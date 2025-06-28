"""
FastAPI application for Slack Hardware Digest System
Includes WebSocket support, REST endpoints, Redis pub/sub, and background tasks
"""

import asyncio
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our mock generator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mock_data.generator import HardwareTeamMockGenerator, MessageCategory, UrgencyLevel
    MOCK_GENERATOR_AVAILABLE = True
    logger.info("Mock generator loaded successfully")
except ImportError:
    logger.warning("Mock generator not available - running without mock data generation")
    MOCK_GENERATOR_AVAILABLE = False
    
    # Create dummy classes for when mock_data is not available
    class HardwareTeamMockGenerator:
        def generate_message(self):
            return None
    
    class MessageCategory:
        pass
    
    class UrgencyLevel:
        pass

# Import AI Agent system
from agents.agent_manager import agent_manager

# Import Enhanced GTM System
from app.gtm_config import gtm_system, initialize_gtm_system

# Global connections
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None

# Pydantic Models
class Message(BaseModel):
    """Slack message model"""
    id: Optional[str] = None
    channel: str
    user: str
    content: str
    timestamp: datetime
    message_type: str
    urgency: str
    category: str
    mentions: List[str] = []
    reactions: List[str] = []
    thread_count: int = 0
    sentiment_score: Optional[float] = None
    priority_level: Optional[str] = None

class RiskAssessment(BaseModel):
    """Risk assessment model"""
    id: Optional[str] = None
    message_id: Optional[str] = None
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    risk_category: str = Field(..., description="Category of risk")
    description: str = Field(..., description="Risk description")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    mitigation_suggestions: List[str] = []
    requires_attention: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AnalysisRequest(BaseModel):
    """Analysis request model"""
    message_ids: Optional[List[str]] = None
    force_reanalysis: bool = False

class SystemHealth(BaseModel):
    """System health status"""
    status: str
    timestamp: datetime
    database: str
    redis: str
    message_count: int
    risk_count: int

class DailyDigest(BaseModel):
    """Daily digest model"""
    date: str
    risk_score: int
    status: str
    critical_alerts: List[Dict[str, Any]]
    wins: List[str]
    priorities: List[str]
    ai_insights: str
    summary_stats: Dict[str, Any]

# WebSocket Manager
class WebSocketManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
            
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# Database and Redis connection functions
async def wait_for_database(max_retries: int = 30, delay: int = 2) -> asyncpg.Pool:
    """Wait for database to be available and create connection pool"""
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/slack_digest")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to database (attempt {attempt + 1}/{max_retries})")
            pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            
            # Test the connection
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            logger.info("Database connection established")
            return pool
            
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            else:
                logger.error("Failed to connect to database after all retries")
                raise

async def wait_for_redis(max_retries: int = 30, delay: int = 2) -> redis.Redis:
    """Wait for Redis to be available and create connection"""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to Redis (attempt {attempt + 1}/{max_retries})")
            client = redis.from_url(redis_url, decode_responses=True)
            
            # Test the connection
            await client.ping()
            
            logger.info("Redis connection established")
            return client
            
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            else:
                logger.error("Failed to connect to Redis after all retries")
                raise

async def get_db() -> asyncpg.Pool:
    """Dependency to get database pool"""
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_pool

async def get_redis() -> redis.Redis:
    """Dependency to get Redis client"""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return redis_client

# Background task for message generation and processing
async def message_generator_task():
    """Background task that generates mock messages and publishes them"""
    if not MOCK_GENERATOR_AVAILABLE:
        logger.info("Mock generator not available - skipping message generation")
        return
        
    generator = HardwareTeamMockGenerator()
    
    while True:
        try:
            # Generate a mock message
            mock_message = generator.generate_message()
            
            # Convert to our Message model
            message = Message(
                channel=mock_message.channel,
                user=mock_message.user,
                content=mock_message.content,
                timestamp=mock_message.timestamp,
                message_type=mock_message.message_type.value,
                urgency=mock_message.urgency.value,
                category=mock_message.category.value,
                mentions=mock_message.mentions or [],
                reactions=mock_message.reactions or [],
                thread_count=mock_message.thread_count,
                sentiment_score=random.uniform(-0.5, 1.0),  # Mock sentiment
                priority_level=mock_message.urgency.value
            )
            
            # Store in database if available
            if db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        # Insert message into database
                        await conn.execute("""
                            INSERT INTO messages (slack_message_id, content, timestamp, is_processed, 
                                                sentiment_score, priority_level, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """, 
                        f"mock_{int(time.time())}_{random.randint(1000, 9999)}",
                        message.content,
                        message.timestamp,
                        False,
                        message.sentiment_score,
                        message.priority_level,
                        json.dumps({
                            "channel": message.channel,
                            "user": message.user,
                            "message_type": message.message_type,
                            "urgency": message.urgency,
                            "category": message.category,
                            "mentions": message.mentions,
                            "reactions": message.reactions
                        })
                        )
                except Exception as e:
                    logger.error(f"Error storing message in database: {e}")
            
            # Publish to Redis for real-time updates
            if redis_client:
                try:
                    await redis_client.publish("messages", message.model_dump_json())
                except Exception as e:
                    logger.error(f"Error publishing to Redis: {e}")
            
            # Broadcast via WebSocket
            await websocket_manager.broadcast(json.dumps({
                "type": "new_message",
                "data": message.model_dump(mode='json')
            }))
            
            # Generate risk assessment for critical/high urgency messages
            if message.urgency in ["critical", "high"]:
                risk = await generate_risk_assessment(message)
                if risk:
                    await websocket_manager.broadcast(json.dumps({
                        "type": "risk_assessment",
                        "data": risk.model_dump(mode='json')
                    }))
            
            # Wait before generating next message (random interval)
            await asyncio.sleep(random.uniform(10, 30))  # 10-30 seconds between messages
            
        except Exception as e:
            logger.error(f"Error in message generator task: {e}")
            await asyncio.sleep(5)  # Wait before retrying

async def generate_risk_assessment(message: Message) -> Optional[RiskAssessment]:
    """Generate AI-powered risk assessment for a message using GTM Risk Commander"""
    try:
        # Convert message to database format for agent analysis
        message_data = {
            "slack_message_id": message.id,
            "content": message.content,
            "timestamp": message.timestamp,
            "priority_level": message.urgency,
            "sentiment_score": message.sentiment_score,
            "metadata": json.dumps({
                "channel": message.channel,
                "user": message.user,
                "message_type": message.message_type,
                "urgency": message.urgency,
                "category": message.category,
                "mentions": message.mentions,
                "reactions": message.reactions
            })
        }
        
        # 🤖 Use AI Agent for analysis instead of mock logic
        try:
            agent_response = await agent_manager.analyze_messages([message_data])
            
            if agent_response.success and agent_response.risk_assessments:
                # Use the first (most critical) risk assessment from AI
                ai_risk = agent_response.risk_assessments[0]
                
                risk = RiskAssessment(
                    message_id=message.id,
                    risk_level=ai_risk.get("level", "medium"),
                    risk_category=ai_risk.get("category", "general"),
                    description=ai_risk.get("description", f"AI analysis: {message.content[:100]}..."),
                    confidence_score=ai_risk.get("confidence", 0.8),
                    mitigation_suggestions=ai_risk.get("recommendations", [
                        "Review AI analysis recommendations",
                        "Implement suggested mitigation strategies"
                    ]),
                    requires_attention=ai_risk.get("requires_attention", message.urgency in ["critical", "high"])
                )
                
                logger.info(f"🤖 AI Risk Assessment generated: {risk.risk_level} risk for {message.category}")
                
            else:
                # Fallback to basic analysis if AI fails
                logger.warning("🤖 AI analysis failed, using fallback analysis")
                risk = await _generate_fallback_risk_assessment(message)
                
        except Exception as ai_error:
            logger.error(f"🤖 AI risk assessment failed: {ai_error}")
            risk = await _generate_fallback_risk_assessment(message)
        
        # Store in database if available
        if db_pool and risk:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO risk_assessments (message_id, risk_level, risk_category, 
                                                    description, confidence_score, requires_attention)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    message.id, risk.risk_level, risk.risk_category,
                    risk.description, risk.confidence_score, risk.requires_attention
                    )
            except Exception as e:
                logger.error(f"Error storing AI risk assessment: {e}")
        
        return risk
        
    except Exception as e:
        logger.error(f"Error generating AI risk assessment: {e}")
        return None

async def _generate_fallback_risk_assessment(message: Message) -> RiskAssessment:
    """Fallback risk assessment when AI is not available"""
    risk_categories = [
        "supply_chain_disruption", "quality_failure", "timeline_delay", 
        "cost_overrun", "compliance_issue", "vendor_risk"
    ]
    
    # Higher urgency = higher risk
    if message.urgency == "critical":
        risk_level = random.choice(["high", "critical"])
    elif message.urgency == "high":
        risk_level = random.choice(["medium", "high"])
    else:
        risk_level = random.choice(["low", "medium"])
    
    return RiskAssessment(
        message_id=message.id,
        risk_level=risk_level,
        risk_category=random.choice(risk_categories),
        description=f"Fallback analysis: {message.content[:100]}...",
        confidence_score=random.uniform(0.3, 0.6),  # Lower confidence for fallback
        mitigation_suggestions=[
            "Review message for hardware risks",
            "Consider escalation if critical",
            "Monitor for similar issues"
        ],
        requires_attention=risk_level in ["high", "critical"]
    )

# Redis subscriber for real-time updates
async def redis_subscriber():
    """Subscribe to Redis channels for real-time updates"""
    if not redis_client:
        return
        
    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("messages", "alerts", "risk_assessments")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                # Broadcast Redis messages to WebSocket clients
                await websocket_manager.broadcast(message["data"])
                
    except Exception as e:
        logger.error(f"Error in Redis subscriber: {e}")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global db_pool, redis_client
    
    # Startup
    logger.info("Starting up Slack Digest System API...")
    
    try:
        # Initialize database connection
        db_pool = await wait_for_database()
        
        # Initialize Redis connection
        redis_client = await wait_for_redis()
        
        # 🤖 Initialize AI Agent System
        ai_success = await initialize_ai_agents()
        if ai_success:
            logger.info("🚀 AI-Powered GTM Analysis System Ready!")
        
        # Start background tasks
        asyncio.create_task(message_generator_task())
        asyncio.create_task(redis_subscriber())
        
        logger.info("Startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    if db_pool:
        await db_pool.close()
    
    if redis_client:
        await redis_client.close()

# Create FastAPI application
app = FastAPI(
    title="Slack Digest System API",
    description="API for Hardware GTM Slack Digest System with real-time WebSocket support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to Slack Digest System",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            await websocket.send_text(json.dumps({
                "type": "echo",
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

# REST API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Slack Digest System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "risk_assessment": "/api/risk-assessment",
            "latest_messages": "/api/messages/latest",
            "analyze": "/api/analyze"
        }
    }

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check database
        db_status = "healthy"
        message_count = 0
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    result = await conn.fetchval("SELECT COUNT(*) FROM messages")
                    message_count = result or 0
            except Exception as e:
                db_status = f"error: {str(e)}"
        else:
            db_status = "not_connected"
        
        # Check Redis
        redis_status = "healthy"
        if redis_client:
            try:
                await redis_client.ping()
            except Exception as e:
                redis_status = f"error: {str(e)}"
        else:
            redis_status = "not_connected"
        
        # Get risk count
        risk_count = 0
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    result = await conn.fetchval("SELECT COUNT(*) FROM risk_assessments")
                    risk_count = result or 0
            except:
                pass
        
        overall_status = "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded"
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            database=db_status,
            redis=redis_status,
            message_count=message_count,
            risk_count=risk_count
        )
        
    except Exception as e:
        return SystemHealth(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            database="error",
            redis="error",
            message_count=0,
            risk_count=0
        )

@app.get("/api/risk-assessment", response_model=List[RiskAssessment])
async def get_risk_assessments(
    limit: int = 20,
    risk_level: Optional[str] = None,
    db: asyncpg.Pool = Depends(get_db)
):
    """Get current risk assessments"""
    try:
        query = """
            SELECT risk_level, risk_category, description, confidence_score, 
                   requires_attention, created_at
            FROM risk_assessments 
        """
        params = []
        
        if risk_level:
            query += " WHERE risk_level = $1"
            params.append(risk_level)
        
        query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        async with db.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        assessments = []
        for row in rows:
            assessments.append(RiskAssessment(
                risk_level=row['risk_level'],
                risk_category=row['risk_category'],
                description=row['description'],
                confidence_score=row['confidence_score'],
                requires_attention=row['requires_attention'],
                created_at=row['created_at']
            ))
        
        return assessments
        
    except Exception as e:
        logger.error(f"Error fetching risk assessments: {e}")
        raise HTTPException(status_code=500, detail="Error fetching risk assessments")

@app.get("/api/messages/latest", response_model=List[Message])
async def get_latest_messages(
    limit: int = 50,
    category: Optional[str] = None,
    urgency: Optional[str] = None,
    db: asyncpg.Pool = Depends(get_db)
):
    """Get latest messages"""
    try:
        query = """
            SELECT slack_message_id, content, timestamp, sentiment_score, 
                   priority_level, metadata
            FROM messages 
        """
        params = []
        conditions = []
        
        if category:
            conditions.append(f"metadata->>'category' = ${len(params) + 1}")
            params.append(category)
        
        if urgency:
            conditions.append(f"priority_level = ${len(params) + 1}")
            params.append(urgency)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY timestamp DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with db.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        messages = []
        for row in rows:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            messages.append(Message(
                id=row['slack_message_id'],
                channel=metadata.get('channel', '#unknown'),
                user=metadata.get('user', 'unknown'),
                content=row['content'],
                timestamp=row['timestamp'],
                message_type=metadata.get('message_type', 'update'),
                urgency=metadata.get('urgency', 'medium'),
                category=metadata.get('category', 'general'),
                mentions=metadata.get('mentions', []),
                reactions=metadata.get('reactions', []),
                sentiment_score=row['sentiment_score'],
                priority_level=row['priority_level']
            ))
        
        return messages
        
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail="Error fetching messages")

@app.post("/api/analyze")
async def trigger_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: asyncpg.Pool = Depends(get_db),
    redis: redis.Redis = Depends(get_redis)
):
    """Trigger AI analysis of messages"""
    try:
        # Add analysis task to background
        background_tasks.add_task(run_analysis, request, db, redis)
        
        return {
            "message": "Analysis triggered successfully",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Error triggering analysis: {e}")
        raise HTTPException(status_code=500, detail="Error triggering analysis")

async def run_analysis(request: AnalysisRequest, db: asyncpg.Pool, redis: redis.Redis):
    """Background task to run analysis"""
    try:
        logger.info("Running analysis task...")
        
        # Mock analysis - in real implementation, this would call OpenAI
        await asyncio.sleep(2)  # Simulate processing time
        
        # Generate mock analysis results
        result = {
            "analysis_id": f"analysis_{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "summary": "Analysis completed successfully",
            "insights": [
                "3 critical supply chain risks identified",
                "Quality metrics trending downward in displays category",
                "EVT milestone delays detected in 2 projects"
            ],
            "recommendations": [
                "Engage alternative suppliers for critical components",
                "Initiate quality improvement process",
                "Review timeline dependencies"
            ]
        }
        
        # Publish results
        await redis.publish("analysis_results", json.dumps(result))
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast(json.dumps({
            "type": "analysis_complete",
            "data": result
        }))
        
        logger.info("Analysis task completed")
        
    except Exception as e:
        logger.error(f"Error in analysis task: {e}")

@app.get("/api/stats")
async def get_system_stats(db: asyncpg.Pool = Depends(get_db)):
    """Get system statistics"""
    try:
        async with db.acquire() as conn:
            # Get message stats
            message_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(CASE WHEN priority_level = 'critical' THEN 1 END) as critical_messages,
                    COUNT(CASE WHEN priority_level = 'high' THEN 1 END) as high_messages,
                    AVG(sentiment_score) as avg_sentiment
                FROM messages
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            
            # Get risk stats
            risk_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_risks,
                    COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_risks,
                    COUNT(CASE WHEN requires_attention = true THEN 1 END) as attention_required
                FROM risk_assessments
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
        
        return {
            "period": "last_24_hours",
            "messages": {
                "total": message_stats['total_messages'] or 0,
                "critical": message_stats['critical_messages'] or 0,
                "high": message_stats['high_messages'] or 0,
                "avg_sentiment": float(message_stats['avg_sentiment'] or 0)
            },
            "risks": {
                "total": risk_stats['total_risks'] or 0,
                "critical": risk_stats['critical_risks'] or 0,
                "attention_required": risk_stats['attention_required'] or 0
            },
            "websocket_connections": len(websocket_manager.active_connections),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Error fetching statistics")

@app.get("/api/digest/daily", response_model=DailyDigest)
async def generate_daily_digest(
    date: Optional[str] = None,
    db: asyncpg.Pool = Depends(get_db)
):
    """Generate AI-powered daily hardware team digest using comprehensive multi-agent analysis"""
    try:
        target_date = date or datetime.utcnow().strftime("%Y-%m-%d")
        
        # Get messages from the last 24 hours
        async with db.acquire() as conn:
            messages = await conn.fetch("""
                SELECT slack_message_id, content, priority_level, metadata, sentiment_score, timestamp
                FROM messages 
                WHERE DATE(timestamp) = CURRENT_DATE
                ORDER BY timestamp DESC
            """)
            
            # Get risk assessments from today
            risks = await conn.fetch("""
                SELECT risk_level, risk_category, description, confidence_score, requires_attention
                FROM risk_assessments
                WHERE DATE(created_at) = CURRENT_DATE
                ORDER BY created_at DESC
            """)
        
        if not messages:
            # Return default digest if no messages
            return DailyDigest(
                date=target_date,
                risk_score=85,
                status="GREEN - No Issues Detected",
                critical_alerts=[],
                wins=["System operational", "No critical alerts"],
                priorities=["Continue monitoring"],
                ai_insights="No messages detected today. System appears operational.",
                summary_stats={"total_messages": 0, "critical_issues": 0, "high_priority": 0, "avg_sentiment": 100.0, "risk_assessments": 0}
            )
        
        # Convert messages to agent format for comprehensive AI analysis
        message_data = []
        for row in messages:
            message_data.append({
                "slack_message_id": row['slack_message_id'],
                "content": row['content'],
                "timestamp": row['timestamp'],
                "priority_level": row['priority_level'],
                "sentiment_score": row['sentiment_score'],
                "metadata": row['metadata']
            })
        
        # 🤖 PERFORM COMPREHENSIVE MULTI-AGENT ANALYSIS
        logger.info(f"🤖 Generating AI-powered digest with {len(message_data)} messages using all 4 agents")
        
        try:
            # Get comprehensive analysis from all 4 agents
            comprehensive_result = await agent_manager.get_comprehensive_analysis(message_data)
            
            # Extract analysis results from each agent
            all_analyses = comprehensive_result["comprehensive_analysis"]
            overall_risk_score = comprehensive_result["overall_risk_score"]
            overall_confidence = comprehensive_result["overall_confidence"]
            successful_analyses = comprehensive_result.get("successful_analyses", 0)
            
            # Use multi-agent intelligence to create digest content
            critical_alerts = await _extract_ai_critical_alerts(all_analyses, message_data)
            wins = await _extract_ai_wins(all_analyses, message_data)
            priorities = await _extract_ai_priorities(all_analyses, message_data)
            ai_insights = await _generate_ai_insights(all_analyses, comprehensive_result, message_data)
            
            # Determine AI-powered status
            if overall_risk_score >= 80:
                status = "🟢 GREEN - AI Assessment: On Track"
            elif overall_risk_score >= 60:
                status = "🟡 YELLOW - AI Assessment: Action Required"
            elif overall_risk_score >= 40:
                status = "🟠 ORANGE - AI Assessment: High Risk"
            else:
                status = "🔴 RED - AI Assessment: Critical Issues"
            
        except Exception as ai_error:
            logger.warning(f"🤖 AI analysis failed, using enhanced fallback: {ai_error}")
            # Enhanced fallback with better analysis
            overall_risk_score, critical_alerts, wins, priorities, ai_insights, status = await _generate_enhanced_fallback_digest(message_data, risks)
        
        # Calculate comprehensive summary stats
        total_messages = len(messages)
        critical_count = sum(1 for m in messages if m['priority_level'] == 'critical')
        high_count = sum(1 for m in messages if m['priority_level'] == 'high')
        avg_sentiment = sum(m['sentiment_score'] or 0.5 for m in messages) / max(len(messages), 1)
        
        summary_stats = {
            "total_messages": total_messages,
            "critical_issues": critical_count,
            "high_priority": high_count,
            "avg_sentiment": round(avg_sentiment * 100, 1),
            "risk_assessments": len(risks),
            "ai_powered": True,
            "agents_used": 4,
            "analysis_confidence": round(overall_confidence * 100, 1) if 'overall_confidence' in locals() else 85,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return DailyDigest(
            date=target_date,
            risk_score=int(overall_risk_score),
            status=status,
            critical_alerts=critical_alerts,
            wins=wins,
            priorities=priorities,
            ai_insights=ai_insights,
            summary_stats=summary_stats
        )
        
    except Exception as e:
        logger.error(f"Error generating AI-powered digest: {e}")
        raise HTTPException(status_code=500, detail="Error generating daily digest")

# AI-Powered Digest Helper Functions
async def _extract_ai_critical_alerts(all_analyses: Dict[str, Any], messages: List[Dict]) -> List[Dict[str, Any]]:
    """Extract critical alerts using AI analysis from all agents"""
    critical_alerts = []
    
    try:
        # Get supply chain alerts
        if "supply_chain_analysis" in all_analyses and all_analyses["supply_chain_analysis"].get("success"):
            supply_analysis = all_analyses["supply_chain_analysis"]
            recommendations = supply_analysis.get("recommendations", [])
            for rec in recommendations[:2]:
                if any(word in rec.lower() for word in ["critical", "shortage", "urgent", "immediate"]):
                    critical_alerts.append({
                        "category": "Supply Chain",
                        "description": rec[:100] + "..." if len(rec) > 100 else rec,
                        "impact": "Supply chain disruption potential",
                        "action": "Review supplier status and alternatives",
                        "urgency": "high",
                        "source": "AI Supply Chain Agent"
                    })
        
        # Get quality alerts
        if "quality_analysis" in all_analyses and all_analyses["quality_analysis"].get("success"):
            quality_analysis = all_analyses["quality_analysis"]
            recommendations = quality_analysis.get("recommendations", [])
            for rec in recommendations[:2]:
                if any(word in rec.lower() for word in ["quality", "defect", "yield", "failure"]):
                    critical_alerts.append({
                        "category": "Quality",
                        "description": rec[:100] + "..." if len(rec) > 100 else rec,
                        "impact": "Product quality risk",
                        "action": "Investigate quality metrics",
                        "urgency": "high",
                        "source": "AI Quality Agent"
                    })
        
        # Get timeline alerts
        if "timeline_analysis" in all_analyses and all_analyses["timeline_analysis"].get("success"):
            timeline_analysis = all_analyses["timeline_analysis"]
            recommendations = timeline_analysis.get("recommendations", [])
            for rec in recommendations[:2]:
                if any(word in rec.lower() for word in ["delay", "timeline", "schedule", "milestone"]):
                    critical_alerts.append({
                        "category": "Timeline",
                        "description": rec[:100] + "..." if len(rec) > 100 else rec,
                        "impact": "Launch timeline risk",
                        "action": "Review project schedule",
                        "urgency": "critical",
                        "source": "AI Timeline Agent"
                    })
        
        # Get GTM alerts
        if "gtm_analysis" in all_analyses and all_analyses["gtm_analysis"].get("success"):
            gtm_analysis = all_analyses["gtm_analysis"]
            risk_assessments = gtm_analysis.get("risk_assessments", [])
            for risk in risk_assessments[:2]:
                if risk.get("level") in ["critical", "high"]:
                    critical_alerts.append({
                        "category": "GTM Risk",
                        "description": risk.get("description", "")[:100] + "..." if len(risk.get("description", "")) > 100 else risk.get("description", ""),
                        "impact": risk.get("impact", "Launch readiness impact"),
                        "action": "Executive review required",
                        "urgency": risk.get("level", "high"),
                        "source": "AI GTM Commander"
                    })
    
    except Exception as e:
        logger.warning(f"Error extracting AI critical alerts: {e}")
    
    # Fallback to message-based alerts if no AI alerts
    if not critical_alerts:
        for msg in messages[:3]:
            if msg['priority_level'] in ['critical', 'high']:
                metadata = json.loads(msg['metadata']) if msg['metadata'] else {}
                category = metadata.get('category', 'General')
                critical_alerts.append({
                    "category": category.replace('_', ' ').title(),
                    "description": msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content'],
                    "impact": "Operational impact",
                    "action": "Review required",
                    "urgency": msg['priority_level'],
                    "source": "Message Analysis"
                })
    
    return critical_alerts[:3]  # Top 3 alerts

async def _extract_ai_wins(all_analyses: Dict[str, Any], messages: List[Dict]) -> List[str]:
    """Extract wins using AI analysis"""
    wins = []
    
    try:
        # Look for positive recommendations and achievements in AI analyses
        for agent_name, analysis in all_analyses.items():
            if analysis.get("success") and "recommendations" in analysis:
                recommendations = analysis["recommendations"]
                for rec in recommendations:
                    if any(word in rec.lower() for word in ["success", "improvement", "completed", "achieved", "positive", "good"]):
                        clean_rec = rec.replace("✅", "").replace("🎉", "").strip()
                        if clean_rec not in wins:
                            wins.append(clean_rec)
        
        # Look for positive sentiment messages
        positive_messages = [m for m in messages if (m['sentiment_score'] or 0) > 0.7]
        for msg in positive_messages[:2]:
            content = msg['content']
            if any(indicator in content.lower() for indicator in ["✅", "completed", "approved", "success", "good news"]):
                clean_content = content.replace("✅", "").replace("🎉", "").strip()[:80]
                if clean_content not in wins:
                    wins.append(clean_content)
    
    except Exception as e:
        logger.warning(f"Error extracting AI wins: {e}")
    
    # Fallback wins if none found
    if not wins:
        wins = [
            "Multi-agent AI analysis completed successfully",
            "System health monitoring operational", 
            "Hardware team communication active"
        ]
    
    return wins[:4]  # Top 4 wins

async def _extract_ai_priorities(all_analyses: Dict[str, Any], messages: List[Dict]) -> List[str]:
    """Extract priorities using AI analysis"""
    priorities = []
    
    try:
        # Extract priorities from each agent's recommendations
        for agent_name, analysis in all_analyses.items():
            if analysis.get("success") and "recommendations" in analysis:
                recommendations = analysis["recommendations"]
                for rec in recommendations[:2]:  # Top 2 from each agent
                    # Convert recommendations to actionable priorities
                    if "supply" in rec.lower() or "component" in rec.lower():
                        priority = f"🔧 Supply Chain: {rec[:60]}..." if len(rec) > 60 else f"🔧 Supply Chain: {rec}"
                    elif "quality" in rec.lower() or "test" in rec.lower():
                        priority = f"🔍 Quality: {rec[:60]}..." if len(rec) > 60 else f"🔍 Quality: {rec}"
                    elif "timeline" in rec.lower() or "schedule" in rec.lower():
                        priority = f"⏰ Timeline: {rec[:60]}..." if len(rec) > 60 else f"⏰ Timeline: {rec}"
                    else:
                        priority = f"📋 Action: {rec[:60]}..." if len(rec) > 60 else f"📋 Action: {rec}"
                    
                    if priority not in priorities:
                        priorities.append(priority)
    
    except Exception as e:
        logger.warning(f"Error extracting AI priorities: {e}")
    
    # Fallback priorities if none found
    if not priorities:
        priorities = [
            "📊 Monitor ongoing hardware component supply chain",
            "🔍 Review daily quality metrics and yields", 
            "⏰ Track project milestone progress",
            "📋 Maintain team communication effectiveness"
        ]
    
    return priorities[:5]  # Top 5 priorities

async def _generate_ai_insights(all_analyses: Dict[str, Any], comprehensive_result: Dict, messages: List[Dict]) -> str:
    """Generate AI insights using comprehensive analysis"""
    try:
        total_messages = len(messages)
        overall_risk_score = comprehensive_result["overall_risk_score"]
        overall_confidence = comprehensive_result["overall_confidence"]
        successful_analyses = comprehensive_result.get("successful_analyses", 0)
        
        # Create dynamic insights based on AI analysis
        insights_parts = []
        
        # Overall assessment
        risk_trend = "improving" if overall_risk_score > 75 else "declining" if overall_risk_score < 50 else "stable"
        insights_parts.append(f"Multi-agent AI analysis of {total_messages} team communications reveals {risk_trend} GTM trajectory.")
        
        # Agent-specific insights
        if "gtm_analysis" in all_analyses and all_analyses["gtm_analysis"].get("success"):
            gtm_summary = all_analyses["gtm_analysis"].get("summary", "")
            if gtm_summary:
                insights_parts.append(f"GTM Intelligence: {gtm_summary[:100]}...")
        
        if "supply_chain_analysis" in all_analyses and all_analyses["supply_chain_analysis"].get("success"):
            supply_summary = all_analyses["supply_chain_analysis"].get("summary", "")
            if "risk" in supply_summary.lower() or "shortage" in supply_summary.lower():
                insights_parts.append("Supply chain monitoring indicates potential component availability concerns.")
        
        if "quality_analysis" in all_analyses and all_analyses["quality_analysis"].get("success"):
            quality_summary = all_analyses["quality_analysis"].get("summary", "")
            if "quality" in quality_summary.lower():
                insights_parts.append("Quality assessment systems are actively monitoring production metrics.")
        
        # Confidence and recommendation
        confidence_level = "high" if overall_confidence > 0.8 else "moderate" if overall_confidence > 0.6 else "low"
        insights_parts.append(f"AI confidence level: {confidence_level} ({overall_confidence*100:.1f}%).")
        
        # Strategic recommendation
        if overall_risk_score > 75:
            insights_parts.append("Recommend maintaining current GTM pace with continued monitoring.")
        elif overall_risk_score > 50:
            insights_parts.append("Recommend accelerated focus on identified risk areas.")
        else:
            insights_parts.append("Recommend immediate escalation and executive review of critical issues.")
        
        return " ".join(insights_parts)
    
    except Exception as e:
        logger.warning(f"Error generating AI insights: {e}")
        return f"AI analysis processed {len(messages)} messages. Comprehensive multi-agent intelligence indicates ongoing GTM monitoring is active. System recommends continued vigilance on hardware supply chain, quality metrics, and timeline adherence."

async def _generate_enhanced_fallback_digest(messages: List[Dict], risks: List) -> tuple:
    """Generate enhanced fallback digest when AI analysis fails"""
    try:
        total_messages = len(messages)
        critical_count = sum(1 for m in messages if m['priority_level'] == 'critical')
        high_count = sum(1 for m in messages if m['priority_level'] == 'high')
        avg_sentiment = sum(m['sentiment_score'] or 0.5 for m in messages) / max(len(messages), 1)
        
        # Enhanced risk calculation
        base_score = 85
        risk_score = max(20, base_score - (critical_count * 10) - (high_count * 5) - int((1 - avg_sentiment) * 25))
        
        # Enhanced critical alerts
        critical_alerts = []
        for msg in messages[:3]:
            if msg['priority_level'] in ['critical', 'high']:
                metadata = json.loads(msg['metadata']) if msg['metadata'] else {}
                category = metadata.get('category', 'General')
                critical_alerts.append({
                    "category": category.replace('_', ' ').title(),
                    "description": msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content'],
                    "impact": "Requires immediate attention",
                    "action": "Review and escalate if needed",
                    "urgency": msg['priority_level'],
                    "source": "Enhanced Analysis"
                })
        
        # Enhanced wins
        wins = []
        positive_messages = [m for m in messages if (m['sentiment_score'] or 0) > 0.7]
        for msg in positive_messages[:3]:
            content = msg['content']
            if any(indicator in content.lower() for indicator in ["✅", "completed", "approved", "success"]):
                clean_content = content.replace("✅", "").replace("🎉", "").strip()[:80]
                wins.append(clean_content)
        
        if not wins:
            wins = ["System operational", "Communications active", "Monitoring functional"]
        
        # Enhanced priorities
        priorities = [
            f"📊 Review {critical_count} critical messages requiring attention",
            f"🔍 Monitor {high_count} high-priority items",
            "⚡ Maintain current communication effectiveness",
            "📋 Continue systematic hardware GTM tracking"
        ]
        
        # Enhanced insights
        risk_trend = "improving" if risk_score > 75 else "concerning" if risk_score < 50 else "stable"
        ai_insights = f"Enhanced analysis of {total_messages} communications shows {risk_trend} trends. {critical_count} critical issues detected. Team sentiment at {avg_sentiment*100:.1f}%. Recommend {'continued monitoring' if risk_score > 75 else 'increased attention' if risk_score > 50 else 'immediate escalation'} for optimal GTM execution."
        
        # Enhanced status
        if risk_score >= 80:
            status = "🟢 GREEN - Enhanced Assessment: On Track"
        elif risk_score >= 60:
            status = "🟡 YELLOW - Enhanced Assessment: Action Required"
        elif risk_score >= 40:
            status = "🟠 ORANGE - Enhanced Assessment: High Risk"
        else:
            status = "🔴 RED - Enhanced Assessment: Critical Issues"
        
        return risk_score, critical_alerts, wins, priorities, ai_insights, status
    
    except Exception as e:
        logger.error(f"Enhanced fallback failed: {e}")
        return 75, [], ["System operational"], ["Monitor status"], "Fallback analysis active", "🟡 YELLOW - Basic Assessment"

@app.get("/api/digest/formatted")
async def get_formatted_digest(
    date: Optional[str] = None,
    db: asyncpg.Pool = Depends(get_db)
):
    """Get beautifully formatted digest for Slack posting"""
    digest = await generate_daily_digest(date, db)
    
    # Format as beautiful text
    formatted = f"""🏭 **Hardware GTM Daily Digest - {digest.date}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **GTM RISK SCORE: {digest.risk_score}/100**
**Status:** {digest.status}

🚨 **CRITICAL ALERTS ({len(digest.critical_alerts)})**"""

    for i, alert in enumerate(digest.critical_alerts, 1):
        urgency_emoji = "🔴" if alert['urgency'] == 'critical' else "🟠"
        formatted += f"""
{i}. {urgency_emoji} **[{alert['category']}]** {alert['description']}
   **Impact:** {alert['impact']}
   **Action:** {alert['action']}"""
    
    formatted += f"""

✅ **WINS ({len(digest.wins)})**"""
    for win in digest.wins:
        formatted += f"\n• {win}"
    
    formatted += f"""

📋 **TODAY'S PRIORITIES**"""
    for priority in digest.priorities:
        formatted += f"\n□ {priority}"
    
    formatted += f"""

💡 **AI INSIGHTS**
"{digest.ai_insights}"

📊 **STATS:** {digest.summary_stats['total_messages']} messages | {digest.summary_stats['critical_issues']} critical | {digest.summary_stats['avg_sentiment']}% sentiment"""
    
    return {"formatted_digest": formatted, "raw_data": digest}

# Initialize AI Agent Manager on startup
async def initialize_ai_agents():
    """Initialize AI Agent Manager with all 4 specialized agents"""
    try:
        logger.info("🤖 Initializing AI Agent System with 4 specialized agents...")
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY", "your-key-here")
        
        # Initialize agent manager with API key
        global agent_manager
        from agents.agent_manager import AgentManager
        agent_manager = AgentManager(openai_api_key)
        
        success = await agent_manager.initialize()
        if success:
            logger.info("✅ Multi-Agent AI System initialized successfully")
            logger.info("🚀 Available Agents: GTM Commander, Supply Chain Intel, Quality Anomaly, Timeline Tracker")
            
            # Initialize AI-powered GTM system
            try:
                from agents.gtm_strategy_agent import GTMStrategyAgent
                gtm_strategy_agent = GTMStrategyAgent(openai_api_key)
                global gtm_system
                gtm_system = initialize_gtm_system(gtm_strategy_agent)
                logger.info("🎯 AI-Powered GTM Strategy System initialized")
            except Exception as e:
                logger.warning(f"⚠️ GTM Strategy Agent initialization failed: {e}")
                gtm_system = initialize_gtm_system()  # Initialize without AI
        else:
            logger.warning("⚠️ AI Agent System initialization failed - falling back to basic analysis")
            gtm_system = initialize_gtm_system()  # Initialize without AI
        return success
    except Exception as e:
        logger.error(f"❌ AI Agent System initialization error: {e}")
        return False

@app.get("/api/agents/status")
async def get_agent_status():
    """Get AI Agent system status and health"""
    try:
        status = await agent_manager.get_agent_status()
        return {
            "ai_system_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail="Error getting AI agent status")

@app.post("/api/agents/analyze")
async def trigger_ai_analysis(
    request: AnalysisRequest,
    db: asyncpg.Pool = Depends(get_db)
):
    """Trigger comprehensive multi-agent AI analysis using all 4 specialized agents"""
    try:
        # Get messages for analysis
        async with db.acquire() as conn:
            if request.message_ids:
                # Analyze specific messages
                query = """
                    SELECT slack_message_id, content, timestamp, priority_level, 
                           sentiment_score, metadata
                    FROM messages 
                    WHERE slack_message_id = ANY($1)
                    ORDER BY timestamp DESC
                """
                rows = await conn.fetch(query, request.message_ids)
            else:
                # Analyze recent critical/high priority messages
                query = """
                    SELECT slack_message_id, content, timestamp, priority_level, 
                           sentiment_score, metadata
                    FROM messages 
                    WHERE priority_level IN ('critical', 'high')
                    AND timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC
                    LIMIT 50
                """
                rows = await conn.fetch(query)
        
        if not rows:
            return {
                "message": "No messages found for AI analysis",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Convert to agent format
        messages = []
        for row in rows:
            messages.append({
                "slack_message_id": row['slack_message_id'],
                "content": row['content'],
                "timestamp": row['timestamp'],
                "priority_level": row['priority_level'],
                "sentiment_score": row['sentiment_score'],
                "metadata": row['metadata']
            })
        
        # 🤖 Perform comprehensive multi-agent analysis
        logger.info(f"🤖 Triggering multi-agent analysis for {len(messages)} messages")
        comprehensive_result = await agent_manager.get_comprehensive_analysis(messages)
        
        return {
            "message": "🤖 Multi-agent AI analysis completed successfully",
            "multi_agent_analysis": {
                "overall_risk_score": comprehensive_result["overall_risk_score"],
                "overall_confidence": comprehensive_result["overall_confidence"],
                "agent_analyses": comprehensive_result["comprehensive_analysis"],
                "multi_agent_intelligence": comprehensive_result.get("multi_agent_intelligence", True),
                "successful_analyses": comprehensive_result.get("successful_analyses", 0),
                "agent_count": comprehensive_result["agent_count"]
            },
            "message_count": len(messages),
            "timestamp": comprehensive_result["analysis_timestamp"]
        }
        
    except Exception as e:
        logger.error(f"AI analysis trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@app.get("/api/agents/multi-agent-demo")
async def multi_agent_demo(
    db: asyncpg.Pool = Depends(get_db)
):
    """Demonstrate all 4 AI agents working together on recent hardware messages"""
    try:
        # Get recent high-priority messages for demonstration
        async with db.acquire() as conn:
            messages = await conn.fetch("""
                SELECT slack_message_id, content, timestamp, priority_level, 
                       sentiment_score, metadata
                FROM messages 
                WHERE priority_level IN ('critical', 'high', 'medium')
                ORDER BY timestamp DESC
                LIMIT 10
            """)
        
        if not messages:
            return {
                "message": "No messages found for multi-agent demonstration",
                "suggestion": "Generate some messages first via the message generator",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Convert to agent format
        message_data = []
        for row in messages:
            message_data.append({
                "slack_message_id": row['slack_message_id'],
                "content": row['content'],
                "timestamp": row['timestamp'],
                "priority_level": row['priority_level'],
                "sentiment_score": row['sentiment_score'],
                "metadata": row['metadata']
            })
        
        # 🤖 Perform comprehensive multi-agent analysis
        logger.info(f"🤖 Multi-agent demo: Analyzing {len(message_data)} messages with all 4 agents")
        comprehensive_result = await agent_manager.get_comprehensive_analysis(message_data)
        
        # Extract individual agent results for detailed view
        agent_results = comprehensive_result["comprehensive_analysis"]
        
        demo_result = {
            "demo_title": "🚀 Multi-Agent Hardware GTM Intelligence System Demo",
            "analysis_overview": {
                "overall_risk_score": comprehensive_result["overall_risk_score"],
                "overall_confidence": comprehensive_result["overall_confidence"],
                "messages_analyzed": len(message_data),
                "agents_deployed": comprehensive_result["agent_count"],
                "successful_analyses": comprehensive_result.get("successful_analyses", 0),
                "multi_agent_intelligence": True
            },
            "agent_breakdown": {
                "gtm_commander": {
                    "role": "🎯 GTM Risk Commander",
                    "specialization": "Holistic launch readiness assessment",
                    "analysis": agent_results.get("gtm_analysis", {})
                },
                "supply_chain_intel": {
                    "role": "📦 Supply Chain Intelligence",
                    "specialization": "Component availability & supplier performance",
                    "analysis": agent_results.get("supply_chain_analysis", {})
                },
                "quality_anomaly": {
                    "role": "🔍 Quality Anomaly Detection",
                    "specialization": "Defect patterns & yield forecasting",
                    "analysis": agent_results.get("quality_analysis", {})
                },
                "timeline_milestone": {
                    "role": "⏰ Timeline & Milestone Tracker",
                    "specialization": "Schedule conflicts & dependency tracking",
                    "analysis": agent_results.get("timeline_analysis", {})
                }
            },
            "sample_messages_analyzed": [
                {
                    "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"],
                    "priority": msg["priority_level"],
                    "metadata": json.loads(msg["metadata"]) if msg["metadata"] else {}
                }
                for msg in message_data[:3]
            ],
            "analysis_timestamp": comprehensive_result["analysis_timestamp"],
            "next_steps": [
                "🔑 Add your OpenAI API key to unlock full AI capabilities",
                "📊 Monitor the Streamlit dashboard for real-time insights",
                "🔄 Enable continuous multi-agent analysis",
                "🚀 Deploy for production GTM intelligence"
            ]
        }
        
        return demo_result
        
    except Exception as e:
        logger.error(f"Multi-agent demo failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-agent demo failed: {str(e)}")

@app.get("/api/gtm/dashboard")
async def get_gtm_dashboard(db: asyncpg.Pool = Depends(get_db)):
    """Get enhanced GTM dashboard with AI-powered dynamic weighting and real-time updates"""
    try:
        # Check if GTM system is initialized
        if gtm_system is None:
            raise HTTPException(status_code=503, detail="GTM system not initialized yet")
        
        # Get recent messages for AI analysis context
        async with db.acquire() as conn:
            recent_messages = await conn.fetch("""
                SELECT slack_message_id, content, timestamp, priority_level, 
                       sentiment_score, metadata
                FROM messages 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                ORDER BY timestamp DESC
                LIMIT 100
            """)
        
        # Analyze message context for AI insights
        ai_context = {
            "supply_chain_risks": False,
            "quality_issues": False,
            "timeline_delays": False
        }
        
        for msg in recent_messages:
            if msg['priority_level'] in ['critical', 'high']:
                metadata = json.loads(msg['metadata']) if msg['metadata'] else {}
                category = metadata.get('category', '')
                
                if 'shortage' in category or 'supplier' in category:
                    ai_context["supply_chain_risks"] = True
                if 'quality' in category or 'defect' in category:
                    ai_context["quality_issues"] = True
                if 'timeline' in category or 'delay' in category:
                    ai_context["timeline_delays"] = True
        
        # Apply AI insights to GTM scoring
        ai_adjustments = gtm_system.apply_ai_insights(ai_context)
        
        # Convert recent_messages to list of dicts for GTM analysis
        message_data = []
        for row in recent_messages:
            message_data.append({
                'slack_message_id': row['slack_message_id'],
                'content': row['content'],
                'timestamp': row['timestamp'],
                'priority_level': row['priority_level'],
                'sentiment_score': row['sentiment_score']
            })
        
        # Calculate comprehensive GTM score using AI-POWERED GTM system
        gtm_data = await gtm_system.calculate_gtm_score(message_data, ai_adjustments)
        
        # Add real-time message stats
        gtm_data["realtime_stats"] = {
            "messages_last_24h": len(recent_messages),
            "critical_alerts": len([m for m in recent_messages if m['priority_level'] == 'critical']),
            "avg_sentiment": sum(m['sentiment_score'] or 0 for m in recent_messages) / max(len(recent_messages), 1),
            "ai_adjustments_applied": ai_adjustments
        }
        
        return gtm_data
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"GTM dashboard generation failed: {e}")
        logger.error(f"Full traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"GTM dashboard failed: {str(e)[:200]}... Full error logged")

@app.get("/api/gtm/strategic-questions")
async def get_strategic_questions(db: asyncpg.Pool = Depends(get_db)):
    """Generate strategic questions for the team to answer"""
    try:
        # Initialize Interactive GTM Agent
        openai_api_key = os.getenv("OPENAI_API_KEY", "your-key-here")
        from agents.interactive_gtm_agent import InteractiveGTMAgent
        interactive_agent = InteractiveGTMAgent(openai_api_key)
        
        # Get current context
        current_context = {
            "days_to_launch": 78,
            "current_phase": "PVT",
            "recent_issues": []
        }
        
        # Generate strategic questions
        questions = await interactive_agent.generate_strategic_questions(current_context)
        
        return {
            "message": "🤖 Strategic questions generated for team engagement",
            "questions": questions,
            "context": current_context,
            "instructions": {
                "how_to_use": "Ask these questions in your team channels",
                "collection_method": "Use /api/gtm/submit-response to collect answers",
                "follow_up": "GTM weights will auto-adjust based on responses"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Strategic question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

@app.post("/api/gtm/submit-response")
async def submit_team_response(
    question_id: str,
    answer: str,
    respondent_role: str,
    confidence: Optional[int] = 5,
    db: asyncpg.Pool = Depends(get_db)
):
    """Submit team response to strategic question"""
    try:
        # Store the response (in real system, you'd save to database)
        response_data = {
            "question_id": question_id,
            "answer": answer,
            "respondent_role": respondent_role,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # For demo, just return acknowledgment
        return {
            "message": "✅ Response recorded successfully",
            "response": response_data,
            "next_steps": [
                "Your input will influence GTM weight adjustments",
                "Check the dashboard for updated priorities",
                "Look for follow-up questions based on your response"
            ]
        }
        
    except Exception as e:
        logger.error(f"Response submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Response submission failed: {str(e)}")

@app.get("/api/gtm/team-engagement-demo")
async def team_engagement_demo():
    """Demonstrate interactive team engagement workflow"""
    try:
        # Demo interactive questions and responses
        demo_questions = [
            {
                "id": "q1",
                "question": "What's your biggest concern for launch readiness this week?",
                "type": "priority",
                "target_audience": "All",
                "responses": [
                    {"role": "Engineering", "answer": "Yield rates are still below 90%", "confidence": 6},
                    {"role": "Supply Chain", "answer": "Backup supplier for OLED panels not qualified", "confidence": 4},
                    {"role": "Quality", "answer": "Need more time for reliability testing", "confidence": 5}
                ]
            },
            {
                "id": "q2", 
                "question": "Rate your confidence in hitting the September launch date (1-10)",
                "type": "confidence",
                "target_audience": "All",
                "responses": [
                    {"role": "Engineering", "answer": "7 - if we solve yield issues", "confidence": 7},
                    {"role": "Supply Chain", "answer": "6 - supplier risks worry me", "confidence": 6},
                    {"role": "Quality", "answer": "8 - quality looks good", "confidence": 8}
                ]
            },
            {
                "id": "q3",
                "question": "Should we prioritize yield improvement or production ramp-up this week?",
                "type": "resource_allocation", 
                "target_audience": "Engineering",
                "responses": [
                    {"role": "Engineering Lead", "answer": "Yield improvement - 90%+ is critical", "confidence": 9},
                    {"role": "Production Manager", "answer": "Yield first, then ramp", "confidence": 8}
                ]
            }
        ]
        
        # Analyze responses to show AI-powered insights
        ai_analysis = {
            "key_insights": [
                "Team confidence averages 6.8/10 - moderate concern",
                "Yield rates are the primary engineering focus",
                "Supply chain backup planning needs immediate attention",
                "Quality team is most confident in their area"
            ],
            "recommended_weight_adjustments": {
                "quality_readiness": "INCREASE to 40% (team confident)",
                "supply_chain_readiness": "INCREASE to 30% (backup supplier risk)",
                "product_readiness": "MAINTAIN at 25% (yield focus needed)",
                "operational_readiness": "DECREASE to 5% (not current priority)"
            },
            "immediate_actions": [
                "🎯 Focus engineering resources on yield improvement",
                "📦 Accelerate backup OLED supplier qualification", 
                "⏰ Schedule daily yield review meetings",
                "🔍 Increase quality testing confidence sharing"
            ],
            "next_questions": [
                "What specific yield improvements are you targeting?",
                "How long will backup supplier qualification take?",
                "What resources do you need for yield improvement?"
            ]
        }
        
        return {
            "demo_title": "🚀 Interactive GTM Strategy Engagement Demo",
            "workflow": {
                "step_1": "AI generates strategic questions based on current context",
                "step_2": "Team members answer questions in Slack channels", 
                "step_3": "AI analyzes responses for insights and priorities",
                "step_4": "GTM weights auto-adjust based on team input",
                "step_5": "Follow-up questions generated for deeper insights"
            },
            "example_questions": demo_questions,
            "ai_analysis": ai_analysis,
            "value_proposition": [
                "🎯 GTM strategy driven by ACTUAL team priorities",
                "📊 Weights adjust based on REAL team confidence",
                "🤖 AI asks smart follow-up questions",
                "⚡ Real-time strategy adaptation",
                "🗣️ Every team member's voice influences strategy"
            ],
            "next_steps": [
                "🔑 Add OpenAI API key to unlock full AI question generation",
                "💬 Deploy questions to team Slack channels",
                "📈 Watch GTM dashboard adapt to team responses",
                "🔄 Enable continuous strategic feedback loops"
            ]
        }
        
    except Exception as e:
        logger.error(f"Team engagement demo failed: {e}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

@app.get("/api/digest/ai-powered")
async def get_ai_powered_digest(
    date: Optional[str] = None,
    db: asyncpg.Pool = Depends(get_db)
):
    """Get AI-powered digest using GTM Risk Commander analysis"""
    try:
        target_date = date or datetime.utcnow().strftime("%Y-%m-%d")
        
        # Get messages from today
        async with db.acquire() as conn:
            messages = await conn.fetch("""
                SELECT slack_message_id, content, timestamp, priority_level, 
                       sentiment_score, metadata
                FROM messages 
                WHERE DATE(timestamp) = CURRENT_DATE
                ORDER BY timestamp DESC
            """)
        
        if not messages:
            return {
                "error": "No messages found for AI digest generation",
                "date": target_date
            }
        
        # Convert to agent format
        message_data = []
        for row in messages:
            message_data.append({
                "slack_message_id": row['slack_message_id'],
                "content": row['content'],
                "timestamp": row['timestamp'],
                "priority_level": row['priority_level'],
                "sentiment_score": row['sentiment_score'],
                "metadata": row['metadata']
            })
        
        # 🤖 Generate AI-powered digest
        logger.info(f"🤖 Generating AI-powered digest for {len(message_data)} messages")
        agent_result = await agent_manager.analyze_messages(message_data)
        
        if not agent_result.success:
            return {
                "error": f"AI digest generation failed: {agent_result.error_message}",
                "date": target_date
            }
        
        # Create AI-enhanced digest
        ai_digest = {
            "date": target_date,
            "ai_powered": True,
            "agent_analysis": {
                "agent_id": agent_result.agent_id,
                "overall_risk_score": agent_result.overall_risk_score,
                "ai_summary": agent_result.summary,
                "ai_recommendations": agent_result.recommendations,
                "confidence": agent_result.confidence,
                "risk_assessments": agent_result.risk_assessments
            },
            "message_stats": {
                "total_analyzed": len(message_data),
                "analysis_timestamp": agent_result.metadata.get("analysis_timestamp"),
                "agent_model": agent_result.metadata.get("agent_model", "gpt-4-turbo-preview")
            }
        }
        
        # Determine AI-powered status
        risk_score = agent_result.overall_risk_score or 50
        if risk_score >= 80:
            status = "🟢 GREEN - AI Assessment: Low Risk"
        elif risk_score >= 60:
            status = "🟡 YELLOW - AI Assessment: Moderate Risk" 
        elif risk_score >= 40:
            status = "🟠 ORANGE - AI Assessment: High Risk"
        else:
            status = "🔴 RED - AI Assessment: Critical Risk"
        
        ai_digest["status"] = status
        ai_digest["risk_level"] = risk_score
        
        # Format for Slack
        formatted_digest = f"""🤖 **AI-Powered Hardware GTM Digest - {target_date}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **AI RISK SCORE: {risk_score}/100** (Confidence: {agent_result.confidence*100:.1f}%)
**Status:** {status}

🧠 **AI INSIGHTS**
{agent_result.summary}

🚨 **AI-IDENTIFIED RISKS ({len(agent_result.risk_assessments)})**"""

        for i, risk in enumerate(agent_result.risk_assessments[:3], 1):
            risk_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(risk.get("level", "medium"), "⚪")
            formatted_digest += f"""
{i}. {risk_emoji} **[{risk.get("category", "Unknown").title()}]** {risk.get("description", "")[:100]}...
   **Impact:** {risk.get("impact", "Assessment pending")}
   **AI Confidence:** {risk.get("confidence", 0)*100:.1f}%"""

        formatted_digest += f"""

🤖 **AI RECOMMENDATIONS**"""
        for rec in agent_result.recommendations[:5]:
            formatted_digest += f"\n• {rec}"

        formatted_digest += f"""

📊 **AI ANALYSIS STATS**
• Model: {agent_result.metadata.get("agent_model", "GPT-4")}
• Messages Analyzed: {len(message_data)}
• Analysis Confidence: {agent_result.confidence*100:.1f}%
• Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}"""

        ai_digest["formatted_digest"] = formatted_digest
        
        return ai_digest
        
    except Exception as e:
        logger.error(f"AI-powered digest generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI digest generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 