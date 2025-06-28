"""
Agent Manager - Central orchestration system for all AI agents
Provides unified interface for agent-powered analysis
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base_agent import HardwareMessage, AgentResponse
from .gtm_commander import GTMRiskCommander
from .supply_chain_agent import SupplyChainIntelligenceAgent
from .quality_agent import QualityAnomalyDetectionAgent
from .timeline_agent import TimelineMilestoneAgent

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Central orchestration system for all AI agents
    Provides unified interface for agent-powered hardware GTM analysis
    """
    
    def __init__(self, openai_api_key: str = "your-key-here"):
        self.agents = {}
        self.analysis_history = []
        self.is_initialized = False
        self.openai_api_key = openai_api_key
        
    async def initialize(self) -> bool:
        """Initialize all AI agents"""
        try:
            logger.info("Initializing AI Agent Manager with 4 specialized agents...")
            
            # Initialize all 4 AI agents
            self.agents['gtm_commander'] = GTMRiskCommander(self.openai_api_key)
            self.agents['supply_chain_intel'] = SupplyChainIntelligenceAgent(self.openai_api_key)
            self.agents['quality_anomaly'] = QualityAnomalyDetectionAgent(self.openai_api_key)
            self.agents['timeline_milestone'] = TimelineMilestoneAgent(self.openai_api_key)
            
            # Perform health checks
            healthy_agents = 0
            for agent_id, agent in self.agents.items():
                try:
                    health = await agent.health_check()
                    if health.get('openai_responsive', False):
                        healthy_agents += 1
                        logger.info(f"Agent {agent_id} initialized successfully")
                    else:
                        logger.warning(f"Agent {agent_id} initialized but OpenAI not responsive")
                except Exception as e:
                    logger.error(f"Agent {agent_id} health check failed: {e}")
            
            self.is_initialized = True
            logger.info(f"Agent Manager initialized with {healthy_agents}/{len(self.agents)} healthy agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Manager: {e}")
            return False
    
    def convert_to_hardware_messages(self, messages: List[Dict[str, Any]]) -> List[HardwareMessage]:
        """Convert database message format to HardwareMessage format"""
        hardware_messages = []
        
        for msg in messages:
            try:
                # Extract metadata
                metadata = msg.get('metadata', {})
                if isinstance(metadata, str):
                    import json
                    metadata = json.loads(metadata)
                
                hardware_msg = HardwareMessage(
                    id=msg.get('slack_message_id', str(msg.get('id', ''))),
                    content=msg.get('content', ''),
                    channel=metadata.get('channel', '#unknown'),
                    user=metadata.get('user', 'unknown'),
                    timestamp=msg.get('timestamp', datetime.utcnow()),
                    urgency=metadata.get('urgency', msg.get('urgency', 'medium')),
                    category=metadata.get('category', 'general'),
                    sentiment_score=msg.get('sentiment_score'),
                    metadata=metadata
                )
                hardware_messages.append(hardware_msg)
            except Exception as e:
                logger.error(f"Failed to convert message to HardwareMessage: {e}")
        
        return hardware_messages
    
    async def analyze_messages(
        self, 
        messages: List[Dict[str, Any]], 
        agent_id: str = "gtm_commander",
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Perform AI-powered analysis using specified agent
        """
        if not self.is_initialized:
            logger.warning("Agent Manager not initialized, attempting to initialize...")
            if not await self.initialize():
                return AgentResponse(
                    agent_id="agent_manager",
                    success=False,
                    error_message="Agent Manager initialization failed"
                )
        
        agent = self.agents.get(agent_id)
        if not agent:
            return AgentResponse(
                agent_id="agent_manager",
                success=False,
                error_message=f"Agent {agent_id} not found"
            )
        
        try:
            # Convert to HardwareMessage format
            hardware_messages = self.convert_to_hardware_messages(messages)
            
            if not hardware_messages:
                return AgentResponse(
                    agent_id=agent_id,
                    success=False,
                    error_message="No valid messages provided for analysis"
                )
            
            logger.info(f"🤖 AI Agent analyzing {len(hardware_messages)} messages with {agent_id}")
            
            # Perform agent analysis
            result = await agent.analyze_messages(hardware_messages, context)
            
            # Store in analysis history
            self.analysis_history.append({
                "timestamp": datetime.utcnow(),
                "agent_id": agent_id,
                "message_count": len(hardware_messages),
                "success": result.success,
                "risk_score": result.overall_risk_score
            })
            
            # Keep only last 100 analyses
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            return result
            
        except Exception as e:
            logger.error(f"Agent analysis failed: {e}")
            return AgentResponse(
                agent_id=agent_id,
                success=False,
                error_message=f"Analysis failed: {str(e)}"
            )
    
    async def get_comprehensive_analysis(
        self, 
        messages: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using all 4 specialized agents
        """
        results = {}
        risk_scores = []
        confidences = []
        
        # Run all 4 agents in parallel for maximum efficiency
        agent_analyses = await asyncio.gather(
            self.analyze_messages(messages, "gtm_commander", context),
            self.analyze_messages(messages, "supply_chain_intel", context),
            self.analyze_messages(messages, "quality_anomaly", context),
            self.analyze_messages(messages, "timeline_milestone", context),
            return_exceptions=True
        )
        
        # Process results from each agent
        agent_names = ["gtm_analysis", "supply_chain_analysis", "quality_analysis", "timeline_analysis"]
        
        for i, (name, result) in enumerate(zip(agent_names, agent_analyses)):
            if isinstance(result, Exception):
                logger.error(f"Agent {name} failed: {result}")
                results[name] = {"error": str(result), "success": False}
            else:
                results[name] = result.dict() if hasattr(result, 'dict') else result.__dict__
                if hasattr(result, 'overall_risk_score') and result.success:
                    risk_scores.append(result.overall_risk_score)
                if hasattr(result, 'confidence') and result.success:
                    confidences.append(result.confidence)
        
        # Calculate weighted overall risk score
        if risk_scores:
            # GTM Commander has highest weight, others contribute equally
            gtm_weight = 0.4
            other_weight = 0.6 / max(len(risk_scores) - 1, 1)
            
            overall_risk_score = (
                risk_scores[0] * gtm_weight +  # GTM Commander
                sum(risk_scores[1:]) * other_weight  # Other agents
            )
        else:
            overall_risk_score = 50  # Default moderate risk
        
        # Calculate average confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "comprehensive_analysis": results,
            "overall_risk_score": min(max(overall_risk_score, 0), 100),
            "overall_confidence": min(max(overall_confidence, 0.0), 1.0),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "agent_count": len(self.agents),
            "successful_analyses": len([r for r in results.values() if r.get("success", False)]),
            "message_count": len(messages),
            "multi_agent_intelligence": True
        }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            "is_initialized": self.is_initialized,
            "agent_count": len(self.agents),
            "analysis_history_count": len(self.analysis_history),
            "agents": {}
        }
        
        for agent_id, agent in self.agents.items():
            try:
                agent_health = await agent.health_check()
                agent_info = agent.get_agent_info()
                status["agents"][agent_id] = {
                    "health": agent_health,
                    "info": agent_info
                }
            except Exception as e:
                status["agents"][agent_id] = {
                    "error": str(e),
                    "health": "unknown"
                }
        
        return status
    
    async def get_analysis_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        return self.analysis_history[-limit:] if self.analysis_history else []
    
    def get_available_agents(self) -> Dict[str, List[str]]:
        """Get list of available agents and their capabilities"""
        agents_info = {}
        
        for agent_id, agent in self.agents.items():
            try:
                capabilities = agent._get_capabilities()
                agents_info[agent_id] = capabilities
            except Exception as e:
                agents_info[agent_id] = [f"Error: {str(e)}"]
        
        return agents_info
    
    async def trigger_real_time_analysis(
        self, 
        message: Dict[str, Any],
        broadcast_callback: Optional[callable] = None
    ) -> Optional[AgentResponse]:
        """
        Trigger real-time analysis for a single message
        Used for live message processing
        """
        try:
            # Check if message requires AI analysis
            urgency = message.get('urgency', 'medium')
            if urgency not in ['critical', 'high']:
                return None  # Skip analysis for low priority messages
            
            # Perform quick analysis
            result = await self.analyze_messages([message], "gtm_commander")
            
            # Broadcast results if callback provided
            if broadcast_callback and result.success:
                await broadcast_callback({
                    "type": "ai_analysis",
                    "data": result.__dict__,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time analysis failed: {e}")
            return None
    
    async def generate_intelligent_digest(
        self, 
        messages: List[Dict[str, Any]], 
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered intelligent digest using agents
        """
        try:
            # Perform comprehensive analysis
            analysis = await self.get_comprehensive_analysis(messages)
            gtm_analysis = analysis["comprehensive_analysis"]["gtm_analysis"]
            
            if not gtm_analysis.get("success", False):
                return {"error": "Agent analysis failed"}
            
            # Extract AI insights
            ai_insights = {
                "risk_assessments": gtm_analysis.get("risk_assessments", []),
                "overall_risk_score": gtm_analysis.get("overall_risk_score", 50),
                "ai_summary": gtm_analysis.get("summary", ""),
                "ai_recommendations": gtm_analysis.get("recommendations", []),
                "confidence": gtm_analysis.get("confidence", 0.0),
                "agent_metadata": gtm_analysis.get("metadata", {})
            }
            
            # Determine AI-powered status
            risk_score = ai_insights["overall_risk_score"]
            if risk_score >= 80:
                status = "GREEN - AI Assessment: Low Risk"
            elif risk_score >= 60:
                status = "YELLOW - AI Assessment: Moderate Risk"
            elif risk_score >= 40:
                status = "ORANGE - AI Assessment: High Risk"
            else:
                status = "RED - AI Assessment: Critical Risk"
            
            return {
                "ai_powered": True,
                "analysis_agent": "gtm_commander",
                "risk_score": risk_score,
                "status": status,
                "ai_insights": ai_insights,
                "generated_by": "AI Agent System",
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Intelligent digest generation failed: {e}")
            return {"error": f"AI digest generation failed: {str(e)}"}

# Global agent manager instance
agent_manager = AgentManager() 