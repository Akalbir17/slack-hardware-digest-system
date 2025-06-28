"""
Base AI Agent Framework for Slack GTM Helper
Provides foundation for specialized hardware analysis agents
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import openai
from pydantic import BaseModel, Field


# Configure logging
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Agent role definitions"""
    RISK_COMMANDER = "gtm_risk_commander"
    SUPPLY_CHAIN_ANALYST = "supply_chain_analyst"
    QUALITY_GUARDIAN = "quality_guardian"
    TIMELINE_TRACKER = "timeline_tracker"

class RiskLevel(Enum):
    """Risk level definitions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HardwareMessage:
    """Hardware team message structure"""
    id: str
    content: str
    channel: str
    user: str
    timestamp: datetime
    urgency: str
    category: str
    sentiment_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    category: str
    level: RiskLevel
    score: int  # 1-100
    confidence: float  # 0.0-1.0
    description: str
    impact: str
    recommendations: List[str]
    urgency: str
    requires_attention: bool

class AgentResponse(BaseModel):
    """Response from an AI agent"""
    agent_id: str
    success: bool
    overall_risk_score: Optional[int] = 50
    summary: Optional[str] = ""
    recommendations: List[str] = []
    confidence: Optional[float] = 0.0
    risk_assessments: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @property
    def analysis_data(self):
        """Fallback property for legacy code accessing analysis_data directly"""
        return self.metadata.get("analysis_data", {})

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in the GTM system
    Provides common functionality for OpenAI integration, message processing, and risk analysis
    """
    
    def __init__(
        self, 
        agent_id: str,
        role: str,
        openai_api_key: str = "your-key-here",
        model: str = "gpt-3.5-turbo",  # Much cheaper model
        max_tokens: int = 1000,        # Reduced for cost efficiency
        temperature: float = 0.3
    ):
        self.agent_id = agent_id
        self.role = role
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        self._initialize_openai()
        
        # Agent state
        self.last_analysis = None
        self.message_history = []
        self.analysis_count = 0
        
    def _initialize_openai(self):
        """Initialize OpenAI client with API key"""
        # Use provided API key, fallback to environment variable
        api_key = self.openai_api_key
        if api_key == "your-key-here":
            api_key = os.getenv("OPENAI_API_KEY", "your-key-here")
            
        if not api_key or api_key == "your-key-here":
            logger.warning(f"OpenAI API key not found for agent {self.agent_id}")
            return
            
        try:
            self.client = openai.AsyncOpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client for {self.agent_id}: {e}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    @abstractmethod
    async def analyze_messages(
        self, 
        messages: List[HardwareMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Analyze hardware messages and return risk assessment"""
        pass
    
    async def _call_openai(
        self, 
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Optional[str]:
        """Make async call to OpenAI API"""
        if not self.client:
            logger.error(f"OpenAI client not available for agent {self.agent_id}")
            return None
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed for {self.agent_id}: {e}")
            return None
    
    async def _call_openai_structured(
        self, 
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Make structured OpenAI call with JSON response format"""
        if not self.client:
            logger.error(f"OpenAI client not available for agent {self.agent_id}")
            return None
            
        try:
            # Use structured outputs for more reliable parsing
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            if response_format:
                params["response_format"] = {"type": "json_object"}
                
            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Structured OpenAI API call failed for {self.agent_id}: {e}")
            return None
    
    def _format_messages_for_analysis(self, messages: List[HardwareMessage]) -> str:
        """Format hardware messages for AI analysis"""
        formatted = []
        for msg in messages:
            formatted.append(f"""
Message ID: {msg.id}
Channel: {msg.channel}
User: {msg.user}
Timestamp: {msg.timestamp}
Urgency: {msg.urgency}
Category: {msg.category}
Content: {msg.content}
Sentiment: {msg.sentiment_score if msg.sentiment_score else 'Unknown'}
---""")
        return "\n".join(formatted)
    
    def _calculate_overall_risk_score(self, risk_assessments: List[RiskAssessment]) -> int:
        """Calculate overall risk score from individual assessments"""
        if not risk_assessments:
            return 50  # Default neutral score
            
        # Weight risk scores by confidence and severity
        weighted_scores = []
        for assessment in risk_assessments:
            weight = assessment.confidence
            if assessment.level == RiskLevel.CRITICAL:
                weight *= 1.5
            elif assessment.level == RiskLevel.HIGH:
                weight *= 1.2
            
            weighted_scores.append(assessment.score * weight)
        
        if weighted_scores:
            return int(sum(weighted_scores) / len(weighted_scores))
        return 50
    
    def _parse_risk_level(self, level_str: str) -> RiskLevel:
        """Parse risk level from string"""
        level_map = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL
        }
        return level_map.get(level_str.lower(), RiskLevel.MEDIUM)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health and OpenAI connectivity"""
        status = {
            "agent_id": self.agent_id,
            "role": self.role,
            "openai_available": self.client is not None,
            "last_analysis": self.last_analysis,
            "analysis_count": self.analysis_count,
            "model": self.model
        }
        
        if self.client:
            try:
                # Test OpenAI connection with minimal call
                test_response = await self._call_openai([
                    {"role": "user", "content": "Hello"}
                ])
                status["openai_responsive"] = test_response is not None
            except Exception as e:
                status["openai_responsive"] = False
                status["openai_error"] = str(e)
        
        return status
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and capabilities"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "model": self.model,
            "capabilities": self._get_capabilities(),
            "analysis_count": self.analysis_count,
            "last_analysis": self.last_analysis
        }
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response text and extract JSON data"""
        try:
            import json
            
            # Try to find JSON in the response
            if not response_text:
                return {}
            
            # Clean the response
            response_text = response_text.strip()
            
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed_data = json.loads(json_text)
            else:
                # Fallback: try to parse the entire response as JSON
                parsed_data = json.loads(response_text)
            
            # Ensure recommendations is always a list
            if "recommendations" in parsed_data:
                recs = parsed_data["recommendations"]
                if isinstance(recs, str):
                    # Convert string to list (split by sentences or newlines)
                    parsed_data["recommendations"] = [recs.strip()]
                elif isinstance(recs, dict):
                    # Extract list from dict values
                    if "actionable_steps" in recs:
                        parsed_data["recommendations"] = recs["actionable_steps"] if isinstance(recs["actionable_steps"], list) else [str(recs["actionable_steps"])]
                    elif "adjustments" in recs:
                        parsed_data["recommendations"] = [recs["adjustments"]] if isinstance(recs["adjustments"], str) else [str(recs["adjustments"])]
                    else:
                        # Convert dict values to list
                        rec_list = []
                        for key, value in recs.items():
                            if isinstance(value, list):
                                rec_list.extend(value)
                            else:
                                rec_list.append(str(value))
                        parsed_data["recommendations"] = rec_list
                elif not isinstance(recs, list):
                    # Convert any other type to list
                    parsed_data["recommendations"] = [str(recs)]
            
            # Also handle other common fields that might need list conversion
            for field in ["corrective_actions", "schedule_recommendations", "actionable_steps"]:
                if field in parsed_data:
                    value = parsed_data[field]
                    if isinstance(value, str):
                        parsed_data[field] = [value]
                    elif isinstance(value, dict):
                        parsed_data[field] = [str(v) for v in value.values()]
                    elif not isinstance(value, list):
                        parsed_data[field] = [str(value)]
                        
            return parsed_data
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            # Return basic fallback structure
            return {
                "risk_score": 50,
                "analysis": "Failed to parse AI response",
                "recommendations": ["Retry analysis", "Check AI response format"],
                "confidence": 0.5
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing AI response: {e}")
            return {
                "error": str(e),
                "risk_score": 50,
                "recommendations": ["Retry analysis"]
            }

    @abstractmethod
    def _get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass 