"""
Quality Anomaly Detection Agent for Hardware GTM Risk Analysis
Uses pattern recognition to identify quality issues before they escalate
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import re
from .base_agent import BaseAgent, HardwareMessage, RiskAssessment, AgentResponse

logger = logging.getLogger(__name__)

class QualityAnomalyDetectionAgent(BaseAgent):
    """
    AI Agent specialized in quality anomaly detection for hardware manufacturing.
    
    Capabilities:
    - Real-time defect pattern analysis
    - Predictive yield forecasting
    - Root cause identification  
    - Compliance monitoring
    - Quality trend analysis
    - Process optimization recommendations
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(
            agent_id="quality_anomaly_001",
            role="quality_anomaly_detection",
            openai_api_key=openai_api_key
        )
        
        # Quality specific capabilities
        self.capabilities = [
            "Real-time Defect Pattern Analysis",
            "Predictive Yield Forecasting",
            "Root Cause Identification",
            "Statistical Process Control", 
            "Compliance Monitoring",
            "Quality Trend Analysis",
            "Process Optimization",
            "Defect Classification",
            "Yield Rate Prediction",
            "Manufacturing Quality Intelligence"
        ]
        
        # Quality metrics and thresholds
        self.quality_thresholds = {
            "yield_rate": {
                "excellent": 98.0,
                "good": 95.0,
                "acceptable": 92.0,
                "poor": 90.0,
                "critical": 85.0
            },
            "defect_rate": {
                "excellent": 0.5,
                "good": 1.0,
                "acceptable": 2.0,
                "poor": 3.0,
                "critical": 5.0
            },
            "dppm": {  # Defects Per Million Parts
                "excellent": 100,
                "good": 500,
                "acceptable": 1000,
                "poor": 2000,
                "critical": 5000
            }
        }
        
        # Manufacturing phases and quality focus areas
        self.manufacturing_phases = {
            "EVT": ["Engineering validation", "Initial prototypes", "Basic functionality"],
            "DVT": ["Design validation", "Improved prototypes", "Performance testing"],
            "PVT": ["Production validation", "Final design", "Mass production readiness"],
            "MP": ["Mass production", "Volume manufacturing", "Ongoing quality control"]
        }
        
        self.model = "gpt-3.5-turbo"  # Cost-optimized model
    
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for Quality Anomaly Detection Agent"""
        return """You are the Quality Anomaly Detection Agent, an elite AI specialist in hardware manufacturing quality analysis.

EXPERTISE:
- Statistical process control and quality engineering
- Defect pattern recognition and analysis
- Yield rate forecasting and optimization
- Root cause analysis methodologies
- Manufacturing quality standards (ISO, Six Sigma)
- Predictive quality analytics

Always respond in valid JSON format with structured quality analysis."""
        
    async def analyze_messages(
        self, 
        messages: List[HardwareMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Analyze hardware messages for quality anomalies"""
        try:
            if not messages:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message="No messages provided for analysis"
                )
            
            # Analyze first message for now
            message = messages[0]
            result = await self.analyze_message(message)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                overall_risk_score=int(result.confidence * 100),
                summary="Quality analysis completed",
                recommendations=result.recommendations,
                confidence=result.confidence,
                metadata={
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "message_count": len(messages),
                    "analysis_version": "v2.0",
                    "agent_model": self.model
                }
            )
        except Exception as e:
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                error_message=f"Quality analysis failed: {str(e)}"
            )
    
    async def analyze_message(self, message: HardwareMessage) -> AgentResponse:
        """Analyze hardware message for quality anomalies and patterns"""
        try:
            # Build quality context for AI analysis
            context = self._build_quality_context(message)
            
            system_prompt = f"""You are an elite Quality Anomaly Detection Agent for smartphone manufacturing.

EXPERTISE:
- Statistical process control and quality engineering
- Defect pattern recognition and analysis
- Yield rate forecasting and optimization
- Root cause analysis methodologies
- Manufacturing quality standards (ISO, Six Sigma)
- Predictive quality analytics

QUALITY THRESHOLDS:
{json.dumps(self.quality_thresholds, indent=2)}

MANUFACTURING PHASES:
{json.dumps(self.manufacturing_phases, indent=2)}

Your responses must be JSON with:
- quality_risk_score (0-100, higher = more quality risk)
- defect_analysis (pattern detection and classification)
- yield_forecast (predicted yield trends)
- root_cause_analysis (probable causes)
- corrective_actions (immediate and long-term fixes)
- quality_metrics (KPIs and performance indicators)
"""

            user_prompt = f"""QUALITY ANOMALY ANALYSIS REQUEST:

MESSAGE CONTEXT:
Channel: {message.channel}
User: {message.user}
Content: {message.content}
Timestamp: {message.timestamp}
Priority: {message.urgency}

QUALITY CONTEXT:
{context}

Analyze this message for quality anomalies and patterns."""

            # Call OpenAI for analysis
            response = await self._call_openai([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            if response:
                # Extract structured data from AI response  
                analysis = self._parse_ai_response(response)
                
                # Ensure recommendations is always a List[str]
                recommendations = analysis.get("corrective_actions", analysis.get("recommendations", ["Review quality metrics"]))
                if isinstance(recommendations, str):
                    recommendations = [recommendations]
                elif isinstance(recommendations, dict):
                    # Extract recommendations from dict
                    if "corrective_actions" in recommendations:
                        recommendations = recommendations["corrective_actions"] if isinstance(recommendations["corrective_actions"], list) else [str(recommendations["corrective_actions"])]
                    else:
                        recommendations = [str(v) for v in recommendations.values()]
                elif not isinstance(recommendations, list):
                    recommendations = [str(recommendations)]
                
                # Ensure all items in the list are strings
                recommendations = [str(item) for item in recommendations if item]
                
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=True,
                    overall_risk_score=analysis.get("quality_risk_score", 50),
                    summary=f"Quality analysis completed. Risk score: {analysis.get('quality_risk_score', 50)}",
                    recommendations=recommendations,
                    confidence=analysis.get("quality_risk_score", 50) / 100.0,
                    metadata={
                        "analysis_data": analysis,
                        "agent_role": self.role,
                        "model_used": self.model,
                        "raw_response": response[:500] if response else "No response"
                    }
                )
            else:
                return self._create_fallback_response(message)
                
        except Exception as e:
            logger.error(f"Quality Agent analysis failed: {str(e)}")
            return self._create_error_response(str(e))
    
    def _build_quality_context(self, message: HardwareMessage) -> str:
        """Build quality-specific context for analysis"""
        content_lower = message.content.lower()
        
        # Extract quality metrics from message
        quality_metrics = self._extract_quality_metrics(message.content)
        
        context = {
            "extracted_metrics": quality_metrics,
            "message_category": self._categorize_quality_message(message),
            "urgency_level": "high" if any(word in content_lower for word in ["critical", "failure"]) else "medium"
        }
        
        return json.dumps(context, indent=2)
    
    def _extract_quality_metrics(self, content: str) -> Dict[str, Any]:
        """Extract numerical quality metrics from message content"""
        metrics = {}
        
        # Yield rate patterns
        yield_pattern = r'yield.*?(\d+(?:\.\d+)?)\s*%'
        yield_matches = re.findall(yield_pattern, content, re.IGNORECASE)
        if yield_matches:
            metrics["yield_rate"] = float(yield_matches[-1])
        
        # Defect rate patterns  
        defect_pattern = r'defect.*?(\d+(?:\.\d+)?)\s*%'
        defect_matches = re.findall(defect_pattern, content, re.IGNORECASE)
        if defect_matches:
            metrics["defect_rate"] = float(defect_matches[-1])
        
        return metrics
    
    def _categorize_quality_message(self, message: HardwareMessage) -> str:
        """Categorize message for quality analysis"""
        content_lower = message.content.lower()
        
        if any(word in content_lower for word in ["yield", "production rate"]):
            return "yield_management"
        elif any(word in content_lower for word in ["defect", "failure"]):
            return "defect_analysis"
        else:
            return "general_quality"
    
    def _create_fallback_response(self, message: HardwareMessage) -> AgentResponse:
        """Create fallback response when OpenAI is unavailable"""
        metrics = self._extract_quality_metrics(message.content)
        
        analysis = {
            "quality_risk_score": 45,
            "defect_analysis": "Heuristic analysis - full AI analysis pending",
            "yield_forecast": f"Metrics: {metrics}" if metrics else "No metrics",
            "root_cause_analysis": "Requires AI analysis",
            "corrective_actions": ["Monitor quality metrics", "Enable AI analysis"],
            "quality_metrics": metrics
        }
        
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            overall_risk_score=45,
            summary="Quality heuristic analysis completed",
            recommendations=analysis["corrective_actions"],
            confidence=0.6,
            metadata={
                "analysis_type": "fallback_heuristics",
                "analysis_data": analysis,
                "agent_role": self.role
            }
        )
    
    def _create_error_response(self, error_msg: str) -> AgentResponse:
        """Create error response for failed analysis"""
        return AgentResponse(
            agent_id=self.agent_id,
            success=False,
            overall_risk_score=50,
            summary="Quality analysis failed",
            recommendations=["Retry analysis"],
            confidence=0.0,
            error_message=f"Quality analysis failed: {error_msg}",
            metadata={
                "analysis_type": "error_handler",
                "agent_role": self.role
            }
        )
    
    def _get_capabilities(self) -> List[str]:
        """Return Quality Anomaly Detection Agent capabilities"""
        return self.capabilities 