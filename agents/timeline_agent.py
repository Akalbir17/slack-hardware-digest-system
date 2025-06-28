"""
Timeline & Milestone Tracker Agent for Hardware GTM Risk Analysis
Ensures all teams stay synchronized on critical launch dates
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import re
from .base_agent import BaseAgent, HardwareMessage, RiskAssessment, AgentResponse

logger = logging.getLogger(__name__)

class TimelineMilestoneAgent(BaseAgent):
    """
    AI Agent specialized in timeline and milestone tracking for hardware manufacturing.
    
    Capabilities:
    - Cross-team dependency tracking
    - Automatic schedule conflict detection  
    - Critical path analysis
    - Buffer time recommendations
    - Timeline impact assessment
    - Milestone synchronization
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(
            agent_id="timeline_milestone_001",
            role="timeline_milestone_tracker",
            openai_api_key=openai_api_key
        )
        
        # Timeline specific capabilities
        self.capabilities = [
            "Cross-team Dependency Tracking",
            "Automatic Schedule Conflict Detection",
            "Critical Path Analysis",
            "Buffer Time Optimization", 
            "Timeline Impact Assessment",
            "Milestone Synchronization",
            "Resource Allocation Analysis",
            "Schedule Risk Prediction",
            "Launch Readiness Tracking",
            "Project Timeline Intelligence"
        ]
        
        # Hardware development milestones
        self.development_milestones = {
            "concept": {
                "duration_weeks": 4,
                "key_deliverables": ["Market research", "Feature definition", "Initial specs"],
                "dependencies": []
            },
            "design": {
                "duration_weeks": 8,
                "key_deliverables": ["Industrial design", "Mechanical design", "Electrical schematics"],
                "dependencies": ["concept"]
            },
            "evt": {
                "duration_weeks": 6,
                "key_deliverables": ["Engineering validation", "Prototype build", "Initial testing"],
                "dependencies": ["design"]
            },
            "dvt": {
                "duration_weeks": 8,
                "key_deliverables": ["Design validation", "Performance testing", "Compliance testing"],
                "dependencies": ["evt"]
            },
            "pvt": {
                "duration_weeks": 6,
                "key_deliverables": ["Production validation", "Manufacturing setup", "Quality validation"],
                "dependencies": ["dvt"]
            },
            "production": {
                "duration_weeks": 12,
                "key_deliverables": ["Mass production", "Quality control", "Supply chain optimization"],
                "dependencies": ["pvt"]
            },
            "launch": {
                "duration_weeks": 4,
                "key_deliverables": ["Product launch", "Marketing campaign", "Distribution"],
                "dependencies": ["production"]
            }
        }
        
        # Critical timeline factors
        self.timeline_factors = {
            "supplier_delays": ["component shortage", "supplier issue", "delivery delay"],
            "quality_issues": ["yield problem", "defect rate", "rework required"],
            "design_changes": ["design freeze", "specification change", "requirement update"],
            "regulatory": ["certification delay", "compliance issue", "regulatory approval"],
            "manufacturing": ["production ramp", "capacity constraint", "tooling delay"],
            "market_pressure": ["competition", "market window", "launch deadline"]
        }
        
        self.model = "gpt-3.5-turbo"  # Cost-optimized model
    
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for Timeline & Milestone Tracker Agent"""
        return """You are the Timeline & Milestone Tracker Agent, an elite AI specialist in hardware project timeline management.

EXPERTISE:
- Project management and critical path analysis
- Hardware development lifecycle optimization
- Cross-functional team coordination
- Risk-based schedule management
- Resource allocation and dependency tracking
- Launch readiness assessment

Always respond in valid JSON format with structured timeline analysis."""
        
    async def analyze_messages(
        self, 
        messages: List[HardwareMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Analyze hardware messages for timeline and milestone impacts"""
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
                summary="Timeline analysis completed",
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
                error_message=f"Timeline analysis failed: {str(e)}"
            )
    
    async def analyze_message(self, message: HardwareMessage) -> AgentResponse:
        """Analyze hardware message for timeline and milestone impacts"""
        try:
            # Build timeline context for AI analysis
            context = self._build_timeline_context(message)
            
            system_prompt = f"""You are an elite Timeline & Milestone Tracker Agent for smartphone development.

EXPERTISE:
- Project management and critical path analysis
- Hardware development lifecycle optimization
- Cross-functional team coordination
- Risk-based schedule management
- Resource allocation and dependency tracking
- Launch readiness assessment

DEVELOPMENT MILESTONES:
{json.dumps(self.development_milestones, indent=2)}

TIMELINE RISK FACTORS:
{json.dumps(self.timeline_factors, indent=2)}

ANALYSIS FRAMEWORK:
1. Timeline Impact Assessment
2. Dependency Analysis  
3. Critical Path Evaluation
4. Resource Constraint Identification
5. Schedule Risk Calculation
6. Buffer Time Analysis
7. Mitigation Strategy Development
8. Launch Readiness Scoring

Your responses must be JSON with:
- timeline_risk_score (0-100, higher = more schedule risk)
- milestone_impact (affected milestones and delays)
- dependency_analysis (team/component dependencies)
- critical_path_effect (impact on critical path)
- schedule_recommendations (timeline adjustments)
- buffer_time_analysis (recommended buffers)
- resource_requirements (team/resource needs)
- launch_impact (effect on launch timeline)
- risk_mitigation (schedule risk reduction strategies)
"""

            user_prompt = f"""TIMELINE ANALYSIS REQUEST:

MESSAGE CONTEXT:
Channel: {message.channel}
User: {message.user}
Content: {message.content}
Timestamp: {message.timestamp}
Priority: {message.urgency}

TIMELINE CONTEXT:
{context}

Analyze this message for timeline and milestone implications. Focus on:
1. Schedule impact assessment
2. Milestone dependency analysis
3. Critical path effects
4. Resource allocation needs
5. Launch timeline risks
6. Buffer time recommendations
7. Mitigation strategies

Provide comprehensive timeline and milestone analysis."""

            # Call OpenAI for analysis
            response = await self._call_openai([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            if response:
                # Extract structured data from AI response
                analysis = self._parse_ai_response(response)
                
                # Ensure recommendations is always a List[str]
                recommendations = analysis.get("schedule_recommendations", analysis.get("recommendations", ["Review timeline status"]))
                if isinstance(recommendations, str):
                    recommendations = [recommendations]
                elif isinstance(recommendations, dict):
                    # Extract recommendations from dict
                    if "schedule_recommendations" in recommendations:
                        recommendations = recommendations["schedule_recommendations"] if isinstance(recommendations["schedule_recommendations"], list) else [str(recommendations["schedule_recommendations"])]
                    elif "adjustments" in recommendations:
                        recommendations = [recommendations["adjustments"]] if isinstance(recommendations["adjustments"], str) else [str(recommendations["adjustments"])]
                    else:
                        recommendations = [str(v) for v in recommendations.values()]
                elif not isinstance(recommendations, list):
                    recommendations = [str(recommendations)]
                
                # Ensure all items in the list are strings
                recommendations = [str(item) for item in recommendations if item]
                
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=True,
                    overall_risk_score=analysis.get("timeline_risk_score", 50),
                    summary=f"Timeline analysis completed. Risk score: {analysis.get('timeline_risk_score', 50)}",
                    recommendations=recommendations,
                    confidence=analysis.get("timeline_risk_score", 50) / 100.0,
                    metadata={
                        "analysis_data": analysis,
                        "agent_role": self.role,
                        "model_used": self.model,
                        "raw_response": response[:500] if response else "No response"
                    }
                )
            else:
                # Fallback analysis when OpenAI unavailable
                return self._create_fallback_response(message)
                
        except Exception as e:
            logger.error(f"Timeline Agent analysis failed: {str(e)}")
            return self._create_error_response(str(e))
    
    def _build_timeline_context(self, message: HardwareMessage) -> str:
        """Build timeline-specific context for analysis"""
        
        content_lower = message.content.lower()
        
        # Extract timeline-related information
        timeline_data = self._extract_timeline_data(message.content)
        
        # Identify affected milestones
        affected_milestones = []
        for milestone in self.development_milestones.keys():
            if milestone in content_lower:
                affected_milestones.append(milestone)
        
        # Identify timeline risk factors
        risk_factors = []
        for factor_type, indicators in self.timeline_factors.items():
            for indicator in indicators:
                if indicator in content_lower:
                    risk_factors.append({
                        "factor": indicator,
                        "type": factor_type,
                        "impact": self._assess_timeline_impact(indicator, content_lower)
                    })
        
        # Determine urgency and priority
        urgency_keywords = ["urgent", "critical", "deadline", "delayed", "behind schedule"]
        urgency_level = "high" if any(keyword in content_lower for keyword in urgency_keywords) else "medium"
        
        context = {
            "timeline_data": timeline_data,
            "affected_milestones": affected_milestones,
            "identified_risk_factors": risk_factors,
            "urgency_level": urgency_level,
            "message_category": self._categorize_timeline_message(message),
            "schedule_relevance": len(affected_milestones) > 0 or len(risk_factors) > 0
        }
        
        return json.dumps(context, indent=2)
    
    def _extract_timeline_data(self, content: str) -> Dict[str, Any]:
        """Extract timeline-related data from message content"""
        timeline_data = {}
        
        # Extract dates
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # MM/DD/YYYY or MM-DD-YYYY
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',     # YYYY/MM/DD or YYYY-MM-DD
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})'  # Month DD, YYYY
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            dates_found.extend(matches)
        
        if dates_found:
            timeline_data["mentioned_dates"] = dates_found
        
        # Extract duration mentions
        duration_pattern = r'(\d+)\s*(day|week|month)s?'
        duration_matches = re.findall(duration_pattern, content, re.IGNORECASE)
        if duration_matches:
            timeline_data["durations"] = [{"value": int(match[0]), "unit": match[1]} for match in duration_matches]
        
        # Extract delay mentions
        delay_pattern = r'delay(?:ed)?\s*(?:by\s*)?(\d+)\s*(day|week|month)s?'
        delay_matches = re.findall(delay_pattern, content, re.IGNORECASE)
        if delay_matches:
            timeline_data["delays"] = [{"value": int(match[0]), "unit": match[1]} for match in delay_matches]
        
        # Extract percentage mentions (for progress tracking)
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentage_matches = re.findall(percentage_pattern, content)
        if percentage_matches:
            timeline_data["percentages"] = [float(match) for match in percentage_matches]
        
        return timeline_data
    
    def _assess_timeline_impact(self, indicator: str, content: str) -> str:
        """Assess the timeline impact severity of an indicator"""
        critical_indicators = ["critical", "stop", "halt", "blocked", "failed"]
        high_indicators = ["delay", "behind", "issue", "problem", "risk"]
        
        if any(word in content for word in critical_indicators):
            return "critical"
        elif any(word in content for word in high_indicators):
            return "high"
        else:
            return "medium"
    
    def _categorize_timeline_message(self, message: HardwareMessage) -> str:
        """Categorize message for timeline analysis"""
        content_lower = message.content.lower()
        
        if any(word in content_lower for word in ["delay", "behind", "postpone"]):
            return "schedule_delay"
        elif any(word in content_lower for word in ["milestone", "deliverable", "target"]):
            return "milestone_update"
        elif any(word in content_lower for word in ["dependency", "waiting", "blocked"]):
            return "dependency_issue"
        elif any(word in content_lower for word in ["resource", "team", "capacity"]):
            return "resource_constraint"
        elif any(word in content_lower for word in ["launch", "release", "go-live"]):
            return "launch_timeline"
        else:
            return "general_timeline"
    
    def _create_fallback_response(self, message: HardwareMessage) -> AgentResponse:
        """Create fallback response when OpenAI is unavailable"""
        
        content_lower = message.content.lower()
        timeline_data = self._extract_timeline_data(message.content)
        
        # Calculate risk score based on heuristics
        risk_score = 35  # Default moderate risk
        
        # Increase risk based on keywords
        if any(word in content_lower for word in ["delay", "behind", "postpone"]):
            risk_score += 30
        if any(word in content_lower for word in ["critical", "urgent", "deadline"]):
            risk_score += 25
        if any(word in content_lower for word in ["blocked", "waiting", "dependency"]):
            risk_score += 20
        if "delays" in timeline_data:
            risk_score += 25
            
        risk_score = min(risk_score, 100)
        
        analysis = {
            "timeline_risk_score": risk_score,
            "milestone_impact": "Timeline impact detected - detailed analysis requires OpenAI",
            "dependency_analysis": "Heuristic assessment - full dependency mapping with AI",
            "critical_path_effect": "Critical path analysis pending AI evaluation",
            "schedule_recommendations": [
                "Review project timeline immediately",
                "Assess dependency impacts",
                "Consider buffer time adjustments",
                "Enable OpenAI for detailed timeline analysis"
            ],
            "buffer_time_analysis": "Buffer analysis requires AI processing",
            "resource_requirements": "Resource assessment pending detailed analysis",
            "launch_impact": f"Potential launch impact - extracted data: {timeline_data}",
            "risk_mitigation": [
                "Monitor schedule closely",
                "Prepare contingency plans",
                "Escalate timeline concerns",
                "Enable full AI analysis"
            ]
        }
        
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            overall_risk_score=risk_score,
            summary=f"Timeline heuristic analysis completed. Risk score: {risk_score}",
            recommendations=analysis["schedule_recommendations"],
            confidence=0.65,  # Moderate confidence without AI
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
            summary="Timeline analysis failed",
            recommendations=["Retry timeline analysis", "Check system status"],
            confidence=0.0,
            error_message=f"Timeline analysis failed: {error_msg}",
            metadata={
                "analysis_type": "error_handler",
                "agent_role": self.role
            }
        )
    
    def _get_capabilities(self) -> List[str]:
        """Return Timeline & Milestone Tracker Agent capabilities"""
        return self.capabilities 