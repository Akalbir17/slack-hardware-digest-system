"""
GTM Risk Commander Agent - The AI Superpower of Hardware GTM Analysis
Specializes in analyzing hardware team communications for GTM risk assessment
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .base_agent import (
    BaseAgent, AgentRole, RiskLevel, HardwareMessage, 
    RiskAssessment, AgentResponse
)

logger = logging.getLogger(__name__)

class GTMRiskCommander(BaseAgent):
    """
    Elite AI Agent specializing in Hardware GTM Risk Analysis
    
    Capabilities:
    - Deep understanding of hardware development phases (EVT/DVT/PVT)
    - Advanced supply chain risk assessment
    - Quality metrics analysis (yield rates, DPPM, test results)
    - Timeline impact prediction
    - Stakeholder sentiment analysis
    - Actionable recommendation generation
    """
    
    def __init__(self, openai_api_key: str = "your-key-here"):
        super().__init__(
            agent_id="gtm_risk_commander_001",
            role="gtm_risk_commander",
            openai_api_key=openai_api_key
        )
        
        # GTM specific configuration - optimized for cost
        self.model = "gpt-3.5-turbo"  # Much cheaper than GPT-4
        self.max_tokens = 1200        # Reduced for cost efficiency
        self.temperature = 0.2        # Lower temperature for more consistent analysis
        
        # Hardware expertise knowledge base
        self.hardware_phases = {
            "EVT": "Engineering Validation Test - Initial design validation",
            "DVT": "Design Validation Test - Design refinement and validation", 
            "PVT": "Production Validation Test - Pre-production validation",
            "MP": "Mass Production - Full production launch"
        }
        
        self.quality_metrics = {
            "yield_rate": {"critical_threshold": 85, "target_threshold": 95},
            "dppm": {"critical_threshold": 5000, "target_threshold": 1000},
            "pass_rate": {"critical_threshold": 85, "target_threshold": 98}
        }
        
        self.supply_chain_components = [
            "display", "oled", "amoled", "mini-led", "lcd",
            "chipset", "soc", "processor", "snapdragon", "mediatek", "exynos",
            "memory", "ddr", "lpddr", "ufs", "emmc", "nand",
            "battery", "li-ion", "lifepo4", "solid-state",
            "sensor", "camera", "gyroscope", "accelerometer", "magnetometer",
            "connector", "usb", "audio", "wireless", "5g", "wifi", "bluetooth"
        ]
        
        self.risk_categories = {
            "supply_chain": ["shortage", "supplier", "lead_time", "allocation", "inventory"],
            "quality": ["yield", "dppm", "defect", "failure", "test", "certification"],
            "timeline": ["delay", "milestone", "evt", "dvt", "pvt", "schedule"],
            "cost": ["price", "margin", "budget", "overrun", "inflation"],
            "regulatory": ["certification", "compliance", "fcc", "ce", "carrier"]
        }
    
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for GTM Risk Commander"""
        return f"""You are the GTM Risk Commander, an elite AI agent specializing in Hardware Go-To-Market risk analysis. You have deep expertise in:

HARDWARE DEVELOPMENT PHASES:
- EVT (Engineering Validation Test): Initial design validation, prototyping
- DVT (Design Validation Test): Design refinement, feature validation  
- PVT (Production Validation Test): Pre-production validation, final testing
- MP (Mass Production): Full production launch

QUALITY METRICS EXPERTISE:
- Yield Rates: Manufacturing success rate (Target: >95%, Critical: <85%)
- DPPM (Defects Per Million Parts): Quality metric (Target: <1000, Critical: >5000)
- Pass Rates: Test success rates (Target: >98%, Critical: <85%)
- Thermal testing, environmental testing, stress testing

SUPPLY CHAIN COMPONENTS:
{', '.join(self.supply_chain_components)}

RISK ASSESSMENT FRAMEWORK:
1. Supply Chain Risk: Component availability, supplier stability, lead times
2. Quality Risk: Yield rates, defect rates, testing failures, certification delays
3. Timeline Risk: Milestone delays, dependency issues, resource constraints
4. Cost Risk: Price inflation, margin compression, budget overruns
5. Regulatory Risk: Certification delays, compliance issues

ANALYSIS INSTRUCTIONS:
- Analyze each message for hardware-specific risks
- Extract quantitative metrics (percentages, DPPM, timelines)
- Identify critical path dependencies
- Assess supplier and vendor impacts
- Calculate risk scores: 1-20 (Low), 21-40 (Medium), 41-70 (High), 71-100 (Critical)
- Provide specific, actionable recommendations
- Consider cascading effects and timeline impacts

RESPONSE FORMAT:
Always respond in valid JSON format with structured risk analysis."""

    async def analyze_messages(
        self, 
        messages: List[HardwareMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Perform comprehensive GTM risk analysis on hardware messages
        """
        try:
            self.analysis_count += 1
            logger.info(f"GTM Risk Commander analyzing {len(messages)} messages")
            
            if not messages:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message="No messages provided for analysis"
                )
            
            # Format messages for AI analysis
            formatted_messages = self._format_messages_for_analysis(messages)
            
            # Enhanced context analysis
            analysis_context = self._build_analysis_context(messages, context)
            
            # Perform OpenAI analysis
            risk_analysis = await self._perform_ai_risk_analysis(formatted_messages, analysis_context)
            
            if not risk_analysis:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message="Failed to perform AI risk analysis"
                )
            
            # Parse and structure the results
            risk_assessments = self._parse_risk_analysis(risk_analysis, messages)
            overall_risk_score = self._calculate_overall_risk_score(risk_assessments)
            
            # Generate executive summary
            summary = await self._generate_executive_summary(risk_assessments, overall_risk_score, messages)
            
            # Compile comprehensive recommendations
            recommendations = self._compile_recommendations(risk_assessments, overall_risk_score)
            
            self.last_analysis = datetime.utcnow()
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                risk_assessments=[asdict(assessment) for assessment in risk_assessments],
                overall_risk_score=overall_risk_score,
                summary=summary or "GTM risk analysis completed",
                recommendations=recommendations,
                confidence=self._calculate_confidence(risk_assessments),
                metadata={
                    "analysis_timestamp": self.last_analysis.isoformat(),
                    "message_count": len(messages),
                    "analysis_version": "v2.0",
                    "agent_model": self.model
                }
            )
            
        except Exception as e:
            logger.error(f"GTM Risk Commander analysis failed: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                error_message=f"Analysis failed: {str(e)}"
            )
    
    async def _perform_ai_risk_analysis(
        self, 
        formatted_messages: str, 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Perform AI-powered risk analysis using OpenAI"""
        
        analysis_prompt = f"""
HARDWARE GTM RISK ANALYSIS REQUEST

CONTEXT:
- Analysis Date: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}
- Message Count: {context.get('message_count', 0)}
- Time Range: {context.get('time_range', 'Unknown')}
- Priority Messages: {context.get('priority_count', 0)}

MESSAGES TO ANALYZE:
{formatted_messages}

ANALYSIS REQUIREMENTS:
1. Identify ALL hardware-specific risks in these messages
2. Extract quantitative metrics (yield rates, DPPM, percentages, timelines)
3. Assess supply chain component risks
4. Evaluate quality and testing issues
5. Analyze timeline and milestone impacts
6. Calculate precise risk scores (1-100) for each identified risk
7. Provide specific recommendations with clear actions

Return your analysis in this JSON format:
{{
    "overall_assessment": {{
        "risk_level": "low|medium|high|critical",
        "confidence": 0.95,
        "summary": "Brief overall assessment"
    }},
    "risk_assessments": [
        {{
            "category": "supply_chain|quality|timeline|cost|regulatory",
            "risk_level": "low|medium|high|critical", 
            "score": 85,
            "confidence": 0.92,
            "description": "Specific risk description",
            "impact": "Business impact description",
            "urgency": "immediate|high|medium|low",
            "requires_attention": true,
            "recommendations": ["Specific action 1", "Specific action 2"],
            "affected_components": ["component1", "component2"],
            "timeline_impact": "2-3 weeks delay potential",
            "quantitative_data": {{
                "yield_rate": 89.2,
                "dppm": 2400,
                "delay_days": 14
            }}
        }}
    ],
    "key_insights": [
        "Critical insight 1",
        "Critical insight 2"
    ],
    "immediate_actions": [
        "Action requiring immediate attention"
    ]
}}
"""
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": analysis_prompt}
        ]
        
        return await self._call_openai_structured(messages, {"type": "json_object"})
    
    def _build_analysis_context(
        self, 
        messages: List[HardwareMessage], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive context for analysis"""
        
        # Analyze message patterns
        priority_messages = [m for m in messages if m.urgency in ['critical', 'high']]
        categories = {}
        channels = set()
        users = set()
        
        for msg in messages:
            categories[msg.category] = categories.get(msg.category, 0) + 1
            channels.add(msg.channel)
            users.add(msg.user)
        
        time_range = "Unknown"
        if messages:
            timestamps = [msg.timestamp for msg in messages]
            start_time = min(timestamps)
            end_time = max(timestamps)
            time_range = f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
        
        analysis_context = {
            "message_count": len(messages),
            "priority_count": len(priority_messages),
            "categories": categories,
            "channels": list(channels),
            "users": list(users),
            "time_range": time_range,
            "hardware_phases_mentioned": self._extract_hardware_phases(messages),
            "components_mentioned": self._extract_components(messages),
            "metrics_found": self._extract_metrics(messages)
        }
        
        if context:
            analysis_context.update(context)
            
        return analysis_context
    
    def _extract_hardware_phases(self, messages: List[HardwareMessage]) -> List[str]:
        """Extract hardware development phases mentioned in messages"""
        phases = []
        phase_pattern = r'\b(EVT|DVT|PVT|MP)\b'
        
        for message in messages:
            found_phases = re.findall(phase_pattern, message.content, re.IGNORECASE)
            phases.extend([phase.upper() for phase in found_phases])
        
        return list(set(phases))
    
    def _extract_components(self, messages: List[HardwareMessage]) -> List[str]:
        """Extract hardware components mentioned in messages"""
        components = []
        
        for message in messages:
            content_lower = message.content.lower()
            for component in self.supply_chain_components:
                if component in content_lower:
                    components.append(component)
        
        return list(set(components))
    
    def _extract_metrics(self, messages: List[HardwareMessage]) -> Dict[str, List[float]]:
        """Extract quantitative metrics from messages"""
        metrics = {
            "yield_rates": [],
            "dppm_values": [],
            "percentages": [],
            "delays_days": []
        }
        
        for message in messages:
            content = message.content
            
            # Extract yield rates
            yield_matches = re.findall(r'yield.*?(\d+(?:\.\d+)?)%', content, re.IGNORECASE)
            metrics["yield_rates"].extend([float(m) for m in yield_matches])
            
            # Extract DPPM values
            dppm_matches = re.findall(r'(\d+(?:,\d+)?)\s*DPPM', content, re.IGNORECASE)
            metrics["dppm_values"].extend([float(m.replace(',', '')) for m in dppm_matches])
            
            # Extract general percentages
            percentage_matches = re.findall(r'(\d+(?:\.\d+)?)%', content)
            metrics["percentages"].extend([float(m) for m in percentage_matches])
            
            # Extract delay mentions
            delay_matches = re.findall(r'(\d+)\s*(?:day|week|month).*?delay', content, re.IGNORECASE)
            metrics["delays_days"].extend([float(m) for m in delay_matches])
        
        return metrics
    
    def _parse_risk_analysis(
        self, 
        ai_response: str, 
        original_messages: List[HardwareMessage]
    ) -> List[RiskAssessment]:
        """Parse AI response into structured risk assessments"""
        risk_assessments = []
        
        try:
            analysis_data = json.loads(ai_response)
            
            for risk_data in analysis_data.get("risk_assessments", []):
                assessment = RiskAssessment(
                    category=risk_data.get("category", "unknown"),
                    level=self._parse_risk_level(risk_data.get("risk_level", "medium")),
                    score=min(100, max(1, risk_data.get("score", 50))),
                    confidence=min(1.0, max(0.0, risk_data.get("confidence", 0.5))),
                    description=risk_data.get("description", "Risk identified"),
                    impact=risk_data.get("impact", "Impact assessment pending"),
                    recommendations=risk_data.get("recommendations", []),
                    urgency=risk_data.get("urgency", "medium"),
                    requires_attention=risk_data.get("requires_attention", False)
                )
                risk_assessments.append(assessment)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response JSON: {e}")
            # Fallback: create basic risk assessment
            fallback_assessment = RiskAssessment(
                category="general",
                level=RiskLevel.MEDIUM,
                score=50,
                confidence=0.3,
                description="AI analysis parsing failed - manual review required",
                impact="Unknown impact due to parsing error",
                recommendations=["Review original AI response", "Perform manual analysis"],
                urgency="medium",
                requires_attention=True
            )
            risk_assessments.append(fallback_assessment)
        
        return risk_assessments
    
    async def _generate_executive_summary(
        self, 
        risk_assessments: List[RiskAssessment], 
        overall_risk_score: int,
        messages: List[HardwareMessage]
    ) -> Optional[str]:
        """Generate executive summary using AI"""
        
        if not self.client:
            return None
        
        summary_prompt = f"""
Generate a concise executive summary for GTM leadership based on this risk analysis:

OVERALL RISK SCORE: {overall_risk_score}/100
RISK ASSESSMENTS COUNT: {len(risk_assessments)}
MESSAGES ANALYZED: {len(messages)}

IDENTIFIED RISKS:
{chr(10).join([f"- {assessment.level.value.upper()}: {assessment.description}" for assessment in risk_assessments[:5]])}

Create a 2-3 sentence executive summary focusing on:
1. Current GTM risk level and trajectory
2. Most critical issues requiring immediate attention
3. Key recommendation for leadership action

Keep it concise and action-oriented for executives.
"""
        
        messages = [
            {"role": "system", "content": "You are an executive communication specialist for hardware GTM teams."},
            {"role": "user", "content": summary_prompt}
        ]
        
        return await self._call_openai(messages)
    
    def _compile_recommendations(
        self, 
        risk_assessments: List[RiskAssessment], 
        overall_risk_score: int
    ) -> List[str]:
        """Compile prioritized recommendations from all risk assessments"""
        
        all_recommendations = []
        
        # Priority recommendations based on risk level
        critical_recommendations = []
        high_recommendations = []
        medium_recommendations = []
        
        for assessment in risk_assessments:
            if assessment.level == RiskLevel.CRITICAL:
                critical_recommendations.extend(assessment.recommendations)
            elif assessment.level == RiskLevel.HIGH:
                high_recommendations.extend(assessment.recommendations)
            else:
                medium_recommendations.extend(assessment.recommendations)
        
        # Add overall risk score based recommendations
        if overall_risk_score >= 70:
            all_recommendations.append("🚨 IMMEDIATE: Escalate to executive leadership for crisis management")
            all_recommendations.append("🚨 IMMEDIATE: Convene emergency GTM risk review meeting")
        elif overall_risk_score >= 50:
            all_recommendations.append("⚠️ HIGH: Accelerate mitigation efforts for identified risks")
            all_recommendations.append("⚠️ HIGH: Increase monitoring frequency for key metrics")
        
        # Combine recommendations in priority order
        all_recommendations.extend(critical_recommendations)
        all_recommendations.extend(high_recommendations)
        all_recommendations.extend(medium_recommendations[:3])  # Limit medium priority
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    def _calculate_confidence(self, risk_assessments: List[RiskAssessment]) -> float:
        """Calculate overall confidence in the analysis"""
        if not risk_assessments:
            return 0.0
        
        confidences = [assessment.confidence for assessment in risk_assessments]
        return sum(confidences) / len(confidences)
    
    def _get_capabilities(self) -> List[str]:
        """Return GTM Risk Commander capabilities"""
        return [
            "Hardware GTM Risk Analysis",
            "Supply Chain Risk Assessment", 
            "Quality Metrics Analysis (Yield, DPPM)",
            "Timeline Impact Prediction",
            "EVT/DVT/PVT Phase Analysis",
            "Component Shortage Detection",
            "Regulatory Risk Assessment",
            "Executive Summary Generation",
            "Actionable Recommendation Engine",
            "Real-time OpenAI Integration"
        ]
        
    async def get_specialized_analysis(
        self, 
        focus_area: str, 
        messages: List[HardwareMessage]
    ) -> Dict[str, Any]:
        """Perform specialized analysis for specific focus areas"""
        
        if focus_area == "supply_chain":
            return await self._analyze_supply_chain_risks(messages)
        elif focus_area == "quality":
            return await self._analyze_quality_risks(messages)
        elif focus_area == "timeline":
            return await self._analyze_timeline_risks(messages)
        else:
            return {"error": f"Unknown focus area: {focus_area}"}
    
    async def _analyze_supply_chain_risks(self, messages: List[HardwareMessage]) -> Dict[str, Any]:
        """Specialized supply chain risk analysis"""
        # Implementation for detailed supply chain analysis
        pass
    
    async def _analyze_quality_risks(self, messages: List[HardwareMessage]) -> Dict[str, Any]:
        """Specialized quality risk analysis"""
        # Implementation for detailed quality analysis
        pass
    
    async def _analyze_timeline_risks(self, messages: List[HardwareMessage]) -> Dict[str, Any]:
        """Specialized timeline risk analysis"""
        # Implementation for detailed timeline analysis
        pass 