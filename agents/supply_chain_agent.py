"""
Supply Chain Intelligence Agent for Hardware GTM Risk Analysis
Monitors component availability and supplier performance in real-time
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from .base_agent import BaseAgent, HardwareMessage, RiskAssessment, AgentResponse

logger = logging.getLogger(__name__)

class SupplyChainIntelligenceAgent(BaseAgent):
    """
    AI Agent specialized in supply chain risk analysis for hardware manufacturing.
    
    Capabilities:
    - Tracks critical component inventory levels
    - Predicts supply disruptions 2-4 weeks ahead  
    - Identifies alternate suppliers automatically
    - Monitors geopolitical risks affecting supply
    - Component shortage detection and mitigation
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(
            agent_id="supply_chain_intel_001",
            role="supply_chain_intelligence",
            openai_api_key=openai_api_key
        )
        
        # Supply chain specific capabilities
        self.capabilities = [
            "Component Inventory Tracking",
            "Supply Disruption Prediction (2-4 weeks ahead)",
            "Alternate Supplier Identification", 
            "Geopolitical Risk Monitoring",
            "Lead Time Analysis",
            "Supplier Performance Scoring",
            "Critical Component Shortage Detection",
            "Supply Chain Cost Optimization",
            "Vendor Risk Assessment",
            "Emergency Procurement Planning"
        ]
        
        # Hardware components knowledge base
        self.critical_components = {
            "displays": ["AMOLED", "OLED", "Mini-LED", "E-ink", "LCD"],
            "chipsets": ["Snapdragon", "MediaTek", "Apple A17", "Exynos", "Kirin"],
            "memory": ["LPDDR5", "UFS 4.0", "eMMC", "NAND Flash"],
            "sensors": ["IMU", "Accelerometer", "Gyroscope", "Magnetometer"],
            "connectivity": ["5G Modem", "WiFi 6E", "Bluetooth 5.3", "NFC"],
            "power": ["Li-ion Battery", "Wireless Charging Coil", "USB-C Port"],
            "camera": ["Image Sensor", "Lens Assembly", "OIS Module", "Flash LED"],
            "audio": ["Speaker", "Microphone", "Audio Codec", "Haptic Motor"]
        }
        
        self.model = "gpt-3.5-turbo"  # Cost-optimized model
    
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for Supply Chain Intelligence Agent"""
        return """You are the Supply Chain Intelligence Agent, an elite AI specialist in hardware manufacturing supply chain analysis.

EXPERTISE:
- Component inventory tracking and forecasting
- Supplier relationship management  
- Supply disruption prediction (2-4 weeks ahead)
- Alternate supplier identification
- Geopolitical risk assessment
- Lead time optimization
- Emergency procurement strategies

Always respond in valid JSON format with structured supply chain analysis."""
        
    async def analyze_messages(
        self, 
        messages: List[HardwareMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Analyze hardware messages for supply chain risks"""
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
                summary="Supply chain analysis completed",
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
                error_message=f"Supply chain analysis failed: {str(e)}"
            )
    
    async def analyze_message(self, message: HardwareMessage) -> AgentResponse:
        """Analyze hardware message for supply chain risks and opportunities"""
        try:
            # Build supply chain context for AI analysis
            context = self._build_supply_chain_context(message)
            
            system_prompt = f"""You are an elite Supply Chain Intelligence Agent for smartphone manufacturing.

EXPERTISE:
- Component inventory tracking and forecasting
- Supplier relationship management  
- Supply disruption prediction (2-4 weeks ahead)
- Alternate supplier identification
- Geopolitical risk assessment
- Lead time optimization
- Emergency procurement strategies

CRITICAL COMPONENTS KNOWLEDGE:
{json.dumps(self.critical_components, indent=2)}

ANALYSIS FRAMEWORK:
1. Component Impact Assessment
2. Supplier Risk Evaluation  
3. Inventory Level Analysis
4. Lead Time Implications
5. Alternative Sourcing Options
6. Geopolitical/Market Factors
7. Cost Impact Analysis
8. Mitigation Recommendations

Your responses must be JSON with:
- supply_chain_risk_score (0-100, higher = more risk)
- component_analysis (affected components and impact)
- supplier_assessment (current supplier status)
- inventory_forecast (projected levels and shortages)  
- alternate_suppliers (backup options with details)
- risk_factors (specific supply chain risks)
- recommendations (actionable steps)
- timeline_impact (effect on production schedule)
- cost_implications (financial impact analysis)
"""

            user_prompt = f"""SUPPLY CHAIN ANALYSIS REQUEST:

MESSAGE CONTEXT:
Channel: {message.channel}
User: {message.user}  
Content: {message.content}
Timestamp: {message.timestamp}
Priority: {message.urgency}

SUPPLY CHAIN CONTEXT:
{context}

Analyze this message for supply chain implications. Focus on:
1. Component availability risks
2. Supplier performance issues
3. Lead time impacts
4. Alternative sourcing needs
5. Cost optimization opportunities
6. Procurement recommendations

Provide detailed supply chain intelligence analysis."""

            # Call OpenAI for analysis
            response = await self._call_openai([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            if response:
                # Extract structured data from AI response
                analysis = self._parse_ai_response(response)
                
                # Ensure recommendations is always a List[str]
                recommendations = analysis.get("recommendations", ["Review supply chain status"])
                if isinstance(recommendations, str):
                    recommendations = [recommendations]
                elif isinstance(recommendations, dict):
                    # Extract recommendations from dict
                    if "actionable_steps" in recommendations:
                        recommendations = recommendations["actionable_steps"] if isinstance(recommendations["actionable_steps"], list) else [str(recommendations["actionable_steps"])]
                    else:
                        recommendations = [str(v) for v in recommendations.values()]
                elif not isinstance(recommendations, list):
                    recommendations = [str(recommendations)]
                
                # Ensure all items in the list are strings
                recommendations = [str(item) for item in recommendations if item]
                
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=True,
                    overall_risk_score=analysis.get("supply_chain_risk_score", 50),
                    summary=f"Supply chain analysis completed. Risk score: {analysis.get('supply_chain_risk_score', 50)}",
                    recommendations=recommendations,
                    confidence=analysis.get("supply_chain_risk_score", 50) / 100.0,
                    metadata={
                        "analysis_data": analysis,
                        "agent_role": self.role,
                        "model_used": self.model,
                        "raw_response": response[:500] if response else "No response"  # Truncate for storage
                    }
                )
            else:
                # Fallback analysis when OpenAI unavailable
                return self._create_fallback_response(message)
                
        except Exception as e:
            logger.error(f"Supply Chain Agent analysis failed: {str(e)}")
            return self._create_error_response(str(e))
    
    def _build_supply_chain_context(self, message: HardwareMessage) -> str:
        """Build supply chain specific context for analysis"""
        
        # Identify mentioned components
        mentioned_components = []
        content_lower = message.content.lower()
        
        for category, components in self.critical_components.items():
            for component in components:
                if component.lower() in content_lower:
                    mentioned_components.append({
                        "component": component,
                        "category": category,
                        "criticality": "high" if category in ["chipsets", "displays"] else "medium"
                    })
        
        # Supply chain risk indicators
        risk_keywords = {
            "shortage": "high",
            "delay": "high", 
            "supplier": "medium",
            "inventory": "medium",
            "lead time": "medium",
            "cost increase": "high",
            "quality issue": "high",
            "yield": "medium",
            "production": "medium"
        }
        
        detected_risks = []
        for keyword, severity in risk_keywords.items():
            if keyword in content_lower:
                detected_risks.append({"keyword": keyword, "severity": severity})
        
        context = {
            "mentioned_components": mentioned_components,
            "detected_risk_signals": detected_risks,
            "message_category": self._categorize_supply_message(message),
            "supply_chain_relevance": len(mentioned_components) > 0 or len(detected_risks) > 0
        }
        
        return json.dumps(context, indent=2)
    
    def _categorize_supply_message(self, message: HardwareMessage) -> str:
        """Categorize the message for supply chain analysis"""
        content_lower = message.content.lower()
        
        if any(word in content_lower for word in ["shortage", "inventory", "stock"]):
            return "inventory_management"
        elif any(word in content_lower for word in ["supplier", "vendor", "procurement"]):
            return "supplier_relations"
        elif any(word in content_lower for word in ["delay", "lead time", "timeline"]):
            return "timeline_impact"
        elif any(word in content_lower for word in ["cost", "price", "budget"]):
            return "cost_management"
        elif any(word in content_lower for word in ["quality", "defect", "yield"]):
            return "quality_supply_impact"
        else:
            return "general_supply_chain"
    
    def _create_fallback_response(self, message: HardwareMessage) -> AgentResponse:
        """Create fallback response when OpenAI is unavailable"""
        
        # Simple heuristic-based analysis
        content_lower = message.content.lower()
        
        risk_score = 30  # Default moderate risk
        
        # Increase risk based on keywords
        if "shortage" in content_lower:
            risk_score += 40
        if "delay" in content_lower:
            risk_score += 30
        if "supplier" in content_lower and "issue" in content_lower:
            risk_score += 25
        if any(word in content_lower for word in ["critical", "urgent", "emergency"]):
            risk_score += 20
            
        risk_score = min(risk_score, 100)
        
        analysis = {
            "supply_chain_risk_score": risk_score,
            "component_analysis": "Supply chain impact detected - full analysis requires OpenAI",
            "supplier_assessment": "Heuristic-based assessment - detailed analysis pending",
            "inventory_forecast": "Basic forecast - enhanced analysis with AI",
            "alternate_suppliers": ["Evaluation pending AI analysis"],
            "risk_factors": ["Component availability", "Supplier performance", "Lead time variance"],
            "recommendations": [
                "Monitor supplier communications closely",
                "Review inventory levels for affected components", 
                "Prepare backup supplier contacts",
                "Enable OpenAI for detailed analysis"
            ],
            "timeline_impact": "Potential 1-2 week impact estimated",
            "cost_implications": "Cost analysis requires detailed AI evaluation"
        }
        
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            overall_risk_score=risk_score,
            summary=f"Supply chain heuristic analysis completed. Risk score: {risk_score}",
            recommendations=analysis["recommendations"],
            confidence=0.6,  # Lower confidence without AI
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
            overall_risk_score=50,  # Default neutral score for errors
            summary="Supply chain analysis failed",
            recommendations=["Retry analysis", "Check system status"],
            confidence=0.0,
            error_message=f"Supply Chain analysis failed: {error_msg}",
            metadata={
                "analysis_type": "error_handler",
                "agent_role": self.role
            }
        )
    
    def _get_capabilities(self) -> List[str]:
        """Return Supply Chain Intelligence Agent capabilities"""
        return self.capabilities 