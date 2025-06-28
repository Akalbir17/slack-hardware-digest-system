"""
GTM Strategy Agent - AI-Powered Dynamic GTM Weighting and Factor Updates
Intelligently adjusts GTM factor weights and scores based on real-time analysis
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, HardwareMessage, AgentResponse

class GTMStrategyAgent(BaseAgent):
    """
    AI-powered GTM Strategy Agent that dynamically adjusts:
    1. Factor weights based on launch phase and timeline
    2. Component scores based on message analysis
    3. Risk assessments based on current situation
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(
            agent_id="gtm_strategy_commander",
            role="GTM Strategy Commander", 
            openai_api_key=openai_api_key
        )
        
        # GTM Strategy-specific configuration
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 1200
        
        # Current project context (would come from project management system)
        self.current_project = {
            "name": "Galaxy X1 Pro",
            "launch_date": "2025-09-15",
            "current_phase": "PVT",  # EVT, DVT, PVT, MP, GTM
            "market_segment": "Premium Smartphone",
            "competitive_pressure": "high",  # low, medium, high, critical
            "regulatory_status": "in_progress"  # pending, in_progress, approved
        }
    
    def get_system_prompt(self) -> str:
        """Get GTM Strategy Agent system prompt"""
        return f"""You are the GTM Strategy Commander, an elite AI agent specializing in dynamic Go-To-Market strategy optimization for hardware launches.

CURRENT PROJECT CONTEXT:
- Project: {self.current_project['name']}
- Launch Date: {self.current_project['launch_date']}
- Current Phase: {self.current_project['current_phase']}
- Market Segment: {self.current_project['market_segment']}
- Competitive Pressure: {self.current_project['competitive_pressure']}

CORE RESPONSIBILITIES:
1. DYNAMIC WEIGHT OPTIMIZATION: Adjust GTM factor weights (Product, Supply Chain, Quality, Market, Operations) based on:
   - Launch phase (EVT=engineering focus, DVT=design focus, PVT=production focus, MP=operations focus)
   - Days to launch (closer = operations/quality matter more)
   - Current crisis situations (supply shortage = supply chain weight ↑)
   - Competitive landscape (high competition = market readiness weight ↑)

2. REAL-TIME FACTOR UPDATES: Analyze hardware team messages to update component scores:
   - EVT completion status from engineering updates
   - Supplier performance from supply chain alerts
   - Yield rates from quality reports
   - Marketing campaign progress from marketing updates

3. STRATEGIC RECOMMENDATIONS: Provide actionable GTM strategy adjustments

ANALYSIS FRAMEWORK:
- Parse hardware messages for GTM-relevant intelligence
- Identify which GTM factors are most critical right now
- Calculate dynamic weights based on current situation
- Update component scores based on real evidence
- Recommend strategic focus areas

WEIGHT ADJUSTMENT LOGIC:
- EVT Phase: Product(40%), Quality(25%), Supply(20%), Market(10%), Operations(5%)
- DVT Phase: Product(35%), Quality(30%), Supply(20%), Market(10%), Operations(5%)
- PVT Phase: Quality(35%), Product(25%), Supply(25%), Operations(10%), Market(5%)
- MP Phase: Operations(40%), Supply(30%), Quality(20%), Market(8%), Product(2%)
- <60 days to launch: Operations(35%), Quality(30%), Supply(25%), Market(10%)
- <30 days to launch: Operations(45%), Quality(35%), Supply(15%), Market(5%)

Respond with structured JSON containing dynamic weights, updated scores, and strategic recommendations."""

    async def analyze_messages(self, messages: List[HardwareMessage]) -> AgentResponse:
        """Analyze messages and provide dynamic GTM strategy optimization"""
        try:
            if not messages:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message="No messages provided for GTM strategy analysis"
                )
            
            # Calculate days to launch
            launch_date = datetime.strptime(self.current_project['launch_date'], '%Y-%m-%d')
            days_to_launch = (launch_date - datetime.utcnow()).days
            
            # Prepare message context for AI analysis
            message_context = []
            gtm_relevant_messages = 0
            
            for msg in messages[-20:]:  # Analyze recent 20 messages
                content = msg.content.lower()
                if any(keyword in content for keyword in [
                    'evt', 'dvt', 'pvt', 'yield', 'supplier', 'component', 
                    'schedule', 'delay', 'quality', 'defect', 'approval',
                    'marketing', 'launch', 'production', 'certification'
                ]):
                    message_context.append({
                        'content': msg.content[:200],  # Limit content length
                        'priority': msg.urgency,
                        'timestamp': msg.timestamp.isoformat() if msg.timestamp else None
                    })
                    gtm_relevant_messages += 1
            
            if gtm_relevant_messages == 0:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message="No GTM-relevant messages found for strategy analysis"
                )
            
            # Create AI prompt for GTM strategy analysis
            prompt = f"""Analyze the following hardware team messages and provide dynamic GTM strategy optimization.

PROJECT CONTEXT:
- Current Phase: {self.current_project['current_phase']}
- Days to Launch: {days_to_launch}
- Market Pressure: {self.current_project['competitive_pressure']}
- Total GTM-Relevant Messages: {gtm_relevant_messages}

RECENT HARDWARE TEAM MESSAGES:
{json.dumps(message_context, indent=2)}

ANALYSIS REQUIRED:
1. Determine optimal GTM factor weights for current situation
2. Update component scores based on message evidence
3. Identify critical focus areas
4. Provide strategic recommendations

Respond with JSON:
{{
    "dynamic_weights": {{
        "product_readiness": <percentage 0-100>,
        "supply_chain_readiness": <percentage 0-100>,
        "quality_readiness": <percentage 0-100>, 
        "market_readiness": <percentage 0-100>,
        "operational_readiness": <percentage 0-100>
    }},
    "component_updates": {{
        "evt_completion": <score 0-100>,
        "dvt_completion": <score 0-100>,
        "pvt_completion": <score 0-100>,
        "yield_rates": <score 0-100>,
        "supplier_qualification": <score 0-100>,
        "component_availability": <score 0-100>,
        "marketing_campaign": <score 0-100>,
        "production_capacity": <score 0-100>
    }},
    "critical_focus_areas": ["area1", "area2", "area3"],
    "strategic_recommendations": ["rec1", "rec2", "rec3"],
    "confidence_level": <0.0-1.0>,
    "rationale": "Why these weights and updates were chosen"
}}"""

            # Call OpenAI API using base agent method
            response_text = await self._call_openai([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ])
            
            # Parse response
            if not response_text:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message="OpenAI API call failed or returned empty response"
                )
            
            analysis_text = response_text.strip()
            
            try:
                # Extract JSON from response
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    analysis_json = json.loads(analysis_text[json_start:json_end])
                else:
                    raise ValueError("No valid JSON found in response")
                
                # Validate required fields
                required_fields = ['dynamic_weights', 'component_updates', 'critical_focus_areas']
                for field in required_fields:
                    if field not in analysis_json:
                        raise ValueError(f"Missing required field: {field}")
                
                # Create summary
                summary = f"GTM Strategy Analysis: Optimized weights for {self.current_project['current_phase']} phase with {days_to_launch} days to launch. Focus areas: {', '.join(analysis_json['critical_focus_areas'][:3])}"
                
                # Create recommendations
                recommendations = analysis_json.get('strategic_recommendations', [])
                
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=True,
                    summary=summary,
                    recommendations=recommendations,
                    confidence=analysis_json.get('confidence_level', 0.8),
                    metadata={
                        "analysis_data": analysis_json,
                        "analysis_timestamp": datetime.utcnow().isoformat(),
                        "days_to_launch": days_to_launch,
                        "current_phase": self.current_project['current_phase'],
                        "messages_analyzed": gtm_relevant_messages,
                        "agent_model": self.model
                    }
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message=f"Failed to parse GTM strategy analysis: {str(e)}",
                    raw_response=analysis_text
                )
                
        except Exception as e:
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                error_message=f"GTM strategy analysis failed: {str(e)}"
            )
    
    def _get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "Dynamic GTM factor weight optimization",
            "Real-time component score updates",
            "Launch phase-aware strategy adjustment",
            "Competitive pressure assessment",
            "Timeline-based priority rebalancing",
            "Strategic focus area identification",
            "GTM risk mitigation recommendations",
            "Hardware launch readiness optimization"
        ]
    
    async def update_project_context(self, project_updates: Dict[str, Any]) -> bool:
        """Update current project context"""
        try:
            for key, value in project_updates.items():
                if key in self.current_project:
                    self.current_project[key] = value
            return True
        except Exception:
            return False
    
    def get_current_phase_weights(self) -> Dict[str, float]:
        """Get default weights for current phase (fallback if AI fails)"""
        phase = self.current_project['current_phase'].upper()
        
        if phase == 'EVT':
            return {
                "product_readiness": 40.0,
                "quality_readiness": 25.0,
                "supply_chain_readiness": 20.0,
                "market_readiness": 10.0,
                "operational_readiness": 5.0
            }
        elif phase == 'DVT':
            return {
                "product_readiness": 35.0,
                "quality_readiness": 30.0,
                "supply_chain_readiness": 20.0,
                "market_readiness": 10.0,
                "operational_readiness": 5.0
            }
        elif phase == 'PVT':
            return {
                "quality_readiness": 35.0,
                "product_readiness": 25.0,
                "supply_chain_readiness": 25.0,
                "operational_readiness": 10.0,
                "market_readiness": 5.0
            }
        elif phase == 'MP':
            return {
                "operational_readiness": 40.0,
                "supply_chain_readiness": 30.0,
                "quality_readiness": 20.0,
                "market_readiness": 8.0,
                "product_readiness": 2.0
            }
        else:
            # Default balanced weights
            return {
                "product_readiness": 25.0,
                "supply_chain_readiness": 25.0,
                "quality_readiness": 20.0,
                "market_readiness": 15.0,
                "operational_readiness": 15.0
            } 