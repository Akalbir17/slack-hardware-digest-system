"""
Interactive GTM Strategy Agent - Proactive Team Engagement for Strategy Updates
Actively engages with hardware teams to understand current priorities and strategy
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, HardwareMessage, AgentResponse

class InteractiveGTMAgent(BaseAgent):
    """
    Interactive GTM Strategy Agent that:
    1. Proactively asks strategic questions to the team
    2. Analyzes team responses to understand current priorities
    3. Dynamically updates GTM weights based on team input
    4. Generates strategic polls and feedback requests
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(
            agent_id="interactive_gtm_strategist",
            role="Interactive GTM Strategist", 
            openai_api_key=openai_api_key
        )
        
        # Interactive GTM-specific configuration
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 1200
        
        # Strategy engagement state
        self.last_strategic_poll = None
        self.pending_questions = []
        self.team_responses = []
        self.strategy_confidence = 0.5
        
        # Current project context
        self.current_project = {
            "name": "Galaxy X1 Pro",
            "launch_date": "2025-09-15",
            "current_phase": "PVT",
            "team_channels": ["#hardware-gtm", "#engineering", "#supply-chain", "#quality"],
            "key_stakeholders": ["Engineering Lead", "Supply Chain Manager", "Quality Director", "GTM Director"]
        }
    
    def get_system_prompt(self) -> str:
        """Get Interactive GTM Agent system prompt"""
        return f"""You are the Interactive GTM Strategist, an AI agent that proactively engages with hardware teams to understand and optimize Go-To-Market strategy.

CORE MISSION: Don't just analyze - ACTIVELY ENGAGE the team to understand their current priorities and concerns.

CURRENT PROJECT CONTEXT:
- Project: {self.current_project['name']}
- Launch Date: {self.current_project['launch_date']} 
- Current Phase: {self.current_project['current_phase']}
- Team Channels: {', '.join(self.current_project['team_channels'])}

PROACTIVE ENGAGEMENT STRATEGIES:

1. STRATEGIC POLLING: Generate targeted questions to understand team priorities
   Examples:
   - "What's your biggest concern for launch readiness this week?"
   - "Rate your confidence in supplier readiness (1-10)"
   - "Should we prioritize quality testing or production ramp-up?"
   - "Which component shortage worries you most?"

2. PRIORITY DISCOVERY: Ask about focus areas and resource allocation
   Examples:
   - "If you had extra resources this week, where would you apply them?"
   - "What keeps you up at night about this launch?"
   - "Which dependencies are blocking your team right now?"

3. SENTIMENT ANALYSIS: Understand team confidence and morale
   Examples:
   - "How confident are you about hitting the launch date?"
   - "What's the team's energy level on this project?"
   - "Are there any silent concerns not being discussed?"

4. STRATEGY VALIDATION: Test assumptions and get feedback
   Examples:
   - "Does the current GTM timeline still make sense?"
   - "Are we focusing on the right quality metrics?"
   - "Should we adjust our risk priorities?"

RESPONSE ANALYSIS FRAMEWORK:
- Parse team responses for strategic insights
- Identify shifting priorities and concerns
- Extract confidence levels and risk perceptions
- Understand resource constraints and blockers
- Detect hidden issues or unspoken concerns

OUTPUT REQUIREMENTS:
Generate strategic questions, analyze responses, and provide dynamic weight recommendations based on team input.

Always respond with actionable strategic intelligence and engagement plans."""

    async def generate_strategic_questions(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic questions to ask the team based on current context"""
        days_to_launch = current_context.get('days_to_launch', 90)
        current_phase = current_context.get('current_phase', 'PVT')
        
        prompt = f"""Generate 3 strategic questions to ask the hardware team right now.

CURRENT CONTEXT:
- Days to Launch: {days_to_launch}
- Current Phase: {current_phase}

Generate questions like:
- "What's your biggest concern for launch readiness this week?"
- "Rate your confidence in supplier readiness (1-10)"
- "Should we prioritize quality testing or production ramp-up?"

Respond with JSON:
{{
    "strategic_questions": [
        {{
            "question": "Question text",
            "type": "priority|confidence|resource_allocation",
            "target_audience": "Engineering|Supply Chain|Quality|All"
        }}
    ]
}}"""

        response_text = await self._call_openai([
            {"role": "user", "content": prompt}
        ])
        
        if response_text:
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    questions_data = json.loads(response_text[json_start:json_end])
                    return questions_data.get('strategic_questions', [])
            except json.JSONDecodeError:
                pass
        
        # Fallback questions
        return [
            {
                "question": "What's your biggest concern for launch readiness this week?",
                "type": "priority",
                "target_audience": "All"
            }
        ]
    
    async def analyze_team_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze team responses to strategic questions and generate GTM adjustments"""
        try:
            if not responses:
                return {"error": "No responses to analyze"}
            
            # Prepare response context for AI analysis
            response_context = []
            for resp in responses:
                response_context.append({
                    "question": resp.get('question', ''),
                    "answer": resp.get('answer', ''),
                    "respondent_role": resp.get('role', 'Unknown'),
                    "confidence": resp.get('confidence', 5),
                    "timestamp": resp.get('timestamp', datetime.utcnow().isoformat())
                })
            
            # Create AI prompt for response analysis
            prompt = f"""Analyze these team responses to strategic questions and provide GTM strategy adjustments.

TEAM RESPONSES:
{json.dumps(response_context, indent=2)}

ANALYSIS REQUIRED:
1. Identify key themes and concerns
2. Assess team confidence levels
3. Determine priority shifts
4. Recommend GTM weight adjustments
5. Suggest action items

Respond with JSON:
{{
    "key_insights": ["insight1", "insight2", "insight3"],
    "team_confidence": {{
        "overall": <1-10>,
        "product_readiness": <1-10>,
        "supply_chain": <1-10>, 
        "quality": <1-10>,
        "timeline": <1-10>
    }},
    "priority_shifts": {{
        "increase_focus": ["area1", "area2"],
        "decrease_focus": ["area3"],
        "new_concerns": ["concern1", "concern2"]
    }},
    "recommended_weights": {{
        "product_readiness": <percentage>,
        "supply_chain_readiness": <percentage>,
        "quality_readiness": <percentage>,
        "market_readiness": <percentage>,
        "operational_readiness": <percentage>
    }},
    "immediate_actions": ["action1", "action2", "action3"],
    "confidence_level": <0.0-1.0>,
    "rationale": "Why these adjustments are recommended"
}}"""

            response_text = await self._call_openai([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ])
            
            if response_text:
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        analysis = json.loads(response_text[json_start:json_end])
                        return analysis
                except json.JSONDecodeError:
                    pass
            
            return {"error": "Failed to parse AI response"}
            
        except Exception as e:
            return {"error": f"Response analysis failed: {str(e)}"}
    
    async def analyze_messages(self, messages: List[HardwareMessage]) -> AgentResponse:
        """Analyze messages and generate strategic engagement plan"""
        try:
            if not messages:
                return AgentResponse(
                    agent_id=self.agent_id,
                    success=False,
                    error_message="No messages provided for interactive analysis"
                )
            
            # Analyze recent messages for strategic context
            strategic_context = self._extract_strategic_context(messages)
            
            # Generate strategic questions for the team
            questions = await self.generate_strategic_questions(strategic_context)
            
            # Create engagement plan
            engagement_plan = {
                "strategic_questions": questions,
                "engagement_strategy": self._create_engagement_strategy(strategic_context),
                "recommended_polling_frequency": self._recommend_polling_frequency(strategic_context),
                "target_stakeholders": self._identify_key_stakeholders(strategic_context)
            }
            
            summary = f"Interactive GTM Strategy: Generated {len(questions)} strategic questions for team engagement. Focus areas: {', '.join([q['type'] for q in questions[:3]])}"
            
            recommendations = [
                "Deploy strategic questions to team channels",
                "Schedule follow-up polls based on responses", 
                "Adjust GTM weights based on team feedback",
                "Monitor sentiment and confidence trends"
            ]
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                summary=summary,
                recommendations=recommendations,
                confidence=0.9,
                metadata={
                    "analysis_data": engagement_plan,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "messages_analyzed": len(messages),
                    "questions_generated": len(questions),
                    "agent_model": self.model
                }
            )
            
        except Exception as e:
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                error_message=f"Interactive GTM analysis failed: {str(e)}"
            )
    
    def _extract_strategic_context(self, messages: List[HardwareMessage]) -> Dict[str, Any]:
        """Extract strategic context from recent messages"""
        context = {
            "recent_issues": [],
            "sentiment_trend": 0.5,
            "urgency_indicators": [],
            "phase_signals": []
        }
        
        for msg in messages[-20:]:  # Last 20 messages
            content = msg.content.lower()
            
            # Extract issues and concerns
            if any(word in content for word in ['issue', 'problem', 'concern', 'delay', 'shortage']):
                context["recent_issues"].append(msg.content[:100])
            
            # Track urgency signals
            if msg.urgency in ['critical', 'high'] or any(word in content for word in ['urgent', 'asap', 'critical']):
                context["urgency_indicators"].append(msg.urgency)
            
            # Phase transition signals
            if any(word in content for word in ['evt', 'dvt', 'pvt', 'mp', 'launch']):
                context["phase_signals"].append(content)
        
        context["current_phase"] = self.current_project["current_phase"]
        context["days_to_launch"] = (datetime.strptime(self.current_project["launch_date"], "%Y-%m-%d") - datetime.utcnow()).days
        
        return context
    
    def _create_engagement_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create engagement strategy based on current context"""
        days_to_launch = context.get('days_to_launch', 90)
        urgency_level = len(context.get('urgency_indicators', []))
        
        if days_to_launch <= 30:
            return {
                "frequency": "daily",
                "focus": "execution_readiness",
                "channels": ["#hardware-gtm", "#engineering"],
                "question_types": ["confidence_check", "risk_identification"]
            }
        elif urgency_level > 3:
            return {
                "frequency": "bi-daily", 
                "focus": "issue_resolution",
                "channels": self.current_project["team_channels"],
                "question_types": ["priority_assessment", "resource_allocation"]
            }
        else:
            return {
                "frequency": "weekly",
                "focus": "strategic_planning",
                "channels": ["#hardware-gtm"],
                "question_types": ["strategy_validation", "priority_assessment"]
            }
    
    def _recommend_polling_frequency(self, context: Dict[str, Any]) -> str:
        """Recommend polling frequency based on context"""
        days_to_launch = context.get('days_to_launch', 90)
        recent_issues = len(context.get('recent_issues', []))
        
        if days_to_launch <= 14:
            return "every_8_hours"
        elif days_to_launch <= 30 or recent_issues > 5:
            return "daily"
        elif days_to_launch <= 60:
            return "every_2_days"
        else:
            return "weekly"
    
    def _identify_key_stakeholders(self, context: Dict[str, Any]) -> List[str]:
        """Identify key stakeholders to engage based on context"""
        return self.current_project["key_stakeholders"]
    
    def _get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "Strategic question generation",
            "Team response analysis", 
            "Dynamic GTM weight adjustment",
            "Proactive team engagement",
            "Sentiment and confidence tracking",
            "Priority shift detection",
            "Interactive strategy validation",
            "Real-time polling and feedback"
        ] 