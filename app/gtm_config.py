"""
GTM (Go-To-Market) Configuration and Scoring System
Enhanced GTM intelligence with AI-powered dynamic weighting and real-time updates
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class GTMProject:
    """GTM Project configuration"""
    project_name: str
    launch_date: datetime
    current_phase: str  # EVT, DVT, PVT, MP, GTM
    market_segment: str
    
class GTMScoringSystem:
    """Enhanced GTM scoring with AI-powered dynamic weighting and real-time updates"""
    
    def __init__(self, gtm_strategy_agent=None):
        # Demo project configuration
        self.current_project = GTMProject(
            project_name="Galaxy X1 Pro",
            launch_date=datetime(2025, 9, 15),  # September 15, 2025
            current_phase="PVT",
            market_segment="Premium Smartphone"
        )
        
        # AI GTM Strategy Agent for dynamic weighting
        self.gtm_strategy_agent = gtm_strategy_agent
        self.last_ai_update = None
        self.ai_cache_duration = 300  # 5 minutes cache
        
        # GTM Readiness Factors (0-100 each)
        self.gtm_factors = {
            "product_readiness": {
                "weight": 25,
                "components": {
                    "design_freeze": 95,      # Design completed
                    "evt_completion": 100,    # Engineering validation done
                    "dvt_completion": 90,     # Design validation almost done
                    "pvt_completion": 70,     # Production validation in progress
                    "regulatory_approval": 60 # FCC, CE certifications
                }
            },
            "supply_chain_readiness": {
                "weight": 25,
                "components": {
                    "component_availability": 75,
                    "supplier_qualification": 85,
                    "inventory_buffer": 60,
                    "backup_suppliers": 70,
                    "cost_targets": 80
                }
            },
            "quality_readiness": {
                "weight": 20,
                "components": {
                    "yield_rates": 85,        # Manufacturing yield
                    "defect_rates": 90,       # Quality metrics
                    "test_completion": 75,    # Testing progress
                    "compliance_status": 80,  # Standards compliance
                    "reliability_testing": 70
                }
            },
            "market_readiness": {
                "weight": 15,
                "components": {
                    "marketing_campaign": 50,
                    "channel_partnerships": 70,
                    "pricing_strategy": 85,
                    "competitive_analysis": 90,
                    "sales_training": 40
                }
            },
            "operational_readiness": {
                "weight": 15,
                "components": {
                    "production_capacity": 80,
                    "logistics_setup": 65,
                    "support_infrastructure": 55,
                    "documentation": 70,
                    "team_readiness": 75
                }
            }
        }
    
    async def calculate_gtm_score(self, messages: List = None, ai_risk_adjustments: Dict[str, float] = None) -> Dict[str, Any]:
        """Calculate comprehensive GTM readiness score with AI-powered dynamic weighting"""
        
        # Calculate days to launch
        days_to_launch = (self.current_project.launch_date - datetime.utcnow()).days
        
        # Get AI-powered dynamic weights and component updates
        dynamic_weights = None
        component_updates = None
        ai_analysis_data = {}
        
        if self.gtm_strategy_agent and messages:
            try:
                # Check if we need fresh AI analysis (cache for 5 minutes)
                need_fresh_analysis = (
                    self.last_ai_update is None or 
                    (datetime.utcnow() - self.last_ai_update).seconds > self.ai_cache_duration
                )
                
                if need_fresh_analysis:
                    # Get AI-powered GTM strategy optimization
                    from agents.base_agent import HardwareMessage
                    
                    # Convert messages to HardwareMessage format
                    hw_messages = []
                    for msg in messages[-50:]:  # Analyze last 50 messages
                        hw_msg = HardwareMessage(
                            id=msg.get('slack_message_id', ''),
                            content=msg.get('content', ''),
                            channel='hardware_team',  # Default channel
                            user='team_member',       # Default user
                            timestamp=msg.get('timestamp', datetime.utcnow()),
                            urgency=msg.get('urgency', 'medium'),
                            category='gtm_analysis',
                            sentiment_score=msg.get('sentiment_score', 0.5)
                        )
                        hw_messages.append(hw_msg)
                    
                    # Get AI analysis
                    ai_response = await self.gtm_strategy_agent.analyze_messages(hw_messages)
                    
                    if ai_response.success and ai_response.metadata.get("analysis_data"):
                        analysis_data = ai_response.metadata.get("analysis_data", {})
                        dynamic_weights = analysis_data.get('dynamic_weights', {})
                        component_updates = analysis_data.get('component_updates', {})
                        ai_analysis_data = analysis_data
                        self.last_ai_update = datetime.utcnow()
                        
                        # Update component scores with AI insights
                        if component_updates:
                            for factor_name, factor_data in self.gtm_factors.items():
                                for comp_name, comp_score in factor_data["components"].items():
                                    if comp_name in component_updates:
                                        # AI provides real-time updates
                                        self.gtm_factors[factor_name]["components"][comp_name] = component_updates[comp_name]
                        
                        # Update weights with AI insights
                        if dynamic_weights:
                            for factor_name in self.gtm_factors.keys():
                                if factor_name in dynamic_weights:
                                    self.gtm_factors[factor_name]["weight"] = dynamic_weights[factor_name]
                        
            except Exception as e:
                print(f"AI GTM analysis failed, using fallback: {e}")
                # Fallback to phase-based weights if AI fails
                if self.gtm_strategy_agent:
                    fallback_weights = self.gtm_strategy_agent.get_current_phase_weights()
                    for factor_name, weight in fallback_weights.items():
                        if factor_name in self.gtm_factors:
                            self.gtm_factors[factor_name]["weight"] = weight
        
        # Calculate factor scores
        factor_scores = {}
        overall_score = 0
        
        for factor_name, factor_data in self.gtm_factors.items():
            # Average component scores
            components = factor_data["components"]
            factor_score = sum(components.values()) / len(components)
            
            # Apply AI risk adjustments if available
            if ai_risk_adjustments and factor_name in ai_risk_adjustments:
                factor_score *= ai_risk_adjustments[factor_name]
                factor_score = max(0, min(100, factor_score))
            
            factor_scores[factor_name] = factor_score
            
            # Weight the factor for overall score (now using AI-dynamic weights)
            weighted_score = factor_score * (factor_data["weight"] / 100)
            overall_score += weighted_score
        
        # Launch readiness status
        if overall_score >= 85:
            status = "🟢 READY TO LAUNCH"
            confidence = "High"
        elif overall_score >= 70:
            status = "🟡 LAUNCH PREPARATION"
            confidence = "Medium"
        elif overall_score >= 55:
            status = "🟠 CRITICAL ISSUES"
            confidence = "Low"
        else:
            status = "🔴 NOT READY"
            confidence = "Very Low"
        
        # Timeline assessment
        if days_to_launch <= 0:
            timeline_status = "🔴 OVERDUE"
        elif days_to_launch <= 30:
            timeline_status = "🟠 FINAL SPRINT"
        elif days_to_launch <= 90:
            timeline_status = "🟡 PREPARATION PHASE"
        else:
            timeline_status = "🟢 ON TRACK"
        
        result = {
            "project_info": {
                "name": self.current_project.project_name,
                "launch_date": self.current_project.launch_date.strftime("%B %d, %Y"),
                "days_to_launch": max(0, days_to_launch),
                "current_phase": self.current_project.current_phase,
                "market_segment": self.current_project.market_segment
            },
            "gtm_score": round(overall_score, 1),
            "launch_readiness": status,
            "confidence_level": confidence,
            "timeline_status": timeline_status,
            "factor_scores": factor_scores,
            "critical_gaps": self._identify_critical_gaps(factor_scores),
            "next_milestones": self._get_next_milestones(days_to_launch),
            "risk_level": self._calculate_risk_level(overall_score, days_to_launch),
            "ai_powered": self.gtm_strategy_agent is not None,
            "ai_analysis": ai_analysis_data if ai_analysis_data else None,
            "dynamic_weights_applied": dynamic_weights is not None,
            "component_updates_applied": component_updates is not None
        }
        
        return result
    
    def _identify_critical_gaps(self, factor_scores: Dict[str, float]) -> List[str]:
        """Identify critical gaps that need attention"""
        gaps = []
        
        for factor_name, score in factor_scores.items():
            if score < 60:
                gap_name = factor_name.replace("_", " ").title()
                gaps.append(f"{gap_name} needs immediate attention ({score:.1f}%)")
        
        # Add specific component gaps
        for factor_name, factor_data in self.gtm_factors.items():
            for comp_name, comp_score in factor_data["components"].items():
                if comp_score < 50:
                    comp_display = comp_name.replace("_", " ").title()
                    gaps.append(f"{comp_display} critically behind ({comp_score}%)")
        
        return gaps[:5]  # Top 5 critical gaps
    
    def _get_next_milestones(self, days_to_launch: int) -> List[str]:
        """Get next critical milestones based on timeline"""
        milestones = []
        
        if days_to_launch > 120:
            milestones = [
                "Complete PVT phase validation",
                "Finalize supplier agreements",
                "Launch marketing campaign preparation"
            ]
        elif days_to_launch > 60:
            milestones = [
                "Production ramp-up initiation",
                "Marketing campaign launch",
                "Sales team training completion",
                "Distribution setup finalization"
            ]
        elif days_to_launch > 30:
            milestones = [
                "Final production validation",
                "Launch event preparation",
                "Inventory buildup completion",
                "Media and PR activities"
            ]
        else:
            milestones = [
                "Launch execution readiness",
                "Final quality checks",
                "Launch event coordination",
                "Post-launch support preparation"
            ]
        
        return milestones
    
    def _calculate_risk_level(self, gtm_score: float, days_to_launch: int) -> str:
        """Calculate overall risk level for launch"""
        
        # Score-based risk
        if gtm_score >= 85:
            score_risk = "low"
        elif gtm_score >= 70:
            score_risk = "medium"
        elif gtm_score >= 55:
            score_risk = "high"
        else:
            score_risk = "critical"
        
        # Timeline-based risk
        if days_to_launch <= 0:
            timeline_risk = "critical"
        elif days_to_launch <= 30:
            timeline_risk = "high"
        elif days_to_launch <= 90:
            timeline_risk = "medium"
        else:
            timeline_risk = "low"
        
        # Combined risk (take higher risk)
        risk_levels = ["low", "medium", "high", "critical"]
        score_idx = risk_levels.index(score_risk)
        timeline_idx = risk_levels.index(timeline_risk)
        
        combined_risk = risk_levels[max(score_idx, timeline_idx)]
        
        return combined_risk.upper()
    
    def apply_ai_insights(self, ai_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Convert AI agent analysis into GTM factor adjustments"""
        adjustments = {}
        
        # Mock AI-based adjustments (would be real with AI analysis)
        if ai_analysis.get("supply_chain_risks"):
            adjustments["supply_chain_readiness"] = 0.9  # 10% penalty
        
        if ai_analysis.get("quality_issues"):
            adjustments["quality_readiness"] = 0.85  # 15% penalty
        
        if ai_analysis.get("timeline_delays"):
            adjustments["product_readiness"] = 0.92  # 8% penalty
        
        return adjustments

# Global GTM scoring system (will be initialized with AI agent in main.py)
gtm_system = None

def initialize_gtm_system(gtm_strategy_agent=None):
    """Initialize the global GTM system with optional AI strategy agent"""
    global gtm_system
    gtm_system = GTMScoringSystem(gtm_strategy_agent)
    return gtm_system 