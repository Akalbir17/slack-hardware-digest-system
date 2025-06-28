"""
Main Streamlit Dashboard for Slack Digest System
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Slack Digest System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_URL = os.getenv("API_URL", "http://fastapi:8000")  # Use internal Docker network

def get_api_data(endpoint):
    """Fetch data from API endpoint"""
    try:
        # Increase timeout for digest generation (AI analysis takes time)
        timeout = 30 if '/digest/' in endpoint else 10
        response = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error(f"Request timed out - AI analysis is taking longer than expected. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def post_api_data(endpoint, data=None):
    """Post data to API endpoint"""
    try:
        # Increase timeout for digest generation (AI analysis takes time)
        timeout = 30 if '/digest/' in endpoint or '/analyze' in endpoint else 10
        response = requests.post(f"{API_URL}{endpoint}", json=data or {}, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error(f"Request timed out - AI analysis is taking longer than expected. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.title("📊 Slack Digest System for Hardware GTM Acceleration")
    st.markdown("---")
    
    # Get real system data
    health_data = get_api_data("/health")
    stats_data = get_api_data("/api/stats")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # API Health Check
        st.subheader("System Status")
        if health_data:
            if health_data.get("status") == "healthy":
                st.success("✅ API Healthy")
            else:
                st.error("❌ API Unhealthy")
                
            # Show real system status
            if health_data.get("database") == "healthy":
                st.success("✅ Database Connected")
            else:
                st.warning("⚠️ Database Not Connected")
                
            if health_data.get("redis") == "healthy":
                st.success("✅ Redis Connected")
            else:
                st.warning("⚠️ Redis Not Connected")
                
            # OpenAI status - check from agent status
            agent_status = get_api_data("/api/agents/status")
            if agent_status:
                ai_status = agent_status.get("ai_system_status", {})
                agents = ai_status.get("agents", {})
                openai_working = False
                
                # Check if any agent has OpenAI responsive
                for agent_data in agents.values():
                    health = agent_data.get("health", {})
                    if health.get("openai_responsive", False):
                        openai_working = True
                        break
                
                if openai_working:
                    st.success("✅ OpenAI Connected")
                else:
                    st.warning("⚠️ OpenAI Not Configured")
            else:
                st.warning("⚠️ OpenAI Status Unknown")
        else:
            st.error("❌ Cannot connect to API")
    
    # Enhanced GTM Dashboard Header
    gtm_data = get_api_data("/api/gtm/dashboard")
    
    if gtm_data:
        # GTM Project Header
        project_info = gtm_data.get("project_info", {})
        st.subheader(f"🚀 {project_info.get('name', 'Hardware Project')} - GTM Command Center")
        
        # GTM Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gtm_score = gtm_data.get("gtm_score", 0)
            status_color = "🟢" if gtm_score >= 85 else "🟡" if gtm_score >= 70 else "🟠" if gtm_score >= 55 else "🔴"
            st.metric(
                label="GTM Readiness Score",
                value=f"{gtm_score}% {status_color}",
                delta=gtm_data.get("launch_readiness", "Unknown")
            )
        
        with col2:
            days_to_launch = project_info.get("days_to_launch", 0)
            timeline_status = gtm_data.get("timeline_status", "Unknown")
            st.metric(
                label="Days to Launch",
                value=f"{days_to_launch} days",
                delta=timeline_status
            )
        
        with col3:
            launch_date = project_info.get("launch_date", "TBD")
            current_phase = project_info.get("current_phase", "Unknown")
            st.metric(
                label="Target Launch",
                value=launch_date,
                delta=f"Current: {current_phase} Phase"
            )
        
        with col4:
            risk_level = gtm_data.get("risk_level", "UNKNOWN")
            confidence = gtm_data.get("confidence_level", "Unknown")
            risk_color = "🔴" if risk_level == "CRITICAL" else "🟠" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
            st.metric(
                label="Launch Risk",
                value=f"{risk_level} {risk_color}",
                delta=f"Confidence: {confidence}"
            )
    else:
        # Fallback to basic metrics if GTM data unavailable
        col1, col2, col3 = st.columns(3)
        
        if stats_data:
            messages_data = stats_data.get("messages", {})
            total_messages = f"{messages_data.get('total', 0):,}"
            critical_high = f"{messages_data.get('critical', 0) + messages_data.get('high', 0)}"
            avg_sentiment = f"{messages_data.get('avg_sentiment', 0)*100:.1f}%"
        else:
            total_messages = "Loading..."
            critical_high = "Loading..."
            avg_sentiment = "Loading..."
        
        with col1:
            st.metric(
                label="Total Messages",
                value=total_messages,
                delta="Hardware Team Communications"
            )
        
        with col2:
            st.metric(
                label="Critical/High Priority", 
                value=critical_high,
                delta="Require Attention"
            )
        
        with col3:
            st.metric(
                label="Avg Sentiment",
                value=avg_sentiment,
                delta="Team Communication Health"
            )
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Overview", "📢 Latest Messages", "⚠️ Risk Assessments", "🤖 AI Agents", "📋 Daily Digest", "🗣️ Interactive Strategy"])
    
    with tab1:
        st.header("GTM Launch Readiness Overview")
        
        # Enhanced GTM Readiness Breakdown
        if gtm_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 GTM Readiness Factors")
                
                # GTM Factor Scores
                factor_scores = gtm_data.get("factor_scores", {})
                if factor_scores:
                    # Create a more detailed breakdown
                    factors_df = pd.DataFrame([
                        {"Factor": "Product Readiness", "Score": factor_scores.get("product_readiness", 0), "Weight": "25%"},
                        {"Factor": "Supply Chain", "Score": factor_scores.get("supply_chain_readiness", 0), "Weight": "25%"},
                        {"Factor": "Quality Assurance", "Score": factor_scores.get("quality_readiness", 0), "Weight": "20%"},
                        {"Factor": "Market Readiness", "Score": factor_scores.get("market_readiness", 0), "Weight": "15%"},
                        {"Factor": "Operations", "Score": factor_scores.get("operational_readiness", 0), "Weight": "15%"}
                    ])
                    
                    # Create color-coded bar chart
                    fig_factors = px.bar(
                        factors_df,
                        x='Factor',
                        y='Score',
                        title='GTM Readiness Factor Breakdown',
                        color='Score',
                        color_continuous_scale=['red', 'orange', 'yellow', 'lightgreen', 'green'],
                        range_color=[0, 100]
                    )
                    fig_factors.update_layout(height=400)
                    st.plotly_chart(fig_factors, use_container_width=True)
                    
                    # Show factor details
                    for _, row in factors_df.iterrows():
                        score = row['Score']
                        status_emoji = "🟢" if score >= 85 else "🟡" if score >= 70 else "🟠" if score >= 55 else "🔴"
                        st.write(f"{status_emoji} **{row['Factor']}**: {score:.1f}% (Weight: {row['Weight']})")
            
            with col2:
                st.subheader("🚨 Critical Gaps")
                critical_gaps = gtm_data.get("critical_gaps", [])
                if critical_gaps:
                    for gap in critical_gaps:
                        st.warning(f"⚠️ {gap}")
                else:
                    st.success("✅ No critical gaps identified")
                
                st.subheader("📋 Next Milestones")
                milestones = gtm_data.get("next_milestones", [])
                for milestone in milestones:
                    st.write(f"□ {milestone}")
        
        # System Stats Section
        if stats_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Message Priority Distribution
                messages_info = stats_data.get("messages", {})
                priority_data = pd.DataFrame({
                    'Priority': ['Critical', 'High', 'Others'],
                    'Count': [
                        messages_info.get('critical', 0),
                        messages_info.get('high', 0),
                        messages_info.get('total', 0) - messages_info.get('critical', 0) - messages_info.get('high', 0)
                    ]
                })
                
                fig_priority = px.pie(
                    priority_data,
                    values='Count',
                    names='Priority',
                    title='Message Priority Distribution',
                    color_discrete_map={
                        'Critical': '#ff4444',
                        'High': '#ff8800', 
                        'Others': '#44ff44'
                    }
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            
            with col2:
                # Risk Assessment Summary
                risks_info = stats_data.get("risks", {})
                risk_data = pd.DataFrame({
                    'Risk Level': ['Critical', 'Attention Required', 'Total Risks'],
                    'Count': [
                        risks_info.get('critical', 0),
                        risks_info.get('attention_required', 0),
                        risks_info.get('total', 0)
                    ]
                })
                
                fig_risks = px.bar(
                    risk_data,
                    x='Risk Level',
                    y='Count',
                    title='Risk Assessment Summary',
                    color='Count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_risks, use_container_width=True)
        else:
            st.warning("Could not load system statistics")
    
    with tab2:
        st.header("Latest Hardware Messages")
        
        # Get real messages
        messages_data = get_api_data("/api/messages/latest?limit=10")
        if messages_data:
            st.subheader(f"Showing {len(messages_data)} Latest Messages")
            
            for msg in messages_data:
                urgency_color = {
                    'critical': '🔴',
                    'high': '🟠', 
                    'medium': '🟡',
                    'low': '🟢'
                }.get(msg.get('urgency', 'medium'), '⚪')
                
                with st.expander(f"{urgency_color} {msg.get('channel', 'Unknown')} - {msg.get('user', 'Unknown')}"):
                    st.write(f"**Content:** {msg.get('content', '')}")
                    st.write(f"**Urgency:** {msg.get('urgency', 'Unknown')}")
                    st.write(f"**Category:** {msg.get('category', 'Unknown')}")
                    st.write(f"**Timestamp:** {msg.get('timestamp', '')}")
                    if msg.get('sentiment_score'):
                        sentiment = "Positive" if msg['sentiment_score'] > 0.5 else "Negative" if msg['sentiment_score'] < 0.3 else "Neutral"
                        st.write(f"**Sentiment:** {sentiment} ({msg['sentiment_score']:.2f})")
        else:
            st.warning("Could not load messages data")
    
    with tab3:
        st.header("Risk Assessments")
        
        # Get real risk assessments
        risks_data = get_api_data("/api/risk-assessment")
        if risks_data:
            st.subheader(f"Current Risk Assessments ({len(risks_data)})")
            
            for risk in risks_data:
                risk_color = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡', 
                    'low': '🟢'
                }.get(risk.get('risk_level', 'medium'), '⚪')
                
                attention_flag = "⚠️ REQUIRES ATTENTION" if risk.get('requires_attention') else ""
                
                with st.expander(f"{risk_color} {risk.get('risk_level', 'Unknown').upper()} - {risk.get('risk_category', 'Unknown')} {attention_flag}"):
                    st.write(f"**Description:** {risk.get('description', '')}")
                    st.write(f"**Confidence:** {risk.get('confidence_score', 0)*100:.1f}%")
                    st.write(f"**Created:** {risk.get('created_at', '')}")
                    if risk.get('requires_attention'):
                        st.error("⚠️ This risk requires immediate attention!")
        else:
            st.warning("Could not load risk assessments")
    
    with tab4:
        st.header("🤖 Multi-Agent AI System")
        
        # Get agent status
        agent_status = get_api_data("/api/agents/status")
        if agent_status:
            ai_status = agent_status.get("ai_system_status", {})
            
            # Agent System Overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Agent Count", ai_status.get("agent_count", 0))
            with col2:
                st.metric("Analysis History", ai_status.get("analysis_history_count", 0))
            with col3:
                initialized = "✅ Ready" if ai_status.get("is_initialized") else "❌ Not Ready"
                st.metric("System Status", initialized)
            
            # Individual Agent Status
            st.subheader("🚀 Specialized AI Agents")
            agents = ai_status.get("agents", {})
            
            # Create 4 columns for 4 agents
            if agents:
                agent_cols = st.columns(4)
                agent_info = [
                    ("gtm_commander", "🎯 GTM Risk Commander", "Holistic launch readiness assessment"),
                    ("supply_chain_intel", "📦 Supply Chain Intelligence", "Component availability & supplier performance"),
                    ("quality_anomaly", "🔍 Quality Anomaly Detection", "Defect patterns & yield forecasting"),
                    ("timeline_milestone", "⏰ Timeline & Milestone Tracker", "Schedule conflicts & dependency tracking")
                ]
                
                for i, (agent_key, agent_name, agent_desc) in enumerate(agent_info):
                    with agent_cols[i]:
                        st.markdown(f"**{agent_name}**")
                        agent_data = agents.get(agent_key, {})
                        health = agent_data.get("health", {})
                        
                        # OpenAI status
                        openai_available = health.get("openai_available", False)
                        openai_responsive = health.get("openai_responsive", False)
                        
                        if openai_responsive:
                            st.success("✅ AI Ready")
                        elif openai_available:
                            st.warning("⚠️ AI Available")
                        else:
                            st.error("❌ OpenAI Needed")
                        
                        # Analysis count
                        analysis_count = health.get("analysis_count", 0)
                        st.metric("Analyses", analysis_count)
                        
                        # Agent capabilities
                        info = agent_data.get("info", {})
                        capabilities = info.get("capabilities", [])
                        if capabilities:
                            with st.expander("View Capabilities"):
                                for cap in capabilities[:5]:  # Show first 5
                                    st.write(f"• {cap}")
                                if len(capabilities) > 5:
                                    st.write(f"+ {len(capabilities) - 5} more...")
            
            # Multi-Agent Demo
            st.subheader("🚀 Multi-Agent Intelligence Demo")
            if st.button("🤖 Run Multi-Agent Demo", type="primary"):
                with st.spinner("Running comprehensive multi-agent analysis..."):
                    demo_result = get_api_data("/api/agents/multi-agent-demo")
                    if demo_result:
                        st.success("🎉 Multi-Agent Analysis Complete!")
                        
                        # Analysis Overview
                        analysis_overview = demo_result.get("analysis_overview", {})
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Overall Risk Score", f"{analysis_overview.get('overall_risk_score', 0):.1f}/100")
                        with col2:
                            st.metric("AI Confidence", f"{analysis_overview.get('overall_confidence', 0)*100:.1f}%")
                        with col3:
                            st.metric("Messages Analyzed", analysis_overview.get('messages_analyzed', 0))
                        with col4:
                            st.metric("Agents Deployed", analysis_overview.get('agents_deployed', 0))
                        
                        # Agent Breakdown
                        st.subheader("🔍 Agent Analysis Breakdown")
                        agent_breakdown = demo_result.get("agent_breakdown", {})
                        
                        for agent_key, agent_info in agent_breakdown.items():
                            with st.expander(f"{agent_info.get('role', 'Unknown Agent')} - {agent_info.get('specialization', '')}"):
                                analysis = agent_info.get("analysis", {})
                                if analysis.get("success"):
                                    st.success("✅ Analysis Successful")
                                    if "overall_risk_score" in analysis:
                                        st.metric("Agent Risk Score", f"{analysis['overall_risk_score']:.1f}/100")
                                    if "confidence" in analysis:
                                        st.metric("Agent Confidence", f"{analysis['confidence']*100:.1f}%")
                                    if "recommendations" in analysis:
                                        st.write("**Recommendations:**")
                                        for rec in analysis["recommendations"][:3]:
                                            st.write(f"• {rec}")
                                else:
                                    error = analysis.get("error", "Unknown error")
                                    if "OpenAI" in error or "API key" in error:
                                        st.warning("⚠️ Requires OpenAI API key for full analysis")
                                    else:
                                        st.error(f"❌ {error}")
                        
                        # Sample Messages
                        sample_messages = demo_result.get("sample_messages_analyzed", [])
                        if sample_messages:
                            st.subheader("📧 Sample Messages Analyzed")
                            for i, msg in enumerate(sample_messages, 1):
                                with st.expander(f"Message {i} - {msg.get('priority', 'unknown')} priority"):
                                    st.write(msg.get("content", ""))
                        
                        # Next Steps
                        next_steps = demo_result.get("next_steps", [])
                        if next_steps:
                            st.subheader("🎯 Next Steps")
                            for step in next_steps:
                                st.write(f"• {step}")
                    else:
                        st.error("Failed to run multi-agent demo. Check API connection.")
        else:
            st.error("Could not connect to AI Agent system")

    with tab5:
        st.header("📋 Daily Hardware GTM Digest")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Controls")
            if st.button("🔄 Generate Digest", type="primary"):
                st.rerun()
            
            if st.button("📤 Copy for Slack", type="secondary"):
                st.info("Digest copied to clipboard!")
        
        with col1:
            # Get the formatted digest
            digest_data = get_api_data("/api/digest/formatted")
            if digest_data:
                formatted_text = digest_data.get("formatted_digest", "")
                raw_data = digest_data.get("raw_data", {})
                
                # Display the beautiful formatted digest
                st.markdown("### 🏭 Hardware GTM Daily Digest")
                
                # Risk Score with color coding
                risk_score = raw_data.get("risk_score", 0)
                status = raw_data.get("status", "Unknown")
                
                if risk_score >= 80:
                    st.success(f"🎯 **GTM RISK SCORE: {risk_score}/100**")
                elif risk_score >= 60:
                    st.warning(f"🎯 **GTM RISK SCORE: {risk_score}/100**")
                else:
                    st.error(f"🎯 **GTM RISK SCORE: {risk_score}/100**")
                
                st.markdown(f"**Status:** {status}")
                st.markdown("---")
                
                # Critical Alerts
                critical_alerts = raw_data.get("critical_alerts", [])
                if critical_alerts:
                    st.markdown(f"### 🚨 CRITICAL ALERTS ({len(critical_alerts)})")
                    for i, alert in enumerate(critical_alerts, 1):
                        urgency_emoji = "🔴" if alert.get('urgency') == 'critical' else "🟠"
                        with st.expander(f"{urgency_emoji} [{alert.get('category', 'Unknown')}] Alert {i}"):
                            st.write(f"**Description:** {alert.get('description', '')}")
                            st.write(f"**Impact:** {alert.get('impact', '')}")
                            st.write(f"**Action:** {alert.get('action', '')}")
                
                # Wins
                wins = raw_data.get("wins", [])
                if wins:
                    st.markdown(f"### ✅ WINS ({len(wins)})")
                    for win in wins:
                        st.markdown(f"• {win}")
                
                # Priorities
                priorities = raw_data.get("priorities", [])
                if priorities:
                    st.markdown(f"### 📋 TODAY'S PRIORITIES")
                    for priority in priorities:
                        st.markdown(f"□ {priority}")
                
                # AI Insights
                ai_insights = raw_data.get("ai_insights", "")
                if ai_insights:
                    st.markdown("### 💡 AI INSIGHTS")
                    st.info(f'"{ai_insights}"')
                
                # Stats
                summary_stats = raw_data.get("summary_stats", {})
                if summary_stats:
                    st.markdown("### 📊 SUMMARY STATISTICS")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Messages", summary_stats.get("total_messages", 0))
                    with col2:
                        st.metric("Critical Issues", summary_stats.get("critical_issues", 0))
                    with col3:
                        st.metric("High Priority", summary_stats.get("high_priority", 0))
                    with col4:
                        st.metric("Sentiment", f"{summary_stats.get('avg_sentiment', 0)}%")
                
                # Raw formatted text for copying
                st.markdown("---")
                st.markdown("### 📋 Formatted Text (for Slack)")
                st.code(formatted_text, language="markdown")
                
            else:
                st.error("Could not generate daily digest. Please check API connection.")

    with tab6:
        st.header("🗣️ Interactive GTM Strategy")
        st.markdown("**Ask the team strategic questions and let AI adjust GTM weights based on their responses!**")
        
        # Get demo data
        demo_data = get_api_data("/api/gtm/team-engagement-demo")
        
        if demo_data:
            st.success("🚀 Interactive GTM Strategy System Ready!")
            
            # Workflow explanation
            st.subheader("💡 How It Works")
            workflow = demo_data.get("workflow", {})
            for step, description in workflow.items():
                st.write(f"**{step.replace('_', ' ').title()}:** {description}")
            
            st.markdown("---")
            
            # Example Questions Section
            st.subheader("🤖 AI-Generated Strategic Questions")
            example_questions = demo_data.get("example_questions", [])
            
            for i, question in enumerate(example_questions, 1):
                with st.expander(f"Question {i}: {question.get('question', '')[:50]}..."):
                    st.write(f"**📝 Question:** {question.get('question', '')}")
                    st.write(f"**🎯 Type:** {question.get('type', '').title()}")
                    st.write(f"**👥 Target:** {question.get('target_audience', '')}")
                    
                    # Show team responses
                    responses = question.get("responses", [])
                    if responses:
                        st.write("**💬 Team Responses:**")
                        for resp in responses:
                            confidence_color = "🟢" if resp.get('confidence', 5) >= 7 else "🟡" if resp.get('confidence', 5) >= 5 else "🔴"
                            st.write(f"• **{resp.get('role', 'Unknown')}** {confidence_color} (Confidence: {resp.get('confidence', 5)}/10): {resp.get('answer', '')}")
            
            st.markdown("---")
            
            # AI Analysis Results
            st.subheader("🧠 AI Analysis of Team Responses")
            ai_analysis = demo_data.get("ai_analysis", {})
            
            # Key Insights
            key_insights = ai_analysis.get("key_insights", [])
            if key_insights:
                st.write("**🔍 Key Insights:**")
                for insight in key_insights:
                    st.info(f"💡 {insight}")
            
            # Weight Adjustments
            weight_adjustments = ai_analysis.get("recommended_weight_adjustments", {})
            if weight_adjustments:
                st.write("**⚖️ AI-Recommended Weight Adjustments:**")
                col1, col2 = st.columns(2)
                
                i = 0
                for factor, adjustment in weight_adjustments.items():
                    with col1 if i % 2 == 0 else col2:
                        factor_name = factor.replace('_', ' ').title()
                        if "INCREASE" in adjustment:
                            st.success(f"📈 **{factor_name}**: {adjustment}")
                        elif "DECREASE" in adjustment:
                            st.warning(f"📉 **{factor_name}**: {adjustment}")
                        else:
                            st.info(f"➡️ **{factor_name}**: {adjustment}")
                    i += 1
            
            # Immediate Actions
            immediate_actions = ai_analysis.get("immediate_actions", [])
            if immediate_actions:
                st.write("**⚡ Immediate Actions:**")
                for action in immediate_actions:
                    st.write(f"• {action}")
            
            # Next Questions
            next_questions = ai_analysis.get("next_questions", [])
            if next_questions:
                st.write("**❓ Follow-up Questions:**")
                for question in next_questions:
                    st.write(f"• {question}")
            
            st.markdown("---")
            
            # Value Proposition
            st.subheader("🎯 Why This is Revolutionary")
            value_props = demo_data.get("value_proposition", [])
            for prop in value_props:
                st.success(prop)
            
            # Interactive Demo Section
            st.subheader("🎮 Try It Yourself")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Generate New Strategic Questions", type="primary"):
                    with st.spinner("AI generating strategic questions..."):
                        questions_data = get_api_data("/api/gtm/strategic-questions")
                        if questions_data:
                            st.success("✅ New questions generated!")
                            questions = questions_data.get("questions", [])
                            for q in questions:
                                st.write(f"🤔 {q.get('question', '')}")
                        else:
                            st.warning("⚠️ Requires OpenAI API key to generate questions")
            
            with col2:
                st.write("**💬 Submit Your Response:**")
                user_role = st.selectbox("Your Role", ["Engineering", "Supply Chain", "Quality", "GTM", "Product"])
                user_answer = st.text_area("Your Answer", placeholder="What's your biggest concern?")
                user_confidence = st.slider("Confidence (1-10)", 1, 10, 5)
                
                if st.button("✅ Submit Response"):
                    if user_answer:
                        # In real system, this would call the submit API
                        st.success(f"✅ Response from {user_role} recorded: '{user_answer}' (Confidence: {user_confidence}/10)")
                        st.info("🔄 GTM weights will adjust based on team responses!")
                    else:
                        st.warning("Please enter your response")
            
            # Next Steps
            st.subheader("🚀 Next Steps")
            next_steps = demo_data.get("next_steps", [])
            for step in next_steps:
                st.write(f"• {step}")
                
        else:
            st.error("Could not load interactive strategy demo")
            st.info("💡 **Concept:** AI asks strategic questions to your team, analyzes responses, and dynamically adjusts GTM weights based on actual team priorities and confidence levels!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 Slack Digest System for Hardware GTM Acceleration | Real-time Hardware Team Communication Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 