# 🚀 Slack Hardware Digest System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**An intelligent AI-powered system for analyzing hardware team communications and accelerating Go-To-Market (GTM) processes.**

This comprehensive system processes, analyzes, and visualizes Slack communications within hardware Go-To-Market teams using advanced AI agents, providing real-time insights, risk assessments, and automated daily digests to accelerate product launches.

## 🎯 Key Features

### 🤖 Multi-Agent AI Intelligence
- **GTM Risk Commander**: Holistic launch readiness assessment
- **Supply Chain Intelligence**: Component availability & supplier performance monitoring  
- **Quality Anomaly Detection**: Defect patterns & yield forecasting
- **Timeline & Milestone Tracker**: Schedule conflicts & dependency tracking

### 📊 Real-Time Analytics
- **Live Risk Scoring**: Dynamic GTM risk assessment with AI confidence metrics
- **Intelligent Message Processing**: Auto-categorization and sentiment analysis
- **Interactive Dashboard**: Beautiful Streamlit interface with multiple analysis views
- **Daily AI Digests**: Automated comprehensive daily summaries

### 🏗️ Enterprise Architecture
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Scalable Infrastructure**: Redis caching + PostgreSQL persistence
- **Containerized Deployment**: Full Docker-compose orchestration
- **Real-time Processing**: Continuous message analysis and alert generation

## 🎬 Demo & Screenshots

### 📹 Video Demo
> **Note**: Currently using simulated hardware team data for demonstration
<div align="center">
  <a href="https://youtu.be/PTcM5NFN3Mk">
    <img src="https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg" alt="Hardware Digest System Demo" style="width:100%;max-width:600px;">
  </a>
  <p><em>▶️ Click to watch the complete system demo (5 minutes)</em></p>
</div>


*Watch the complete system walkthrough showing real-time message analysis, AI agent coordination, and automated digest generation.*

### 🎭 Mock Data Simulation

**For demonstration purposes, this system uses an advanced mock data generator that creates realistic hardware team communications.**

The [`mock_data/generator.py`](mock_data/generator.py) creates authentic-looking Slack messages including:

- **🔧 Component Shortages**: "🚨 CRITICAL: Snapdragon 8 Gen 3 shortage at Foxconn. Current inventory: 250 units, need 1,200 for EVT"
- **📊 Quality Issues**: "📉 Quality alert: AMOLED 6.1\" Samsung E7 yield rate dropped to 89.2% (target: 94.5%)"  
- **📅 Timeline Updates**: "📅 DVT milestone update for Phoenix: delayed - additional testing required"
- **🌏 Supply Chain Alerts**: "🚢 Logistics update: 7 day delay on Li-ion 4500mAh shipment from Shenzhen"
- **🧪 Testing Results**: "🧪 Reliability testing: IMU BMI088 completed 3,500 cycles, 2 failures"

**Features of Mock Generator:**
- 15+ realistic hardware team personas
- 50+ actual component specifications (displays, chipsets, batteries, sensors)
- 14+ real manufacturing vendors (Foxconn, Pegatron, Inventec, etc.)
- Project codenames and milestone tracking (EVT, DVT, PVT, MP)
- Authentic urgency levels and team communication patterns

**Pre-loaded Crisis Day Scenario:**
The system includes [`crisis_day_messages.json`](crisis_day_messages.json) - a curated dataset simulating a challenging day in hardware development with multiple concurrent issues:
- Critical component shortages affecting production schedules
- Quality issues discovered in testing phases  
- Supply chain disruptions from geopolitical events
- Timeline adjustments across multiple product lines

This allows you to see the full system capabilities including crisis management and multi-risk coordination without needing actual Slack integration during evaluation.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/slack-hardware-digest-system.git
cd slack-hardware-digest-system

# Copy environment template
cp env.template .env
```

### 2. Configure Environment
Edit `.env` with your configuration:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Database Configuration  
POSTGRES_DB=slack_digest
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password

# Redis Configuration
REDIS_URL=redis://redis:6379

# Optional: Slack Integration
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
```

### 3. Launch System
```bash
# Start all services
docker-compose up -d

# Check system health
curl http://localhost:8000/health
```

### 4. Access Applications
- **🎯 Dashboard**: http://localhost:8501
- **🔧 API**: http://localhost:8000
- **📚 API Docs**: http://localhost:8000/docs

> **✨ Ready to Explore!** The system comes pre-loaded with 169 simulated hardware team messages and 65 AI-generated risk assessments, so you can immediately see all features in action without needing real Slack data.

## 🏗️ Architecture

```mermaid
graph TB
    A[Slack Messages] --> B[Message Processor]
    B --> C[AI Agent Manager]
    C --> D[GTM Commander]
    C --> E[Supply Chain Intel]
    C --> F[Quality Detection]
    C --> G[Timeline Tracker]
    D --> H[Risk Assessment DB]
    E --> H
    F --> H
    G --> H
    H --> I[Dashboard]
    H --> J[Daily Digest]
    H --> K[REST API]
```

## 📁 Project Structure

```
slack-hardware-digest-system/
├── 🤖 agents/              # AI Agent System
│   ├── base_agent.py       # Base agent framework
│   ├── gtm_commander.py    # GTM risk assessment
│   ├── supply_chain_agent.py # Supply chain intelligence
│   ├── quality_agent.py    # Quality anomaly detection
│   └── timeline_agent.py   # Timeline & milestone tracking
├── 🚀 app/                 # FastAPI Backend
│   ├── main.py            # API server
│   ├── health.py          # Health monitoring
│   └── gtm_config.py      # GTM configuration
├── 📊 dashboard/           # Streamlit Frontend
│   └── main.py            # Interactive dashboard
├── 🐳 docker/              # Docker Configuration
│   ├── Dockerfile.api     # API container
│   ├── Dockerfile.dashboard # Dashboard container
│   └── init.sql           # Database schema
├── 📝 mock_data/           # Development Data
│   └── generator.py       # Mock message generator
├── ⚙️ scripts/             # Utility Scripts
│   ├── start-dev.sh       # Development startup
│   └── start-dev.bat      # Windows development startup
├── 📋 requirements.txt     # Python dependencies
├── 🔗 SLACK_INTEGRATION.md # Complete Slack setup guide
└── 📖 README.md           # Main documentation
```

## 🔧 Configuration Guide

### OpenAI Setup
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add to `.env`: `OPENAI_API_KEY=sk-proj-your-key-here`
3. System automatically validates and enables AI features

### 🔗 Slack Integration

**Connect to your hardware team's Slack workspace for real-time intelligence.**

For complete setup instructions, see our detailed **[Slack Integration Guide](SLACK_INTEGRATION.md)**.

#### Quick Setup Summary:
1. Create Slack app with hardware bot permissions
2. Configure event subscriptions for message monitoring  
3. Install app and copy tokens to `.env`
4. Invite bot to your hardware team channels
5. Test with sample messages and verify dashboard updates

#### What You Get:
- 📡 **Real-time monitoring** of hardware team discussions
- 🤖 **AI analysis** of supply chain, quality, and timeline risks
- 📊 **Daily digest posts** with team activity summaries
- ⚠️ **Instant alerts** for critical issues requiring attention

## 🔄 How It Works in Practice

### Real-Time Message Processing
1. **Message Ingestion**: Bot monitors your configured Slack channels 24/7
2. **AI Classification**: Each message is analyzed by specialized AI agents:
   - Supply chain mentions (delays, shortages, vendor issues)
   - Quality concerns (defects, yields, testing failures) 
   - Timeline discussions (deadlines, dependencies, blockers)
   - Risk indicators (urgent, critical, blocked, delayed)

3. **Smart Aggregation**: Related messages are grouped and analyzed together
4. **Risk Scoring**: Dynamic risk assessment based on message content, urgency, and patterns
5. **Automated Reporting**: Daily digest generation and critical alert notifications

### Practical Use Cases

#### 🚨 Early Warning System
- **Supplier Risk**: "Chip shortage may delay Q3 delivery" → Immediate supply chain alert
- **Quality Issues**: "Yield dropped to 78% overnight" → Quality team notification
- **Timeline Slips**: "Testing phase needs extra 2 weeks" → Schedule impact analysis

#### 📈 Trend Analysis
- Track recurring supplier issues over time
- Monitor quality metrics and improvement trends
- Identify bottlenecks in your GTM process

#### 🤖 Intelligent Summaries
- **Daily Digest**: Comprehensive overview of all hardware team activity
- **Weekly Trends**: Pattern analysis and risk trajectory
- **Monthly Reports**: Strategic insights for leadership

### Sample Analysis Output
```json
{
  "message_analysis": {
    "original": "Components from AcmeCorp delayed again, 3rd time this month",
    "category": "supply_chain_risk",
    "risk_level": "high",
    "entities": ["AcmeCorp", "components", "delay"],
    "sentiment": "negative",
    "urgency": "medium",
    "ai_insights": [
      "Recurring supplier reliability issue",
      "Pattern suggests vendor relationship review needed",
      "Recommend backup supplier evaluation"
    ]
  }
}
```

## 🧪 Development

### Local Development
```bash
# Start infrastructure
docker-compose up -d postgres redis

# Run API (Terminal 1)
cd app && uvicorn main:app --reload --port 8000

# Run Dashboard (Terminal 2) 
cd dashboard && streamlit run main.py --server.port 8501
```

### Testing
```bash
# Run test suite
python -m pytest tests/

# Test API endpoints
curl http://localhost:8000/api/agents/multi-agent-demo
```

## 📊 API Reference

### Core Endpoints
- `GET /health` - System health check
- `GET /api/messages/latest` - Recent messages
- `GET /api/risk-assessment` - Current risk assessments
- `GET /api/agents/status` - AI agent system status
- `GET /api/agents/multi-agent-demo` - Run multi-agent analysis
- `GET /api/digest/daily` - Generate daily digest

### Dashboard Features
- **📈 Overview**: System metrics and KPIs
- **📢 Latest Messages**: Real-time message feed
- **⚠️ Risk Assessments**: AI-generated risk analysis
- **🤖 AI Agents**: Multi-agent intelligence system
- **📋 Daily Digest**: Automated comprehensive summaries
- **🗣️ Interactive Strategy**: AI-powered team engagement

## 🚢 Deployment

### Production Deployment
```bash
# Production build
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale fastapi=3
```

### Environment Variables
```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Security
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=your-domain.com

# Monitoring
SENTRY_DSN=your-sentry-dsn
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT models that power our AI agents
- **FastAPI** for the excellent async Python framework
- **Streamlit** for the beautiful and intuitive dashboard framework
- **Docker** for containerization and deployment simplicity

## 📞 Support

- 📧 **Issues**: [GitHub Issues](https://github.com/Akalbir17/slack-hardware-digest-system/issues)
- 📖 **Documentation**: [Wiki](https://github.com/Akalbir17/slack-hardware-digest-system/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Akalbir17/slack-hardware-digest-system/discussions)

---

**Built with ❤️ for GTM Teams**

*Accelerating hardware product launches through intelligent communication analysis* 
