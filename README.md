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

## 🎬 Demo

### System Overview
![System Status](docs/screenshots/overview.png)

### Multi-Agent Intelligence
![AI Agents](docs/screenshots/agents.png)

### Daily Digest
![Daily Digest](docs/screenshots/digest.png)

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
└── 📋 requirements.txt     # Python dependencies
```

## 🔧 Configuration Guide

### OpenAI Setup
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add to `.env`: `OPENAI_API_KEY=sk-proj-your-key-here`
3. System automatically validates and enables AI features

### Slack Integration (Optional)
1. Create a Slack app in your workspace
2. Enable bot scopes: `channels:read`, `channels:history`, `chat:write`
3. Install app and copy tokens to `.env`

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

- 📧 **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/slack-hardware-digest-system/issues)
- 📖 **Documentation**: [Wiki](https://github.com/YOUR_USERNAME/slack-hardware-digest-system/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/slack-hardware-digest-system/discussions)

---

**Built with ❤️ for Hardware GTM Teams**

*Accelerating hardware product launches through intelligent communication analysis* 