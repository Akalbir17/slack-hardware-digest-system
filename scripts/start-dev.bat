@echo off
REM Slack Digest System - Development Startup Script (Windows)
REM This script stops existing containers and starts fresh with latest builds

echo 🚀 Starting Slack Digest System in Development Mode...
echo ==================================================

REM Change to project directory
pushd "%~dp0.."
echo 📂 Working directory: %cd%

REM Stop existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Clean up any orphaned containers
echo 🧹 Cleaning up orphaned containers...
docker-compose down --remove-orphans

REM Build and start services
echo 🔨 Building and starting services...
docker-compose up --build

echo ✅ Development environment started!
echo.
echo 🌐 Access your applications:
echo    • FastAPI: http://localhost:8000
echo    • API Docs: http://localhost:8000/docs
echo    • Streamlit: http://localhost:8501
echo    • PostgreSQL: localhost:5432
echo    • Redis: localhost:6379
echo.
echo 💡 To stop: Press Ctrl+C or run 'docker-compose down'

popd
pause 