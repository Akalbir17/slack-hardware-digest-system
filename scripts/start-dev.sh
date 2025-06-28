#!/bin/bash

# Slack Digest System - Development Startup Script
# This script stops existing containers and starts fresh with latest builds

echo "🚀 Starting Slack Digest System in Development Mode..."
echo "=================================================="

# Change to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📂 Working directory: $(pwd)"

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Clean up any orphaned containers
echo "🧹 Cleaning up orphaned containers..."
docker-compose down --remove-orphans

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build

echo "✅ Development environment started!"
echo ""
echo "🌐 Access your applications:"
echo "   • FastAPI: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo "   • Streamlit: http://localhost:8501"
echo "   • PostgreSQL: localhost:5432"
echo "   • Redis: localhost:6379"
echo ""
echo "💡 To stop: Press Ctrl+C or run 'docker-compose down'" 