#!/bin/bash

# DV Photo Validator - Quick Start Script

echo "🚀 Starting DV Photo Validator..."
echo ""

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 is not installed."
    echo "📦 Install it with: brew install python@3.12"
    exit 1
fi

echo "✅ Python 3.12 found"
echo ""

# Setup backend
cd backend

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.12 -m venv .venv
fi

echo "📦 Activating virtual environment..."
source .venv/bin/activate

if [ ! -f ".venv/bin/uvicorn" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Backend ready!"
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo ""
echo "📋 Next steps:"
echo "   1. Backend is starting now..."
echo "   2. In another terminal, run: cd frontend && python -m http.server 5173"
echo "   3. Open http://localhost:5173 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload

