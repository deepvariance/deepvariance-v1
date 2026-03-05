#!/bin/bash
# Start ML Pipeline Service
# This script starts the ML Pipeline microservice on port 8001

echo "🤖 Starting ML Pipeline Service..."
echo ""

# Navigate to the service directory
cd "$(dirname "$0")"

# Activate virtual environment from parent dv-backend
if [ -f "../../venv/bin/activate" ]; then
    echo "✓ Activating virtual environment..."
    source ../../venv/bin/activate
else
    echo "❌ Virtual environment not found at ../../venv"
    echo "Please create a virtual environment in dv-backend root:"
    echo "  cd ../.. && python -m venv venv && source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$GROQ_API_KEY" ] || [ -z "$DATABASE_URL" ]; then
    echo "⚠️  Environment variables not loaded"
    if [ -f "../../.env" ]; then
        echo "✓ Loading from ../../.env"
        export $(cat ../../.env | grep -v '^#' | xargs)
    else
        echo "❌ .env file not found in dv-backend root"
        echo "Please create .env file with GROQ_API_KEY and DATABASE_URL"
        exit 1
    fi
fi

echo "✓ Environment variables loaded"
echo ""

# Start the service
echo "Starting FastAPI service on http://localhost:8001"
echo "Docs available at: http://localhost:8001/docs"
echo ""

python -m app.main
