#!/bin/bash

echo "🔑 Loading API keys from .env file..."

# Source the .env file from parent directory
set -a
source /Users/saaivigneshp/Desktop/dv-backend/.env
set +a

echo "✅ API keys loaded"
echo ""

# Navigate to the service directory
cd /Users/saaivigneshp/Desktop/dv-backend/services/ml_pipeline_service

echo "🚀 Starting ML Pipeline Test..."
echo ""

# Run the test with all passed arguments
python3 test_credit_pipeline.py "$@"

echo ""
echo "✅ Test completed!"
