#!/bin/bash

echo "================================================"
echo "ML Pipeline Test Setup"
echo "================================================"
echo ""

# Prompt for OpenAI API Key
echo "Enter your OpenAI API Key (starts with 'sk-'):"
read -r OPENAI_KEY

# Prompt for Groq API Key
echo "Enter your Groq API Key (starts with 'gsk_'):"
read -r GROQ_KEY

echo ""
echo "✅ API Keys configured"
echo ""

# Export for current session
export OPENAI_API_KEY="$OPENAI_KEY"
export GROQ_API_KEY="$GROQ_KEY"

# Run the test
echo "================================================"
echo "Starting ML Pipeline Test..."
echo "================================================"
echo ""

python3 test_credit_pipeline.py

echo ""
echo "================================================"
echo "Test Complete!"
echo "================================================"
