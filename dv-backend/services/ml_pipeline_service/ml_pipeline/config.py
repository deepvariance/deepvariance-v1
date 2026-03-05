"""
Configuration settings for the ML pipeline.
"""

import os

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY", "")

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY", "")
