"""
ML Pipeline Microservice
Standalone FastAPI service that uses DeepVariance PostgreSQL database
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', '..'))

from config import validate_config, get_config
from database import initialize_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("\n🤖 Starting ML Pipeline Service...")

    try:
        validate_config()
    except Exception as e:
        print(f"\n❌ Failed to start: Configuration validation failed")
        print(f"Error: {e}\n")
        sys.exit(1)

    print("✓ Configuration validated successfully")
    print("\n📊 Initializing database connection...")
    initialize_db()
    print("✓ Database connection initialized")
    print("\n✅ ML Pipeline Service started successfully!\n")

    yield

    # Shutdown
    print("\n🛑 Shutting down ML Pipeline Service...")
    print("✓ Service stopped gracefully\n")

app = FastAPI(
    title="ML Pipeline Service",
    description="AutoML service for DeepVariance - LLM-driven 8-stage ML pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration (using centralized config)
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
from app.routes import router
app.include_router(router, prefix="/api/ml-pipeline", tags=["ML Pipeline"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Pipeline Service",
        "version": "1.0.0",
        "description": "AutoML service with 8-stage LLM-driven pipeline",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ml-pipeline"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # Different port from main DeepVariance backend
        reload=True,
        reload_excludes=["models/*", "results/*", "data/*"]
    )
