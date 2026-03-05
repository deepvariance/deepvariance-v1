"""
DeepVariance FastAPI Backend
Main application entry point
"""
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_config, validate_config
from database import DatasetDB, JobDB, initialize_db
from job_worker import get_worker_pool
from routers import datasets, jobs, models, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup - Validate configuration first
    print("\n🔧 Starting DeepVariance Backend...")

    try:
        validate_config()
    except Exception as e:
        print(f"\n❌ Failed to start: Configuration validation failed")
        print(f"Error: {e}\n")
        sys.exit(1)

    print("✓ Configuration validated successfully")
    print("\n📊 Initializing database...")
    initialize_db()
    print("✓ Database initialized successfully")

    # Start background JobResumer (periodic retry/resubmit of pending jobs)
    print("\n🔁 Starting Job Resumer service...")
    from services.job_resumer import JobResumer
    resumer = JobResumer()
    # Start the resumer on the current asyncio loop (FastAPI uses one)
    import asyncio
    try:
        resumer.start(asyncio.get_event_loop())
        app.state.job_resumer = resumer
        print(
            f"✓ Job Resumer started (poll interval: {resumer.poll_interval}s)")
    except Exception as e:
        print(f"⚠️  Failed to start Job Resumer: {e}")

    # Resume queued jobs from previous session (keep for backward compatibility)
    print("\n🔄 Checking for queued jobs...")
    queued_jobs = JobDB.get_all()
    queued_count = 0
    for job in queued_jobs:
        # Resume jobs that are queued or pending (not yet started)
        if job.get('status') in ['queued', 'pending']:
            queued_count += 1
            job_type = job.get('job_type', 'training')
            print(
                f"  - Found {job.get('status')} job: {job['id']} (type: {job_type})")

            dataset_id = job.get('dataset_id')
            if dataset_id:
                dataset_obj = DatasetDB.get_by_id(dataset_id)
                if dataset_obj:
                    # Route to appropriate worker pool based on job type
                    if job_type == 'automl_training':
                        # Submit to ML Pipeline worker
                        from ml_training_worker import submit_ml_training_job
                        config = job.get('config', {})
                        dataset_path = dataset_obj.get(
                            'file_path') or dataset_obj.get('path')

                        # Find CSV file in dataset directory
                        from pathlib import Path
                        dataset_dir = Path(dataset_path)
                        csv_files = list(dataset_dir.glob("*.csv"))

                        if csv_files:
                            submit_ml_training_job(
                                job_id=job['id'],
                                dataset_path=str(csv_files[0]),
                                target_column=config.get('target_column', ''),
                                model_id=job.get('model_id'),
                                dataset_id=dataset_id
                            )
                            print(f"    ✓ Resumed ML training job")
                        else:
                            print(f"    ✗ No CSV file found for ML job")
                    else:
                        # Submit to DL worker (original logic)
                        pool = get_worker_pool()
                        config = job.get('config', {})
                        job_data = {
                            'job_id': job['id'],
                            'dataset': dataset_obj,
                            'hyperparameters': config.get('hyperparameters', {}),
                            'model_id': job.get('model_id'),
                            'model_name': job.get('model_name'),
                            'task': config.get('task', 'classification'),
                            'strategy': config.get('strategy', 'auto')
                        }
                        pool.submit_job(job_data)
                        print(f"    ✓ Resumed DL training job")

    if queued_count > 0:
        print(f"✓ Resumed {queued_count} queued job(s)")
    else:
        print("✓ No queued jobs found")

    print("\n✅ Server started successfully!\n")

    yield

    # Shutdown
    print("\n🛑 Shutting down server...")

    # Stop Job Resumer if running
    try:
        res = getattr(app.state, 'job_resumer', None)
        if res:
            print("Stopping Job Resumer...")
            await res.stop()
            print("✓ Job Resumer stopped")
    except Exception as e:
        print(f"Error stopping Job Resumer: {e}")

    print("✓ Server stopped gracefully\n")

app = FastAPI(
    title="DeepVariance API",
    description="REST API for CNN Model Training and Dataset Management",
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

# Include routers
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Training Jobs"])
app.include_router(system.router, prefix="/api/system", tags=["System"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DeepVariance API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Exclude generated model files from auto-reload to prevent interruption during training
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["models/*", "results/*", "data/*"]
    )
