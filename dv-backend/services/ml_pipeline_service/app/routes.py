"""
ML Pipeline Service API Routes
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', '..'))

from database import JobDB, ModelDB, DatasetDB

router = APIRouter()

class MLPipelineRequest(BaseModel):
    """Request model for ML Pipeline training"""
    job_id: str
    dataset_id: str
    target_column: str

class MLPipelineResponse(BaseModel):
    """Response model for ML Pipeline jobs"""
    job_id: str
    status: str
    message: str

@router.post("/train", response_model=MLPipelineResponse)
async def train_with_ml_pipeline(request: MLPipelineRequest):
    """
    Trigger ML Pipeline training for a job

    - **job_id**: DeepVariance Job ID (UUID)
    - **dataset_id**: DeepVariance Dataset ID (UUID)
    - **target_column**: Target column name for prediction
    """
    # Validate job exists
    job = JobDB.get_by_id(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found")

    # Validate dataset exists and is CSV
    dataset = DatasetDB.get_by_id(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

    if dataset.get('file_format') != 'csv':
        raise HTTPException(
            status_code=400,
            detail="ML Pipeline only supports CSV datasets"
        )

    # Update job status to running
    JobDB.update(request.job_id, {'status': 'running'})

    # Import and run pipeline adapter
    from app.pipeline_adapter import MLPipelineAdapter

    try:
        adapter = MLPipelineAdapter()
        result = await adapter.run_pipeline(
            job_id=request.job_id,
            dataset_id=request.dataset_id,
            target_column=request.target_column
        )

        return MLPipelineResponse(
            job_id=request.job_id,
            status="completed",
            message="ML Pipeline completed successfully"
        )
    except Exception as e:
        # Update job as failed
        JobDB.update(request.job_id, {
            'status': 'failed',
            'error': str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Get ML Pipeline job status

    Returns current status and progress from DeepVariance database
    """
    job = JobDB.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Extract pipeline stages from config
    config = job.get('config', {})
    pipeline_stages = config.get('pipeline_stages', [])

    return {
        "job_id": job_id,
        "status": job.get('status'),
        "progress": job.get('progress', 0),
        "current_iteration": job.get('current_iteration', 0),
        "total_iterations": job.get('total_iterations', 10),
        "pipeline_stages": pipeline_stages,
        "metrics": {
            "accuracy": job.get('best_accuracy'),
            "loss": job.get('best_loss')
        },
        "error": job.get('error')
    }
