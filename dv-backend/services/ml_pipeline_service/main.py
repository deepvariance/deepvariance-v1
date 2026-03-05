"""
ML Pipeline Service - FastAPI Application
AutoGluon-based ML training service for DeepVariance
"""

from services.ml_pipeline_service.autogluon_pipeline import \
    run_autogluon_pipeline
from database import DatasetDB, JobDB, ModelDB
from config import get_config
import os
import sys
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


# Initialize FastAPI app
app = FastAPI(
    title="DeepVariance ML Pipeline Service",
    description="AutoGluon-based ML training service for tabular data",
    version="2.0.0"
)

# CORS configuration
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AutoMLTrainRequest(BaseModel):
    """Request model for AutoML training"""
    job_id: str = Field(..., description="Job ID from main backend")
    dataset_id: str = Field(..., description="Dataset ID")
    target_column: str = Field(..., description="Target column name")
    selected_models: Optional[List[str]] = Field(
        None,
        description="Optional list of AutoGluon model codes (e.g., ['GBM', 'CAT', 'XGB', 'LR', 'NN_TORCH'])"
    )
    time_limit: Optional[int] = Field(
        None,
        description="Training time limit in seconds (None = no limit)"
    )
    preset: str = Field(
        "medium_quality",
        description="AutoGluon preset: 'best_quality', 'high_quality', 'good_quality', 'medium_quality', or 'fast'"
    )


class AutoMLTrainResponse(BaseModel):
    """Response model for AutoML training"""
    success: bool
    job_id: str
    model_id: Optional[str] = None
    best_model: Optional[str] = None
    metrics: Optional[dict] = None
    message: str


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ml-pipeline-service",
        "version": "2.0.0",
        "pipeline": "autogluon"
    }


@app.post("/api/ml-pipeline/train", response_model=AutoMLTrainResponse)
async def train_automl(request: AutoMLTrainRequest):
    """
    Train AutoML models using AutoGluon.

    This endpoint:
    1. Loads the dataset from storage
    2. Trains AutoGluon models (automatic model selection or user-selected models)
    3. Evaluates on test set
    4. Saves best model
    5. Updates job and model records in database

    Returns training results and metrics.
    """
    print(f"\n{'='*60}")
    print(f"[TRAIN] Received AutoML training request")
    print(f"[TRAIN] Job ID: {request.job_id}")
    print(f"[TRAIN] Dataset ID: {request.dataset_id}")
    print(f"[TRAIN] Target: {request.target_column}")
    print(f"{'='*60}\n")

    try:
        # Get job from database
        job = JobDB.get_by_id(request.job_id)
        if not job:
            raise HTTPException(
                status_code=404, detail=f"Job {request.job_id} not found")

        # Get dataset from database
        dataset = DatasetDB.get_by_id(request.dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"Dataset {request.dataset_id} not found")

        # Validate dataset is CSV/tabular
        if dataset.get("domain") != "tabular":
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be tabular (CSV/Parquet). Got: {dataset.get('domain')}"
            )

        # Get dataset file path - support both CSV and Parquet
        dataset_dir = Path(config.DATA_DIR) / request.dataset_id
        data_files = (
            list(dataset_dir.glob("*.csv")) +
            list(dataset_dir.glob("*.parquet")) +
            list(dataset_dir.glob("*.pq"))
        )

        if not data_files:
            raise HTTPException(
                status_code=404,
                detail=f"No CSV or Parquet file found in dataset directory: {dataset_dir}"
            )

        dataset_path = str(data_files[0])  # Use first data file
        print(f"[TRAIN] Dataset path: {dataset_path}")

        # Prepare model output directory
        model_output_dir = Path(config.MODELS_DIR) / request.job_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_output_path = str(model_output_dir)

        # Update job status to running
        JobDB.update(request.job_id, {
            "status": "running",
            "progress": 0.0,
            "config": {
                "target_column": request.target_column,
                "selected_models": request.selected_models,
                "time_limit": request.time_limit,
                "preset": request.preset,
                "pipeline_type": "autogluon"
            }
        })

        # Progress callback to update job in database
        def progress_callback(stage: str, progress: float, message: str):
            """Update job progress in database"""
            JobDB.update(request.job_id, {
                "progress": progress,
                "config": {
                    "target_column": request.target_column,
                    "selected_models": request.selected_models,
                    "current_stage": stage,
                    "stage_message": message
                }
            })
            print(f"[PROGRESS] {stage}: {progress}% - {message}")

        # Run AutoGluon pipeline
        print(f"[TRAIN] Starting AutoGluon pipeline...")
        results = run_autogluon_pipeline(
            dataset_path=dataset_path,
            target_column=request.target_column,
            model_output_path=model_output_path,
            selected_models=request.selected_models,
            time_limit=request.time_limit,
            preset=request.preset,
            progress_callback=progress_callback
        )

        print(f"[TRAIN] Pipeline completed with status: {results['status']}")

        # Handle pipeline results
        if results["status"] == "completed":
            metrics = results.get("metrics", {})
            best_model = metrics.get("best_model", "unknown")
            problem_type = metrics.get("problem_type", "unknown")

            # Determine accuracy metric based on problem type
            accuracy = None
            if "class" in problem_type or "binary" in problem_type:
                accuracy = metrics.get("accuracy")
            elif "regression" in problem_type:
                accuracy = metrics.get("r2")

            # Update model record
            model_id = job.get("model_id")
            if model_id:
                ModelDB.update(model_id, {
                    "status": "ready",
                    "framework": "autogluon",
                    "accuracy": accuracy,
                    "metrics": metrics,
                    "model_path": model_output_path,
                    "config": {
                        "best_model": best_model,
                        "problem_type": problem_type,
                        "num_models_trained": metrics.get("num_models_trained"),
                        "leaderboard": results.get("leaderboard", [])
                    }
                })
                print(
                    f"[TRAIN] Updated model {model_id}: {best_model}, accuracy={accuracy}")

            # Update job to completed
            JobDB.update(request.job_id, {
                "status": "completed",
                "progress": 100.0,
                "result": {
                    "success": True,
                    "best_model": best_model,
                    "metrics": metrics,
                    "stages": results.get("stages", []),
                    "duration_seconds": results.get("total_duration_seconds")
                }
            })

            print(f"[TRAIN] ✓ Training completed successfully")
            print(f"[TRAIN] Best model: {best_model}")
            print(f"[TRAIN] Metrics: {metrics}")

            return AutoMLTrainResponse(
                success=True,
                job_id=request.job_id,
                model_id=model_id,
                best_model=best_model,
                metrics=metrics,
                message=f"Training completed. Best model: {best_model}"
            )

        else:
            # Training failed
            error_msg = "; ".join(results.get("errors", ["Unknown error"]))

            # Update model to failed
            model_id = job.get("model_id")
            if model_id:
                ModelDB.update(model_id, {
                    "status": "failed",
                    "error": error_msg
                })

            # Update job to failed
            JobDB.update(request.job_id, {
                "status": "failed",
                "error": error_msg,
                "result": results
            })

            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] Training failed: {error_msg}\n")

        # Update job to failed
        try:
            JobDB.update(request.job_id, {
                "status": "failed",
                "error": error_msg
            })
        except:
            pass

        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    # Start the service
    port = int(os.getenv("ML_PIPELINE_PORT", 8001))
    print(f"\n{'='*60}")
    print(f"Starting ML Pipeline Service on port {port}")
    print(f"Pipeline: AutoGluon")
    print(f"{'='*60}\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
