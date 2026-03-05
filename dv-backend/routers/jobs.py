"""
Training Job Endpoints
Manage and monitor training jobs
"""
import asyncio
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from database import DatasetDB, JobDB, ModelDB
from db_config import SessionLocal
from db_models import TrainingRun
from job_logger import JobLogger
from job_worker import submit_training_job
from ml_training_worker import submit_ml_training_job
from models import (HyperparametersConfig, JobStatus, MessageResponse,
                    TrainingJobCreate, TrainingJobResponse)

router = APIRouter()


@router.get("", response_model=List[TrainingJobResponse])
async def list_jobs(status: Optional[JobStatus] = None):
    """
    List all training jobs with optional status filter

    - **status**: Filter by job status (pending, running, completed, failed)
    """
    jobs = JobDB.get_all(status=status.value if status else None)
    return jobs


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_job(job_id: str):
    """
    Get a specific training job by ID
    """
    job = JobDB.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.post("", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(
    job_request: TrainingJobCreate
):
    """
    Start a new training job

    Creates a training job that will train a CNN model on the specified dataset.
    The job runs in the background and can be monitored via the job ID.

    - **dataset_id**: ID of the dataset to train on
    - **model_name**: Optional name for the resulting model
    - **hyperparameters**: Optional hyperparameter configuration
    - **task**: Model task type (classification, regression, etc.)
    """
    try:
        # Validate dataset exists
        dataset = DatasetDB.get_by_id(job_request.dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {job_request.dataset_id} not found"
            )

        # Check dataset is ready
        if dataset.get("readiness") not in ["ready", "draft"]:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be in 'ready' or 'draft' state, currently: {dataset.get('readiness')}"
            )

        # Set default hyperparameters if not provided
        hyperparams = job_request.hyperparameters or HyperparametersConfig()

        # Get task value (handle both enum and string)
        task_value = job_request.task.value if hasattr(
            job_request.task, 'value') else str(job_request.task)

        # Create model record with queued status
        model_name = job_request.model_name or f"{dataset.get('name')}_model"
        model_data = {
            "name": model_name,
            "task": task_value,
            "framework": "pytorch",  # lowercase to match database constraint
            "tags": [dataset.get("name"), task_value],
            "description": f"CNN model training on {dataset.get('name')}",
            "status": "queued",
            "dataset_id": job_request.dataset_id,
        }
        new_model = ModelDB.create(model_data)

        # Create job record
        job_data = {
            "dataset_id": job_request.dataset_id,
            "model_id": new_model["id"],
            "status": "pending",
            "progress": 0.0,
            "current_iteration": 0,
            "total_iterations": hyperparams.max_iterations,
            "current_accuracy": None,
            "best_accuracy": None,
            "config": {
                "hyperparameters": hyperparams.model_dump(),
                "task": task_value,
            },
            "error": None
        }

        new_job = JobDB.create(job_data)

        # Submit training job to worker pool (runs in separate process)
        submit_training_job(
            job_id=new_job["id"],
            dataset=dataset,
            hyperparams=hyperparams.model_dump(),
            model_id=new_model["id"],
            model_name=model_name,
            task=job_request.task.value if hasattr(
                job_request.task, 'value') else str(job_request.task),
            strategy='auto'
        )

        return new_job
    except HTTPException:
        raise
    except Exception as e:
        import logging
        import traceback
        logging.error(
            f"Error creating training job: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.post("/{job_id}/cancel", response_model=MessageResponse)
async def cancel_job(job_id: str):
    """
    Cancel a running training job
    """
    job = JobDB.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.get("status") not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in status: {job.get('status')}"
        )

    # Terminate the training process
    from job_worker import get_worker_pool
    from ml_training_worker import get_ml_worker_pool

    # Determine which worker pool to use based on job type
    job_type = job.get("job_type", "training")
    pool = get_ml_worker_pool() if job_type == "automl_training" else get_worker_pool()
    process_terminated = pool.cancel_job(job_id)

    # Update job status to stopped
    JobDB.update(job_id, {
        "status": "stopped",
        "completed_at": datetime.now().isoformat(),
        "error_message": "Job stopped by user"
    })

    # Also update the associated model status to stopped
    if job.get("model_id"):
        ModelDB.update(job["model_id"], {
            "status": "stopped"
        })

        # Update the training run status to stopped (find by model_id)
        db = SessionLocal()
        try:
            training_run = db.query(TrainingRun).filter(
                TrainingRun.model_id == job["model_id"],
                TrainingRun.status.in_(["running", "pending"])
            ).first()
            if training_run:
                from datetime import timezone
                training_run.status = "stopped"
                training_run.completed_at = datetime.now(timezone.utc)

                # Use job's elapsed_time to calculate duration (more accurate than timestamps)
                # Parse elapsed_time string (format: "5m 25s", "1h 30m", etc.)
                elapsed_time_str = job.get("config", {}).get(
                    "elapsed_time") if job.get("config") else None
                if elapsed_time_str:
                    import re
                    total_seconds = 0
                    hours_match = re.search(r'(\d+)h', elapsed_time_str)
                    if hours_match:
                        total_seconds += int(hours_match.group(1)) * 3600
                    minutes_match = re.search(r'(\d+)m', elapsed_time_str)
                    if minutes_match:
                        total_seconds += int(minutes_match.group(1)) * 60
                    seconds_match = re.search(r'(\d+)s', elapsed_time_str)
                    if seconds_match:
                        total_seconds += int(seconds_match.group(1))

                    if total_seconds > 0:
                        training_run.duration_seconds = total_seconds
                        print(
                            f"[API] Using job elapsed_time for duration: {total_seconds}s")
                    else:
                        # Fallback to timestamp calculation
                        if training_run.started_at and training_run.completed_at:
                            duration = training_run.completed_at - training_run.started_at
                            training_run.duration_seconds = int(
                                duration.total_seconds())
                else:
                    # Fallback to timestamp calculation if no elapsed_time in job
                    if training_run.started_at and training_run.completed_at:
                        duration = training_run.completed_at - training_run.started_at
                        training_run.duration_seconds = int(
                            duration.total_seconds())

                db.commit()
                db.refresh(training_run)
                print(
                    f"[API] Updated training_run {training_run.id} status to stopped with duration {training_run.duration_seconds}s")
            else:
                print(
                    f"[API] No running/pending training_run found for model {job['model_id']}")
        except Exception as e:
            print(f"[API] Error updating training_run: {e}")
            db.rollback()
        finally:
            db.close()

    detail = "Training job has been stopped"
    if process_terminated:
        detail += " (process terminated)"
    else:
        detail += " (process was not running or already completed)"

    return MessageResponse(
        message=f"Job {job_id} cancelled",
        detail=detail
    )


@router.delete("/{job_id}", response_model=MessageResponse)
async def delete_job(job_id: str):
    """
    Delete a training job record
    """
    job = JobDB.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Don't allow deletion of running jobs
    if job.get("status") == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running job. Cancel it first."
        )

    success = JobDB.delete(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete job")

    return MessageResponse(
        message=f"Job {job_id} deleted successfully"
    )


@router.post("/{job_id}/restart", response_model=TrainingJobResponse)
async def restart_job(job_id: str):
    """
    Restart a failed or stopped job by creating a new job with the same configuration

    This creates a completely new job (with new ID) using the configuration from the original job.
    Useful for retrying failed training runs or re-running stopped jobs.

    Args:
        job_id: ID of the failed/stopped job to restart

    Returns:
        JobResponse: The newly created job

    Raises:
        HTTPException: If job not found, still running, or restart fails
    """
    # Get the original job
    original_job = JobDB.get_by_id(job_id)
    if not original_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Don't allow restarting running or queued jobs
    if original_job.get("status") in ["running", "queued", "pending"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot restart a {original_job.get('status')} job. Cancel it first."
        )

    # Get dataset
    dataset_id = original_job.get('dataset_id')
    if not dataset_id:
        raise HTTPException(
            status_code=400, detail="Original job has no dataset_id")

    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found")

    # Get original config
    config = original_job.get('config', {})
    job_type = original_job.get('job_type', 'training')
    model_id = original_job.get('model_id')

    try:
        # Route based on job type
        if job_type == 'automl_training':
            # Restart ML Pipeline job
            from pathlib import Path

            from ml_training_worker import submit_ml_training_job

            # Find CSV file
            dataset_path = dataset.get('file_path') or dataset.get('path')
            dataset_dir = Path(dataset_path)
            csv_files = list(dataset_dir.glob("*.csv"))

            if not csv_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"No CSV file found in dataset directory: {dataset_path}"
                )

            # Get target column from config or dataset metadata
            target_column = config.get('target_column')
            if not target_column:
                # Fallback to dataset metadata
                metadata = dataset.get('metadata', {})
                target_column = metadata.get('target_column', '')

            if not target_column:
                raise HTTPException(
                    status_code=400,
                    detail="Target column not found in job config or dataset metadata"
                )

            # Create new job record
            new_job_data = {
                "job_type": "automl_training",
                "dataset_id": dataset_id,
                "model_id": model_id,
                "status": "pending",
                "progress": 0.0,
                "current_iteration": 0,
                "total_iterations": 10,
                "config": {
                    "target_column": target_column,
                    "pipeline_stages": []
                },
            }
            new_job = JobDB.create(new_job_data)

            # Submit to ML worker
            submit_ml_training_job(
                job_id=new_job['id'],
                dataset_path=str(csv_files[0]),
                target_column=target_column,
                model_id=model_id,
                dataset_id=dataset_id
            )

            # Update model status
            ModelDB.update(model_id, {'status': 'queued'})

            return new_job

        else:
            # Restart DL training job
            from job_worker import get_worker_pool

            # Create new job record
            new_job_data = {
                "job_type": "training",
                "dataset_id": dataset_id,
                "model_id": model_id,
                "status": "pending",
                "progress": 0.0,
                "current_iteration": 0,
                "total_iterations": config.get('epochs', 10),
                "config": {
                    "task": config.get('task', 'classification'),
                    "strategy": config.get('strategy', 'auto'),
                    "hyperparameters": config.get('hyperparameters', {}),
                    "epochs": config.get('epochs', 10),
                    "batch_size": config.get('batch_size', 32),
                },
            }
            new_job = JobDB.create(new_job_data)

            # Submit to DL worker
            pool = get_worker_pool()
            job_data = {
                'job_id': new_job['id'],
                'dataset': dataset,
                'hyperparameters': config.get('hyperparameters', {}),
                'model_id': model_id,
                'model_name': original_job.get('model_name'),
                'task': config.get('task', 'classification'),
                'strategy': config.get('strategy', 'auto')
            }
            pool.submit_job(job_data)

            # Update model status
            ModelDB.update(model_id, {'status': 'queued'})

            return new_job

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart job: {str(e)}"
        )


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    max_lines: Optional[int] = Query(
        default=500, description="Maximum number of log lines to return (most recent)")
):
    """
    Get training logs for a job

    Returns real-time logs from the training process.
    Logs are streamed as they are generated and can be polled for updates.

    - **job_id**: ID of the training job
    - **max_lines**: Maximum number of log lines to return (default: 500, most recent lines)
    """
    job = JobDB.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Read logs from file
    logs = JobLogger.read_logs(job_id, max_lines=max_lines)

    return {
        "job_id": job_id,
        "logs": logs,
        "total_lines": len(logs)
    }

# ============= AUTO-ML TRAINING ENDPOINT =============


class AutoMLTrainingRequest(BaseModel):
    """Request model for AutoML training"""
    dataset_id: str
    model_name: Optional[str] = None
    target_column: str  # Required for AutoML


@router.post("/train-automl", response_model=TrainingJobResponse, status_code=201)
async def create_automl_training_job(request: AutoMLTrainingRequest, background_tasks: BackgroundTasks):
    """
    Start a new AutoML training job using ML Pipeline

    Creates a training job that will use the 10-stage LLM-driven ML Pipeline
    to automatically preprocess data, select the best model, and train it.

    **Same async pattern as DL training** - job runs in background worker process.

    **Requirements:**
    - Dataset must be in CSV format
    - Must specify target column for prediction

    **Process:**
    1. Creates Job and Model records in DeepVariance database
    2. Submits job to ML worker pool (runs in separate process)
    3. Worker updates Job/Model records during 10-stage execution
    4. Returns job information immediately for real-time monitoring

    - **dataset_id**: ID of the CSV dataset to train on
    - **target_column**: Name of the target column for prediction
    - **model_name**: Optional name for the resulting model
    """
    try:
        # Validate dataset exists
        dataset = DatasetDB.get_by_id(request.dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {request.dataset_id} not found"
            )

        # Check dataset is tabular (CSV)
        if dataset.get("domain") != "tabular":
            raise HTTPException(
                status_code=400,
                detail=f"AutoML only supports tabular datasets. This dataset domain is: {dataset.get('domain')}"
            )

        # Check dataset is ready
        if dataset.get("readiness") not in ["ready", "draft"]:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be in 'ready' or 'draft' state, currently: {dataset.get('readiness')}"
            )

        # Validate dataset file exists
        dataset_path = dataset.get("file_path") or dataset.get("path")
        if not dataset_path:
            raise HTTPException(
                status_code=400,
                detail="Dataset file path not found"
            )

        from pathlib import Path
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset directory not found: {dataset_path}"
            )

        # Find CSV file in the dataset directory
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            raise HTTPException(
                status_code=400,
                detail=f"No CSV file found in dataset directory: {dataset_path}"
            )

        csv_path = csv_files[0]  # Use first CSV file found

        # Validate AutoML requirements
        from validators import DatasetValidator
        try:
            validation_result = DatasetValidator.validate_automl_requirements(
                csv_path=csv_path,
                target_column=request.target_column
            )
            print(f"AutoML validation passed: {validation_result['message']}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset validation failed: {str(e)}"
            )

        # Create model record with queued status
        model_name = request.model_name or f"{dataset.get('name')}_automl_model"
        model_data = {
            "name": model_name,
            "task": "unknown",  # Will be auto-detected from target column before training
            "framework": "sklearn",     # ML Pipeline uses sklearn
            "tags": [dataset.get("name"), "automl"],
            "description": f"AutoML model training on {dataset.get('name')} using 8-stage pipeline",
            "status": "queued",
            "dataset_id": request.dataset_id,
        }
        new_model = ModelDB.create(model_data)

        # Create job record
        job_data = {
            "job_type": "automl_training",
            "dataset_id": request.dataset_id,
            "model_id": new_model["id"],
            "status": "pending",
            "progress": 0.0,
            "current_iteration": 0,
            "total_iterations": 10,  # 10 pipeline stages
            "config": {
                "target_column": request.target_column,
                "pipeline_stages": []
            },
            "error": None
        }

        new_job = JobDB.create(job_data)

        # Define background task to call ML Pipeline Service
        async def call_ml_pipeline_service():
            """Call ML Pipeline with retries and backoff; do not mark job failed on first transient error."""
            import httpx

            from config import get_config
            cfg = get_config()
            ml_service_url = cfg.ML_PIPELINE_URL

            # Configurable retry/backoff
            max_attempts = cfg.MAX_SUBMISSION_ATTEMPTS
            backoff_base = 2

            attempt = 0
            last_exc = None

            while attempt < max_attempts:
                attempt += 1
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{ml_service_url}/api/ml-pipeline/train",
                            json={
                                "job_id": new_job["id"],
                                "dataset_id": request.dataset_id,
                                "target_column": request.target_column,
                                "selected_models": None,  # Auto-select all models
                                "time_limit": None,  # No time limit
                                "preset": "medium_quality"
                            },
                            timeout=None  # Training can take a while
                        )
                        print(f"ML Pipeline response: {response.status_code}")

                        if response.status_code == 200 or response.status_code == 201:
                            # success
                            return
                        else:
                            last_exc = Exception(
                                f"ML pipeline returned status {response.status_code}")

                except Exception as e:
                    last_exc = e
                    print(
                        f"Attempt {attempt}/{max_attempts} - Error calling ML Pipeline service: {e}")

                # Exponential backoff before next attempt
                if attempt < max_attempts:
                    wait_seconds = backoff_base ** attempt
                    print(f"Retrying ML Pipeline call in {wait_seconds}s...")
                    await asyncio.sleep(wait_seconds)

            # If we reach here, all attempts failed — update job but keep as pending so resumer can retry
            print(f"All attempts to call ML Pipeline failed: {last_exc}")
            JobDB.update(new_job["id"], {
                "status": "pending",
                "error": f"Failed to call ML Pipeline after {max_attempts} attempts: {last_exc}",
                "config": {**(new_job.get('config') or {}), 'submission_attempts': max_attempts}
            })

        # Add task to background
        background_tasks.add_task(call_ml_pipeline_service)

        return new_job

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error creating AutoML training job: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Failed to create AutoML training job: {str(e)}")
