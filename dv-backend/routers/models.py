"""
Model Management Endpoints
Operations for trained models
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from models import (
    TrainedModelResponse, TrainedModelUpdate,
    MessageResponse, ModelTask, ModelStatus
)
from database import ModelDB
from db_config import SessionLocal
from db_models import TrainingRun
from sqlalchemy.orm import Session
import shutil
from pathlib import Path

router = APIRouter()

@router.get("", response_model=List[TrainedModelResponse])
async def list_models(
    task: Optional[ModelTask] = None,
    status: Optional[ModelStatus] = None,
    search: Optional[str] = None
):
    """
    List all trained models with optional filters

    - **task**: Filter by model task (classification, regression, etc.)
    - **status**: Filter by model status (active, training, etc.)
    - **search**: Search by model name
    """
    models = ModelDB.get_all(
        task=task.value if task else None,
        status=status.value if status else None,
        search=search
    )
    return models

@router.get("/{model_id}", response_model=TrainedModelResponse)
async def get_model(model_id: str):
    """
    Get a specific model by ID
    """
    model = ModelDB.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model

@router.put("/{model_id}", response_model=TrainedModelResponse)
async def update_model(model_id: str, model_update: TrainedModelUpdate):
    """
    Update model metadata (name, tags, description)
    """
    # Check if model exists
    existing = ModelDB.get_by_id(model_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Update model
    update_data = model_update.model_dump(exclude_unset=True)
    updated_model = ModelDB.update(model_id, update_data)

    return updated_model

@router.patch("/{model_id}/name", response_model=TrainedModelResponse)
async def update_model_name(model_id: str, name: str):
    """
    Update only the model name (convenience endpoint)
    """
    existing = ModelDB.get_by_id(model_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    updated_model = ModelDB.update(model_id, {"name": name})
    return updated_model

@router.delete("/{model_id}", response_model=MessageResponse)
async def delete_model(model_id: str):
    """
    Delete a trained model

    Deletes both the model record and associated model files from storage.
    Stopped models can be deleted. Active/training models cannot be deleted.
    """
    # Check if model exists
    model = ModelDB.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Check if model is in a state that can be deleted
    model_status = model.get("status")
    if model_status in ["training", "queued"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete model with status '{model_status}'. Stop the training first."
        )

    # Delete model files
    files_deleted = False
    model_path = model.get("model_path")
    if model_path:
        model_file = Path(model_path)
        if model_file.exists():
            try:
                model_file.unlink()
                files_deleted = True
                print(f"Deleted model file: {model_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete model file: {str(e)}"
                )

    # Delete from database
    success = ModelDB.delete(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model from database")

    return MessageResponse(
        message=f"Model '{model.get('name')}' deleted successfully",
        detail=f"Database record and files removed" if files_deleted else "Database record removed (no files found)"
    )

@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """
    Download model file (placeholder - would return actual file in production)
    """
    model = ModelDB.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    model_path = model.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Model file not found"
        )

    # In production, would use FileResponse to return the actual file
    # from fastapi.responses import FileResponse
    # return FileResponse(model_path, filename=f"{model['name']}.py")

    return MessageResponse(
        message="Model download endpoint",
        detail=f"Model path: {model_path}"
    )

@router.get("/{model_id}/training-history")
async def get_training_history(model_id: str):
    """
    Get training history for a model

    Returns all training runs for this model with their metrics and status.
    """
    # Check if model exists
    model = ModelDB.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Get training runs from database
    db: Session = SessionLocal()
    try:
        training_runs = db.query(TrainingRun).filter(
            TrainingRun.model_id == model_id
        ).order_by(
            TrainingRun.created_at.desc()
        ).all()

        # Convert to response format
        history = []
        for run in training_runs:
            # Calculate duration
            duration_seconds = run.duration_seconds
            if not duration_seconds:
                if run.started_at and run.completed_at:
                    # Completed job - use completed_at
                    duration = run.completed_at - run.started_at
                    duration_seconds = int(duration.total_seconds())
                elif run.started_at and run.status in ['running', 'pending']:
                    # Running job - calculate from now
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    duration = now - run.started_at
                    duration_seconds = int(duration.total_seconds())

            history.append({
                "id": str(run.id),
                "run_number": run.run_number,
                "status": run.status,
                "progress": float(run.progress) if run.progress else 0,
                "current_epoch": run.current_epoch,
                "total_epochs": run.total_epochs,
                "final_loss": float(run.final_loss) if run.final_loss else None,
                "final_accuracy": float(run.final_accuracy) if run.final_accuracy else None,
                "best_loss": float(run.best_loss) if run.best_loss else None,
                "best_accuracy": float(run.best_accuracy) if run.best_accuracy else None,
                "duration_seconds": duration_seconds,
                "error_message": run.error_message,
                "config": run.config,
                "epoch_metrics": run.epoch_metrics,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "dataset_id": str(run.dataset_id) if run.dataset_id else None,
            })

        return history
    finally:
        db.close()
