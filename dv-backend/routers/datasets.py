"""
Dataset Management Endpoints

This module provides CRUD operations for dataset management including:
- Listing and filtering datasets
- Creating datasets from file uploads
- Updating dataset metadata
- Deleting datasets and associated files
"""
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from database import DatasetDB
from models import (DatasetDomain, DatasetReadiness, DatasetResponse,
                    DatasetUpdate, MessageResponse)
from task_inference import (detect_supported_tasks_at_upload,
                            get_task_confidence, infer_task_from_dataset)
from validators import DatasetValidator, ValidationError

router = APIRouter()

# Constants
DATA_DIR = Path("./data")
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file streaming


def _calculate_dataset_size(dataset_dir: Path) -> tuple[int, int]:
    """
    Calculate dataset size by counting files and total bytes.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        Tuple of (file_count, total_bytes)
    """
    file_count = 0
    total_bytes = 0

    for root, _, files in os.walk(dataset_dir):
        # Skip hidden files and macOS metadata
        visible_files = [f for f in files if not f.startswith(
            '.') and f != '.DS_Store']
        file_count += len(visible_files)

        for file_name in visible_files:
            file_path = Path(root) / file_name
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                # Skip files that can't be accessed
                continue

    return file_count, total_bytes


def _parse_tags(tags: Optional[str]) -> List[str]:
    """
    Parse comma-separated tags string into a list.

    Args:
        tags: Comma-separated string of tags

    Returns:
        List of trimmed tag strings
    """
    if not tags:
        return []
    return [tag.strip() for tag in tags.split(',') if tag.strip()]


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    domain: Optional[DatasetDomain] = None,
    readiness: Optional[DatasetReadiness] = None,
    search: Optional[str] = None
):
    """
    List all datasets with optional filters.

    Args:
        domain: Filter by domain (tabular, vision, text, audio)
        readiness: Filter by readiness status
        search: Search by dataset name

    Returns:
        List of datasets matching the filters
    """
    datasets = DatasetDB.get_all(
        domain=domain.value if domain else None,
        readiness=readiness.value if readiness else None,
        search=search
    )
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """
    Get a specific dataset by ID.

    Args:
        dataset_id: Unique dataset identifier

    Returns:
        Dataset details

    Raises:
        HTTPException: If dataset not found
    """
    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )
    return dataset


@router.get("/{dataset_id}/columns")
async def get_dataset_columns(dataset_id: str):
    """
    Get column names and shape from a tabular dataset (CSV).

    Args:
        dataset_id: Unique dataset identifier

    Returns:
        Column names and dataset shape information

    Raises:
        HTTPException: If dataset not found or not tabular
    """
    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Check if dataset is tabular
    if dataset.get("domain") != "tabular":
        raise HTTPException(
            status_code=400,
            detail="This endpoint only works for tabular datasets"
        )

    # Find CSV file in dataset directory
    dataset_path = dataset.get("file_path") or dataset.get("path")
    if not dataset_path:
        raise HTTPException(
            status_code=400,
            detail="Dataset path not found"
        )

    from pathlib import Path

    import pandas as pd

    dataset_dir = Path(dataset_path)
    # Support both CSV and Parquet files
    from file_utils import get_data_files
    data_files = get_data_files(dataset_dir)
    csv_files = data_files  # Keep variable name for compatibility

    if not data_files:
        raise HTTPException(
            status_code=400,
            detail="No CSV or Parquet file found in dataset directory"
        )

    try:
        # Read data file to get full shape information
        from file_utils import read_dataframe
        df = read_dataframe(data_files[0])
        columns = df.columns.tolist()

        # Get data types for each column
        dtypes = {col: str(df[col].dtype) for col in columns}

        return {
            "dataset_id": dataset_id,
            "columns": columns,
            "total_columns": len(columns),
            "total_rows": len(df),
            "shape": {"rows": len(df), "columns": len(columns)},
            "dtypes": dtypes
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read CSV file: {str(e)}"
        )


@router.post("", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    name: str = Form(...),
    domain: DatasetDomain = Form(...),
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None)
):
    """
    Upload and create a new dataset.

    Supports large file uploads up to 100GB with streaming.
    Automatically extracts ZIP archives and calculates dataset size.

    Args:
        name: Dataset name (required)
        domain: Dataset domain - vision, tabular, text, or audio (required)
        file: Dataset file (ZIP archive recommended for large datasets) (required)
        tags: Comma-separated tags (optional)
        description: Dataset description (optional)
        target_column: Target column name for tabular datasets (optional)

    Returns:
        DatasetResponse: Created dataset with metadata

    Raises:
        HTTPException: If file save/extraction fails or dataset already exists

    Notes:
        - The backend automatically creates storage directory at ./data/{uuid}/
        - ZIP files are extracted and the archive is removed
        - Dataset size is calculated by walking the directory tree
        - Vision datasets are marked as 'ready', others as 'draft'
    """
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Generate UUID for this dataset upfront (before file operations)
    import uuid
    dataset_id = str(uuid.uuid4())

    # Create dataset-specific directory using UUID instead of name
    dataset_dir = DATA_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_path = dataset_dir / file.filename

    try:
        # Stream file to disk for efficient large file handling
        with file_path.open("wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                buffer.write(chunk)
    except Exception as e:
        # Clean up on failure
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    finally:
        await file.close()

    # Extract ZIP files automatically
    if file.filename.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            file_path.unlink()  # Remove ZIP after extraction
            print(f"Extracted ZIP file to {dataset_dir}")
        except zipfile.BadZipFile as e:
            shutil.rmtree(dataset_dir)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ZIP file: {str(e)}"
            )
        except Exception as e:
            shutil.rmtree(dataset_dir)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract ZIP file: {str(e)}"
            )

    # Calculate dataset statistics
    file_count, total_bytes = _calculate_dataset_size(dataset_dir)

    # Parse tags
    tag_list = _parse_tags(tags)

    # Validate dataset structure
    validation_result = None
    readiness = "draft"

    try:
        validation_result = DatasetValidator.validate_dataset(
            dataset_path=dataset_dir,
            domain=domain.value,
            task=None  # Task not specified at upload time
        )
        readiness = "ready"
        print(f"Validation passed: {validation_result.get('message')}")
    except ValidationError as e:
        # Validation failed - clean up and return error
        error_message = str(e)
        print(f"Validation failed: {error_message}")

        # Delete uploaded files
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            print(f"Cleaned up invalid dataset: {dataset_dir}")

        raise HTTPException(
            status_code=400,
            detail=f"Dataset validation failed: {error_message}"
        )
    except Exception as e:
        # Unexpected validation error - clean up and fail
        print(f"Validation error (unexpected): {str(e)}")

        # Delete uploaded files
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            print(f"Cleaned up dataset after error: {dataset_dir}")

        raise HTTPException(
            status_code=500,
            detail=f"Dataset validation error: {str(e)}"
        )

    # Build dataset record
    size_mb = total_bytes / (1024**2)

    # Use validation info if available
    if validation_result:
        file_count = validation_result.get('total_samples', file_count)

    dataset_description = description
    if not dataset_description:
        if validation_result:
            dataset_description = validation_result.get(
                'message', f"Dataset: {name}")
        else:
            dataset_description = f"Dataset: {name} ({file_count} files, {size_mb:.2f} MB)"

    # Auto-detect supported tasks based on structure
    task_detection = detect_supported_tasks_at_upload(
        dataset_path=dataset_dir,
        domain=domain.value,
        structure=validation_result.get(
            'structure', {}) if validation_result else {}
    )

    dataset_data = {
        "id": dataset_id,  # Use our pre-generated UUID
        "name": name,
        "domain": domain.value,
        "storage": "local",
        "path": str(dataset_dir),
        "size": file_count,
        "tags": tag_list,
        "description": dataset_description,
        "readiness": readiness,
        "structure": {
            **(validation_result.get('structure', {}) if validation_result else {}),
            "supported_tasks": task_detection['supported_tasks'],
            "recommended_task": task_detection['recommended_task'],
            "task_reasoning": task_detection['reasoning']
        },
        "metadata": {
            "target_column": target_column
        } if target_column else None
    }

    new_dataset = DatasetDB.create(dataset_data)

    print(
        f"Dataset created: {name} | Files: {file_count} | Size: {size_mb:.2f} MB | Status: {readiness}")

    return new_dataset


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(dataset_id: str, dataset_update: DatasetUpdate):
    """
    Update dataset metadata (name, tags, description, readiness).

    Args:
        dataset_id: Unique dataset identifier
        dataset_update: Fields to update

    Returns:
        Updated dataset details

    Raises:
        HTTPException: If dataset not found
    """
    # Check if dataset exists
    existing = DatasetDB.get_by_id(dataset_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Update dataset
    update_data = dataset_update.model_dump(exclude_unset=True)
    updated_dataset = DatasetDB.update(dataset_id, update_data)

    return updated_dataset


@router.patch("/{dataset_id}/name", response_model=DatasetResponse)
async def update_dataset_name(dataset_id: str, name: str):
    """
    Update only the dataset name (convenience endpoint).

    Args:
        dataset_id: Unique dataset identifier
        name: New dataset name

    Returns:
        Updated dataset details

    Raises:
        HTTPException: If dataset not found
    """
    existing = DatasetDB.get_by_id(dataset_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    updated_dataset = DatasetDB.update(dataset_id, {"name": name})
    return updated_dataset


@router.post("/{dataset_id}/validate")
async def validate_dataset_for_task(dataset_id: str, task: Optional[str] = None):
    """
    Validate a dataset for a specific task.

    Args:
        dataset_id: Unique dataset identifier
        task: Task type (classification, regression, clustering, detection)

    Returns:
        Validation result with details

    Raises:
        HTTPException: If dataset not found or validation fails
    """
    # Check if dataset exists
    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    dataset_path = Path(dataset.get("path", ""))
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset files not found on disk"
        )

    try:
        validation_result = DatasetValidator.validate_dataset(
            dataset_path=dataset_path,
            domain=dataset.get("domain"),
            task=task
        )

        # Update dataset readiness if validation passed
        if validation_result.get("valid"):
            DatasetDB.update(dataset_id, {"readiness": "ready"})

        return {
            "valid": True,
            "dataset_id": dataset_id,
            "validation": validation_result
        }
    except ValidationError as e:
        # Update dataset to error status
        error_desc = f"Validation error: {str(e)}"
        DatasetDB.update(dataset_id, {
            "readiness": "error",
            "description": f"{dataset.get('description', '')} | {error_desc}"
        })
        return {
            "valid": False,
            "dataset_id": dataset_id,
            "error": str(e),
            "message": "Dataset validation failed"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )


@router.delete("/{dataset_id}", response_model=MessageResponse)
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset and all associated files.

    Args:
        dataset_id: Unique dataset identifier

    Returns:
        Success message

    Raises:
        HTTPException: If dataset not found or deletion fails
    """
    # Check if dataset exists
    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Delete files from storage
    files_deleted = False
    # Default to local storage if not specified (since storage field is not in DB schema)
    storage_type = dataset.get("storage", "local")
    if storage_type == "local":
        # Use file_path from database (which is mapped to 'path' in model_to_dict)
        dataset_path = Path(dataset.get("path", ""))
        if dataset_path and dataset_path.exists():
            try:
                shutil.rmtree(dataset_path)
                files_deleted = True
                print(f"Deleted dataset files: {dataset_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete dataset files: {str(e)}"
                )

    # Delete from database
    success = DatasetDB.delete(dataset_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete dataset from database"
        )

    detail = "Database record and files removed" if files_deleted else "Database record removed (no files found)"
    return MessageResponse(
        message=f"Dataset '{dataset.get('name')}' deleted successfully",
        detail=detail
    )


@router.get("/{dataset_id}/suggest-task")
async def suggest_task_for_dataset(dataset_id: str):
    """
    Suggest the most appropriate ML task type for a dataset

    Uses smart inference based on:
    - Dataset name patterns
    - Dataset structure (classes, annotations, etc.)
    - Domain type (vision, tabular, etc.)
    - Description keywords

    Returns:
        - suggested_task: The recommended task type
        - confidence_scores: Confidence for each task type
        - reasoning: Explanation of the suggestion
    """
    # Get dataset
    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found")

    # Infer task
    suggested_task = infer_task_from_dataset(dataset)
    confidence_scores = get_task_confidence(dataset)

    # Generate reasoning
    reasoning = []
    name = dataset.get('name', '').lower()
    domain = dataset.get('domain', '')
    structure = dataset.get('structure', {})

    if any(kw in name for kw in ['classification', 'mnist', 'cifar', 'flower']):
        reasoning.append(
            f"Dataset name '{dataset.get('name')}' suggests classification task")

    if any(kw in name for kw in ['regression', 'prediction', 'forecast', 'price']):
        reasoning.append(
            f"Dataset name '{dataset.get('name')}' suggests regression task")

    if domain == 'vision' and structure.get('classes'):
        num_classes = len(structure.get('classes', []))
        reasoning.append(
            f"Vision dataset with {num_classes} classes indicates classification")

    if 'annotations' in structure or 'labels' in structure:
        reasoning.append(
            "Dataset contains annotations, suitable for detection tasks")

    if not reasoning:
        reasoning.append(f"Default suggestion based on domain: {domain}")

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.get('name'),
        "suggested_task": suggested_task,
        "confidence_scores": confidence_scores,
        "reasoning": reasoning,
        "all_tasks": list(confidence_scores.keys())
    }
