"""
Pydantic Models and Schemas
Data validation and serialization models for the API
"""
from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator

# ============= Enums =============


class DatasetDomain(str, Enum):
    """Dataset domain types"""
    TABULAR = "tabular"
    VISION = "vision"


class DatasetReadiness(str, Enum):
    """Dataset readiness status"""
    READY = "ready"
    PROFILING = "profiling"
    PROCESSING = "processing"
    DRAFT = "draft"
    ERROR = "error"


class StorageType(str, Enum):
    """Storage provider types"""
    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"


class ModelTask(str, Enum):
    """Model task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DETECTION = "detection"
    UNKNOWN = "unknown"  # Temporary state before auto-detection


class ModelStatus(str, Enum):
    """Model status"""
    ACTIVE = "active"
    READY = "ready"
    TRAINING = "training"
    QUEUED = "queued"
    DRAFT = "draft"
    FAILED = "failed"
    STOPPED = "stopped"


class JobStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPED = "stopped"

# ============= Dataset Schemas =============


class DatasetBase(BaseModel):
    """Base dataset schema"""
    name: str = Field(..., min_length=1, max_length=255)
    domain: DatasetDomain
    size: Optional[int] = Field(
        None, ge=0, description="Number of samples/rows")
    storage: StorageType = StorageType.LOCAL
    path: Optional[str] = Field(None, description="Path to dataset on storage")
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    metadata: Optional[dict] = Field(
        None, description="Additional metadata (e.g., target_column)")


class DatasetCreate(DatasetBase):
    """Schema for creating a dataset"""
    pass


class DatasetUpdate(BaseModel):
    """Schema for updating a dataset"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    readiness: Optional[DatasetReadiness] = None
    metadata: Optional[dict] = Field(
        None, description="Additional metadata (e.g., target_column for tabular datasets)"
    )


class DatasetResponse(DatasetBase):
    """Schema for dataset response"""
    id: str
    readiness: DatasetReadiness = DatasetReadiness.DRAFT
    created_at: datetime
    updated_at: datetime
    last_modified: Optional[str] = None
    freshness: Optional[str] = None
    metadata: Optional[dict] = None

    class Config:
        from_attributes = True

# ============= Model Schemas =============


class TrainedModelBase(BaseModel):
    """Base trained model schema"""
    name: str = Field(..., min_length=1, max_length=255)
    task: ModelTask
    framework: str = "PyTorch"
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class TrainedModelCreate(TrainedModelBase):
    """Schema for creating a model (via training job)"""
    dataset_id: str


class TrainedModelUpdate(BaseModel):
    """Schema for updating a model"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    tags: Optional[List[str]] = None
    description: Optional[str] = None


class TrainedModelResponse(TrainedModelBase):
    """Schema for model response"""
    id: str
    version: str = "v0.1.0"
    status: ModelStatus = ModelStatus.DRAFT
    accuracy: Optional[float] = Field(None, ge=0.0, le=100.0)
    created_at: datetime
    updated_at: datetime
    last_trained: Optional[datetime] = None
    dataset_id: Optional[str] = None
    model_path: Optional[str] = None

    class Config:
        from_attributes = True

# ============= Training Job Schemas =============


class HyperparametersConfig(BaseModel):
    """Hyperparameters configuration"""
    learning_rate: float = Field(1e-3, gt=0, le=1)
    batch_size: int = Field(32, ge=1, le=512)
    optimizer: Literal["Adam", "SGD", "RMSprop"] = "Adam"
    dropout_rate: float = Field(0.2, ge=0.0, le=0.9)
    epochs: int = Field(3, ge=1, le=100)
    max_iterations: int = Field(
        10, ge=1, le=50, description="Max agent iterations")
    target_accuracy: float = Field(1.0, ge=0.0, le=1.0)


class TrainingJobCreate(BaseModel):
    """Schema for creating a training job"""
    dataset_id: str
    model_name: Optional[str] = None
    hyperparameters: Optional[HyperparametersConfig] = None
    task: ModelTask = ModelTask.CLASSIFICATION


class TrainingJobResponse(BaseModel):
    """Schema for training job response"""
    id: str
    job_type: str = "training"  # Job type: 'training' or 'automl_training'
    dataset_id: str
    model_id: Optional[str] = None
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=100.0)
    current_iteration: int = 0
    total_iterations: int = 10
    current_accuracy: Optional[float] = None
    best_accuracy: Optional[float] = None
    current_loss: Optional[float] = None  # Current training loss
    best_loss: Optional[float] = None  # Best (lowest) loss achieved
    precision: Optional[float] = None  # Precision metric
    recall: Optional[float] = None  # Recall metric
    f1_score: Optional[float] = None  # F1-Score metric
    # Job configuration including pipeline_stages for ML
    config: Optional[dict] = None
    hyperparameters: Optional[HyperparametersConfig] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    # Human-readable elapsed time (e.g., "2h 15m")
    elapsed_time: Optional[str] = None
    # Human-readable estimated time (e.g., "45m 32s")
    estimated_remaining: Optional[str] = None

    class Config:
        from_attributes = True


class TrainingMetrics(BaseModel):
    """Training metrics"""
    accuracy: float
    loss: float
    inference_speed: float
    cpu_usage: float
    ram_peak: float
    stability: float

# ============= System Schemas =============


class SystemMetrics(BaseModel):
    """System monitoring metrics"""
    cpu_temp: float
    cpu_memory_used: float
    cpu_memory_total: float
    cpu_memory_percent: float
    gpu_temp: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_usage_percent: Optional[float] = None

# ============= Common Response Schemas =============


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    status_code: int
