"""
SQLAlchemy ORM Models for PostgreSQL Database
Database table definitions using SQLAlchemy ORM
"""
import uuid

from sqlalchemy import (DECIMAL, TIMESTAMP, BigInteger, Boolean,
                        CheckConstraint, Column, Date, ForeignKey, Index,
                        Integer, String, Text)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from db_config import Base

# ============= DATASETS TABLE =============


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    domain = Column(String(50), nullable=False)
    readiness = Column(String(20), nullable=False, default='draft')
    description = Column(Text)

    # Dataset structure
    structure = Column(JSONB)  # Flexible schema info

    # File information
    file_path = Column(String(500))
    file_size = Column(BigInteger)  # Size in bytes
    file_format = Column(String(50))

    # Statistics
    total_samples = Column(Integer, default=0)
    train_samples = Column(Integer)
    val_samples = Column(Integer)
    test_samples = Column(Integer)

    # Metadata
    tags = Column(ARRAY(Text))
    # Use 'meta_data' as attribute, 'metadata' as column name
    meta_data = Column('metadata', JSONB)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True),
                        server_default=func.now(), onupdate=func.now())
    last_modified = Column(Date)
    freshness = Column(Date)

    # Relationships
    models = relationship("Model", back_populates="dataset")
    training_runs = relationship("TrainingRun", back_populates="dataset")

    # Constraints
    __table_args__ = (
        CheckConstraint('domain IN (\'vision\', \'tabular\')',
                        name='valid_domain'),
        CheckConstraint(
            'readiness IN (\'draft\', \'processing\', \'ready\', \'failed\')', name='valid_readiness'),
        CheckConstraint('total_samples >= 0', name='valid_samples'),
        CheckConstraint('file_size IS NULL OR file_size > 0',
                        name='valid_file_size'),
        Index('idx_datasets_domain', 'domain'),
        Index('idx_datasets_readiness', 'readiness'),
        Index('idx_datasets_name', 'name'),
        Index('idx_datasets_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', domain='{self.domain}')>"


# ============= MODELS TABLE =============

class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Model configuration
    task = Column(String(50), nullable=False)
    framework = Column(String(50), nullable=False)
    version = Column(String(50), default='v0.1.0')

    # Status tracking
    status = Column(String(20), nullable=False, default='draft')

    # Training results
    accuracy = Column(DECIMAL(5, 2))  # 0-100%
    loss = Column(DECIMAL(10, 6))

    # Relationships
    dataset_id = Column(UUID(as_uuid=True), ForeignKey(
        'datasets.id', ondelete='SET NULL'))

    # Storage
    model_path = Column(String(500))

    # Metadata
    tags = Column(ARRAY(Text))
    hyperparameters = Column(JSONB)
    metrics = Column(JSONB)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True),
                        server_default=func.now(), onupdate=func.now())
    last_trained = Column(TIMESTAMP(timezone=True))

    # Relationships
    dataset = relationship("Dataset", back_populates="models")
    training_runs = relationship(
        "TrainingRun", back_populates="model", cascade="all, delete-orphan")
    versions = relationship(
        "ModelVersion", back_populates="model", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            'task IN (\'classification\', \'regression\', \'clustering\', \'detection\', \'unknown\')', name='valid_task'),
        CheckConstraint(
            'framework IN (\'pytorch\', \'tensorflow\', \'sklearn\')', name='valid_framework'),
        CheckConstraint(
            'status IN (\'draft\', \'queued\', \'training\', \'ready\', \'active\', \'failed\', \'stopped\')', name='valid_status'),
        CheckConstraint(
            'accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 100)', name='valid_accuracy'),
        Index('idx_models_task', 'task'),
        Index('idx_models_status', 'status'),
        Index('idx_models_framework', 'framework'),
        Index('idx_models_dataset_id', 'dataset_id'),
        Index('idx_models_name', 'name'),
        Index('idx_models_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<Model(id={self.id}, name='{self.name}', task='{self.task}', status='{self.status}')>"


# ============= TRAINING_RUNS TABLE =============

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_number = Column(Integer, autoincrement=True)

    # Relationships
    model_id = Column(UUID(as_uuid=True), ForeignKey(
        'models.id', ondelete='CASCADE'), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey(
        'datasets.id', ondelete='SET NULL'))

    # Run configuration
    config = Column(JSONB, nullable=False)

    # Status tracking
    status = Column(String(20), nullable=False, default='pending')
    progress = Column(DECIMAL(5, 2), default=0)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)

    # Results
    final_loss = Column(DECIMAL(10, 6))
    final_accuracy = Column(DECIMAL(5, 2))
    best_loss = Column(DECIMAL(10, 6))
    best_accuracy = Column(DECIMAL(5, 2))

    # Metrics over time
    epoch_metrics = Column(JSONB)  # Array of epoch metrics

    # Performance tracking
    duration_seconds = Column(Integer)
    estimated_time_remaining = Column(Integer)

    # Error tracking
    error_message = Column(Text)
    error_traceback = Column(Text)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True),
                        server_default=func.now(), onupdate=func.now())

    # Relationships
    model = relationship("Model", back_populates="training_runs")
    dataset = relationship("Dataset", back_populates="training_runs")
    logs = relationship(
        "TrainingLog", back_populates="training_run", cascade="all, delete-orphan")
    version = relationship(
        "ModelVersion", back_populates="training_run", uselist=False)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            'status IN (\'pending\', \'queued\', \'running\', \'completed\', \'failed\', \'stopped\')', name='valid_status'),
        CheckConstraint('progress >= 0 AND progress <= 100',
                        name='valid_progress'),
        CheckConstraint(
            'current_epoch >= 0 AND (total_epochs IS NULL OR current_epoch <= total_epochs)', name='valid_epochs'),
        Index('idx_training_runs_model_id', 'model_id'),
        Index('idx_training_runs_status', 'status'),
        Index('idx_training_runs_created_at', 'created_at'),
        Index('idx_training_runs_model_status', 'model_id', 'status'),
    )

    def __repr__(self):
        return f"<TrainingRun(id={self.id}, model_id={self.model_id}, status='{self.status}')>"


# ============= TRAINING_LOGS TABLE =============

class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_run_id = Column(UUID(as_uuid=True), ForeignKey(
        'training_runs.id', ondelete='CASCADE'), nullable=False)

    # Log details
    log_level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)

    # Context
    epoch = Column(Integer)
    batch = Column(Integer)

    # Timestamp
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    training_run = relationship("TrainingRun", back_populates="logs")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            'log_level IN (\'DEBUG\', \'INFO\', \'WARNING\', \'ERROR\', \'CRITICAL\')', name='valid_log_level'),
        Index('idx_training_logs_run_id', 'training_run_id', 'created_at'),
        Index('idx_training_logs_level', 'log_level'),
    )

    def __repr__(self):
        return f"<TrainingLog(id={self.id}, level='{self.log_level}', message='{self.message[:50]}...')>"


# ============= MODEL_VERSIONS TABLE =============

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey(
        'models.id', ondelete='CASCADE'), nullable=False)
    training_run_id = Column(UUID(as_uuid=True), ForeignKey(
        'training_runs.id', ondelete='SET NULL'))

    version = Column(String(50), nullable=False)
    is_current = Column(Boolean, default=False)

    # Version metrics
    accuracy = Column(DECIMAL(5, 2))
    loss = Column(DECIMAL(10, 6))

    # Storage
    model_path = Column(String(500))
    checkpoint_path = Column(String(500))

    # Metadata
    notes = Column(Text)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    model = relationship("Model", back_populates="versions")
    training_run = relationship("TrainingRun", back_populates="version")

    # Constraints
    __table_args__ = (
        Index('idx_model_versions_model_id', 'model_id'),
        Index('idx_model_versions_current', 'model_id', 'is_current'),
    )

    def __repr__(self):
        return f"<ModelVersion(id={self.id}, model_id={self.model_id}, version='{self.version}')>"


# ============= JOBS TABLE =============

class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default='pending')

    # Progress
    progress = Column(DECIMAL(5, 2), default=0)
    current_iteration = Column(Integer, default=0)
    total_iterations = Column(Integer, default=10)

    # Training metrics
    current_accuracy = Column(DECIMAL(10, 6))
    best_accuracy = Column(DECIMAL(10, 6))
    current_loss = Column(DECIMAL(10, 6))
    best_loss = Column(DECIMAL(10, 6))
    precision = Column(DECIMAL(10, 6))
    recall = Column(DECIMAL(10, 6))
    f1_score = Column(DECIMAL(10, 6))

    # Relationships
    model_id = Column(UUID(as_uuid=True), ForeignKey(
        'models.id', ondelete='CASCADE'))
    dataset_id = Column(UUID(as_uuid=True), ForeignKey(
        'datasets.id', ondelete='CASCADE'))

    # Job data
    config = Column(JSONB)
    result = Column(JSONB)
    error = Column(Text)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))

    # Constraints
    __table_args__ = (
        CheckConstraint(
            'status IN (\'pending\', \'running\', \'completed\', \'failed\', \'stopped\')', name='valid_status'),
        CheckConstraint('progress >= 0 AND progress <= 100',
                        name='valid_progress'),
        Index('idx_jobs_status', 'status'),
        Index('idx_jobs_type', 'job_type'),
        Index('idx_jobs_model_id', 'model_id'),
    )

    def __repr__(self):
        return f"<Job(id={self.id}, type='{self.job_type}', status='{self.status}')>"
