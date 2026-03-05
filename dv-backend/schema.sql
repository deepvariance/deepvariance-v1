-- DeepVariance PostgreSQL Database Schema
-- PostgreSQL 15+ recommended

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============= DATASETS TABLE =============
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(50) NOT NULL CHECK (domain IN ('vision', 'tabular')),
    readiness VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (readiness IN ('draft', 'processing', 'ready', 'failed')),
    description TEXT,

    -- Dataset structure
    structure JSONB, -- Stores flexible schema info (columns for tabular, classes for vision)

    -- File information
    file_path VARCHAR(500),
    file_size BIGINT, -- Size in bytes
    file_format VARCHAR(50), -- csv, zip, etc.

    -- Statistics
    total_samples INTEGER DEFAULT 0,
    train_samples INTEGER,
    val_samples INTEGER,
    test_samples INTEGER,

    -- Metadata
    tags TEXT[], -- Array of tags
    metadata JSONB, -- Flexible metadata storage

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_modified DATE,
    freshness DATE,

    -- Constraints
    CONSTRAINT valid_samples CHECK (total_samples >= 0),
    CONSTRAINT valid_file_size CHECK (file_size IS NULL OR file_size > 0)
);

-- Index for common queries
CREATE INDEX idx_datasets_domain ON datasets(domain);
CREATE INDEX idx_datasets_readiness ON datasets(readiness);
CREATE INDEX idx_datasets_name ON datasets(name);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX idx_datasets_tags ON datasets USING GIN(tags); -- GIN index for array search

-- ============= MODELS TABLE =============
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Model configuration
    task VARCHAR(50) NOT NULL CHECK (task IN ('classification', 'regression', 'clustering', 'detection', 'unknown')),
    framework VARCHAR(50) NOT NULL CHECK (framework IN ('pytorch', 'tensorflow', 'sklearn')),
    version VARCHAR(50) DEFAULT 'v0.1.0',

    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'queued', 'training', 'ready', 'active', 'failed')),

    -- Training results
    accuracy DECIMAL(5, 2), -- 0-100%
    loss DECIMAL(10, 6),

    -- Relationships
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,

    -- Storage
    model_path VARCHAR(500),

    -- Metadata
    tags TEXT[],
    hyperparameters JSONB, -- Store training hyperparameters
    metrics JSONB, -- Store additional metrics (precision, recall, f1, etc.)

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_trained TIMESTAMP WITH TIME ZONE,

    -- Constraints
    CONSTRAINT valid_accuracy CHECK (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 100))
);

-- Indexes for models
CREATE INDEX idx_models_task ON models(task);
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_framework ON models(framework);
CREATE INDEX idx_models_dataset_id ON models(dataset_id);
CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_created_at ON models(created_at DESC);
CREATE INDEX idx_models_tags ON models USING GIN(tags);

-- ============= TRAINING_RUNS TABLE =============
-- Tracks individual training runs for each model
CREATE TABLE training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_number SERIAL, -- Auto-incrementing run number per model

    -- Relationships
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,

    -- Run configuration
    config JSONB NOT NULL, -- Training configuration (epochs, batch_size, learning_rate, etc.)

    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed', 'stopped')),
    progress DECIMAL(5, 2) DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER,

    -- Results
    final_loss DECIMAL(10, 6),
    final_accuracy DECIMAL(5, 2),
    best_loss DECIMAL(10, 6),
    best_accuracy DECIMAL(5, 2),

    -- Metrics over time (stored as JSONB array)
    epoch_metrics JSONB, -- [{epoch: 1, loss: 0.5, accuracy: 80}, ...]

    -- Performance tracking
    duration_seconds INTEGER, -- Total duration in seconds
    estimated_time_remaining INTEGER, -- Estimated seconds remaining

    -- Error tracking
    error_message TEXT,
    error_traceback TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_progress CHECK (progress >= 0 AND progress <= 100),
    CONSTRAINT valid_epochs CHECK (current_epoch >= 0 AND (total_epochs IS NULL OR current_epoch <= total_epochs))
);

-- Indexes for training runs
CREATE INDEX idx_training_runs_model_id ON training_runs(model_id);
CREATE INDEX idx_training_runs_status ON training_runs(status);
CREATE INDEX idx_training_runs_created_at ON training_runs(created_at DESC);
CREATE INDEX idx_training_runs_model_status ON training_runs(model_id, status); -- Composite index

-- ============= TRAINING_LOGS TABLE =============
-- Stores detailed logs for each training run
CREATE TABLE training_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_run_id UUID NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,

    -- Log details
    log_level VARCHAR(20) NOT NULL CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,

    -- Context
    epoch INTEGER,
    batch INTEGER,

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient log retrieval
CREATE INDEX idx_training_logs_run_id ON training_logs(training_run_id, created_at);
CREATE INDEX idx_training_logs_level ON training_logs(log_level);

-- ============= MODEL_VERSIONS TABLE =============
-- Track different versions of the same model
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    training_run_id UUID REFERENCES training_runs(id) ON DELETE SET NULL,

    version VARCHAR(50) NOT NULL,
    is_current BOOLEAN DEFAULT FALSE,

    -- Version metrics
    accuracy DECIMAL(5, 2),
    loss DECIMAL(10, 6),

    -- Storage
    model_path VARCHAR(500),
    checkpoint_path VARCHAR(500),

    -- Metadata
    notes TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_model_version UNIQUE(model_id, version)
);

-- Indexes
CREATE INDEX idx_model_versions_model_id ON model_versions(model_id);
CREATE INDEX idx_model_versions_current ON model_versions(model_id, is_current);

-- ============= JOBS TABLE (for backward compatibility) =============
-- Can be used for general background jobs
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL, -- 'training', 'evaluation', 'deployment', etc.
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    -- Progress
    progress DECIMAL(5, 2) DEFAULT 0,

    -- Relationships (flexible)
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,

    -- Job data
    config JSONB,
    result JSONB,
    error TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_progress CHECK (progress >= 0 AND progress <= 100)
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type ON jobs(job_type);
CREATE INDEX idx_jobs_model_id ON jobs(model_id);

-- ============= UPDATE TRIGGERS =============
-- Automatically update updated_at timestamp

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to tables
CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_runs_updated_at BEFORE UPDATE ON training_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============= VIEWS FOR COMMON QUERIES =============

-- Active training runs with model and dataset info
CREATE VIEW active_training_runs AS
SELECT
    tr.id,
    tr.run_number,
    tr.status,
    tr.progress,
    tr.current_epoch,
    tr.total_epochs,
    tr.started_at,
    tr.estimated_time_remaining,
    m.id as model_id,
    m.name as model_name,
    m.task,
    m.framework,
    d.id as dataset_id,
    d.name as dataset_name,
    d.total_samples
FROM training_runs tr
JOIN models m ON tr.model_id = m.id
LEFT JOIN datasets d ON tr.dataset_id = d.id
WHERE tr.status IN ('running', 'queued');

-- Model statistics
CREATE VIEW model_statistics AS
SELECT
    m.id,
    m.name,
    m.task,
    m.framework,
    m.status,
    COUNT(tr.id) as total_runs,
    COUNT(CASE WHEN tr.status = 'completed' THEN 1 END) as successful_runs,
    COUNT(CASE WHEN tr.status = 'failed' THEN 1 END) as failed_runs,
    MAX(tr.final_accuracy) as best_accuracy,
    MIN(tr.final_loss) as best_loss
FROM models m
LEFT JOIN training_runs tr ON m.id = tr.model_id
GROUP BY m.id, m.name, m.task, m.framework, m.status;

-- ============= SAMPLE DATA (for testing) =============
-- Uncomment to insert sample data

-- INSERT INTO datasets (name, domain, readiness, description, total_samples, structure)
-- VALUES
--     ('MNIST Handwritten Digits', 'vision', 'ready', 'Classic handwritten digit recognition dataset', 70000, '{"classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], "image_size": "28x28"}'),
--     ('Iris Flower Dataset', 'tabular', 'ready', 'Classic classification dataset with iris flower measurements', 150, '{"columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]}');
