# DeepVariance Backend - Directory Structure

## 📁 Core Application Structure

```
dv-backend/
├── main.py                      # FastAPI application entry point
├── config.py                    # Configuration management
├── models.py                    # Pydantic schemas
├── database.py                  # Database operations layer
├── db_config.py                 # Database connection config
├── db_models.py                 # SQLAlchemy ORM models
├── schema.sql                   # PostgreSQL schema definition
└── requirements.txt             # Python dependencies
```

## 🔌 API Layer

```
routers/
├── __init__.py
├── datasets.py                  # Dataset CRUD endpoints
├── models.py                    # Model management endpoints
├── jobs.py                      # Training job endpoints
└── system.py                    # System health/info endpoints
```

## 🔧 Core Services

```
├── job_worker.py                # DL training worker pool
├── ml_training_worker.py        # AutoML training worker pool
├── job_logger.py                # Job logging service
├── training_runner.py           # Legacy training orchestration
└── task_inference.py            # Task type detection
```

## 🛠️ Utilities

```
├── validators.py                # Dataset validation
├── file_utils.py                # File operations utilities
├── metrics_utils.py             # Metrics calculation
└── hardware_utils.py            # Hardware detection
```

## 📊 Data Storage (gitignored)

```
data/                            # Dataset storage
├── .gitkeep                     # Preserves directory in git
├── {dataset-id}/                # Individual datasets
│   ├── train/
│   └── test/
└── *.csv                        # Uploaded CSV files

models/                          # Trained model storage
├── .gitkeep
├── generated_model_*.py         # LLM-generated CNN code
├── best_model_*.py              # Best iteration models
└── ml_pipeline_model_*.pkl      # AutoML models

results/                         # Training results
├── .gitkeep
└── {job-id}/                    # Per-job results
    ├── metrics.json
    └── checkpoints/

autogluon_models_*/              # AutoGluon training artifacts
logs/                            # Application logs
db/                              # PostgreSQL data (if local)
```

## 🧪 Testing

```
scripts/
└── inspect_jobs.py              # Job debugging utility

test_data/                       # Test datasets

├── test_simple.py               # Basic functionality tests
├── test_training_pipeline.py   # Training pipeline tests
├── test_autogluon_pipeline.py  # AutoGluon tests
└── setup_test_dataset.py       # Test data setup
```

## 🌐 Microservices

```
services/
├── job_resumer.py               # Background job resumer
└── ml_pipeline_service/         # ML Pipeline service (port 8001)
    ├── start.sh
    ├── app/
    │   ├── main.py
    │   ├── routes.py
    │   ├── models.py
    │   └── database.py
    └── ml_pipeline/
        ├── pipeline.py
        └── llm_agents.py
```

## 🗂️ Supporting

```
migrations/                      # Database migrations
training_pipeline/               # Modular training system
├── core/
│   ├── llm_training.py
│   └── trainer.py
└── utils/
```

## 🔐 Configuration Files

```
.env                             # Environment variables (gitignored)
.env.example                     # Example configuration
.gitignore                       # Git ignore patterns
schema.sql                       # Database schema
DeepVariance_API.postman_collection.json  # API testing
```

## 📋 Best Practices

### Storage Guidelines

1. **Never commit** `data/`, `models/`, or `results/` directories
2. **Keep** `.gitkeep` files to preserve directory structure
3. **Use UUID-based** directory names for isolation
4. **Clean up** old training artifacts periodically

### Code Organization

1. **Routers**: Handle HTTP requests, validation only
2. **Database layer**: All DB operations through `database.py`
3. **Workers**: Process-based isolation for training jobs
4. **Services**: Background tasks and microservices

### Data Flow

```
API Request → Router → Database Layer → Worker Pool → Training Process
                ↓                           ↓              ↓
            Validation                Job Queue      Job Logger
                                                          ↓
                                                     Model Storage
```

## 🧹 Cleanup Commands

```bash
# Remove all generated models (be careful!)
rm -rf autogluon_models_* models/*.pkl models/*.py results/*

# Remove test data
rm -rf data/test_*

# Remove logs
rm -rf logs/*.log

# Reset to clean state (preserves structure)
find data models results -type f ! -name '.gitkeep' -delete
```

## 📈 Monitoring Directories

Watch these directories for growth:

- `autogluon_models_*/` - Can grow to GBs per training run
- `models/*.pkl` - Each AutoML model can be 50-500MB
- `results/` - Accumulates training metadata
- `logs/` - Log files can accumulate quickly

Regular cleanup recommended!
