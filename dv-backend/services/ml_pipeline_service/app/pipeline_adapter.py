"""
ML Pipeline Adapter
Adapts ML Pipeline to use DeepVariance PostgreSQL schema
"""
from database import DatasetDB, JobDB, ModelDB
from ml_pipeline.pipeline import run_pipeline
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MLPipelineAdapter:
    """
    Adapter that bridges ML Pipeline with DeepVariance database schema

    Maps ML Pipeline's 8-stage execution to DeepVariance Job/Model tables
    """

    def __init__(self):
        self.stage_names = [
            "Data Loading",
            "Type Conversion",
            "Missing Value Handling",
            "Data Sampling",
            "Profile Generation (Sampled)",
            "Preprocessing Insights",
            "Preprocessing Execution",
            "Profile Generation (Preprocessed)",
            "Model Recommendation",
            "Model Training"
        ]

    async def run_pipeline(
        self,
        job_id: str,
        dataset_id: str,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Run ML Pipeline using DeepVariance database for tracking

        Args:
            job_id: DeepVariance Job.id (UUID)
            dataset_id: DeepVariance Dataset.id (UUID)
            target_column: Target column for prediction

        Returns:
            Dict with metrics and model information
        """
        # 1. Fetch job and dataset from DeepVariance database
        job = JobDB.get_by_id(job_id)
        dataset = DatasetDB.get_by_id(dataset_id)

        if not job or not dataset:
            raise ValueError(f"Job {job_id} or Dataset {dataset_id} not found")

        # 2. Load dataset from file
        dataset_path = dataset.get('file_path')
        if not dataset_path or not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        df = pd.read_csv(dataset_path)

        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # 3. Update job config with target column and initialize pipeline stages
        JobDB.update(job_id, {
            'config': {
                **job.get('config', {}),
                'target_column': target_column,
                'pipeline_stages': []
            },
            'status': 'running',
            'total_iterations': 10,
            'current_iteration': 0,
            'progress': 0
        })

        # 4. Define status callback for pipeline
        def status_callback(stage_name: str, status: str):
            """
            Callback called by ML Pipeline on stage start/complete/error

            Args:
                stage_name: Name of the pipeline stage
                status: 'start', 'complete', or 'error'
            """
            job = JobDB.get_by_id(job_id)
            config = job.get('config', {})
            stages = config.get('pipeline_stages', [])

            # Find stage number from name
            try:
                stage_num = self.stage_names.index(stage_name) + 1
            except ValueError:
                stage_num = len(stages) + 1

            if status == 'start':
                # Add new stage
                stages.append({
                    'stage': stage_num,
                    'name': stage_name,
                    'status': 'running'
                })
                JobDB.update(job_id, {
                    'config': {**config, 'pipeline_stages': stages},
                    'current_iteration': stage_num,
                    'progress': ((stage_num - 1) / 10) * 100
                })

            elif status == 'complete':
                # Update last stage to completed
                if stages:
                    stages[-1]['status'] = 'completed'
                    JobDB.update(job_id, {
                        'config': {**config, 'pipeline_stages': stages},
                        'progress': (stage_num / 10) * 100
                    })

            elif status == 'error':
                # Update last stage to failed
                if stages:
                    stages[-1]['status'] = 'failed'
                    JobDB.update(job_id, {
                        'config': {**config, 'pipeline_stages': stages},
                        'status': 'failed'
                    })

        # 5. Run ML Pipeline with callback
        try:
            metrics, model = run_pipeline(
                X=df,
                target_field=target_column,
                status_callback=status_callback
            )

            # 6. Save model (pickle) to backend's models directory
            model_id = job.get('model_id')
            model_filename = f"ml_pipeline_model_{model_id}.pkl"
            # Path relative to ml_pipeline_service directory
            model_path = os.path.join(os.path.dirname(
                __file__), '../../..', 'models', model_filename)
            # Resolve to absolute path
            model_path = os.path.abspath(model_path)

            # Ensure models directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save model using joblib/pickle
            import joblib
            joblib.dump(model, model_path)

            # 7. Update Model record with results
            ModelDB.update(model_id, {
                'status': 'ready',
                # Convert to percentage
                'accuracy': metrics.get('accuracy', 0) * 100,
                'loss': metrics.get('loss'),
                'metrics': metrics,
                'model_path': model_path,
                'framework': 'sklearn'  # ML Pipeline uses sklearn models
            })

            # 8. Update Job as completed
            JobDB.update(job_id, {
                'status': 'completed',
                'progress': 100,
                'current_iteration': 10,
                'best_accuracy': metrics.get('accuracy', 0) * 100,
                'best_loss': metrics.get('loss'),
                'result': metrics
            })

            return {
                'metrics': metrics,
                'model_path': model_path,
                'job_id': job_id,
                'model_id': model_id
            }

        except Exception as e:
            # Update job as failed
            JobDB.update(job_id, {
                'status': 'failed',
                'error': str(e)
            })
            raise e
