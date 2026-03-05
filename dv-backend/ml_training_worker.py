"""
ML Pipeline Training Worker
Runs AutoML training jobs asynchronously in separate processes (same pattern as DL training)
"""

import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [ML-Worker] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_ml_training_job_worker(job_data: Dict):
    """
    Worker function that runs ML Pipeline training in a separate process
    This mirrors the DL training worker pattern
    """
    import os
    import sys
    from pathlib import Path

    # Add ML Pipeline to Python path
    ml_pipeline_path = os.path.join(
        os.getcwd(), 'services', 'ml_pipeline_service')
    sys.path.insert(0, ml_pipeline_path)

    # Import ML pipeline modules
    import joblib
    import pandas as pd
    from ml_pipeline.pipeline import run_pipeline

    from database import JobDB, ModelDB, TrainingRunDB
    from job_logger import JobLogger

    job_id = job_data['job_id']
    dataset_path = job_data['dataset_path']
    target_column = job_data['target_column']
    model_id = job_data['model_id']
    dataset_id = job_data.get('dataset_id')

    # Create job logger for real-time log streaming
    job_logger = JobLogger(job_id)
    job_logger.info(f"Starting ML job {job_id}")

    start_time = time.time()

    try:
        # Create training run record
        training_run = TrainingRunDB.create({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "config": {"target_column": target_column, "pipeline_type": "automl"},
            "status": "pending",
            "total_epochs": 8,  # 8 stages for ML Pipeline
        })
        training_run_id = training_run.get("id")
        job_logger.info(f"Created training run {training_run_id}")

        # Pre-initialize all 8 pipeline stages with 'pending' status
        # This ensures stages are visible from the start and preserves state on retries
        config = job_data.get('config', {})
        existing_stages = config.get('pipeline_stages', [])
        if not existing_stages:
            # First run - initialize all stages
            stage_names = [
                "Type Conversion",
                "Data Sampling",
                "Profile Generation (Sampled)",
                "Preprocessing Insights Generation",
                "Preprocessing Code Execution",
                "Profile Generation (Preprocessed)",
                "Model Recommendation",
                "Model Training"
            ]
            pipeline_stages = [
                {
                    'stage': i + 1,
                    'name': name,
                    'status': 'pending'
                }
                for i, name in enumerate(stage_names)
            ]
        else:
            # Retry - preserve existing stages and reset incomplete ones to pending
            pipeline_stages = []
            for stage in existing_stages:
                if stage['status'] in ['running', 'failed']:
                    # Reset incomplete/failed stages to pending
                    pipeline_stages.append({
                        'stage': stage['stage'],
                        'name': stage['name'],
                        'status': 'pending'
                    })
                else:
                    # Keep completed stages as-is
                    pipeline_stages.append(stage)

        # Update job status to running with started_at timestamp
        JobDB.update(job_id, {
            'status': 'running',
            'progress': 0.0,
            'started_at': datetime.now().isoformat(),
            'config': {
                **job_data.get('config', {}),
                'pipeline_stages': pipeline_stages,
                'training_run_id': training_run_id
            }
        })

        # Load dataset
        from file_utils import get_file_format, read_dataframe

        job_logger.info(f"Loading dataset from {dataset_path}")
        file_format = get_file_format(dataset_path)
        job_logger.info(f"Detected file format: {file_format}")

        df = read_dataframe(dataset_path)
        job_logger.info(
            f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Validate target column
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # Detect task type from target column
        job_logger.info(
            f"Detecting task type for target column: {target_column}")
        target_data = df[target_column]

        # Check if target is numeric and continuous (regression) or categorical (classification)
        if pd.api.types.is_numeric_dtype(target_data):
            # Count unique values
            unique_count = target_data.nunique()
            total_count = len(target_data)
            unique_ratio = unique_count / total_count if total_count > 0 else 0

            # If more than 10 unique values OR more than 5% unique values, it's likely regression
            if unique_count > 10 or unique_ratio > 0.05:
                task_type = 'regression'
                job_logger.info(
                    f"Detected REGRESSION task ({unique_count} unique values, {unique_ratio:.2%} unique ratio)")
            else:
                task_type = 'classification'
                job_logger.info(
                    f"Detected CLASSIFICATION task ({unique_count} unique classes)")
        else:
            # Non-numeric target is classification
            task_type = 'classification'
            unique_count = target_data.nunique()
            job_logger.info(
                f"Detected CLASSIFICATION task (non-numeric target, {unique_count} classes)")

        # Update model with detected task type
        ModelDB.update(model_id, {'task': task_type})
        job_logger.info(f"Updated model task type to: {task_type}")

        # Define 8 stage names for progress tracking (matches pipeline.py exactly)
        stage_names = [
            "Type Conversion",
            "Data Sampling",
            "Profile Generation (Sampled)",
            "Preprocessing Insights Generation",
            "Preprocessing Code Execution",
            "Profile Generation (Preprocessed)",
            "Model Recommendation",  # LLM selects which AutoGluon models to use
            "Model Training"  # AutoGluon trains the selected models
        ]

        # Status callback to update database during pipeline execution
        def status_callback(stage_name: str, status: str):
            """
            Callback invoked by ML Pipeline on stage start/complete/error
            Updates Job and Model records in database
            """
            try:
                job = JobDB.get_by_id(job_id)
                if not job:
                    return

                config = job.get('config') or {}
                stages = config.get('pipeline_stages', [])

                # Find stage number from name
                try:
                    stage_num = stage_names.index(stage_name) + 1
                except ValueError:
                    stage_num = len(stages) + 1

                if status == 'start':
                    # Update existing stage to running (stages are pre-initialized)
                    stage_index = stage_num - 1
                    if stage_index < len(stages):
                        stages[stage_index].update({
                            'status': 'running',
                            'started_at': datetime.now().isoformat()
                        })
                    else:
                        # Fallback: if stage doesn't exist (shouldn't happen), add it
                        stages.append({
                            'stage': stage_num,
                            'name': stage_name,
                            'status': 'running',
                            'started_at': datetime.now().isoformat()
                        })

                    # Progress when starting a stage: (stage_num - 1) / 8 * 100
                    # Stage 1 starts at 0%, Stage 2 starts at 12.5%, etc.
                    progress = ((stage_num - 1) / 8) * 100

                    # Current iteration shows completed stages (stage_num - 1)
                    current_iteration = stage_num - 1

                    JobDB.update(job_id, {
                        'config': {**config, 'pipeline_stages': stages},
                        'current_iteration': current_iteration,
                        'progress': progress,
                        'status': 'running'
                    })

                    # Update model status
                    ModelDB.update(model_id, {'status': 'training'})

                    job_logger.info(
                        f"Stage {stage_num}/8 started: {stage_name} ({progress:.1f}%)")

                elif status == 'complete':
                    # Update stage to completed
                    stage_index = stage_num - 1
                    if stage_index < len(stages):
                        completed_stage_num = stages[stage_index]['stage']
                        stages[stage_index].update({
                            'status': 'completed',
                            'completed_at': datetime.now().isoformat()
                        })
                    else:
                        # Fallback: shouldn't happen with pre-initialized stages
                        completed_stage_num = stage_num

                    # Progress when completing a stage: stage_num / 8 * 100
                    # Stage 1 completes at 12.5%, Stage 2 completes at 25%, etc.
                    progress = (completed_stage_num / 8) * 100

                    # Current iteration now shows number of COMPLETED stages
                    current_iteration = completed_stage_num

                    # Calculate elapsed time
                    current_elapsed = int(time.time() - start_time)
                    hours = current_elapsed // 3600
                    minutes = (current_elapsed % 3600) // 60
                    seconds = current_elapsed % 60
                    if hours > 0:
                        elapsed_str = f"{hours}h {minutes}m {seconds}s"
                    elif minutes > 0:
                        elapsed_str = f"{minutes}m {seconds}s"
                    else:
                        elapsed_str = f"{seconds}s"

                    # Estimate remaining time
                    if completed_stage_num > 0:
                        avg_time_per_stage = current_elapsed / completed_stage_num
                        remaining_stages = 8 - completed_stage_num
                        remaining_seconds = int(
                            avg_time_per_stage * remaining_stages)
                        rem_hours = remaining_seconds // 3600
                        rem_minutes = (remaining_seconds % 3600) // 60
                        rem_seconds = remaining_seconds % 60
                        if rem_hours > 0:
                            remaining_str = f"{rem_hours}h {rem_minutes}m"
                        elif rem_minutes > 0:
                            remaining_str = f"{rem_minutes}m {rem_seconds}s"
                        else:
                            remaining_str = f"{rem_seconds}s"
                    else:
                        remaining_str = "Calculating..."

                    JobDB.update(job_id, {
                        'config': {
                            **config,
                            'pipeline_stages': stages,
                            'elapsed_time': elapsed_str,
                            'estimated_remaining': remaining_str
                        },
                        'current_iteration': current_iteration,
                        'progress': progress
                    })

                    job_logger.info(
                        f"Stage {completed_stage_num}/8 completed: {stage_name} ({progress:.1f}%)")

                elif status == 'error':
                    # Update stage to failed
                    stage_index = stage_num - 1
                    if stage_index < len(stages):
                        failed_stage_num = stages[stage_index]['stage']
                        stages[stage_index].update({
                            'status': 'failed',
                            'completed_at': datetime.now().isoformat()
                        })
                    else:
                        failed_stage_num = stage_num

                    # Current iteration shows completed stages before failure
                    current_iteration = failed_stage_num - 1

                    JobDB.update(job_id, {
                        'config': {**config, 'pipeline_stages': stages},
                        'current_iteration': current_iteration,
                        'status': 'failed'
                    })

                    ModelDB.update(model_id, {'status': 'failed'})

            except Exception as e:
                logger.error(f"Error in status callback: {e}")

        # Run ML Pipeline with status callback and default 25% sampling
        job_logger.info(
            f"Starting ML Pipeline with target column: {target_column}")
        job_logger.info(f"Using default 25% sampling for optimal performance")
        metrics, trained_model = run_pipeline(
            X=df,
            target_field=target_column,
            status_callback=status_callback,
            sample_percentage=25.0  # Default 25% sampling like command-line test
        )

        job_logger.info(f"ML Pipeline completed successfully")
        job_logger.info(f"📊 Extracted Metrics:")
        if isinstance(metrics, dict):
            if 'accuracy' in metrics:
                job_logger.info(f"   • Accuracy: {metrics['accuracy']:.4f}")
            if 'f1_macro' in metrics:
                job_logger.info(f"   • F1 Macro: {metrics['f1_macro']:.4f}")
            if 'roc_auc' in metrics:
                job_logger.info(f"   • ROC-AUC: {metrics['roc_auc']:.4f}")
            if 'r2' in metrics:
                job_logger.info(f"   • R²: {metrics['r2']:.4f}")
            if 'mae' in metrics:
                job_logger.info(f"   • MAE: {metrics['mae']:.4f}")
            if 'rmse' in metrics:
                job_logger.info(f"   • RMSE: {metrics['rmse']:.4f}")
            if 'best_model' in metrics:
                job_logger.info(f"   • Best Model: {metrics['best_model']}")
            if 'top_models' in metrics:
                job_logger.info(
                    f"   • Top 3 Models: {[m.get('model') for m in metrics['top_models']]}")
        else:
            job_logger.info(f"   Raw metrics: {metrics}")

        # Save model using joblib
        model_filename = f"ml_pipeline_model_{model_id}.pkl"
        model_path = os.path.join(os.getcwd(), 'models', model_filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        job_logger.info(f"Saving model to {model_path}")
        joblib.dump(trained_model, model_path)

        # Extract metrics (handle various formats from ML Pipeline)
        accuracy = None
        precision = None
        recall = None
        f1_score = None
        roc_auc = None

        if isinstance(metrics, dict):
            # Try different accuracy keys
            for key in ['accuracy', 'test_accuracy', 'Accuracy', 'acc']:
                if key in metrics:
                    acc_value = metrics[key]
                    if isinstance(acc_value, (int, float)):
                        accuracy = acc_value if acc_value > 1 else acc_value * 100
                    break

            # Extract F1 score (ML Pipeline returns 'f1_macro')
            for key in ['f1_macro', 'f1_score', 'f1-score', 'f1', 'F1']:
                if key in metrics:
                    f1_value = metrics[key]
                    if isinstance(f1_value, (int, float)):
                        f1_score = f1_value if f1_value > 1 else f1_value * 100
                    break

            # Extract ROC-AUC (ML Pipeline returns 'roc_auc')
            if 'roc_auc' in metrics:
                roc_value = metrics['roc_auc']
                if isinstance(roc_value, (int, float)):
                    roc_auc = roc_value if roc_value > 1 else roc_value * 100

            # Extract precision and recall if available
            for key in ['precision', 'Precision']:
                if key in metrics:
                    prec_value = metrics[key]
                    if isinstance(prec_value, (int, float)):
                        precision = prec_value if prec_value > 1 else prec_value * 100
                    break

            for key in ['recall', 'Recall']:
                if key in metrics:
                    rec_value = metrics[key]
                    if isinstance(rec_value, (int, float)):
                        recall = rec_value if rec_value > 1 else rec_value * 100
                    break

        # Normalize accuracy to 0-1 range for database storage
        accuracy_normalized = accuracy / 100 if accuracy and accuracy > 1 else accuracy
        precision_normalized = precision / \
            100 if precision and precision > 1 else precision
        recall_normalized = recall / 100 if recall and recall > 1 else recall
        f1_normalized = f1_score / 100 if f1_score and f1_score > 1 else f1_score

        # Log extracted and normalized metrics for database
        job_logger.info(f"📈 Normalized Metrics (for storage):")
        if accuracy_normalized is not None:
            job_logger.info(
                f"   • Accuracy: {accuracy_normalized:.4f} (0-1 range)")
        if precision_normalized is not None:
            job_logger.info(
                f"   • Precision: {precision_normalized:.4f} (0-1 range)")
        if recall_normalized is not None:
            job_logger.info(
                f"   • Recall: {recall_normalized:.4f} (0-1 range)")
        if f1_normalized is not None:
            job_logger.info(f"   • F1 Score: {f1_normalized:.4f} (0-1 range)")
        if roc_auc is not None:
            roc_normalized = roc_auc / 100 if roc_auc > 1 else roc_auc
            job_logger.info(f"   • ROC-AUC: {roc_normalized:.4f} (0-1 range)")

        # Clean metrics dict - replace NaN with None for JSON compatibility
        def clean_nan_values(obj):
            """Recursively replace NaN values with None for JSON serialization"""
            import math
            if isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            elif isinstance(obj, float) and math.isinf(obj):
                return None
            else:
                return obj

        cleaned_metrics = clean_nan_values(metrics) if isinstance(
            metrics, dict) else {'raw': str(metrics)}

        # Detect task type from metrics
        task_type = 'classification'  # default
        if isinstance(cleaned_metrics, dict):
            # If metrics contain regression-specific fields, it's regression
            if 'r2' in cleaned_metrics or 'rmse' in cleaned_metrics or 'mae' in cleaned_metrics:
                task_type = 'regression'
            # If metrics contain classification-specific fields, it's classification
            elif 'f1_score' in cleaned_metrics or 'precision' in cleaned_metrics or 'confusion_matrix' in cleaned_metrics:
                task_type = 'classification'

        # Update Model record with results
        ModelDB.update(model_id, {
            'status': 'ready',
            'task': task_type,  # Update with detected task type
            'accuracy': accuracy_normalized,
            'metrics': cleaned_metrics,
            'model_path': model_path,
            'framework': 'sklearn'
        })

        # Calculate elapsed time
        elapsed_seconds = int(time.time() - start_time)
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        if hours > 0:
            elapsed_time = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            elapsed_time = f"{minutes}m {seconds}s"
        else:
            elapsed_time = f"{seconds}s"

        # Extract loss if available
        loss = None
        if isinstance(metrics, dict):
            for key in ['loss', 'Loss', 'test_loss']:
                if key in metrics:
                    loss = metrics[key]
                    break

        # Get current job config to preserve pipeline_stages
        current_job = JobDB.get_by_id(job_id)
        if not current_job:
            raise Exception(f"Job {job_id} not found in database")

        current_config = current_job.get('config') or {}
        stages = current_config.get('pipeline_stages', [])

        # Calculate final progress
        completed_stages_count = len(
            [s for s in stages if s.get('status') == 'completed']
        )

        # Update Job as completed with all metrics
        job_update = {
            'status': 'completed',
            'progress': 100,
            'current_iteration': 8,
            'current_accuracy': accuracy_normalized,
            'best_accuracy': accuracy_normalized,
            'current_loss': loss,
            'best_loss': loss,
            'completed_at': datetime.now().isoformat(),
            'config': {
                **current_config,
                'elapsed_time': elapsed_time,
                'estimated_remaining': '0s'
            }
        }

        # Add optional metrics if available
        if precision_normalized is not None:
            job_update['precision'] = precision_normalized
        if recall_normalized is not None:
            job_update['recall'] = recall_normalized
        if f1_normalized is not None:
            job_update['f1_score'] = f1_normalized

        # Add additional ML Pipeline metrics
        if isinstance(cleaned_metrics, dict):
            if 'best_model' in cleaned_metrics:
                job_update['config']['best_model'] = cleaned_metrics['best_model']
            if 'roc_auc' in cleaned_metrics:
                job_update['config']['roc_auc'] = cleaned_metrics['roc_auc']
            if 'top_models' in cleaned_metrics:
                job_update['config']['top_models'] = cleaned_metrics['top_models']

        JobDB.update(job_id, job_update)

        # Update training run as completed with detailed metrics
        # Include pipeline_stages in config so historical runs can display them
        training_run_update = {
            'status': 'completed',
            'current_epoch': 8,  # All 8 stages completed
            'duration_seconds': elapsed_seconds,
            'completed_at': datetime.now(),
            'final_accuracy': accuracy_normalized,
            'final_loss': loss,
            'best_accuracy': accuracy_normalized,
            'best_loss': loss,
            'config': {
                **current_config,  # Includes pipeline_stages
                'target_column': target_column,
                'pipeline_type': 'automl'
            }
        }

        TrainingRunDB.update(training_run_id, training_run_update)

        job_logger.info(
            f"Worker process {os.getpid()} completed ML job {job_id}")
        job_logger.close()

    except Exception as e:
        job_logger.error(
            f"Worker process {os.getpid()} failed ML job {job_id}: {e}")
        import traceback
        error_trace = traceback.format_exc()
        job_logger.error(error_trace)
        job_logger.close()

        # Get current config and stages
        current_job = JobDB.get_by_id(job_id)
        if not current_job:
            return

        config = current_job.get('config') or {}
        stages = config.get('pipeline_stages', [])

        # Mark last running stage as failed
        failed_stage_num = None
        if stages:
            last_stage = stages[-1]
            if last_stage.get('status') == 'running':
                failed_stage_num = last_stage['stage']
                last_stage['status'] = 'failed'
                last_stage['completed_at'] = datetime.now().isoformat()
                current_iteration = failed_stage_num - 1
            else:
                current_iteration = current_job.get('current_iteration', 0)
        else:
            current_iteration = current_job.get('current_iteration', 0)

        # Update job and model as failed
        JobDB.update(job_id, {
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat(),
            'current_iteration': current_iteration,
            'config': {**config, 'pipeline_stages': stages}
        })

        ModelDB.update(model_id, {
            'status': 'failed'
        })

        # Update training run as failed
        # Get current job state to preserve pipeline_stages
        failed_job = JobDB.get_by_id(job_id)
        failed_config = failed_job.get('config', {}) if failed_job else {}

        # Count how many stages completed before failure
        pipeline_stages = failed_config.get('pipeline_stages', [])
        completed_stages = len(
            [s for s in pipeline_stages if s.get('status') == 'completed'])

        TrainingRunDB.update(training_run_id, {
            'status': 'failed',
            'current_epoch': completed_stages,  # Number of stages that completed
            'completed_at': datetime.now(),
            'error_message': str(e),
            'config': {
                **failed_config,  # Includes pipeline_stages with current state
                'target_column': target_column,
                'pipeline_type': 'automl'
            }
        })


class MLTrainingWorkerPool:
    """
    Manages worker processes for ML Pipeline training jobs
    Same pattern as DL JobWorkerPool
    """

    def __init__(self, max_workers: int = 1):
        """
        Initialize the ML worker pool

        Args:
            max_workers: Maximum number of parallel ML training jobs (default 1 due to LLM API limits)
        """
        self.max_workers = max_workers
        # job_id -> {'process': Process, 'started_at': timestamp}
        self.active_jobs: Dict[str, Dict] = {}
        logger.info(f"Initialized ML worker pool with {max_workers} worker(s)")

    def submit_job(self, job_data: Dict) -> bool:
        """
        Submit an ML training job to the worker pool

        Args:
            job_data: Dictionary containing job information

        Returns:
            True if job was submitted successfully
        """
        job_id = job_data['job_id']

        # Check if job is already running
        if job_id in self.active_jobs:
            logger.warning(f"ML job {job_id} is already running")
            return False

        # Create a dedicated process for this job
        process = multiprocessing.Process(
            target=run_ml_training_job_worker,
            args=(job_data,),
            name=f"MLTrainingJob-{job_id}"
        )

        logger.info(f"Starting ML job {job_id} in process {process.name}")
        process.start()

        self.active_jobs[job_id] = {
            'process': process,
            'started_at': time.time()
        }

        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running ML training job

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled successfully
        """
        if job_id not in self.active_jobs:
            logger.warning(f"ML job {job_id} not found in active jobs")
            return False

        job_info = self.active_jobs[job_id]
        process = job_info['process']

        if process.is_alive():
            logger.info(f"Terminating ML job {job_id} (PID: {process.pid})")
            process.terminate()
            process.join(timeout=5)

            if process.is_alive():
                logger.warning(f"Force killing ML job {job_id}")
                process.kill()
                process.join()

            logger.info(f"ML job {job_id} terminated")

        del self.active_jobs[job_id]
        return True

    def cleanup_completed_jobs(self):
        """Remove completed jobs from active jobs tracking"""
        completed = [
            job_id for job_id, job_info in self.active_jobs.items()
            if not job_info['process'].is_alive()
        ]

        for job_id in completed:
            process = self.active_jobs[job_id]['process']
            if process.exitcode != 0:
                logger.error(
                    f"ML job {job_id} failed with exit code {process.exitcode}")
            del self.active_jobs[job_id]

    def get_active_job_count(self) -> int:
        """Get number of currently running ML jobs"""
        self.cleanup_completed_jobs()
        return len(self.active_jobs)

    def shutdown(self):
        """Shutdown the ML worker pool gracefully"""
        logger.info("Shutting down ML worker pool")
        for job_id in list(self.active_jobs.keys()):
            self.cancel_job(job_id)


# Global ML worker pool instance
_ml_worker_pool: Optional[MLTrainingWorkerPool] = None


def get_ml_worker_pool(max_workers: int = 1) -> MLTrainingWorkerPool:
    """
    Get or create the global ML worker pool instance

    Args:
        max_workers: Maximum number of parallel ML training jobs

    Returns:
        MLTrainingWorkerPool instance
    """
    global _ml_worker_pool

    if _ml_worker_pool is None:
        _ml_worker_pool = MLTrainingWorkerPool(max_workers=max_workers)

    return _ml_worker_pool


def submit_ml_training_job(
    job_id: str,
    dataset_path: str,
    target_column: str,
    model_id: str,
    dataset_id: str = None,
    config: dict = None
) -> bool:
    """
    Submit an ML training job to the worker pool

    This function is called from the FastAPI endpoint to queue an ML job

    Args:
        job_id: Job ID
        dataset_path: Path to CSV dataset file
        target_column: Target column name
        model_id: Model ID
        dataset_id: Dataset ID for training run tracking
        config: Additional configuration

    Returns:
        True if job was submitted successfully
    """
    job_data = {
        'job_id': job_id,
        'dataset_path': dataset_path,
        'target_column': target_column,
        'model_id': model_id,
        'dataset_id': dataset_id,
        'config': config or {}
    }

    pool = get_ml_worker_pool()
    return pool.submit_job(job_data)
