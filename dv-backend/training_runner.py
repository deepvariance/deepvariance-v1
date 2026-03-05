"""
Training Job Runner
Background task runner using the plugin-based training pipeline
"""

import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from database import DatasetDB, JobDB, ModelDB, TrainingRunDB
from hardware_utils import get_optimal_device, log_device_info
from job_logger import JobLogger
from models import HyperparametersConfig, ModelTask
from training_pipeline import (ProgressUpdate, TrainingConfig,
                               TrainingOrchestrator)

# Global flag for graceful shutdown
_training_cancelled = False
_monitoring_thread = None


def _signal_handler(signum, frame):
    """Handle termination signals - immediately exit"""
    print(f"\n[SIGNAL] Received signal {signum}, terminating immediately...")
    sys.stdout.flush()
    os._exit(1)  # Force immediate exit


def _monitor_cancellation(job_id: str, check_interval: float = 0.5):
    """Monitor job status in database and force exit if cancelled"""
    global _training_cancelled
    while not _training_cancelled:
        try:
            job = JobDB.get_by_id(job_id)
            if job and job.get('status') == 'failed':
                # Job was marked as failed (cancelled) externally
                print(
                    f"\n[MONITOR] Job {job_id} cancelled, terminating process...")
                sys.stdout.flush()
                os._exit(1)  # Force immediate exit
        except Exception as e:
            print(f"[MONITOR] Error checking job status: {e}")
        time.sleep(check_interval)  # Check every 0.5 seconds


def run_training_job(
    job_id: str,
    dataset: dict,
    hyperparams: HyperparametersConfig,
    model_id: str = None,
    model_name: str = None,
    task: ModelTask = ModelTask.CLASSIFICATION,
    strategy: str = 'auto'
):
    """
    Run a training job using the plugin-based training pipeline

    Args:
        job_id: Job ID for tracking
        dataset: Dataset dictionary from database
        hyperparams: Hyperparameters configuration
        model_id: Model ID (created before training)
        model_name: Model name
        task: Training task (classification, regression, etc.)
        strategy: Training strategy (llm, native, auto)
    """
    import time
    start_time = time.time()

    # Register signal handlers for immediate termination
    global _training_cancelled, _monitoring_thread
    _training_cancelled = False
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Start monitoring thread to check for cancellation in database
    _monitoring_thread = threading.Thread(
        target=_monitor_cancellation,
        args=(job_id,),
        daemon=True  # Daemon thread will be killed when main process exits
    )
    _monitoring_thread.start()

    # Create logger for this job
    logger = JobLogger(job_id)

    try:
        logger.info(f"Starting training job {job_id}")

        # Update model status to training
        if model_id:
            ModelDB.update(model_id, {"status": "training"})
            logger.info(f"Model {model_id} status updated to 'training'")

        # Update job status to running
        JobDB.update(job_id, {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "progress": 0.0
        })
        logger.info("Job status updated to 'running'")

        # Extract dataset info
        dataset_id = dataset.get("id")
        dataset_path = Path(dataset.get("path"))
        dataset_name = dataset.get("name")
        dataset_domain = dataset.get("domain")
        num_samples = dataset.get("size", 0)

        # Infer number of classes from dataset structure (for vision datasets)
        num_classes = _infer_num_classes(dataset_path, dataset_domain)

        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"  - Samples: {num_samples}")
        logger.info(f"  - Classes: {num_classes}")
        logger.info(f"  - Domain: {dataset_domain}")

        # Create training run record (after dataset_id is extracted)
        training_run = TrainingRunDB.create({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "config": hyperparams.model_dump() if hyperparams else {},
            "status": "pending",
            "total_epochs": hyperparams.max_iterations if hyperparams else 10,
        })
        training_run_id = training_run.get("id")
        logger.info(f"Created training run {training_run_id}")

        # Detect optimal hardware device
        device, device_description = get_optimal_device()
        logger.info("")
        log_device_info(logger)
        logger.info("")
        logger.info(f"Using device: {device} - {device_description}")

        # Build training configuration
        config = TrainingConfig(
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            dataset_domain=dataset_domain,
            num_classes=num_classes,
            num_samples=num_samples,
            model_id=model_id,
            model_name=model_name or f"{dataset_name}_model",
            task=task.value if isinstance(task, ModelTask) else task,
            # Hyperparameters (optional - strategy may determine)
            learning_rate=hyperparams.learning_rate if hyperparams else None,
            batch_size=hyperparams.batch_size if hyperparams else None,
            epochs=hyperparams.epochs if hyperparams else None,
            optimizer=hyperparams.optimizer if hyperparams else None,
            dropout_rate=hyperparams.dropout_rate if hyperparams else None,
            # Training config
            max_iterations=hyperparams.max_iterations if hyperparams else 10,
            target_accuracy=hyperparams.target_accuracy if hyperparams else 1.0,
            device=device,  # Adaptive: CUDA → MPS → CPU
            # Platform integration
            job_id=job_id,
            strategy=strategy
        )

        # Create orchestrator
        orchestrator = TrainingOrchestrator()

        # Define progress callback
        def on_progress(progress: ProgressUpdate):
            """Update job progress in database"""
            try:
                # Calculate overall progress percentage
                progress_percent = (progress.iteration /
                                    progress.total_iterations) * 100

                # Calculate time tracking
                elapsed_seconds = int(time.time() - start_time)

                # Estimate remaining time based on progress
                if progress_percent > 0:
                    total_estimated_seconds = int(
                        (elapsed_seconds / progress_percent) * 100)
                    remaining_seconds = max(
                        0, total_estimated_seconds - elapsed_seconds)
                else:
                    remaining_seconds = 0

                # Format time strings
                elapsed_time = _format_time(elapsed_seconds)
                estimated_remaining = _format_time(remaining_seconds)

                # Update job in database
                update_data = {
                    "progress": progress_percent,
                    "current_iteration": progress.iteration,
                    "total_iterations": progress.total_iterations,
                    "current_accuracy": progress.current_accuracy,
                    "current_loss": progress.current_loss,
                }

                # Update best metrics
                if progress.best_accuracy is not None:
                    update_data["best_accuracy"] = progress.best_accuracy

                if progress.best_loss is not None:
                    update_data["best_loss"] = progress.best_loss

                # Update classification metrics
                if progress.precision is not None:
                    update_data["precision"] = progress.precision

                if progress.recall is not None:
                    update_data["recall"] = progress.recall

                if progress.f1_score is not None:
                    update_data["f1_score"] = progress.f1_score

                if progress.status == 'failed':
                    update_data["status"] = "failed"
                    update_data["error_message"] = progress.message

                # Store time tracking in config JSONB field
                job = JobDB.get_by_id(job_id)
                if job:
                    config = job.get("config") or {}
                    config["elapsed_seconds"] = elapsed_seconds
                    config["remaining_seconds"] = remaining_seconds
                    config["elapsed_time"] = elapsed_time
                    config["estimated_remaining"] = estimated_remaining
                    update_data["config"] = config

                JobDB.update(job_id, update_data)

                # Also update training run - use nonlocal to access outer scope variable
                if training_run_id:
                    # Get current epoch_metrics or initialize empty list
                    current_run = TrainingRunDB.get_by_id(training_run_id)
                    if current_run:
                        epoch_metrics = current_run.get('epoch_metrics') or []
                    else:
                        epoch_metrics = []

                    # Ensure epoch_metrics is a list
                    if not isinstance(epoch_metrics, list):
                        epoch_metrics = []

                    # Append current iteration metrics
                    iteration_metrics = {
                        "iteration": progress.iteration,
                        "accuracy": progress.current_accuracy,
                        "loss": progress.current_loss,
                        "precision": progress.precision,
                        "recall": progress.recall,
                        "f1_score": progress.f1_score,
                        "hyperparameters": progress.hyperparameters,
                        "timestamp": datetime.now().isoformat()
                    }
                    epoch_metrics.append(iteration_metrics)

                    run_update = {
                        "progress": progress_percent,
                        "current_epoch": progress.iteration,
                        "status": "running" if progress.status != 'failed' else "failed",
                        "epoch_metrics": epoch_metrics
                    }

                    # Set started_at timestamp on first iteration (when status changes to running)
                    if progress.iteration == 1 and progress.status != 'failed':
                        run_update["started_at"] = datetime.now()

                    if progress.current_loss is not None:
                        run_update["final_loss"] = progress.current_loss

                    if progress.current_accuracy is not None:
                        run_update["final_accuracy"] = progress.current_accuracy

                    if progress.best_loss is not None:
                        run_update["best_loss"] = progress.best_loss

                    if progress.best_accuracy is not None:
                        run_update["best_accuracy"] = progress.best_accuracy

                    if progress.status == 'failed':
                        run_update["error_message"] = progress.message

                    TrainingRunDB.update(training_run_id, run_update)

                # Log progress with each metric on a separate line for readability
                logger.info("")
                logger.info(
                    f"=== Iteration {progress.iteration}/{progress.total_iterations} ===")

                # Current metrics
                if progress.current_accuracy is not None:
                    logger.info(
                        f"  Accuracy:  {progress.current_accuracy:.4f}")
                if progress.current_loss is not None:
                    logger.info(f"  Loss:      {progress.current_loss:.4f}")
                if progress.precision is not None:
                    logger.info(f"  Precision: {progress.precision:.4f}")
                if progress.recall is not None:
                    logger.info(f"  Recall:    {progress.recall:.4f}")
                if progress.f1_score is not None:
                    logger.info(f"  F1 Score:  {progress.f1_score:.4f}")

                # Best metrics
                if progress.best_accuracy is not None or progress.best_loss is not None:
                    logger.info("")
                    if progress.best_accuracy is not None:
                        logger.info(
                            f"  Best Acc:  {progress.best_accuracy:.4f}")
                    if progress.best_loss is not None:
                        logger.info(f"  Best Loss: {progress.best_loss:.4f}")

                # Time tracking
                logger.info("")
                logger.info(f"  Elapsed:   {elapsed_time}")
                logger.info(f"  Remaining: {estimated_remaining}")
                logger.info("")

            except Exception as e:
                logger.error(f"Error updating progress: {e}")

        # Execute training
        logger.info("=" * 50)
        logger.info(f"Starting training with strategy: {strategy}")
        logger.info("")
        logger.info("Training Configuration:")
        logger.info(
            f"  Learning Rate: {config.learning_rate if config.learning_rate else 'auto'}")
        logger.info(
            f"  Batch Size: {config.batch_size if config.batch_size else 'auto'}")
        logger.info(f"  Epochs: {config.epochs if config.epochs else 'auto'}")
        logger.info(
            f"  Optimizer: {config.optimizer if config.optimizer else 'auto'}")
        logger.info(
            f"  Dropout Rate: {config.dropout_rate if config.dropout_rate else 'auto'}")
        logger.info(f"  Max Iterations: {config.max_iterations}")
        logger.info(f"  Target Accuracy: {config.target_accuracy}")
        logger.info("=" * 50)
        logger.info("")
        result = orchestrator.train(config, progress_callback=on_progress)

        # Check if training succeeded
        if result.success:
            logger.info("=" * 50)
            logger.info("Training completed successfully!")
            if result.final_accuracy:
                logger.info(f"Final accuracy: {result.final_accuracy:.4f}")
            if result.best_accuracy:
                logger.info(f"Best accuracy: {result.best_accuracy:.4f}")
            if result.hyperparameters:
                logger.info(f"Final hyperparameters: {result.hyperparameters}")
            if result.model_path:
                logger.info(f"Model saved to: {result.model_path}")
            logger.info("=" * 50)

            # Update model with final results
            if model_id:
                model_update = {
                    "status": "ready",
                    "last_trained": datetime.now().isoformat(),
                    "accuracy": result.final_accuracy * 100 if result.final_accuracy else None,
                    "loss": result.final_loss,
                    "model_path": str(result.model_path) if result.model_path else None,
                    "hyperparameters": result.hyperparameters,
                    "metrics": result.metrics,
                }
                ModelDB.update(model_id, model_update)
                logger.info(f"Model {model_id} updated with final results")

            # Update job as completed (preserve iteration counts!)
            job_data = JobDB.get_by_id(job_id)
            elapsed_seconds = int(time.time() - start_time)

            # Normalize metrics from result.metrics if present
            def _normalize_value(v: object) -> float | None:
                try:
                    if v is None:
                        return None
                    val = float(v)
                    # If value looks like a percentage (>1 and <=1000), convert to ratio
                    if val > 1 and val <= 1000:
                        return val / 100.0
                    return val
                except Exception:
                    return None

            metrics = result.metrics or {}

            # Try multiple possible keys for each metric
            def _extract_metric(keys: list[str]):
                for k in keys:
                    if k in metrics and metrics[k] is not None:
                        return _normalize_value(metrics[k])
                return None

            current_acc = result.final_accuracy or result.best_accuracy or _extract_metric(
                ['current_accuracy', 'accuracy', 'Accuracy%', 'Accuracy', 'test_accuracy', 'acc'])
            # ensure ratio (0..1)
            if isinstance(current_acc, (int, float)) and current_acc > 1:
                current_acc = current_acc / 100.0

            current_loss = result.final_loss or _extract_metric(
                ['current_loss', 'loss', 'Loss', 'test_loss'])

            precision = _extract_metric(
                ['precision', 'Precision', 'prec', 'precision_score'])
            recall = _extract_metric(['recall', 'Recall', 'recall_score'])
            f1 = _extract_metric(['f1_score', 'f1', 'F1'])

            job_update = {
                'status': 'completed',
                'progress': 100.0,
                'current_iteration': job_data.get('total_iterations') or config.max_iterations,
                'total_iterations': job_data.get('total_iterations') or config.max_iterations,
                'best_accuracy': result.best_accuracy,
                'completed_at': datetime.now().isoformat()
            }

            if current_acc is not None:
                job_update['current_accuracy'] = current_acc
            if current_loss is not None:
                job_update['current_loss'] = current_loss
            if precision is not None:
                job_update['precision'] = precision
            if recall is not None:
                job_update['recall'] = recall
            if f1 is not None:
                job_update['f1_score'] = f1

            JobDB.update(job_id, job_update)

            # Update training run as completed (if exists)
            if 'training_run_id' in locals():
                tr_update = {
                    'status': 'completed',
                    'progress': 100.0,
                    'final_accuracy': current_acc if current_acc is not None else result.best_accuracy,
                    'final_loss': current_loss if current_loss is not None else result.final_loss,
                    'best_accuracy': result.best_accuracy,
                    'best_loss': current_loss if current_loss is not None else result.final_loss,
                    'duration_seconds': elapsed_seconds,
                    'completed_at': datetime.now()
                }

                if precision is not None:
                    tr_update['precision'] = precision
                if recall is not None:
                    tr_update['recall'] = recall
                if f1 is not None:
                    tr_update['f1_score'] = f1

                TrainingRunDB.update(training_run_id, tr_update)
                logger.info(
                    f"Training run {training_run_id} marked as completed")

            logger.info(f"Job {job_id} completed successfully!")

        else:
            # Training failed
            logger.error(f"Training failed: {result.error}")

            # Update model status to failed
            if model_id:
                ModelDB.update(model_id, {"status": "failed"})
                logger.info(f"Model {model_id} marked as failed")

            # Update job as failed
            JobDB.update(job_id, {
                "status": "failed",
                "error_message": result.error,
                "completed_at": datetime.now().isoformat()
            })
            logger.info(f"Job {job_id} marked as failed")

            # Update training run as failed
            if 'training_run_id' in locals():
                elapsed_seconds = int(time.time() - start_time)
                TrainingRunDB.update(training_run_id, {
                    "status": "failed",
                    "error_message": result.error,
                    "error_traceback": result.error_traceback,
                    "duration_seconds": elapsed_seconds,
                    "completed_at": datetime.now()
                })
                logger.info(f"Training run {training_run_id} marked as failed")

    except KeyboardInterrupt:
        # Handle user cancellation
        logger.info("Training stopped by user")

        # Update model status to stopped
        if model_id:
            ModelDB.update(model_id, {"status": "stopped"})

        # Update job as stopped
        JobDB.update(job_id, {
            "status": "stopped",
            "error_message": "Training stopped by user",
            "completed_at": datetime.now().isoformat()
        })

        # Update training run as stopped
        if 'training_run_id' in locals():
            elapsed_seconds = int(time.time() - start_time)
            TrainingRunDB.update(training_run_id, {
                "status": "stopped",
                "error_message": "Training stopped by user",
                "duration_seconds": elapsed_seconds,
                "completed_at": datetime.now()
            })

    except Exception as e:
        # Handle unexpected errors
        error_msg = str(e)
        error_trace = traceback.format_exc()

        logger.error(f"Unexpected error: {error_msg}")
        logger.error("Full traceback:")
        for line in error_trace.split('\n'):
            if line.strip():
                logger.error(f"  {line}")

        # Update model status to failed
        if model_id:
            ModelDB.update(model_id, {"status": "failed"})

        # Update job as failed
        JobDB.update(job_id, {
            "status": "failed",
            "error_message": error_msg,
            "completed_at": datetime.now().isoformat()
        })

        # Update training run as failed
        if 'training_run_id' in locals():
            elapsed_seconds = int(time.time() - start_time)
            TrainingRunDB.update(training_run_id, {
                "status": "failed",
                "error_message": error_msg,
                "error_traceback": error_trace,
                "duration_seconds": elapsed_seconds,
                "completed_at": datetime.now()
            })

    finally:
        # Stop monitoring thread
        _training_cancelled = True

        # Close logger
        logger.close()


def _infer_num_classes(dataset_path: Path, dataset_domain: str) -> int:
    """
    Infer number of classes from dataset structure

    For vision datasets, assumes structure:
        dataset/
            train/
                class1/
                class2/
                ...
    """
    if dataset_domain == 'vision':
        # Look for class directories in train folder
        train_dir = dataset_path / 'train'
        if train_dir.exists() and train_dir.is_dir():
            # Count subdirectories (each is a class)
            class_dirs = [d for d in train_dir.iterdir(
            ) if d.is_dir() and not d.name.startswith('.')]
            return len(class_dirs)

        # Fallback: count top-level directories
        class_dirs = [d for d in dataset_path.iterdir(
        ) if d.is_dir() and not d.name.startswith('.')]
        return max(len(class_dirs), 2)  # At least binary classification

    # Default for other domains
    return 2


def _format_time(seconds: int) -> str:
    """
    Format seconds into human-readable time string

    Args:
        seconds: Total seconds

    Returns:
        Formatted time string (e.g., "2h 15m", "45m 32s", "12s")
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


# Legacy functions for backward compatibility (deprecated)
def run_training_job_real(*args, **kwargs):
    """Deprecated: Use run_training_job instead"""
    print("[WARNING] run_training_job_real is deprecated. Use run_training_job instead.")
    return run_training_job(*args, **kwargs)
