"""
Job Worker Service
Runs training jobs asynchronously in separate processes
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
    format='[%(asctime)s] [Worker] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_training_job_worker(job_data: Dict):
    """
    Worker function that runs in a separate process
    This is the actual training execution that runs independently
    """
    # Import inside worker to avoid pickling issues
    from models import HyperparametersConfig, ModelTask
    from training_runner import run_training_job

    # Set GROQ API key if available
    if 'GROQ_API_KEY' in os.environ:
        os.environ['GROQ_API_KEY'] = os.environ['GROQ_API_KEY']

    logger.info(
        f"Worker process {os.getpid()} starting job {job_data['job_id']}")

    try:
        # Reconstruct hyperparams
        hyperparams = HyperparametersConfig(**job_data['hyperparameters'])

        # Reconstruct task enum
        task = ModelTask(job_data['task'])

        # Run the training
        run_training_job(
            job_id=job_data['job_id'],
            dataset=job_data['dataset'],
            hyperparams=hyperparams,
            model_id=job_data['model_id'],
            model_name=job_data['model_name'],
            task=task,
            strategy=job_data.get('strategy', 'auto')
        )

        logger.info(
            f"Worker process {os.getpid()} completed job {job_data['job_id']}")

    except Exception as e:
        logger.error(
            f"Worker process {os.getpid()} failed job {job_data['job_id']}: {e}")
        import traceback
        traceback.print_exc()


class JobWorkerPool:
    """
    Manages a pool of worker processes for running training jobs
    """

    def __init__(self, max_workers: int = 2):
        """
        Initialize the worker pool

        Args:
            max_workers: Maximum number of parallel training jobs
        """
        self.max_workers = max_workers
        self.pool = multiprocessing.Pool(processes=max_workers)
        # job_id -> {'result': AsyncResult, 'process': Process}
        self.active_jobs: Dict[str, Dict] = {}
        logger.info(
            f"Initialized job worker pool with {max_workers} worker(s)")

    def submit_job(self, job_data: Dict) -> bool:
        """
        Submit a training job to the worker pool

        Args:
            job_data: Dictionary containing job information

        Returns:
            True if job was submitted successfully
        """
        job_id = job_data['job_id']

        # Check if job is already running
        if job_id in self.active_jobs:
            logger.warning(f"Job {job_id} is already running")
            return False

        # Create a dedicated process for this job (instead of pool)
        process = multiprocessing.Process(
            target=run_training_job_worker,
            args=(job_data,),
            name=f"TrainingJob-{job_id}"
        )

        logger.info(f"Starting job {job_id} in process {process.name}")
        process.start()

        self.active_jobs[job_id] = {
            'process': process,
            'started_at': time.time()
        }

        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running training job

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled successfully
        """
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return False

        job_info = self.active_jobs[job_id]
        process = job_info['process']

        if process.is_alive():
            logger.info(f"Terminating job {job_id} (PID: {process.pid})")
            process.terminate()
            process.join(timeout=5)

            if process.is_alive():
                logger.warning(f"Force killing job {job_id}")
                process.kill()
                process.join()

            logger.info(f"Job {job_id} terminated")

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
                    f"Job {job_id} failed with exit code {process.exitcode}")
            del self.active_jobs[job_id]

    def get_active_job_count(self) -> int:
        """Get number of currently running jobs"""
        self.cleanup_completed_jobs()
        return len(self.active_jobs)

    def shutdown(self):
        """Shutdown the worker pool gracefully"""
        logger.info("Shutting down worker pool")
        for job_id in list(self.active_jobs.keys()):
            self.cancel_job(job_id)
        self.pool.close()
        self.pool.join()


# Global worker pool instance
_worker_pool: Optional[JobWorkerPool] = None


def get_worker_pool(max_workers: int = 2) -> JobWorkerPool:
    """
    Get or create the global worker pool instance

    Args:
        max_workers: Maximum number of parallel training jobs

    Returns:
        JobWorkerPool instance
    """
    global _worker_pool

    if _worker_pool is None:
        _worker_pool = JobWorkerPool(max_workers=max_workers)

    return _worker_pool


def submit_training_job(
    job_id: str,
    dataset: dict,
    hyperparams: dict,
    model_id: str,
    model_name: str,
    task: str,
    strategy: str = 'auto'
) -> bool:
    """
    Submit a training job to the worker pool

    This function is called from the FastAPI endpoint to queue a job

    Args:
        job_id: Job ID
        dataset: Dataset dictionary
        hyperparams: Hyperparameters as dict
        model_id: Model ID
        model_name: Model name
        task: Task type (classification, etc.)
        strategy: Training strategy

    Returns:
        True if job was submitted successfully
    """
    job_data = {
        'job_id': job_id,
        'dataset': dataset,
        'hyperparameters': hyperparams,
        'model_id': model_id,
        'model_name': model_name,
        'task': task,
        'strategy': strategy
    }

    pool = get_worker_pool()
    return pool.submit_job(job_data)
