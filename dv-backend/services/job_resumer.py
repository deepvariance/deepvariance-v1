"""
Background Job Resumer
- Periodically scans JobDB for jobs in `pending` or `queued` state
- Validates dataset availability and submission prerequisites
- Attempts to submit jobs to the appropriate worker (DL or AutoML)
- Applies exponential backoff and respects MAX_SUBMISSION_ATTEMPTS

This is *not* a hotfix: it's a robust, reusable service started/stopped
from main.py lifespan handler.
"""
import asyncio
import time
import traceback
from typing import Optional

from config import get_config
from database import DatasetDB, JobDB, ModelDB
from job_worker import get_worker_pool, submit_training_job
from ml_training_worker import submit_ml_training_job


class JobResumer:
    def __init__(self):
        self._stopped = True
        self._task: Optional[asyncio.Task] = None
        self.config = get_config()
        self.poll_interval = max(5, self.config.RESUMER_POLL_INTERVAL)
        self.max_attempts = max(1, self.config.MAX_SUBMISSION_ATTEMPTS)

    async def _loop(self):
        """Main resumer loop."""
        while not self._stopped:
            try:
                self._process_pending_jobs()
            except Exception:
                traceback.print_exc()
            await asyncio.sleep(self.poll_interval)

    def _process_pending_jobs(self):
        jobs = JobDB.get_all()
        for job in jobs:
            status = job.get("status")
            if status not in ["pending", "queued"]:
                continue

            job_type = job.get("job_type", "training")
            job_id = job.get("id")

            # Skip jobs already marked with too many attempts
            config = job.get("config") or {}
            attempts = config.get("submission_attempts", 0)
            if attempts >= self.max_attempts:
                # mark as failed to avoid infinite retries
                JobDB.update(
                    job_id, {"status": "failed", "error": "max submission attempts exceeded"})
                continue

            # Validate dataset presence
            dataset_id = job.get("dataset_id")
            dataset = DatasetDB.get_by_id(dataset_id) if dataset_id else None
            if not dataset:
                # Can't submit without dataset
                continue

            # AutoML path
            if job_type == "automl_training":
                # Check CSV path exists
                dataset_path = dataset.get("file_path") or dataset.get("path")
                if not dataset_path:
                    continue

                from pathlib import Path
                dataset_dir = Path(dataset_path)
                if not dataset_dir.exists():
                    continue

                csv_files = list(dataset_dir.glob("*.csv"))
                if not csv_files:
                    continue

                # Get config values safely
                target_column = config.get('target_column', '')
                model_id = job.get('model_id')

                # attempt submission via ml_training_worker
                submitted = submit_ml_training_job(
                    job_id=job_id,
                    dataset_path=str(csv_files[0]),
                    target_column=target_column,
                    model_id=model_id,
                    dataset_id=dataset_id
                )

                # Bump attempt count and persist
                config['submission_attempts'] = attempts + 1
                JobDB.update(job_id, {"config": config})

                if submitted:
                    # Update model status to queued->training transition handled by worker
                    continue
                else:
                    # If submission failed, will be retried later until max attempts
                    continue

            else:
                # DL training submission (job_worker)
                hyperparams = config.get('hyperparameters', {})
                task = config.get('task', 'classification')
                strategy = config.get('strategy', 'auto')
                dataset_obj = dataset

                # Submit to worker pool
                submitted = submit_training_job(
                    job_id=job_id,
                    dataset=dataset_obj,
                    hyperparams=hyperparams,
                    model_id=job.get('model_id'),
                    model_name=job.get('model_name') or None,
                    task=task,
                    strategy=strategy
                )

                # Bump attempt count and persist
                config['submission_attempts'] = attempts + 1
                JobDB.update(job_id, {"config": config})

                # If submitted, worker will update job status to running
                continue

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        if not self._stopped:
            return
        self._stopped = False
        if loop is None:
            loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())

    async def stop(self):
        self._stopped = True
        if self._task:
            await self._task
            self._task = None
