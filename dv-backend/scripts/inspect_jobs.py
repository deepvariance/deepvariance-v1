#!/usr/bin/env python3
"""Inspect jobs, related models, recent logs and worker processes."""
import json
import subprocess

from database import JobDB, ModelDB
from job_logger import JobLogger


def summarize_job(j):
    return {
        "id": j.get("id"),
        "status": j.get("status"),
        "job_type": j.get("job_type"),
        "model_id": j.get("model_id"),
        "dataset_id": j.get("dataset_id"),
        "progress": j.get("progress"),
        "created_at": j.get("created_at")
    }


if __name__ == "__main__":
    jobs = JobDB.get_all()
    print("Total jobs:", len(jobs))

    for s in [summarize_job(j) for j in jobs]:
        print(json.dumps(s, indent=2))

    queued = [j for j in jobs if j.get("status") in ["queued", "pending"]]
    print("\nQueued/pending count:", len(queued))

    for j in queued:
        m = ModelDB.get_by_id(j.get("model_id")) if j.get("model_id") else None
        print(f"\n=== Job: {j.get('id')} (status={j.get('status')}) ===")
        print("Model status:", m.get('status') if m else 'N/A')

        logs = JobLogger.read_logs(j.get('id'), max_lines=200)
        if logs:
            print("Last log lines:")
            for line in logs[-20:]:
                print(line)
        else:
            print("  (no logs found)")

    # Show OS processes that look like training workers
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines(
    ) if "TrainingJob-" in l or "run_training_job" in l or "ml_training_worker" in l or "python main.py" in l]
    print("\nMatching worker processes:", len(lines))
    for l in lines:
        print(l)
