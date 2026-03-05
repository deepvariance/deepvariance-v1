#!/usr/bin/env python3
"""
OpenML Benchmark Score Collection
=================================

This script queries the OpenML REST API to get the best benchmark scores for datasets.
Follows the GitHub Copilot requirements exactly:
1. Uses known dataset IDs
2. Gets tasks via /api/v1/json/task/list/data_id/:data_id
3. Gets evaluations via /api/v1/json/evaluation/list/function/<metric>/task/<task_id>
4. Finds best values (max for accuracy/AUC/F1, min for RMSE)
5. Handles pagination with limit/offset
6. Outputs JSON with specified schema
"""

import json
import time
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

# Configuration
BASE_URL = "https://www.openml.org/api/v1/json"
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1.5
PAGE_LIMIT = 10000

# Target datasets
DATASETS = [
    {"dataset_id": 37, "dataset_name": "diabetes"},
    {"dataset_id": 1464, "dataset_name": "blood-transfusion-service-center"},
    {"dataset_id": 23381, "dataset_name": "dresses-sales"},
    {"dataset_id": 31, "dataset_name": "credit-g"},
    {"dataset_id": 1461, "dataset_name": "bank-marketing"},
    {"dataset_id": 1504, "dataset_name": "steel-plates-fault"},
    {"dataset_id": 6332, "dataset_name": "cylinder-bands"},
]

# Supported metrics and optimization direction
METRICS = {
    "predictive_accuracy": "max",
    "area_under_roc_curve": "max",
    "f_measure": "max",
    "root_mean_squared_error": "min"
}


def make_request_with_retry(url: str) -> Optional[Dict[str, Any]]:
    """Make HTTP request with retries and error handling."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=HTTP_TIMEOUT)

            # Handle OpenML specific error responses
            if response.status_code == 412:
                try:
                    error_data = response.json()
                    if (error_data.get('error', {}).get('code') == '542' and
                            'No results' in error_data.get('error', {}).get('message', '')):
                        return None  # No results available
                except ValueError:
                    pass

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY ** attempt)
                continue
            break

    print(
        f"⚠️  Request failed after {MAX_RETRIES} attempts: {url} - {last_error}")
    return None


def get_tasks_for_dataset(dataset_id: int) -> List[int]:
    """Get all task IDs for a dataset using the specified API endpoint."""
    url = f"{BASE_URL}/task/list/data_id/{dataset_id}"
    data = make_request_with_retry(url)

    if not data:
        return []

    # Handle the nested structure: tasks.task can be dict or list
    tasks = data.get("tasks", {}).get("task", [])
    if isinstance(tasks, dict):
        tasks = [tasks]

    task_ids = []
    for task in tasks:
        task_id = task.get("task_id") or task.get("id")
        if task_id:
            try:
                task_ids.append(int(task_id))
            except (ValueError, TypeError):
                continue

    return list(set(task_ids))  # Remove duplicates


def get_all_runs_for_task(task_id: int) -> Dict[int, Dict[str, float]]:
    """Get all runs for a task with all their metric scores."""
    runs_data = {}

    # Collect evaluations for each metric
    for metric in METRICS:
        all_evaluations = []
        offset = 0

        while True:
            url = f"{BASE_URL}/evaluation/list/function/{metric}/task/{task_id}"
            params = f"?limit={PAGE_LIMIT}&offset={offset}"
            full_url = url + params

            data = make_request_with_retry(full_url)
            if not data:
                break

            # Handle nested structure: evaluations.evaluation
            evaluations = data.get("evaluations", {}).get("evaluation", [])
            if isinstance(evaluations, dict):
                evaluations = [evaluations]

            if not evaluations:
                break

            all_evaluations.extend(evaluations)

            # Check if we've reached the end
            if len(evaluations) < PAGE_LIMIT:
                break

            offset += PAGE_LIMIT

        # Store metric values for each run
        for evaluation in all_evaluations:
            try:
                run_id = evaluation.get("run_id")
                value = float(evaluation.get("value", 0))

                if run_id is None:
                    continue

                run_id = int(run_id)

                if run_id not in runs_data:
                    runs_data[run_id] = {}

                runs_data[run_id][metric] = value

            except (ValueError, TypeError):
                continue

    return runs_data


def find_best_run(runs_data: Dict[int, Dict[str, float]]) -> Optional[int]:
    """Find the best run ID based on composite scoring (prioritizing accuracy)."""
    if not runs_data:
        return None

    best_run_id = None
    best_score = -1

    for run_id, metrics in runs_data.items():
        # Calculate composite score prioritizing accuracy
        score = 0.0
        weight_sum = 0.0

        # Accuracy gets highest weight (50%)
        if "predictive_accuracy" in metrics:
            score += metrics["predictive_accuracy"] * 0.5
            weight_sum += 0.5

        # AUC gets medium weight (30%)
        if "area_under_roc_curve" in metrics:
            score += metrics["area_under_roc_curve"] * 0.3
            weight_sum += 0.3

        # F1 gets lower weight (20%)
        if "f_measure" in metrics:
            score += metrics["f_measure"] * 0.2
            weight_sum += 0.2

        # Normalize score
        if weight_sum > 0:
            final_score = score / weight_sum

            if final_score > best_score:
                best_score = final_score
                best_run_id = run_id

    return best_run_id


def process_dataset(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single dataset and return benchmark results from the best run."""
    dataset_id = dataset["dataset_id"]
    dataset_name = dataset["dataset_name"]

    print(f"🔄 Processing {dataset_name} (ID: {dataset_id})")

    # Get tasks for this dataset
    task_ids = get_tasks_for_dataset(dataset_id)
    if not task_ids:
        print(f"  ⚠️  No tasks found for {dataset_name}")
        return []

    print(f"  📋 Found {len(task_ids)} tasks")
    all_runs_data = {}

    # Collect all runs data from all tasks
    with tqdm(total=len(task_ids), desc=f"  🎯 {dataset_name}", leave=False) as pbar:
        for task_id in task_ids:
            task_runs = get_all_runs_for_task(task_id)

            # Merge task runs into overall runs data, keeping track of task_id
            for run_id, metrics in task_runs.items():
                if run_id not in all_runs_data:
                    all_runs_data[run_id] = {
                        "task_id": task_id, "metrics": metrics}
                else:
                    # If run exists, merge metrics (though this should be rare)
                    all_runs_data[run_id]["metrics"].update(metrics)

            pbar.update(1)

    if not all_runs_data:
        print(f"  ⚠️  No runs found for {dataset_name}")
        return []

    # Find the best run across all tasks
    best_run_id = find_best_run(
        {rid: data["metrics"] for rid, data in all_runs_data.items()})

    if not best_run_id or best_run_id not in all_runs_data:
        print(f"  ⚠️  Could not determine best run for {dataset_name}")
        return []

    best_run_data = all_runs_data[best_run_id]
    best_metrics = best_run_data["metrics"]
    best_task_id = best_run_data["task_id"]

    # Create result with all metrics from the best run
    result = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_id": best_task_id,
        "run_id": best_run_id,
        "metrics": best_metrics
    }

    print(
        f"  ✅ Best run {best_run_id} from task {best_task_id} with {len(best_metrics)} metrics")
    return [result]


def main():
    """Main function to collect OpenML benchmark scores."""
    print("🚀 OpenML Benchmark Score Collection")
    print("📊 Finding the single best run for each dataset")
    print("🎯 Extracting all metric scores from that best run\n")

    all_results = []

    # Process each dataset
    for dataset in tqdm(DATASETS, desc="📦 Datasets"):
        results = process_dataset(dataset)
        all_results.extend(results)

    # Save results to file
    output_file = "openml_best_run_scores.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n🎉 Processing complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Best runs found: {len(all_results)}")

    if all_results:
        # Show detailed results
        print(f"\n🔍 Best Run Results:")
        for result in all_results:
            print(
                f"  🏆 {result['dataset_name']} (Dataset {result['dataset_id']})")
            print(
                f"      Run ID: {result['run_id']} | Task ID: {result['task_id']}")
            print(f"      Metrics:")

            for metric, value in result['metrics'].items():
                print(f"        • {metric}: {value:.4f}")
            print()

    # Print final JSON result as requested
    print(f"\n{'='*60}")
    print("Final JSON Results:")
    print("="*60)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
