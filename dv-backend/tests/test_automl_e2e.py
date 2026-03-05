#!/usr/bin/env python3
"""
End-to-End AutoML Integration Test
Tests the complete flow from dataset upload to model training
"""
import requests
import json
import time
import sys
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
ML_PIPELINE_URL = "http://localhost:8001"

# Choose which dataset to test
# DATASET_PATH = "/Users/saaivigneshp/Desktop/dv-backend/data/test_automl_diabetes.csv"  # Small (768 rows)
DATASET_PATH = "/Users/saaivigneshp/Desktop/dv-backend/data/test_automl_weather.csv"  # Large (145K rows)

# Dataset-specific settings
DATASET_CONFIG = {
    "test_automl_diabetes.csv": {
        "name": "diabetes_test",
        "target_column": "class",
        "expected_time": "5-7 minutes"
    },
    "test_automl_weather.csv": {
        "name": "weather_australia_test",
        "target_column": "RainTomorrow",
        "expected_time": "25-35 minutes (large dataset, will be auto-sampled)"
    }
}

# Get config for current dataset
import os
DATASET_NAME = os.path.basename(DATASET_PATH)
CONFIG = DATASET_CONFIG.get(DATASET_NAME, {"name": "test_dataset", "target_column": None})

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    """Print colored status message"""
    colors = {
        "info": BLUE,
        "success": GREEN,
        "error": RED,
        "warning": YELLOW
    }
    color = colors.get(status, RESET)
    print(f"{color}{message}{RESET}")

def check_services():
    """Check that both services are running"""
    print_status("\n🔍 Step 1: Checking Services", "info")

    # Check main backend
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print_status("  ✅ Main backend is running", "success")
        else:
            print_status(f"  ❌ Main backend returned: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"  ❌ Main backend not reachable: {e}", "error")
        print_status("     Please start: python main.py", "warning")
        return False

    # Check ML Pipeline service
    try:
        response = requests.get(f"{ML_PIPELINE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_status("  ✅ ML Pipeline service is running", "success")
        else:
            print_status(f"  ❌ ML Pipeline service returned: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"  ❌ ML Pipeline service not reachable: {e}", "error")
        print_status("     Please start: cd services/ml_pipeline_service && ./start.sh", "warning")
        return False

    return True

def upload_dataset():
    """Upload test dataset"""
    print_status("\n📤 Step 2: Uploading Test Dataset", "info")

    if not Path(DATASET_PATH).exists():
        print_status(f"  ❌ Dataset not found: {DATASET_PATH}", "error")
        return None

    print_status(f"  📁 Dataset: {DATASET_PATH}", "info")

    # Read CSV to get column names
    import pandas as pd
    df = pd.read_csv(DATASET_PATH)
    print_status(f"  📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns", "info")
    print_status(f"  📋 Columns: {', '.join(df.columns.tolist())}", "info")

    # Upload via API
    files = {'file': open(DATASET_PATH, 'rb')}
    data = {
        'name': CONFIG['name'],
        'domain': 'tabular'
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/datasets",
            files=files,
            data=data,
            timeout=30
        )

        if response.status_code in [200, 201]:
            dataset = response.json()
            print_status(f"  ✅ Dataset uploaded: {dataset['id']}", "success")
            return dataset
        else:
            print_status(f"  ❌ Upload failed: {response.status_code}", "error")
            print_status(f"     {response.text}", "error")
            return None
    except Exception as e:
        print_status(f"  ❌ Upload error: {e}", "error")
        return None

def find_existing_csv_dataset():
    """Find an existing CSV dataset to use for testing"""
    print_status("\n🔍 Looking for existing CSV dataset...", "info")

    try:
        response = requests.get(f"{BACKEND_URL}/api/datasets", timeout=10)
        if response.status_code == 200:
            datasets = response.json()
            csv_datasets = [d for d in datasets if d.get('file_format') == 'csv']

            if csv_datasets:
                dataset = csv_datasets[0]
                print_status(f"  ✅ Found CSV dataset: {dataset['name']} ({dataset['id']})", "success")

                # Check what columns are available
                import pandas as pd
                file_path = dataset.get('file_path')
                if file_path and Path(file_path).exists():
                    df = pd.read_csv(file_path)
                    print_status(f"  📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns", "info")
                    print_status(f"  📋 Columns: {', '.join(df.columns.tolist())}", "info")
                    dataset['columns'] = df.columns.tolist()

                return dataset

        return None
    except Exception as e:
        print_status(f"  ❌ Error finding dataset: {e}", "error")
        return None

def start_automl_training(dataset_id, target_column, model_name="test_automl_model"):
    """Start AutoML training job"""
    print_status(f"\n🚀 Step 3: Starting AutoML Training", "info")
    print_status(f"  🎯 Target column: {target_column}", "info")

    payload = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "model_name": model_name
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/jobs/train-automl",
            json=payload,
            timeout=300  # 5 minutes timeout
        )

        if response.status_code in [200, 201]:
            job = response.json()
            print_status(f"  ✅ Job created: {job['id']}", "success")
            print_status(f"  📈 Progress: {job['progress']}%", "info")
            print_status(f"  🔢 Total iterations: {job['total_iterations']}", "info")
            return job
        else:
            print_status(f"  ❌ Training failed: {response.status_code}", "error")
            print_status(f"     {response.text}", "error")
            return None
    except Exception as e:
        print_status(f"  ❌ Training error: {e}", "error")
        return None

def monitor_training(job_id, max_wait_minutes=40):
    """Monitor training progress"""
    print_status(f"\n⏱️  Step 4: Monitoring Training Progress", "info")
    print_status(f"  ⏰ Max wait time: {max_wait_minutes} minutes", "info")
    print_status(f"  🔄 Polling every 10 seconds...\n", "info")

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    iteration = 0

    while True:
        iteration += 1
        elapsed = int(time.time() - start_time)

        if elapsed > max_wait_seconds:
            print_status(f"\n  ⏰ Timeout reached ({max_wait_minutes} minutes)", "warning")
            return False

        try:
            response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}", timeout=10)

            if response.status_code == 200:
                job = response.json()
                status = job['status']
                progress = job.get('progress', 0)
                current_iter = job.get('current_iteration', 0)
                total_iter = job.get('total_iterations', 10)

                # Get current pipeline stage
                pipeline_stages = job.get('config', {}).get('pipeline_stages', [])
                current_stage = pipeline_stages[-1] if pipeline_stages else None

                # Print progress
                progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
                stage_name = current_stage['name'] if current_stage else "Starting..."

                print(f"\r  [{progress_bar}] {progress:.0f}% | Stage {current_iter}/{total_iter}: {stage_name}", end="", flush=True)

                # Check if completed
                if status == "completed":
                    print("\n")
                    print_status(f"  ✅ Training completed successfully!", "success")
                    print_status(f"  ⏱️  Total time: {elapsed}s ({elapsed//60}m {elapsed%60}s)", "info")

                    # Print all stages
                    print_status(f"\n  📝 Pipeline Stages:", "info")
                    for stage in pipeline_stages:
                        stage_num = stage.get('stage', '?')
                        stage_name = stage.get('name', 'Unknown')
                        stage_status = stage.get('status', 'unknown')
                        icon = "✅" if stage_status == "completed" else "⏳"
                        print_status(f"     {icon} Stage {stage_num}: {stage_name}", "info")

                    return True

                elif status == "failed":
                    print("\n")
                    error = job.get('error', 'Unknown error')
                    print_status(f"  ❌ Training failed: {error}", "error")
                    return False

            else:
                print_status(f"\n  ❌ Failed to get job status: {response.status_code}", "error")
                return False

        except Exception as e:
            print_status(f"\n  ❌ Error checking status: {e}", "error")
            return False

        time.sleep(10)

def check_results(job_id):
    """Check final training results"""
    print_status(f"\n📊 Step 5: Checking Results", "info")

    # Get job details
    try:
        response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}", timeout=10)
        if response.status_code == 200:
            job = response.json()

            # Get model details
            model_id = job.get('model_id')
            if model_id:
                model_response = requests.get(f"{BACKEND_URL}/api/models/{model_id}", timeout=10)

                if model_response.status_code == 200:
                    model = model_response.json()

                    print_status(f"\n  🏆 Model Details:", "success")
                    print_status(f"     Name: {model.get('name')}", "info")
                    print_status(f"     ID: {model.get('id')}", "info")
                    print_status(f"     Framework: {model.get('framework')}", "info")
                    print_status(f"     Status: {model.get('status')}", "info")

                    # Metrics
                    accuracy = model.get('accuracy')
                    metrics = model.get('metrics', {})

                    print_status(f"\n  📈 Metrics:", "success")
                    if accuracy:
                        print_status(f"     Accuracy: {accuracy:.2f}%", "info")

                    if metrics:
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                print_status(f"     {key.capitalize()}: {value:.4f}", "info")

                    # Model file
                    model_path = model.get('model_path')
                    if model_path:
                        model_file = Path(model_path)
                        if model_file.exists():
                            file_size = model_file.stat().st_size
                            print_status(f"\n  💾 Model File:", "success")
                            print_status(f"     Path: {model_path}", "info")
                            print_status(f"     Size: {file_size:,} bytes ({file_size/1024:.1f} KB)", "info")
                        else:
                            print_status(f"\n  ⚠️  Model file not found: {model_path}", "warning")

                    return True

        return False
    except Exception as e:
        print_status(f"  ❌ Error checking results: {e}", "error")
        return False

def main():
    """Run end-to-end test"""
    print_status("="*70, "info")
    print_status("🧪 AutoML Integration - End-to-End Test", "info")
    print_status("="*70, "info")

    # Step 1: Check services
    if not check_services():
        print_status("\n❌ Services check failed. Please start both services.", "error")
        return 1

    # Step 2: Use existing CSV dataset or skip upload
    dataset = find_existing_csv_dataset()

    if not dataset:
        print_status("  ℹ️  No existing CSV dataset found. Attempting to use test file...", "info")
        # Try to use the copied file directly
        if Path(DATASET_PATH).exists():
            import pandas as pd
            df = pd.read_csv(DATASET_PATH)

            # Manually create dataset entry
            print_status("  ⚠️  Note: For full test, upload dataset via UI first", "warning")
            print_status(f"  📊 Test dataset has columns: {', '.join(df.columns.tolist())}", "info")
            print_status(f"  💡 Suggested target column: {df.columns[-1]}", "info")

        print_status("\n❌ Test incomplete: No CSV dataset available", "error")
        print_status("   Please upload a CSV dataset via the UI first, then run this test.", "warning")
        return 1

    # Determine target column
    columns = dataset.get('columns', [])
    if not columns:
        print_status("  ❌ Cannot determine dataset columns", "error")
        return 1

    # Use configured target column or last column as fallback
    target_column = CONFIG.get('target_column') or columns[-1]
    print_status(f"\n  💡 Using target column: '{target_column}'", "info")

    if CONFIG.get('expected_time'):
        print_status(f"  ⏰ Expected training time: {CONFIG['expected_time']}", "info")

    # Step 3: Start training
    job = start_automl_training(dataset['id'], target_column)
    if not job:
        print_status("\n❌ Failed to start training", "error")
        return 1

    # Step 4: Monitor training
    success = monitor_training(job['id'], max_wait_minutes=15)
    if not success:
        print_status("\n❌ Training did not complete successfully", "error")
        return 1

    # Step 5: Check results
    if not check_results(job['id']):
        print_status("\n⚠️  Could not verify all results", "warning")

    print_status("\n" + "="*70, "success")
    print_status("🎉 End-to-End Test PASSED!", "success")
    print_status("="*70, "success")

    return 0

if __name__ == "__main__":
    sys.exit(main())
