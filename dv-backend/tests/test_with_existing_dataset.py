#!/usr/bin/env python3
"""
Real-Scenario AutoML Test
Uses existing dataset from database (no upload needed)
"""
import requests
import json
import time
import sys

# Configuration
BACKEND_URL = "http://localhost:8000"
ML_PIPELINE_URL = "http://localhost:8001"

# ============================================================
# CONFIGURE YOUR DATASET HERE
# ============================================================
# Option 1: Set a specific dataset ID if you know it
DATASET_ID = None  # e.g., "abc-123-def-456..."

# Option 2: Auto-find CSV dataset (will use first one found)
AUTO_FIND_CSV = True

# Target column for your dataset
# Common targets: "class", "label", "target", "RainTomorrow", etc.
TARGET_COLUMN = None  # Will auto-detect from dataset name if None
# ============================================================

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, color=BLUE):
    print(f"{color}{message}{RESET}")

def check_services():
    """Check both services are running"""
    print_status("="*70)
    print_status("🧪 AutoML Real-Scenario Test", BLUE)
    print_status("="*70)
    print_status("\n🔍 Step 1: Checking Services", BLUE)

    try:
        r1 = requests.get(f"{BACKEND_URL}/health", timeout=5)
        print_status("  ✅ Main backend running (8000)", GREEN)
    except:
        print_status("  ❌ Main backend not running", RED)
        print_status("     Start: cd /Users/saaivigneshp/Desktop/dv-backend && python main.py", YELLOW)
        return False

    try:
        r2 = requests.get(f"{ML_PIPELINE_URL}/health", timeout=5)
        print_status("  ✅ ML Pipeline service running (8001)", GREEN)
    except:
        print_status("  ❌ ML Pipeline service not running", RED)
        print_status("     Start: cd services/ml_pipeline_service && ./start.sh", YELLOW)
        return False

    return True

def find_csv_dataset():
    """Find a CSV dataset in the database"""
    print_status("\n📊 Step 2: Finding Dataset", BLUE)

    if DATASET_ID:
        print_status(f"  Using specified dataset: {DATASET_ID}", BLUE)
        # Verify it exists
        response = requests.get(f"{BACKEND_URL}/api/datasets/{DATASET_ID}")
        if response.status_code == 200:
            dataset = response.json()
            print_status(f"  ✅ Dataset found: {dataset['name']}", GREEN)
            return dataset
        else:
            print_status(f"  ❌ Dataset {DATASET_ID} not found", RED)
            return None

    # Auto-find CSV dataset
    print_status("  🔍 Searching for CSV datasets in database...", BLUE)
    response = requests.get(f"{BACKEND_URL}/api/datasets")

    if response.status_code != 200:
        print_status(f"  ❌ Failed to list datasets: {response.status_code}", RED)
        return None

    datasets = response.json()
    csv_datasets = [d for d in datasets if d.get('file_format') == 'csv']

    if not csv_datasets:
        print_status("  ❌ No CSV datasets found in database", RED)
        print_status("\n  💡 To add a dataset:", YELLOW)
        print_status("     1. Upload via UI at http://localhost:5173", YELLOW)
        print_status("     2. Or use: curl -X POST http://localhost:8000/api/datasets -F file=@yourdata.csv -F name=test -F domain=tabular", YELLOW)
        return None

    # Use first CSV dataset
    dataset = csv_datasets[0]
    print_status(f"  ✅ Found CSV dataset: {dataset['name']}", GREEN)
    print_status(f"     ID: {dataset['id']}", BLUE)
    print_status(f"     Rows: {dataset.get('total_samples', 'N/A')}", BLUE)
    print_status(f"     Format: {dataset.get('file_format', 'N/A')}", BLUE)

    # Show available datasets
    if len(csv_datasets) > 1:
        print_status(f"\n  ℹ️  Other CSV datasets available: {len(csv_datasets) - 1}", YELLOW)
        for i, d in enumerate(csv_datasets[1:4], 1):  # Show up to 3 more
            print_status(f"     {i}. {d['name']} ({d['id']})", YELLOW)

    return dataset

def detect_target_column(dataset):
    """Try to detect target column from dataset"""
    print_status("\n🎯 Step 3: Detecting Target Column", BLUE)

    if TARGET_COLUMN:
        print_status(f"  Using specified target: {TARGET_COLUMN}", GREEN)
        return TARGET_COLUMN

    # Try to read CSV to get columns
    file_path = dataset.get('file_path')
    if not file_path:
        print_status("  ⚠️  Dataset has no file_path, cannot auto-detect", YELLOW)
        return None

    try:
        import pandas as pd
        from pathlib import Path

        if not Path(file_path).exists():
            print_status(f"  ⚠️  File not found: {file_path}", YELLOW)
            return None

        df = pd.read_csv(file_path, nrows=1)  # Just read header
        columns = df.columns.tolist()

        print_status(f"  📋 Dataset columns ({len(columns)}):", BLUE)
        print_status(f"     {', '.join(columns)}", BLUE)

        # Common target column names
        common_targets = ['class', 'label', 'target', 'y', 'output',
                         'RainTomorrow', 'Survived', 'diagnosis', 'outcome']

        # Check if any common target exists
        for col in columns:
            if col.lower() in [t.lower() for t in common_targets]:
                print_status(f"\n  💡 Auto-detected target: {col}", GREEN)
                print_status(f"     (Common classification target name)", BLUE)
                return col

        # Use last column as fallback
        target = columns[-1]
        print_status(f"\n  💡 Using last column as target: {target}", GREEN)
        print_status(f"     (Standard convention - adjust if needed)", YELLOW)
        return target

    except Exception as e:
        print_status(f"  ⚠️  Could not read CSV: {e}", YELLOW)
        return None

def start_training(dataset_id, target_column):
    """Start AutoML training"""
    print_status(f"\n🚀 Step 4: Starting AutoML Training", BLUE)
    print_status(f"  Dataset ID: {dataset_id}", BLUE)
    print_status(f"  Target Column: {target_column}", BLUE)

    payload = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "model_name": f"automl_model_{int(time.time())}"
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/jobs/train-automl",
            json=payload,
            timeout=300
        )

        if response.status_code in [200, 201]:
            job = response.json()
            print_status(f"  ✅ Job created: {job['id']}", GREEN)
            print_status(f"  📊 Total stages: {job.get('total_iterations', 10)}", BLUE)
            return job
        else:
            print_status(f"  ❌ Failed to start training: {response.status_code}", RED)
            error_detail = response.json().get('detail', response.text)
            print_status(f"     Error: {error_detail}", RED)
            return None

    except Exception as e:
        print_status(f"  ❌ Error: {e}", RED)
        return None

def monitor_training(job_id):
    """Monitor training with real-time progress"""
    print_status(f"\n⏱️  Step 5: Monitoring Training Progress", BLUE)
    print_status(f"  Job ID: {job_id}", BLUE)
    print_status(f"  Polling every 10 seconds...\n", BLUE)

    start_time = time.time()
    last_stage = 0

    while True:
        try:
            response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}", timeout=10)

            if response.status_code != 200:
                print_status(f"\n  ❌ Failed to get job status", RED)
                return False

            job = response.json()
            status = job['status']
            progress = job.get('progress', 0)
            current_iter = job.get('current_iteration', 0)
            total_iter = job.get('total_iterations', 10)

            # Get pipeline stages
            stages = job.get('config', {}).get('pipeline_stages', [])

            # Print new stage info when stage changes
            if current_iter > last_stage and stages:
                elapsed = int(time.time() - start_time)
                current_stage = stages[-1]
                stage_name = current_stage.get('name', 'Unknown')
                print_status(f"\n  [{elapsed:3d}s] Stage {current_iter}/{total_iter}: {stage_name}", GREEN)
                last_stage = current_iter

            # Progress bar
            bar_length = 20
            filled = int(progress / 5)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Current stage name
            stage_name = stages[-1]['name'] if stages else "Initializing..."

            print(f"\r  [{bar}] {progress:5.1f}% | {stage_name[:40]:<40}", end="", flush=True)

            # Check completion
            if status == "completed":
                elapsed = int(time.time() - start_time)
                print("\n")
                print_status(f"  ✅ Training completed successfully!", GREEN)
                print_status(f"  ⏱️  Total time: {elapsed//60}m {elapsed%60}s", GREEN)

                # Show all stages
                print_status(f"\n  📝 Pipeline Execution Summary:", BLUE)
                for stage in stages:
                    stage_num = stage.get('stage', '?')
                    stage_name = stage.get('name', 'Unknown')
                    stage_status = stage.get('status', 'unknown')
                    icon = "✅" if stage_status == "completed" else "❌"
                    print_status(f"     {icon} Stage {stage_num:2d}: {stage_name}", BLUE)

                return True

            elif status == "failed":
                print("\n")
                error = job.get('error', 'Unknown error')
                print_status(f"  ❌ Training failed", RED)
                print_status(f"     Error: {error}", RED)
                return False

        except Exception as e:
            print_status(f"\n  ❌ Monitoring error: {e}", RED)
            return False

        time.sleep(10)

def show_results(job_id):
    """Display final results"""
    print_status(f"\n📊 Step 6: Results", BLUE)

    # Get job details
    response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}")
    if response.status_code != 200:
        print_status("  ❌ Could not fetch job details", RED)
        return

    job = response.json()
    model_id = job.get('model_id')

    if not model_id:
        print_status("  ⚠️  No model ID found", YELLOW)
        return

    # Get model details
    response = requests.get(f"{BACKEND_URL}/api/models/{model_id}")
    if response.status_code != 200:
        print_status("  ❌ Could not fetch model details", RED)
        return

    model = response.json()

    print_status(f"\n  🏆 Model Information:", GREEN)
    print_status(f"     Name: {model.get('name')}", BLUE)
    print_status(f"     ID: {model.get('id')}", BLUE)
    print_status(f"     Framework: {model.get('framework')}", BLUE)
    print_status(f"     Status: {model.get('status')}", BLUE)

    # Metrics
    accuracy = model.get('accuracy')
    metrics = model.get('metrics', {})

    if accuracy or metrics:
        print_status(f"\n  📈 Performance Metrics:", GREEN)
        if accuracy:
            print_status(f"     Accuracy: {accuracy:.2f}%", BLUE)

        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ['accuracy']:
                    print_status(f"     {key.capitalize()}: {value:.4f}", BLUE)

    # Model file
    model_path = model.get('model_path')
    if model_path:
        from pathlib import Path
        model_file = Path(model_path)
        if model_file.exists():
            file_size = model_file.stat().st_size
            print_status(f"\n  💾 Model File:", GREEN)
            print_status(f"     Path: {model_path}", BLUE)
            print_status(f"     Size: {file_size:,} bytes ({file_size/1024:.1f} KB)", BLUE)

def main():
    """Main test execution"""
    # Step 1: Check services
    if not check_services():
        return 1

    # Step 2: Find dataset
    dataset = find_csv_dataset()
    if not dataset:
        return 1

    dataset_id = dataset['id']

    # Step 3: Detect target column
    target_column = detect_target_column(dataset)
    if not target_column:
        print_status("\n  ❌ Could not determine target column", RED)
        print_status("     Please set TARGET_COLUMN at top of script", YELLOW)
        return 1

    # Step 4: Start training
    job = start_training(dataset_id, target_column)
    if not job:
        return 1

    # Step 5: Monitor training
    success = monitor_training(job['id'])
    if not success:
        return 1

    # Step 6: Show results
    show_results(job['id'])

    print_status("\n" + "="*70, GREEN)
    print_status("🎉 Test Completed Successfully!", GREEN)
    print_status("="*70, GREEN)
    print_status("\n💡 Next steps:", YELLOW)
    print_status("   - View model in UI: http://localhost:5173/models", YELLOW)
    print_status("   - Check job details: curl http://localhost:8000/api/jobs/" + job['id'], YELLOW)
    print_status("   - Try different datasets by setting DATASET_ID in script\n", YELLOW)

    return 0

if __name__ == "__main__":
    sys.exit(main())
