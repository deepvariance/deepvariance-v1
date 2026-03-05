#!/usr/bin/env python3
"""
Simple test that uploads dataset automatically
"""
import requests
import json
import time
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
ML_PIPELINE_URL = "http://localhost:8001"

# Use diabetes for quick test (switch to weather for full test)
DATASET_PATH = "/Users/saaivigneshp/Desktop/dv-backend/data/test_automl_diabetes.csv"
TARGET_COLUMN = "class"  # For diabetes
# DATASET_PATH = "/Users/saaivigneshp/Desktop/dv-backend/data/test_automl_weather.csv"
# TARGET_COLUMN = "RainTomorrow"  # For weather

print("="*70)
print("🧪 AutoML Simple Test")
print("="*70)

# Step 1: Check services
print("\n🔍 Checking services...")
try:
    r1 = requests.get(f"{BACKEND_URL}/health", timeout=5)
    r2 = requests.get(f"{ML_PIPELINE_URL}/health", timeout=5)
    print("  ✅ Both services running")
except:
    print("  ❌ Services not running. Please start:")
    print("     Terminal 1: python main.py")
    print("     Terminal 2: cd services/ml_pipeline_service && ./start.sh")
    exit(1)

# Step 2: Upload dataset
print(f"\n📤 Uploading dataset...")
print(f"  File: {Path(DATASET_PATH).name}")

with open(DATASET_PATH, 'rb') as f:
    files = {'file': f}
    data = {'name': f'automl_test_{int(time.time())}', 'domain': 'tabular'}

    response = requests.post(f"{BACKEND_URL}/api/datasets", files=files, data=data)

    if response.status_code not in [200, 201]:
        print(f"  ❌ Upload failed: {response.status_code}")
        print(response.text)
        exit(1)

    dataset = response.json()
    dataset_id = dataset['id']
    print(f"  ✅ Dataset uploaded: {dataset_id}")

# Step 3: Start AutoML training
print(f"\n🚀 Starting AutoML training...")
print(f"  Target column: {TARGET_COLUMN}")

payload = {
    "dataset_id": dataset_id,
    "target_column": TARGET_COLUMN,
    "model_name": f"test_model_{int(time.time())}"
}

response = requests.post(f"{BACKEND_URL}/api/jobs/train-automl", json=payload, timeout=300)

if response.status_code not in [200, 201]:
    print(f"  ❌ Training failed: {response.status_code}")
    print(response.text)
    exit(1)

job = response.json()
job_id = job['id']
print(f"  ✅ Job created: {job_id}")

# Step 4: Monitor progress
print(f"\n⏱️  Monitoring progress (polling every 10s)...\n")

start_time = time.time()
while True:
    response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}")
    if response.status_code != 200:
        print("  ❌ Failed to get status")
        break

    job = response.json()
    status = job['status']
    progress = job.get('progress', 0)
    current_iter = job.get('current_iteration', 0)

    # Get current stage
    stages = job.get('config', {}).get('pipeline_stages', [])
    current_stage = stages[-1]['name'] if stages else "Starting..."

    # Progress bar
    bar_length = 20
    filled = int(progress / 5)
    bar = "█" * filled + "░" * (bar_length - filled)

    print(f"\r  [{bar}] {progress:.0f}% | Stage {current_iter}/10: {current_stage}".ljust(80), end="", flush=True)

    if status == "completed":
        print("\n")
        elapsed = int(time.time() - start_time)
        print(f"  ✅ Training completed! ({elapsed//60}m {elapsed%60}s)")

        # Get model details
        model_id = job.get('model_id')
        if model_id:
            model_resp = requests.get(f"{BACKEND_URL}/api/models/{model_id}")
            if model_resp.status_code == 200:
                model = model_resp.json()
                print(f"\n📊 Results:")
                print(f"  Model: {model.get('name')}")
                print(f"  Framework: {model.get('framework')}")
                print(f"  Accuracy: {model.get('accuracy', 0):.2f}%")

                metrics = model.get('metrics', {})
                if metrics:
                    print(f"  Precision: {metrics.get('precision', 0):.4f}")
                    print(f"  Recall: {metrics.get('recall', 0):.4f}")
                    print(f"  F1: {metrics.get('f1', 0):.4f}")

        print(f"\n✅ Test PASSED!")
        break

    elif status == "failed":
        print("\n")
        print(f"  ❌ Training failed: {job.get('error', 'Unknown error')}")
        exit(1)

    time.sleep(10)

print("="*70)
