#!/usr/bin/env python3
"""
Setup Test Dataset
Registers the weather dataset in the database for testing
"""
import requests
import sys
from pathlib import Path

BACKEND_URL = "http://localhost:8000"
DATASET_PATH = "/Users/saaivigneshp/Desktop/dv-backend/data/test_automl_weather.csv"

print("="*70)
print("📦 Setting Up Test Dataset for AutoML")
print("="*70)

# Check file exists
dataset_file = Path(DATASET_PATH)
if not dataset_file.exists():
    print(f"\n❌ Dataset file not found: {DATASET_PATH}")
    sys.exit(1)

file_size_mb = dataset_file.stat().st_size / (1024 * 1024)
print(f"\n📁 Dataset File:")
print(f"   Path: {DATASET_PATH}")
print(f"   Size: {file_size_mb:.2f} MB")

# Check backend is running
print(f"\n🔍 Checking backend...")
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    print(f"   ✅ Backend running on {BACKEND_URL}")
except:
    print(f"   ❌ Backend not running")
    print(f"   Start: cd /Users/saaivigneshp/Desktop/dv-backend && python main.py")
    sys.exit(1)

# Upload dataset
print(f"\n📤 Uploading weather dataset to database...")

with open(DATASET_PATH, 'rb') as f:
    files = {'file': f}
    data = {
        'name': 'weather_australia_test',
        'domain': 'tabular',
        'description': 'Weather data from Australia - predict rain tomorrow'
    }

    response = requests.post(
        f"{BACKEND_URL}/api/datasets",
        files=files,
        data=data,
        timeout=120  # Large file, give it time
    )

if response.status_code not in [200, 201]:
    print(f"   ❌ Upload failed: {response.status_code}")
    print(f"   Error: {response.text}")
    sys.exit(1)

dataset = response.json()

print(f"   ✅ Dataset registered successfully!")
print(f"\n📊 Dataset Details:")
print(f"   ID: {dataset['id']}")
print(f"   Name: {dataset['name']}")
print(f"   Domain: {dataset.get('domain', 'N/A')}")
print(f"   Format: {dataset.get('file_format', 'N/A')}")
print(f"   Rows: {dataset.get('total_samples', 'N/A')}")
print(f"   Status: {dataset.get('readiness', 'N/A')}")

print(f"\n✅ Setup Complete!")
print(f"\n💡 Next Steps:")
print(f"   1. The dataset is now in your database")
print(f"   2. Run the test:")
print(f"      python test_with_existing_dataset.py")
print(f"   3. The test will auto-find this dataset and use 'RainTomorrow' as target")

print(f"\n📝 Dataset ID for reference:")
print(f"   {dataset['id']}")

print("="*70)
