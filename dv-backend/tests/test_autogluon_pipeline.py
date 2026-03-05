#!/usr/bin/env python3
"""
Comprehensive test file for AutoGluon ML Pipeline
Tests end-to-end functionality of the new AutoGluon-based pipeline
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
MAIN_BACKEND_URL = "http://localhost:8000"
ML_SERVICE_URL = "http://localhost:8001"
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_success(message):
    """Print success message"""
    print(f"✓ {message}")


def print_error(message):
    """Print error message"""
    print(f"✗ {message}")


def print_info(message):
    """Print info message"""
    print(f"→ {message}")


def check_service_health():
    """Check if both services are running"""
    print_section("HEALTH CHECK")

    try:
        # Check main backend
        print_info("Checking main backend...")
        response = requests.get(f"{MAIN_BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success(f"Main backend is healthy: {response.json()}")
        else:
            print_error(f"Main backend returned status {response.status_code}")
            return False

        # Check ML pipeline service
        print_info("Checking ML Pipeline service...")
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"ML Pipeline service is healthy: {data}")
            print_info(f"  Pipeline type: {data.get('pipeline', 'unknown')}")
            print_info(f"  Version: {data.get('version', 'unknown')}")
        else:
            print_error(f"ML Pipeline service returned status {response.status_code}")
            return False

        return True

    except requests.exceptions.ConnectionError as e:
        print_error(f"Connection error: {e}")
        print_error("Make sure both services are running:")
        print_info("  Terminal 1: cd /path/to/dv-backend && python main.py")
        print_info("  Terminal 2: cd /path/to/dv-backend/services/ml_pipeline_service && python main.py")
        return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def create_test_dataset_classification():
    """Create a synthetic classification dataset"""
    print_section("CREATE TEST DATASET (Classification)")

    np.random.seed(42)
    n_samples = 1000

    # Generate features
    data = {
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.randint(20000, 150000, n_samples),
        "credit_score": np.random.randint(300, 850, n_samples),
        "debt_to_income": np.random.uniform(0, 1, n_samples),
        "num_accounts": np.random.randint(0, 10, n_samples),
        "years_employed": np.random.randint(0, 40, n_samples),
        "has_mortgage": np.random.choice([0, 1], n_samples),
        "education_level": np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        "employment_type": np.random.choice(['Full-Time', 'Part-Time', 'Self-Employed', 'Unemployed'], n_samples),
    }

    # Generate target (loan default) with some correlation to features
    risk_score = (
        (data["credit_score"] < 600).astype(int) * 0.3 +
        (data["debt_to_income"] > 0.5).astype(int) * 0.3 +
        (data["income"] < 40000).astype(int) * 0.2 +
        np.random.random(n_samples) * 0.2
    )
    data["default"] = (risk_score > 0.5).astype(int)

    df = pd.DataFrame(data)

    # Save to CSV
    TEST_DATA_DIR.mkdir(exist_ok=True)
    csv_path = TEST_DATA_DIR / "test_classification.csv"
    df.to_csv(csv_path, index=False)

    print_success(f"Created classification dataset: {len(df)} rows × {len(df.columns)} columns")
    print_info(f"Saved to: {csv_path}")
    print_info(f"Target column: 'default'")
    print_info(f"Target distribution: {df['default'].value_counts().to_dict()}")

    return csv_path


def create_test_dataset_regression():
    """Create a synthetic regression dataset"""
    print_section("CREATE TEST DATASET (Regression)")

    np.random.seed(42)
    n_samples = 1000

    # Generate features
    data = {
        "square_feet": np.random.randint(800, 4000, n_samples),
        "bedrooms": np.random.randint(1, 6, n_samples),
        "bathrooms": np.random.randint(1, 4, n_samples),
        "age_years": np.random.randint(0, 100, n_samples),
        "lot_size": np.random.randint(1000, 20000, n_samples),
        "garage_spaces": np.random.randint(0, 3, n_samples),
        "has_pool": np.random.choice([0, 1], n_samples),
        "location": np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        "condition": np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
    }

    # Generate target (house price) with correlation to features
    base_price = 50000
    data["price"] = (
        base_price +
        data["square_feet"] * 150 +
        data["bedrooms"] * 20000 +
        data["bathrooms"] * 15000 +
        (100 - data["age_years"]) * 500 +
        data["lot_size"] * 2 +
        data["garage_spaces"] * 10000 +
        data["has_pool"] * 30000 +
        np.random.normal(0, 20000, n_samples)  # Add noise
    )
    data["price"] = np.maximum(data["price"], 50000)  # Ensure positive prices

    df = pd.DataFrame(data)

    # Save to CSV
    TEST_DATA_DIR.mkdir(exist_ok=True)
    csv_path = TEST_DATA_DIR / "test_regression.csv"
    df.to_csv(csv_path, index=False)

    print_success(f"Created regression dataset: {len(df)} rows × {len(df.columns)} columns")
    print_info(f"Saved to: {csv_path}")
    print_info(f"Target column: 'price'")
    print_info(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print_info(f"Price mean: ${df['price'].mean():.2f}")

    return csv_path


def upload_dataset(csv_path, dataset_name):
    """Upload dataset to main backend"""
    print_section(f"UPLOAD DATASET: {dataset_name}")

    # Create ZIP file (datasets must be uploaded as ZIP)
    import zipfile
    zip_path = csv_path.parent / f"{csv_path.stem}.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(csv_path, csv_path.name)

    print_info(f"Created ZIP file: {zip_path}")

    # Upload to main backend
    try:
        with open(zip_path, 'rb') as f:
            files = {'file': (zip_path.name, f, 'application/zip')}
            data = {
                'name': dataset_name,
                'domain': 'tabular',
                'task': 'classification' if 'classification' in dataset_name else 'regression'
            }

            response = requests.post(
                f"{MAIN_BACKEND_URL}/api/datasets",
                files=files,
                data=data,
                timeout=60
            )

        if response.status_code in [200, 201]:
            dataset = response.json()
            dataset_id = dataset['id']
            print_success(f"Dataset uploaded successfully")
            print_info(f"Dataset ID: {dataset_id}")
            print_info(f"Name: {dataset['name']}")
            print_info(f"Readiness: {dataset['readiness']}")
            return dataset_id
        else:
            print_error(f"Upload failed with status {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print_error(f"Upload failed: {e}")
        return None


def create_automl_training_job(dataset_id, target_column, model_name, selected_models=None):
    """Create an AutoML training job"""
    print_section(f"CREATE AUTOML JOB: {model_name}")

    try:
        data = {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "model_name": model_name
        }

        if selected_models:
            data["selected_models"] = selected_models
            print_info(f"Selected models: {', '.join(selected_models)}")

        response = requests.post(
            f"{MAIN_BACKEND_URL}/api/jobs/train-automl",
            json=data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            job = response.json()
            job_id = job['id']
            print_success(f"Training job created successfully")
            print_info(f"Job ID: {job_id}")
            print_info(f"Model ID: {job.get('model_id')}")
            print_info(f"Status: {job['status']}")
            return job_id, job.get('model_id')
        else:
            print_error(f"Job creation failed with status {response.status_code}: {response.text}")
            return None, None

    except Exception as e:
        print_error(f"Job creation failed: {e}")
        return None, None


def monitor_training_job(job_id, timeout=600):
    """Monitor training job progress"""
    print_section(f"MONITOR TRAINING JOB: {job_id}")

    start_time = time.time()
    last_progress = -1
    last_stage = None

    while True:
        try:
            response = requests.get(f"{MAIN_BACKEND_URL}/api/jobs/{job_id}", timeout=10)

            if response.status_code != 200:
                print_error(f"Failed to get job status: {response.status_code}")
                return False

            job = response.json()
            status = job['status']
            progress = job.get('progress', 0)
            config = job.get('config', {})
            current_stage = config.get('current_stage', 'Unknown')

            # Print progress updates
            if progress != last_progress or current_stage != last_stage:
                if current_stage != last_stage:
                    print_info(f"Stage: {current_stage}")
                print_info(f"Progress: {progress:.1f}% - {config.get('stage_message', '')}")
                last_progress = progress
                last_stage = current_stage

            # Check if completed
            if status == "completed":
                result = job.get('result', {})
                print_success(f"Training completed!")
                print_info(f"Duration: {result.get('duration_seconds', 0):.2f}s")
                print_info(f"Best model: {result.get('best_model', 'unknown')}")

                metrics = result.get('metrics', {})
                if metrics:
                    print_info("Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print_info(f"  {key}: {value}")

                return True

            # Check if failed
            if status == "failed":
                print_error(f"Training failed: {job.get('error', 'Unknown error')}")
                return False

            # Check timeout
            if time.time() - start_time > timeout:
                print_error(f"Training timeout after {timeout}s")
                return False

            # Wait before next poll
            time.sleep(5)

        except KeyboardInterrupt:
            print_info("\nMonitoring interrupted by user")
            return False
        except Exception as e:
            print_error(f"Error monitoring job: {e}")
            time.sleep(5)


def get_model_details(model_id):
    """Get model details from database"""
    print_section(f"MODEL DETAILS: {model_id}")

    try:
        response = requests.get(f"{MAIN_BACKEND_URL}/api/models/{model_id}", timeout=10)

        if response.status_code == 200:
            model = response.json()
            print_success("Model details retrieved")
            print_info(f"Name: {model['name']}")
            print_info(f"Framework: {model.get('framework', 'unknown')}")
            print_info(f"Status: {model['status']}")
            print_info(f"Accuracy: {model.get('accuracy', 'N/A')}")

            metrics = model.get('metrics', {})
            if metrics:
                print_info("Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print_info(f"  {key}: {value}")

            config = model.get('config', {})
            if config:
                print_info(f"Best model type: {config.get('best_model', 'unknown')}")
                print_info(f"Problem type: {config.get('problem_type', 'unknown')}")
                print_info(f"Models trained: {config.get('num_models_trained', 'unknown')}")

            return True
        else:
            print_error(f"Failed to get model: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Error getting model details: {e}")
        return False


def run_full_test():
    """Run full end-to-end test"""
    print(f"\n{'#'*70}")
    print(f"#  AutoGluon ML Pipeline - End-to-End Test")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # Step 1: Health check
    if not check_service_health():
        print_error("\nServices are not healthy. Exiting.")
        return False

    # Step 2: Test Classification
    print_section("TEST 1: CLASSIFICATION TASK")

    csv_path_class = create_test_dataset_classification()
    dataset_id_class = upload_dataset(csv_path_class, "test_loan_default")

    if not dataset_id_class:
        print_error("Classification dataset upload failed")
        return False

    # Test with automatic model selection
    job_id_class, model_id_class = create_automl_training_job(
        dataset_id=dataset_id_class,
        target_column="default",
        model_name="loan_default_predictor_auto"
    )

    if not job_id_class:
        print_error("Classification job creation failed")
        return False

    if not monitor_training_job(job_id_class, timeout=600):
        print_error("Classification training failed")
        return False

    if not get_model_details(model_id_class):
        print_error("Failed to get classification model details")
        return False

    # Step 3: Test Regression
    print_section("TEST 2: REGRESSION TASK")

    csv_path_reg = create_test_dataset_regression()
    dataset_id_reg = upload_dataset(csv_path_reg, "test_house_prices")

    if not dataset_id_reg:
        print_error("Regression dataset upload failed")
        return False

    # Test with selected models
    job_id_reg, model_id_reg = create_automl_training_job(
        dataset_id=dataset_id_reg,
        target_column="price",
        model_name="house_price_predictor_selected",
        selected_models=['GBM', 'CAT', 'XGB', 'LR']
    )

    if not job_id_reg:
        print_error("Regression job creation failed")
        return False

    if not monitor_training_job(job_id_reg, timeout=600):
        print_error("Regression training failed")
        return False

    if not get_model_details(model_id_reg):
        print_error("Failed to get regression model details")
        return False

    # Final summary
    print_section("TEST SUMMARY")
    print_success("All tests completed successfully!")
    print_info(f"Classification Model ID: {model_id_class}")
    print_info(f"Regression Model ID: {model_id_reg}")

    return True


if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
