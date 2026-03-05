#!/usr/bin/env python3
"""
Quick test script to verify ML Pipeline Service setup
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("\n🧪 Test 1: Checking Python imports...")

    try:
        import httpx
        print("  ✅ httpx")
    except ImportError as e:
        print(f"  ❌ httpx: {e}")
        return False

    try:
        import pandas
        print("  ✅ pandas")
    except ImportError as e:
        print(f"  ❌ pandas: {e}")
        return False

    try:
        import sklearn
        print("  ✅ scikit-learn")
    except ImportError as e:
        print(f"  ❌ scikit-learn: {e}")
        return False

    try:
        import joblib
        print("  ✅ joblib")
    except ImportError as e:
        print(f"  ❌ joblib: {e}")
        return False

    return True

def test_ml_pipeline_files():
    """Test that ML Pipeline files exist"""
    print("\n🧪 Test 2: Checking ML Pipeline files...")

    service_dir = "services/ml_pipeline_service"
    required_files = [
        f"{service_dir}/app/main.py",
        f"{service_dir}/app/routes.py",
        f"{service_dir}/app/pipeline_adapter.py",
        f"{service_dir}/ml_pipeline/pipeline.py",
        f"{service_dir}/ml_pipeline/llm_agents.py",
        f"{service_dir}/start.sh"
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - NOT FOUND")
            all_exist = False

    return all_exist

def test_main_backend_integration():
    """Test that main backend has AutoML endpoint"""
    print("\n🧪 Test 3: Checking main backend integration...")

    try:
        with open("routers/jobs.py", "r") as f:
            content = f.read()

        if "train-automl" in content:
            print("  ✅ AutoML endpoint found in routers/jobs.py")
        else:
            print("  ❌ AutoML endpoint NOT found in routers/jobs.py")
            return False

        if "AutoMLTrainingRequest" in content:
            print("  ✅ AutoMLTrainingRequest model found")
        else:
            print("  ❌ AutoMLTrainingRequest model NOT found")
            return False

        if "import httpx" in content:
            print("  ✅ httpx import found")
        else:
            print("  ❌ httpx import NOT found")
            return False

        return True
    except Exception as e:
        print(f"  ❌ Error reading jobs.py: {e}")
        return False

def test_environment():
    """Test environment variables"""
    print("\n🧪 Test 4: Checking environment variables...")

    groq_key = os.getenv("GROQ_API_KEY")
    db_url = os.getenv("DATABASE_URL")

    if groq_key:
        masked_key = f"{groq_key[:8]}...{groq_key[-4:]}" if len(groq_key) > 12 else "***"
        print(f"  ✅ GROQ_API_KEY: {masked_key}")
    else:
        print("  ⚠️  GROQ_API_KEY not set (will be loaded from .env)")

    if db_url:
        print(f"  ✅ DATABASE_URL: {db_url[:30]}...")
    else:
        print("  ⚠️  DATABASE_URL not set (will be loaded from .env)")

    # Check .env file exists
    if os.path.exists(".env"):
        print("  ✅ .env file exists")
        return True
    else:
        print("  ❌ .env file NOT found - services may fail to start")
        return False

def test_database_models():
    """Test that database models support AutoML"""
    print("\n🧪 Test 5: Checking database schema compatibility...")

    try:
        # Import database models
        sys.path.insert(0, os.getcwd())
        from db_models import Job, Model

        # Check Job table has required fields
        job_columns = [c.name for c in Job.__table__.columns]

        required_job_fields = ['id', 'job_type', 'status', 'progress',
                               'current_iteration', 'total_iterations', 'config']

        all_present = True
        for field in required_job_fields:
            if field in job_columns:
                print(f"  ✅ Job.{field}")
            else:
                print(f"  ❌ Job.{field} - NOT FOUND")
                all_present = False

        # Check Model table has framework field
        model_columns = [c.name for c in Model.__table__.columns]
        if 'framework' in model_columns:
            print("  ✅ Model.framework")
        else:
            print("  ❌ Model.framework - NOT FOUND")
            all_present = False

        return all_present
    except Exception as e:
        print(f"  ❌ Error checking database models: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("ML Pipeline Integration - Pre-Flight Checks")
    print("="*60)

    results = {
        "Imports": test_imports(),
        "ML Pipeline Files": test_ml_pipeline_files(),
        "Backend Integration": test_main_backend_integration(),
        "Environment": test_environment(),
        "Database Schema": test_database_models()
    }

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All pre-flight checks passed!")
        print("\nNext steps:")
        print("1. Start main backend:")
        print("   python main.py")
        print("\n2. In another terminal, start ML Pipeline service:")
        print("   cd services/ml_pipeline_service && ./start.sh")
        print("\n3. Test health endpoints:")
        print("   curl http://localhost:8000/health")
        print("   curl http://localhost:8001/health")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
