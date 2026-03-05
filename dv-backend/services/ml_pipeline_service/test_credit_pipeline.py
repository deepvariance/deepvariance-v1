#!/usr/bin/env python3
"""
Test ML Pipeline with Credit Default Dataset

Tests the complete ML pipeline with the credit_default_150k_encoded.csv dataset,
displaying comprehensive metrics and run statistics.

Usage:
    python test_credit_pipeline.py                    # Uses default 25% sampling
    python test_credit_pipeline.py --sample 10        # Uses 10% sampling
    python test_credit_pipeline.py --sample 50        # Uses 50% sampling
    python test_credit_pipeline.py --no-sample        # Uses full dataset (no sampling)
"""

import os
import sys
import time
import json
import platform
import psutil
import traceback
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Add ml_pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_pipeline.pipeline import run_pipeline


class RunStats:
    """Track and display comprehensive run statistics."""

    def __init__(self):
        self.t0 = time.time()
        self.proc = psutil.Process(os.getpid())
        self.start_memory = self.proc.memory_info().rss / 1024**2  # MB
        self.start_cpu_times = self.proc.cpu_times()

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.proc.memory_info().rss / 1024**2

    def get_memory_delta(self):
        """Get memory increase since start in MB."""
        return self.get_memory_usage() - self.start_memory

    def get_elapsed_time(self):
        """Get elapsed time in seconds."""
        return time.time() - self.t0

    def get_cpu_percent(self):
        """Get CPU usage percentage."""
        return self.proc.cpu_percent(interval=0.1)

    def print_summary(self):
        """Print comprehensive run statistics."""
        elapsed = self.get_elapsed_time()
        memory_used = self.get_memory_delta()
        current_memory = self.get_memory_usage()

        print("\n" + "="*80)
        print("📊 RUN STATISTICS SUMMARY")
        print("="*80)

        # System Information
        print("\n🖥️  SYSTEM INFORMATION:")
        print(f"   Platform: {platform.system()} {platform.release()}")
        print(f"   Python: {platform.python_version()}")
        print(f"   CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        print(f"   Total RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")

        # Timing Information
        print("\n⏱️  TIMING:")
        print(f"   Total Elapsed Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"   Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.t0))}")
        print(f"   End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

        # Memory Information
        print("\n💾 MEMORY USAGE:")
        print(f"   Initial Memory: {self.start_memory:.2f} MB")
        print(f"   Current Memory: {current_memory:.2f} MB")
        print(f"   Memory Increase: {memory_used:.2f} MB")
        print(f"   Peak Memory: {self.proc.memory_info().rss / 1024**2:.2f} MB")

        # CPU Information
        cpu_times = self.proc.cpu_times()
        print("\n🔧 CPU USAGE:")
        print(f"   User CPU Time: {cpu_times.user:.2f} seconds")
        print(f"   System CPU Time: {cpu_times.system:.2f} seconds")
        print(f"   Total CPU Time: {cpu_times.user + cpu_times.system:.2f} seconds")
        print(f"   CPU Utilization: {(cpu_times.user + cpu_times.system) / elapsed * 100:.1f}%")

        print("\n" + "="*80)


def print_metrics_summary(metrics: dict, model: any):
    """Print formatted metrics summary."""
    print("\n" + "="*80)
    print("🎯 MODEL PERFORMANCE METRICS")
    print("="*80)

    if not metrics:
        print("❌ No metrics available")
        return

    # Detect problem type
    is_classification = 'accuracy' in metrics or 'precision' in metrics
    is_regression = 'mse' in metrics or 'rmse' in metrics

    if is_classification:
        print("\n📈 CLASSIFICATION METRICS:")
        print(f"   Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else f"   Accuracy:  {metrics.get('accuracy', 'N/A')}")
        print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else f"   Precision: {metrics.get('precision', 'N/A')}")
        print(f"   Recall:    {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), (int, float)) else f"   Recall:    {metrics.get('recall', 'N/A')}")
        print(f"   F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), (int, float)) else f"   F1 Score:  {metrics.get('f1_score', 'N/A')}")

        if 'auc' in metrics and metrics['auc'] is not None:
            print(f"   AUC:       {metrics['auc']:.4f}")

        if 'loss' in metrics and metrics['loss'] is not None:
            print(f"   Loss:      {metrics['loss']:.4f}")

    elif is_regression:
        print("\n📉 REGRESSION METRICS:")
        print(f"   MSE:  {metrics.get('mse', 'N/A'):.4f}" if isinstance(metrics.get('mse'), (int, float)) else f"   MSE:  {metrics.get('mse', 'N/A')}")
        print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else f"   RMSE: {metrics.get('rmse', 'N/A')}")
        print(f"   MAE:  {metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else f"   MAE:  {metrics.get('mae', 'N/A')}")
        print(f"   R²:   {metrics.get('r2', 'N/A'):.4f}" if isinstance(metrics.get('r2'), (int, float)) else f"   R²:   {metrics.get('r2', 'N/A')}")

    # Sample size
    if 'sample_size' in metrics:
        print(f"\n📊 Test Sample Size: {metrics['sample_size']}")

    # Confusion Matrix (if available)
    if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
        print("\n🔢 CONFUSION MATRIX:")
        cm = metrics['confusion_matrix']
        if isinstance(cm, list):
            for i, row in enumerate(cm):
                print(f"   {row}")

    # Model information
    if model:
        print(f"\n🤖 MODEL INFORMATION:")
        print(f"   Model Type: {type(model).__name__}")
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                print(f"   Parameters: {len(params)} configured")
            except:
                pass

    # Additional metrics
    other_metrics = {k: v for k, v in metrics.items()
                    if k not in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'loss',
                                'mse', 'rmse', 'mae', 'r2', 'sample_size', 'confusion_matrix']}

    if other_metrics:
        print("\n📋 ADDITIONAL METRICS:")
        for key, value in other_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    print("\n" + "="*80)


def save_results(metrics: dict, run_stats: RunStats, output_file: str = "test_results.json"):
    """Save test results to JSON file."""
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": "credit_default_150k_encoded.csv",
        "metrics": metrics,
        "run_stats": {
            "elapsed_time_seconds": run_stats.get_elapsed_time(),
            "memory_used_mb": run_stats.get_memory_delta(),
            "current_memory_mb": run_stats.get_memory_usage(),
            "platform": platform.system(),
            "python_version": platform.python_version()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test ML Pipeline with Credit Default Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Use default 25%% sampling
  %(prog)s --sample 10        Use 10%% sampling
  %(prog)s --sample 50        Use 50%% sampling
  %(prog)s --no-sample        Use full dataset (no sampling)
        """
    )

    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        '--sample', '-s',
        type=float,
        metavar='PERCENT',
        help='Percentage of data to sample (0-100). Default: 25.0'
    )
    sampling_group.add_argument(
        '--no-sample',
        action='store_true',
        help='Use full dataset without sampling'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default="/Users/saaivigneshp/Desktop/research/credit_default_150k_encoded.csv",
        help='Path to the dataset CSV file'
    )

    parser.add_argument(
        '--target',
        type=str,
        default="default",
        help='Name of the target column. Default: default'
    )

    return parser.parse_args()


def main():
    """Main test function."""
    # Parse command line arguments
    args = parse_args()

    print("\n" + "="*80)
    print("🧪 ML PIPELINE TEST - CREDIT DEFAULT DATASET")
    print("="*80)

    # Initialize run statistics
    run_stats = RunStats()

    # Configuration from arguments
    dataset_path = args.dataset
    target_column = args.target

    # Determine sampling percentage
    if args.no_sample:
        sample_percentage = None
    elif args.sample is not None:
        sample_percentage = args.sample
        # Validate percentage range
        if not (0 < sample_percentage <= 100):
            print(f"\n❌ ERROR: Sample percentage must be between 0 and 100, got {sample_percentage}")
            sys.exit(1)
    else:
        # Default: 25%
        sample_percentage = 25.0

    print(f"\n📁 Dataset: {dataset_path}")
    print(f"🎯 Target Column: {target_column}")
    if sample_percentage is not None:
        print(f"🎲 Sampling: {sample_percentage}% of data")
    else:
        print(f"🎲 Sampling: Disabled (using full dataset)")

    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    try:
        # Load dataset
        print("\n📥 Loading dataset...")
        load_start = time.time()
        df = pd.read_csv(dataset_path)
        load_time = time.time() - load_start

        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        print(f"   Shape: {df.shape} (rows: {df.shape[0]:,}, columns: {df.shape[1]})")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Display data info
        print("\n📊 Dataset Information:")
        print(f"   Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
        print(f"   Missing values: {df.isnull().sum().sum()}")

        if target_column in df.columns:
            print(f"\n🎯 Target Variable '{target_column}':")
            print(f"   Type: {df[target_column].dtype}")
            print(f"   Unique values: {df[target_column].nunique()}")
            print(f"   Distribution:\n{df[target_column].value_counts()}")
        else:
            print(f"\n⚠️  WARNING: Target column '{target_column}' not found in dataset")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)

        # Run pipeline
        print("\n" + "="*80)
        print("�� STARTING ML PIPELINE")
        print("="*80 + "\n")

        pipeline_start = time.time()

        # Run with status callback
        stage_times = {}
        current_stage = {"name": None, "start": None}

        def status_callback(stage_name: str, status: str):
            """Track stage timing."""
            if status == "start":
                current_stage["name"] = stage_name
                current_stage["start"] = time.time()
                print(f"\n▶️  {stage_name} - Starting...")
            elif status == "complete":
                if current_stage["start"]:
                    elapsed = time.time() - current_stage["start"]
                    stage_times[stage_name] = elapsed
                    print(f"✅ {stage_name} - Complete ({elapsed:.2f}s)")
            elif status == "error":
                print(f"❌ {stage_name} - Failed")

        metrics, model = run_pipeline(
            df,
            target_field=target_column,
            status_callback=status_callback,
            sample_percentage=sample_percentage
        )

        pipeline_time = time.time() - pipeline_start

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\n⏱️  Total Pipeline Time: {pipeline_time:.2f} seconds ({pipeline_time/60:.2f} minutes)")

        # Display stage times
        if stage_times:
            print("\n📊 STAGE TIMING BREAKDOWN:")
            for stage, duration in stage_times.items():
                percentage = (duration / pipeline_time * 100) if pipeline_time > 0 else 0
                print(f"   {stage:.<40} {duration:>8.2f}s ({percentage:>5.1f}%)")

        # Display metrics
        print_metrics_summary(metrics, model)

        # Display run stats
        run_stats.print_summary()

        # Save results
        output_file = f"test_results_{int(time.time())}.json"
        save_results(metrics, run_stats, output_file)

        print("\n" + "="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\n📋 Stack Trace:")
        print(traceback.format_exc())

        # Still print run stats even on failure
        run_stats.print_summary()

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
