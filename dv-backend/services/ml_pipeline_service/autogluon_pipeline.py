"""
AutoGluon-based ML Pipeline for DeepVariance
Production-ready implementation based on research team's stable version.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# AutoGluon imports
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    TabularPredictor = None


class AutoGluonPipeline:
    """
    AutoGluon-based ML Pipeline for tabular data.

    Simplified, production-ready implementation that:
    - Uses AutoGluon for automatic model selection and training
    - Supports both classification and regression tasks
    - Provides detailed metrics and leaderboard
    - Tracks progress and resource usage
    """

    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        model_output_path: str,
        test_size: float = 0.2,
        time_limit: Optional[int] = None,
        preset: str = "medium_quality",
        selected_models: Optional[List[str]] = None
    ):
        """
        Initialize the AutoGluon pipeline.

        Args:
            dataset_path: Path to dataset file (CSV or Parquet)
            target_column: Name of target column
            model_output_path: Directory to save trained models
            test_size: Fraction of data for testing (default: 0.2)
            time_limit: Time limit in seconds for training (None = no limit)
            preset: AutoGluon preset ('best_quality', 'high_quality', 'good_quality', 'medium_quality', 'fast')
            selected_models: Optional list of model codes to train (e.g., ['GBM', 'CAT', 'XGB'])
        """
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError(
                "AutoGluon is not installed. Please install: pip install autogluon")

        self.dataset_path = dataset_path
        self.target_column = target_column
        self.model_output_path = model_output_path
        self.test_size = test_size
        self.time_limit = time_limit
        self.preset = preset
        self.selected_models = selected_models

        self.predictor = None
        self.train_data = None
        self.test_data = None
        self.start_time = None
        self.end_time = None

    def load_data(self) -> pd.DataFrame:
        """Load dataset (CSV or Parquet)."""
        print(f"\n{'='*60}")
        print(f"[LOAD] Data Loading Stage")
        print(f"{'='*60}")
        print(f"[LOAD] Dataset path: {self.dataset_path}")
        print(f"[LOAD] Checking file existence...")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Detect file format
        file_ext = Path(self.dataset_path).suffix.lower()
        file_format = 'Parquet' if file_ext in ['.parquet', '.pq'] else 'CSV'

        print(f"[LOAD] ✓ File found, reading {file_format}...")
        load_start = time.time()

        if file_ext in ['.parquet', '.pq']:
            df = pd.read_parquet(self.dataset_path)
        else:
            df = pd.read_csv(self.dataset_path)

        load_duration = time.time() - load_start

        print(
            f"[LOAD] ✓ Loaded {len(df)} rows × {len(df.columns)} columns in {load_duration:.2f}s")
        print(
            f"[LOAD] Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
        print(
            f"[LOAD] Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Validate target column
        print(f"[LOAD] Validating target column '{self.target_column}'...")
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        print(f"[LOAD] ✓ Target column '{self.target_column}' found")
        print(f"[LOAD] Target data type: {df[self.target_column].dtype}")
        print(
            f"[LOAD] Missing values in dataset: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%)")

        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets."""
        print(f"\n{'='*60}")
        print(f"[SPLIT] Data Splitting Stage")
        print(f"{'='*60}")
        print(
            f"[SPLIT] Total dataset size: {len(df)} rows × {len(df.columns)} columns")
        print(
            f"[SPLIT] Split ratio: {1-self.test_size:.0%} train, {self.test_size:.0%} test")

        # Check if target column exists for stratification
        target_col = self.target_column
        stratify = None

        if target_col in df.columns:
            # Check if classification task (for stratification)
            n_unique = df[target_col].nunique()
            print(
                f"[SPLIT] Target column '{target_col}' has {n_unique} unique values")

            if n_unique < 100:  # Likely classification
                stratify = df[target_col]
                print(f"[SPLIT] Using stratified split for classification")

        print(f"[SPLIT] Performing train/test split...")
        train_data, test_data = train_test_split(
            df,
            test_size=self.test_size,
            random_state=42,
            stratify=stratify
        )

        print(f"[SPLIT] ✓ Split completed:")
        print(
            f"[SPLIT]   - Train set: {len(train_data)} rows ({len(train_data)/len(df)*100:.1f}%)")
        print(
            f"[SPLIT]   - Test set: {len(test_data)} rows ({len(test_data)/len(df)*100:.1f}%)")

        if target_col in df.columns:
            print(f"[SPLIT] Target distribution:")
            print(
                f"[SPLIT]   - Train: {dict(train_data[target_col].value_counts().head(5))}")
            print(
                f"[SPLIT]   - Test: {dict(test_data[target_col].value_counts().head(5))}")

        self.train_data = train_data
        self.test_data = test_data

        return train_data, test_data

    def train(self, progress_callback=None) -> Dict[str, Any]:
        """
        Train AutoGluon models.

        Args:
            progress_callback: Optional callback function(stage, progress, message)

        Returns:
            Dictionary with training results
        """
        self.start_time = time.time()
        results = {
            "status": "running",
            "stages": [],
            "metrics": {},
            "errors": []
        }

        try:
            # Stage 1: Load Data
            if progress_callback:
                progress_callback("Data Loading", 10, "Loading dataset...")

            df = self.load_data()
            results["stages"].append({
                "stage": "Data Loading",
                "status": "completed",
                "duration_seconds": round(time.time() - self.start_time, 2)
            })

            # Stage 2: Split Data
            if progress_callback:
                progress_callback("Data Splitting", 20,
                                  "Splitting train/test sets...")

            train_data, test_data = self.split_data(df)
            results["stages"].append({
                "stage": "Data Splitting",
                "status": "completed",
                "train_size": len(train_data),
                "test_size": len(test_data),
                "duration_seconds": round(time.time() - self.start_time, 2)
            })

            # Stage 3: Train Models
            if progress_callback:
                progress_callback("Model Training", 30,
                                  "Training AutoGluon models...")

            print(f"\n{'='*60}")
            print(f"[TRAIN] Model Training Stage")
            print(f"{'='*60}")
            print(f"[TRAIN] Configuration:")
            print(f"[TRAIN]   - Target column: {self.target_column}")
            print(f"[TRAIN]   - Preset quality: {self.preset}")
            print(
                f"[TRAIN]   - Time limit: {self.time_limit if self.time_limit else 'No limit'}{'s' if self.time_limit else ''}")
            print(f"[TRAIN]   - Model output path: {self.model_output_path}")

            # Prepare hyperparameters if specific models selected
            hyperparameters = None
            if self.selected_models:
                hyperparameters = {model: {} for model in self.selected_models}
                print(
                    f"[TRAIN]   - Model selection: Manual ({len(self.selected_models)} models)")
                print(
                    f"[TRAIN]   - Selected models: {', '.join(self.selected_models)}")
            else:
                print(
                    f"[TRAIN]   - Model selection: Automatic (AutoGluon will select best models)")

            print(f"\n[TRAIN] Initializing AutoGluon TabularPredictor...")
            print(f"[TRAIN] Starting training process...")
            print(f"[TRAIN] {'─'*60}")

            # Train predictor
            train_start = time.time()
            self.predictor = TabularPredictor(
                label=self.target_column,
                path=self.model_output_path
            ).fit(
                train_data=train_data,
                time_limit=self.time_limit,
                presets=self.preset,
                hyperparameters=hyperparameters,
                verbosity=2
            )
            train_duration = time.time() - train_start

            print(f"\n[TRAIN] {'─'*60}")
            print(f"[TRAIN] ✓ Training completed successfully")
            print(
                f"[TRAIN] Total training time: {train_duration:.2f}s ({train_duration/60:.1f} minutes)")
            print(
                f"[TRAIN] Models trained: {len(self.predictor.model_names())}")
            print(
                f"[TRAIN] Best model selected: {self.predictor.get_model_best()}")

            results["stages"].append({
                "stage": "Model Training",
                "status": "completed",
                "duration_seconds": round(train_duration, 2)
            })

            # Stage 4: Evaluate Models
            if progress_callback:
                progress_callback("Model Evaluation", 70,
                                  "Evaluating models on test set...")

            print(f"\n{'='*60}")
            print(f"[EVAL] Model Evaluation Stage")
            print(f"{'='*60}")
            print(f"[EVAL] Evaluating all trained models on test set...")
            print(f"[EVAL] Test set size: {len(test_data)} samples")

            eval_start = time.time()

            # Get performance metrics
            print(f"[EVAL] Computing model performance metrics...")
            performance = self.predictor.evaluate(test_data, silent=True)

            print(f"[EVAL] Generating model leaderboard...")
            leaderboard = self.predictor.leaderboard(test_data, silent=True)

            eval_duration = time.time() - eval_start
            print(f"[EVAL] ✓ Evaluation completed in {eval_duration:.2f}s")

            # Get best model info
            best_model = self.predictor.get_model_best()
            problem_type = self.predictor.problem_type.lower()

            print(f"\n[EVAL] Results Summary:")
            print(f"[EVAL]   - Problem type: {problem_type}")
            print(f"[EVAL]   - Best model: {best_model}")
            print(
                f"[EVAL]   - Evaluation metric: {self.predictor.eval_metric.name}")
            print(f"[EVAL]   - Total models evaluated: {len(leaderboard)}")

            # Display top 5 models from leaderboard
            print(f"\n[EVAL] Top 5 Models Leaderboard:")
            for idx, row in leaderboard.head(5).iterrows():
                model_name = row.get('model', 'Unknown')
                score_val = row.get('score_val', row.get('score_test', 'N/A'))
                print(f"[EVAL]   {idx+1}. {model_name}: {score_val}")

            # Compute additional metrics
            print(f"\n[EVAL] Computing detailed metrics...")
            y_true = test_data[self.target_column]
            y_pred = self.predictor.predict(test_data)

            metrics = {
                "best_model": best_model,
                "problem_type": problem_type,
                "eval_metric": self.predictor.eval_metric.name,
                "num_models_trained": len(leaderboard)
            }

            if "class" in problem_type or "binary" in problem_type:
                # Classification metrics
                print(f"[EVAL] Classification task detected - computing metrics...")
                acc = accuracy_score(y_true, y_pred)
                metrics["accuracy"] = round(acc, 4)
                print(f"[EVAL]   - Accuracy: {acc:.4f} ({acc*100:.2f}%)")

                try:
                    f1 = f1_score(y_true, y_pred, average='macro')
                    metrics["f1_macro"] = round(f1, 4)
                    print(f"[EVAL]   - F1 Score (macro): {f1:.4f}")
                except:
                    pass

                # ROC AUC for binary classification
                if len(np.unique(y_true)) == 2:
                    try:
                        y_proba = self.predictor.predict_proba(test_data)
                        if isinstance(y_proba, pd.DataFrame):
                            y_proba = y_proba.iloc[:, 1].values
                        elif len(y_proba.shape) > 1:
                            y_proba = y_proba[:, 1]
                        roc = roc_auc_score(y_true, y_proba)
                        metrics["roc_auc"] = round(roc, 4)
                        print(f"[EVAL]   - ROC AUC: {roc:.4f}")
                    except Exception as e:
                        print(f"[EVAL]   - Could not compute ROC AUC: {e}")

            elif "regression" in problem_type:
                # Regression metrics
                print(f"[EVAL] Regression task detected - computing metrics...")
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                metrics["r2"] = round(r2, 4)
                metrics["mae"] = round(mae, 4)
                metrics["rmse"] = round(rmse, 4)

                print(f"[EVAL]   - R² Score: {r2:.4f}")
                print(f"[EVAL]   - Mean Absolute Error (MAE): {mae:.4f}")
                print(f"[EVAL]   - Root Mean Squared Error (RMSE): {rmse:.4f}")

            results["metrics"] = metrics
            results["leaderboard"] = leaderboard.to_dict('records')

            results["stages"].append({
                "stage": "Model Evaluation",
                "status": "completed",
                "duration_seconds": round(time.time() - eval_start, 2)
            })

            # Print final summary
            total_duration = time.time() - self.start_time
            print(f"\n{'='*60}")
            print(f"[COMPLETE] Pipeline Execution Summary")
            print(f"{'='*60}")
            print(f"[COMPLETE] Status: SUCCESS ✓")
            print(
                f"[COMPLETE] Total execution time: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
            print(f"\n[COMPLETE] Stage Breakdown:")
            for stage_info in results["stages"]:
                stage_name = stage_info.get("stage", "Unknown")
                stage_duration = stage_info.get("duration_seconds", 0)
                stage_pct = (stage_duration / total_duration *
                             100) if total_duration > 0 else 0
                print(
                    f"[COMPLETE]   - {stage_name}: {stage_duration:.2f}s ({stage_pct:.1f}%)")

            print(f"\n[COMPLETE] Final Model:")
            print(f"[COMPLETE]   - Best model: {best_model}")
            print(f"[COMPLETE]   - Problem type: {problem_type}")
            print(f"[COMPLETE]   - Models trained: {len(leaderboard)}")

            if "class" in problem_type or "binary" in problem_type:
                if "accuracy" in metrics:
                    print(
                        f"[COMPLETE]   - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            elif "regression" in problem_type:
                if "r2" in metrics:
                    print(f"[COMPLETE]   - R² Score: {metrics['r2']:.4f}")

            print(f"\n[COMPLETE] Model saved to: {self.model_output_path}")
            print(f"{'='*60}\n")

            # Final progress
            if progress_callback:
                progress_callback(
                    "Completed", 100, f"Training completed. Best model: {best_model}")

            results["status"] = "completed"

        except Exception as e:
            error_msg = str(e)
            print(f"\n[ERROR] Pipeline failed: {error_msg}")
            results["status"] = "failed"
            results["errors"].append(error_msg)

            if progress_callback:
                progress_callback("Failed", 0, f"Error: {error_msg}")

            raise

        finally:
            self.end_time = time.time()
            results["total_duration_seconds"] = round(
                self.end_time - self.start_time, 2)

        return results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.predictor.predict(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification)."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.predictor.predict_proba(data)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the best model."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")

        try:
            return self.predictor.feature_importance(self.test_data)
        except Exception as e:
            print(f"[WARN] Could not compute feature importance: {e}")
            return pd.DataFrame()

    def save_summary(self, output_path: str):
        """Save training summary to JSON file."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": self.dataset_path,
            "target_column": self.target_column,
            "problem_type": self.predictor.problem_type,
            "best_model": self.predictor.get_model_best(),
            "models_trained": self.predictor.model_names(),
            "training_duration_seconds": round(self.end_time - self.start_time, 2) if self.end_time else None
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[SAVE] Summary saved to: {output_path}")


def run_autogluon_pipeline(
    dataset_path: str,
    target_column: str,
    model_output_path: str,
    selected_models: Optional[List[str]] = None,
    time_limit: Optional[int] = None,
    preset: str = "medium_quality",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Convenience function to run the AutoGluon pipeline.

    Args:
        dataset_path: Path to dataset file (CSV or Parquet)
        target_column: Target column name
        model_output_path: Directory to save models
        selected_models: Optional list of models to train (e.g., ['GBM', 'CAT', 'XGB', 'LR', 'NN_TORCH'])
        time_limit: Training time limit in seconds
        preset: AutoGluon preset quality level
        progress_callback: Optional callback(stage, progress, message)

    Returns:
        Training results dictionary
    """
    pipeline = AutoGluonPipeline(
        dataset_path=dataset_path,
        target_column=target_column,
        model_output_path=model_output_path,
        time_limit=time_limit,
        preset=preset,
        selected_models=selected_models
    )

    results = pipeline.train(progress_callback=progress_callback)

    # Save summary
    summary_path = os.path.join(model_output_path, "training_summary.json")
    pipeline.save_summary(summary_path)

    return results
