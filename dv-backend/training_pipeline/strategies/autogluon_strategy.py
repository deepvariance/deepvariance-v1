"""
AutoGluon Training Strategy

Implements automated machine learning using AutoGluon's TabularPredictor.
Suitable for tabular datasets with automatic model selection and hyperparameter tuning.
"""

import os
import tempfile
from typing import Any, Dict, Optional

import pandas as pd
from autogluon.tabular import TabularPredictor

from training_pipeline.base import (
    BaseTrainingStrategy,
    ProgressUpdate,
    TrainingConfig,
    TrainingResult,
)


class AutoGluonStrategy(BaseTrainingStrategy):
    """
    Training strategy using AutoGluon for automated machine learning.

    Best suited for:
    - Tabular datasets
    - Classification and regression tasks
    - Automatic model selection and ensembling
    """

    def validate_config(self, config: TrainingConfig) -> bool:
        """
        Validate that the configuration is suitable for AutoGluon training.

        Args:
            config: Training configuration to validate

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # AutoGluon is primarily for tabular data
        if config.dataset.domain not in ['tabular', 'general']:
            raise ValueError(
                f"AutoGluon strategy requires tabular dataset, got: {config.dataset.domain}"
            )

        # Check for required target column
        if not config.target_column:
            raise ValueError("AutoGluon strategy requires a target column")

        # AutoGluon supports classification and regression
        if config.model.task_type not in ['classification', 'regression']:
            raise ValueError(
                f"AutoGluon supports classification/regression, got: {config.model.task_type}"
            )

        return True

    def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[callable] = None
    ) -> TrainingResult:
        """
        Train models using AutoGluon's TabularPredictor.

        Args:
            config: Training configuration
            progress_callback: Optional callback for progress updates

        Returns:
            TrainingResult with metrics and trained predictor
        """
        self.validate_config(config)

        # Send initial progress
        if progress_callback:
            progress_callback(ProgressUpdate(
                status='running',
                progress=0.0,
                current_iteration=0,
                total_iterations=1,
                message='Loading dataset for AutoGluon training...'
            ))

        # Load dataset
        df = self._load_dataset(config)

        if progress_callback:
            progress_callback(ProgressUpdate(
                status='running',
                progress=0.2,
                current_iteration=0,
                total_iterations=1,
                message=f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns'
            ))

        # Validate target column exists
        if config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{config.target_column}' not found in dataset"
            )

        # Split dataset
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        if progress_callback:
            progress_callback(ProgressUpdate(
                status='running',
                progress=0.3,
                current_iteration=0,
                total_iterations=1,
                message=f'Train/test split: {len(train_data)}/{len(test_data)} samples'
            ))

        # Prepare hyperparameters
        hyperparameters = self._prepare_hyperparameters(config)

        # Determine problem type
        problem_type = self._determine_problem_type(config, df[config.target_column])

        # Create temporary directory for AutoGluon models
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, f'ag_model_{config.model.id}')

            if progress_callback:
                progress_callback(ProgressUpdate(
                    status='running',
                    progress=0.4,
                    current_iteration=0,
                    total_iterations=1,
                    message=f'Starting AutoGluon training with {problem_type}...'
                ))

            # Train predictor
            predictor = TabularPredictor(
                label=config.target_column,
                path=model_path,
                problem_type=problem_type
            ).fit(
                train_data=train_data,
                hyperparameters=hyperparameters,
                verbosity=0  # Silent mode for cleaner logs
            )

            if progress_callback:
                progress_callback(ProgressUpdate(
                    status='running',
                    progress=0.8,
                    current_iteration=0,
                    total_iterations=1,
                    message='Evaluating model performance...'
                ))

            # Evaluate on test set
            performance = predictor.evaluate(test_data, silent=True)
            leaderboard = predictor.leaderboard(test_data, silent=True)

            # Extract metrics based on problem type
            metrics = self._extract_metrics(performance, problem_type, predictor, test_data)

            if progress_callback:
                progress_callback(ProgressUpdate(
                    status='running',
                    progress=1.0,
                    current_iteration=1,
                    total_iterations=1,
                    message='Training complete!',
                    metrics=metrics
                ))

            # Prepare result
            result = TrainingResult(
                model_id=config.model.id,
                metrics=metrics,
                best_model=predictor.info()['best_model'],
                trained_model=predictor,
                additional_info={
                    'leaderboard': leaderboard.to_dict('records'),
                    'problem_type': problem_type,
                    'num_models_trained': len(leaderboard),
                    'feature_importance': predictor.feature_importance(test_data).to_dict() if hasattr(predictor, 'feature_importance') else {}
                }
            )

            return result

    def _load_dataset(self, config: TrainingConfig) -> pd.DataFrame:
        """Load dataset from file path."""
        file_path = config.dataset.file_path

        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def _prepare_hyperparameters(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Prepare hyperparameters for AutoGluon.

        If no hyperparameters are specified, uses default models (GBM, CAT, XGB).
        """
        if config.hyperparameters:
            # Use custom hyperparameters if provided
            return config.hyperparameters

        # Default: Use ensemble of gradient boosting models
        return {
            'GBM': {},
            'CAT': {},
            'XGB': {}
        }

    def _determine_problem_type(self, config: TrainingConfig, target_series: pd.Series) -> str:
        """
        Determine AutoGluon problem type from config and target variable.
        """
        if config.model.task_type == 'classification':
            # Check if binary or multiclass
            n_classes = target_series.nunique()
            return 'binary' if n_classes == 2 else 'multiclass'
        elif config.model.task_type == 'regression':
            return 'regression'
        else:
            # Let AutoGluon auto-detect
            return None

    def _extract_metrics(
        self,
        performance: Dict[str, float],
        problem_type: str,
        predictor: TabularPredictor,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract and normalize metrics from AutoGluon evaluation.
        """
        metrics = {}

        if problem_type in ['binary', 'multiclass']:
            # Classification metrics
            metrics['accuracy'] = float(performance.get('accuracy', 0.0))
            metrics['f1_score'] = float(performance.get('f1', 0.0))
            metrics['precision'] = float(performance.get('precision', 0.0))
            metrics['recall'] = float(performance.get('recall', 0.0))

            # Try to get AUC if available
            if 'roc_auc' in performance:
                metrics['auc'] = float(performance['roc_auc'])

            # Loss (negative log loss)
            if 'log_loss' in performance:
                metrics['loss'] = float(performance['log_loss'])

        elif problem_type == 'regression':
            # Regression metrics
            metrics['rmse'] = float(performance.get('root_mean_squared_error', 0.0))
            metrics['mae'] = float(performance.get('mean_absolute_error', 0.0))
            metrics['r2'] = float(performance.get('r2', 0.0))
            metrics['loss'] = metrics['rmse']  # Use RMSE as loss

        # Add sample size
        metrics['sample_size'] = len(test_data)

        return metrics
