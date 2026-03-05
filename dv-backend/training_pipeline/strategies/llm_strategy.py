"""
LLM-based Training Strategy
Uses GROQ API to generate and iteratively refine CNN architectures
"""

import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config

from ..base import (BaseTrainingStrategy, ProgressUpdate, TrainingConfig,
                    TrainingResult)
from ..core.llm_training import run_llm_training


class LLMStrategy(BaseTrainingStrategy):
    """
    LLM-powered CNN generation strategy

    Uses GROQ API to generate PyTorch CNN code and train it iteratively.
    Core training logic is extracted from cnn_new.py for better integration.
    """

    def __init__(self):
        super().__init__(name="LLM-powered CNN Generation")

    def validate(self, config: TrainingConfig) -> bool:
        """
        LLM strategy works best for:
        - Vision datasets (currently)
        - Small to medium datasets (< 100k samples)
        - Classification tasks
        """
        # Check if dataset is vision
        if config.dataset_domain not in ['vision']:
            return False

        # Check if task is classification
        if config.task not in ['classification']:
            return False

        # Check if GROQ API key is set
        try:
            app_config = get_config()
            if not app_config.GROQ_API_KEY:
                print("[LLMStrategy] Warning: GROQ_API_KEY not set, LLM strategy may fail")
                return False
        except Exception as e:
            print(f"[LLMStrategy] Warning: Failed to get config: {e}")
            return False

        return True

    def get_default_hyperparameters(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        LLM strategy determines hyperparameters dynamically,
        but we provide defaults that will be used if no override
        """
        return {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 3,
            'optimizer': 'Adam',
            'dropout_rate': 0.2,
            'max_iterations': config.max_iterations,
            'target_accuracy': config.target_accuracy,
        }

    def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> TrainingResult:
        """
        Execute LLM-based training using core training module

        The training will:
        1. Use GROQ API to generate CNN architecture
        2. Train and evaluate the model
        3. Iteratively refine based on accuracy
        4. Report progress via callbacks
        5. Return final metrics and model path
        """
        try:
            print(
                f"[LLMStrategy] Starting training for model {config.model_id}")

            # Report initial progress
            self._report_progress(
                progress_callback,
                ProgressUpdate(
                    iteration=0,
                    total_iterations=config.max_iterations,
                    status='training',
                    message='Starting LLM-based CNN generation'
                )
            )

            # Create progress callback wrapper to convert dict to ProgressUpdate
            def progress_wrapper(progress_dict: dict):
                """Convert dict progress to ProgressUpdate and report"""
                if progress_dict.get('type') == 'progress':
                    self._report_progress(
                        progress_callback,
                        ProgressUpdate(
                            iteration=progress_dict['iteration'],
                            total_iterations=progress_dict['total_iterations'],
                            current_accuracy=progress_dict.get(
                                'current_accuracy'),
                            best_accuracy=progress_dict.get('best_accuracy'),
                            current_loss=progress_dict.get('current_loss'),
                            best_loss=progress_dict.get('best_loss'),
                            precision=progress_dict.get('precision'),
                            recall=progress_dict.get('recall'),
                            f1_score=progress_dict.get('f1_score'),
                            status=progress_dict.get('status', 'training'),
                            message=progress_dict.get('message', '')
                        )
                    )

            # Get GROQ API key from config
            app_config = get_config()

            # Run core LLM training
            result = run_llm_training(
                dataset_path=config.dataset_path,
                model_id=config.model_id,
                groq_api_key=app_config.GROQ_API_KEY,
                max_iterations=config.max_iterations,
                target_accuracy=config.target_accuracy,
                device=config.device,
                resize_to=(224, 224),  # Default for vision
                num_workers=0,  # Safe default
                progress_callback=progress_wrapper
            )

            # Check if training succeeded
            if result['success']:
                print(f"[LLMStrategy] Training completed successfully!")
                print(
                    f"[LLMStrategy] Final accuracy: {result['best_accuracy']:.4f}")
                print(f"[LLMStrategy] Model saved to: {result['model_path']}")

                # Build TrainingResult
                return TrainingResult(
                    success=True,
                    model_path=result['model_path'],
                    final_accuracy=result['best_accuracy'],
                    best_accuracy=result['best_accuracy'],
                    hyperparameters=result['best_config'],
                    metrics=result['metrics'],
                    training_history=result['experiment_history']
                )

            else:
                # Training failed
                error_msg = result.get('error', 'Unknown error')
                print(f"[LLMStrategy] Training failed: {error_msg}")

                return TrainingResult(
                    success=False,
                    error=error_msg,
                    training_history=result.get('experiment_history', [])
                )

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()

            print(f"[LLMStrategy] Training failed with exception: {error_msg}")
            print(error_trace)

            # Report failure
            self._report_progress(
                progress_callback,
                ProgressUpdate(
                    iteration=0,
                    total_iterations=config.max_iterations,
                    status='failed',
                    message=f'Training failed: {error_msg}'
                )
            )

            return TrainingResult(
                success=False,
                error=error_msg,
                error_traceback=error_trace
            )
