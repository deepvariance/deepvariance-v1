"""
Base Training Strategy Interface
Defines the contract that all training strategies must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Dataset info
    dataset_id: str
    dataset_path: Path
    dataset_domain: str  # vision, tabular, text, audio
    num_classes: int
    num_samples: int

    # Model info
    model_id: str
    model_name: str
    task: str  # classification, regression, clustering, detection

    # Hyperparameters (optional - strategy may determine these)
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None  # Adam, SGD, RMSprop
    dropout_rate: Optional[float] = None

    # Training configuration
    max_iterations: int = 10  # For iterative strategies like LLM
    target_accuracy: float = 1.0  # Target to achieve
    device: str = 'cpu'  # cpu, cuda, mps

    # Platform integration
    job_id: Optional[str] = None

    # Strategy selection
    strategy: str = 'auto'  # llm, native, transfer, auto

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dataset_id': self.dataset_id,
            'dataset_path': str(self.dataset_path),
            'dataset_domain': self.dataset_domain,
            'num_classes': self.num_classes,
            'num_samples': self.num_samples,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'task': self.task,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'dropout_rate': self.dropout_rate,
            'max_iterations': self.max_iterations,
            'target_accuracy': self.target_accuracy,
            'device': self.device,
            'job_id': self.job_id,
            'strategy': self.strategy,
        }


@dataclass
class ProgressUpdate:
    """Progress update during training"""
    iteration: int
    total_iterations: int
    current_accuracy: Optional[float] = None
    best_accuracy: Optional[float] = None
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    hyperparameters: Optional[Dict[str, Any]] = None  # Per-iteration hyperparameters
    status: str = 'training'  # training, completed, failed
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output"""
        return {
            'type': 'progress',
            'iteration': self.iteration,
            'total_iterations': self.total_iterations,
            'current_accuracy': self.current_accuracy,
            'best_accuracy': self.best_accuracy,
            'current_loss': self.current_loss,
            'best_loss': self.best_loss,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'hyperparameters': self.hyperparameters,
            'status': self.status,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'progress_percent': (self.iteration / self.total_iterations) * 100,
        }


@dataclass
class TrainingResult:
    """Final result of training"""
    success: bool
    model_path: Optional[Path] = None
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    best_accuracy: Optional[float] = None

    # Final hyperparameters used (important for LLM strategies)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Training history
    training_history: List[Dict[str, Any]] = field(default_factory=list)

    # Error info (if failed)
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'model_path': str(self.model_path) if self.model_path else None,
            'final_accuracy': self.final_accuracy,
            'final_loss': self.final_loss,
            'best_accuracy': self.best_accuracy,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'error': self.error,
            'error_traceback': self.error_traceback,
        }


class BaseTrainingStrategy(ABC):
    """
    Base class for all training strategies

    Strategies are pluggable training approaches (LLM, native, transfer learning, etc.)
    Each strategy implements its own training logic while adhering to this interface.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def validate(self, config: TrainingConfig) -> bool:
        """
        Validate if this strategy can handle the given configuration

        Args:
            config: Training configuration

        Returns:
            True if this strategy can handle the config, False otherwise
        """
        pass

    @abstractmethod
    def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> TrainingResult:
        """
        Execute training and return results

        Args:
            config: Training configuration
            progress_callback: Optional callback for progress updates

        Returns:
            TrainingResult with final metrics and model path
        """
        pass

    @abstractmethod
    def get_default_hyperparameters(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Get default hyperparameters for this strategy

        Args:
            config: Training configuration

        Returns:
            Dictionary of default hyperparameters
        """
        pass

    def _report_progress(
        self,
        callback: Optional[Callable[[ProgressUpdate], None]],
        progress: ProgressUpdate
    ):
        """Helper to report progress if callback is provided"""
        if callback:
            callback(progress)

    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
