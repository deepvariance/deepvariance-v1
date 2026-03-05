"""
Training Strategies
Different approaches to model training
"""

from .llm_strategy import LLMStrategy
from .autogluon_strategy import AutoGluonStrategy

__all__ = ['LLMStrategy', 'AutoGluonStrategy']
