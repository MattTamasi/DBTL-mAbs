"""
mAbs Optimization Package

A comprehensive package for antibody formulation optimization using
Bayesian optimization and BoTorch.
"""

__version__ = "1.0.0"
__author__ = "mAbs Research Team"

from .main import main
from .pipeline import BoTorchPipeline, DataProcessor, ModelTrainer, OptimizationEngine

__all__ = [
    "main",
    "BoTorchPipeline", 
    "DataProcessor",
    "ModelTrainer", 
    "OptimizationEngine"
]