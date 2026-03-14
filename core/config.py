"""
Configuration management for experiments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for MF experiments."""
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Data paths
    data_dir: str = "data/movielens_100k"
    
    # Model parameters
    latent_dim: int = 10  # Number of latent factors (k)
    
    # SGD parameters
    learning_rate: float = 0.01
    regularization: float = 0.01  # Lambda (L2 regularization)
    n_iterations: int = 100
    
    # Evaluation
    train_split: str = "u1.base"
    test_split: str = "u1.test"
    
    def __post_init__(self):
        """Set random seed after initialization."""
        np.random.seed(self.random_seed)
    
    def get_train_path(self) -> str:
        """Get full path to training file."""
        return f"{self.data_dir}/{self.train_split}"
    
    def get_test_path(self) -> str:
        """Get full path to test file."""
        return f"{self.data_dir}/{self.test_split}"


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)

