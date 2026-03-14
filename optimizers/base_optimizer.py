"""
Base optimizer interface for Matrix Factorization.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers.
    
    All optimizers must implement the optimize method that updates
    the model parameters (U, V embeddings).
    """
    
    @abstractmethod
    def optimize(self, model, train_ratings: np.ndarray, **kwargs) -> dict:
        """
        Optimize model parameters.
        
        Args:
            model: MFModel instance to optimize
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Dictionary with optimization history (e.g., losses, iterations)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get optimizer name.
        
        Returns:
            Optimizer name string
        """
        pass

