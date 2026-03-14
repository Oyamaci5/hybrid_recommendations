"""
PSO-MF method: Matrix Factorization with Particle Swarm Optimization.

STATUS: LOCKED - PSO-MF baseline implementation finalized.
- Global optimization baseline for comparison
- Reuses MF model, data loader, metrics, and evaluation pipeline
- Result logging format matches MF-SGD (loss_curve.csv + summary.json)
- DO NOT modify this implementation.
"""

import numpy as np
from typing import Dict, Tuple
from models.mf_model import MFModel
from optimizers.pso import PSOOptimizer
from core.metrics import evaluate_predictions


class PSOMF:
    """
    PSO-MF method combining Matrix Factorization model with PSO optimizer.
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 swarm_size: int = 30, inertia_weight_max: float = 0.9,
                 inertia_weight_min: float = 0.2, cognitive_coeff: float = 2.0,
                 social_coeff: float = 2.0, regularization: float = 0.01,
                 boundary: float = 1.0, random_seed: int = 42):
        """
        Initialize PSO-MF method.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            latent_dim: Number of latent factors (k)
            swarm_size: Number of particles in swarm
            inertia_weight_max: Maximum PSO inertia weight (wMax)
            inertia_weight_min: Minimum PSO inertia weight (wMin)
            cognitive_coeff: PSO cognitive coefficient (c1)
            social_coeff: PSO social coefficient (c2)
            regularization: L2 regularization coefficient
            boundary: Boundary constraint for embeddings
            random_seed: Random seed for reproducibility
        """
        self.model = MFModel(n_users, n_items, latent_dim, random_seed)
        self.optimizer = PSOOptimizer(
            swarm_size=swarm_size,
            inertia_weight_max=inertia_weight_max,
            inertia_weight_min=inertia_weight_min,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            regularization=regularization,
            boundary=boundary
        )
        self.random_seed = random_seed
    
    def fit(self, train_ratings: np.ndarray, n_iterations: int = 100,
            verbose: bool = False) -> Dict:
        """
        Train the model on training data using PSO.
        
        Args:
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            n_iterations: Number of PSO iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Optimize
        history = self.optimizer.optimize(
            self.model, train_ratings, n_iterations, verbose
        )
        
        return history
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for given user-item pairs.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices
            
        Returns:
            Predicted ratings array
        """
        return self.model.predict(user_ids, item_ids)
    
    def evaluate(self, test_ratings: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_ratings: Test ratings array of shape (n_ratings, 3)
                         with columns [user_id, item_id, rating]
            
        Returns:
            Tuple of (RMSE, MAE)
        """
        user_ids = test_ratings[:, 0].astype(int)
        item_ids = test_ratings[:, 1].astype(int)
        true_ratings = test_ratings[:, 2]
        
        pred_ratings = self.predict(user_ids, item_ids)
        
        return evaluate_predictions(true_ratings, pred_ratings)
    
    def get_model(self) -> MFModel:
        """Get the underlying MF model."""
        return self.model
