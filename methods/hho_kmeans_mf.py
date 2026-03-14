"""
HHO-KMeans-MF method: Matrix Factorization with Harris Hawks Optimization initialization
followed by K-Means clustering refinement.

Two-phase approach:
1. HHO phase: Global search for good initial embeddings (U, V)
2. K-Means phase: Clustering-based refinement from HHO-initialized embeddings
"""

import numpy as np
from typing import Dict, Tuple
from models.mf_model import MFModel
from optimizers.hho import HHOOptimizer
from optimizers.kmeans import KMeansOptimizer
from core.metrics import evaluate_predictions


class HHOKMeansMF:
    """
    HHO-KMeans-MF method combining HHO initialization with K-Means refinement.
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 n_hawks: int = 30, escape_energy_initial: float = 1.0,
                 n_clusters_users: int = None, n_clusters_items: int = None,
                 learning_rate: float = 0.1, regularization: float = 0.01,
                 boundary: float = 1.0, random_seed: int = 42):
        """
        Initialize HHO-KMeans-MF method.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            latent_dim: Number of latent factors (k)
            n_hawks: Number of hawks in HHO population
            escape_energy_initial: Initial escape energy E0 for HHO
            n_clusters_users: Number of clusters for user embeddings
            n_clusters_items: Number of clusters for item embeddings
            learning_rate: K-Means learning rate
            regularization: L2 regularization coefficient
            boundary: Boundary constraint for embeddings
            random_seed: Random seed for reproducibility
        """
        self.model = MFModel(n_users, n_items, latent_dim, random_seed)
        self.hho_optimizer = HHOOptimizer(
            n_hawks=n_hawks,
            escape_energy_initial=escape_energy_initial,
            regularization=regularization,
            boundary=boundary
        )
        self.kmeans_optimizer = KMeansOptimizer(
            n_clusters_users=n_clusters_users,
            n_clusters_items=n_clusters_items,
            learning_rate=learning_rate,
            regularization=regularization
        )
        self.random_seed = random_seed
    
    def fit(self, train_ratings: np.ndarray, hho_iterations: int = 50,
            kmeans_iterations: int = 100, verbose: bool = False) -> Dict:
        """
        Train the model using HHO initialization followed by K-Means refinement.
        
        Args:
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            hho_iterations: Number of HHO iterations for initialization
            kmeans_iterations: Number of K-Means iterations for refinement
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history (HHO and K-Means phases)
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Phase 1: HHO initialization
        if verbose:
            print("=" * 60)
            print("Phase 1: HHO Initialization")
            print("=" * 60)
        
        hho_history = self.hho_optimizer.optimize(
            self.model, train_ratings, hho_iterations, verbose
        )
        
        # Phase 2: K-Means refinement
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 2: K-Means Refinement")
            print("=" * 60)
        
        kmeans_history = self.kmeans_optimizer.optimize(
            self.model, train_ratings, kmeans_iterations, verbose
        )
        
        # Combine histories
        history = {
            'hho_losses': hho_history['losses'],
            'hho_iterations': hho_history['iterations'],
            'kmeans_losses': kmeans_history['losses'],
            'kmeans_iterations': kmeans_history['iterations']
        }
        
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

