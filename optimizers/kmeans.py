"""
K-Means clustering optimizer for Matrix Factorization refinement.

K-Means is used as a refinement strategy: clustering user/item embeddings
and moving them towards cluster centers.
"""

import numpy as np
from typing import Dict
from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel
from sklearn.cluster import KMeans


class KMeansOptimizer(BaseOptimizer):
    """
    K-Means optimizer for Matrix Factorization refinement.
    
    Uses K-Means clustering to refine embeddings by moving them towards cluster centers.
    """
    
    def __init__(self, n_clusters_users: int = None, n_clusters_items: int = None,
                 learning_rate: float = 0.1, regularization: float = 0.01,
                 max_iter: int = 100):
        """
        Initialize K-Means optimizer.
        
        Args:
            n_clusters_users: Number of clusters for user embeddings (None = auto)
            n_clusters_items: Number of clusters for item embeddings (None = auto)
            learning_rate: Learning rate for moving towards cluster centers
            regularization: L2 regularization coefficient
            max_iter: Maximum iterations for K-Means algorithm
        """
        self.n_clusters_users = n_clusters_users
        self.n_clusters_items = n_clusters_items
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
    
    def optimize(self, model: MFModel, train_ratings: np.ndarray,
                 n_iterations: int = 100, verbose: bool = False) -> Dict:
        """
        Optimize model using K-Means clustering refinement.
        
        Args:
            model: MFModel instance to optimize
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            n_iterations: Number of K-Means refinement iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization history
        """
        n_users = model.n_users
        n_items = model.n_items
        latent_dim = model.latent_dim
        
        user_ids = train_ratings[:, 0].astype(int)
        item_ids = train_ratings[:, 1].astype(int)
        ratings = train_ratings[:, 2]
        
        # Auto-determine number of clusters if not specified
        # Ensure clusters <= number of samples
        if self.n_clusters_users is None:
            self.n_clusters_users = max(2, min(int(np.sqrt(n_users)), n_users))
        else:
            self.n_clusters_users = min(self.n_clusters_users, n_users)
        
        if self.n_clusters_items is None:
            self.n_clusters_items = max(2, min(int(np.sqrt(n_items)), n_items))
        else:
            self.n_clusters_items = min(self.n_clusters_items, n_items)
        
        # Ensure at least 2 clusters (or 1 if only 1 sample)
        self.n_clusters_users = max(1, min(self.n_clusters_users, n_users))
        self.n_clusters_items = max(1, min(self.n_clusters_items, n_items))
        
        history = {
            'losses': [],
            'iterations': []
        }
        
        for iteration in range(n_iterations):
            # Cluster user embeddings
            kmeans_users = KMeans(n_clusters=self.n_clusters_users, 
                                 max_iter=self.max_iter, 
                                 random_state=42 + iteration,
                                 n_init=10)
            user_clusters = kmeans_users.fit_predict(model.U)
            user_centers = kmeans_users.cluster_centers_
            
            # Cluster item embeddings
            kmeans_items = KMeans(n_clusters=self.n_clusters_items,
                                max_iter=self.max_iter,
                                random_state=42 + iteration,
                                n_init=10)
            item_clusters = kmeans_items.fit_predict(model.V)
            item_centers = kmeans_items.cluster_centers_
            
            # Move embeddings towards cluster centers
            for u in range(n_users):
                cluster_id = user_clusters[u]
                center = user_centers[cluster_id]
                # Update: move towards center with learning rate
                model.U[u] = (1 - self.learning_rate) * model.U[u] + self.learning_rate * center
            
            for v in range(n_items):
                cluster_id = item_clusters[v]
                center = item_centers[cluster_id]
                # Update: move towards center with learning rate
                model.V[v] = (1 - self.learning_rate) * model.V[v] + self.learning_rate * center
            
            # Apply regularization
            if self.regularization > 0:
                model.U -= self.regularization * model.U
                model.V -= self.regularization * model.V
            
            # Compute loss for history
            loss = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
            history['losses'].append(loss)
            history['iterations'].append(iteration)
            
            if verbose:
                print(f"K-Means Iteration {iteration}: Loss = {loss:.6f}")
        
        return history
    
    def get_name(self) -> str:
        """Get optimizer name."""
        return "KMeans"

