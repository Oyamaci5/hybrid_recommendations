"""
Stochastic Gradient Descent optimizer for Matrix Factorization.
"""

import numpy as np
from typing import Dict
from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class SGDOptimizer(BaseOptimizer):
    """
    SGD optimizer for Matrix Factorization.
    
    Updates user and item embeddings using gradient descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, regularization: float = 0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate (alpha)
            regularization: L2 regularization coefficient (lambda)
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
    
    def optimize(self, model: MFModel, train_ratings: np.ndarray, 
                 n_iterations: int = 100, verbose: bool = False) -> Dict:
        """
        Optimize model using SGD.
        
        Args:
            model: MFModel instance to optimize
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            n_iterations: Number of SGD iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization history
        """
        n_ratings = len(train_ratings)
        user_ids = train_ratings[:, 0].astype(int)
        item_ids = train_ratings[:, 1].astype(int)
        ratings = train_ratings[:, 2]
        
        history = {
            'losses': [],
            'iterations': []
        }
        
        for iteration in range(n_iterations):
            # Shuffle training data
            indices = np.random.permutation(n_ratings)
            shuffled_users = user_ids[indices]
            shuffled_items = item_ids[indices]
            shuffled_ratings = ratings[indices]
            
            # SGD updates
            for i in range(n_ratings):
                u = shuffled_users[i]
                v = shuffled_items[i]
                r = shuffled_ratings[i]
                
                # Prediction (with bias if model.use_bias)
                pred = np.dot(model.U[u], model.V[v])
                if model.use_bias:
                    pred += model.mu + model.b_u[u] + model.b_i[v]
                
                # Error
                error = r - pred
                
                # Gradients
                grad_U_u = -error * model.V[v] + self.regularization * model.U[u]
                grad_V_v = -error * model.U[u] + self.regularization * model.V[v]
                
                # Update embeddings
                model.U[u] -= self.learning_rate * grad_U_u
                model.V[v] -= self.learning_rate * grad_V_v
            
            # Compute loss for history (always collect, print only if verbose)
            loss = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
            history['losses'].append(loss)
            history['iterations'].append(iteration)
            
            if verbose:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        return history
    
    def get_name(self) -> str:
        """Get optimizer name."""
        return "SGD"

