"""
MF-SGD method: Matrix Factorization with Stochastic Gradient Descent.
Supports bias terms and content-based initialization for cold start.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from models.mf_model import MFModel
from optimizers.sgd import SGDOptimizer
from core.metrics import evaluate_predictions


class MFSGD:
    """
    MF-SGD method combining Matrix Factorization model with SGD optimizer.
    Optional: bias terms, content-based V/U initialization for cold start.
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 learning_rate: float = 0.01, regularization: float = 0.01,
                 random_seed: int = 42,
                 use_bias: bool = False,
                 X_items: Optional[np.ndarray] = None,
                 X_users: Optional[np.ndarray] = None):
        """
        Initialize MF-SGD method.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            latent_dim: Number of latent factors (k)
            learning_rate: SGD learning rate
            regularization: L2 regularization coefficient
            random_seed: Random seed for reproducibility
            use_bias: If True, use μ, b_u, b_i (set via set_biases before fit)
            X_items: Optional genre matrix (n_items, n_genres) for V content-init
            X_users: Optional user features (n_users, n_features) for U content-init
        """
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.random_seed = random_seed
        
        V_init = None
        U_init = None
        W_item = None
        
        if X_items is not None:
            from core.utils import genre_matrix_to_embedding
            V_init, W_item = genre_matrix_to_embedding(
                X_items, latent_dim, random_seed
            )
        if X_users is not None:
            from core.utils import user_matrix_to_embedding
            U_init = user_matrix_to_embedding(
                X_users, latent_dim, random_seed
            )
        
        self.model = MFModel(
            n_users, n_items, latent_dim,
            random_seed=random_seed,
            use_bias=use_bias,
            V_init=V_init,
            U_init=U_init,
            W_item=W_item
        )
        self.optimizer = SGDOptimizer(learning_rate, regularization)
    
    def fit(self, train_ratings: np.ndarray, n_iterations: int = 100,
            verbose: bool = False) -> Dict:
        """
        Train the model on training data.
        
        Args:
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            n_iterations: Number of SGD iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Compute and set biases if use_bias
        if self.use_bias:
            from core.utils import compute_biases
            mu, b_u, b_i = compute_biases(
                train_ratings, self.n_users, self.n_items
            )
            self.model.set_biases(mu, b_u, b_i)
        
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
    
    def fold_in_new_user(self, user_ratings: np.ndarray,
                        regularization: float = 0.01) -> np.ndarray:
        """
        Compute U embedding for a new user (cold start with 1+ ratings).
        
        Args:
            user_ratings: Array of shape (n_ratings, 2) with [item_id, rating]
            regularization: L2 for closed-form solve
            
        Returns:
            U vector of shape (latent_dim,)
        """
        return self.model.fold_in_new_user(
            user_ratings, self.model.V, regularization
        )
    
    def add_new_item(self, genre_vec: np.ndarray) -> int:
        """
        Add a new item (cold start) using genre embedding.
        Requires content-based init (X_items) at construction.
        
        Args:
            genre_vec: Genre multi-hot vector (n_genres,)
            
        Returns:
            New item index
        """
        return self.model.add_new_item(genre_vec)

