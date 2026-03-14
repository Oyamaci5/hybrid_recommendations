"""
HHO-SGD-MF method: Matrix Factorization with Harris Hawks Optimization initialization
followed by Stochastic Gradient Descent refinement.
Supports bias terms and content-based initialization for cold start.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from models.mf_model import MFModel
from optimizers.hho import HHOOptimizer
from optimizers.sgd import SGDOptimizer
from core.metrics import evaluate_predictions


class HHOSGDMF:
    """
    HHO-SGD-MF method combining HHO initialization with SGD refinement.
    Optional: bias terms, content-based V/U initialization for cold start.
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 n_hawks: int = 30, escape_energy_initial: float = 1.0,
                 learning_rate: float = 0.01, regularization: float = 0.01,
                 boundary: float = 1.0, random_seed: int = 42,
                 use_bias: bool = False,
                 X_items: Optional[np.ndarray] = None,
                 X_users: Optional[np.ndarray] = None):
        """
        Initialize HHO-SGD-MF method.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            latent_dim: Number of latent factors (k)
            n_hawks: Number of hawks in HHO population
            escape_energy_initial: Initial escape energy E0 for HHO
            learning_rate: SGD learning rate
            regularization: L2 regularization coefficient
            boundary: Boundary constraint for embeddings
            random_seed: Random seed for reproducibility
            use_bias: If True, use μ, b_u, b_i
            X_items: Optional genre matrix for V content-init
            X_users: Optional user features for U content-init
        """
        self.n_users = n_users
        self.n_items = n_items
        self.use_bias = use_bias
        self.random_seed = random_seed
        
        V_init = None
        U_init = None
        W_item = None
        if X_items is not None:
            from core.utils import genre_matrix_to_embedding
            V_init, W_item = genre_matrix_to_embedding(X_items, latent_dim, random_seed)
        if X_users is not None:
            from core.utils import user_matrix_to_embedding
            U_init = user_matrix_to_embedding(X_users, latent_dim, random_seed)
        
        self.model = MFModel(
            n_users, n_items, latent_dim,
            random_seed=random_seed,
            use_bias=use_bias,
            V_init=V_init,
            U_init=U_init,
            W_item=W_item
        )
        self.hho_optimizer = HHOOptimizer(
            n_hawks=n_hawks,
            escape_energy_initial=escape_energy_initial,
            regularization=regularization,
            boundary=boundary
        )
        self.sgd_optimizer = SGDOptimizer(learning_rate, regularization)
        self.random_seed = random_seed
    
    def fit(self, train_ratings: np.ndarray, hho_iterations: int = 50,
            sgd_iterations: int = 100, verbose: bool = False) -> Dict:
        """
        Train the model using HHO initialization followed by SGD refinement.
        
        Args:
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            hho_iterations: Number of HHO iterations for initialization
            sgd_iterations: Number of SGD iterations for refinement
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history (HHO and SGD phases)
        """
        np.random.seed(self.random_seed)
        
        if self.use_bias:
            from core.utils import compute_biases
            mu, b_u, b_i = compute_biases(train_ratings, self.n_users, self.n_items)
            self.model.set_biases(mu, b_u, b_i)
        
        # Phase 1: HHO initialization
        if verbose:
            print("=" * 60)
            print("Phase 1: HHO Initialization")
            print("=" * 60)
        
        hho_history = self.hho_optimizer.optimize(
            self.model, train_ratings, hho_iterations, verbose
        )
        
        # Phase 2: SGD refinement
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 2: SGD Refinement")
            print("=" * 60)
        
        sgd_history = self.sgd_optimizer.optimize(
            self.model, train_ratings, sgd_iterations, verbose
        )
        
        # Combine histories
        history = {
            'hho_losses': hho_history['losses'],
            'hho_iterations': hho_history['iterations'],
            'sgd_losses': sgd_history['losses'],
            'sgd_iterations': sgd_history['iterations']
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
    
    def fold_in_new_user(self, user_ratings: np.ndarray,
                         regularization: float = 0.01) -> np.ndarray:
        """Compute U embedding for a new user (cold start with 1+ ratings)."""
        return self.model.fold_in_new_user(
            user_ratings, self.model.V, regularization
        )
    
    def add_new_item(self, genre_vec: np.ndarray) -> int:
        """Add a new item (cold start) using genre embedding."""
        return self.model.add_new_item(genre_vec)

