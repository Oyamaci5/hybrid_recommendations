"""
Matrix Factorization model for collaborative filtering.
Supports bias terms and content-based initialization for cold start.
"""

import numpy as np
from typing import Tuple, Optional


class MFModel:
    """
    Matrix Factorization model with user and item latent embeddings.
    
    Predicts rating as: r_ui = μ + b_u[u] + b_i[i] + U[u] @ V[i]
    When use_bias=False: r_ui = U[u] @ V[i] (backward compatible)
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 random_seed: Optional[int] = None,
                 use_bias: bool = True,
                 V_init: Optional[np.ndarray] = None,
                 U_init: Optional[np.ndarray] = None,
                 W_item: Optional[np.ndarray] = None):
        """
        Initialize MF model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            latent_dim: Number of latent factors (k)
            random_seed: Random seed for initialization (optional)
            use_bias: If True, use μ, b_u, b_i in prediction
            V_init: Optional initial V matrix (n_items, latent_dim) for content-based init
            U_init: Optional initial U matrix (n_users, latent_dim) for content-based init
            W_item: Optional projection matrix (n_genres, latent_dim) for add_new_item cold start
        """
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        scale = np.sqrt(1.0 / latent_dim)
        
        # User embeddings
        if U_init is not None:
            assert U_init.shape == (n_users, latent_dim)
            self.U = U_init.copy().astype(np.float32)
        else:
            self.U = np.random.normal(0, scale, (n_users, latent_dim)).astype(np.float32)
        
        # Item embeddings
        if V_init is not None:
            assert V_init.shape == (n_items, latent_dim)
            self.V = V_init.copy().astype(np.float32)
        else:
            self.V = np.random.normal(0, scale, (n_items, latent_dim)).astype(np.float32)
        
        # Bias terms (computed/updated separately; init to zero)
        self.mu = 0.0
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)
        
        # For cold start: projection from genre vec to latent (used by add_new_item)
        self.W_item = W_item.copy() if W_item is not None else None
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for given user-item pairs.
        
        r_ui = μ + b_u[u] + b_i[i] + U[u] @ V[i]
        """
        user_ids = user_ids.astype(int)
        item_ids = item_ids.astype(int)
        user_embeddings = self.U[user_ids]
        item_embeddings = self.V[item_ids]
        predictions = np.sum(user_embeddings * item_embeddings, axis=1)
        
        if self.use_bias:
            predictions += self.mu + self.b_u[user_ids] + self.b_i[item_ids]
        
        return predictions
    
    def predict_all(self) -> np.ndarray:
        """
        Predict all ratings: R = μ + b_u + b_i + U @ V^T
        """
        R = self.U @ self.V.T
        if self.use_bias:
            R += self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
        return R
    
    def compute_loss(self, user_ids: np.ndarray, item_ids: np.ndarray, 
                     ratings: np.ndarray, regularization: float = 0.0) -> float:
        """
        Compute squared error loss with optional L2 regularization.
        """
        predictions = self.predict(user_ids, item_ids)
        squared_error = np.mean((ratings - predictions) ** 2)
        
        if regularization > 0:
            l2_reg = regularization * (
                np.mean(self.U ** 2) + np.mean(self.V ** 2)
            )
            if self.use_bias:
                l2_reg += regularization * (
                    np.mean(self.b_u ** 2) + np.mean(self.b_i ** 2)
                )
            return squared_error + l2_reg
        
        return squared_error
    
    def set_biases(self, mu: float, b_u: np.ndarray, b_i: np.ndarray):
        """Set bias terms (e.g. from compute_biases)."""
        self.mu = float(mu)
        self.b_u = np.asarray(b_u, dtype=np.float32)
        self.b_i = np.asarray(b_i, dtype=np.float32)
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get model parameters (U, V)."""
        return self.U.copy(), self.V.copy()
    
    def set_parameters(self, U: np.ndarray, V: np.ndarray):
        """Set model parameters (U, V)."""
        assert U.shape == (self.n_users, self.latent_dim)
        assert V.shape == (self.n_items, self.latent_dim)
        self.U = U.copy()
        self.V = V.copy()
    
    def fold_in_new_user(self, user_ratings: np.ndarray, V: np.ndarray,
                         regularization: float = 0.01) -> np.ndarray:
        """
        Compute U embedding for a new user given their ratings (folding-in).
        V is fixed (item embeddings). Solves: U[i] = (V^T V + λI)^{-1} V^T (r - μ - b_i).
        
        Args:
            user_ratings: Array of shape (n_ratings, 2) with [item_id, rating]
            V: Item matrix (n_items, latent_dim) - typically model.V
            regularization: L2 regularization for the closed-form solve
            
        Returns:
            U vector of shape (latent_dim,) for the new user
        """
        item_ids = user_ratings[:, 0].astype(int)
        ratings = user_ratings[:, 1]
        
        # Residual: r - μ - b_i (b_u for new user is 0)
        resid = ratings.copy()
        if self.use_bias:
            resid -= self.mu + self.b_i[item_ids]
        
        V_sub = V[item_ids]  # (n_ratings, latent_dim)
        k = V_sub.shape[1]
        # U = (V^T V + λI)^{-1} V^T r
        VtV = V_sub.T @ V_sub + regularization * np.eye(k)
        Vtr = V_sub.T @ resid
        U_new = np.linalg.solve(VtV, Vtr)
        return U_new.astype(np.float32)
    
    def add_new_item(self, genre_vec: np.ndarray) -> int:
        """
        Add a new item (cold start) using genre embedding.
        Requires W_item to have been set during content-based initialization.
        
        Args:
            genre_vec: Genre multi-hot vector of shape (n_genres,)
            
        Returns:
            New item index (previous n_items)
        """
        if self.W_item is None:
            raise ValueError("add_new_item requires W_item (content-based init)")
        genre_vec = np.asarray(genre_vec, dtype=np.float32)
        if genre_vec.ndim == 1:
            genre_vec = genre_vec.reshape(1, -1)
        V_new = (genre_vec @ self.W_item).flatten()
        # Append to V and b_i
        self.V = np.vstack([self.V, V_new.astype(np.float32)])
        self.b_i = np.append(self.b_i, 0.0)
        self.n_items += 1
        return self.n_items - 1
