"""
Coati Optimization Algorithm (COA) optimizer for Matrix Factorization initialization.

COA is used ONLY as a global initializer to find good starting U, V matrices.
"""

import numpy as np
from typing import Dict, Tuple
from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class COAOptimizer(BaseOptimizer):
    """
    COA optimizer for Matrix Factorization initialization.
    
    Each coati represents a candidate initialization for U and V matrices.
    COA searches for good initial embeddings using two-phase strategy:
    - Phase 1: Hunting and attacking strategy on the iguana (Exploration)
    - Phase 2: Escaping from predators (Exploitation)
    """
    
    def __init__(self, n_coatis: int = 30, regularization: float = 0.01,
                 boundary: float = 1.0):
        """
        Initialize COA optimizer.
        
        Args:
            n_coatis: Number of coatis in the population
            regularization: L2 regularization coefficient
            boundary: Boundary constraint for embeddings ([-boundary, boundary])
        """
        self.n_coatis = n_coatis
        self.regularization = regularization
        self.boundary = boundary
    
    def optimize(self, model: MFModel, train_ratings: np.ndarray,
                 n_iterations: int = 50, verbose: bool = False) -> Dict:
        """
        Optimize initial embeddings using COA.
        
        Args:
            model: MFModel instance (will be initialized with best COA solution)
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            n_iterations: Number of COA iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization history (best fitness per iteration)
        """
        n_users = model.n_users
        n_items = model.n_items
        latent_dim = model.latent_dim
        
        # Extract training data
        user_ids = train_ratings[:, 0].astype(int)
        item_ids = train_ratings[:, 1].astype(int)
        ratings = train_ratings[:, 2]
        
        # Dimension of search space: U (n_users * latent_dim) + V (n_items * latent_dim)
        dim_U = n_users * latent_dim
        dim_V = n_items * latent_dim
        dim_total = dim_U + dim_V
        
        # Initialize coatis (each coati = candidate initialization)
        # Position: [U.flatten(), V.flatten()]
        ub = self.boundary
        lb = -self.boundary
        coatis = (ub - lb) * np.random.rand(self.n_coatis, dim_total) + lb
        coatis = coatis.astype(np.float32)
        
        # Warm start: use model's current U,V for first coati if available
        U_cur, V_cur = model.get_parameters()
        init_pos = np.concatenate([U_cur.flatten(), V_cur.flatten()])
        init_pos = np.clip(init_pos, lb, ub).astype(np.float32)
        coatis[0] = init_pos
        
        # Evaluate initial population
        fitnesses = np.zeros(self.n_coatis, dtype=np.float32)
        for i in range(self.n_coatis):
            fitnesses[i] = self._evaluate_coati(coatis[i], model, n_users, n_items, latent_dim,
                                                user_ids, item_ids, ratings)
        
        # Find best coati (best solution)
        best_idx = np.argmin(fitnesses)
        best_position = coatis[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        
        # History
        history = {
            'losses': [best_fitness],
            'iterations': [0]
        }
        
        # COA main loop
        for iteration in range(1, n_iterations + 1):
            # Update iguana position (based on best member)
            iguana_position = best_position.copy()
            
            # Phase 1: Hunting and attacking strategy on the iguana (Exploration)
            # First half of population: i = 1 to N/2
            n_half = self.n_coatis // 2
            for i in range(n_half):
                # Calculate new position using Eq. (4): X_i^P1 = X_i + r * (Iguana - I * X_i)
                # where I is a random binary vector
                I = np.random.randint(0, 2, dim_total)
                r = np.random.rand(dim_total)
                X_i_P1 = coatis[i] + r * (iguana_position - I * coatis[i])
                
                # Update position using Eq. (7): X_i = X_i^P1 if better, else keep X_i
                fitness_P1 = self._evaluate_coati(X_i_P1, model, n_users, n_items, latent_dim,
                                                 user_ids, item_ids, ratings)
                if fitness_P1 < fitnesses[i]:
                    coatis[i] = X_i_P1
                    fitnesses[i] = fitness_P1
                    if fitnesses[i] < best_fitness:
                        best_fitness = fitnesses[i]
                        best_position = coatis[i].copy()
            
            # Second half of population: i = N/2 + 1 to N
            for i in range(n_half, self.n_coatis):
                # Generate random position of the iguana using Eq. (5)
                # Iguana_random = lb + r * (ub - lb)
                r = np.random.rand(dim_total)
                iguana_random = lb + r * (ub - lb)
                
                # Calculate new position using Eq. (6): X_i^P1 = X_i + r * (Iguana_random - I * X_i)
                I = np.random.randint(0, 2, dim_total)
                r = np.random.rand(dim_total)
                X_i_P1 = coatis[i] + r * (iguana_random - I * coatis[i])
                
                # Apply boundary constraints
                X_i_P1 = np.clip(X_i_P1, lb, ub)
                
                # Update position using Eq. (7)
                fitness_P1 = self._evaluate_coati(X_i_P1, model, n_users, n_items, latent_dim,
                                                 user_ids, item_ids, ratings)
                if fitness_P1 < fitnesses[i]:
                    coatis[i] = X_i_P1
                    fitnesses[i] = fitness_P1
                    if fitnesses[i] < best_fitness:
                        best_fitness = fitnesses[i]
                        best_position = coatis[i].copy()
            
            # Phase 2: Escaping from predators (Exploitation)
            # Calculate local bounds using Eq. (8)
            # Local bounds around best solution
            alpha = 1.0 - (iteration / n_iterations)  # Decreases over time
            local_lb = best_position - alpha * (ub - lb) / 2
            local_ub = best_position + alpha * (ub - lb) / 2
            local_lb = np.clip(local_lb, lb, ub)
            local_ub = np.clip(local_ub, lb, ub)
            
            # Update all coatis
            for i in range(self.n_coatis):
                # Calculate new position using Eq. (9): X_i^P2 = lb_local + r * (ub_local - lb_local)
                r = np.random.rand(dim_total)
                X_i_P2 = local_lb + r * (local_ub - local_lb)
                
                # Update position using Eq. (10): X_i = X_i^P2 if better, else keep X_i
                fitness_P2 = self._evaluate_coati(X_i_P2, model, n_users, n_items, latent_dim,
                                                 user_ids, item_ids, ratings)
                if fitness_P2 < fitnesses[i]:
                    coatis[i] = X_i_P2
                    fitnesses[i] = fitness_P2
                    if fitnesses[i] < best_fitness:
                        best_fitness = fitnesses[i]
                        best_position = coatis[i].copy()
            
            # Record history
            history['losses'].append(best_fitness)
            history['iterations'].append(iteration)
            
            if verbose:
                print(f"COA Iteration {iteration}: Best Fitness = {best_fitness:.6f}")
        
        # Set model parameters to best COA solution (initialization)
        U_best, V_best = self._position_to_matrices(best_position, n_users, n_items, latent_dim)
        model.set_parameters(U_best, V_best)
        
        return history
    
    def _position_to_matrices(self, position: np.ndarray, n_users: int,
                             n_items: int, latent_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert coati position vector to U and V matrices.
        
        Args:
            position: Flattened position vector [U.flatten(), V.flatten()]
            n_users: Number of users
            n_items: Number of items
            latent_dim: Latent dimension
            
        Returns:
            Tuple of (U, V) matrices
        """
        dim_U = n_users * latent_dim
        U_flat = position[:dim_U]
        V_flat = position[dim_U:]
        
        U = U_flat.reshape(n_users, latent_dim)
        V = V_flat.reshape(n_items, latent_dim)
        
        return U, V
    
    def _evaluate_coati(self, position: np.ndarray, model: MFModel, n_users: int, n_items: int,
                       latent_dim: int, user_ids: np.ndarray, item_ids: np.ndarray,
                       ratings: np.ndarray) -> float:
        """
        Evaluate fitness of a coati (RMSE + L2 regularization).
        """
        U, V = self._position_to_matrices(position, n_users, n_items, latent_dim)
        user_embeddings = U[user_ids]
        item_embeddings = V[item_ids]
        predictions = np.sum(user_embeddings * item_embeddings, axis=1)
        if model.use_bias:
            predictions += model.mu + model.b_u[user_ids] + model.b_i[item_ids]
        
        rmse = np.sqrt(np.mean((ratings - predictions) ** 2))
        
        # Add regularization
        if self.regularization > 0:
            l2_reg = self.regularization * (np.mean(U ** 2) + np.mean(V ** 2))
            return rmse + l2_reg
        
        return rmse
    
    def get_name(self) -> str:
        """Get optimizer name."""
        return "COA"
