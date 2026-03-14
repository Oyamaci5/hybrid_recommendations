"""
Harris Hawks Optimization optimizer for Matrix Factorization initialization.

HHO is used ONLY as a global initializer to find good starting U, V matrices.
"""

import numpy as np
from typing import Dict, Tuple
from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class HHOOptimizer(BaseOptimizer):
    """
    HHO optimizer for Matrix Factorization initialization.
    
    Each hawk represents a candidate initialization for U and V matrices.
    HHO searches for good initial embeddings to avoid poor local minima.
    """
    
    def __init__(self, n_hawks: int = 30, escape_energy_initial: float = 1.0,
                 regularization: float = 0.01, boundary: float = 1.0):
        """
        Initialize HHO optimizer.
        
        Args:
            n_hawks: Number of hawks in the population
            escape_energy_initial: Initial escape energy E0
            regularization: L2 regularization coefficient
            boundary: Boundary constraint for embeddings ([-boundary, boundary])
        """
        self.n_hawks = n_hawks
        self.escape_energy_initial = escape_energy_initial
        self.regularization = regularization
        self.boundary = boundary
    
    def optimize(self, model: MFModel, train_ratings: np.ndarray,
                 n_iterations: int = 50, verbose: bool = False) -> Dict:
        """
        Optimize initial embeddings using HHO.
        
        Args:
            model: MFModel instance (will be initialized with best HHO solution)
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            n_iterations: Number of HHO iterations
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
        
        # Initialize hawks (each hawk = candidate initialization)
        # Position: [U.flatten(), V.flatten()]
        ub = self.boundary
        lb = -self.boundary
        hawks = (ub - lb) * np.random.rand(self.n_hawks, dim_total) + lb
        hawks = hawks.astype(np.float32)
        
        # Warm start: use model's current U,V for first hawk if available
        U_cur, V_cur = model.get_parameters()
        init_pos = np.concatenate([U_cur.flatten(), V_cur.flatten()])
        init_pos = np.clip(init_pos, lb, ub).astype(np.float32)
        hawks[0] = init_pos
        
        # Evaluate initial population
        fitnesses = np.zeros(self.n_hawks, dtype=np.float32)
        for i in range(self.n_hawks):
            fitnesses[i] = self._evaluate_hawk(hawks[i], model, n_users, n_items, latent_dim,
                                              user_ids, item_ids, ratings)
        
        # Find best hawk (rabbit)
        rabbit_idx = np.argmin(fitnesses)
        rabbit_position = hawks[rabbit_idx].copy()
        rabbit_fitness = fitnesses[rabbit_idx]
        
        # History
        history = {
            'losses': [rabbit_fitness],
            'iterations': [0]
        }
        
        # HHO main loop
        for iteration in range(1, n_iterations + 1):
            # Update escape energy: E = 2E0(1 - t/T)
            E = 2 * self.escape_energy_initial * (1 - iteration / n_iterations)
            
            # Update each hawk
            for i in range(self.n_hawks):
                # Random number for exploration/exploitation selection
                q = np.random.rand()
                r = np.random.rand()
                
                if abs(E) >= 1.0:
                    # Exploration phase
                    if q >= 0.5:
                        # Random walk around random hawk
                        rand_hawk_idx = np.random.randint(0, self.n_hawks)
                        hawks[i] = hawks[rand_hawk_idx] - r * abs(
                            hawks[rand_hawk_idx] - 2 * r * hawks[i]
                        )
                    else:
                        # Random walk around rabbit (best solution)
                        hawks[i] = rabbit_position - np.mean(hawks, axis=0) - r * (
                            lb + r * (ub - lb)
                        )
                else:
                    # Exploitation phase
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    
                    if r >= 0.5 and abs(E) < 0.5:
                        # Soft besiege
                        hawks[i] = (rabbit_position - hawks[i]) - E * abs(
                            rabbit_position - hawks[i]
                        )
                    elif r >= 0.5 and abs(E) >= 0.5:
                        # Hard besiege
                        hawks[i] = rabbit_position - E * abs(
                            rabbit_position - hawks[i]
                        )
                    elif r < 0.5 and abs(E) < 0.5:
                        # Soft besiege with progressive rapid dives
                        Y = rabbit_position - E * abs(rabbit_position - hawks[i])
                        Z = Y + np.random.rand(dim_total) * levy_flight(dim_total)
                        if self._evaluate_hawk(Y, model, n_users, n_items, latent_dim,
                                             user_ids, item_ids, ratings) < \
                           self._evaluate_hawk(Z, model, n_users, n_items, latent_dim,
                                             user_ids, item_ids, ratings):
                            hawks[i] = Y
                        else:
                            hawks[i] = Z
                    else:
                        # Hard besiege with progressive rapid dives
                        Y = rabbit_position - E * abs(
                            rabbit_position - np.mean(hawks, axis=0)
                        )
                        Z = Y + np.random.rand(dim_total) * levy_flight(dim_total)
                        if self._evaluate_hawk(Y, model, n_users, n_items, latent_dim,
                                             user_ids, item_ids, ratings) < \
                           self._evaluate_hawk(Z, model, n_users, n_items, latent_dim,
                                             user_ids, item_ids, ratings):
                            hawks[i] = Y
                        else:
                            hawks[i] = Z
                
                # Apply boundary constraints
                hawks[i] = np.clip(hawks[i], lb, ub)
                
                # Evaluate updated hawk
                fitnesses[i] = self._evaluate_hawk(hawks[i], model, n_users, n_items, latent_dim,
                                                  user_ids, item_ids, ratings)
                
                # Update rabbit (best solution)
                if fitnesses[i] < rabbit_fitness:
                    rabbit_fitness = fitnesses[i]
                    rabbit_position = hawks[i].copy()
            
            # Record history
            history['losses'].append(rabbit_fitness)
            history['iterations'].append(iteration)
            
            if verbose:
                print(f"HHO Iteration {iteration}: Best Fitness = {rabbit_fitness:.6f}, E = {E:.4f}")
        
        # Set model parameters to best HHO solution (initialization)
        U_best, V_best = self._position_to_matrices(rabbit_position, n_users, n_items, latent_dim)
        model.set_parameters(U_best, V_best)
        
        return history
    
    def _position_to_matrices(self, position: np.ndarray, n_users: int,
                             n_items: int, latent_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert hawk position vector to U and V matrices.
        
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
    
    def _evaluate_hawk(self, position: np.ndarray, model: MFModel, n_users: int, n_items: int,
                      latent_dim: int, user_ids: np.ndarray, item_ids: np.ndarray,
                      ratings: np.ndarray) -> float:
        """Evaluate fitness of a hawk (MSE + regularization)."""
        U, V = self._position_to_matrices(position, n_users, n_items, latent_dim)
        user_embeddings = U[user_ids]
        item_embeddings = V[item_ids]
        predictions = np.sum(user_embeddings * item_embeddings, axis=1)
        if model.use_bias:
            predictions += model.mu + model.b_u[user_ids] + model.b_i[item_ids]
        
        mse = np.mean((ratings - predictions) ** 2)
        
        # Add regularization
        if self.regularization > 0:
            l2_reg = self.regularization * (np.mean(U ** 2) + np.mean(V ** 2))
            return mse + l2_reg
        
        return mse
    
    def get_name(self) -> str:
        """Get optimizer name."""
        return "HHO"


def levy_flight(dim: int) -> np.ndarray:
    """
    Generate Levy flight step.
    
    Args:
        dim: Dimension of the step
        
    Returns:
        Levy flight step vector
    """
    import math
    
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v) ** (1 / beta))
    
    return step.astype(np.float32)

