"""
Particle Swarm Optimization optimizer for Matrix Factorization.

STATUS: LOCKED - PSO-MF baseline implementation finalized.
- Follows standard PSO structure (MATLAB-based reference implementation)
- Adaptive inertia weight (linear decrease from wMax to wMin)
- Global optimizer over MF latent embeddings (U, V)
- Each particle represents complete MF solution
- DO NOT modify hyperparameters or add hybridization.
"""

import numpy as np
from typing import Dict
from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class PSOOptimizer(BaseOptimizer):
    """
    PSO optimizer for Matrix Factorization.
    
    Each particle represents a complete MF solution (U and V matrices).
    Particle position: flattened concatenation of U and V.
    """
    
    def __init__(self, swarm_size: int = 30, inertia_weight_max: float = 0.9,
                 inertia_weight_min: float = 0.2, cognitive_coeff: float = 2.0,
                 social_coeff: float = 2.0, regularization: float = 0.01,
                 boundary: float = 1.0):
        """
        Initialize PSO optimizer.
        
        Args:
            swarm_size: Number of particles in the swarm
            inertia_weight_max: Maximum inertia weight (wMax)
            inertia_weight_min: Minimum inertia weight (wMin)
            cognitive_coeff: Cognitive coefficient (c1)
            social_coeff: Social coefficient (c2)
            regularization: L2 regularization coefficient
            boundary: Boundary constraint for embeddings ([-boundary, boundary])
        """
        self.swarm_size = swarm_size
        self.inertia_weight_max = inertia_weight_max
        self.inertia_weight_min = inertia_weight_min
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.regularization = regularization
        self.boundary = boundary
    
    def optimize(self, model: MFModel, train_ratings: np.ndarray,
                 n_iterations: int = 100, verbose: bool = False) -> Dict:
        """
        Optimize model using PSO.
        
        Args:
            model: MFModel instance to optimize
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            n_iterations: Number of PSO iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization history
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
        
        # Initialize swarm (following MATLAB PSO structure)
        # Each particle position: [U.flatten(), V.flatten()]
        # Position initialization: (ub - lb) * rand + lb
        ub = self.boundary
        lb = -self.boundary
        positions = (ub - lb) * np.random.rand(self.swarm_size, dim_total) + lb
        positions = positions.astype(np.float32)
        
        # Initialize velocities to zero (standard PSO initialization)
        velocities = np.zeros((self.swarm_size, dim_total), dtype=np.float32)
        
        # Velocity limits: vMax = (ub - lb) * 0.2, vMin = -vMax
        velocity_max = (ub - lb) * 0.2
        velocity_min = -velocity_max
        
        # Personal best positions and fitness
        pbest_positions = positions.copy()
        pbest_fitness = np.full(self.swarm_size, np.inf, dtype=np.float32)
        
        # Global best position and fitness
        gbest_position = None
        gbest_fitness = np.inf
        
        # History
        history = {
            'losses': [],
            'iterations': []
        }
        
        # Evaluate initial swarm
        for i in range(self.swarm_size):
            # Decode particle position to (U, V) matrices
            U_particle, V_particle = self._position_to_matrices(positions[i], n_users, n_items, latent_dim)
            
            # Update model parameters with particle-specific (U, V) BEFORE computing fitness
            model.set_parameters(U_particle, V_particle)
            
            # Evaluate fitness using model's compute_loss (reconstruction error + L2 regularization)
            fitness = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
            
            pbest_fitness[i] = fitness
            pbest_positions[i] = positions[i].copy()
            
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = positions[i].copy()
        
        # Ensure gbest_position is initialized (should always be after initial evaluation)
        if gbest_position is None:
            raise RuntimeError("Failed to initialize global best position")
        
        # PSO main loop (following MATLAB PSO structure)
        for iteration in range(n_iterations):
            # Calculate adaptive inertia weight (linear decrease from wMax to wMin)
            # w = wMax - t * ((wMax - wMin) / maxIter)
            inertia_weight = (self.inertia_weight_max - 
                            iteration * ((self.inertia_weight_max - self.inertia_weight_min) / n_iterations))
            
            # PHASE 1: Evaluate all particles and update pbest/gbest
            # (Following MATLAB structure: evaluate first, then update)
            for i in range(self.swarm_size):
                # Decode particle position to (U, V) matrices
                U_particle, V_particle = self._position_to_matrices(positions[i], n_users, n_items, latent_dim)
                
                # Update model parameters with particle-specific (U, V) BEFORE computing fitness
                model.set_parameters(U_particle, V_particle)
                
                # Evaluate fitness using model's compute_loss (reconstruction error + L2 regularization)
                fitness = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
                
                # Update personal best
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if fitness < gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = positions[i].copy()
            
            # PHASE 2: Update velocities and positions for all particles
            # (Following MATLAB structure: update after evaluation)
            for i in range(self.swarm_size):
                # Update velocity (standard PSO equation with adaptive inertia weight)
                r1 = np.random.rand(dim_total)
                r2 = np.random.rand(dim_total)
                
                velocities[i] = (inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (pbest_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (gbest_position - positions[i]))
                
                # Clamp velocity (following MATLAB: check and clip to vMax/vMin)
                # index1 = find(V > vMax), index2 = find(V < vMin)
                index1 = velocities[i] > velocity_max
                index2 = velocities[i] < velocity_min
                velocities[i][index1] = velocity_max
                velocities[i][index2] = velocity_min
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Apply boundary constraints (following MATLAB: clip to ub/lb)
                # index1 = find(X > ub), index2 = find(X < lb)
                index1 = positions[i] > ub
                index2 = positions[i] < lb
                positions[i][index1] = ub
                positions[i][index2] = lb
            
            # Record history
            history['losses'].append(gbest_fitness)
            history['iterations'].append(iteration)
            
            # Debug logging for first 3 iterations
            if iteration < 3:
                # Compute mean fitness across all particles (using model for consistency)
                fitnesses = []
                for i in range(self.swarm_size):
                    U_particle, V_particle = self._position_to_matrices(positions[i], n_users, n_items, latent_dim)
                    model.set_parameters(U_particle, V_particle)
                    fitness = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
                    fitnesses.append(fitness)
                mean_fitness = np.mean(fitnesses)
                
                # Compute mean velocity norm (verify velocity is not collapsing to zero)
                mean_velocity_norm = np.mean([np.linalg.norm(velocities[i]) for i in range(self.swarm_size)])
                
                if verbose:
                    print(f"Iteration {iteration}: Best Loss = {gbest_fitness:.6f}, "
                          f"Mean Loss = {mean_fitness:.6f}, Mean Velocity Norm = {mean_velocity_norm:.6f}")
            elif verbose:
                print(f"Iteration {iteration}: Best Loss = {gbest_fitness:.6f}")
        
        # Set model parameters to global best solution
        U_best, V_best = self._position_to_matrices(gbest_position, n_users, n_items, latent_dim)
        model.set_parameters(U_best, V_best)
        
        return history
    
    def _position_to_matrices(self, position: np.ndarray, n_users: int, 
                             n_items: int, latent_dim: int):
        """
        Convert particle position vector to U and V matrices.
        
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
    
    
    def get_name(self) -> str:
        """Get optimizer name."""
        return "PSO"
