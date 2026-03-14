"""
Hybrid COA + SMA optimizer for Matrix Factorization.

Fikir:
  - Alt popülasyon (COA benzeri faz 1/2) geniş arama yapar.
  - Üst popülasyon (SMA benzeri) en iyi çözümler etrafında dar arama yapar.
  - İterasyon ilerledikçe SMA oranı artar (exploration → exploitation).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class HybridCOASMAOptimizer(BaseOptimizer):
    """
    COA + SMA hibrit optimizer for Matrix Factorization.

    Her ajan, tam bir MF çözümünü temsil eder:
        position = [U.flatten(), V.flatten()]
    """

    def __init__(
        self,
        n_agents: int = 40,
        sma_ratio_min: float = 0.1,
        sma_ratio_max: float = 0.9,
        regularization: float = 0.01,
        boundary: float = 1.0,
    ) -> None:
        self.n_agents = n_agents
        self.sma_ratio_min = sma_ratio_min
        self.sma_ratio_max = sma_ratio_max
        self.regularization = regularization
        self.boundary = boundary

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        model: MFModel,
        train_ratings: np.ndarray,
        n_iterations: int = 50,
        verbose: bool = False,
    ) -> Dict:
        """
        Optimize MF parameters using COA+SMA hybrid population.

        Args:
            model: MFModel instance
            train_ratings: (n_ratings, 3) [user_id, item_id, rating]
            n_iterations: metaheuristic iteration count
            verbose: whether to print logs
        """
        n_users = model.n_users
        n_items = model.n_items
        latent_dim = model.latent_dim

        user_ids = train_ratings[:, 0].astype(int)
        item_ids = train_ratings[:, 1].astype(int)
        ratings = train_ratings[:, 2]

        dim_U = n_users * latent_dim
        dim_V = n_items * latent_dim
        dim_total = dim_U + dim_V

        ub = self.boundary
        lb = -self.boundary

        # Initial population
        positions = (ub - lb) * np.random.rand(self.n_agents, dim_total) + lb
        positions = positions.astype(np.float32)

        # Warm-start first agent with current model params
        U_cur, V_cur = model.get_parameters()
        init_pos = np.concatenate([U_cur.flatten(), V_cur.flatten()])
        init_pos = np.clip(init_pos, lb, ub).astype(np.float32)
        positions[0] = init_pos

        # Initial fitness
        fitnesses = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            fitnesses[i] = self._evaluate_position(
                positions[i],
                model,
                n_users,
                n_items,
                latent_dim,
                user_ids,
                item_ids,
                ratings,
            )

        best_idx = int(np.argmin(fitnesses))
        best_pos = positions[best_idx].copy()
        best_fit = float(fitnesses[best_idx])

        history = {"losses": [best_fit], "iterations": [0]}

        for it in range(1, n_iterations + 1):
            # Sort population by fitness (best first)
            sort_idx = np.argsort(fitnesses)
            positions = positions[sort_idx]
            fitnesses = fitnesses[sort_idx]

            # SMA ratio schedule (exploration -> exploitation)
            if n_iterations > 1:
                t_ratio = (it - 1) / float(n_iterations - 1)
            else:
                t_ratio = 1.0
            sma_ratio = self.sma_ratio_min + (self.sma_ratio_max - self.sma_ratio_min) * t_ratio
            sma_ratio = float(np.clip(sma_ratio, 0.0, 1.0))

            n_sma = int(round(sma_ratio * self.n_agents))
            n_sma = max(0, min(n_sma, self.n_agents))
            n_coa = self.n_agents - n_sma

            # Split into SMA (top) and COA (bottom) subpopulations
            sma_positions = positions[:n_sma].copy() if n_sma > 0 else np.empty((0, dim_total), dtype=np.float32)
            sma_fitnesses = fitnesses[:n_sma].copy() if n_sma > 0 else np.empty((0,), dtype=np.float32)

            coa_positions = positions[n_sma:].copy() if n_coa > 0 else np.empty((0, dim_total), dtype=np.float32)
            coa_fitnesses = fitnesses[n_sma:].copy() if n_coa > 0 else np.empty((0,), dtype=np.float32)

            # COA-like wide search on lower subpopulation
            if n_coa > 0:
                coa_positions = self._coa_phase_update(
                    coa_positions,
                    coa_fitnesses,
                    lb,
                    ub,
                    iteration=it,
                    max_iter=n_iterations,
                )

            # SMA-like exploitation on upper subpopulation
            if n_sma > 0:
                global_best = positions[0].copy()  # best of full population
                sma_positions = self._sma_update(
                    sma_positions,
                    sma_fitnesses,
                    global_best,
                    lb,
                    ub,
                    iteration=it,
                    max_iter=n_iterations,
                )

            # Merge back
            new_positions = np.zeros_like(positions)
            if n_sma > 0:
                new_positions[:n_sma] = sma_positions
            if n_coa > 0:
                new_positions[n_sma:] = coa_positions

            new_positions = np.clip(new_positions, lb, ub)

            # Evaluate
            new_fitnesses = np.zeros_like(fitnesses)
            for i in range(self.n_agents):
                new_fitnesses[i] = self._evaluate_position(
                    new_positions[i],
                    model,
                    n_users,
                    n_items,
                    latent_dim,
                    user_ids,
                    item_ids,
                    ratings,
                )

            positions = new_positions
            fitnesses = new_fitnesses

            # Update global best
            cur_best_idx = int(np.argmin(fitnesses))
            cur_best_fit = float(fitnesses[cur_best_idx])
            if cur_best_fit < best_fit:
                best_fit = cur_best_fit
                best_pos = positions[cur_best_idx].copy()

            history["losses"].append(best_fit)
            history["iterations"].append(it)

            if verbose:
                print(
                    f"Hybrid COA+SMA Iter {it}: "
                    f"Best Loss = {best_fit:.6f}, "
                    f"SMA ratio = {sma_ratio:.2f}"
                )

        # Set model parameters to best solution
        U_best, V_best = self._position_to_matrices(
            best_pos,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U_best, V_best)

        return history

    def get_name(self) -> str:
        return "Hybrid COA+SMA"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _position_to_matrices(
        self,
        position: np.ndarray,
        n_users: int,
        n_items: int,
        latent_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dim_U = n_users * latent_dim
        U_flat = position[:dim_U]
        V_flat = position[dim_U:]
        U = U_flat.reshape(n_users, latent_dim)
        V = V_flat.reshape(n_items, latent_dim)
        return U, V

    def _evaluate_position(
        self,
        position: np.ndarray,
        model: MFModel,
        n_users: int,
        n_items: int,
        latent_dim: int,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
    ) -> float:
        """Fitness: model.compute_loss (reconstruction + L2)."""
        U, V = self._position_to_matrices(
            position,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U, V)
        loss = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
        return float(loss)

    def _coa_phase_update(
        self,
        positions: np.ndarray,
        fitnesses: np.ndarray,
        lb: float,
        ub: float,
        iteration: int,
        max_iter: int,
    ) -> np.ndarray:
        """
        Basitleştirilmiş COA faz 1/2: gruplar halinde, lokal lider etrafında geniş arama.
        """
        n, dim = positions.shape
        if n == 0:
            return positions

        new_positions = positions.copy()

        # Step size schedule
        step_max = 0.2 * (ub - lb)
        step_min = 0.01 * (ub - lb)
        if max_iter > 1:
            t_ratio = iteration / float(max_iter)
        else:
            t_ratio = 1.0
        step = step_max - (step_max - step_min) * t_ratio

        # Random grouping
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_groups = max(2, n // 5)
        groups = np.array_split(indices, n_groups)

        for g in groups:
            if len(g) == 0:
                continue

            # Local best inside group
            g_fits = fitnesses[g]
            best_local_idx = g[np.argmin(g_fits)]
            best_local = positions[best_local_idx]

            for i in g:
                if i == best_local_idx:
                    # Leader explores around its position
                    direction = np.random.randn(dim)
                    pos_new = best_local + step * direction
                else:
                    # Move towards leader with some randomness
                    r1 = np.random.rand(dim)
                    r2 = np.random.randn(dim)
                    pos_new = positions[i] + step * (
                        r1 * (best_local - positions[i]) + r2
                    )

                new_positions[i] = np.clip(pos_new, lb, ub)

        return new_positions.astype(np.float32)

    def _sma_update(
        self,
        positions: np.ndarray,
        fitnesses: np.ndarray,
        global_best: np.ndarray,
        lb: float,
        ub: float,
        iteration: int,
        max_iter: int,
    ) -> np.ndarray:
        """
        Basitleştirilmiş SMA: fitliğe göre ağırlıklı, global/elit çevresinde dar arama.
        """
        n, dim = positions.shape
        if n == 0:
            return positions

        new_positions = positions.copy()

        # Normalize fitness to weights (better -> larger weight)
        f_min = float(np.min(fitnesses))
        f_max = float(np.max(fitnesses))
        if f_max - f_min == 0.0:
            weights = np.ones_like(fitnesses, dtype=np.float32)
        else:
            weights = (f_max - fitnesses) / (f_max - f_min)

        # Elites: best quarter of SMA subpopulation
        idx_sorted = np.argsort(fitnesses)
        n_elite = max(1, n // 4)
        elite_idx = idx_sorted[:n_elite]
        elites = positions[elite_idx]

        # Decaying coefficient a
        a_max = 1.0
        a_min = 0.1
        if max_iter > 1:
            t_ratio = iteration / float(max_iter)
        else:
            t_ratio = 1.0
        a = a_max - (a_max - a_min) * t_ratio

        for i in range(n):
            w = weights[i]
            elite = elites[np.random.randint(0, n_elite)]

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            exploitation_move = r1 * (global_best - positions[i]) + r2 * (
                elite - positions[i]
            )
            noise = 0.01 * (ub - lb) * np.random.randn(dim)

            pos_new = positions[i] + a * w * exploitation_move + noise
            new_positions[i] = np.clip(pos_new, lb, ub)

        return new_positions.astype(np.float32)

