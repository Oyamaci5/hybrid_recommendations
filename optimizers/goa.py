"""
Grasshopper Optimization Algorithm (GOA) optimizer for Matrix Factorization initialization.

Her grasshopper, U ve V matrisleri için bir aday başlangıcı temsil eder:
    position = [U.flatten(), V.flatten()]

Bu sınıf, Saremi et al. (2017) ve GOA-k-means makalesindeki (Ambikesh et al., 2024)
denklemlerden uyarlanmıştır.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class GOAOptimizer(BaseOptimizer):
    """
    GOA optimizer for Matrix Factorization initialization.

    Not:
      - Burada amaç, COAOptimizer gibi yalnızca iyi bir global başlangıç bulmak;
        dolayısıyla tek fazlı bir GOA döngüsü ile en iyi U, V seçilip modele set edilir.
    """

    def __init__(
        self,
        n_grasshoppers: int = 40,
        regularization: float = 0.01,
        boundary: float = 1.0,
        c_min: float = 0.00004,
        c_max: float = 1.0,
    ) -> None:
        """
        Initialize GOA optimizer.

        Args:
            n_grasshoppers: Population size (search agents)
            regularization: L2 regularization coefficient
            boundary: Search space bound ([-boundary, boundary])
            c_min: minimum shrinking coefficient
            c_max: maximum shrinking coefficient
        """
        self.n_grasshoppers = n_grasshoppers
        self.regularization = regularization
        self.boundary = boundary
        self.c_min = c_min
        self.c_max = c_max

    def optimize(
        self,
        model: MFModel,
        train_ratings: np.ndarray,
        n_iterations: int = 50,
        verbose: bool = False,
    ) -> Dict:
        """
        Optimize initial embeddings using GOA.

        Args:
            model: MFModel instance (U, V will be set to best GOA solution)
            train_ratings: (n_ratings, 3) [user_id, item_id, rating]
            n_iterations: number of GOA iterations
            verbose: whether to print progress
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

        # Başlangıç popülasyonu
        positions = (ub - lb) * np.random.rand(self.n_grasshoppers, dim_total) + lb
        positions = positions.astype(np.float32)

        # Warm start: mevcut model parametreleri ilk ajana konur
        U_cur, V_cur = model.get_parameters()
        init_pos = np.concatenate([U_cur.flatten(), V_cur.flatten()])
        init_pos = np.clip(init_pos, lb, ub).astype(np.float32)
        positions[0] = init_pos

        # Başlangıç fitness'leri
        fitnesses = np.zeros(self.n_grasshoppers, dtype=np.float32)
        for i in range(self.n_grasshoppers):
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

        # GOA ana döngüsü
        for t in range(1, n_iterations + 1):
            # Shrinking coefficient c (Eq. 11)
            if n_iterations > 0:
                c = self.c_max - t * (self.c_max - self.c_min) / float(n_iterations)
            else:
                c = self.c_min

            # Normalized positions için referans: en iyi ajan (T_d)
            T = best_pos.copy()

            new_positions = np.zeros_like(positions)

            for i in range(self.n_grasshoppers):
                Xi = positions[i]

                # Sosyal etkileşim kısmı (Eq. 10 içindeki toplam)
                S_component = np.zeros(dim_total, dtype=np.float32)
                for j in range(self.n_grasshoppers):
                    if j == i:
                        continue
                    Xj = positions[j]
                    dist_vec = Xj - Xi
                    dist = np.linalg.norm(dist_vec) + 1e-12

                    # d_ij normalize distance (scaled to [0,1])
                    # Orijinal makalede genelde |xj - xi| / R_max tarzı,
                    # burada basitçe bir ölçekleme sabiti kullanıyoruz.
                    r = dist / (ub - lb + 1e-12)
                    # s(r) fonksiyonu (Eq. 6)
                    f = 0.5  # attraction intensity (heuristic)
                    l = 1.5  # attractive range (heuristic)
                    s_r = f * np.exp(-r / l) - np.exp(-r)

                    S_component += (s_r * dist_vec / dist)

                # Eq. 10: xi = c * ( (UBd - LBd)/2 * S_component ) + T_d
                Xi_new = c * ((ub - lb) / 2.0 * S_component) + T

                # Sınırlar
                Xi_new = np.clip(Xi_new, lb, ub)
                new_positions[i] = Xi_new.astype(np.float32)

            # Yeni fitness'ler
            for i in range(self.n_grasshoppers):
                fitnesses[i] = self._evaluate_position(
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

            # Global en iyiyi güncelle
            cur_best_idx = int(np.argmin(fitnesses))
            cur_best_fit = float(fitnesses[cur_best_idx])
            if cur_best_fit < best_fit:
                best_fit = cur_best_fit
                best_pos = positions[cur_best_idx].copy()

            history["losses"].append(best_fit)
            history["iterations"].append(t)

            if verbose:
                print(f"GOA Iteration {t}: Best Fitness = {best_fit:.6f}, c = {c:.6f}")

        # En iyi çözümü modele set et
        U_best, V_best = self._position_to_matrices(
            best_pos,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U_best, V_best)

        return history

    # ------------------------------------------------------------------
    # Yardımcı fonksiyonlar
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
        """Fitness: RMSE + L2 regularization (via model.compute_loss)."""
        U, V = self._position_to_matrices(
            position,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U, V)
        loss = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
        return float(loss)

    def get_name(self) -> str:
        return "GOA"

