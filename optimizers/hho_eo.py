"""
Hybrid HHO + EO optimizer for Matrix Factorization.

Amaç:
  - HHO'nun güçlü keşif/sömürü dinamiğini,
  - EO'nun equilibrium candidate havuzu ve eksponansiyel terimi ile
birleştirerek MF (U, V) parametrelerinin global optimizasyonunu yapmak.

Bu optimizer:
  - Tamamen meta-sezgisel, saf işbirlikçi filtreleme (MF) parametre optimizasyonu içindir.
  - Loss: model.compute_loss(...) (rekonstrüksiyon hatası + L2).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import math

from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class HHOEOOptimizer(BaseOptimizer):
    """
    HHO+EO hibrit optimizer for Matrix Factorization.

    Her ajan (hawk), tam bir MF çözümünü temsil eder:
      position = [U.flatten(), V.flatten()]
    """

    def __init__(
        self,
        n_agents: int = 40,
        escape_energy_initial: float = 1.0,
        regularization: float = 0.01,
        boundary: float = 1.0,
    ) -> None:
        self.n_agents = n_agents
        self.escape_energy_initial = escape_energy_initial
        self.regularization = regularization
        self.boundary = boundary

    # ------------------------------------------------------------------
    # BaseOptimizer arayüzü
    # ------------------------------------------------------------------

    def optimize(
        self,
        model: MFModel,
        train_ratings: np.ndarray,
        n_iterations: int = 50,
        verbose: bool = False,
    ) -> Dict:
        """
        HHO+EO hibriti ile MF parametrelerini optimize et.

        Args:
            model: MFModel instance
            train_ratings: (n_ratings, 3) [user_id, item_id, rating]
            n_iterations: metaheuristic iterasyon sayısı
            verbose: log basılsın mı
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
        positions = (ub - lb) * np.random.rand(self.n_agents, dim_total) + lb
        positions = positions.astype(np.float32)

        # İlk ajanı mevcut model parametreleri ile doldur (warm-start)
        U_cur, V_cur = model.get_parameters()
        init_pos = np.concatenate([U_cur.flatten(), V_cur.flatten()])
        init_pos = np.clip(init_pos, lb, ub).astype(np.float32)
        positions[0] = init_pos

        # Fitness hesapla
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

        # En iyi ajan (rabbit)
        rabbit_idx = int(np.argmin(fitnesses))
        rabbit_pos = positions[rabbit_idx].copy()
        rabbit_fit = float(fitnesses[rabbit_idx])

        history = {
            "losses": [rabbit_fit],
            "iterations": [0],
        }

        for it in range(1, n_iterations + 1):
            # EO: equilibrium pool oluştur (en iyi 4 + ortalamaları)
            eq_pool = self._build_equilibrium_pool(positions, fitnesses)

            # Ortalama pozisyon (HHO formüllerinde kullanılıyor)
            mean_pos = np.mean(positions, axis=0)

            # Kaçış enerjisi (HHO) – pseudo-kodla uyumlu:
            # E0 = 2*rand() - 1, E = 2*E0*(1 - t/T)
            E0 = 2.0 * np.random.rand() - 1.0
            E = 2.0 * E0 * (1.0 - it / float(n_iterations))
            # mevcut parametreyi ölçekleyici olarak kullan
            E *= self.escape_energy_initial

            for i in range(self.n_agents):
                Xi = positions[i]

                # --- KEŞİF FAZI (|E| >= 1) ---
                if abs(E) >= 1.0:
                    q = np.random.rand()
                    if q >= 0.5:
                        # X_i = X_rabbit - X_mean - E * |J * X_rabbit - X_i|
                        J = 2.0 * np.random.rand(dim_total) - 1.0
                        Xi_new = rabbit_pos - mean_pos - E * np.abs(J * rabbit_pos - Xi)
                    else:
                        # X_r = rastgele birey
                        rand_idx = np.random.randint(0, self.n_agents)
                        X_r = positions[rand_idx]
                        # X_i = X_r - E * |X_r - 2 * rand() * X_i|
                        r_vec = np.random.rand(dim_total)
                        Xi_new = X_r - E * np.abs(X_r - 2.0 * r_vec * Xi)

                # --- SÖMÜRÜ FAZI (|E| < 1) ---
                else:
                    alpha = abs(E)  # hibrit ağırlık

                    # ----- HHO sömürü bileşeni -----
                    r1 = np.random.rand()
                    J = 2.0 * np.random.rand(dim_total) - 1.0
                    if r1 >= 0.5 and abs(E) >= 0.5:
                        # Yumuşak besiege:
                        # X_HHO = X_rabbit - E * |J * X_rabbit - X_i|
                        X_HHO = rabbit_pos - E * np.abs(J * rabbit_pos - Xi)
                    else:
                        # Sert besiege + Lévy:
                        # X_HHO = X_rabbit - E * |J * X_rabbit - X_mean| + rand * Lévy(dim)
                        levy_step = self._levy_flight(dim_total)
                        X_HHO = rabbit_pos - E * np.abs(J * rabbit_pos - mean_pos) + np.random.rand() * levy_step

                    # ----- EO sömürü bileşeni -----
                    a1 = 2.0
                    GP = 0.5
                    dim = dim_total
                    iter_ratio = it / float(n_iterations)
                    t_eo = (1.0 - iter_ratio) * (iter_ratio**1.0)

                    lambda_vec = np.random.rand(dim)
                    Er = np.random.rand(dim)
                    F = a1 * np.sign(Er - 0.5) * (np.exp(-lambda_vec * t_eo) - 1.0)

                    # Rasgele Ceq seç
                    rand_idx = np.random.randint(0, eq_pool.shape[0])
                    Ceq = eq_pool[rand_idx]

                    r1_eo = np.random.rand(dim)
                    r2_eo = np.random.rand(dim)
                    GCP = np.where(r2_eo >= GP, 0.5 * r1_eo, 0.0)
                    G0 = GCP * (Ceq - lambda_vec * Xi)
                    G = G0 * F

                    # Pseudo-koda yakın bir EO güncellemesi:
                    # X_EO = Ceq + (X_i - Ceq) * F + (G / (lambda_vec + eps)) * (1 - F)
                    eps = 1e-12
                    X_EO = Ceq + (Xi - Ceq) * F + (G / (lambda_vec + eps)) * (1.0 - F)

                    # ----- Hibrit birleştirme -----
                    Xi_new = alpha * X_HHO + (1.0 - alpha) * X_EO

                # Sınır kontrolü
                Xi_new = np.clip(Xi_new, lb, ub).astype(np.float32)

                # Yeni fitness
                fit_new = self._evaluate_position(
                    Xi_new,
                    model,
                    n_users,
                    n_items,
                    latent_dim,
                    user_ids,
                    item_ids,
                    ratings,
                )

                positions[i] = Xi_new
                fitnesses[i] = fit_new

                if fit_new < rabbit_fit:
                    rabbit_fit = float(fit_new)
                    rabbit_pos = Xi_new.copy()

            history["losses"].append(rabbit_fit)
            history["iterations"].append(it)

            if verbose:
                print(f"HHO+EO Iter {it}: Best Loss = {rabbit_fit:.6f}, E = {E:.4f}")

        # En iyi çözümü modele set et
        U_best, V_best = self._position_to_matrices(
            rabbit_pos,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U_best, V_best)

        return history

    def get_name(self) -> str:
        return "HHO+EO"

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
        """Pozisyonun fitness'i: model.compute_loss (MSE + L2)."""
        U, V = self._position_to_matrices(
            position,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U, V)
        loss = model.compute_loss(user_ids, item_ids, ratings, self.regularization)
        return float(loss)

    @staticmethod
    def _build_equilibrium_pool(
        positions: np.ndarray,
        fitnesses: np.ndarray,
    ) -> np.ndarray:
        """En iyi 4 ajan + ortalamaları (EO fikri)."""
        idx_sorted = np.argsort(fitnesses)
        best_indices = idx_sorted[:4]
        top4 = positions[best_indices]
        avg = top4.mean(axis=0)
        pool = np.vstack([top4, avg[None, :]])
        return pool.astype(np.float32)

    @staticmethod
    def _levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
        """
        Lévy flight adımı (Mantegna yöntemi).

        Kullanım: büyük adımlı rastgele sıçramalar için.
        """
        sigma_u = (
            (
                math.gamma(1 + beta)
                * math.sin(math.pi * beta / 2)
                / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
            )
            ** (1.0 / beta)
        )
        sigma_v = 1.0
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, sigma_v, size=dim)
        step = u / (np.abs(v) ** (1.0 / beta) + 1e-12)
        return step.astype(np.float32)

