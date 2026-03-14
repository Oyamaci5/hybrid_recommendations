"""
Hybrid HHO + DBO optimizer for Matrix Factorization.

Hedef:
- HHO'nun kaçış enerjisi temelli keşif/sömürü dengesini,
- DBO'nun dört ajan tipine (ball-rolling, brood ball, small beetle, thief)
  dayalı zengin arama stratejileri ile birleştirmek.

Her ajan (dung beetle / hawk), tam bir MF çözümünü temsil eder:
    position = [U.flatten(), V.flatten()]
"""

from __future__ import annotations

from typing import Dict, Tuple

import math
import numpy as np

from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class HHODBOOptimizer(BaseOptimizer):
    """
    HHO+DBO hibrit optimizer for Matrix Factorization.

    Temel fikir:
    - Popülasyon: N ajan
    - Her iterasyonda:
      * HHO tarafı için kaçış enerjisi E hesaplanır (exploration / exploitation)
      * DBO tarafında ajanlar dört role ayrılır:
        - Ball-rolling beetles
        - Brood balls
        - Small beetles (foraging)
        - Thieves
      * Her ajan için:
        - HHO güncellemesi (Xi_hho)
        - DBO rolüne göre güncelleme (Xi_dbo)
        - Hibrit pozisyon: Xi_new = alpha * Xi_hho + (1 - alpha) * Xi_dbo
          (alpha = |E|; E küçüldükçe DBO bileşeni daha baskın olur).
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

    def optimize(
        self,
        model: MFModel,
        train_ratings: np.ndarray,
        n_iterations: int = 50,
        verbose: bool = False,
    ) -> Dict:
        """
        HHO+DBO hibriti ile MF parametrelerini optimize et.

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

        # Bir önceki iterasyondaki pozisyonlar (DBO'nun xi(t-1) terimleri için)
        prev_positions = positions.copy()

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

        # En iyi ve en kötü ajan (global best sadece burada initialize edilir)
        best_idx = int(np.argmin(fitnesses))
        best_pos = positions[best_idx].copy()
        best_fit = float(fitnesses[best_idx])

        worst_idx = int(np.argmax(fitnesses))
        worst_pos = positions[worst_idx].copy()

        history = {
            "losses": [best_fit],
            "iterations": [0],
        }

        # DBO oranları (toplam N ajanı dört role böler)
        # Örnek: N=30 -> 6,6,7,11 (makaledeki örneğe yakın)
        def split_indices(n: int) -> Tuple[int, int, int, int]:
            n_ball = max(1, int(0.2 * n))
            n_brood = max(1, int(0.2 * n))
            n_small = max(1, int(0.2 * n))
            n_thief = n - (n_ball + n_brood + n_small)
            if n_thief <= 0:
                n_thief = 1
                n_small = max(1, n_small - 1)
            return n_ball, n_brood, n_small, n_thief

        n_ball, n_brood, n_small, n_thief = split_indices(self.n_agents)

        for it in range(1, n_iterations + 1):
            # Ortalama pozisyon (HHO formüllerinde kullanılıyor)
            mean_pos = np.mean(positions, axis=0)

            # HHO kaçış enerjisi:
            # E0_rand = 2 * rand() - 1, E = 2 * E0_rand * (1 - t/T) * escape_energy_initial
            E0_rand = 2.0 * np.random.rand() - 1.0
            E = 2.0 * E0_rand * (1.0 - it / float(n_iterations))
            E *= self.escape_energy_initial

            # DBO parametreleri
            k_dbo = 0.1  # deflection coefficient
            b_dbo = 0.3
            S_thief = 1.0
            R = 1.0 - it / float(n_iterations)

            # Elitizm: global en iyi ajanı popülasyonda koru
            best_idx = int(np.argmin(fitnesses))
            positions[best_idx] = best_pos.copy()
            fitnesses[best_idx] = best_fit

            # En kötü ajanı güncelle (DBO rollerinde X_worst için)
            worst_idx = int(np.argmax(fitnesses))
            worst_pos = positions[worst_idx].copy()

            # DBO için X* (local best) ve Xb (global best) aynı alınabilir
            X_star = best_pos
            X_best = best_pos
            X_worst = worst_pos

            new_positions = np.zeros_like(positions)
            new_fitnesses = np.zeros_like(fitnesses)

            # Rol sınırları
            idx_ball_end = n_ball
            idx_brood_end = idx_ball_end + n_brood
            idx_small_end = idx_brood_end + n_small

            for i in range(self.n_agents):
                Xi = positions[i]
                Xi_prev = prev_positions[i]

                # --------------------------------------------------
                # 1) HHO bileşeni (Xi_hho)
                # --------------------------------------------------
                if abs(E) >= 1.0:
                    # Keşif fazı
                    q = np.random.rand()
                    if q >= 0.5:
                        # X_i = X_best - X_mean - E * |J * X_best - X_i|
                        J_vec = 2.0 * np.random.rand(dim_total) - 1.0
                        Xi_hho = X_best - mean_pos - E * np.abs(J_vec * X_best - Xi)
                    else:
                        # X_r = rastgele ajan
                        rand_idx = np.random.randint(0, self.n_agents)
                        X_r = positions[rand_idx]
                        r_vec = np.random.rand(dim_total)
                        Xi_hho = X_r - E * np.abs(X_r - 2.0 * r_vec * Xi)
                else:
                    # Sömürü fazı
                    r1 = np.random.rand()
                    J_vec = 2.0 * np.random.rand(dim_total) - 1.0
                    if r1 >= 0.5 and abs(E) >= 0.5:
                        # Yumuşak besiege
                        Xi_hho = X_best - E * np.abs(J_vec * X_best - Xi)
                    else:
                        # Sert besiege + Lévy
                        levy_step = self._levy_flight(dim_total)
                        Xi_hho = (
                            X_best
                            - E * np.abs(J_vec * X_best - mean_pos)
                            + np.random.rand() * levy_step
                        )

                # --------------------------------------------------
                # 2) DBO bileşeni (Xi_dbo) – role bağlı
                # --------------------------------------------------
                if i < idx_ball_end:
                    # Ball-rolling dung beetles (Eşitlik (1) + opsiyonel dans (2))
                    alpha_sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    delta_x = np.abs(Xi - X_worst)
                    Xi_dbo = Xi + alpha_sign * k_dbo * Xi_prev + b_dbo * delta_x

                    # Basit bir "dance" adımı: küçük bir olasılıkla eq. (2)
                    if np.random.rand() < 0.3:
                        theta = np.random.rand() * math.pi
                        Xi_dbo = Xi_dbo + math.tan(theta) * np.abs(Xi - Xi_prev)
                elif i < idx_brood_end:
                    # Brood balls (Eşitlik (3)-(4))
                    Lb_star = np.maximum(X_star * (1.0 - R), lb)
                    Ub_star = np.minimum(X_star * (1.0 + R), ub)
                    b1 = np.random.rand(dim_total)
                    b2 = np.random.rand(dim_total)
                    Xi_dbo = X_star + b1 * (Xi - Lb_star) + b2 * (Xi - Ub_star)
                elif i < idx_small_end:
                    # Small dung beetles (foraging) (Eşitlik (5)-(6))
                    Lb_b = np.maximum(X_best * (1.0 - R), lb)
                    Ub_b = np.minimum(X_best * (1.0 + R), ub)
                    C1 = np.random.normal(0.0, 1.0, size=dim_total)
                    C2 = np.random.rand(dim_total)
                    Xi_dbo = Xi + C1 * (Xi - Lb_b) + C2 * (Xi - Ub_b)
                else:
                    # Thieves (Eşitlik (7))
                    g = np.random.normal(0.0, 1.0, size=dim_total)
                    Xi_dbo = X_best + S_thief * g * (
                        np.abs(Xi - X_star) + np.abs(Xi - X_best)
                    )

                # --------------------------------------------------
                # 3) Hibrit birleştirme
                # --------------------------------------------------
                alpha = min(1.0, max(0.0, abs(E)))
                Xi_new = alpha * Xi_hho + (1.0 - alpha) * Xi_dbo

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

                new_positions[i] = Xi_new
                new_fitnesses[i] = fit_new

                if fit_new < best_fit:
                    best_fit = float(fit_new)
                    best_pos = Xi_new.copy()

            # Iterasyon sonu: pozisyonları ve fitness'leri güncelle
            prev_positions = positions.copy()
            positions = new_positions
            fitnesses = new_fitnesses

            history["losses"].append(best_fit)
            history["iterations"].append(it)

            if verbose:
                print(f"HHO+DBO Iter {it}: Best Loss = {best_fit:.6f}, E = {E:.4f}")

        # En iyi çözümü modele set et
        U_best, V_best = self._position_to_matrices(
            best_pos,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U_best, V_best)

        return history

    def get_name(self) -> str:
        return "HHO+DBO"

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
    def _levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
        """
        Lévy flight adımı (Mantegna yöntemi).
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

