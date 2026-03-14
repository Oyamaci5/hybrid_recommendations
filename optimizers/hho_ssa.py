"""
Hybrid HHO + SSA optimizer for Matrix Factorization.

Her ajan (sparrow / hawk), tam bir MF çözümünü temsil eder:
    position = [U.flatten(), V.flatten()]

Hibrit mantık (özet):
- HHO tarafı:
  * Kaçış enerjisi E ile global keşif/sömürü dengesi
- SSA tarafı:
  * Üreticiler (producers): güvenli / tehlikeli bölgeye göre güncelleme
  * Takipçiler (scroungers): iyi/kötü konuma göre farklı strateji
  * Farkındalık (awareness): en iyiye yaklaşma / en kötüden kaçış
- Hibrit:
  * |E| < 1 iken SSA sömürü adımının üstüne HHO sömürü bileşeni
    eklenir ve |E| azaldıkça SSA ağırlığı artar.
"""

from __future__ import annotations

from typing import Dict, Tuple

import math
import numpy as np

from optimizers.base_optimizer import BaseOptimizer
from models.mf_model import MFModel


class HHOSSAOptimizer(BaseOptimizer):
    """
    HHO+SSA hibrit optimizer for Matrix Factorization.
    """

    def __init__(
        self,
        n_agents: int = 40,
        escape_energy_initial: float = 1.0,
        regularization: float = 0.01,
        boundary: float = 2.0,
        safety_threshold: float = 0.7,
        producer_ratio: float = 0.2,
        awareness_ratio: float = 0.1,
        ssa_alpha: float = 0.8,
    ) -> None:
        """
        Args:
            n_agents: Popülasyon boyutu (sparrow/hawk sayısı)
            escape_energy_initial: HHO için başlangıç kaçış enerjisi katsayısı
            regularization: L2 regularization katsayısı (MF loss için)
            boundary: Arama uzayı sınırı ([-boundary, boundary])
            safety_threshold: SSA güvenlik eşiği (ST, ~[0.6, 0.8])
            producer_ratio: Üretici oranı (PR, ~0.2)
            awareness_ratio: Farkındalık oranı (AR, ~0.1)
            ssa_alpha: Üretici exp() güncellemesindeki ölçek katsayısı (α_ssa)
        """
        self.n_agents = n_agents
        self.escape_energy_initial = escape_energy_initial
        self.regularization = regularization
        self.boundary = boundary
        self.safety_threshold = safety_threshold
        self.producer_ratio = producer_ratio
        self.awareness_ratio = awareness_ratio
        self.ssa_alpha = max(1e-6, float(ssa_alpha))

    # ------------------------------------------------------------------
    # BaseOptimizer arayüzü
    # ------------------------------------------------------------------

    def optimize(
        self,
        model: MFModel,
        train_ratings: np.ndarray,
        n_iterations: int = 200,
        verbose: bool = False,
    ) -> Dict:
        """
        HHO+SSA hibriti ile MF parametrelerini optimize et.

        Args:
            model: MFModel instance
            train_ratings: (n_ratings, 3) [user_id, item_id, rating]
            n_iterations: metaheuristic iterasyon sayısı (T)
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

        # Akıllı başlangıç:
        # - Tüm popülasyon önce küçük normal dağılımla başlar
        # - Ajan 0: mevcut model (warm-start)
        # - Ajan 1..N/2-1: warm-start etrafında küçük pertürbasyon
        # - Ajan N/2..N-1: geniş keşif için rastgele
        scale = 1.0 / math.sqrt(latent_dim)
        positions = np.random.normal(
            0.0, scale, size=(self.n_agents, dim_total)
        ).astype(np.float32)
        positions = np.clip(positions, lb, ub)

        # positions[0]: mevcut model parametreleri (warm-start)
        U_cur, V_cur = model.get_parameters()
        init_pos = np.concatenate(
            [U_cur.flatten(), V_cur.flatten()]
        ).astype(np.float32)
        init_pos = np.clip(init_pos, lb, ub)
        positions[0] = init_pos

        # positions[1..half-1]: warm-start etrafında küçük pertürbasyon
        half = self.n_agents // 2
        for i in range(1, half):
            noise = np.random.normal(
                0.0, 0.05, size=dim_total
            ).astype(np.float32)
            positions[i] = np.clip(init_pos + noise, lb, ub)

        # positions[half..N-1]: geniş keşif için rastgele
        for i in range(half, self.n_agents):
            positions[i] = np.clip(
                np.random.normal(0.0, scale, size=dim_total),
                lb,
                ub,
            ).astype(np.float32)

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

        # En iyi / en kötü ajan
        best_idx = int(np.argmin(fitnesses))
        best_pos = positions[best_idx].copy()
        best_fit = float(fitnesses[best_idx])

        worst_idx = int(np.argmax(fitnesses))
        worst_pos = positions[worst_idx].copy()

        history = {
            "losses": [best_fit],
            "iterations": [0],
        }

        # SSA rol sayıları
        n_producers = max(1, int(self.producer_ratio * self.n_agents))
        n_awareness = max(1, int(self.awareness_ratio * self.n_agents))
        if n_producers + n_awareness >= self.n_agents:
            # En az bir takipçi kalmasını garanti et
            n_awareness = max(1, self.n_agents - n_producers - 1)
        n_scroungers = self.n_agents - n_producers - n_awareness

        eps = 1e-12

        # Global en iyi çözümü hep koru
        global_best_pos = best_pos.copy()
        global_best_fit = best_fit
        last_improvement = global_best_fit
        stagnation_counter = 0
        stagnation_patience = 10

        for it in range(1, n_iterations + 1):
            # Popülasyon ortalaması
            mean_pos = np.mean(positions, axis=0)

            # HHO kaçış enerjisi:
            # E_abs sadece iterasyona bağlı, monoton azalır
            E_abs = 2.0 * (1.0 - it / float(n_iterations))
            # Yön için rastgele işaret
            E = E_abs * np.sign(2.0 * np.random.rand() - 1.0)

            # Fitness'a göre sırala (en iyi -> en kötü)
            sorted_idx = np.argsort(fitnesses)
            producers_idx = sorted_idx[:n_producers]
            awareness_idx = sorted_idx[-n_awareness:]
            scrounger_mask = np.ones(self.n_agents, dtype=bool)
            scrounger_mask[producers_idx] = False
            scrounger_mask[awareness_idx] = False
            scroungers_idx = np.where(scrounger_mask)[0]

            # Set'ler ile daha net rol kontrolü
            producers_set = set(producers_idx.tolist())
            scroungers_set = set(scroungers_idx.tolist())
            awareness_set = set(awareness_idx.tolist())

            # Bu iterasyondaki best yerine TÜM ZAMANLARIN en iyisini kullan
            best_pos = global_best_pos.copy()
            best_fit = global_best_fit

            worst_idx = int(sorted_idx[-1])
            worst_pos = positions[worst_idx].copy()
            worst_fit = float(fitnesses[worst_idx])

            # Greedy güncelleme için kopyayı temel al
            new_positions = positions.copy()
            new_fitnesses = fitnesses.copy()

            for rank, i in enumerate(sorted_idx):
                Xi = positions[i]

                # --------------------------------------------------
                # 1) HHO EXPLORATION (E_abs >= 1)  -> Keşif fazı
                #    Pseudo-koddaki 15. adım
                # --------------------------------------------------
                if E_abs >= 1.0:
                    q = np.random.rand()
                    if q >= 0.5:
                        # Rastgele bireyden uzaklaşma
                        rand_idx = np.random.randint(0, self.n_agents)
                        X_rand = positions[rand_idx]
                        J_vec = 2.0 * np.random.rand(dim_total)  # [0,2]
                        Xi_new = X_rand - E * np.abs(J_vec * best_pos - Xi)
                    else:
                        # Popülasyon ortalamasından uzaklaşma
                        r_vec = np.random.rand(dim_total)
                        Xi_new = (best_pos - mean_pos) - E * np.abs(
                            lb + r_vec * (ub - lb) - Xi
                        )

                # --------------------------------------------------
                # 2) SSA SÖMÜRÜ (E_abs < 1)  -> rollerle güncelleme
                #    Pseudo-koddaki 16–19. adımlar
                # --------------------------------------------------
                else:
                    # SSA rolleri
                    if i in producers_set:
                        # Producer (keşifçi)
                        R2 = np.random.rand()
                        if R2 < self.safety_threshold:
                            # Güvenli bölge: exp ile küçülme
                            # i rank'i 0..N-1, formülde 1..N
                            idx1 = rank + 1.0
                            decay = math.exp(
                                -idx1 / (self.ssa_alpha * float(n_iterations) + eps)
                            )
                            Xi_ssa = Xi * decay
                        else:
                            # Tehlike: normal dağılımla kaçış
                            Xi_ssa = Xi + np.random.randn(dim_total).astype(
                                np.float32
                            )
                    elif i in scroungers_set:
                        # Scrounger (takipçi)
                        if rank >= self.n_agents // 2:
                            # Kötü pozisyondaki takipçi
                            # Xi = randn * exp((X_worst - Xi) / i^2)
                            idx1 = rank + 1.0
                            step = np.exp((worst_pos - Xi) / (idx1**2 + eps))
                            # Patlamayı önlemek için sınırla
                            step = np.clip(step, -10.0, 10.0)
                            Xi_ssa = np.random.randn(dim_total).astype(
                                np.float32
                            ) * step
                        else:
                            # İyi pozisyondaki takipçi
                            A = np.random.choice([-1.0, 1.0], size=dim_total)
                            # A+: 1xD vektör için psödo-invers: A^T / (A A^T)
                            denom = float(np.sum(A * A)) + eps
                            A_pseudo = A / denom  # (D,)
                            L_vec = np.ones(dim_total, dtype=np.float32)
                            scalar = float(np.dot(A_pseudo, L_vec))
                            # scalar'ı makul aralıkta tut
                            scalar = np.clip(scalar, -1.0, 1.0)
                            Xi_ssa = best_pos + np.abs(Xi - best_pos) * scalar
                    elif i in awareness_set:
                        # Awareness (anti-predator)
                        fi = float(fitnesses[i])
                        if fi > best_fit:
                            # En iyi değil, en iyiye yaklaş
                            beta = np.random.randn()
                            beta = np.clip(beta, -2.0, 2.0)
                            Xi_ssa = best_pos + beta * np.abs(Xi - best_pos)
                        else:
                            # En kötüye yakın, kaç
                            K = (2.0 * np.random.rand() - 1.0)  # [-1,1]
                            Xi_ssa = Xi + K * (
                                np.abs(Xi - worst_pos)
                                / (fi - worst_fit + eps)
                            )

                    # --------------------------------------------------
                    # 3) HHO SÖMÜRÜ BİLEŞENİ ve HİBRİT (20. adım)
                    # --------------------------------------------------
                    r1 = np.random.rand()
                    J_vec = 2.0 * np.random.rand(dim_total)  # [0,2]
                    if r1 >= 0.5 and E_abs >= 0.5:
                        # Yumuşak kuşatma
                        X_hho = best_pos - E * np.abs(J_vec * best_pos - Xi_ssa)
                    else:
                        # Sert kuşatma + Lévy uçuşu
                        levy_step = self._levy_flight(dim_total)
                        X_hho = (
                            best_pos
                            - E * np.abs(J_vec * best_pos - mean_pos)
                            + np.random.rand() * levy_step
                        )

                    # Dinamik hibrit ağırlıklar
                    w_hho = min(1.0, max(0.0, E_abs))
                    w_ssa = 1.0 - w_hho
                    Xi_new = w_hho * X_hho + w_ssa * Xi_ssa

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

                # Greedy seçim: warm-start ajanı için katı, diğerleri için esnek
                if i == 0:
                    # Warm-start ajanı: sadece iyileşme kabul et
                    if fit_new < new_fitnesses[i]:
                        new_positions[i] = Xi_new
                        new_fitnesses[i] = fit_new
                        if fit_new < global_best_fit:
                            global_best_fit = float(fit_new)
                            global_best_pos = Xi_new.copy()
                else:
                    # Diğer ajanlar: her zaman güncelle (keşif serbestçe olsun)
                    new_positions[i] = Xi_new
                    new_fitnesses[i] = fit_new
                    if fit_new < global_best_fit:
                        global_best_fit = float(fit_new)
                        global_best_pos = Xi_new.copy()

            # Iterasyon sonu
            positions = new_positions
            fitnesses = new_fitnesses

            # Stagnation kontrolü: 20 iterasyon iyileşme yoksa en kötü %25'i yenile
            if global_best_fit < last_improvement - 1e-10:
                last_improvement = global_best_fit
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_patience:
                n_reset = max(1, int(0.25 * self.n_agents))
                # En kötü ajanları bul (fitness büyükten küçüğe)
                worst_order = np.argsort(fitnesses)[::-1]
                reset_indices = worst_order[:n_reset]
                for idx in reset_indices:
                    positions[idx] = (ub - lb) * np.random.rand(dim_total) + lb
                    positions[idx] = positions[idx].astype(np.float32)
                    new_fit = self._evaluate_position(
                        positions[idx],
                        model,
                        n_users,
                        n_items,
                        latent_dim,
                        user_ids,
                        item_ids,
                        ratings,
                    )
                    fitnesses[idx] = new_fit
                    if new_fit < global_best_fit:
                        global_best_fit = float(new_fit)
                        global_best_pos = positions[idx].copy()
                        last_improvement = global_best_fit
                        stagnation_counter = 0
                # Sonraki iterasyona taze popülasyonla devam

            history["losses"].append(global_best_fit)
            history["iterations"].append(it)

            if verbose:
                print(
                    f"HHO+SSA Iter {it}: "
                    f"Best Loss = {global_best_fit:.6f}, "
                    f"E_abs = {E_abs:.4f}"
                )

        # En iyi çözümü modele set et
        U_best, V_best = self._position_to_matrices(
            global_best_pos,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U_best, V_best)

        return history

    def get_name(self) -> str:
        return "HHO+SSA"

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
        """
        Pozisyonun fitness'i: RMSE + L2 reg.

        Not: RMSE, SGD'nin optimize ettiği metrikle aynı olması için
        doğrudan predict üzerinden hesaplanıyor.
        """
        U, V = self._position_to_matrices(
            position,
            n_users,
            n_items,
            latent_dim,
        )
        model.set_parameters(U, V)

        # Tahminler ve hata
        predictions = model.predict(user_ids, item_ids)
        errors = ratings - predictions

        rmse = float(np.sqrt(np.mean(errors**2)))

        # Regularization (U ve V üzerinde L2, MFModel.compute_loss ile aynı ölçekte)
        reg = self.regularization * (
            float(np.mean(U**2)) + float(np.mean(V**2))
        )

        return rmse + reg

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

