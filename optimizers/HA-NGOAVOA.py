#!/usr/bin/env python
# ============================================================
#  HA-NGOAVOA: Hybrid Adaptive Northern Goshawk + African Vultures
#  Optimization Algorithm
#
#  Tasarım:
#  - İlk %50 iterasyon : AVOA exploration dominant (F >= 1 bölgesi)
#  - Orta %30           : AVOA exploitation + NGO Phase 1 karma
#  - Son %20            : NGO Phase 2 (Levy flight) dominant
#  - fitness_stagnation_counter > 20 olduğunda NGO fazına zorla geç
#
#  Kıyaslanacak algoritmalar:
#    1. OriginalAVOA  (ebeveyn)
#    2. OriginalNGO   (ebeveyn)
#    3. OriginalGWO   (klasik literatür)
#    4. OriginalOOA   (güçlü rakip)
#    5. OriginalHHO   (güçlü rakip)
# ============================================================

import numpy as np
from mealpy.swarm_based.AVOA import OriginalAVOA


class HA_NGOAVOA(OriginalAVOA):
    """
    Hybrid Adaptive NGO-AVOA (HA-NGOAVOA)

    Faz yapısı
    ----------
    ratio = epoch / max_epoch

    ratio < 0.5  →  AVOA exploration (|F| >= 1) dominant
                    NGO Phase 1 yardımcı (greedy kabul)

    0.5 <= ratio < 0.8  →  AVOA exploitation (|F| < 1) dominant
                            NGO Phase 1 ek çeşitlilik sağlar

    ratio >= 0.8  →  NGO Phase 2 (Levy flight) dominant
                     AVOA exploitation yedek

    Stagnation mekanizması
    ----------------------
    fitness_stagnation_counter : her iterasyonda gbest değişmezse +1,
                                  iyileşme olursa sıfırlanır.
    counter >= stagnation_threshold (varsayılan 20) →
        switch_to_ngo = True → NGO Phase 2'ye zorla geç

    Parametreler
    ------------
    epoch : int       - maksimum iterasyon sayısı (varsayılan 500)
    pop_size : int    - popülasyon büyüklüğü (varsayılan 30)
    p1 : float        - AVOA faz 1 olasılığı (varsayılan 0.6)
    p2 : float        - AVOA faz 2 olasılığı (varsayılan 0.4)
    p3 : float        - AVOA faz 3 olasılığı (varsayılan 0.6)
    alpha : float     - AVOA en iyi bireyi seçme ağırlığı (varsayılan 0.8)
    gama : float      - AVOA F hesaplama faktörü (varsayılan 2.5)
    stagnation_threshold : int - NGO'ya geçiş eşiği (varsayılan 20)
    """

    def __init__(
        self,
        epoch: int = 500,
        pop_size: int = 30,
        p1: float = 0.6,
        p2: float = 0.4,
        p3: float = 0.6,
        alpha: float = 0.8,
        gama: float = 2.5,
        stagnation_threshold: int = 5,
        **kwargs
    ) -> None:
        super().__init__(
            epoch=epoch, pop_size=pop_size, p1=p1, p2=p2, p3=p3, alpha=alpha, gama=gama, **kwargs
        )
        self.stagnation_threshold = self.validator.check_int(
            "stagnation_threshold", stagnation_threshold, [1, 1000]
        )
        self.set_parameters([
            "epoch", "pop_size", "p1", "p2", "p3",
            "alpha", "gama", "stagnation_threshold"
        ])
        self.sort_flag = False
        self.is_parallelizable = False

        # Stagnation takip değişkenleri — solve() başlamadan önce sıfırlanır
        self.fitness_stagnation_counter = 0
        self.switch_to_ngo = False
        self._prev_gbest_fitness = None
        self._ngo_trigger_count = 0
        self._ngo_accept_count = 0

    # ------------------------------------------------------------------
    # Yardımcı: AVOA faz faktörü F hesapla
    # ------------------------------------------------------------------
    def _avoa_compute_F(self, epoch: int) -> float:
        a = (
            self.generator.uniform(-2, 2)
            * (
                (np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama)
                + np.cos((np.pi / 2) * (epoch / self.epoch))
                - 1
            )
        )
        ppp = (2 * self.generator.random() + 1) * (1 - epoch / self.epoch) + a
        return ppp * (2 * self.generator.random() - 1)

    # ------------------------------------------------------------------
    # Yardımcı: AVOA exploration güncelleme (|F| >= 1)
    # ------------------------------------------------------------------
    def _avoa_exploration(self, idx: int, rand_pos: np.ndarray, F: float) -> np.ndarray:
        if self.generator.random() < self.p1:
            pos_new = rand_pos - (
                np.abs(
                    (2 * self.generator.random()) * rand_pos
                    - self.pop[idx].solution
                )
                * F
            )
        else:
            pos_new = rand_pos - F + self.generator.random() * (
                (self.problem.ub - self.problem.lb) * self.generator.random()
                + self.problem.lb
            )
        return pos_new

    # ------------------------------------------------------------------
    # Yardımcı: AVOA exploitation güncelleme (|F| < 1)
    # ------------------------------------------------------------------
    def _avoa_exploitation(
        self,
        idx: int,
        rand_pos: np.ndarray,
        best_x1: np.ndarray,
        best_x2: np.ndarray,
        F: float,
    ) -> np.ndarray:
        if np.abs(F) < 0.5:  # Phase 1 — rotasyon
            if self.generator.random() < self.p2:
                A = best_x1 - (
                    (best_x1 * self.pop[idx].solution)
                    / (best_x1 - self.pop[idx].solution ** 2 + self.EPSILON)
                ) * F
                B = best_x2 - (
                    (best_x2 * self.pop[idx].solution)
                    / (best_x2 - self.pop[idx].solution ** 2 + self.EPSILON)
                ) * F
                pos_new = (A + B) / 2
            else:
                pos_new = rand_pos - np.abs(
                    rand_pos - self.pop[idx].solution
                ) * F * self.get_levy_flight_step(
                    beta=1.5, multiplier=1.0,
                    size=self.problem.n_dims, case=-1
                )
        else:  # Phase 2 — salınım
            if self.generator.random() < self.p3:
                pos_new = np.abs(
                    (2 * self.generator.random()) * rand_pos - self.pop[idx].solution
                ) * (F + self.generator.random()) - (rand_pos - self.pop[idx].solution)
            else:
                s1 = (
                    rand_pos
                    * (self.generator.random() * self.pop[idx].solution / (2 * np.pi))
                    * np.cos(self.pop[idx].solution)
                )
                s2 = (
                    rand_pos
                    * (self.generator.random() * self.pop[idx].solution / (2 * np.pi))
                    * np.sin(self.pop[idx].solution)
                )
                pos_new = rand_pos - (s1 + s2)
        return pos_new

    # ------------------------------------------------------------------
    # Yardımcı: NGO Phase 1 — exploration (av takibi)
    # ------------------------------------------------------------------
    def _ngo_phase1(self, idx: int) -> np.ndarray:
        kk = self.generator.permutation(self.pop_size)[0]
        if self.compare_target(
            self.pop[kk].target, self.pop[idx].target, self.problem.minmax
        ):
            pos_new = self.pop[idx].solution + self.generator.random(
                self.problem.n_dims
            ) * (
                self.pop[kk].solution
                - self.generator.integers(1, 3) * self.pop[idx].solution
            )
        else:
            pos_new = self.pop[idx].solution + self.generator.random(
                self.problem.n_dims
            ) * (self.pop[idx].solution - self.pop[kk].solution)
        return pos_new

    # ------------------------------------------------------------------
    # Yardımcı: NGO Phase 2 — exploitation (Levy flight)
    # ------------------------------------------------------------------
    def _ngo_phase2(self, idx: int, epoch: int) -> np.ndarray:
        R = 0.02 * (1.0 - epoch / self.epoch)
        pos_new = self.pop[idx].solution + (
            -R + 2 * R * self.generator.random(self.problem.n_dims)
        ) * self.pop[idx].solution
        return pos_new

    # ------------------------------------------------------------------
    # Yardımcı: AVOA'nın orijinal evolve akışı (mealpy OriginalAVOA ile aynı)
    # ------------------------------------------------------------------
    def _avoa_full_evolve(self, epoch: int):
        a = self.generator.uniform(-2, 2) * (
            (np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama)
            + np.cos((np.pi / 2) * (epoch / self.epoch))
            - 1
        )
        ppp = (2 * self.generator.random() + 1) * (1 - epoch / self.epoch) + a
        _, best_list, _ = self.get_special_agents(
            self.pop, n_best=2, minmax=self.problem.minmax
        )
        pop_new = []
        for idx in range(0, self.pop_size):
            F = ppp * (2 * self.generator.random() - 1)
            rand_idx = self.generator.choice([0, 1], p=[self.alpha, 1 - self.alpha])
            rand_pos = best_list[rand_idx].solution
            if np.abs(F) >= 1:  # Exploration
                if self.generator.random() < self.p1:
                    pos_new = rand_pos - (
                        np.abs((2 * self.generator.random()) * rand_pos - self.pop[idx].solution)
                    ) * F
                else:
                    pos_new = rand_pos - F + self.generator.random() * (
                        (self.problem.ub - self.problem.lb) * self.generator.random()
                        + self.problem.lb
                    )
            else:  # Exploitation
                if np.abs(F) < 0.5:  # Phase 1
                    best_x1 = best_list[0].solution
                    best_x2 = best_list[1].solution
                    if self.generator.random() < self.p2:
                        A = best_x1 - (
                            (best_x1 * self.pop[idx].solution)
                            / (best_x1 - self.pop[idx].solution ** 2 + self.EPSILON)
                        ) * F
                        B = best_x2 - (
                            (best_x2 * self.pop[idx].solution)
                            / (best_x2 - self.pop[idx].solution ** 2 + self.EPSILON)
                        ) * F
                        pos_new = (A + B) / 2
                    else:
                        pos_new = rand_pos - np.abs(rand_pos - self.pop[idx].solution) * F * \
                                  self.get_levy_flight_step(
                                      beta=1.5, multiplier=1.0, size=self.problem.n_dims, case=-1
                                  )
                else:  # Phase 2
                    if self.generator.random() < self.p3:
                        pos_new = np.abs(
                            (2 * self.generator.random()) * rand_pos - self.pop[idx].solution
                        ) * (F + self.generator.random()) - (
                            rand_pos - self.pop[idx].solution
                        )
                    else:
                        s1 = rand_pos * (
                            self.generator.random() * self.pop[idx].solution / (2 * np.pi)
                        ) * np.cos(self.pop[idx].solution)
                        s2 = rand_pos * (
                            self.generator.random() * self.pop[idx].solution / (2 * np.pi)
                        ) * np.sin(self.pop[idx].solution)
                        pos_new = rand_pos - (s1 + s2)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)

    # ------------------------------------------------------------------
    # Yardımcı: En kötü n bireyin indekslerini bul
    # ------------------------------------------------------------------
    def _get_worst_indices(self, n_replace: int):
        if self.pop is None or len(self.pop) == 0:
            return []
        pairs = [(idx, float(agent.target.fitness)) for idx, agent in enumerate(self.pop)]
        reverse = self.problem.minmax == "min"
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=reverse)
        return [idx for idx, _ in pairs_sorted[:n_replace]]

    # ------------------------------------------------------------------
    # Stagnation güncelle
    # ------------------------------------------------------------------
    def _update_stagnation(self):
        current_fitness = self.g_best.target.fitness
        if self._prev_gbest_fitness is None:
            self._prev_gbest_fitness = current_fitness
            return

        improved = (
            current_fitness < self._prev_gbest_fitness
            if self.problem.minmax == "min"
            else current_fitness > self._prev_gbest_fitness
        )

        if improved:
            self.fitness_stagnation_counter = 0
            self.switch_to_ngo = False
            self._prev_gbest_fitness = current_fitness
        else:
            self.fitness_stagnation_counter += 1
            if self.fitness_stagnation_counter >= self.stagnation_threshold:
                self.switch_to_ngo = True

    # ------------------------------------------------------------------
    # Ana evrim döngüsü
    # ------------------------------------------------------------------
    def evolve(self, epoch: int):
        """
        Her iterasyonda çağrılır. epoch: 0-indexed mevcut iterasyon.
        """
        # 1. AVOA'nın orijinal evolve'unu çalıştır
        super().evolve(epoch)

        # 2. NGO perturbation — AVOA'nın güncellediği pop üzerinde
        ratio = epoch / self.epoch
        ngo_rate = 0.3 * (1 - ratio)
        n_replace = max(1, int(ngo_rate * self.pop_size))

        sorted_pop = sorted(
            range(self.pop_size),
            key=lambda i: self.pop[i].target.fitness,
            reverse=(self.problem.minmax == "min"),
        )
        worst_indices = sorted_pop[:n_replace]

        for idx in worst_indices:
            pos_new = self._ngo_phase2(idx, epoch)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self._ngo_accept_count += 1