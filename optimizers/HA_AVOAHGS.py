#!/usr/bin/env python
# ============================================================
#  HA-AVOAHGS: Hybrid Adaptive AVOA + HGS
#
#  Tasarım:
#  - OriginalAVOA'dan türer → super().evolve() ile AVOA tam çalışır
#  - Her iterasyonda HGS hunger-weighted adımı EN İYİ %30 bireye uygulanır
#  - İterasyon ilerledikçe HGS oranı artar (shrink azalır → exploitation dominant)
#  - Greedy selection: HGS adımı ancak iyileşme sağlarsa kabul edilir
#
#  Neden en iyi %30?
#  - AVOA multimodal/exploration'da zaten güçlü → en kötü bireyleri
#    tekrar karıştırmak NGO+AVOA'da işe yaramadı
#  - HGS'nin W1*g_best hareketi g_best'e yakın bireylerde en verimli
#  - Unimodal fonksiyonlarda AVOA'nın eksik kaldığı exploitation'ı
#    HGS hunger mekanizması tamamlar
#
#  NGO+AVOA deneyinden öğrenilenler:
#  - super().evolve() şart — internal state (p1,p2,p3,alpha,gama) korunmalı
#  - Stagnation'a bağlama — her iterasyonda aktif olmalı
#  - En kötü değil, en iyi bireylere uygula
# ============================================================

import numpy as np
from mealpy.swarm_based.AVOA import OriginalAVOA
from mealpy.utils.agent import Agent


class HA_AVOAHGS(OriginalAVOA):
    """
    Hybrid Adaptive AVOA-HGS (HA-AVOAHGS)

    AVOA'nın exploration + multimodal gücü ile
    HGS'nin hunger-weighted exploitation gücünü birleştirir.

    Parametreler
    ------------
    epoch     : int   - maksimum iterasyon (varsayılan 500)
    pop_size  : int   - popülasyon büyüklüğü (varsayılan 30)
    p1        : float - AVOA faz 1 olasılığı (varsayılan 0.6)
    p2        : float - AVOA faz 2 olasılığı (varsayılan 0.4)
    p3        : float - AVOA faz 3 olasılığı (varsayılan 0.6)
    alpha     : float - AVOA en iyi seçme ağırlığı (varsayılan 0.8)
    gama      : float - AVOA F faktörü (varsayılan 2.5)
    PUP       : float - HGS pozisyon güncelleme olasılığı (varsayılan 0.08)
    LH        : float - HGS en büyük açlık eşiği (varsayılan 10000)
    hgs_rate  : float - her iterasyonda HGS uygulanan en iyi birey oranı
                        (varsayılan 0.3 → en iyi %30)
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
        PUP: float = 0.08,
        LH: float = 10000,
        hgs_rate: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(
            epoch=epoch, pop_size=pop_size,
            p1=p1, p2=p2, p3=p3,
            alpha=alpha, gama=gama,
            **kwargs
        )
        self.PUP = self.validator.check_float("PUP", PUP, (0, 1.0))
        self.LH  = self.validator.check_float("LH",  LH,  [1, 20000])
        self.hgs_rate = self.validator.check_float("hgs_rate", hgs_rate, (0, 1.0))
        self.set_parameters([
            "epoch", "pop_size", "p1", "p2", "p3",
            "alpha", "gama", "PUP", "LH", "hgs_rate"
        ])

        # Sayaçlar — test/debug için
        self._hgs_apply_count = 0
        self._hgs_accept_count = 0

    # ------------------------------------------------------------------
    # HGS yardımcı: sech fonksiyonu
    # ------------------------------------------------------------------
    def _sech(self, x: float) -> float:
        if np.abs(x) > 50:
            return 0.5
        return 2.0 / (np.exp(x) + np.exp(-x))

    # ------------------------------------------------------------------
    # HGS yardımcı: açlık değerlerini güncelle
    # ------------------------------------------------------------------
    def _update_hunger(self, g_best: Agent, g_worst: Agent) -> None:
        """
        HGS Eq.(2.8)-(2.9): her bireyin hunger değerini güncelle.
        Agent'larda .hunger attribute yoksa 1.0 ile başlat.
        """
        space = float(np.mean(self.problem.ub - self.problem.lb))
        fit_range = (
            g_worst.target.fitness - g_best.target.fitness + self.EPSILON
        )
        for agent in self.pop:
            if getattr(agent, "hunger", None) is None:
                agent.hunger = 1.0
            r = self.generator.random()
            H = (
                (agent.target.fitness - g_best.target.fitness)
                / fit_range * r * 2 * space
            )
            if H < self.LH:
                H = self.LH * (1 + r)
            agent.hunger += H
            if g_best.target.fitness == agent.target.fitness:
                agent.hunger = 0.0

    # ------------------------------------------------------------------
    # HGS adımı: tek birey için yeni pozisyon üret
    # ------------------------------------------------------------------
    def _hgs_step(self, idx: int, epoch: int,
                  g_best: Agent, total_hunger: float,
                  shrink: float) -> np.ndarray:
        """
        HGS Eq.(2.1)-(2.4): hunger-weighted pozisyon güncellemesi.
        """
        agent = self.pop[idx]
        if getattr(agent, "hunger", None) is None:
            agent.hunger = 1.0

        E  = self._sech(agent.target.fitness - g_best.target.fitness)
        R  = 2 * shrink * self.generator.random() - shrink   # Eq.(2.3)
        r1 = self.generator.random()
        r2 = self.generator.random()

        if r1 < self.PUP:
            W1 = (
                agent.hunger * self.pop_size
                / (total_hunger + self.EPSILON)
                * self.generator.random()
            )
        else:
            W1 = 1.0

        W2 = (
            (1 - np.exp(-np.abs(agent.hunger - total_hunger)))
            * self.generator.random() * 2
        )

        if r1 < self.PUP:
            pos_new = agent.solution * (1 + self.generator.normal(0, 1))
        else:
            diff = np.abs(g_best.solution - agent.solution)
            if r2 > E:
                pos_new = W1 * g_best.solution + R * W2 * diff
            else:
                pos_new = W1 * g_best.solution - R * W2 * diff

        return pos_new

    # ------------------------------------------------------------------
    # Ana evolve döngüsü
    # ------------------------------------------------------------------
    def evolve(self, epoch: int) -> None:
        """
        1. AVOA tam evolve (super())
        2. HGS hunger-weighted adımı en iyi %hgs_rate bireye uygula
        """
        # ── Adım 1: AVOA orijinal evolve ──────────────────────────────
        super().evolve(epoch)

        # ── Adım 2: HGS perturbation ───────────────────────────────────
        _, (g_best,), (g_worst,) = self.get_special_agents(
            self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax
        )

        # Açlık değerlerini güncelle
        self._update_hunger(g_best, g_worst)

        total_hunger = sum(
            (a.hunger if getattr(a, "hunger", None) is not None else 1.0) for a in self.pop
        )

        # HGS'nin shrink parametresi — iterasyon ilerledikçe azalır
        # Bu exploitation'ı iterasyon sonuna doğru artırır
        shrink = 2.0 * (1.0 - epoch / self.epoch)

        # En iyi %hgs_rate bireyi seç
        n_apply = max(1, int(self.hgs_rate * self.pop_size))
        sorted_indices = sorted(
            range(self.pop_size),
            key=lambda i: self.pop[i].target.fitness,
            reverse=(self.problem.minmax == "max"),
        )
        best_indices = sorted_indices[:n_apply]

        # HGS adımını uygula, greedy kabul et
        for idx in best_indices:
            self._hgs_apply_count += 1
            pos_new = self._hgs_step(
                idx, epoch, g_best, total_hunger, shrink
            )
            pos_new = self.correct_solution(pos_new)
            agent   = self.generate_agent(pos_new)
            if self.compare_target(
                agent.target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = agent
                self._hgs_accept_count += 1