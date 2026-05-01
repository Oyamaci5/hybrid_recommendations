"""
optimizers/doa.py
-----------------
Dhole Optimization Algorithm (DOA) — küme merkezi optimizasyonu için.

Kaynak: Mohammed, Aghdasi, Salehpour — Cluster Computing (2025) 28:430

MF embedding'lerini değil, küme centroid'lerini optimize eder.
Bu nedenle BaseOptimizer interface'ini kullanmaz; mevcut
clustering-CF pipeline'ına entegre olmak üzere tasarlanmıştır.

Interface::

    from core.fitness import make_fitness_fn
    from optimizers.doa import DOA

    fn  = make_fitness_fn(train_matrix, K=30)
    doa = DOA(pop_size=30, max_iter=100, dim=K * n_items, lb=1.0, ub=5.0)
    best_pos, best_fit, curve = doa.optimize(fn, init_population=init_pop)

Dönüş değeri `best_pos` shape (K * n_items,) düzleştirilmiş centroid vektörü.
models/cluster_manager.py ile küme atamasına dönüştürülür.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


class DOA:
    """
    Dhole Optimization Algorithm.

    Makale Algorithm 1'ine birebir karşılık gelir.
    Yorum satırlarındaki Eq. numaraları makale denklemleriyle eşleştirilmiştir.

    Üç faz
    ------
    Arama  (Searching)   [Eq.6-7]  : vocalization < 0.5 VE PMN < 10
    Çevreleme (Encircling)[Eq.8-9] : vocalization < 0.5 VE PMN >= 10
    Saldırı (Attacking)  [Eq.10-13]: vocalization >= 0.5

    Parameters
    ----------
    pop_size : int
        Popülasyon büyüklüğü N.
    max_iter : int
        Maksimum iterasyon T.
    dim : int
        Arama uzayı boyutu (K_clusters * n_items).
    lb : float
        Alt sınır — rating uzayı için 1.0.
    ub : float
        Üst sınır — rating uzayı için 5.0.
    C1 : float
        Prey size kontrol sabiti [Eq.4]. Varsayılan: 1.0.
    mu : float
        En uygun paket üye sayısı μ [Eq.4]. Varsayılan: 25.0.
    K_param : float
        Avcılık verimlilik katsayısı [Eq.4]. Varsayılan: 0.5.
    C3 : float
        Prey factor (sabit = 3) [Eq.10].
    EF : float
        Çevresel faktör [Eq.4]. Varsayılan: 1.0.
    seed : int
        Tekrarlanabilirlik tohumu.
    verbose : bool
        Her 10 iterasyonda fitness yazdır.
    """

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 100,
        dim: int = None,
        lb: float = 1.0,
        ub: float = 5.0,
        C1: float = 1.0,
        mu: float = 25.0,
        K_param: float = 0.5,
        C3: float = 3.0,
        EF: float = 1.0,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if dim is None:
            raise ValueError("dim gereklidir: K_clusters * n_items.")
        self.N        = pop_size
        self.T        = max_iter
        self.dim      = dim
        self.lb       = lb
        self.ub       = ub
        self.C1       = C1
        self.mu       = mu
        self.K_param  = K_param
        self.C3       = C3
        self.EF       = EF
        self.verbose  = verbose
        self.rng      = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Yardımcı: Pack Member Number  [Eq.3]
    # ------------------------------------------------------------------
    def _pmn(self) -> int:
        """Eq.3 — PMN = round(rand * 15 + 5)  ∈ [5, 20]"""
        return int(round(self.rng.random() * 15 + 5))

    # ------------------------------------------------------------------
    # Yardımcı: Suitable Hunting Time  [Eq.4]
    # ------------------------------------------------------------------
    def _ps(self, PMN: int) -> float:
        """Eq.4 — ps = (C1 / (1 + exp(K*(PMN-mu))))^2 * EF"""
        sigmoid = self.C1 / (1.0 + np.exp(self.K_param * (PMN - self.mu)))
        return float((sigmoid ** 2) * self.EF)

    # ------------------------------------------------------------------
    # Yardımcı: Prey pozisyonu  [Eq.5]
    # ------------------------------------------------------------------
    def _prey(self, local_best: np.ndarray, global_best: np.ndarray) -> np.ndarray:
        """Eq.5 — prey = (prey_local + prey_global) / 2"""
        return (local_best + global_best) / 2.0

    # ------------------------------------------------------------------
    # Faz 1 — Arama  [Eq.6, Eq.7]
    # ------------------------------------------------------------------
    def _searching(self, x: np.ndarray, prey: np.ndarray, t: int) -> np.ndarray:
        """Eq.7: C2 = 1 - t/T  |  Eq.6: x_new = x + C2*rand*(prey - x)"""
        C2   = 1.0 - t / self.T                        # Eq.7
        rand = self.rng.random(self.dim)
        return x + C2 * rand * (prey - x)              # Eq.6

    # ------------------------------------------------------------------
    # Faz 2 — Çevreleme  [Eq.8, Eq.9]
    # ------------------------------------------------------------------
    def _encircling(self, x: np.ndarray, pop: np.ndarray,
                    prey: np.ndarray, i: int) -> np.ndarray:
        """Eq.9: z != i rastgele  |  Eq.8: x_new = x - pop[z] + prey"""
        z = i
        while z == i:
            z = int(round(self.rng.random() * (self.N - 1)))   # Eq.9
        return x - pop[z] + prey                                # Eq.8

    # ------------------------------------------------------------------
    # Faz 3 — Saldırı  [Eq.10–13]
    # ------------------------------------------------------------------
    def _attacking(
        self,
        x: np.ndarray,
        fit_x: float,
        fit_prey: float,
        local_best: np.ndarray,
        global_best: np.ndarray,
        ps: float,
    ) -> np.ndarray:
        """
        Eq.10: S = C3 * rand * (fit_x / fit_prey)
        Eq.11: W  = exp(-1/S) * prey_local          [S > threshold]
        Eq.12: x += W*ps*(cos-sin)*W*ps             [S > threshold]
        Eq.13: x  = (x - global)*ps + ps*rand*x     [S <= threshold]
        """
        fit_prey = fit_prey if abs(fit_prey) > 1e-10 else 1e-10
        S = self.C3 * self.rng.random() * (fit_x / fit_prey)   # Eq.10
        threshold = (self.C3 + 1) / 2.0                        # = 2.0

        if S > threshold:
            W    = np.exp(-1.0 / S) * local_best               # Eq.11
            r1   = self.rng.random(self.dim)
            r2   = self.rng.random(self.dim)
            cs   = np.cos(2 * np.pi * r1) - np.sin(2 * np.pi * r2)
            return x + W * ps * cs * W * ps                    # Eq.12
        else:
            rand = self.rng.random(self.dim)
            return (x - global_best) * ps + ps * rand * x      # Eq.13

    # ------------------------------------------------------------------
    # Ana optimizasyon döngüsü — Algorithm 1
    # ------------------------------------------------------------------
    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        init_population: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        DOA ana döngüsü.

        Parameters
        ----------
        fitness_fn : Callable[[np.ndarray shape (dim,)], float]
            Minimize edilecek fonksiyon.
            Genellikle ``core.fitness.make_fitness_fn`` ile oluşturulur.
        init_population : np.ndarray | None, shape (pop_size, dim)
            Verilirse başlangıç popülasyonu olarak kullanılır. Verilmezse
            [lb, ub] aralığında rastgele başlatma yapılır.

        Returns
        -------
        best_solution : np.ndarray, shape (dim,)
            En iyi düzleştirilmiş centroid vektörü.
        best_fitness : float
            En düşük fitness değeri.
        convergence_curve : list[float]
            Her iterasyondaki en iyi fitness.
        """
        # Adım 3: Popülasyon başlatma
        if init_population is not None:
            pop = np.asarray(init_population, dtype=np.float64)
            if pop.shape != (self.N, self.dim):
                raise ValueError(
                    f"init_population shape must be ({self.N}, {self.dim}), "
                    f"got {pop.shape}"
                )
            pop = np.clip(pop, self.lb, self.ub)
        else:
            pop = self.rng.uniform(self.lb, self.ub, (self.N, self.dim))
        fit = np.array([fitness_fn(pop[i]) for i in range(self.N)])

        g_best_pos = pop[np.argmin(fit)].copy()
        g_best_fit = float(fit.min())
        curve: list[float] = []

        # Adım 6: t = 1..T
        for t in range(1, self.T + 1):
            PMN = self._pmn()                               # Adım 7  [Eq.3]
            ps  = self._ps(PMN)

            # prey: mevcut lokal en iyi + global en iyi  [Eq.5]
            l_best_pos = pop[np.argmin(fit)].copy()
            prey = self._prey(l_best_pos, g_best_pos)      # Adım 8

            fit_prey = fitness_fn(g_best_pos)

            for i in range(self.N):                         # Adım 9
                vocal = self.rng.random()                   # Adım 10

                if vocal < 0.5:                             # Adım 11
                    if PMN < 10:                            # Adım 12
                        x_new = self._searching(pop[i], prey, t)
                    else:
                        x_new = self._encircling(pop[i], pop, prey, i)
                else:                                       # Adım 17
                    x_new = self._attacking(
                        pop[i], fit[i], fit_prey,
                        l_best_pos, g_best_pos, ps,
                    )

                x_new = np.clip(x_new, self.lb, self.ub)   # sınır
                f_new = fitness_fn(x_new)                   # Adım 27

                if f_new < fit[i]:
                    pop[i] = x_new
                    fit[i] = f_new
                    if f_new < g_best_fit:
                        g_best_pos = x_new.copy()
                        g_best_fit = f_new

            curve.append(g_best_fit)
            if self.verbose and t % 10 == 0:
                print(f"  DOA [{t:>4}/{self.T}] fitness={g_best_fit:.4f}")

        return g_best_pos, g_best_fit, curve

    def get_name(self) -> str:
        return "DOA"
