"""
models/baselines.py
-------------------
Clustering-CF karşılaştırması için baseline algoritmalar.

Her sınıf DOA ile aynı interface'i uygular::

    optimize(fitness_fn) → (best_solution, best_fitness, convergence_curve)

Sınıflar
--------
KMeansBaseline   : sklearn KMeans — centroid'leri PCC fitness ile değerlendirir.
PSOBaseline      : Particle Swarm Optimization — standart PSO.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable


# ────────────────────────────────────────────────────────────────────────────
# K-Means Baseline
# ────────────────────────────────────────────────────────────────────────────

class KMeansBaseline:
    """
    sklearn KMeans sarmalayıcı.

    KMeans Öklid mesafesiyle kümeleme yapar; döndürülen centroid'ler
    PCC fitness fonksiyonuyla değerlendirilir — adil karşılaştırma için.

    Parameters
    ----------
    rating_matrix : np.ndarray, shape (n_users, n_items)
        KMeans'in fit edeceği veri.
    K : int
        Küme sayısı.
    max_iter : int
        KMeans iç iterasyon limiti.
    n_init : int
        KMeans yeniden başlatma sayısı.
    seed : int
    """

    def __init__(
        self,
        rating_matrix: np.ndarray,
        K: int = 30,
        max_iter: int = 300,
        n_init: int = 10,
        seed: int = 42,
    ) -> None:
        self.rating_matrix = rating_matrix
        self.K = K
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed

    def optimize(
        self, fitness_fn: Callable[[np.ndarray], float]
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        KMeans centroid'lerini bulur ve PCC fitness ile değerlendirir.

        Returns
        -------
        best_solution : np.ndarray, shape (K * n_items,)
        best_fitness : float
        convergence_curve : list[float]  (boş — tek aşamalı)
        """
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=self.K,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.seed,
        )
        km.fit(self.rating_matrix)
        flat = km.cluster_centers_.ravel()
        return flat, float(fitness_fn(flat)), []

    def get_name(self) -> str:
        return "KMeans"


# ────────────────────────────────────────────────────────────────────────────
# PSO Baseline
# ────────────────────────────────────────────────────────────────────────────

class _Particle:
    """Tek bir PSO parçacığı."""

    def __init__(self, position: np.ndarray, fitness: float) -> None:
        self.pos       = position.copy()
        self.vel       = np.zeros_like(position)
        self.best_pos  = position.copy()
        self.best_fit  = fitness

    def update(
        self,
        gbest: np.ndarray,
        w: float, c1: float, c2: float,
        rng: np.random.Generator,
        lb: float, ub: float,
    ) -> None:
        r1 = rng.random(self.pos.shape)
        r2 = rng.random(self.pos.shape)
        self.vel = (
            w * self.vel
            + c1 * r1 * (self.best_pos - self.pos)
            + c2 * r2 * (gbest - self.pos)
        )
        self.pos = np.clip(self.pos + self.vel, lb, ub)


class PSOBaseline:
    """
    Standart Particle Swarm Optimization — DOA ile aynı interface.

    Parameters
    ----------
    pop_size : int
    max_iter : int
    dim : int
    lb : float
    ub : float
    w : float — atalet katsayısı
    c1 : float — bilişsel faktör
    c2 : float — sosyal faktör
    seed : int
    verbose : bool
    """

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 100,
        dim: int = None,
        lb: float = 1.0,
        ub: float = 5.0,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if dim is None:
            raise ValueError("dim gereklidir.")
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim      = dim
        self.lb       = lb
        self.ub       = ub
        self.w        = w
        self.c1       = c1
        self.c2       = c2
        self.verbose  = verbose
        self.rng      = np.random.default_rng(seed)

    def optimize(
        self, fitness_fn: Callable[[np.ndarray], float]
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        PSO döngüsü.

        Returns
        -------
        best_solution : np.ndarray, shape (dim,)
        best_fitness : float
        convergence_curve : list[float]
        """
        swarm = [
            _Particle(
                self.rng.uniform(self.lb, self.ub, self.dim),
                float("inf"),
            )
            for _ in range(self.pop_size)
        ]
        for p in swarm:
            p.best_fit = fitness_fn(p.pos)

        gbest = min(swarm, key=lambda p: p.best_fit)
        gbest_pos = gbest.best_pos.copy()
        gbest_fit = gbest.best_fit
        curve: list[float] = []

        for t in range(1, self.max_iter + 1):
            for p in swarm:
                p.update(gbest_pos, self.w, self.c1, self.c2,
                         self.rng, self.lb, self.ub)
                f = fitness_fn(p.pos)
                if f < p.best_fit:
                    p.best_fit = f
                    p.best_pos = p.pos.copy()
                    if f < gbest_fit:
                        gbest_fit = f
                        gbest_pos = p.pos.copy()
            curve.append(gbest_fit)
            if self.verbose and t % 10 == 0:
                print(f"  PSO [{t:>4}/{self.max_iter}] fitness={gbest_fit:.4f}")

        return gbest_pos, gbest_fit, curve

    def get_name(self) -> str:
        return "PSO"


@dataclass
class BaselineRecord:
    """
    Lightweight evaluation record for assignment-based pipeline reporting.
    """

    algorithm: str
    strategy: str
    mae: float
    rmse: float
    precision_at_n: float
    recall_at_n: float
    n_predictions: int

    def to_dict(self) -> dict:
        return asdict(self)


def records_to_numpy_table(records: list[BaselineRecord]) -> np.ndarray:
    """
    Convert records into a compact object table for quick CSV export.
    """
    headers = np.array(
        [
            "algorithm",
            "strategy",
            "mae",
            "rmse",
            "precision_at_n",
            "recall_at_n",
            "n_predictions",
        ],
        dtype=object,
    )
    rows = [
        np.array(
            [
                r.algorithm,
                r.strategy,
                r.mae,
                r.rmse,
                r.precision_at_n,
                r.recall_at_n,
                r.n_predictions,
            ],
            dtype=object,
        )
        for r in records
    ]
    return np.vstack([headers, *rows]) if rows else headers.reshape(1, -1)
