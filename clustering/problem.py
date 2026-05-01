"""Kümeleme uygunluğu ve atama için problem sarmalı."""

from __future__ import annotations

import numpy as np

from core.fitness import FitnessEvaluator


class ClusteringProblem:
    """Düzleştirilmiş centroid vektörü için metaheuristic uyumlu nesne."""

    def __init__(self, clustering_matrix: np.ndarray, K: int, metric: str = "pearson") -> None:
        self.K = K
        self.n_items = int(clustering_matrix.shape[1])
        self._eval = FitnessEvaluator(clustering_matrix, K, distance_metric=metric)

    @property
    def dim(self) -> int:
        return int(self.K * self.n_items)

    def fitness(self, flat: np.ndarray) -> float:
        return float(self._eval(np.asarray(flat, dtype=np.float64).ravel()))

    def assign(self, flat: np.ndarray) -> np.ndarray:
        C = flat.reshape(self.K, self.n_items)
        return self._eval.assign_clusters(C)
