"""
Cluster assignment management utilities.

`ClusterManager` supports two modes:
1) Fit mode: infer labels from centroids + matrix (legacy flow).
2) Assignment mode: load existing `assignments.npy` and optional
   `gray_sheep_mask.npy` (generate_assignments-based flow).
"""

from __future__ import annotations

import numpy as np
from typing import Literal


class ClusterManager:
    """
    Optimize edilmiş centroid'leri kullanarak küme atamasını yönetir.

    Parameters
    ----------
    K : int
        Küme sayısı.
    """

    def __init__(self, K: int) -> None:
        self.K = K
        self._labels: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._gray_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        centroids: np.ndarray,
        train_matrix: np.ndarray,
        fitness_evaluator=None,
        distance_metric: Literal["pearson", "euclidean"] = "pearson",
    ) -> np.ndarray:
        """
        Centroid'leri kaydeder ve kullanıcı atamasını yapar.

        Atama için ``fitness_evaluator.assign_clusters`` (core.fitness.FitnessEvaluator)
        kullanılır; yoksa pearson_distance tabanlı brute-force atama yapılır.

        Parameters
        ----------
        centroids : np.ndarray, shape (K, n_items)
        train_matrix : np.ndarray, shape (n_users, n_items)
        fitness_evaluator : FitnessEvaluator | None

        Returns
        -------
        labels : np.ndarray, shape (n_users,)
        """
        self._centroids = centroids.copy()

        if fitness_evaluator is not None:
            self._labels = fitness_evaluator.assign_clusters(centroids)
        else:
            self._labels = self._assign_brute(
                centroids, train_matrix, distance_metric=distance_metric
            )

        return self._labels

    def load_assignments(
        self,
        assignments: np.ndarray,
        gray_sheep_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Load precomputed cluster assignments from disk or memory.

        Parameters
        ----------
        assignments : np.ndarray, shape (n_users,)
            Cluster IDs for each user.
        gray_sheep_mask : np.ndarray | None, shape (n_users,)
            Optional boolean mask. True means gray sheep.
        """
        labels = np.asarray(assignments, dtype=np.int32).reshape(-1)
        if labels.size == 0:
            raise ValueError("assignments cannot be empty")
        # 0..K-1: white clusters, K: gray sheep cluster (optional)
        if np.any(labels < 0) or np.any(labels > self.K):
            raise ValueError(f"assignments must be in [0, {self.K}]")
        self._labels = labels

        if gray_sheep_mask is not None:
            gray = np.asarray(gray_sheep_mask).reshape(-1).astype(bool)
            if gray.shape[0] != labels.shape[0]:
                raise ValueError("gray_sheep_mask length must match assignments")
            self._gray_mask = gray
        else:
            self._gray_mask = np.zeros_like(labels, dtype=bool)

        return self._labels

    def _assign_brute(
        self,
        centroids: np.ndarray,
        rating_matrix: np.ndarray,
        distance_metric: Literal["pearson", "euclidean"] = "pearson",
    ) -> np.ndarray:
        """Seçilen mesafe metriği ile brute-force küme atama (fallback)."""
        from core.metrics import pearson_distance

        n_users = rating_matrix.shape[0]
        labels = np.zeros(n_users, dtype=int)
        for u in range(n_users):
            if distance_metric == "euclidean":
                idx = np.where(rating_matrix[u] != 0)[0]
                if len(idx) == 0:
                    dists = [float("inf")] * self.K
                else:
                    dists = [
                        float(np.linalg.norm(centroids[k, idx] - rating_matrix[u, idx]))
                        for k in range(self.K)
                    ]
            else:
                dists = [
                    pearson_distance(rating_matrix[u], centroids[k])
                    for k in range(self.K)
                ]
            labels[u] = int(np.argmin(dists))
        return labels

    # ------------------------------------------------------------------
    # Sorgular
    # ------------------------------------------------------------------

    def get_user_cluster(self, user_id: int) -> int:
        """Kullanıcının küme ID'sini döndürür."""
        self._check_fitted()
        return int(self._labels[user_id])

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """Kümedeki tüm kullanıcı indekslerini döndürür."""
        self._check_fitted()
        return np.where(self._labels == cluster_id)[0]

    def get_white_members(self, cluster_id: int) -> np.ndarray:
        """Return non-gray users in a cluster."""
        self._check_fitted()
        self._check_gray_ready()
        members = self.get_cluster_members(cluster_id)
        return members[~self._gray_mask[members]]

    def get_gray_members(self, cluster_id: int) -> np.ndarray:
        """Return gray sheep users in a cluster."""
        self._check_fitted()
        self._check_gray_ready()
        members = self.get_cluster_members(cluster_id)
        return members[self._gray_mask[members]]

    @property
    def labels(self) -> np.ndarray:
        self._check_fitted()
        return self._labels

    @property
    def centroids(self) -> np.ndarray:
        self._check_fitted()
        return self._centroids

    @property
    def gray_mask(self) -> np.ndarray:
        self._check_fitted()
        self._check_gray_ready()
        return self._gray_mask

    # ------------------------------------------------------------------
    # İstatistik
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Küme boyut istatistiklerini döndürür."""
        self._check_fitted()
        max_label = int(np.max(self._labels)) if len(self._labels) > 0 else self.K - 1
        upper = max(self.K - 1, max_label)
        sizes = {k: int((self._labels == k).sum()) for k in range(upper + 1)}
        vals = list(sizes.values())
        gray_ratio = None
        if self._gray_mask is not None and len(self._gray_mask) == len(self._labels):
            gray_ratio = float(self._gray_mask.mean())
        return {
            "per_cluster": sizes,
            "min_size":    min(vals),
            "max_size":    max(vals),
            "mean_size":   round(float(np.mean(vals)), 1),
            "empty":       sum(1 for v in vals if v == 0),
            "gray_ratio":  gray_ratio,
        }

    def _check_fitted(self) -> None:
        if self._labels is None:
            raise RuntimeError("fit() henüz çağrılmadı.")

    def _check_gray_ready(self) -> None:
        if self._gray_mask is None:
            raise RuntimeError("gray sheep mask is not loaded.")
