"""Kümeli CF öneri katmanı."""

from __future__ import annotations

import numpy as np

from core.metrics import predict_rating


class CFRecommender:
    """Dokümantasyondaki Yol 1–3 CF adımı (küme içi komşular + ağırlıklı tahmin)."""

    def __init__(
        self,
        train_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
        top_k: int = 30,
        distance_metric: str = "pearson",
    ) -> None:
        self.train_matrix = np.asarray(train_matrix, dtype=np.float32)
        self.cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.top_k = int(top_k)
        self.distance_metric = distance_metric

    def predict(self, user_id: int, item_id: int) -> float:
        u = int(user_id)
        i = int(item_id)
        return float(
            predict_rating(
                self.train_matrix[u],
                self.cluster_labels,
                self.train_matrix,
                self.centroids,
                i,
                top_k=self.top_k,
                distance_metric=self.distance_metric,
            )
        )
