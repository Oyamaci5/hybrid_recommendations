"""Tek noktadan değerlendirme (MAE, RMSE, P@N, R@N vb.)."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.metrics import compute_topn_metrics, evaluate_cf, evaluate_predictions, predict_rating


class Evaluator:
    def __init__(self, relevance_threshold: float = 3.5, top_neighbors: int = 30, at_n: int = 10):
        self.relevance_threshold = float(relevance_threshold)
        self.top_neighbors = int(top_neighbors)
        self.at_n = int(at_n)

    def rating_errors(
        self,
        test_ratings: np.ndarray,
        train_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
        distance_metric: str = "pearson",
    ) -> tuple[np.ndarray, np.ndarray]:
        preds, acts = [], []
        for row in test_ratings:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            pred = predict_rating(
                train_matrix[u],
                cluster_labels,
                train_matrix,
                centroids,
                i,
                top_k=self.top_neighbors,
                distance_metric=distance_metric,
            )
            preds.append(pred)
            acts.append(r)
        return np.asarray(acts), np.asarray(preds)

    def cluster_cf_summary(
        self,
        test_ratings: np.ndarray,
        train_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
        distance_metric: str = "pearson",
    ) -> dict[str, Any]:
        return evaluate_cf(
            test_ratings,
            train_matrix,
            cluster_labels,
            centroids,
            top_k=self.top_neighbors,
            N=self.at_n,
            relevance_threshold=self.relevance_threshold,
            distance_metric=distance_metric,
        )

    def from_pred_rows(self, eval_rows: np.ndarray, top_n: int | None = None) -> tuple[float, float, float, float]:
        """Satırlar [user, item, rating_true, rating_pred]."""
        tn = int(top_n) if top_n is not None else self.at_n
        return compute_topn_metrics(eval_rows, top_n=tn, threshold=self.relevance_threshold)

    @staticmethod
    def regression_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        return evaluate_predictions(np.asarray(y_true), np.asarray(y_pred))
