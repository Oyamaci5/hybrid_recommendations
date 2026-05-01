"""Kümeleme algoritma yönlendirmesi (KMeans / FCM / meta problemi)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.cluster import KMeans

from clustering.fcm import fuzzy_cmeans
from clustering.problem import ClusteringProblem


@dataclass
class ClusteringModule:
    """
    Dokümantasyondaki ``clustering.space`` ile uyum için:
      - latent + kmeans|fcm: sklearn/KMeans veya fuzzy_cmeans üzerinden sert cluster.
      - meta_*: centroid vektörü zaten Optimize edilmişse ``assign`` doğrudan problem üzerinden.
    """

    n_clusters: int = 50
    algorithm: str = "kmeans"
    fcm_m: float = 2.0
    kmeans_seed: int = 42

    def fit_predict(
        self,
        embeddings: np.ndarray,
        *,
        centroid_flat: np.ndarray | None = None,
        problem: ClusteringProblem | None = None,
        space: Literal["raw", "latent"] = "latent",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        algo = self.algorithm.lower().replace("-", "_")
        X = embeddings.astype(np.float64)

        if algo.startswith("meta"):
            if problem is None or centroid_flat is None:
                raise ValueError("meta_* algoritmalar centroid vektörü + ClusteringProblem gerektirir.")
            labs = problem.assign(np.asarray(centroid_flat))
            meta = {"centroids_shape": (problem.K, problem.n_items)}
            return labs, meta

        if algo == "fcm":
            _, centers = fuzzy_cmeans(X, self.n_clusters, m=self.fcm_m, random_state=self.kmeans_seed)
            dist = np.linalg.norm(X[:, None] - centers[None, :, :], axis=2)
            labs = np.argmin(dist, axis=1).astype(np.int32)
            return labs, {"centers": centers}

        # default sklearn kmeans (+ meta_kmeans centroid path yoksa)
        km = KMeans(
            n_clusters=min(self.n_clusters, X.shape[0]),
            random_state=self.kmeans_seed,
            n_init=min(10, max(4, X.shape[0] // 10)),
            max_iter=300,
        )
        labs = km.fit_predict(X)
        return labs.astype(np.int32), {"centers": km.cluster_centers_}
