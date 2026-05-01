"""Kümeleme algoritma yönlendirmesi (KMeans / FCM / meta problemi)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from sklearn.cluster import KMeans

from clustering.fcm import fuzzy_cmeans
from clustering.problem import ClusteringProblem
from preprocess.mkmeans_plus_plus import make_mkmeans_init_population


@dataclass
class ClusteringModule:
    """
    Dokümantasyondaki ``clustering.space`` ile uyum için:
      - latent + kmeans|fcm: sklearn/KMeans veya fuzzy_cmeans üzerinden sert cluster.
      - meta_*: centroid vektörü zaten Optimize edilmişse ``assign`` doğrudan problem üzerinden.
    """

    optimizer: Any = None
    n_clusters: int = 50
    algorithm: str = "kmeans"
    fcm_m: float = 2.0
    kmeans_inner_iter: int = 5
    space: str = "latent"
    kmeans_seed: int = 42
    init_method: str = "kmeans++"
    mkmeans_init_max_iter: int = 50

    def _space_bounds(self, X: np.ndarray, space: str) -> tuple[np.ndarray, np.ndarray]:
        dim = int(self.n_clusters * X.shape[1])
        if (space or self.space).strip().lower() == "raw":
            return np.ones(dim, dtype=np.float64), np.full(dim, 5.0, dtype=np.float64)
        eps = 1e-6
        lo = float(np.min(X)) - eps
        hi = float(np.max(X)) + eps
        if abs(hi - lo) <= eps:
            absmax = max(abs(lo), abs(hi), 1.0)
            lo, hi = -absmax, absmax
        return np.full(dim, lo, dtype=np.float64), np.full(dim, hi, dtype=np.float64)

    def _optimize_meta(
        self,
        optimizer: Any,
        problem: ClusteringProblem,
        X: np.ndarray,
        space: str,
    ) -> np.ndarray:
        lb, ub = self._space_bounds(X, space)
        if hasattr(optimizer, "minimize"):
            result = optimizer.minimize(problem.fitness, problem.dim, lb, ub)
            vec = getattr(result, "best_vector", None)
            if vec is None:
                raise ValueError("Optimizer minimize() returned no best_vector.")
            return np.asarray(vec, dtype=np.float64).ravel()

        if hasattr(optimizer, "optimize"):
            out = optimizer.optimize(problem.fitness)
            if isinstance(out, tuple) and len(out) >= 1:
                return np.asarray(out[0], dtype=np.float64).ravel()
            return np.asarray(out, dtype=np.float64).ravel()

        if callable(optimizer):
            materialized = optimizer(problem.dim, lb, ub)
            return self._optimize_meta(materialized, problem, X, space)

        raise TypeError("Unsupported optimizer type for meta clustering.")

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
            if problem is None:
                problem = ClusteringProblem(X, int(self.n_clusters))
            if centroid_flat is None:
                if self.optimizer is None:
                    raise AssertionError("meta_* için optimizer gerekli")
                centroid_flat = self._optimize_meta(self.optimizer, problem, X, space)
            labs = problem.assign(np.asarray(centroid_flat))
            centers = np.asarray(centroid_flat, dtype=np.float64).reshape(problem.K, problem.n_items)
            meta = {"centroids_shape": (problem.K, problem.n_items), "centers": centers}
            return labs, meta

        if algo == "fcm":
            membership, centers = fuzzy_cmeans(X, self.n_clusters, m=self.fcm_m, random_state=self.kmeans_seed)
            dist = np.linalg.norm(X[:, None] - centers[None, :, :], axis=2)
            labs = np.argmin(dist, axis=1).astype(np.int32)
            return labs, {"centers": centers, "membership": membership}

        # default sklearn kmeans (+ meta_kmeans centroid path yoksa)
        init_centers = "k-means++"
        if (self.init_method or "kmeans++").strip().lower() in (
            "mkmeans_plus_plus_init",
            "mkmeans++",
            "mkmeans_plus_plus",
        ):
            pop = make_mkmeans_init_population(
                X,
                K=min(self.n_clusters, X.shape[0]),
                pop_size=1,
                max_iter=max(int(self.mkmeans_init_max_iter), 1),
                seed=int(self.kmeans_seed),
            )
            init_centers = np.asarray(pop[0], dtype=np.float64).reshape(min(self.n_clusters, X.shape[0]), X.shape[1])

        km = KMeans(
            n_clusters=min(self.n_clusters, X.shape[0]),
            init=init_centers,
            random_state=self.kmeans_seed,
            n_init=1 if isinstance(init_centers, np.ndarray) else min(10, max(4, X.shape[0] // 10)),
            max_iter=max(int(self.kmeans_inner_iter), 1),
        )
        labs = km.fit_predict(X)
        return labs.astype(np.int32), {"centers": km.cluster_centers_}
