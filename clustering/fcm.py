"""Numpy ile basit fuzzy c-means (Euclidean)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def fuzzy_cmeans(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Üyelik U (n, K), merkez V (K, d) döner.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    K = max(2, min(int(n_clusters), n))

    rng = np.random.default_rng(random_state)
    U = rng.random(size=(n, K))
    U /= U.sum(axis=1, keepdims=True) + 1e-15
    m_f = float(m)
    expo = float(2.0 / max(m_f - 1.0, 1e-6))

    centers = rng.uniform(X.min(axis=0), X.max(axis=0), size=(K, d))

    for _ in range(int(max_iter)):
        Um = U**m_f
        denom = np.sum(Um, axis=0, keepdims=True).T.clip(1e-15)
        centers = (Um.T @ X) / denom
        dist = cdist(X, centers, metric="euclidean").astype(np.float64)
        dist = np.maximum(dist, 1e-12)
        U_new = 1.0 / np.sum(
            (dist[:, :, None] / dist[:, None, :]) ** expo,
            axis=2,
            dtype=np.float64,
        ).clip(min=1e-15)

        delta = np.linalg.norm(U_new - U, ord=np.inf)
        U = U_new
        if delta < tol:
            break

    return U, centers
