"""
Clustering fitness utilities used by custom optimizers.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from core.metrics import pearson_distance

_INF_PROXY = 1e6


def calculate_clustering_fitness(
    centroids: np.ndarray,
    user_data: np.ndarray,
    k: int,
) -> float:
    """
    Evaluate centroid quality using Pearson distance assignments.
    """
    total_error = 0.0
    cluster_assignments = [[] for _ in range(int(k))]

    for user in user_data:
        distances = [pearson_distance(user, c) for c in centroids]
        best_cluster_idx = int(np.argmin(distances))
        cluster_assignments[best_cluster_idx].append(float(distances[best_cluster_idx]))

    for i in range(int(k)):
        if len(cluster_assignments[i]) == 0:
            total_error += 1000.0
        else:
            total_error += float(np.sum(cluster_assignments[i]))

    return float(total_error)


# ---------------------------------------------------------------------------
# Vectorised fitness evaluator — DOA / metaheuristic clustering için
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """
    Kullanıcı istatistiklerini bir kez önden hesaplayarak DOA/PSO döngüsünde
    tekrarlanan fitness çağrılarını hızlandırır.

    Her kullanıcı için sadece oyladığı itemler üzerinden PCC hesaplar;
    centroid bu konumlarda yeniden merkezlenir.

    Parameters
    ----------
    rating_matrix : np.ndarray, shape (n_users, n_items)
        Train rating matrisi (0 = oylanmamış).
    K : int
        Küme sayısı.
    """

    def __init__(
        self,
        rating_matrix: np.ndarray,
        K: int,
        distance_metric: Literal["pearson", "euclidean"] = "pearson",
    ) -> None:
        if distance_metric not in {"pearson", "euclidean"}:
            raise ValueError("distance_metric must be 'pearson' or 'euclidean'")
        self.K = K
        self.n_items = rating_matrix.shape[1]
        self.n_users = rating_matrix.shape[0]
        self.distance_metric = distance_metric
        self._rating_matrix = rating_matrix
        self._stats = self._precompute(rating_matrix)

    def _precompute(self, R: np.ndarray) -> list:
        stats = []
        for u in range(self.n_users):
            idx = np.where(R[u] != 0)[0]
            if len(idx) == 0:
                stats.append((idx, np.array([]), 0.0))
                continue
            rated = R[u, idx].astype(np.float64)
            centered = rated - rated.mean()
            norm = np.linalg.norm(centered)
            stats.append((idx, centered, norm))
        return stats

    def _dist_matrix(self, centroids: np.ndarray) -> np.ndarray:
        """(n_users, K) PCC mesafe matrisi — K boyutunda vektörize."""
        if self.distance_metric == "euclidean":
            return self._euclidean_dist_matrix(centroids)

        dist = np.ones((self.n_users, self.K))
        for u, (idx, r_c, norm_u) in enumerate(self._stats):
            if len(idx) == 0 or norm_u < 1e-10:
                continue
            c_at_u = centroids[:, idx]                        # (K, n_rated)
            c_c = c_at_u - c_at_u.mean(axis=1, keepdims=True)
            numer = c_c @ r_c                                 # (K,)
            c_norms = np.linalg.norm(c_c, axis=1)
            denom = c_norms * norm_u
            with np.errstate(invalid="ignore", divide="ignore"):
                pcc = np.where(denom > 1e-10, numer / denom, 0.0)
            dist[u] = 1.0 - pcc
        return dist

    def _euclidean_dist_matrix(self, centroids: np.ndarray) -> np.ndarray:
        """
        (n_users, K) Euclidean mesafe matrisi.
        Her kullanıcı için sadece oylanan itemler üzerinde hesaplanır.
        """
        dist = np.full((self.n_users, self.K), fill_value=np.inf, dtype=np.float64)
        for u in range(self.n_users):
            idx = np.where(self._rating_matrix[u] != 0)[0]
            if len(idx) == 0:
                continue
            diff = centroids[:, idx] - self._rating_matrix[u, idx]
            dist[u] = np.sqrt((diff * diff).sum(axis=1))
        return dist

    def __call__(self, flat_centroids: np.ndarray) -> float:
        """
        Optimizer'ın çağıracağı interface.

        Parameters
        ----------
        flat_centroids : np.ndarray, shape (K * n_items,)

        Returns
        -------
        float — toplam intra-cluster PCC mesafesi (minimize edilir).
        """
        centroids = flat_centroids.reshape(self.K, self.n_items)
        dist = self._dist_matrix(centroids)
        labels = np.argmin(dist, axis=1)
        return float(dist[np.arange(self.n_users), labels].sum())

    def assign_clusters(self, centroids: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        centroids : np.ndarray, shape (K, n_items)

        Returns
        -------
        labels : np.ndarray, shape (n_users,)
        """
        return np.argmin(self._dist_matrix(centroids), axis=1)


def make_fitness_fn(
    rating_matrix: np.ndarray,
    K: int,
    distance_metric: Literal["pearson", "euclidean"] = "pearson",
):
    """
    FitnessEvaluator'ı oluşturur ve callable olarak döndürür.

    Kullanım::

        fn = make_fitness_fn(train_matrix, K=30)
        best_pos, best_fit, curve = doa.optimize(fn)

    Parameters
    ----------
    rating_matrix : np.ndarray, shape (n_users, n_items)
    K : int

    Returns
    -------
    FitnessEvaluator instance (callable)
    """
    return FitnessEvaluator(rating_matrix, K, distance_metric=distance_metric)


def _euclidean_distance_on_rated(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Euclidean distance computed only on co-rated items."""
    common = np.where((vector_a != 0) & (vector_b != 0))[0]
    if len(common) == 0:
        return float("inf")
    diff = vector_a[common] - vector_b[common]
    return float(np.sqrt(np.sum(diff * diff)))


def _distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    distance_metric: Literal["pearson", "pcc", "euclidean"],
) -> float:
    if distance_metric in {"pearson", "pcc"}:
        return float(pearson_distance(vector_a, vector_b))
    return _euclidean_distance_on_rated(vector_a, vector_b)


def _decode_solution(position: np.ndarray, K: int, num_items: int) -> np.ndarray:
    return position.reshape(K, num_items)


def _assign_users_to_clusters(
    rating_matrix: np.ndarray,
    centroids: np.ndarray,
    distance_metric: Literal["pearson", "pcc", "euclidean"] = "pearson",
) -> np.ndarray:
    n_users = rating_matrix.shape[0]
    K = centroids.shape[0]
    labels = np.zeros(n_users, dtype=np.int32)

    for u in range(n_users):
        distances = np.array(
            [_distance(rating_matrix[u], centroids[k], distance_metric) for k in range(K)],
            dtype=np.float64,
        )
        distances = np.where(np.isinf(distances), _INF_PROXY, distances)
        labels[u] = int(np.argmin(distances))

    return labels


def compute_silhouette(
    rating_matrix: np.ndarray,
    labels: np.ndarray,
    distance_metric: Literal["pearson", "pcc", "euclidean"] = "pearson",
    sample_size: int = 200,
    seed: int = 42,
) -> float:
    """
    Compute silhouette score in [-1, 1] for clustering quality.
    """
    n_users = rating_matrix.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    sampled = rng.choice(n_users, min(sample_size, n_users), replace=False)
    scores: list[float] = []

    for u in sampled:
        cluster_u = labels[u]
        same = np.where(labels == cluster_u)[0]
        same = same[same != u]
        if len(same) == 0:
            scores.append(0.0)
            continue

        a_vals = [
            _distance(rating_matrix[u], rating_matrix[v], distance_metric) for v in same
        ]
        a_vals = [_INF_PROXY if np.isinf(x) else x for x in a_vals]
        a = float(np.mean(a_vals))

        b_candidates: list[float] = []
        for k in unique_labels:
            if k == cluster_u:
                continue
            other = np.where(labels == k)[0]
            if len(other) == 0:
                continue
            b_vals = [
                _distance(rating_matrix[u], rating_matrix[v], distance_metric)
                for v in other
            ]
            b_vals = [_INF_PROXY if np.isinf(x) else x for x in b_vals]
            b_candidates.append(float(np.mean(b_vals)))

        if not b_candidates:
            scores.append(0.0)
            continue

        b = min(b_candidates)
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 1e-10 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def compute_davies_bouldin(
    rating_matrix: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    distance_metric: Literal["pearson", "pcc", "euclidean"] = "pearson",
) -> float:
    """
    Compute Davies-Bouldin index (lower is better).
    """
    K = centroids.shape[0]
    if K == 0:
        return float("inf")

    S = np.zeros(K, dtype=np.float64)
    for k in range(K):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue
        dists = [_distance(rating_matrix[u], centroids[k], distance_metric) for u in members]
        dists = [_INF_PROXY if np.isinf(x) else x for x in dists]
        S[k] = float(np.mean(dists))

    db_sum = 0.0
    for i in range(K):
        max_ratio = 0.0
        for j in range(K):
            if i == j:
                continue
            m_ij = _distance(centroids[i], centroids[j], distance_metric)
            if np.isinf(m_ij) or m_ij < 1e-10:
                continue
            ratio = (S[i] + S[j]) / m_ij
            if ratio > max_ratio:
                max_ratio = float(ratio)
        db_sum += max_ratio

    return float(db_sum / K)


def compute_all_metrics(
    rating_matrix: np.ndarray,
    best_position: np.ndarray,
    K: int,
    distance_metric: Literal["pearson", "pcc", "euclidean"] = "pearson",
    silhouette_sample: int = 200,
) -> dict:
    """
    Compute WCSS-like fitness, silhouette, DB index and cluster metadata.
    """
    n_items = rating_matrix.shape[1]
    centroids = _decode_solution(best_position, K, n_items)
    labels = _assign_users_to_clusters(rating_matrix, centroids, distance_metric)

    evaluator = FitnessEvaluator(
        rating_matrix,
        K,
        distance_metric="euclidean" if distance_metric == "euclidean" else "pearson",
    )
    fitness_value = float(evaluator(best_position))
    sil = compute_silhouette(
        rating_matrix,
        labels,
        distance_metric=distance_metric,
        sample_size=silhouette_sample,
    )
    db = compute_davies_bouldin(
        rating_matrix,
        centroids,
        labels,
        distance_metric=distance_metric,
    )

    cluster_sizes = {k: int((labels == k).sum()) for k in range(K)}
    return {
        "fitness": round(fitness_value, 4),
        "Silhouette": round(sil, 4),
        "Davies_Bouldin": round(db, 4),
        "cluster_sizes": cluster_sizes,
        "labels": labels,
        "centroids": centroids,
    }

