"""
mkmeans_plus_plus.py
--------------------
MkMeans++ tabanli baslangic populasyonu uretir.

Kaynak:
"Proposing improved meta-heuristic algorithms for clustering and separating
users in recommender systems", Electronic Commerce Research (2022) 22:623-648
"""

from __future__ import annotations

import numpy as np


def _rated_count_per_user(rating_matrix: np.ndarray) -> np.ndarray:
    return np.count_nonzero(rating_matrix != 0, axis=1)


def _rated_count_per_item(rating_matrix: np.ndarray) -> np.ndarray:
    return np.count_nonzero(rating_matrix != 0, axis=0)


def _pcc_similarity(user_a: np.ndarray, user_b: np.ndarray) -> float:
    mask = (user_a != 0) & (user_b != 0)
    if np.count_nonzero(mask) < 2:
        return 0.0
    a = user_a[mask].astype(np.float64)
    b = user_b[mask].astype(np.float64)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < 1e-10:
        return 0.0
    return float(np.clip(np.dot(a_c, b_c) / denom, -1.0, 1.0))


def _masked_euclidean(user: np.ndarray, centroid: np.ndarray) -> float:
    mask = (user != 0) & (centroid != 0)
    if np.count_nonzero(mask) == 0:
        return float("inf")
    diff = user[mask] - centroid[mask]
    return float(np.sqrt(np.dot(diff, diff)))


def _select_first_center(power_weights: np.ndarray, rng: np.random.Generator) -> int:
    max_pw = float(np.max(power_weights))
    candidates = np.where(np.isclose(power_weights, max_pw))[0]
    if len(candidates) == 1:
        return int(candidates[0])
    return int(rng.choice(candidates))


def _kmeanspp_next_center(
    rating_matrix: np.ndarray,
    selected_ids: list[int],
    rng: np.random.Generator,
) -> int:
    n_users = rating_matrix.shape[0]
    dist_sq = np.zeros(n_users, dtype=np.float64)

    for u in range(n_users):
        if u in selected_ids:
            dist_sq[u] = 0.0
            continue
        d_min = float("inf")
        for c_id in selected_ids:
            d = _masked_euclidean(rating_matrix[u], rating_matrix[c_id])
            if d < d_min:
                d_min = d
        if not np.isfinite(d_min):
            d_min = 0.0
        dist_sq[u] = d_min * d_min

    total = float(dist_sq.sum())
    if total <= 1e-12:
        remaining = np.setdiff1d(np.arange(n_users), np.array(selected_ids))
        return int(rng.choice(remaining))

    probs = dist_sq / total
    return int(rng.choice(np.arange(n_users), p=probs))


def _assign_labels(rating_matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    n_users = rating_matrix.shape[0]
    K = centroids.shape[0]
    labels = np.zeros(n_users, dtype=np.int32)

    for u in range(n_users):
        dists = np.empty(K, dtype=np.float64)
        for k in range(K):
            dists[k] = _masked_euclidean(rating_matrix[u], centroids[k])
            if not np.isfinite(dists[k]):
                dists[k] = 1e9
        labels[u] = int(np.argmin(dists))
    return labels


def _update_centroids(
    rating_matrix: np.ndarray,
    labels: np.ndarray,
    K: int,
    prev_centroids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_items = rating_matrix.shape[1]
    new_centroids = np.zeros((K, n_items), dtype=np.float64)

    for k in range(K):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            new_centroids[k] = rating_matrix[rng.integers(0, rating_matrix.shape[0])]
            continue
        new_centroids[k] = rating_matrix[members].mean(axis=0)

    # Bos item kolonlarinda oldugu gibi 0 kalmasi normaldir.
    _ = prev_centroids
    return new_centroids


def _single_mkmeans(
    rating_matrix: np.ndarray,
    K: int,
    max_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_users, n_items = rating_matrix.shape
    if K <= 0 or K > n_users:
        raise ValueError(f"K must be in [1, n_users], got K={K}, n_users={n_users}")

    user_counts = _rated_count_per_user(rating_matrix)
    _ = _rated_count_per_item(rating_matrix)  # pI feature extracted conceptually

    pu_id = int(np.argmax(user_counts))
    wu_id = int(np.argmin(user_counts))

    power_weights = np.array(
        [
            _pcc_similarity(rating_matrix[pu_id], rating_matrix[u])
            + _pcc_similarity(rating_matrix[wu_id], rating_matrix[u])
            for u in range(n_users)
        ],
        dtype=np.float64,
    )

    selected_ids: list[int] = [_select_first_center(power_weights, rng)]
    while len(selected_ids) < K:
        nxt = _kmeanspp_next_center(rating_matrix, selected_ids, rng)
        if nxt not in selected_ids:
            selected_ids.append(nxt)

    centroids = rating_matrix[selected_ids].astype(np.float64).copy()

    prev_labels = None
    for _ in range(max_iter):
        labels = _assign_labels(rating_matrix, centroids)
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels
        centroids = _update_centroids(rating_matrix, labels, K, centroids, rng)

    return centroids.reshape(K * n_items)


def make_mkmeans_init_population(
    rating_matrix: np.ndarray,
    K: int,
    pop_size: int,
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    MkMeans++ ile DOA icin baslangic populasyonu uretir.

    Parameters
    ----------
    rating_matrix : np.ndarray
        Shape (n_users, n_items)
    K : int
        Kume sayisi.
    pop_size : int
        Uretilecek cozum adedi (P).
    max_iter : int
        Her MkMeans calismasi icin maksimum kmeans iterasyonu.
    seed : int
        Rastgelelik tohumu.

    Returns
    -------
    np.ndarray
        Shape (pop_size, K * n_items)
    """
    if pop_size <= 0:
        raise ValueError("pop_size must be positive")

    base_rng = np.random.default_rng(seed)
    population = []
    for _ in range(pop_size):
        run_seed = int(base_rng.integers(0, 2**31 - 1))
        run_rng = np.random.default_rng(run_seed)
        flat = _single_mkmeans(rating_matrix, K, max_iter=max_iter, rng=run_rng)
        population.append(flat)
    return np.asarray(population, dtype=np.float64)
