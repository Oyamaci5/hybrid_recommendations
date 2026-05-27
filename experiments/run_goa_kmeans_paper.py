"""
GOA-k-means implementation following the paper's pipeline:

- User-mean imputation + column z-score
- PCA-50 feature extraction
- GOA-based centroid search + k-means refinement
- Cluster-average rating prediction (MAE/RMSE)

ML-100K u1.base / u1.test (same split as mealpy paper-mode eval).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config, set_random_seed
from core.utils import load_train_test_split, get_data_info
from core.metrics import evaluate_predictions


def build_user_item_matrix(
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
) -> np.ndarray:
    M = np.full((n_users, n_items), np.nan, dtype=np.float32)
    for u, i, r in ratings:
        M[int(u), int(i)] = float(r)
    return M


def zscore_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    return (X - mean) / (std + eps)


def prepare_paper_features(
    train_ratings: np.ndarray,
    n_users: int,
    n_items: int,
    pca_components: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    train_M = build_user_item_matrix(train_ratings, n_users, n_items)
    user_means = np.nanmean(train_M, axis=1, keepdims=True)
    global_mean_train = np.nanmean(train_M)
    user_means[np.isnan(user_means)] = global_mean_train
    M_imputed = np.where(np.isnan(train_M), user_means, train_M)
    X = zscore_normalize(M_imputed)
    if pca_components is not None and pca_components > 0:
        n_comp = min(pca_components, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=0)
        X_feat = pca.fit_transform(X)
    else:
        X_feat = X
    return train_M, X_feat.astype(np.float32)


def goa_kmeans_optimize_features(
    X: np.ndarray,
    n_clusters: int = 3,
    n_agents: int = 40,
    n_iterations: int = 100,
    c_min: float = 0.00004,
    c_max: float = 1.0,
    f: float = 0.5,
    l: float = 1.5,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, dim = X.shape
    total_dim = n_clusters * dim

    lb = X.min(axis=0).min()
    ub = X.max(axis=0).max()
    if lb == ub:
        lb, ub = -1.0, 1.0

    positions = ((ub - lb) * np.random.rand(n_agents, total_dim) + lb).astype(np.float32)

    def decode_centers(position: np.ndarray) -> np.ndarray:
        return position.reshape(n_clusters, dim)

    def fitness(position: np.ndarray) -> float:
        centers = decode_centers(position)
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = 0.0
        for k in range(n_clusters):
            mask = labels == k
            if not np.any(mask):
                continue
            diff = X[mask] - centers[k]
            inertia += np.sum(diff * diff)
        return float(inertia / max(1, n_samples))

    fitnesses = np.array([fitness(pos) for pos in positions], dtype=np.float32)
    best_idx = int(np.argmin(fitnesses))
    best_pos = positions[best_idx].copy()
    best_fit = float(fitnesses[best_idx])

    for t in range(1, n_iterations + 1):
        c = c_max - t * (c_max - c_min) / float(n_iterations)
        new_positions = np.zeros_like(positions)
        for i in range(n_agents):
            Xi = positions[i]
            S_component = np.zeros(total_dim, dtype=np.float32)
            for j in range(n_agents):
                if j == i:
                    continue
                Xj = positions[j]
                dist_vec = Xj - Xi
                dist = np.linalg.norm(dist_vec) + 1e-12
                r_norm = dist / (ub - lb + 1e-12)
                s_r = f * np.exp(-r_norm / l) - np.exp(-r_norm)
                S_component += (s_r * dist_vec / dist)
            Xi_new = c * ((ub - lb) / 2.0 * S_component) + best_pos
            new_positions[i] = np.clip(Xi_new, lb, ub).astype(np.float32)

        positions = new_positions
        fitnesses = np.array([fitness(pos) for pos in positions], dtype=np.float32)
        cur_best_idx = int(np.argmin(fitnesses))
        cur_best_fit = float(fitnesses[cur_best_idx])
        if cur_best_fit < best_fit:
            best_fit = cur_best_fit
            best_pos = positions[cur_best_idx].copy()
        if verbose and (t % 10 == 0 or t == 1):
            print(f"GOA Iteration {t}: Best Inertia = {best_fit:.6f}, c = {c:.6f}")

    init_centers = decode_centers(best_pos)
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        max_iter=100,
        random_state=0,
    )
    labels = kmeans.fit_predict(X)
    return kmeans.cluster_centers_, labels


def predict_cluster_average(
    labels: np.ndarray,
    train_M: np.ndarray,
    test_ratings: np.ndarray,
    n_users: int,
    n_items: int,
    n_clusters: int,
) -> dict:
    cluster_profiles = np.full((n_clusters, n_items), np.nan, dtype=np.float32)
    cluster_counts = np.zeros((n_clusters, n_items), dtype=np.int32)

    for u in range(n_users):
        k = int(labels[u])
        for i in range(n_items):
            r = train_M[u, i]
            if not np.isnan(r):
                if np.isnan(cluster_profiles[k, i]):
                    cluster_profiles[k, i] = r
                else:
                    cluster_profiles[k, i] += r
                cluster_counts[k, i] += 1

    for k in range(n_clusters):
        for i in range(n_items):
            if cluster_counts[k, i] > 0:
                cluster_profiles[k, i] /= cluster_counts[k, i]

    global_mean = float(np.nanmean(train_M))
    with np.errstate(all='ignore'):
        item_means = np.nanmean(train_M, axis=0)
        user_means_full = np.nanmean(train_M, axis=1)
    item_means = np.where(np.isnan(item_means), global_mean, item_means)
    user_means_full = np.where(np.isnan(user_means_full), global_mean, user_means_full)

    user_ids = test_ratings[:, 0].astype(int)
    item_ids = test_ratings[:, 1].astype(int)
    true_ratings = test_ratings[:, 2]
    preds = np.empty_like(true_ratings, dtype=np.float32)
    for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
        k = int(labels[u])
        val = cluster_profiles[k, i]
        if not np.isnan(val):
            preds[idx] = val
        elif not np.isnan(item_means[i]):
            preds[idx] = item_means[i]
        elif not np.isnan(user_means_full[u]):
            preds[idx] = user_means_full[u]
        else:
            preds[idx] = global_mean

    rmse, mae = evaluate_predictions(true_ratings, preds)
    return {'rmse': float(rmse), 'mae': float(mae), 'labels': labels}


def evaluate_goa_kmeans_rs(
    train_ratings: np.ndarray,
    test_ratings: np.ndarray,
    n_users: int,
    n_items: int,
    n_clusters: int = 3,
    n_agents: int = 40,
    n_iterations: int = 100,
    pca_components: int = 50,
    verbose: bool = True,
) -> dict:
    train_M, X_feat = prepare_paper_features(
        train_ratings, n_users, n_items, pca_components,
    )
    if verbose:
        print("=" * 60)
        print("GOA-k-means (paper replica)")
        print("=" * 60)
        print(f"Users: {n_users}, Items: {n_items}, PCA dim: {X_feat.shape[1]}, K: {n_clusters}")
        print(f"Agents: {n_agents}, Iterations: {n_iterations}")
        print("=" * 60)

    _, labels = goa_kmeans_optimize_features(
        X_feat, n_clusters, n_agents, n_iterations, verbose=verbose,
    )
    metrics = predict_cluster_average(
        labels, train_M, test_ratings, n_users, n_items, n_clusters,
    )
    if verbose:
        print(f"\n[GOA-k-means] MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}")
    return {
        'method': 'GOA-k-means',
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'n_clusters': n_clusters,
    }


def evaluate_kmeans_baseline(
    train_ratings: np.ndarray,
    test_ratings: np.ndarray,
    n_users: int,
    n_items: int,
    n_clusters: int,
    pca_components: int = 50,
    verbose: bool = True,
) -> dict:
    train_M, X_feat = prepare_paper_features(
        train_ratings, n_users, n_items, pca_components,
    )
    km = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=100,
        random_state=0,
    )
    labels = km.fit_predict(X_feat)
    metrics = predict_cluster_average(
        labels, train_M, test_ratings, n_users, n_items, n_clusters,
    )
    if verbose:
        print(f"[K-means++]   MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}")
    return {
        'method': 'K-means++ (PCA)',
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'n_clusters': n_clusters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='GOA-k-means paper replica (ML-100K).')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--agents', type=int, default=40)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--pca_components', type=int, default=50)
    parser.add_argument('--data-dir', type=str, default='data/ml-100k')
    parser.add_argument('--compare-kmeans', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    config = Config(random_seed=args.seed, data_dir=args.data_dir)
    train_ratings, test_ratings = load_train_test_split(
        config.get_train_path(), config.get_test_path(),
    )
    n_users, n_items, _ = get_data_info(train_ratings)

    print(f"Dataset: {args.data_dir}  |  train={len(train_ratings):,}  test={len(test_ratings):,}")

    results = []
    if args.compare_kmeans:
        results.append(evaluate_kmeans_baseline(
            train_ratings, test_ratings, n_users, n_items,
            args.k, args.pca_components,
        ))
    results.append(evaluate_goa_kmeans_rs(
        train_ratings, test_ratings, n_users, n_items,
        n_clusters=args.k,
        n_agents=args.agents,
        n_iterations=args.iters,
        pca_components=args.pca_components,
        verbose=True,
    ))

    if len(results) > 1:
        print('\n' + '=' * 60)
        print('SUMMARY')
        print('=' * 60)
        for r in results:
            print(f"  {r['method']:24s}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}")
        best = min(results, key=lambda x: x['mae'])
        print(f"\nBest MAE: {best['method']} ({best['mae']:.4f})")


if __name__ == '__main__':
    main()
