"""
GOA-k-means implementation following the paper's pipeline:

- Z-score normalization
- PCA feature extraction
- GOA-based centroid initialization
- k-means refinement
- Evaluation on MovieLens 1M with MAE/RMSE

Bu script MF tabanlı yapından bağımsız, makaledeki
özellik-uzayı + GOA-k-means yaklaşımını yeniden uygular.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Add project root to path (same as other experiment scripts)
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
    """
    Dense user–item matrix with NaN for missing entries.
    Bu sayede eksik değerleri istatistiklerde ayrı tutabiliyoruz.
    """
    M = np.full((n_users, n_items), np.nan, dtype=np.float32)
    for u, i, r in ratings:
        M[int(u), int(i)] = float(r)
    return M


def zscore_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Column-wise z-score normalization, ignoring NaNs.
    """
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    return (X - mean) / (std + eps)


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
    """
    GOA-k-means optimizasyonu:
    - X: (n_samples, d) PCA ile elde edilmiş özellikler
    - Çıktı: (centers, labels) k-means sonrası nihai merkezler ve atamalar
    """
    n_samples, dim = X.shape
    total_dim = n_clusters * dim

    # Arama uzayı sınırları (özellik aralığı)
    lb = X.min(axis=0).min()
    ub = X.max(axis=0).max()
    if lb == ub:
        lb, ub = -1.0, 1.0

    # Başlangıç popülasyonu: her ajan k merkez vektörünü taşır
    positions = (ub - lb) * np.random.rand(n_agents, total_dim) + lb
    positions = positions.astype(np.float32)

    def decode_centers(position: np.ndarray) -> np.ndarray:
        return position.reshape(n_clusters, dim)

    def fitness(position: np.ndarray) -> float:
        centers = decode_centers(position)
        # her örnek için en yakın merkeze uzaklık karesinin ortalaması
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

    # İlk fitness'ler
    fitnesses = np.array([fitness(pos) for pos in positions], dtype=np.float32)
    best_idx = int(np.argmin(fitnesses))
    best_pos = positions[best_idx].copy()
    best_fit = float(fitnesses[best_idx])

    history_best = [best_fit]

    for t in range(1, n_iterations + 1):
        # shrinking coefficient c (Eq. 11)
        c = c_max - t * (c_max - c_min) / float(n_iterations)

        new_positions = np.zeros_like(positions)
        for i in range(n_agents):
            Xi = positions[i]

            # Sosyal etkileşim (Eq. 10)
            S_component = np.zeros(total_dim, dtype=np.float32)
            for j in range(n_agents):
                if j == i:
                    continue
                Xj = positions[j]
                dist_vec = Xj - Xi
                dist = np.linalg.norm(dist_vec) + 1e-12
                # normalize distance r
                r_norm = dist / (ub - lb + 1e-12)
                s_r = f * np.exp(-r_norm / l) - np.exp(-r_norm)
                S_component += (s_r * dist_vec / dist)

            Xi_new = c * ((ub - lb) / 2.0 * S_component) + best_pos
            Xi_new = np.clip(Xi_new, lb, ub).astype(np.float32)

            new_positions[i] = Xi_new

        positions = new_positions
        fitnesses = np.array([fitness(pos) for pos in positions], dtype=np.float32)

        cur_best_idx = int(np.argmin(fitnesses))
        cur_best_fit = float(fitnesses[cur_best_idx])
        if cur_best_fit < best_fit:
            best_fit = cur_best_fit
            best_pos = positions[cur_best_idx].copy()

        history_best.append(best_fit)
        if verbose and (t % 10 == 0 or t == 1):
            print(f"GOA Iteration {t}: Best Inertia = {best_fit:.6f}, c = {c:.6f}")

    # GOA sonucu merkezleri ile k-means başlat
    init_centers = decode_centers(best_pos)
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        max_iter=100,
        random_state=0,
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return centers, labels


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
    """
    GOA-k-means tabanlı film öneri modeli:
    - Kullanıcı rating vektörleri üzerinde çalışır.
    - Kullanıcıları kümeleyip, her küme için ortalama rating profili çıkarır.
    - Tahminler bu küme profillerinden elde edilir.
    """
    # Kullanıcı–öğe matrisleri (NaN = missing)
    train_M = build_user_item_matrix(train_ratings, n_users, n_items)

    # Z-score + PCA (kullanıcı özellik uzayı)
    # Eksik değerler için önce kullanıcı ortalaması ile imputasyon,
    # ardından z-score normalizasyonu ve PCA.
    user_means = np.nanmean(train_M, axis=1, keepdims=True)
    # Eğer bir kullanıcının hiç rating'i yoksa, global ortalamayı kullan
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

    if verbose:
        print("=" * 60)
        print("GOA-k-means (paper-like) on MovieLens 1M")
        print("=" * 60)
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Feature dim after PCA: {X_feat.shape[1]}")
        print(f"Clusters (k): {n_clusters}, Agents: {n_agents}, Iterations: {n_iterations}")
        print("=" * 60)

    # GOA + k-means ile kullanıcı kümeleri
    centers, labels = goa_kmeans_optimize_features(
        X_feat,
        n_clusters=n_clusters,
        n_agents=n_agents,
        n_iterations=n_iterations,
        verbose=verbose,
    )

    # Her küme için ortalama rating profili (sadece gözlenen rating'ler üzerinden)
    cluster_profiles = np.full((n_clusters, n_items), np.nan, dtype=np.float32)
    cluster_counts = np.zeros((n_clusters, n_items), dtype=np.int32)

    for u in range(n_users):
        k = labels[u]
        for i in range(n_items):
            r = train_M[u, i]
            if not np.isnan(r):
                if np.isnan(cluster_profiles[k, i]):
                    cluster_profiles[k, i] = r
                else:
                    cluster_profiles[k, i] += r
                cluster_counts[k, i] += 1

    # Ortalama al
    for k in range(n_clusters):
        for i in range(n_items):
            if cluster_counts[k, i] > 0:
                cluster_profiles[k, i] /= cluster_counts[k, i]

    # Global / item / user ortalama rating'ler (çoklu fallback için)
    global_mean = float(np.nanmean(train_M))

    # RuntimeWarning almamak ve tamamen boş item/user'lar için global_mean kullanmak üzere
    with np.errstate(all="ignore"):
        item_means = np.nanmean(train_M, axis=0)
        user_means_full = np.nanmean(train_M, axis=1)

    # Tamamen boş kalanları global_mean ile doldur
    item_means = np.where(np.isnan(item_means), global_mean, item_means)
    user_means_full = np.where(np.isnan(user_means_full), global_mean, user_means_full)

    # Test set üzerinde tahminler
    user_ids = test_ratings[:, 0].astype(int)
    item_ids = test_ratings[:, 1].astype(int)
    true_ratings = test_ratings[:, 2]

    preds = np.empty_like(true_ratings, dtype=np.float32)
    for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
        k = labels[u]
        val = cluster_profiles[k, i]

        if not np.isnan(val):
            preds[idx] = val
        else:
            # Küme profili yoksa önce item ortalaması, o da yoksa user, en son global
            if not np.isnan(item_means[i]):
                preds[idx] = item_means[i]
            elif not np.isnan(user_means_full[u]):
                preds[idx] = user_means_full[u]
            else:
                preds[idx] = global_mean

    rmse, mae = evaluate_predictions(true_ratings, preds)

    if verbose:
        print("\nEVALUATION (GOA-k-means, paper-like):")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE : {mae:.6f}")

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "n_clusters": int(n_clusters),
        "n_agents": int(n_agents),
        "n_iterations": int(n_iterations),
        "pca_components": int(pca_components),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-like GOA-k-means on MovieLens 1M.")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters (k).")
    parser.add_argument("--agents", type=int, default=40, help="Number of GOA search agents.")
    parser.add_argument("--iters", type=int, default=100, help="GOA iterations.")
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components (0 = no PCA).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)

    # Varsayılan Config: MovieLens 100K (data/movielens_100k, u1.base/u1.test)
    # Makaledeki 100K senaryosuna daha yakın olması için burada override etmiyoruz.
    config = Config(random_seed=args.seed)

    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)

    n_users, n_items, _ = get_data_info(train_ratings)

    _ = evaluate_goa_kmeans_rs(
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        n_users=n_users,
        n_items=n_items,
        n_clusters=args.k,
        n_agents=args.agents,
        n_iterations=args.iters,
        pca_components=args.pca_components,
        verbose=True,
    )


if __name__ == "__main__":
    main()

