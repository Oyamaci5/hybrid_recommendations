"""
Test cluster-aware matrix factorization on MovieLens 100K with different clusterings.

This script demonstrates that cluster-aware MF properly differentiates
between different clustering algorithms because user factors are tied to cluster structure.
"""

import numpy as np
import os
from scipy.sparse import csr_matrix, load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
import sqlite3

from cluster_aware_mf import ClusterAwareMF, SimplePureClusterMF


def load_ratings_matrix(dataset_path="data/ml-100k/u.data", n_users=943, n_items=1682):
    """Load ratings matrix from MovieLens format."""
    print(f"Loading ratings from {dataset_path}")

    ratings = np.zeros((n_users, n_items))
    with open(dataset_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user_id = int(parts[0]) - 1
            item_id = int(parts[1]) - 1
            rating = int(parts[2])
            ratings[user_id, item_id] = rating

    print(f"Loaded: {np.count_nonzero(ratings)} ratings")
    return csr_matrix(ratings)


def load_cluster_data(algo, k, base_dir="mealpy/results/assignments_lof/ml100k"):
    """Load cluster assignments and centroids from a specific algorithm."""
    dir_name = f"{base_dir}/{algo}_pruneu5_i10_euc_imkpp_minmax_wnmf20_k{k}"

    assignments_path = os.path.join(dir_name, "assignments.npy")
    best_sol_path = os.path.join(dir_name, "best_sol.npy")
    user_vectors_path = os.path.join(dir_name, "wnmf_user_vectors.npy")

    if not all(os.path.exists(p) for p in [assignments_path, best_sol_path, user_vectors_path]):
        return None

    assignments = np.load(assignments_path)
    best_sol = np.load(best_sol_path)
    user_vectors = np.load(user_vectors_path)

    # Decode centroids: reshape from (k*dim,) to (k, dim)
    dim = user_vectors.shape[1]
    centroids = best_sol.reshape(k, dim)

    return {
        'assignments': assignments,
        'centroids': centroids,
        'dim': dim
    }


def test_algorithm(algo, k, ratings_train, ratings_val, model_class="cluster_aware"):
    """Test an algorithm with cluster-aware MF."""
    print(f"\n{'='*70}")
    print(f"Testing {algo} (K={k})")
    print(f"{'='*70}")

    # Load cluster data
    cluster_data = load_cluster_data(algo, k)
    if cluster_data is None:
        print(f"Cluster data not found for {algo} K={k}")
        return None

    assignments = cluster_data['assignments']
    centroids = cluster_data['centroids']
    dim = cluster_data['dim']

    print(f"Cluster assignments loaded: {len(assignments)} users, {len(np.unique(assignments))} clusters")
    print(f"Cluster centroids shape: {centroids.shape}")

    # Create and train model
    if model_class == "pure":
        model = SimplePureClusterMF(
            n_latent=dim,
            n_epochs=30,
            learning_rate=0.05,
            lambda_cluster=0.01,
            lambda_item=0.01,
            seed=42
        )
        print(f"\nTraining SimplePureClusterMF (users share cluster factors)")
    else:
        model = ClusterAwareMF(
            n_latent=dim,
            n_epochs=30,
            learning_rate=0.05,
            lambda_user=0.01,
            lambda_item=0.01,
            lambda_cluster=0.1,
            seed=42
        )
        print(f"\nTraining ClusterAwareMF (lambda_cluster=0.1)")

    if model_class == "pure":
        model.fit(ratings_train, assignments, val_ratings=ratings_val, verbose=True)
    else:
        model.fit(ratings_train, centroids, assignments, val_ratings=ratings_val, verbose=True)

    # Evaluate
    print(f"\nEvaluation:")
    rows_val, cols_val = ratings_val.nonzero()
    actual_val = np.array(ratings_val[rows_val, cols_val]).flatten()
    pred_val = model.predict_batch(rows_val, cols_val)

    mae = mean_absolute_error(actual_val, pred_val)
    rmse = np.sqrt(mean_squared_error(actual_val, pred_val))

    print(f"  Validation MAE:  {mae:.6f}")
    print(f"  Validation RMSE: {rmse:.6f}")

    return {
        'algo': algo,
        'k': k,
        'mae': mae,
        'rmse': rmse,
        'n_clusters': len(np.unique(assignments))
    }


def main():
    print("CLUSTER-AWARE MATRIX FACTORIZATION EVALUATION")
    print("=" * 70)
    print()
    print("This test compares different clustering algorithms when used with")
    print("cluster-aware matrix factorization. With this approach, different")
    print("clusterings SHOULD produce different recommendation quality.")
    print()

    # Load ratings
    ratings_matrix = load_ratings_matrix()

    # Split into train/val
    rows, cols = ratings_matrix.nonzero()
    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_rows, train_cols = rows[train_idx], cols[train_idx]
    val_rows, val_cols = rows[val_idx], cols[val_idx]

    ratings_train = csr_matrix(
        (ratings_matrix.data[train_idx], (train_rows, train_cols)),
        shape=ratings_matrix.shape
    )
    ratings_val = csr_matrix(
        (ratings_matrix.data[val_idx], (val_rows, val_cols)),
        shape=ratings_matrix.shape
    )

    print(f"Train ratings: {ratings_train.nnz}")
    print(f"Val ratings: {ratings_val.nnz}")
    print()

    # Test different algorithms and K values
    algorithms = ['B0_KMEANS', 'HA_AVOAHGS', 'IWO_HHO', 'LIT_GWO']
    k_values = [4, 10, 27]
    results_pure = []
    results_aware = []

    # Test SimplePureClusterMF first
    print("\n" + "=" * 70)
    print("PART 1: SimplePureClusterMF (users in same cluster = same factors)")
    print("=" * 70)

    for k in k_values:
        for algo in algorithms:
            result = test_algorithm(algo, k, ratings_train, ratings_val, model_class="pure")
            if result:
                results_pure.append(result)

    # Test ClusterAwareMF
    print("\n" + "=" * 70)
    print("PART 2: ClusterAwareMF (users regularized toward cluster centroid)")
    print("=" * 70)

    for k in k_values:
        for algo in algorithms:
            result = test_algorithm(algo, k, ratings_train, ratings_val, model_class="cluster_aware")
            if result:
                results_aware.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SimplePureClusterMF Results")
    print("=" * 70)
    if results_pure:
        for k in k_values:
            print(f"\nK = {k}:")
            k_results = [r for r in results_pure if r['k'] == k]
            if k_results:
                for r in sorted(k_results, key=lambda x: x['mae']):
                    print(f"  {r['algo']:15s} | MAE={r['mae']:.6f} RMSE={r['rmse']:.6f}")

                mae_values = [r['mae'] for r in k_results]
                mae_range = max(mae_values) - min(mae_values)
                mae_pct = (mae_range / min(mae_values)) * 100
                print(f"  MAE range: {mae_range:.6f} ({mae_pct:.2f}% of minimum)")

    print("\n" + "=" * 70)
    print("SUMMARY: ClusterAwareMF Results")
    print("=" * 70)
    if results_aware:
        for k in k_values:
            print(f"\nK = {k}:")
            k_results = [r for r in results_aware if r['k'] == k]
            if k_results:
                for r in sorted(k_results, key=lambda x: x['mae']):
                    print(f"  {r['algo']:15s} | MAE={r['mae']:.6f} RMSE={r['rmse']:.6f}")

                mae_values = [r['mae'] for r in k_results]
                mae_range = max(mae_values) - min(mae_values)
                mae_pct = (mae_range / min(mae_values)) * 100
                print(f"  MAE range: {mae_range:.6f} ({mae_pct:.2f}% of minimum)")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("If SimplePureClusterMF shows high MAE variation (>5%),")
    print("then clustering differences matter for recommendation quality.")
    print()
    print("If ClusterAwareMF shows variation similar to SimplePureClusterMF,")
    print("then the cluster regularization is working correctly.")
    print()
    print("If variation is LOWER than traditional cluster_avg (0.95%),")
    print("something is still masking the cluster differences.")
    print()


if __name__ == "__main__":
    main()
