"""
Compare clustering methods: baselines (PCA-KMeans, SOM, PCA-SOM) vs meta-algorithms.

Tests on MovieLens 100K with WNMF latent features.
"""

import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, davies_bouldin_score, silhouette_score
import sqlite3
import time

from baseline_clustering import PCAKMeans, SOMCluster, PCASOM


def load_wnmf_features(k, algo=None):
    """Load WNMF latent features."""
    if algo:
        dir_name = f"mealpy/results/assignments_lof/ml100k/{algo}_pruneu5_i10_euc_imkpp_minmax_wnmf20_k{k}"
        features_path = os.path.join(dir_name, "wnmf_user_vectors.npy")
    else:
        # Use baseline features (from first algorithm)
        dir_name = f"mealpy/results/assignments_lof/ml100k/B0_KMEANS_pruneu5_i10_euc_imkpp_minmax_wnmf20_k{k}"
        features_path = os.path.join(dir_name, "wnmf_user_vectors.npy")

    if os.path.exists(features_path):
        return np.load(features_path)
    return None


def load_ratings_matrix(dataset_path="data/ml-100k/u.data"):
    """Load ratings from MovieLens."""
    ratings = np.zeros((943, 1682))
    with open(dataset_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            u, i, r = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
            ratings[u, i] = r
    return csr_matrix(ratings)


def evaluate_clustering(X, assignments, name):
    """Evaluate clustering quality metrics."""
    db_score = davies_bouldin_score(X, assignments)
    sil_score = silhouette_score(X, assignments)
    return {'db_score': db_score, 'sil_score': sil_score}


def main():
    print("COMPREHENSIVE CLUSTERING COMPARISON")
    print("=" * 90)
    print()

    # Load data
    print("Loading data...")
    X = load_wnmf_features(k=4)  # Use K=4 for initial comparison
    if X is None:
        print("ERROR: Could not load WNMF features")
        return

    ratings_matrix = load_ratings_matrix()
    rows, cols = ratings_matrix.nonzero()
    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    ratings_val = csr_matrix(
        (ratings_matrix.data[val_idx], (rows[val_idx], cols[val_idx])),
        shape=ratings_matrix.shape
    )

    print(f"Features shape: {X.shape}")
    print(f"Validation ratings: {ratings_val.nnz}\n")

    # === BASELINE METHODS ===
    print("=" * 90)
    print("BASELINE METHODS (K=4)")
    print("=" * 90)

    baseline_results = {}

    # PCA-KMeans
    print("\n1. PCA-KMeans")
    print("-" * 90)
    start = time.time()
    pca_kmeans = PCAKMeans(n_clusters=4, pca_components=20, random_state=42)
    pca_kmeans.fit(X, verbose=True)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.2f}s")

    assignments_pca_kmeans = pca_kmeans.get_labels()
    evals_pca_kmeans = evaluate_clustering(X, assignments_pca_kmeans, "PCA-KMeans")
    baseline_results['PCA-KMeans'] = {
        'assignments': assignments_pca_kmeans,
        'centers': pca_kmeans.get_centers(),
        'evals': evals_pca_kmeans,
        'time': elapsed
    }

    # SOM-Cluster
    print("\n2. SOM-Cluster")
    print("-" * 90)
    start = time.time()
    som = SOMCluster(n_clusters=4, grid_size=2, n_epochs=50, random_state=42)
    som.fit(X, verbose=True)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.2f}s")

    assignments_som = som.get_labels()
    evals_som = evaluate_clustering(X, assignments_som, "SOM-Cluster")
    baseline_results['SOM-Cluster'] = {
        'assignments': assignments_som,
        'centers': som.get_centers(),
        'evals': evals_som,
        'time': elapsed
    }

    # PCA-SOM
    print("\n3. PCA-SOM")
    print("-" * 90)
    start = time.time()
    pca_som = PCASOM(n_clusters=4, pca_components=20, n_epochs=50, random_state=42)
    pca_som.fit(X, verbose=True)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.2f}s")

    assignments_pca_som = pca_som.get_labels()
    evals_pca_som = evaluate_clustering(X, assignments_pca_som, "PCA-SOM")
    baseline_results['PCA-SOM'] = {
        'assignments': assignments_pca_som,
        'centers': pca_som.get_centers(),
        'evals': evals_pca_som,
        'time': elapsed
    }

    # === META-ALGORITHMS (from previous runs) ===
    print("\n" + "=" * 90)
    print("META-ALGORITHMS (K=4)")
    print("=" * 90)

    meta_results = {}
    algorithms = ['B0_KMEANS', 'HA_AVOAHGS', 'IWO_HHO', 'LIT_GWO']

    for algo in algorithms:
        print(f"\n{algo}")
        print("-" * 90)

        dir_name = f"mealpy/results/assignments_lof/ml100k/{algo}_pruneu5_i10_euc_imkpp_minmax_wnmf20_k4"
        assignments_path = os.path.join(dir_name, "assignments.npy")
        best_sol_path = os.path.join(dir_name, "best_sol.npy")

        if os.path.exists(assignments_path) and os.path.exists(best_sol_path):
            assignments = np.load(assignments_path)
            best_sol = np.load(best_sol_path)
            centers = best_sol.reshape(4, 20)

            evals = evaluate_clustering(X, assignments, algo)

            print(f"  Cluster sizes: {np.bincount(assignments)}")
            print(f"  Davies-Bouldin Index: {evals['db_score']:.4f}")
            print(f"  Silhouette Score: {evals['sil_score']:.4f}")

            meta_results[algo] = {
                'assignments': assignments,
                'centers': centers,
                'evals': evals,
                'time': 0  # Not measured
            }

    # === SUMMARY TABLE ===
    print("\n" + "=" * 90)
    print("SUMMARY: Clustering Quality Metrics (K=4)")
    print("=" * 90)

    print("\nDavies-Bouldin Index (lower is better):")
    print("-" * 90)
    all_methods = list(baseline_results.keys()) + list(meta_results.keys())
    db_scores = []
    for method in all_methods:
        if method in baseline_results:
            db = baseline_results[method]['evals']['db_score']
        else:
            db = meta_results[method]['evals']['db_score']
        db_scores.append((method, db))
        print(f"  {method:20s}: {db:.4f}")

    print("\nSilhouette Score (higher is better, range -1 to 1):")
    print("-" * 90)
    sil_scores = []
    for method in all_methods:
        if method in baseline_results:
            sil = baseline_results[method]['evals']['sil_score']
        else:
            sil = meta_results[method]['evals']['sil_score']
        sil_scores.append((method, sil))
        print(f"  {method:20s}: {sil:.4f}")

    print("\nTraining Time (seconds, for baselines only):")
    print("-" * 90)
    for method in all_methods:
        if method in baseline_results:
            t = baseline_results[method]['time']
            print(f"  {method:20s}: {t:.2f}s")

    # === RANKING ===
    print("\n" + "=" * 90)
    print("RANKINGS")
    print("=" * 90)

    print("\nBest Davies-Bouldin Score (lower is better):")
    for i, (method, score) in enumerate(sorted(db_scores, key=lambda x: x[1])[:5], 1):
        method_type = "Baseline" if method in baseline_results else "Meta-algorithm"
        print(f"  {i}. {method:20s} ({method_type:15s}): {score:.4f}")

    print("\nBest Silhouette Score (higher is better):")
    for i, (method, score) in enumerate(sorted(sil_scores, key=lambda x: x[1], reverse=True)[:5], 1):
        method_type = "Baseline" if method in baseline_results else "Meta-algorithm"
        print(f"  {i}. {method:20s} ({method_type:15s}): {score:.4f}")

    # === ANALYSIS ===
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    baseline_db = np.mean([baseline_results[m]['evals']['db_score'] for m in baseline_results])
    meta_db = np.mean([meta_results[m]['evals']['db_score'] for m in meta_results])

    baseline_sil = np.mean([baseline_results[m]['evals']['sil_score'] for m in baseline_results])
    meta_sil = np.mean([meta_results[m]['evals']['sil_score'] for m in meta_results])

    print(f"\nAverage Davies-Bouldin Score:")
    print(f"  Baselines:       {baseline_db:.4f}")
    print(f"  Meta-algorithms: {meta_db:.4f}")
    print(f"  Difference:      {abs(baseline_db - meta_db):.4f} ({abs(baseline_db - meta_db) / meta_db * 100:.1f}%)")

    print(f"\nAverage Silhouette Score:")
    print(f"  Baselines:       {baseline_sil:.4f}")
    print(f"  Meta-algorithms: {meta_sil:.4f}")
    print(f"  Difference:      {abs(baseline_sil - meta_sil):.4f} ({abs(baseline_sil - meta_sil) / meta_sil * 100:.1f}%)")

    print("\nKey Insights:")
    if baseline_db < meta_db:
        print(f"  - Simpler baseline methods achieve BETTER Davies-Bouldin scores")
        print(f"    (Baselines are {(meta_db - baseline_db) / baseline_db * 100:.1f}% better)")
    else:
        print(f"  - Meta-algorithms achieve BETTER Davies-Bouldin scores")
        print(f"    (Meta-algorithms are {(baseline_db - meta_db) / meta_db * 100:.1f}% better)")

    if baseline_sil > meta_sil:
        print(f"  - Baselines achieve HIGHER Silhouette scores")
        print(f"    (Baselines are {(baseline_sil - meta_sil) / abs(meta_sil) * 100:.1f}% better)")
    else:
        print(f"  - Meta-algorithms achieve HIGHER Silhouette scores")
        print(f"    (Meta-algorithms are {(meta_sil - baseline_sil) / baseline_sil * 100:.1f}% better)")

    print("\nImplications:")
    print("  If baselines score similarly to meta-algorithms, then sophisticated")
    print("  optimization may not be necessary for clustering in this domain.")
    print("  Simple PCA+KMeans might be 'good enough' with much faster training.")

    return baseline_results, meta_results


if __name__ == "__main__":
    baseline_results, meta_results = main()
