"""
Baseline clustering methods for comparison with meta-heuristic algorithms.

Implements:
- KMeans: direct KMeans on feature space
- PCA-KMeans: PCA dimensionality reduction + KMeans clustering
- SOM-Cluster: Self-Organizing Maps for clustering
- PCA-SOM: PCA + Self-Organizing Maps
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')


class KMeansCluster:
    """Direct KMeans clustering on the original feature space."""

    def __init__(self, n_clusters=4, kmeans_init='k-means++', random_state=42):
        self.n_clusters = n_clusters
        self.kmeans_init = kmeans_init
        self.random_state = random_state

        self.kmeans = None
        self.labels = None
        self.centers = None

    def fit(self, X, verbose=True):
        n_samples, n_features = X.shape

        if verbose:
            print(f"[KMeans] Input shape: {X.shape}")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=self.kmeans_init,
            n_init=10,
            random_state=self.random_state,
            verbose=0,
        )
        self.labels = self.kmeans.fit_predict(X)
        self.centers = self.kmeans.cluster_centers_

        if verbose:
            print(f"  Cluster sizes: {np.bincount(self.labels)}")
            db_score = davies_bouldin_score(X, self.labels)
            print(f"  Davies-Bouldin Index: {db_score:.4f}")

        return self

    def predict(self, X):
        return self.kmeans.predict(X)

    def get_centers(self):
        return self.centers.copy()

    def get_labels(self):
        return self.labels.copy()


class PCAKMeans:
    """PCA-based dimensionality reduction followed by KMeans clustering."""

    def __init__(self, n_clusters=4, pca_components=None, kmeans_init='k-means++',
                 random_state=42):
        """
        Args:
            n_clusters: number of clusters
            pca_components: number of PCA components (None = auto, or int)
            kmeans_init: initialization method for KMeans
            random_state: random seed
        """
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.kmeans_init = kmeans_init
        self.random_state = random_state

        self.pca = None
        self.kmeans = None
        self.labels = None
        self.centers = None

    def fit(self, X, verbose=True):
        """
        Fit PCA-KMeans to data.

        Args:
            X: data matrix of shape (n_samples, n_features)
            verbose: print progress
        """
        n_samples, n_features = X.shape

        # Auto-select PCA components if not specified
        if self.pca_components is None:
            # Keep 95% of variance
            pca_temp = PCA(n_components=min(n_samples, n_features), random_state=self.random_state)
            pca_temp.fit(X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum >= 0.95) + 1
            pca_components = max(2, min(n_comp, n_features // 2))
        else:
            pca_components = self.pca_components

        if verbose:
            print(f"[PCA-KMeans] Input shape: {X.shape}")
            print(f"  PCA components: {pca_components}")

        # Apply PCA
        self.pca = PCA(n_components=pca_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)

        if verbose:
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"  Explained variance: {explained_var:.4f}")

        # Apply KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=self.kmeans_init,
            n_init=10,
            random_state=self.random_state,
            verbose=0
        )
        self.labels = self.kmeans.fit_predict(X_pca)

        # Transform cluster centers back to original space
        self.centers = self.pca.inverse_transform(self.kmeans.cluster_centers_)

        if verbose:
            print(f"  Cluster sizes: {np.bincount(self.labels)}")
            db_score = davies_bouldin_score(X, self.labels)
            print(f"  Davies-Bouldin Index: {db_score:.4f}")

        return self

    def predict(self, X):
        """Predict cluster assignments for new data."""
        X_pca = self.pca.transform(X)
        return self.kmeans.predict(X_pca)

    def get_centers(self):
        """Return cluster centers in original space."""
        return self.centers.copy()

    def get_labels(self):
        """Return cluster assignments."""
        return self.labels.copy()


class SOMCluster:
    """Self-Organizing Maps for clustering."""

    def __init__(self, n_clusters=4, grid_size=None, learning_rate=0.5,
                 n_epochs=100, radius_start=None, random_state=42):
        """
        Args:
            n_clusters: target number of clusters
            grid_size: SOM grid size (None = auto)
            learning_rate: initial learning rate
            n_epochs: number of training epochs
            radius_start: initial radius (None = auto)
            random_state: random seed
        """
        self.n_clusters = n_clusters
        self.grid_size = grid_size or int(np.sqrt(n_clusters * 1.5))
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.radius_start = radius_start or max(self.grid_size / 2, 1.0)
        self.random_state = random_state

        self.weights = None
        self.labels = None
        self.centers = None

    def fit(self, X, verbose=True):
        """
        Fit SOM to data and extract clusters via KMeans on neuron weights.

        Args:
            X: data matrix of shape (n_samples, n_features)
            verbose: print progress
        """
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        if verbose:
            print(f"[SOM-Cluster] Input shape: {X.shape}")
            print(f"  Grid size: {self.grid_size}x{self.grid_size}")
            print(f"  Training epochs: {self.n_epochs}")

        # Initialize SOM weights
        n_neurons = self.grid_size * self.grid_size
        self.weights = np.random.randn(n_neurons, n_features) * 0.1

        # Normalize input
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Train SOM
        for epoch in range(self.n_epochs):
            # Decay learning rate and radius
            lr = self.learning_rate * np.exp(-epoch / self.n_epochs)
            radius = self.radius_start * np.exp(-epoch / self.n_epochs)

            # Random sample
            idx = np.random.randint(0, n_samples)
            x = X_norm[idx]

            # Find best matching unit (BMU)
            distances = np.linalg.norm(self.weights - x, axis=1)
            bmu_idx = np.argmin(distances)
            bmu_pos = np.array([bmu_idx // self.grid_size, bmu_idx % self.grid_size])

            # Update weights
            for i in range(n_neurons):
                neuron_pos = np.array([i // self.grid_size, i % self.grid_size])
                dist_to_bmu = np.linalg.norm(neuron_pos - bmu_pos)

                if dist_to_bmu < radius:
                    influence = np.exp(-(dist_to_bmu ** 2) / (2 * radius ** 2))
                    self.weights[i] += lr * influence * (x - self.weights[i])

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1:3d}: LR={lr:.4f}, Radius={radius:.4f}")

        # Cluster neurons using KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        neuron_clusters = kmeans.fit_predict(self.weights)
        self.centers = kmeans.cluster_centers_

        # Assign samples to closest neuron, then to cluster
        distances = np.linalg.norm(X_norm[:, None, :] - self.weights[None, :, :], axis=2)
        neuron_assignments = np.argmin(distances, axis=1)
        self.labels = neuron_clusters[neuron_assignments]

        # Store centers in original scale
        self.centers = self.centers * X.std(axis=0) + X.mean(axis=0)

        if verbose:
            print(f"  Cluster sizes: {np.bincount(self.labels)}")
            db_score = davies_bouldin_score(X, self.labels)
            print(f"  Davies-Bouldin Index: {db_score:.4f}")

        return self

    def predict(self, X):
        """Predict cluster assignments for new data."""
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        distances = np.linalg.norm(X_norm[:, None, :] - self.weights[None, :, :], axis=2)
        neuron_assignments = np.argmin(distances, axis=1)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=1)
        kmeans.fit(self.weights)
        neuron_clusters = kmeans.predict(self.weights)
        return neuron_clusters[neuron_assignments]

    def get_centers(self):
        """Return cluster centers."""
        return self.centers.copy()

    def get_labels(self):
        """Return cluster assignments."""
        return self.labels.copy()


class PCASOM:
    """PCA-based dimensionality reduction followed by SOM clustering."""

    def __init__(self, n_clusters=4, pca_components=None, grid_size=None,
                 learning_rate=0.5, n_epochs=100, random_state=42):
        """
        Args:
            n_clusters: target number of clusters
            pca_components: number of PCA components (None = auto)
            grid_size: SOM grid size (None = auto)
            learning_rate: SOM learning rate
            n_epochs: SOM training epochs
            random_state: random seed
        """
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.grid_size = grid_size or int(np.sqrt(n_clusters * 1.5))
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.pca = None
        self.som = None
        self.labels = None
        self.centers = None

    def fit(self, X, verbose=True):
        """
        Fit PCA-SOM to data.

        Args:
            X: data matrix of shape (n_samples, n_features)
            verbose: print progress
        """
        n_samples, n_features = X.shape

        # Auto-select PCA components
        if self.pca_components is None:
            pca_temp = PCA(n_components=min(n_samples, n_features), random_state=self.random_state)
            pca_temp.fit(X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum >= 0.95) + 1
            pca_components = max(2, min(n_comp, n_features // 2))
        else:
            pca_components = self.pca_components

        if verbose:
            print(f"[PCA-SOM] Input shape: {X.shape}")
            print(f"  PCA components: {pca_components}")

        # Apply PCA
        self.pca = PCA(n_components=pca_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)

        if verbose:
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"  Explained variance: {explained_var:.4f}")

        # Apply SOM on PCA-reduced data
        self.som = SOMCluster(
            n_clusters=self.n_clusters,
            grid_size=self.grid_size,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            random_state=self.random_state
        )
        self.som.fit(X_pca, verbose=False)

        self.labels = self.som.get_labels()

        # Transform centers back to original space
        som_centers_pca = self.som.get_centers()
        self.centers = self.pca.inverse_transform(som_centers_pca)

        if verbose:
            print(f"  Cluster sizes: {np.bincount(self.labels)}")
            db_score = davies_bouldin_score(X, self.labels)
            print(f"  Davies-Bouldin Index: {db_score:.4f}")

        return self

    def predict(self, X):
        """Predict cluster assignments for new data."""
        X_pca = self.pca.transform(X)
        return self.som.predict(X_pca)

    def get_centers(self):
        """Return cluster centers in original space."""
        return self.centers.copy()

    def get_labels(self):
        """Return cluster assignments."""
        return self.labels.copy()


def _load_user_features(k: int, wnmf_dim: int = 20) -> np.ndarray:
    repo = os.path.dirname(os.path.abspath(__file__))
    assign_root = os.path.join(repo, "mealpy", "results", "assignments", "ml100k")
    preferred = os.path.join(
        assign_root,
        f"B0_KMEANS_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf{wnmf_dim}_k{k}",
        "user_features.npy",
    )
    if os.path.isfile(preferred):
        return np.load(preferred)

    pattern = f"*wnmf{wnmf_dim}_k{k}"
    for entry in sorted(os.listdir(assign_root)):
        if not entry.endswith(f"_wnmf{wnmf_dim}_k{k}"):
            continue
        path = os.path.join(assign_root, entry, "user_features.npy")
        if os.path.isfile(path):
            return np.load(path)

    raise FileNotFoundError(
        f"user_features.npy bulunamadi (k={k}, wnmf_dim={wnmf_dim}, root={assign_root})"
    )


def _cluster_metrics(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> dict:
    wcss = float(
        np.sum((X - centers[labels]) ** 2)
    )
    return {
        "wcss": wcss,
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "silhouette": float(silhouette_score(X, labels)),
    }


def _eval_cluster_avg(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
) -> dict:
    from wnmf.meta_dual_cf import _baseline_cluster_avg_user

    m = _baseline_cluster_avg_user(train, test, labels)
    return {
        "mae": round(m["mae"], 4),
        "rmse": round(m["rmse"], 4),
        "precision_at_10": round(m["precision_at_10"], 4),
        "recall_at_10": round(m["recall_at_10"], 4),
        "ndcg_at_10": round(m["ndcg_at_10"], 4),
    }


def _fit_baselines_for_k(
    X: np.ndarray,
    k: int,
    *,
    pca_components: int = 20,
    som_epochs: int = 50,
    random_state: int = 42,
) -> list[dict]:
    grid = max(4, int(np.ceil(np.sqrt(k * 2))))
    methods = [
        ("KMeans", lambda: KMeansCluster(n_clusters=k, random_state=random_state)),
        ("PCA-KMeans", lambda: PCAKMeans(
            n_clusters=k, pca_components=pca_components, random_state=random_state,
        )),
        ("SOM-Cluster", lambda: SOMCluster(
            n_clusters=k, grid_size=grid, n_epochs=som_epochs, random_state=random_state,
        )),
        ("PCA-SOM", lambda: PCASOM(
            n_clusters=k,
            pca_components=pca_components,
            grid_size=grid,
            n_epochs=som_epochs,
            random_state=random_state,
        )),
    ]

    fitted: list[dict] = []
    for name, factory in methods:
        t0 = time.time()
        model = factory()
        model.fit(X, verbose=False)
        fit_sec = time.time() - t0
        labels = model.get_labels()
        centers = model.get_centers()
        metrics = _cluster_metrics(X, labels, centers)
        fitted.append({
            "K": k,
            "method": name,
            "labels": labels,
            "fit_sec": round(fit_sec, 2),
            **{key: round(val, 4) for key, val in metrics.items()},
        })
    return fitted


def _aggregate_cv5(fold_rows: list[dict]) -> pd.DataFrame:
    rec_cols = ["mae", "rmse", "precision_at_10", "recall_at_10", "ndcg_at_10"]
    df = pd.DataFrame(fold_rows)
    agg = df.groupby(["K", "method"], as_index=False).agg(
        wcss=("wcss", "first"),
        davies_bouldin=("davies_bouldin", "first"),
        silhouette=("silhouette", "first"),
        fit_sec=("fit_sec", "first"),
        **{c: (c, "mean") for c in rec_cols},
        **{f"{c}_std": (c, "std") for c in rec_cols},
    )
    for c in rec_cols:
        agg[c] = agg[c].round(4)
        agg[f"{c}_std"] = agg[f"{c}_std"].round(4)
    return agg


def run_k_sweep(
    k_list: list[int],
    *,
    wnmf_dim: int = 20,
    pca_components: int = 20,
    som_epochs: int = 50,
    fold: int | None = 1,
    n_folds: int = 1,
    csv_path: str | None = None,
) -> pd.DataFrame:
    repo = os.path.dirname(os.path.abspath(__file__))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from wnmf.wnmf_utils import load_ratings_100k_all

    data_path = os.path.join(repo, "data", "ml-100k", "u.data")
    rec_cols = ["mae", "rmse", "precision_at_10", "recall_at_10", "ndcg_at_10"]
    cluster_cols = ["K", "method", "wcss", "davies_bouldin", "silhouette", "fit_sec"]

    if n_folds > 1:
        folds = list(range(1, n_folds + 1))
    elif fold is not None:
        folds = [fold]
    else:
        folds = [1]

    fold_rows: list[dict] = []
    for k in k_list:
        X = _load_user_features(k, wnmf_dim=wnmf_dim)
        fitted = _fit_baselines_for_k(
            X, k,
            pca_components=pca_components,
            som_epochs=som_epochs,
        )
        for f in folds:
            train, test = load_ratings_100k_all(data_path, random_seed=42, fold=f)
            for item in fitted:
                rec = _eval_cluster_avg(train, test, item["labels"])
                fold_rows.append({
                    "fold": f,
                    "K": item["K"],
                    "method": item["method"],
                    "wcss": item["wcss"],
                    "davies_bouldin": item["davies_bouldin"],
                    "silhouette": item["silhouette"],
                    "fit_sec": item["fit_sec"],
                    **rec,
                })

    if n_folds > 1:
        df_detail = pd.DataFrame(fold_rows)
        df = _aggregate_cv5(fold_rows)

        print("\nBaseline Clustering — K sweep (cluster metrikleri)")
        print("=" * 88)
        print(
            df[cluster_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

        print(f"\nBaseline Clustering — 5-fold CV ortalamasi (cluster_avg, seed=42)")
        print("=" * 100)
        summary_cols = ["K", "method"] + rec_cols
        print(
            df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

        print(f"\nStandart sapmalar (5-fold)")
        print("=" * 100)
        std_cols = ["K", "method"] + [f"{c}_std" for c in rec_cols]
        print(
            df[std_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    else:
        df_detail = pd.DataFrame(fold_rows)
        df = df_detail.drop(columns=["fold"])
        print("\nBaseline Clustering — K sweep (cluster metrikleri)")
        print("=" * 88)
        print(
            df[cluster_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        print(f"\nBaseline Clustering — K sweep (cluster_avg, fold={folds[0]})")
        print("=" * 88)
        print(
            df[["K", "method"] + rec_cols].to_string(
                index=False, float_format=lambda x: f"{x:.4f}",
            )
        )

    if csv_path:
        out = csv_path if os.path.isabs(csv_path) else os.path.join(repo, csv_path)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nCSV (ozet): {out}")
        if n_folds > 1:
            detail_path = out.replace(".csv", "_folds.csv")
            df_detail.to_csv(detail_path, index=False)
            print(f"CSV (fold detay): {detail_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline clustering K sweep")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[3, 6, 9, 10, 11, 12, 14],
        help="K degerleri",
    )
    parser.add_argument("--wnmf-dim", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--som-epochs", type=int, default=50)
    parser.add_argument("--fold", type=int, default=None, help="Tek fold (1-5); --cv5 ile birlikte kullanilmaz")
    parser.add_argument("--cv5", action="store_true", help="5-fold CV ortalamasi")
    parser.add_argument("--csv", default="results/baseline_clustering_k_sweep.csv")
    args = parser.parse_args()

    run_k_sweep(
        args.k,
        wnmf_dim=args.wnmf_dim,
        pca_components=args.pca_components,
        som_epochs=args.som_epochs,
        fold=args.fold if not args.cv5 else None,
        n_folds=5 if args.cv5 else 1,
        csv_path=args.csv,
    )
