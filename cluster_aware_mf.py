"""
Cluster-Aware Matrix Factorization for Recommendations

Ties user factors to clustering results so that different clusterings
produce different recommendation predictions.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class ClusterAwareMF:
    """
    Matrix Factorization with cluster regularization.

    User factors are initialized from and regularized toward their cluster centroid.
    This makes clustering quality DIRECTLY impact prediction accuracy.
    """

    def __init__(self, n_latent=20, n_epochs=50, learning_rate=0.01,
                 lambda_user=0.01, lambda_item=0.01, lambda_cluster=0.1, seed=42):
        """
        Args:
            n_latent: dimensionality of latent factors
            n_epochs: number of training epochs
            learning_rate: SGD learning rate
            lambda_user: L2 regularization for user factors
            lambda_item: L2 regularization for item factors
            lambda_cluster: regularization strength pulling users toward cluster centroid
                          (higher = stronger cluster influence)
            seed: random seed
        """
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.lambda_cluster = lambda_cluster
        self.seed = seed
        np.random.seed(seed)

        self.user_factors = None
        self.item_factors = None
        self.cluster_centroids = None
        self.user_assignments = None

    def fit(self, ratings_matrix, cluster_centroids, user_assignments,
            val_ratings=None, verbose=True):
        """
        Train the model with cluster regularization.

        Args:
            ratings_matrix: sparse matrix of shape (n_users, n_items) with ratings
            cluster_centroids: array of shape (n_clusters, n_latent) with cluster centers
            user_assignments: array of shape (n_users,) with cluster assignments
            val_ratings: optional sparse matrix for validation
            verbose: print progress
        """
        self.cluster_centroids = cluster_centroids
        self.user_assignments = user_assignments

        n_users, n_items = ratings_matrix.shape
        n_clusters = cluster_centroids.shape[0]

        # Initialize user factors from cluster centroids
        self.user_factors = np.zeros((n_users, self.n_latent))
        for u in range(n_users):
            cluster_id = user_assignments[u]
            # Start at cluster centroid + small random offset
            self.user_factors[u] = cluster_centroids[cluster_id] + \
                                   np.random.randn(self.n_latent) * 0.01

        # Initialize item factors randomly
        self.item_factors = np.random.randn(n_items, self.n_latent) * 0.01

        # Convert to lil format for efficient iteration
        R_lil = ratings_matrix.tolil()

        if verbose:
            print(f"Training cluster-aware MF:")
            print(f"  Users: {n_users}, Items: {n_items}, Latent: {self.n_latent}")
            print(f"  Clusters: {n_clusters}")
            print(f"  Lambda cluster (regularization strength): {self.lambda_cluster}")

        best_val_mae = float('inf')

        for epoch in range(self.n_epochs):
            # Stochastic gradient descent on ratings
            for u in range(n_users):
                # Get rated items for this user
                rated_items = R_lil[u].nonzero()[1]
                if len(rated_items) == 0:
                    continue

                for i in rated_items:
                    rating = ratings_matrix[u, i]
                    pred = np.dot(self.user_factors[u], self.item_factors[i])
                    error = rating - pred

                    # Clip error to prevent explosion
                    error = np.clip(error, -5, 5)

                    # Gradient updates with smaller step
                    self.user_factors[u] += self.learning_rate * 0.1 * (
                        error * self.item_factors[i] -
                        self.lambda_user * self.user_factors[u]
                    )

                    self.item_factors[i] += self.learning_rate * 0.1 * (
                        error * self.user_factors[u] -
                        self.lambda_item * self.item_factors[i]
                    )

            # Apply cluster regularization: pull users toward their cluster centroid
            for u in range(n_users):
                cluster_id = user_assignments[u]
                cluster_center = cluster_centroids[cluster_id]

                # Gradient to move toward cluster center
                deviation = self.user_factors[u] - cluster_center
                self.user_factors[u] -= self.learning_rate * 0.1 * self.lambda_cluster * deviation

            # Clip factors to prevent explosion
            self.user_factors = np.clip(self.user_factors, -10, 10)
            self.item_factors = np.clip(self.item_factors, -10, 10)

            # Evaluate
            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_mae = self._compute_mae(ratings_matrix)

                if val_ratings is not None:
                    val_mae = self._compute_mae(val_ratings)
                    if verbose:
                        print(f"  Epoch {epoch+1:3d}: Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")

                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                else:
                    if verbose:
                        print(f"  Epoch {epoch+1:3d}: Train MAE={train_mae:.4f}")

        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair."""
        if user_id >= len(self.user_factors) or item_id >= len(self.item_factors):
            return 0.0
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])

    def predict_batch(self, user_ids, item_ids):
        """Predict ratings for multiple pairs."""
        preds = np.zeros(len(user_ids))
        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            preds[idx] = self.predict(u, i)
        return preds

    def _compute_mae(self, ratings_matrix):
        """Compute MAE on a ratings matrix."""
        rows, cols = ratings_matrix.nonzero()
        actual = np.array(ratings_matrix[rows, cols]).flatten()
        predicted = self.predict_batch(rows, cols)
        return mean_absolute_error(actual, predicted)

    def get_user_factors(self):
        """Return learned user factors."""
        return self.user_factors.copy()

    def get_item_factors(self):
        """Return learned item factors."""
        return self.item_factors.copy()


class SimplePureClusterMF:
    """
    Simplest cluster-aware approach: users in same cluster have identical factors.

    Prediction = cluster_factor . item_factor

    This is the purest test of whether clustering structure matters.
    """

    def __init__(self, n_latent=20, n_epochs=50, learning_rate=0.01,
                 lambda_cluster=0.01, lambda_item=0.01, seed=42):
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.lambda_cluster = lambda_cluster
        self.lambda_item = lambda_item
        self.seed = seed
        np.random.seed(seed)

        self.cluster_factors = None
        self.item_factors = None
        self.user_assignments = None

    def fit(self, ratings_matrix, user_assignments, val_ratings=None, verbose=True):
        """
        Train cluster factors and item factors.

        All users in same cluster share a factor vector.
        """
        self.user_assignments = user_assignments

        n_users, n_items = ratings_matrix.shape
        n_clusters = len(np.unique(user_assignments))

        # Initialize cluster factors
        self.cluster_factors = np.random.randn(n_clusters, self.n_latent) * 0.01
        self.item_factors = np.random.randn(n_items, self.n_latent) * 0.01

        R_lil = ratings_matrix.tolil()

        if verbose:
            print(f"Training pure cluster MF:")
            print(f"  Users: {n_users}, Items: {n_items}, Latent: {self.n_latent}")
            print(f"  Clusters: {n_clusters}")
            print(f"  (All users in same cluster share identical factors)")

        for epoch in range(self.n_epochs):
            for u in range(n_users):
                cluster_id = user_assignments[u]
                rated_items = R_lil[u].nonzero()[1]
                if len(rated_items) == 0:
                    continue

                for i in rated_items:
                    rating = ratings_matrix[u, i]
                    pred = np.dot(self.cluster_factors[cluster_id], self.item_factors[i])
                    error = rating - pred

                    # Clip error to prevent explosion
                    error = np.clip(error, -5, 5)

                    # Update cluster factor with smaller step
                    self.cluster_factors[cluster_id] += self.learning_rate * 0.1 * (
                        error * self.item_factors[i] -
                        self.lambda_cluster * self.cluster_factors[cluster_id]
                    )

                    # Update item factor with smaller step
                    self.item_factors[i] += self.learning_rate * 0.1 * (
                        error * self.cluster_factors[cluster_id] -
                        self.lambda_item * self.item_factors[i]
                    )

            # Clip factors to prevent explosion
            self.cluster_factors = np.clip(self.cluster_factors, -10, 10)
            self.item_factors = np.clip(self.item_factors, -10, 10)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_mae = self._compute_mae(ratings_matrix)
                if verbose:
                    print(f"  Epoch {epoch+1:3d}: Train MAE={train_mae:.4f}")

        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair."""
        cluster_id = self.user_assignments[user_id]
        return np.dot(self.cluster_factors[cluster_id], self.item_factors[item_id])

    def predict_batch(self, user_ids, item_ids):
        """Predict ratings for multiple pairs."""
        preds = np.zeros(len(user_ids))
        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            preds[idx] = self.predict(u, i)
        return preds

    def _compute_mae(self, ratings_matrix):
        """Compute MAE on a ratings matrix."""
        rows, cols = ratings_matrix.nonzero()
        actual = np.array(ratings_matrix[rows, cols]).flatten()
        predicted = self.predict_batch(rows, cols)
        return mean_absolute_error(actual, predicted)


if __name__ == "__main__":
    print("Cluster-Aware Matrix Factorization")
    print("=" * 70)
    print()
    print("This module implements two cluster-aware recommendation approaches:")
    print()
    print("1. ClusterAwareMF:")
    print("   - User factors initialized from and regularized toward cluster centroid")
    print("   - lambda_cluster controls strength of cluster influence")
    print("   - Higher lambda_cluster = stronger cluster effect on predictions")
    print()
    print("2. SimplePureClusterMF:")
    print("   - All users in same cluster share identical learned factors")
    print("   - Purest test: prediction = cluster_factor . item_factor")
    print("   - Clustering quality directly determines recommendation quality")
    print()
    print("Expected behavior:")
    print("   B0_KMEANS vs HA_AVOAHGS vs IWO_HHO vs LIT_GWO")
    print("   (different cluster assignments)")
    print("   --> Different learned cluster factors")
    print("   --> Different predictions")
    print("   --> Different MAE scores")
    print()
