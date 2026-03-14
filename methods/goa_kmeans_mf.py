"""
GOA-KMeans-MF method: Matrix Factorization with Grasshopper Optimization Algorithm
initialization followed by K-Means clustering refinement.

Bu sınıf, Ambikesh et al. (2024) GOA-k-means fikrini MF tabanlı sisteme uyarlar:
  1. GOA: U, V için iyi bir global başlangıç bulur.
  2. K-Means: Kullanıcı/öğe gömlemelerini kümelere yaklaştırarak rafine eder.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from models.mf_model import MFModel
from optimizers.goa import GOAOptimizer
from optimizers.kmeans import KMeansOptimizer
from core.metrics import evaluate_predictions


class GOAKMeansMF:
    """
    GOA-KMeans-MF method combining GOA initialization with K-Means refinement.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        latent_dim: int,
        n_grasshoppers: int = 40,
        n_clusters_users: int | None = None,
        n_clusters_items: int | None = None,
        learning_rate: float = 0.1,
        regularization: float = 0.01,
        boundary: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize GOA-KMeans-MF method.

        Args:
            n_users: Number of users
            n_items: Number of items
            latent_dim: Number of latent factors (k)
            n_grasshoppers: Number of search agents in GOA population
            n_clusters_users: Number of clusters for user embeddings
            n_clusters_items: Number of clusters for item embeddings
            learning_rate: K-Means learning rate
            regularization: L2 regularization coefficient
            boundary: Boundary constraint for embeddings
            random_seed: Random seed for reproducibility
        """
        self.model = MFModel(n_users, n_items, latent_dim, random_seed)
        self.goa_optimizer = GOAOptimizer(
            n_grasshoppers=n_grasshoppers,
            regularization=regularization,
            boundary=boundary,
        )
        self.kmeans_optimizer = KMeansOptimizer(
            n_clusters_users=n_clusters_users,
            n_clusters_items=n_clusters_items,
            learning_rate=learning_rate,
            regularization=regularization,
        )
        self.random_seed = random_seed

    def fit(
        self,
        train_ratings: np.ndarray,
        goa_iterations: int = 50,
        kmeans_iterations: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """
        Train the model using GOA initialization followed by K-Means refinement.

        Args:
            train_ratings: Training ratings array of shape (n_ratings, 3)
                          with columns [user_id, item_id, rating]
            goa_iterations: Number of GOA iterations for initialization
            kmeans_iterations: Number of K-Means iterations for refinement
            verbose: Whether to print progress

        Returns:
            Dictionary with training history (GOA and K-Means phases)
        """
        np.random.seed(self.random_seed)

        # Phase 1: GOA initialization
        if verbose:
            print("=" * 60)
            print("Phase 1: GOA Initialization")
            print("=" * 60)

        goa_history = self.goa_optimizer.optimize(
            self.model,
            train_ratings,
            n_iterations=goa_iterations,
            verbose=verbose,
        )

        # Phase 2: K-Means refinement
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 2: K-Means Refinement")
            print("=" * 60)

        kmeans_history = self.kmeans_optimizer.optimize(
            self.model,
            train_ratings,
            n_iterations=kmeans_iterations,
            verbose=verbose,
        )

        history = {
            "goa_losses": goa_history["losses"],
            "goa_iterations": goa_history["iterations"],
            "kmeans_losses": kmeans_history["losses"],
            "kmeans_iterations": kmeans_history["iterations"],
        }

        return history

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        return self.model.predict(user_ids, item_ids)

    def evaluate(self, test_ratings: np.ndarray) -> Tuple[float, float]:
        user_ids = test_ratings[:, 0].astype(int)
        item_ids = test_ratings[:, 1].astype(int)
        true_ratings = test_ratings[:, 2]

        pred_ratings = self.predict(user_ids, item_ids)
        return evaluate_predictions(true_ratings, pred_ratings)

    def get_model(self) -> MFModel:
        """Get the underlying MF model."""
        return self.model

