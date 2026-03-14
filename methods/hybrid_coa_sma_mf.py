"""
Hybrid-COA-SMA-MF method: Matrix Factorization with hybrid COA+SMA optimizer.

Tamamen meta-sezgisel bir işbirlikçi filtreleme yaklaşımı:
  - MFModel: U, V latent faktörleri
  - HybridCOASMAOptimizer: U, V'yi global olarak optimize eder

Not:
  - MF-SGD ve HHO-EO-MF ile benzer fit/predict/evaluate arayüzünü takip eder.
  - Bias ve içerik tabanlı başlangıç (X_items, X_users) desteklenir.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from models.mf_model import MFModel
from optimizers.hybrid_coa_sma import HybridCOASMAOptimizer
from core.metrics import evaluate_predictions


class HybridCOASMAMF:
    """
    Hybrid COA+SMA optimizer ile MF modeli.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        latent_dim: int,
        n_agents: int = 40,
        sma_ratio_min: float = 0.1,
        sma_ratio_max: float = 0.9,
        regularization: float = 0.01,
        boundary: float = 1.0,
        random_seed: int = 42,
        use_bias: bool = False,
        X_items: Optional[np.ndarray] = None,
        X_users: Optional[np.ndarray] = None,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.random_seed = random_seed

        V_init = None
        U_init = None
        W_item = None

        if X_items is not None:
            from core.utils import genre_matrix_to_embedding

            V_init, W_item = genre_matrix_to_embedding(
                X_items,
                latent_dim,
                random_seed,
            )
        if X_users is not None:
            from core.utils import user_matrix_to_embedding

            U_init = user_matrix_to_embedding(X_users, latent_dim, random_seed)

        self.model = MFModel(
            n_users,
            n_items,
            latent_dim,
            random_seed=random_seed,
            use_bias=use_bias,
            V_init=V_init,
            U_init=U_init,
            W_item=W_item,
        )

        self.optimizer = HybridCOASMAOptimizer(
            n_agents=n_agents,
            sma_ratio_min=sma_ratio_min,
            sma_ratio_max=sma_ratio_max,
            regularization=regularization,
            boundary=boundary,
        )

    def fit(
        self,
        train_ratings: np.ndarray,
        n_iterations: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """
        Train MF model using hybrid COA+SMA optimizer.

        Args:
            train_ratings: (n_ratings, 3) [user_id, item_id, rating]
            n_iterations: metaheuristic iterasyon sayısı
            verbose: log basılsın mı
        """
        np.random.seed(self.random_seed)

        if self.use_bias:
            from core.utils import compute_biases

            mu, b_u, b_i = compute_biases(
                train_ratings,
                self.n_users,
                self.n_items,
            )
            self.model.set_biases(mu, b_u, b_i)

        history = self.optimizer.optimize(
            self.model,
            train_ratings,
            n_iterations=n_iterations,
            verbose=verbose,
        )

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
        return self.model

