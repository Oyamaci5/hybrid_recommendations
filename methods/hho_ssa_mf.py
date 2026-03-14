"""
HHO-SSA-MF method: Matrix Factorization with hybrid HHO+SSA optimizer.

Tamamen meta-sezgisel bir işbirlikçi filtreleme yaklaşımı:
  - MFModel: U, V latent faktörleri
  - HHOSSAOptimizer: U, V'yi global olarak optimize eder

Not:
  - Bu sınıf, MF-SGD'ye benzer bir arayüz sunar (fit/predict/evaluate).
  - Bias ve içerik tabanlı başlangıç (X_items, X_users) desteklenir.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from models.mf_model import MFModel
from optimizers.hho_ssa import HHOSSAOptimizer
from core.metrics import evaluate_predictions


class HHOSSAMF:
    """
    HHO-SSA-MF method combining MF model with hybrid HHO+SSA optimizer.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        latent_dim: int,
        n_agents: int = 40,
        escape_energy_initial: float = 1.0,
        regularization: float = 0.01,
        boundary: float = 2.0,
        safety_threshold: float = 0.7,
        producer_ratio: float = 0.2,
        awareness_ratio: float = 0.1,
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
                X_items, latent_dim, random_seed
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

        self.optimizer = HHOSSAOptimizer(
            n_agents=n_agents,
            escape_energy_initial=escape_energy_initial,
            regularization=regularization,
            boundary=boundary,
            safety_threshold=safety_threshold,
            producer_ratio=producer_ratio,
            awareness_ratio=awareness_ratio,
        )

    def fit(
        self,
        train_ratings: np.ndarray,
        n_iterations: int = 300,
        verbose: bool = False,
    ) -> Dict:
        """
        Train MF model using HHO+SSA optimizer.
        """
        np.random.seed(self.random_seed)

        if self.use_bias:
            from core.utils import compute_biases

            mu, b_u, b_i = compute_biases(
                train_ratings, self.n_users, self.n_items
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

