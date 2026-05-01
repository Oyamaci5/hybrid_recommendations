from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from models.base import MatrixFactorizationBase
from utils.config import ModelConfig


class SVDModel(MatrixFactorizationBase):
    """
    FunkSVD / biased matrix factorization (SGD ile).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.mu_: float = 0.0
        self.bu_: np.ndarray | None = None
        self.bi_: np.ndarray | None = None
        self.P_: np.ndarray | None = None
        self.Q_: np.ndarray | None = None

    def fit(self, R, mask=None) -> "SVDModel":
        rng = np.random.default_rng(int(self.cfg.random_state))
        A = R.toarray().astype(np.float64) if issparse(R) else np.asarray(R, dtype=np.float64)
        G = (A != 0) if mask is None else np.asarray(mask, dtype=bool)

        self.mu_ = float(A[G].mean()) if np.any(G) else 3.0
        n_users, n_items = A.shape
        k = int(self.cfg.n_components)
        lr, reg = float(self.cfg.lr), float(self.cfg.reg)
        epochs = int(self.cfg.n_epochs)

        if self.cfg.use_bias:
            self.bu_, self.bi_ = np.zeros(n_users), np.zeros(n_items)
        else:
            self.bu_, self.bi_ = np.zeros(n_users), np.zeros(n_items)
        sd = 0.1
        self.P_ = rng.normal(0, sd, size=(n_users, k))
        self.Q_ = rng.normal(0, sd, size=(n_items, k))

        idx = list(zip(*np.nonzero(G)))
        rng.shuffle(idx)
        for ep in range(epochs):
            for u, i in idx:
                r_ui = float(A[u, i])
                bias = (
                    float(self.mu_ + self.bu_[u] + self.bi_[i])
                    if self.cfg.use_bias
                    else self.mu_
                )
                pred = bias + float(self.P_[u] @ self.Q_[i])
                e = r_ui - pred
                if self.cfg.use_bias:
                    self.bu_[u] += lr * (e - reg * self.bu_[u])
                    self.bi_[i] += lr * (e - reg * self.bi_[i])
                gu = lr * (e * self.Q_[i] - reg * self.P_[u])
                gi = lr * (e * self.P_[u] - reg * self.Q_[i])
                self.P_[u] += gu
                self.Q_[i] += gi
        return self

    def predict(self, u: int, i: int) -> float:
        bias = (
            float(self.mu_ + self.bu_[u] + self.bi_[i])
            if self.cfg.use_bias
            else float(self.mu_)
        )
        return float(np.clip(bias + float(self.P_[u] @ self.Q_[i]), 1.0, 5.0))

    def predict_all(self) -> np.ndarray:
        PQ = self.P_ @ self.Q_.T
        out = np.full_like(PQ, self.mu_, dtype=np.float64) + PQ
        if self.cfg.use_bias:
            out += self.bu_[:, None] + self.bi_[None, :]
        return np.clip(out.astype(np.float32), 1.0, 5.0)

    def get_user_factors(self) -> np.ndarray:
        return self.P_

    def get_item_factors(self) -> np.ndarray:
        return self.Q_
