from __future__ import annotations

import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD

from models.base import MatrixFactorizationBase
from utils.config import ModelConfig


class WNMF(MatrixFactorizationBase):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.W_: np.ndarray | None = None
        self.H_: np.ndarray | None = None

    def _init_factors(self, R: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.cfg.random_state)
        if self.cfg.init == "nndsvda":
            filled = R.copy()
            obs = filled > 0
            mean_val = float(filled[obs].mean()) if np.any(obs) else 0.0
            filled[~obs] = mean_val
            svd = TruncatedSVD(n_components=k, random_state=self.cfg.random_state)
            U = np.abs(svd.fit_transform(filled))
            S = svd.singular_values_
            Vt = np.abs(svd.components_)
            W = np.clip(U, 1e-10, None)
            H = np.clip((Vt.T * S).T, 1e-10, None)
            return W, H
        W = np.abs(rng.normal(0, 0.01, size=(R.shape[0], k))) + 0.1
        H = np.abs(rng.normal(0, 0.01, size=(k, R.shape[1]))) + 0.1
        return W, H

    def fit(self, R, mask=None):
        A = R.toarray().astype(float) if issparse(R) else np.asarray(R, dtype=float)
        G = (A > 0).astype(float) if mask is None else np.asarray(mask, dtype=float)
        k = int(self.cfg.n_components)
        W, H = self._init_factors(A, k)
        alpha, beta = self.cfg.alpha, self.cfg.beta
        prev = np.inf
        for it in range(self.cfg.max_iter):
            WH = W @ H
            num_h = W.T @ (G * A)
            den_h = W.T @ (G * WH) + beta * H + 1e-10
            H *= num_h / den_h
            H = np.clip(H, 1e-10, None)

            WH = W @ H
            num_w = (G * A) @ H.T
            den_w = (G * WH) @ H.T + alpha * W + 1e-10
            W *= num_w / den_w
            W = np.clip(W, 1e-10, None)

            if it % 10 == 0:
                WH = W @ H
                loss = np.sum((G * (A - WH)) ** 2) + alpha * np.sum(W**2) + beta * np.sum(H**2)
                if abs(prev - loss) < self.cfg.tol:
                    break
                prev = loss
        self.W_, self.H_ = W, H
        return self

    def predict(self, u: int, i: int) -> float:
        return float(self.W_[u] @ self.H_[:, i])

    def predict_all(self) -> np.ndarray:
        return self.W_ @ self.H_

    def get_user_factors(self) -> np.ndarray:
        return self.W_

    def get_item_factors(self) -> np.ndarray:
        return self.H_.T
