from __future__ import annotations

import numpy as np

from utils.config import GraySheepConfig


class GraySheepDetector:
    """Simple detector based on user mean/std profile z-scores."""

    def __init__(self, cfg: GraySheepConfig) -> None:
        self.cfg = cfg
        self._gray_mask: np.ndarray | None = None

    def fit(self, R_train: np.ndarray) -> None:
        R = np.asarray(R_train, dtype=np.float64)
        n_users = R.shape[0]
        user_mean = np.zeros(n_users, dtype=np.float64)
        user_std = np.zeros(n_users, dtype=np.float64)
        sparsity = np.zeros(n_users, dtype=np.float64)

        for u in range(n_users):
            rated = R[u, R[u] > 0]
            if rated.size == 0:
                user_mean[u] = 0.0
                user_std[u] = 0.0
                sparsity[u] = 1.0
                continue
            user_mean[u] = float(rated.mean())
            user_std[u] = float(rated.std())
            sparsity[u] = 1.0 - (float(rated.size) / float(R.shape[1]))

        def _z(x: np.ndarray) -> np.ndarray:
            sx = float(x.std())
            if sx <= 1e-12:
                return np.zeros_like(x)
            return (x - float(x.mean())) / sx

        # pcc_std-ish profile anomaly score.
        score = np.abs(_z(user_mean)) + np.abs(_z(user_std)) + np.abs(_z(sparsity))
        thr = float(self.cfg.threshold)
        self._gray_mask = score >= thr

    def get_mask(self) -> np.ndarray:
        if self._gray_mask is None:
            raise RuntimeError("GraySheepDetector.fit must be called before get_mask().")
        return self._gray_mask
