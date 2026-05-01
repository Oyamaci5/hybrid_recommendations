from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import issparse

from utils.config import PreprocessConfig


@dataclass
class BiasState:
    global_mean: float
    user_bias: np.ndarray
    item_bias: np.ndarray


class Preprocessor:
    """Dense preprocessing + reversible per-rating transform."""

    def __init__(self, cfg: PreprocessConfig | None = None) -> None:
        self.cfg = cfg or PreprocessConfig()
        self._norm_kind = "none"
        self._norm_global_mean = 0.0
        self._norm_global_std = 1.0
        self._norm_user_means: np.ndarray | None = None
        self._norm_user_stds: np.ndarray | None = None

    @staticmethod
    def _to_dense(R: Any) -> np.ndarray:
        return R.toarray().astype(np.float64) if issparse(R) else np.asarray(R, dtype=np.float64)

    @staticmethod
    def _safe_mean(arr: np.ndarray) -> float:
        return float(arr.mean()) if arr.size else 0.0

    def _normalize(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kind = (self.cfg.normalization or "none").strip().lower()
        self._norm_kind = kind
        Y = X.copy()

        if kind == "none":
            return Y
        if kind == "global_mean":
            g = self._safe_mean(X[mask])
            self._norm_global_mean = g
            Y[mask] = X[mask] - g
            return Y
        if kind in ("zscore", "z_score", "z-score"):
            g = self._safe_mean(X[mask])
            s = float(np.std(X[mask])) if np.any(mask) else 1.0
            if s <= 1e-12:
                s = 1.0
            self._norm_kind = "zscore"
            self._norm_global_mean = g
            self._norm_global_std = s
            Y[mask] = (X[mask] - g) / s
            return Y
        if kind == "user_mean":
            means = np.zeros(X.shape[0], dtype=np.float64)
            for u in range(X.shape[0]):
                mm = mask[u]
                means[u] = self._safe_mean(X[u, mm])
                if np.any(mm):
                    Y[u, mm] = X[u, mm] - means[u]
            self._norm_user_means = means
            return Y
        if kind in ("user_zscore", "user_z_score", "user-zscore", "user-z-score"):
            means = np.zeros(X.shape[0], dtype=np.float64)
            stds = np.ones(X.shape[0], dtype=np.float64)
            for u in range(X.shape[0]):
                mm = mask[u]
                if np.any(mm):
                    means[u] = self._safe_mean(X[u, mm])
                    std = float(np.std(X[u, mm]))
                    if std > 1e-12:
                        stds[u] = std
                    Y[u, mm] = (X[u, mm] - means[u]) / stds[u]
            self._norm_kind = "user_zscore"
            self._norm_user_means = means
            self._norm_user_stds = stds
            return Y

        # Backward-compatible aliases from previous code.
        if kind == "user_mean_center":
            means = np.zeros(X.shape[0], dtype=np.float64)
            for u in range(X.shape[0]):
                mm = mask[u]
                means[u] = self._safe_mean(X[u, mm])
                if np.any(mm):
                    Y[u, mm] = X[u, mm] - means[u]
            self._norm_kind = "user_mean"
            self._norm_user_means = means
            return Y

        raise ValueError(f"Unknown normalization: {kind}")

    def _remove_bias(self, X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, BiasState]:
        if not self.cfg.apply_bias_removal or not np.any(mask):
            return X, BiasState(global_mean=0.0, user_bias=np.zeros(X.shape[0]), item_bias=np.zeros(X.shape[1]))

        g = self._safe_mean(X[mask])
        u_bias = np.zeros(X.shape[0], dtype=np.float64)
        i_bias = np.zeros(X.shape[1], dtype=np.float64)

        for u in range(X.shape[0]):
            mm = mask[u]
            if np.any(mm):
                u_bias[u] = self._safe_mean(X[u, mm]) - g
        for i in range(X.shape[1]):
            mm = mask[:, i]
            if np.any(mm):
                i_bias[i] = self._safe_mean(X[mm, i]) - g

        Y = X.copy()
        rows, cols = np.where(mask)
        Y[rows, cols] = X[rows, cols] - (g + u_bias[rows] + i_bias[cols])
        return Y, BiasState(global_mean=g, user_bias=u_bias, item_bias=i_bias)

    def fit_transform(self, R: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        X = self._to_dense(R)
        mask = X > 0

        Xn = self._normalize(X, mask)
        Xb, bias_state = self._remove_bias(Xn, mask)
        return Xb, mask.astype(np.float64), {"bias_state": bias_state}

    def inverse_transform_single(
        self,
        value: float,
        bias: dict[str, Any] | None,
        user_id: int,
        item_id: int,
    ) -> float:
        out = float(value)
        u = int(user_id)
        i = int(item_id)

        bstate = (bias or {}).get("bias_state")
        if isinstance(bstate, BiasState):
            out += float(bstate.global_mean + bstate.user_bias[u] + bstate.item_bias[i])

        if self._norm_kind == "global_mean":
            out += float(self._norm_global_mean)
        elif self._norm_kind == "zscore":
            out = out * float(self._norm_global_std) + float(self._norm_global_mean)
        elif self._norm_kind == "user_mean" and self._norm_user_means is not None:
            out += float(self._norm_user_means[u])
        elif (
            self._norm_kind == "user_zscore"
            and self._norm_user_means is not None
            and self._norm_user_stds is not None
        ):
            out = out * float(self._norm_user_stds[u]) + float(self._norm_user_means[u])
        return out
