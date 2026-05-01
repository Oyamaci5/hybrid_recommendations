"""PCA tabanli boyut indirgeme; sklearn wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from utils.config import PCAConfig


class PCAReducer:
    """
    Ham rating matrisini PCA ile dusuk boyutlu uzaya indirger.

    Kullanim:
        reducer = PCAReducer(cfg.pca)
        reducer.fit(R_train)
        X = reducer.transform(R_train)  # (n_users, n_components)
    """

    def __init__(self, cfg: PCAConfig):
        self.cfg = cfg
        self._pca = PCA(
            n_components=cfg.n_components,
            whiten=cfg.whiten,
            random_state=cfg.random_state,
        )
        self.is_fitted_ = False

    def fit(self, R: np.ndarray) -> "PCAReducer":
        """
        R: (n_users, n_items) dense numpy array.
        Missing degerler (0) oldugu gibi kalir; zero_fill + zscore
        sonrasi PCA akisiyla uyumludur.
        """
        self._pca.fit(R)
        self.is_fitted_ = True
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.n_components_actual_ = self._pca.n_components_
        return self

    def transform(self, R: np.ndarray) -> np.ndarray:
        """(n_users, n_items) -> (n_users, n_components)"""
        return self._pca.transform(R)

    def fit_transform(self, R: np.ndarray) -> np.ndarray:
        self.fit(R)
        return self.transform(R)

    def explained_variance_summary(self) -> dict:
        """Kac bilesenin ne kadar varyansi acikladigini dondur."""
        cumsum = np.cumsum(self.explained_variance_ratio_)
        return {
            "n_components": self.n_components_actual_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_variance": cumsum.tolist(),
            "total_explained": float(cumsum[-1]),
        }
