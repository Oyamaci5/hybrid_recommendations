from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from models.base import MatrixFactorizationBase
from utils.config import ModelConfig


class PMFModel(MatrixFactorizationBase):
    """
    MAP tabanlı PMF ile uyumlu, SGD ile çözülen probit olmayan regüler latent model.
    ``sigma_*`` parametreleri L2 gücünü yaklaşık kontrol eder.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.mu_: float = 0.0
        self.U_: np.ndarray | None = None
        self.V_: np.ndarray | None = None

    def fit(self, R, mask=None) -> "PMFModel":
        rng = np.random.default_rng(int(self.cfg.random_state))
        A = R.toarray().astype(np.float64) if issparse(R) else np.asarray(R, dtype=np.float64)
        G = (A != 0) if mask is None else np.asarray(mask, dtype=bool)
        self.mu_ = float(A[G].mean()) if np.any(G) else 3.0

        nu, nv = float(self.cfg.sigma_U_sq), float(self.cfg.sigma_V_sq)
        nz = float(self.cfg.sigma_sq)
        lam_u = 1.0 / max(nu, 1e-6)
        lam_v = 1.0 / max(nv, 1e-6)
        noise_prec = 1.0 / max(nz, 1e-6)

        n_users, n_items = A.shape
        k = int(self.cfg.n_components)
        lr_base = float(self.cfg.lr)

        sd = np.sqrt(min(nu, nv))
        self.U_ = rng.normal(0, float(sd / np.sqrt(max(k, 1))), size=(n_users, k))
        self.V_ = rng.normal(0, float(sd / np.sqrt(max(k, 1))), size=(n_items, k))

        idx = list(zip(*np.nonzero(G)))
        rng.shuffle(idx)
        for ep in range(int(self.cfg.n_iter)):
            rng.shuffle(idx)
            for u, i in idx:
                r_ui = float(A[u, i])
                pred = float(self.mu_ + self.U_[u] @ self.V_[i])
                e = (r_ui - pred) * noise_prec
                gu = lr_base * (e * self.V_[i] - lam_u * self.U_[u])
                gv = lr_base * (e * self.U_[u] - lam_v * self.V_[i])
                self.U_[u] += gu
                self.V_[i] += gv
        return self

    def predict(self, u: int, i: int) -> float:
        return float(
            np.clip(self.mu_ + float(self.U_[u] @ self.V_[i]), 1.0, 5.0)
        )

    def predict_all(self) -> np.ndarray:
        return np.clip(self.mu_ + self.U_ @ self.V_.T, 1.0, 5.0).astype(np.float32)

    def get_user_factors(self) -> np.ndarray:
        return self.U_

    def get_item_factors(self) -> np.ndarray:
        return self.V_
