"""Yol 4: MF hiperparametre araması için doğruluk problemi."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import issparse

from models.base import MatrixFactorizationBase
from utils.config import Config, ModelConfig


def _build_model(kind: str, mc: ModelConfig) -> MatrixFactorizationBase:
    k = kind.lower().strip()
    if k == "wnmf":
        from models.wnmf import WNMF

        return WNMF(mc)
    if k == "svd":
        from models.svd import SVDModel

        return SVDModel(mc)
    if k == "pmf":
        from models.pmf import PMFModel

        return PMFModel(mc)
    raise ValueError(f"Bilinmeyen model tipi: {kind}")


@dataclass
class MFTuningProblem:
    """
    Gözlümler içinden doğrulanan küçük bir dilim seçilir (MAE doğrusu).
    Eğitim matrisinde bu dizinler 0’a çekilir; model yalnızca kalan puanları görür.
    """

    train_matrix: np.ndarray
    full_cfg: Config
    val_fraction: float = 0.15
    random_state: int = 42
    _vu: np.ndarray = field(init=False, default=np.array([]))
    _vi: np.ndarray = field(init=False, default=np.array([]))
    _R_tr: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        R = np.asarray(self.train_matrix, dtype=np.float64)
        if issparse(R):
            raise TypeError("MFTuningProblem yoğun (dense) matris bekliyor.")
        nz = np.argwhere(R > 0)
        if len(nz) == 0:
            self._R_tr = np.zeros_like(R)
            return
        rng = np.random.default_rng(int(self.random_state))
        perm = rng.permutation(len(nz))
        vn = len(nz)
        min_train_keep = max(16, vn // min(100, vn))
        cand = max(16, min(int(float(self.val_fraction) * vn), vn - min_train_keep))
        nv = int(np.clip(cand, 1, vn - 1))
        take = nz[perm[:nv]]
        self._vu = take[:, 0].astype(np.int64)
        self._vi = take[:, 1].astype(np.int64)

        Rt = R.copy()
        Rt[self._vu, self._vi] = 0.0
        self._R_tr = Rt

    def _decode_vec(self, x: np.ndarray) -> dict[str, float]:
        x = np.clip(np.asarray(x, dtype=float).ravel(), 0.0, 1.0)
        pb = self.full_cfg.mf_tuning.param_bounds
        keys = sorted(pb.keys())
        nd = len(keys)
        arr = np.resize(x, nd) if nd else x
        out: dict[str, float] = {}
        for i, key in enumerate(keys):
            lo, hi = float(pb[key][0]), float(pb[key][1])
            out[key] = lo + arr[i] * max(hi - lo, 1e-12)
        return out

    def decode_to_model_cfg(self, x: np.ndarray) -> ModelConfig:
        d = self._decode_vec(x)
        base = ModelConfig(**{**self.full_cfg.model.__dict__})
        if "n_components" in d:
            base.n_components = max(1, int(round(d["n_components"])))
        for att in ("alpha", "beta", "lr", "reg", "sigma_sq", "sigma_U_sq", "sigma_V_sq"):
            if att in d:
                setattr(base, att, float(d[att]))
        return base

    def fitness(self, x: np.ndarray) -> float:
        R = np.asarray(self.train_matrix, dtype=np.float64)
        if self._vu.size == 0:
            return 10.0
        mc = self.decode_to_model_cfg(x)
        mod = _build_model(self.full_cfg.model.name, mc)
        mod.fit(self._R_tr, mask=None)
        errs = [
            abs(float(mod.predict(int(u_g), int(i_g))) - float(R[u_g, i_g]))
            for u_g, i_g in zip(self._vu.tolist(), self._vi.tolist())
        ]
        return float(np.mean(errs))


def boxed_bounds(dim: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(dim), np.ones(dim)
