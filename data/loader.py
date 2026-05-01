from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import DataConfig


class DatasetLoader:
    """Loads dataset and returns dense rating matrix + id maps."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg

    def _resolve_ml100k_file(self) -> Path:
        base = Path(self.cfg.path)
        candidates = [
            base / "ml-100k" / "u.data",
            base / "ml100k" / "u.data",
            base / "u.data",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"ml100k ratings file not found under: {base}")

    def _resolve_ml1m_file(self) -> Path:
        base = Path(self.cfg.path)
        candidates = [
            base / "ml-1m" / "ratings.dat",
            base / "ml1m" / "ratings.dat",
            base / "ratings.dat",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"ml1m ratings file not found under: {base}")

    @staticmethod
    def _to_dense(rows: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
        users = np.unique(rows[:, 0].astype(np.int64))
        items = np.unique(rows[:, 1].astype(np.int64))
        user_map = {int(uid): idx for idx, uid in enumerate(users.tolist())}
        item_map = {int(iid): idx for idx, iid in enumerate(items.tolist())}
        R = np.zeros((len(users), len(items)), dtype=np.float64)
        for uid, iid, rating in rows:
            R[user_map[int(uid)], item_map[int(iid)]] = float(rating)
        return R, user_map, item_map

    def _load_ml1m(self) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
        path = self._resolve_ml1m_file()
        df = pd.read_csv(
            path,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            usecols=["user_id", "item_id", "rating"],
            engine="python",
        )
        rows = df[["user_id", "item_id", "rating"]].to_numpy(dtype=np.float64)
        rows[:, 0] -= 1.0
        rows[:, 1] -= 1.0
        return self._to_dense(rows)

    def load(self) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
        name = self.cfg.dataset.strip().lower()
        if name == "ml100k":
            path = self._resolve_ml100k_file()
            df = pd.read_csv(
                path,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
                usecols=["user_id", "item_id", "rating"],
            )
            rows = df[["user_id", "item_id", "rating"]].to_numpy(dtype=np.float64)
            rows[:, 0] -= 1.0
            rows[:, 1] -= 1.0
            return self._to_dense(rows)

        if name == "ml1m":
            return self._load_ml1m()

        raise NotImplementedError(
            f"DatasetLoader supports: ml100k, ml1m — got: {name!r}"
        )
