"""
ML-100K resmi 5-fold yükleyici — sklearn'siz, saf pandas/numpy.

Resmi GroupLens bölmesi: u{fold}.base / u{fold}.test  (fold = 1..5).
Dosya formatı: user_id  item_id  rating  timestamp  (tab ile ayrılmış).
Döndürülen array'lerde user_id ve item_id 0-indexed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ML100K_DIR = REPO / "data" / "ml-100k"

# ML-100K sabit boyutları (u.info ile uyumlu)
N_USERS = 943
N_ITEMS = 1682


def _read_ml100k(path: Path) -> np.ndarray:
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        usecols=["user_id", "item_id", "rating"],
    )
    df["user_id"] -= 1  # 1-indexed -> 0-indexed
    df["item_id"] -= 1
    return df[["user_id", "item_id", "rating"]].values.astype(np.float32)


def load_official_fold(fold: int, data_dir: Path | str = ML100K_DIR):
    """
    Resmi fold'u yükle.

    Döndürür
    --------
    train, test : her biri (N, 3) [user_id, item_id, rating] (0-indexed)
    n_users, n_items : int
    """
    if not (1 <= fold <= 5):
        raise ValueError(f"fold 1..5 olmalı, gelen: {fold}")
    data_dir = Path(data_dir)
    train = _read_ml100k(data_dir / f"u{fold}.base")
    test = _read_ml100k(data_dir / f"u{fold}.test")
    return train, test, N_USERS, N_ITEMS
