"""
Dataset-specific loaders for MovieLens and FilmTrust.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from core.utils import load_ratings, create_train_test_split


def load_ratings_100k(base_path: str, test_path: str, fold: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if fold != 1:
        data_dir = os.path.dirname(os.path.abspath(base_path))
        base_path = os.path.join(data_dir, f"u{fold}.base")
        test_path = os.path.join(data_dir, f"u{fold}.test")

    def _read(path: str) -> np.ndarray:
        df = pd.read_csv(
            path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            usecols=["user_id", "item_id", "rating"],
        )
        df["user_id"] -= 1
        df["item_id"] -= 1
        return df[["user_id", "item_id", "rating"]].values.astype(np.float32)

    return _read(base_path), _read(test_path)


def load_ratings_1m(
    ratings_path: str,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    fold: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if fold is not None and not (1 <= fold <= 5):
        raise ValueError(f"load_ratings_1m: fold None veya 1..5 olmalı, gelen: {fold}")
    rows = []
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) >= 3:
                rows.append((int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
    if fold is None or fold == 1:
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=random_seed, shuffle=True
        )
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        train_idx, test_idx = list(kf.split(df))[fold - 1]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
    combo = pd.concat([train_df, test_df], axis=0)
    unique_items = np.sort(combo["item_id"].unique())
    i_map = {int(it): j for j, it in enumerate(unique_items)}
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["item_id"] = train_df["item_id"].map(i_map)
    test_df["item_id"] = test_df["item_id"].map(i_map)
    return (
        train_df[["user_id", "item_id", "rating"]].values.astype(np.float32),
        test_df[["user_id", "item_id", "rating"]].values.astype(np.float32),
    )


@dataclass
class BaseLoader:
    def load(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class ML100KLoader(BaseLoader):
    base_path: str
    test_path: str
    fold: int = 1

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return load_ratings_100k(self.base_path, self.test_path, fold=self.fold)


@dataclass
class ML1MLoader(BaseLoader):
    ratings_path: str
    test_ratio: float = 0.2
    random_seed: int = 42
    fold: int | None = None

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return load_ratings_1m(
            self.ratings_path,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
            fold=self.fold,
        )


@dataclass
class FilmTrustLoader(BaseLoader):
    ratings_path: str
    test_ratio: float = 0.2
    random_seed: int = 42

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        ratings = load_ratings(self.ratings_path)
        return create_train_test_split(
            ratings, test_ratio=self.test_ratio, random_seed=self.random_seed
        )


def get_loader(dataset: str, data_root: str = "data", fold: int | None = None) -> BaseLoader:
    ds = dataset.strip().lower()
    if ds == "100k":
        base_path = os.path.join(data_root, "ml-100k", "u1.base")
        test_path = os.path.join(data_root, "ml-100k", "u1.test")
        return ML100KLoader(base_path=base_path, test_path=test_path, fold=fold or 1)
    if ds == "1m":
        ratings_path = os.path.join(data_root, "ml-1m", "ratings.dat")
        return ML1MLoader(ratings_path=ratings_path, fold=fold)
    if ds == "filmtrust":
        ratings_path = os.path.join(data_root, "filmtrust", "ratings.txt")
        return FilmTrustLoader(ratings_path=ratings_path)
    raise ValueError(f"Unsupported dataset: {dataset}")

