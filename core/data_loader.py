"""
core/data_loader.py
-------------------
Clustering-CF pipeline için veri yükleme ve ön işleme.

core/utils.py'deki genel fonksiyonları wrap eder ve rating matrix
pipeline'ını tek bir çağrıyla sunar.

Kullanım::

    from core.data_loader import load_dataset

    train_matrix, test_ratings, info = load_dataset(
        "data/ml-100k/u.data", test_ratio=0.2, seed=42
    )
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from core.utils import (
    load_ratings,
    create_train_test_split,
    create_rating_matrix,
    get_data_info,
)
from core.loaders import BaseLoader, ML100KLoader, ML1MLoader, FilmTrustLoader, get_loader


def load_dataset(
    ratings_path: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    normalize: Literal["none", "user-mean"] = "none",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Veriyi yükler, train/test böler ve rating matrisini oluşturur.

    Parameters
    ----------
    ratings_path : str
        u.data (100K) veya ratings.dat (1M) dosyasının yolu.
    test_ratio : float
        Test oranı (varsayılan 0.20 → %80 train / %20 test).
    seed : int
        Tekrarlanabilirlik tohumu.

    Returns
    -------
    train_matrix : np.ndarray, shape (n_users, n_items)
        Ham train rating matrisi — test pozisyonları 0, eksik değerler 0.
    test_ratings : np.ndarray, shape (n_test, 3)
        Test gözlemleri — sütunlar: [user_id, item_id, rating] (0-indexed).
    info : dict
        n_users, n_items, n_train, n_test, sparsity, normalize ve
        clustering pipeline'ı için yardımcı alanlar.
    """
    all_ratings = load_ratings(ratings_path)
    train_ratings, test_ratings = create_train_test_split(
        all_ratings, test_ratio=test_ratio, random_seed=seed
    )

    n_users, n_items, _ = get_data_info(all_ratings)
    train_matrix = create_rating_matrix(train_ratings, n_users, n_items)

    if normalize not in {"none", "user-mean"}:
        raise ValueError("normalize must be one of: 'none', 'user-mean'")

    observed_mask = train_matrix != 0
    user_means = np.zeros(n_users, dtype=np.float32)
    rated_counts = observed_mask.sum(axis=1)
    valid_users = rated_counts > 0
    if np.any(valid_users):
        user_means[valid_users] = (
            train_matrix[valid_users].sum(axis=1) / rated_counts[valid_users]
        ).astype(np.float32)

    if normalize == "user-mean":
        clustering_matrix = np.where(
            observed_mask, train_matrix - user_means[:, None], 0.0
        ).astype(np.float32)
    else:
        clustering_matrix = train_matrix.copy()

    sparsity = 1.0 - float(observed_mask.sum()) / (n_users * n_items)
    info = {
        "n_users":  n_users,
        "n_items":  n_items,
        "n_train":  len(train_ratings),
        "n_test":   len(test_ratings),
        "sparsity": round(sparsity, 4),
        "normalize": normalize,
        "clustering_matrix": clustering_matrix,
        "user_means": user_means,
        "observed_mask": observed_mask,
    }

    return train_matrix, test_ratings, info


