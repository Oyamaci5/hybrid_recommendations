"""
Küme-ortalaması tahmini (Thakrar et al. 2025, Algoritma 6) — saf NumPy.

Bir (u, i) çifti için tahmin:
  1. Kullanıcı u'nun kümesi c'yi bul.
  2. c kümesindeki kullanıcıların item i'ye verdiği oyların ortalamasını al.
  3. Hiç oy yoksa global ortalamaya düş (fallback).

NOT: Aynı kümedeki tüm kullanıcılar belirli bir item için AYNI tahmini alır
(item-küme ortalaması). Kişiselleştirme küme çözünürlüğündedir.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def cluster_average_predict(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    *,
    clip: tuple[float, float] | None = (1.0, 5.0),
) -> dict:
    """
    Parametreler
    ------------
    train       : (N, 3) [user_id, item_id, rating] (0-indexed).
    test        : (M, 3) aynı format.
    assignments : (n_users,) her kullanıcının küme etiketi.
    clip        : tahminleri bu aralığa kırp (None = kırpma).

    Döndürür
    --------
    dict: mae, rmse, n_test, cluster_mean_pct, global_fallback_pct, global_mean
    """
    global_mean = float(train[:, 2].mean())

    # (küme, item) -> toplam ve sayı  (hızlı ortalama için ön-toplama)
    cl_train = assignments[train[:, 0].astype(np.int64)]
    ci_sum: dict[tuple[int, int], float] = defaultdict(float)
    ci_cnt: dict[tuple[int, int], int] = defaultdict(int)
    for c, i, r in zip(cl_train, train[:, 1].astype(np.int64), train[:, 2]):
        key = (int(c), int(i))
        ci_sum[key] += float(r)
        ci_cnt[key] += 1

    preds = np.empty(len(test), dtype=np.float64)
    truths = test[:, 2].astype(np.float64)
    n_cluster_hit = 0

    for row, (u, i, _r) in enumerate(test):
        c = int(assignments[int(u)])
        key = (c, int(i))
        cnt = ci_cnt.get(key, 0)
        if cnt > 0:
            pred = ci_sum[key] / cnt
            n_cluster_hit += 1
        else:
            pred = global_mean
        if clip is not None:
            pred = min(max(pred, clip[0]), clip[1])
        preds[row] = pred

    err = truths - preds
    n_test = len(test)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n_test": n_test,
        "cluster_mean_pct": 100.0 * n_cluster_hit / max(n_test, 1),
        "global_fallback_pct": 100.0 * (n_test - n_cluster_hit) / max(n_test, 1),
        "global_mean": global_mean,
    }
