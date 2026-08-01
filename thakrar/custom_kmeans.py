"""
Custom K-Means (Lloyd) — saf NumPy, sklearn'siz.

Thakrar et al. (2025 BigComp) Algoritma 5'in birebir karşılığı:
  - Atama adımı: her noktayı en yakın centroid'e ata (öklid mesafe).
  - Güncelleme adımı: her centroid'i kümesindeki noktaların ortalamasına taşı.
  - Yakınsama veya max_iter'a kadar tekrarla.

Ek olarak boş küme onarımı yapılır (sklearn bunu otomatik yapar; biz elle).
Centroid başlatma dışarıdan verilebilir (CSO/HHO meta-sezgisel çıktısı için).
"""

from __future__ import annotations

import numpy as np


def _pairwise_sq_dist(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(n, k) karesel öklid mesafe matrisi: ||x||^2 + ||c||^2 - 2 x·c."""
    xx = np.sum(X * X, axis=1, keepdims=True)        # (n, 1)
    cc = np.sum(centroids * centroids, axis=1)        # (k,)
    cross = X @ centroids.T                           # (n, k)
    d2 = xx + cc[None, :] - 2.0 * cross
    np.maximum(d2, 0.0, out=d2)                       # negatif yuvarlama hatalarını kırp
    return d2


def random_init(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """k adet rastgele veri noktasını başlangıç centroid'i olarak seç."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].astype(np.float64).copy()


def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """k-means++ başlatma (öklid). Yayılı, çeşitli centroidler verir."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    first = int(rng.integers(n))
    centroids = [X[first].astype(np.float64)]
    closest_sq = _pairwise_sq_dist(X, X[first][None, :].astype(np.float64))[:, 0]
    for _ in range(1, k):
        total = closest_sq.sum()
        if total <= 0:  # tüm noktalar mevcut centroidlerle çakışık
            idx = int(rng.integers(n))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n, p=probs))
        centroids.append(X[idx].astype(np.float64))
        new_sq = _pairwise_sq_dist(X, X[idx][None, :].astype(np.float64))[:, 0]
        closest_sq = np.minimum(closest_sq, new_sq)
    return np.asarray(centroids, dtype=np.float64)


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    init: str = "kmeans++",
    init_centroids: np.ndarray | None = None,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Lloyd K-Means.

    Parametreler
    ------------
    X              : (n, d) öznitelik matrisi (örn. MF kullanıcı-latent P).
    k              : küme sayısı.
    init           : 'kmeans++' veya 'random' (init_centroids verilmezse).
    init_centroids : (k, d) dışarıdan başlangıç centroidleri (CSO/HHO çıktısı).
                     Verilirse 'init' yok sayılır.
    max_iter       : maksimum iterasyon.
    tol            : centroid kayması bu eşiğin altına inince dur.
    seed           : tekrarlanabilirlik.

    Döndürür
    --------
    labels    : (n,) int32 küme atamaları
    centroids : (k, d) float64 final centroidler
    inertia   : WCSS (küme-içi karesel mesafe toplamı)
    n_iter    : çalışan iterasyon sayısı
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    if init_centroids is not None:
        centroids = np.asarray(init_centroids, dtype=np.float64).reshape(k, d).copy()
    elif init == "kmeans++":
        centroids = kmeans_plus_plus_init(X, k, seed)
    elif init == "random":
        centroids = random_init(X, k, seed)
    else:
        raise ValueError(f"Bilinmeyen init: {init!r}")

    labels = np.zeros(n, dtype=np.int32)
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        d2 = _pairwise_sq_dist(X, centroids)
        labels = np.argmin(d2, axis=1).astype(np.int32)

        new_centroids = centroids.copy()
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = X[mask].mean(axis=0)
            else:
                # Boş küme onarımı: mevcut centroid'ine en uzak noktayı yeni
                # centroid yap ve o noktayı bu kümeye ata.
                far = int(np.argmax(np.min(d2, axis=1)))
                new_centroids[c] = X[far]
                labels[far] = c

        shift = float(np.sqrt(np.sum((new_centroids - centroids) ** 2)))
        centroids = new_centroids
        if shift <= tol:
            break

    # Final atama ve inertia (WCSS)
    d2 = _pairwise_sq_dist(X, centroids)
    labels = np.argmin(d2, axis=1).astype(np.int32)
    inertia = float(d2[np.arange(n), labels].sum())
    return labels, centroids, inertia, n_iter
