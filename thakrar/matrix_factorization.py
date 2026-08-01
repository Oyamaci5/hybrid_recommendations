"""
Matrix Factorization (Thakrar et al. 2025, Algoritma 4) — saf NumPy SGD.

R (kullanıcı-item) matrisini P (n×L) ve Q (m×L) latent matrislerine ayırır.
Yalnızca GÖZLEMLENEN oylar üzerinden gradient descent yapılır.

  tahmin:  r̂_ui = P[u] · Q[i]
  hata:    e_ui  = r_ui - r̂_ui
  güncelle (standart düzenlileştirme işaretiyle):
      P[u] += lr · (e_ui · Q[i] - reg · P[u])
      Q[i] += lr · (e_ui · P[u] - reg · Q[i])

NOT: Makalede MF SADECE kümeleme özniteliği (P) üretmek için kullanılır.
Tahmin, MF yeniden-yapılandırması (P·Q) ile DEĞİL, küme-ortalaması ile yapılır
(bkz. predict.cluster_average_predict). Bu yüzden Q tahminde kullanılmaz; yine
de eksiksizlik ve olası alternatifler için döndürülür.
"""

from __future__ import annotations

import numpy as np


def matrix_factorization(
    train: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    latent_dim: int = 10,
    n_epochs: int = 50,
    lr: float = 0.01,
    reg: float = 0.01,
    seed: int = 42,
    shuffle: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parametreler
    ------------
    train      : (N, 3) array, sütunlar [user_id, item_id, rating] (0-indexed).
    n_users    : kullanıcı sayısı (P satır sayısı).
    n_items    : item sayısı (Q satır sayısı).
    latent_dim : L, latent faktör sayısı.
    n_epochs   : T, iterasyon sayısı.
    lr         : öğrenme oranı (α).
    reg        : düzenlileştirme (λ).
    shuffle    : her epoch'ta örnek sırasını karıştır (standart SGD).

    Döndürür
    --------
    P : (n_users, latent_dim) kullanıcı-latent matrisi  → kümeleme özniteliği
    Q : (n_items, latent_dim) item-latent matrisi
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(0.0, 0.1, (n_users, latent_dim)).astype(np.float64)
    Q = rng.normal(0.0, 0.1, (n_items, latent_dim)).astype(np.float64)

    u_idx = train[:, 0].astype(np.int64)
    i_idx = train[:, 1].astype(np.int64)
    ratings = train[:, 2].astype(np.float64)
    order = np.arange(len(train))

    for epoch in range(n_epochs):
        if shuffle:
            rng.shuffle(order)
        sse = 0.0
        for idx in order:
            u = u_idx[idx]
            i = i_idx[idx]
            err = ratings[idx] - float(P[u] @ Q[i])
            pu = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * pu - reg * Q[i])
            sse += err * err
        if verbose:
            rmse = np.sqrt(sse / len(train))
            print(f"  [MF] epoch {epoch + 1:3d}/{n_epochs}  train_rmse={rmse:.4f}")

    return P.astype(np.float32), Q.astype(np.float32)
