"""
Thakrar Algoritma 2 — tek konfigürasyon uçtan uca pipeline (sklearn'siz).

    MF (Alg.4)  →  centroid init  →  kendi K-Means (Alg.5)  →  küme-ort. (Alg.6)
"""

from __future__ import annotations

import time

import numpy as np

from . import custom_kmeans, matrix_factorization, predict


_META_MODES = ("avoa", "hho", "cso")


def run_config(
    train: np.ndarray,
    test: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    k: int = 14,
    latent_dim: int = 10,
    mf_epochs: int = 50,
    lr: float = 0.01,
    reg: float = 0.01,
    init_mode: str = "kmeans++",
    meta_epoch: int = 100,
    meta_pop: int = 30,
    kmeans_max_iter: int = 300,
    clip: tuple[float, float] | None = (1.0, 5.0),
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    init_mode:
        'kmeans++' | 'random'  → custom_kmeans içi başlatma
        'avoa' | 'hho' | 'cso' → meta-sezgisel WCSS centroidleri (Alg.7 yerine),
                                  ardından kendi K-Means ile rafine (Alg.5).
    meta_epoch / meta_pop: meta-sezgisel iterasyon ve popülasyon (init modları için).
    """
    t0 = time.time()

    # --- Adım 1: Matrix Factorization (Alg.4) — kümeleme özniteliği P ---
    P, _Q = matrix_factorization.matrix_factorization(
        train, n_users, n_items,
        latent_dim=latent_dim, n_epochs=mf_epochs, lr=lr, reg=reg,
        seed=seed, verbose=verbose,
    )

    # --- Adım 2: centroid başlatma + kendi K-Means (Alg.5) ---
    init_centroids = None
    km_init = init_mode
    if init_mode in _META_MODES:
        from . import meta_init
        init_centroids = meta_init.metaheuristic_init(
            P, k, algo=init_mode, epoch=meta_epoch, pop_size=meta_pop, seed=seed,
        )
        km_init = "kmeans++"  # init_centroids verildiğinde yok sayılır

    labels, _centroids, inertia, n_iter = custom_kmeans.kmeans(
        P, k, init=km_init, init_centroids=init_centroids,
        max_iter=kmeans_max_iter, seed=seed,
    )

    # --- Adım 3: küme-ortalaması tahmini (Alg.6) ---
    metrics = predict.cluster_average_predict(train, test, labels, clip=clip)

    # küme boyut istatistikleri
    sizes = np.bincount(labels, minlength=k)
    return {
        "k": k,
        "latent_dim": latent_dim,
        "mf_epochs": mf_epochs,
        "lr": lr,
        "reg": reg,
        "init_mode": init_mode,
        "wcss": float(inertia),
        "kmeans_iters": int(n_iter),
        "cluster_min": int(sizes.min()),
        "cluster_max": int(sizes.max()),
        "n_empty_clusters": int(np.sum(sizes == 0)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "cluster_mean_pct": metrics["cluster_mean_pct"],
        "global_fallback_pct": metrics["global_fallback_pct"],
        "seconds": time.time() - t0,
    }
