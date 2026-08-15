"""
KLASIK KUMELEME BASELINE'LARI — HHO-K-means makalesinin kiyas seti.

Eklenenler: k-means (random init, tek kosu = literatur hali), KMeans++ (n_init=10),
PCA-k-means, SOM-Cluster, PCA-SOM  +  AVOA (bizim yontem).
Hepsi AYNI tahmin hattinda (repair kapasite, soft top-2, kume-MF + kNN karisimi)
degerlendirilir -> fark yalnizca kumeleme yonteminden gelir.

SOM: minimal batch-SOM (dikdortgen izgara, gaussian komsuluk, lineer sogutma).
Kullanim: python algo_selection_v2/klasik_baselines.py --k 6 [--resume]
Cikti: results/klasik_baselines.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
import cluster_mf as cm  # noqa: E402
import tabloB_plus as bp  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402


def grid_for(K):
    """K dugum icin en dengeli dikdortgen izgara."""
    best = (1, K)
    for r in range(1, int(np.sqrt(K)) + 1):
        if K % r == 0:
            best = (r, K // r)
    return best


def som_fit(X, K, seed=42, epochs=30):
    """Minimal batch-SOM; dondurur: dugum agirliklari (K x D)."""
    rng = np.random.default_rng(seed)
    r, c = grid_for(K)
    pos = np.array([[i, j] for i in range(r) for j in range(c)], float)
    W = X[rng.choice(len(X), K, replace=False)].copy()
    sigma0 = max(r, c) / 2.0
    for e in range(epochs):
        sigma = max(sigma0 * (1 - e / epochs), 0.5)
        d = ((X[:, None, :] - W[None, :, :]) ** 2).sum(-1)
        bmu = d.argmin(1)
        pd2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        h = np.exp(-pd2 / (2 * sigma ** 2))          # K x K komsuluk
        Hn = h[bmu]                                   # N x K
        num = Hn.T @ X
        den = Hn.sum(0)[:, None]
        W = np.where(den > 1e-9, num / np.maximum(den, 1e-9), W)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data, args.fold)
    X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
    S = build_sims(ctx)
    K = args.k
    bp.K = K; cm.K = K
    out = RESULTS / "klasik_baselines.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K))

    Xp = PCA(n_components=10, random_state=42).fit_transform(X)  # PCA varyantlari

    def add(name, cents):
        if (name, K) in done:
            return
        _, m = bp.evaluate(ctx, X, S, cents)
        rows.append({"yontem": name, "K": K, "fold": args.fold,
                     "seed": args.seed, **m})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{name:16s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
              f"NDCG={m['ndcg10']:.4f} P@10={m['prec10']:.4f}", flush=True)

    # 1) literatur hali: random-init, tek kosu
    km_r = KMeans(K, init="random", n_init=1, random_state=args.seed).fit(X)
    add("kmeans_random", km_r.cluster_centers_)
    # 2) guclu baseline
    km_p = KMeans(K, init="k-means++", n_init=10, random_state=args.seed).fit(X)
    add("kmeans++_n10", km_p.cluster_centers_)
    # 3) PCA-k-means: PCA uzayinda kumele, merkezleri geri tasi
    kmp = KMeans(K, init="k-means++", n_init=10, random_state=args.seed).fit(Xp)
    cent_back = np.vstack([X[kmp.labels_ == c].mean(0) if (kmp.labels_ == c).any()
                           else X.mean(0) for c in range(K)])
    add("pca_kmeans", cent_back)
    # 4) SOM
    add("som_cluster", som_fit(X, K, args.seed))
    # 5) PCA-SOM
    Wp = som_fit(Xp, K, args.seed)
    lab = ((Xp[:, None, :] - Wp[None, :, :]) ** 2).sum(-1).argmin(1)
    cent_back2 = np.vstack([X[lab == c].mean(0) if (lab == c).any() else X.mean(0)
                            for c in range(K)])
    add("pca_som", cent_back2)
    # 6) bizim yontem
    if ("AVOA_warm", K) not in done:
        from mealpy import FloatVar
        cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
        lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
        rng = np.random.default_rng(args.seed)
        c0 = km_p.cluster_centers_
        starts = np.clip(np.vstack(
            [c0.ravel()] + [c0.ravel() + rng.normal(0, 0.08 * X.std(), c0.size)
                            for _ in range(14)]), lb, ub)
        g = cls(epoch=3000, pop_size=15).solve(
            {"obj_func": bp.make_fitness(ctx, X), "bounds": FloatVar(lb=lb, ub=ub),
             "minmax": "min", "log_to": None},
            seed=args.seed, termination={"max_fe": 1200},
            starting_solutions=starts)
        add("AVOA_warm", np.asarray(g.solution).reshape(K, X.shape[1]))


if __name__ == "__main__":
    main()
