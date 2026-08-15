"""
KUME-BASINA ALS-MF KARISIMI — algoritma farkini KORUYAN guclu tahminci.

Sorun: global ALS-MF kumelemeden bagimsiz -> meta farkini ~4 kat sonumluyor.
Cozum: her kumenin KENDI MF modeli (uyelerinin ic-train puanlariyla egitilir).
Tahmin: p = beta * kume-ici kNN + (1-beta) * kume-MF(u,i)   [beta ic-val'de]
Tahminin iki bileseni de kumelemeye bagli -> algoritma farki tahmine akar.

Sistem B adayi: K=30, dengeli fitness (cap=2.5/K), nmf+genre, soft top-2,
kNN k=20. Yontemler: B0, AVOA, NGO, HGS. Seed 42, fold 1.
Cikti: results/cluster_mf.csv
Kullanim: python algo_selection_v2/cluster_mf.py [--resume]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from eksikler_deney import SEED, Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile, hard_fitness  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402

K, KNN_K, F, LAM, ITERS = 30, 20, 10, 10.0, 6
N_I = 1682          # ML-100K film sayisi; ML-1M icin disaridan cm.N_I = 3952
ALGOS = ["AVOA.OriginalAVOA", "NGO.OriginalNGO", "HGS.OriginalHGS"]


def cluster_mf_models(ctx, labels):
    """Kume basina bias'li ALS; dondurdugu fonksiyon (u,i) tahmini verir."""
    models = {}
    for c in np.unique(labels):
        uc = np.flatnonzero(labels == c)
        keep = np.isin(ctx.iu, uc)
        iu, ii, ir = ctx.iu[keep], ctx.ii[keep], ctx.ir[keep]
        if len(ir) < 50:
            models[c] = None
            continue
        mu = float(ir.mean())
        remap = -np.ones(len(ctx.R), int); remap[uc] = np.arange(len(uc))
        iuf = remap[iu]
        bu = np.zeros(len(uc)); bi = np.zeros(N_I)
        rng = np.random.default_rng(42)
        P = rng.normal(0, .1, (len(uc), F)); Q = rng.normal(0, .1, (N_I, F))
        ui = [np.flatnonzero(iuf == x) for x in range(len(uc))]
        items = np.unique(ii)
        iu_ = {i: np.flatnonzero(ii == i) for i in items}
        I = LAM * np.eye(F)
        for _ in range(ITERS):
            for x in range(len(uc)):
                e = ui[x]
                if not len(e):
                    continue
                j = ii[e]; r = ir[e] - mu - bi[j]
                bu[x] = (r - P[x] @ Q[j].T).sum() / (len(e) + LAM)
                P[x] = np.linalg.solve(Q[j].T @ Q[j] + I, Q[j].T @ (r - bu[x]))
            for i in items:
                e = iu_[i]
                j = iuf[e]; r = ir[e] - mu - bu[j]
                bi[i] = (r - (P[j] * Q[i]).sum(1)).sum() / (len(e) + LAM)
                Q[i] = np.linalg.solve(P[j].T @ P[j] + I, P[j].T @ (r - bi[i]))
        models[c] = (mu, bu, bi, P, Q, remap)
    return models


def cmf_predict(models, labels, us, is_, fallback):
    p = np.empty(len(us))
    for j, (u, i) in enumerate(zip(us, is_)):
        m = models.get(labels[u])
        if m is None:
            p[j] = fallback[j]; continue
        mu, bu, bi, P, Q, remap = m
        x = remap[u]
        p[j] = mu + bu[x] + bi[i] + P[x] @ Q[i] if x >= 0 else fallback[j]
    return np.clip(p, 1, 5)


def knn_preds(ctx, S, L, near, us, is_):
    p = np.empty(len(us))
    for j, (u, i) in enumerate(zip(us, is_)):
        cand = ctx.raters[i]
        cand = cand[(L[cand] == near[u, 0]) | (L[cand] == near[u, 1])]
        pk = np.nan
        if len(cand):
            s = S[u, cand]; t = np.argsort(-s)[:KNN_K]
            s, cc = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                pk = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
        p[j] = pk
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data)
    X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
    S = build_sims(ctx)
    out = RESULTS / "cluster_mf.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(old.yontem)

    cents_all = {}
    km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
    cents_all["B0"] = km.cluster_centers_
    cls = discover_algorithms(ALGOS)
    for a in ALGOS:
        kisa = a.split(".")[0]
        if kisa in done:
            continue
        pos, _, _, _ = solve_meta(cls[a], hard_fitness(ctx, X, K, 2.5 / K),
                                  np.tile(X.min(0), K), np.tile(X.max(0), K),
                                  3000, 30, SEED, max_fe=2000)
        cents_all[kisa] = np.asarray(pos).reshape(K, X.shape[1])

    fbv = ctx.um[ctx.vu] + ctx.dev_g[ctx.vi]
    fbe = ctx.um[ctx.eu] + ctx.dev_g[ctx.ei]
    for yn, cent in cents_all.items():
        if yn in done:
            continue
        d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        L = d.argmin(1); near = np.argsort(d, 1)[:, :2]
        models = cluster_mf_models(ctx, L)
        pmf_v = cmf_predict(models, L, ctx.vu, ctx.vi, fbv)
        pmf_e = cmf_predict(models, L, ctx.eu, ctx.ei, fbe)
        pk_v = knn_preds(ctx, S, L, near, ctx.vu, ctx.vi)
        pk_e = knn_preds(ctx, S, L, near, ctx.eu, ctx.ei)
        best_b, best = None, np.inf
        for b in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pv = np.where(np.isnan(pk_v), pmf_v, b * np.nan_to_num(pk_v)
                          + (1 - b) * pmf_v)
            v = float(np.abs(np.clip(pv, 1, 5) - ctx.vr).mean())
            if v < best:
                best_b, best = b, v
        p = np.where(np.isnan(pk_e), pmf_e, best_b * np.nan_to_num(pk_e)
                     + (1 - best_b) * pmf_e)
        p = np.clip(p, 1, 5)
        fb = 100 * float(np.isnan(pk_e).mean())
        pool = np.array([(np.isin(L, near[u])).sum()
                         for u in range(len(X))]).mean()
        sz = sorted(np.bincount(L, minlength=K), reverse=True)
        r = {"yontem": yn, "beta": best_b, "val_mae": round(best, 4),
             "havuz": int(pool), "boyut_max": sz[0],
             **full_metrics(ctx, p, fb, [np.arange(int(pool))])}
        rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
        print(yn, f"b={best_b}", round(r["mae"], 4), round(r["ndcg10"], 4),
              f"havuz={int(pool)} maxk={sz[0]}", flush=True)


if __name__ == "__main__":
    main()
