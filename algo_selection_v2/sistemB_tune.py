"""
SISTEM B ON-AYAR + UCLU KARISIM TESTI (gercek run oncesi son ayarlar).

--exp k     : Sistem B K taramasi {20,30,40}, sert ceza (50x), B0 + HGS
--exp f     : kume-MF faktor taramasi f {5,10,20}, secilen K, HGS
--exp three : SAMPIYON sistemde uclu karisim (kNN + kume-MF + global MF),
              K=10, B0 + AVOA — "kume-MF sampiyonda kullanilabilir mi?" cevabi
Secimler ic-val MAE ile; test metrikleri de kaydedilir (bilgi amacli).
Cikti: results/sistemB_tune.csv
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
import cluster_mf as cm  # noqa: E402
from eksikler_deney import SEED, Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402


def sert_fitness(ctx, X, K, cap, w=50.0):
    N = len(X)

    def fit(sol):
        c = np.asarray(sol).reshape(K, X.shape[1])
        lab = sse_wcss(X, c)[1]
        s_ = np.zeros((K, 1682)); n_ = np.zeros((K, 1682))
        np.add.at(s_, (lab[ctx.iu], ctx.ii), ctx.ir - ctx.um[ctx.iu])
        np.add.at(n_, (lab[ctx.iu], ctx.ii), 1.0)
        d = np.where(n_[lab[ctx.vu], ctx.vi] > 0,
                     s_[lab[ctx.vu], ctx.vi] / np.maximum(n_[lab[ctx.vu], ctx.vi], 1),
                     ctx.dev_g[ctx.vi])
        mae = float(np.abs(np.clip(ctx.um[ctx.vu] + d, 1, 5) - ctx.vr).mean())
        sh = np.bincount(lab, minlength=K) / N
        return mae + w * np.clip(sh - cap, 0, None).sum() \
            + 0.1 * (K - len(np.unique(lab)))
    return fit


def eval_sistemB(ctx, X, S, cent, K, f, out_probs=None):
    """Sistem B: kNN + kume-MF; istege bagli global MF ile uclu (out_probs)."""
    cm.F = f
    d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    L = d.argmin(1); near = np.argsort(d, 1)[:, :2]
    models = cm.cluster_mf_models(ctx, L)
    fbv = ctx.um[ctx.vu] + ctx.dev_g[ctx.vi]
    fbe = ctx.um[ctx.eu] + ctx.dev_g[ctx.ei]
    pmf_v = cm.cmf_predict(models, L, ctx.vu, ctx.vi, fbv)
    pmf_e = cm.cmf_predict(models, L, ctx.eu, ctx.ei, fbe)
    pk_v = cm.knn_preds(ctx, S, L, near, ctx.vu, ctx.vi)
    pk_e = cm.knn_preds(ctx, S, L, near, ctx.eu, ctx.ei)
    grid_b = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    best = (None, np.inf)
    if out_probs is None:
        for b in grid_b:
            pv = np.where(np.isnan(pk_v), pmf_v,
                          b * np.nan_to_num(pk_v) + (1 - b) * pmf_v)
            v = float(np.abs(np.clip(pv, 1, 5) - ctx.vr).mean())
            if v < best[1]:
                best = ((b,), v)
        (b,), v = best
        p = np.where(np.isnan(pk_e), pmf_e,
                     b * np.nan_to_num(pk_e) + (1 - b) * pmf_e)
        params = {"b_knn": b}
    else:
        gm_v, gm_e = out_probs
        for b1 in (0.2, 0.3, 0.4):
            for b2 in (0.0, 0.1, 0.2, 0.3):
                pv = b1 * np.nan_to_num(pk_v) + b2 * pmf_v + (1 - b1 - b2) * gm_v
                pv = np.where(np.isnan(pk_v), b2/(1-b1+1e-9)*pmf_v
                              + (1-b1-b2)/(1-b1+1e-9)*gm_v, pv)
                v = float(np.abs(np.clip(pv, 1, 5) - ctx.vr).mean())
                if v < best[1]:
                    best = ((b1, b2), v)
        (b1, b2), v = best
        p = b1 * np.nan_to_num(pk_e) + b2 * pmf_e + (1 - b1 - b2) * gm_e
        p = np.where(np.isnan(pk_e), b2/(1-b1+1e-9)*pmf_e
                     + (1-b1-b2)/(1-b1+1e-9)*gm_e, p)
        params = {"b_knn": b1, "b_cmf": b2}
    p = np.clip(p, 1, 5)
    fb = 100 * float(np.isnan(pk_e).mean())
    pool = int(np.array([(np.isin(L, near[u])).sum()
                         for u in range(len(X))]).mean())
    sz = sorted(np.bincount(L, minlength=K), reverse=True)
    m = full_metrics(ctx, p, fb, [np.arange(pool)])
    return {"val_mae": round(v, 4), **params, "havuz": pool, "maxk": sz[0], **m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["k", "f", "three"], required=True)
    ap.add_argument("--best-k", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data)
    X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
    S = build_sims(ctx)
    out = RESULTS / "sistemB_tune.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(old.config)

    def save(cfg, r):
        rows.append({"config": cfg, **r})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(cfg, {k: v for k, v in r.items()
                    if k in ("val_mae", "mae", "ndcg10", "havuz", "maxk",
                             "b_knn", "b_cmf")}, flush=True)

    if args.exp == "k":
        for K in (20, 30, 40):
            cm.K = K
            cfg = f"B0_K{K}"
            if cfg not in done:
                km = KMeans(K, init="k-means++", n_init=10,
                            random_state=SEED).fit(X)
                save(cfg, eval_sistemB(ctx, X, S, km.cluster_centers_, K, 10))
            cfg = f"HGS_K{K}"
            if cfg not in done:
                cls = discover_algorithms(["HGS.OriginalHGS"])["HGS.OriginalHGS"]
                pos, _, _, _ = solve_meta(cls, sert_fitness(ctx, X, K, 2.5 / K),
                                          np.tile(X.min(0), K),
                                          np.tile(X.max(0), K),
                                          3000, 30, SEED, max_fe=2000)
                save(cfg, eval_sistemB(ctx, X, S,
                                       np.asarray(pos).reshape(K, X.shape[1]),
                                       K, 10))

    elif args.exp == "f":
        K = args.best_k
        cm.K = K
        cls = discover_algorithms(["HGS.OriginalHGS"])["HGS.OriginalHGS"]
        pos, _, _, _ = solve_meta(cls, sert_fitness(ctx, X, K, 2.5 / K),
                                  np.tile(X.min(0), K), np.tile(X.max(0), K),
                                  3000, 30, SEED, max_fe=2000)
        cent = np.asarray(pos).reshape(K, X.shape[1])
        for f in (5, 10, 20):
            cfg = f"HGS_K{K}_f{f}"
            if cfg not in done:
                save(cfg, eval_sistemB(ctx, X, S, cent, K, f))

    elif args.exp == "three":
        K = 10
        cm.K = K
        z = np.load(RESULTS / "pacr_cache.npz")
        gm = (z["pm_val"], z["pm_te"])
        km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
        if "B0_uclu_K10" not in done:
            save("B0_uclu_K10", eval_sistemB(ctx, X, S, km.cluster_centers_,
                                             K, 10, out_probs=gm))
        cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
        pos, _, _, _ = solve_meta(cls, sert_fitness(ctx, X, K, 2.5 / K),
                                  np.tile(X.min(0), K), np.tile(X.max(0), K),
                                  3000, 30, SEED, max_fe=2000)
        if "AVOA_uclu_K10" not in done:
            save("AVOA_uclu_K10", eval_sistemB(ctx, X, S,
                                               np.asarray(pos).reshape(K, X.shape[1]),
                                               K, 10, out_probs=gm))


if __name__ == "__main__":
    main()
