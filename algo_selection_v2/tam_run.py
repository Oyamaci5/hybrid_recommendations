"""
TAM RUN — makale ana tablolari (cok-seed x 5 resmi fold).

Tablo A (mutlak performans): K=10, nmf+genre, soft top-2, kNN k=20,
  GLOBAL ALS-MF (f=80, l=10), beta izgara 0.2-0.7 (ic-val).
  Yontemler: B0, AVOA (+KNN_ALL referansi fold basina bir kez).
Tablo B (algoritma farki): K=30, KUME-MF (f=10), sert ceza w=50, kNN k=20.
  Yontemler: B0, AVOA, NGO, HGS, HHO, GWO.

Cikti:
  results/tam_run_A.csv, tam_run_B.csv        (satir = yontem x fold x seed)
  results/tam_run_userr_{A,B}.npz             (kullanici-bazli |hata| ortalamalari
                                               -> kullanici-bazli Wilcoxon icin)
Kullanim (kaldigi yerden devam eder):
  python algo_selection_v2/tam_run.py --table B --folds 1 2 3 4 5 --seeds 42 43 ... --resume
Analiz: tam_run_analiz.py (ozet + Friedman/Holm + Wilcoxon + grafikler)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
import cluster_mf as cm  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile, hard_fitness  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402
from sampiyon import als_fit  # noqa: E402
from sistemB_tune import sert_fitness  # noqa: E402

BETAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
A_ALGOS = ["AVOA.OriginalAVOA", "NGO.OriginalNGO", "HHO.OriginalHHO",
           "HGS.OriginalHGS", "GWO.OriginalGWO"]
B_ALGOS = ["AVOA.OriginalAVOA", "NGO.OriginalNGO", "HGS.OriginalHGS",
           "HHO.OriginalHHO", "GWO.OriginalGWO"]


def user_mae_vec(ctx, p):
    v = np.full(943, np.nan)
    for u in range(943):
        idx = ctx.by_user[u]
        if idx:
            v[u] = float(np.abs(p[np.array(idx)] - ctx.er[np.array(idx)]).mean())
    return v


def knn_soft(ctx, S, X, cent, us, is_, k=20):
    d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    L = d.argmin(1)
    nc = min(2, cent.shape[0])          # tek kume (KNN_ALL) durumunda 1
    near = np.argsort(d, 1)[:, :nc]
    if nc == 1:                          # tum havuz: ikinci sutunu birinciyle doldur
        near = np.hstack([near, near])
    p = np.empty(len(us))
    for j, (u, i) in enumerate(zip(us, is_)):
        cand = ctx.raters[i]
        cand = cand[(L[cand] == near[u, 0]) | (L[cand] == near[u, 1])]
        pk = np.nan
        if len(cand):
            s = S[u, cand]; t = np.argsort(-s)[:k]
            s, cc = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                pk = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
        p[j] = pk
    pool = int(np.mean([(np.isin(L, near[u])).sum() for u in range(len(X))]))
    return p, L, near, pool


def blend_pick(pk_v, alt_v, vr, pk_e, alt_e):
    best_b, best = None, np.inf
    for b in BETAS:
        pv = np.where(np.isnan(pk_v), alt_v, b * np.nan_to_num(pk_v) + (1 - b) * alt_v)
        v = float(np.abs(np.clip(pv, 1, 5) - vr).mean())
        if v < best:
            best_b, best = b, v
    p = np.where(np.isnan(pk_e), alt_e,
                 best_b * np.nan_to_num(pk_e) + (1 - best_b) * alt_e)
    return np.clip(p, 1, 5), best_b, best


def save_user_vec(npz_path, key, vec):
    d = dict(np.load(npz_path)) if npz_path.exists() else {}
    d[key] = vec
    np.savez_compressed(npz_path, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["A", "B"], required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-100k"))
    ap.add_argument("--uzay", choices=["nmf", "nmf+genre"], default="nmf",
                    help="nmf = saf CF (varsayilan), nmf+genre = tur bilgisiyle")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(args.data_dir)

    etiket = args.table + ("_genre" if args.uzay == "nmf+genre" else "")
    out = RESULTS / f"tam_run_{etiket}.csv"
    upath = RESULTS / f"tam_run_userr_{etiket}.npz"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.fold, old.seed))

    algos = discover_algorithms(A_ALGOS if args.table == "A" else B_ALGOS)

    for fold in args.folds:
        need = [s for s in args.seeds
                if any((y.split(".")[0], fold, s) not in done
                       for y in (["B0"] + list(algos)))]
        if not need and ("KNN_ALL", fold, args.seeds[0]) in done:
            continue
        t0 = time.time()
        ctx = Ctx(data, fold)
        X = build_space(ctx, genre_profile(ctx, data), args.uzay)
        S = build_sims(ctx)
        print(f"[fold {fold}] baglam {time.time()-t0:.0f}s", flush=True)

        if args.table == "A":
            acache = RESULTS / f"tam_als_f{fold}.npz"
            if acache.exists():
                z = np.load(acache); pm_v, pm_e = z["v"], z["e"]
            else:
                mf = als_fit(ctx, 80, 10)
                pm_v, pm_e = mf(ctx.vu, ctx.vi), mf(ctx.eu, ctx.ei)
                np.savez(acache, v=pm_v, e=pm_e)

        def emit(yontem, fold, seed, p, extra):
            r = {"yontem": yontem, "fold": fold, "seed": seed, **extra,
                 **full_metrics(ctx, p, extra.get("fallback_in", 0), [np.arange(1)])}
            r.pop("havuz_ort", None); r.pop("fallback_in", None)
            rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            save_user_vec(upath, f"{yontem}|f{fold}|s{seed}", user_mae_vec(ctx, p))
            print(f"  {yontem} f{fold} s{seed} MAE={r['mae']:.4f} "
                  f"NDCG={r['ndcg10']:.4f}", flush=True)

        for seed in args.seeds:
            # --- B0 ---
            K = 10 if args.table == "A" else 30
            if ("B0", fold, seed) not in done:
                km = KMeans(K, init="k-means++", n_init=10,
                            random_state=seed).fit(X)
                cent = km.cluster_centers_
                pk_v, _, _, _ = knn_soft(ctx, S, X, cent, ctx.vu, ctx.vi)
                pk_e, L, near, pool = knn_soft(ctx, S, X, cent, ctx.eu, ctx.ei)
                if args.table == "A":
                    alt_v, alt_e = pm_v, pm_e
                else:
                    cm.K, cm.F = K, 10
                    models = cm.cluster_mf_models(ctx, L)
                    fbv = ctx.um[ctx.vu] + ctx.dev_g[ctx.vi]
                    fbe = ctx.um[ctx.eu] + ctx.dev_g[ctx.ei]
                    alt_v = cm.cmf_predict(models, L, ctx.vu, ctx.vi, fbv)
                    alt_e = cm.cmf_predict(models, L, ctx.eu, ctx.ei, fbe)
                p, b, v = blend_pick(pk_v, alt_v, ctx.vr, pk_e, alt_e)
                emit("B0", fold, seed, p,
                     {"beta": b, "val_mae": round(v, 4), "havuz": pool,
                      "fallback_in": 100 * float(np.isnan(pk_e).mean())})
            # --- metalar ---
            for aname, cls in algos.items():
                kisa = aname.split(".")[0]
                if (kisa, fold, seed) in done:
                    continue
                if args.table == "A":
                    fitf = hard_fitness(ctx, X, K, 2.5 / K)
                else:
                    fitf = sert_fitness(ctx, X, K, 2.5 / K, w=50.0)
                pos, _, _, _ = solve_meta(cls, fitf, np.tile(X.min(0), K),
                                          np.tile(X.max(0), K),
                                          3000, 30, seed, max_fe=2000)
                cent = np.asarray(pos).reshape(K, X.shape[1])
                pk_v, _, _, _ = knn_soft(ctx, S, X, cent, ctx.vu, ctx.vi)
                pk_e, L, near, pool = knn_soft(ctx, S, X, cent, ctx.eu, ctx.ei)
                if args.table == "A":
                    alt_v, alt_e = pm_v, pm_e
                else:
                    cm.K, cm.F = K, 10
                    models = cm.cluster_mf_models(ctx, L)
                    fbv = ctx.um[ctx.vu] + ctx.dev_g[ctx.vi]
                    fbe = ctx.um[ctx.eu] + ctx.dev_g[ctx.ei]
                    alt_v = cm.cmf_predict(models, L, ctx.vu, ctx.vi, fbv)
                    alt_e = cm.cmf_predict(models, L, ctx.eu, ctx.ei, fbe)
                p, b, v = blend_pick(pk_v, alt_v, ctx.vr, pk_e, alt_e)
                mx = int(np.bincount(L, minlength=K).max())
                emit(kisa, fold, seed, p,
                     {"beta": b, "val_mae": round(v, 4), "havuz": pool,
                      "maxk": mx,
                      "fallback_in": 100 * float(np.isnan(pk_e).mean())})
        # KNN_ALL referansi (Tablo A, fold basina bir kez, seed'den bagimsiz)
        if args.table == "A" and ("KNN_ALL", fold, args.seeds[0]) not in done:
            pk_v, _, _, _ = knn_soft(ctx, S, X, X.mean(0, keepdims=True),
                                     ctx.vu, ctx.vi)  # tek kume = tum havuz
            pk_e, _, _, _ = knn_soft(ctx, S, X, X.mean(0, keepdims=True),
                                     ctx.eu, ctx.ei)
            p, b, v = blend_pick(pk_v, pm_v, ctx.vr, pk_e, pm_e)
            emit("KNN_ALL", fold, args.seeds[0], p,
                 {"beta": b, "val_mae": round(v, 4), "havuz": 943,
                  "fallback_in": 100 * float(np.isnan(pk_e).mean())})


if __name__ == "__main__":
    main()
