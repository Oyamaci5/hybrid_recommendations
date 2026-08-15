
"""
DOGRUDAN ATAMA ARAMASI + REPAIRSIZ COK-ALGORITMA KIYASI (ML-100K, hizli).

--exp atama : Meta-sezgisel MERKEZ degil, dogrudan ATAMA arar.
    Kodlama: her kullanici icin surekli bir sayi x_u in [0,K); etiket = floor(x_u).
    Boyut = N (943), merkez aramada K*D (=120-800). Warm start: B0 etiketleri.
    Amac: "merkez parametrizasyonu ne kaybettiriyor?" sorusuna cevap.

--exp repairsiz : Ayni merkez aramasi ama fitness ve degerlendirmede repair YOK
    (serbest Voronoi). 5 algoritma + B0. Amac: kisit olmadan siralamayi gormek.

Kullanim:
  python algo_selection_v2/dogrudan_atama.py --exp atama --k 6
  python algo_selection_v2/dogrudan_atama.py --exp repairsiz --k 6
Cikti: results/dogrudan_atama.csv
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
import tabloB_plus as bp  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402

N_I = 1682
ALGOS = ["AVOA.OriginalAVOA", "NGO.OriginalNGO", "HHO.OriginalHHO",
         "HGS.OriginalHGS", "GWO.OriginalGWO"]


def serbest_ata(X, cents):
    d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).astype(np.int32), np.argsort(d, 1)[:, :2].astype(np.int32)


def near_from_labels(X, L, K):
    """Etiketlerden merkez turet, ikinci en yakin kumeyi bul (havuz icin)."""
    cent = np.vstack([X[L == c].mean(0) if (L == c).any() else X.mean(0)
                      for c in range(K)])
    d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d, 1)
    near = np.empty((len(X), 2), np.int32)
    near[:, 0] = L
    ilk = order[:, 0]
    near[:, 1] = np.where(ilk != L, ilk, order[:, 1])
    return near, cent


def degerlendir(ctx, X, L, near, K, S):
    cm.K = K
    models = cm.cluster_mf_models(ctx, L)
    flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
    n = np.bincount(flat, minlength=K * N_I)
    sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu], minlength=K * N_I)

    def bias(us, is_):
        a = near[us, 0].astype(np.int64) * N_I + is_
        b = near[us, 1].astype(np.int64) * N_I + is_
        den = n[a] + n[b]
        dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1),
                       ctx.dev_g[is_])
        return ctx.um[us] + dev

    p_bias = bias(ctx.eu, ctx.ei)
    p_mf = cm.cmf_predict(models, L, ctx.eu, ctx.ei, p_bias)
    p_knn = cm.knn_preds(ctx, S, L, near, ctx.eu, ctx.ei)
    fb = np.isnan(p_knn)
    p_knn = np.where(fb, p_bias, np.nan_to_num(p_knn))
    p = np.clip(0.5 * p_knn + 0.5 * p_mf, 1, 5)
    havuz = int(np.mean([(np.isin(L, near[u])).sum()
                         for u in range(0, len(X), 5)]))
    r = full_metrics(ctx, p, 100 * float(fb.mean()), [np.arange(1)])
    r.pop("havuz_ort", None)
    sz = np.bincount(L, minlength=K)
    r.update({"havuz": havuz, "maxk": int(sz.max()), "mink": int(sz.min())})
    return r


def fitness_atama(ctx, X, K):
    """Etiket vektoru uzerinden val-MAE (soft top-2 bias vekili)."""
    def fit(sol):
        L = np.clip(np.floor(np.asarray(sol)).astype(np.int32), 0, K - 1)
        if len(np.unique(L)) < 2:
            return 5.0
        near, _ = near_from_labels(X, L, K)
        flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
        n = np.bincount(flat, minlength=K * N_I)
        sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu],
                         minlength=K * N_I)
        a = near[ctx.vu, 0].astype(np.int64) * N_I + ctx.vi
        b = near[ctx.vu, 1].astype(np.int64) * N_I + ctx.vi
        den = n[a] + n[b]
        dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1),
                       ctx.dev_g[ctx.vi])
        return float(np.abs(np.clip(ctx.um[ctx.vu] + dev, 1, 5)
                            - ctx.vr).mean())
    return fit


def fitness_merkez_serbest(ctx, X, K):
    """Merkez arama, repair YOK (serbest Voronoi)."""
    def fit(sol):
        c = np.asarray(sol).reshape(K, X.shape[1])
        L, near = serbest_ata(X, c)
        if len(np.unique(L)) < 2:
            return 5.0
        flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
        n = np.bincount(flat, minlength=K * N_I)
        sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu],
                         minlength=K * N_I)
        a = near[ctx.vu, 0].astype(np.int64) * N_I + ctx.vi
        b = near[ctx.vu, 1].astype(np.int64) * N_I + ctx.vi
        den = n[a] + n[b]
        dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1),
                       ctx.dev_g[ctx.vi])
        return float(np.abs(np.clip(ctx.um[ctx.vu] + dev, 1, 5)
                            - ctx.vr).mean())
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["atama", "repairsiz"], required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fe", type=int, default=1200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data, args.fold)
    X = build_space(ctx, genre_profile(ctx, data), "nmf")
    S = build_sims(ctx)
    K, N = args.k, len(X)
    km = KMeans(K, init="k-means++", n_init=10, random_state=args.seed).fit(X)
    c0 = km.cluster_centers_
    L0 = km.labels_.astype(np.int32)

    out = RESULTS / "dogrudan_atama.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.deney, old.yontem, old.K))

    from mealpy import FloatVar
    algos = discover_algorithms(ALGOS)
    rng = np.random.default_rng(args.seed)

    # --- B0 referansi (ilgili modda) ---
    ad = f"B0_{args.exp}"
    if (args.exp, ad, K) not in done:
        if args.exp == "atama":
            near, _ = near_from_labels(X, L0, K)
            r = degerlendir(ctx, X, L0, near, K, S)
        else:
            L, near = serbest_ata(X, c0)
            r = degerlendir(ctx, X, L, near, K, S)
        rows.append({"deney": args.exp, "yontem": ad, "K": K, **r})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{ad:16s} MAE={r['mae']:.4f} NDCG={r['ndcg10']:.4f} "
              f"havuz={r['havuz']} max/min={r['maxk']}/{r['mink']}", flush=True)

    for aname, cls in algos.items():
        kisa = aname.split(".")[0] + "_" + args.exp
        if (args.exp, kisa, K) in done:
            continue
        t0 = time.time()
        if args.exp == "atama":
            lb, ub = np.zeros(N), np.full(N, K - 1e-6)
            starts = np.clip(np.vstack(
                [L0 + 0.5] + [L0 + 0.5 + rng.normal(0, 0.6, N)
                              for _ in range(14)]), lb, ub)
            g = cls(epoch=5000, pop_size=15).solve(
                {"obj_func": fitness_atama(ctx, X, K),
                 "bounds": FloatVar(lb=lb, ub=ub), "minmax": "min",
                 "log_to": None},
                seed=args.seed, termination={"max_fe": args.max_fe},
                starting_solutions=starts)
            L = np.clip(np.floor(np.asarray(g.solution)).astype(np.int32),
                        0, K - 1)
            near, _ = near_from_labels(X, L, K)
        else:
            lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
            starts = np.clip(np.vstack(
                [c0.ravel()] + [c0.ravel() + rng.normal(0, 0.08 * X.std(),
                                                        c0.size)
                                for _ in range(14)]), lb, ub)
            g = cls(epoch=5000, pop_size=15).solve(
                {"obj_func": fitness_merkez_serbest(ctx, X, K),
                 "bounds": FloatVar(lb=lb, ub=ub), "minmax": "min",
                 "log_to": None},
                seed=args.seed, termination={"max_fe": args.max_fe},
                starting_solutions=starts)
            L, near = serbest_ata(X, np.asarray(g.solution).reshape(K, X.shape[1]))
        r = degerlendir(ctx, X, L, near, K, S)
        rows.append({"deney": args.exp, "yontem": kisa, "K": K,
                     "sure_s": round(time.time() - t0, 1), **r})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{kisa:16s} MAE={r['mae']:.4f} NDCG={r['ndcg10']:.4f} "
              f"havuz={r['havuz']} max/min={r['maxk']}/{r['mink']}", flush=True)


if __name__ == "__main__":
    main()
