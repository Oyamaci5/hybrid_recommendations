"""
NDCG-HIZALI FITNESS PILOTU (madde 3).

Hipotez: Oneri sisteminin isi siralamadir; incelenen 6 makalenin 6'si da yalniz
hata metrigi (MAE/RMSE) optimize ediyor. Fitness dogrudan val-NDCG olursa
siralama kalitesi ve algoritma farki artar mi?

Iskele Tablo B+ ile ayni (repair esit havuz + warm start, K=30, kume-MF, beta val'de);
TEK DEGISEN: fitness.
  fit_mae  : val MAE  (mevcut referans)
  fit_ndcg : 1 - val NDCG@10   (siralama hedefi)
  fit_hib  : 0.5*MAE_norm + 0.5*(1-NDCG)   (karma)
Fitness ic tahminci = soft bias vekili (hizli); degerlendirme tam sistemle.

Kullanim: python algo_selection_v2/ndcg_fitness.py --fitness ndcg --seeds 42 43
Cikti: results/ndcg_fitness.csv
NOT: tabloB_plus.py kosarken bunu AYRI terminalde calistirmayin (CPU paylasimi);
sirayla calistirin.
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
from eksikler_deney import THR, TOPN, Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402
from tabloB_plus import K, evaluate, repair_assign  # noqa: E402

ALGOS = ["AVOA.OriginalAVOA", "HGS.OriginalHGS", "GWO.OriginalGWO"]


def val_index(ctx):
    """Val kayitlarini kullanici bazinda grupla (NDCG icin)."""
    by = {}
    for j, u in enumerate(ctx.vu):
        by.setdefault(u, []).append(j)
    # en az 2 kaydi ve >=1 ilgili filmi olan kullanicilar
    keep = {}
    for u, idx in by.items():
        idx = np.array(idx)
        if len(idx) >= 2 and (ctx.vr[idx] >= THR).any():
            keep[u] = idx
    return keep


def soft_bias_preds(ctx, L, near, us, is_):
    s_ = np.zeros((K, 1682)); n_ = np.zeros((K, 1682))
    np.add.at(s_, (L[ctx.iu], ctx.ii), ctx.ir - ctx.um[ctx.iu])
    np.add.at(n_, (L[ctx.iu], ctx.ii), 1.0)
    a, b = near[us, 0], near[us, 1]
    num = s_[a, is_] + s_[b, is_]; den = n_[a, is_] + n_[b, is_]
    dev = np.where(den > 0, num / np.maximum(den, 1), ctx.dev_g[is_])
    return np.clip(ctx.um[us] + dev, 1, 5)


def make_fitness(ctx, X, mode, vidx):
    disc_cache = {}

    def ndcg_of(p):
        tot, n = 0.0, 0
        for u, idx in vidx.items():
            order = idx[np.argsort(-p[idx])]
            rel = (ctx.vr[order] >= THR).astype(float)
            m = min(TOPN, len(order))
            if m not in disc_cache:
                disc_cache[m] = np.log2(np.arange(2, m + 2))
            d = disc_cache[m]
            idcg = (np.sort(rel)[::-1][:m] / d).sum()
            if idcg > 0:
                tot += (rel[:m] / d).sum() / idcg; n += 1
        return tot / max(n, 1)

    def fit(sol):
        c = np.asarray(sol).reshape(K, X.shape[1])
        L, near = repair_assign(X, c)
        p = soft_bias_preds(ctx, L, near, ctx.vu, ctx.vi)
        if mode == "mae":
            return float(np.abs(p - ctx.vr).mean())
        nd = ndcg_of(p)
        if mode == "ndcg":
            return 1.0 - nd
        return 0.5 * float(np.abs(p - ctx.vr).mean()) + 0.5 * (1.0 - nd)
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitness", choices=["mae", "ndcg", "hib"], default="ndcg")
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--max-fe", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(ROOT.parent / "data" / "ml-100k")
    out = RESULTS / "ndcg_fitness.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.fitness, old.fold, old.seed))

    from mealpy import FloatVar
    algos = discover_algorithms(ALGOS)

    for fold in args.folds:
        ctx = Ctx(data, fold)
        X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
        S = build_sims(ctx)
        vidx = val_index(ctx)
        fitf = make_fitness(ctx, X, args.fitness, vidx)
        lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
        for seed in args.seeds:
            km = KMeans(K, init="k-means++", n_init=10, random_state=seed).fit(X)
            c0 = km.cluster_centers_
            rng = np.random.default_rng(seed)
            starts = np.clip(np.vstack(
                [c0.ravel()] + [c0.ravel() + rng.normal(0, 0.08 * X.std(), c0.size)
                                for _ in range(14)]), lb, ub)
            for aname, cls in algos.items():
                kisa = aname.split(".")[0]
                if (kisa, args.fitness, fold, seed) in done:
                    continue
                model = cls(epoch=3000, pop_size=15)
                g = model.solve({"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub),
                                 "minmax": "min", "log_to": None},
                                seed=seed, termination={"max_fe": args.max_fe},
                                starting_solutions=starts)
                _, m = evaluate(ctx, X, S,
                                np.asarray(g.solution).reshape(K, X.shape[1]))
                rows.append({"yontem": kisa, "fitness": args.fitness, "fold": fold,
                             "seed": seed, "fit_val": round(float(g.target.fitness), 4),
                             **m})
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"  {kisa} fit={args.fitness} f{fold} s{seed} "
                      f"MAE={m['mae']:.4f} NDCG={m['ndcg10']:.4f} "
                      f"P@10={m['prec10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
