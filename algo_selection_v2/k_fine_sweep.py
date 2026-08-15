"""
INCE K TARAMASI + ALGORITMA KIYASI — sampiyon pipeline uzerinde.

Sabitler (sampiyon.csv'den): nmf+genre uzayi, soft top-2, kNN k=20,
ALS-MF f=80 lam=10, karisim beta=0.4, denge cap=2.5/K, NFE=2000, seed 42.
Degisenler: K = 10,12,...,30  x  yontem = {B0, AVOA, GWO, NGO}

Cikti: results/k_fine.csv  +  results/plots/k_fine_*.png
Kullanim: python algo_selection_v2/k_fine_sweep.py [--resume]
          python algo_selection_v2/k_fine_sweep.py --plot   (sadece grafik)
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
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from eksikler_deney import SEED, Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile, hard_fitness  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta  # noqa: E402
from sampiyon import als_fit  # noqa: E402

KNN_K, BETA, MAXFE = 20, 0.4, 2000
ALGOS = ["AVOA.OriginalAVOA", "GWO.OriginalGWO", "NGO.OriginalNGO"]


def knn_test_preds(ctx, S, L, near, us, is_):
    p = np.empty(len(us))
    for j, (u, i) in enumerate(zip(us, is_)):
        cand = ctx.raters[i]
        m = (L[cand] == near[u, 0]) | (L[cand] == near[u, 1])
        cand = cand[m]
        if len(cand):
            s = S[u, cand]; t = np.argsort(-s)[:KNN_K]
            s, cc = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                p[j] = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
                continue
        p[j] = np.nan
    return p


def run(args):
    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data)
    G = genre_profile(ctx, data)
    X = build_space(ctx, G, "nmf+genre")
    S = build_sims(ctx, iuf=False, sig=False)

    mf_npz = RESULTS / "als_f80_preds.npz"
    if mf_npz.exists():
        z = np.load(mf_npz); pm_te = z["pm_te"]
    else:
        mf = als_fit(ctx, 80, 10)
        pm_te = mf(ctx.eu, ctx.ei)
        np.savez(mf_npz, pm_te=pm_te)

    out = RESULTS / "k_fine.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(old.config)
    algo_cls = discover_algorithms(ALGOS)

    def eval_cfg(cfg, K, yontem, cents):
        d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        L = d.argmin(1); near = np.argsort(d, 1)[:, :2]
        pool_sz = np.array([(np.isin(L, near[u])).sum() for u in range(len(X))])
        pk = knn_test_preds(ctx, S, L, near, ctx.eu, ctx.ei)
        p = np.clip(np.where(np.isnan(pk), pm_te,
                             BETA * np.nan_to_num(pk) + (1 - BETA) * pm_te), 1, 5)
        fbp = 100 * float(np.isnan(pk).mean())
        r = {"config": cfg, "K": K, "yontem": yontem,
             "havuz_ort": int(pool_sz.mean()),
             "havuz_pct": round(100 * pool_sz.mean() / len(X), 1),
             **full_metrics(ctx, p, fbp, [np.arange(s) for s in pool_sz])}
        del r["havuz_ort"]  # full_metrics kendi havuzunu yazar; bizimkini koru
        r["havuz_ort"] = int(pool_sz.mean())
        rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
        print(cfg, round(r["mae"], 4), round(r["ndcg10"], 4),
              f"havuz={r['havuz_ort']}", flush=True)

    for K in range(10, 31, 2):
        cfg = f"B0_K{K}"
        if cfg not in done:
            km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
            eval_cfg(cfg, K, "B0", km.cluster_centers_)
        for aname in ALGOS:
            kisa = aname.split(".")[0]
            cfg = f"{kisa}_K{K}"
            if cfg in done:
                continue
            pos, _, _, _ = solve_meta(algo_cls[aname],
                                      hard_fitness(ctx, X, K, 2.5 / K),
                                      np.tile(X.min(0), K), np.tile(X.max(0), K),
                                      3000, 30, SEED, max_fe=MAXFE)
            eval_cfg(cfg, K, kisa, np.asarray(pos).reshape(K, X.shape[1]))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(RESULTS / "k_fine.csv")
    renk = {"B0": "tab:gray", "AVOA": "tab:red", "GWO": "tab:blue",
            "NGO": "tab:green"}
    panels = [("mae", "Test MAE (dusuk iyi)"), ("ndcg10", "NDCG@10 (yuksek iyi)"),
              ("prec10", "Precision@10"), ("havuz_pct", "Komsu havuzu (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, baslik) in zip(axes.ravel(), panels):
        for y, g in df.groupby("yontem"):
            g = g.sort_values("K")
            ax.plot(g.K, g[col], "o-", label=y, color=renk.get(y), lw=1.8, ms=4)
        if col == "mae":
            ax.axhline(0.7467, ls="--", c="k", lw=1, label="kumesiz kNN")
        if col == "ndcg10":
            ax.axhline(0.8352, ls="--", c="k", lw=1)
        ax.set_xlabel("K"); ax.set_title(baslik); ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("Ince K taramasi — sampiyon pipeline (kNN+MF, beta=0.4, seed 42)")
    fig.tight_layout()
    fig.savefig(PLOTS / "k_fine_sweep.png", dpi=150)
    print("Grafik ->", PLOTS / "k_fine_sweep.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot()
    else:
        run(a)
