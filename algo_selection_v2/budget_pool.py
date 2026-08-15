"""
ESIT BUTCELI HAVUZ KIYASI — algoritma farkini geri kazanma deneyi.

Sorun (K9): havuz boyutu yontemin gizli serbestlik derecesi; GWO buyuk havuzla
"kazaniyor". Cozum: SABIT butce B — her kullanicinin komsu-aday havuzu tam B kisi.
Havuz kurma: kumeler merkez-uzakligina gore siralanir, sirayla eklenir; butceyi
asan son kume, kullaniciya (ozellik uzayinda) en yakin uyeleriyle kirpilir.
Boylece TEK degisken kumeleme kalitesi = hangi B kisi secildigi.

Sabitler: nmf+genre, K=20 (ince granul), kNN k=20, ALS-MF f80/l10, beta=0.4,
denge cap=2.5/K, NFE=2000, seed 42.
Degisenler: butce B_pct in {15, 25, 35} x yontem {B0, AVOA, GWO, NGO}.

Cikti: results/budget_pool.csv + plots/budget_pool.png
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
from recompute_scores import discover_algorithms, solve_meta  # noqa: E402
from sampiyon import als_fit  # noqa: E402

K, KNN_K, BETA, MAXFE = 20, 20, 0.4, 2000
ALGOS = ["AVOA.OriginalAVOA", "GWO.OriginalGWO", "NGO.OriginalNGO"]


def budget_pools(X, cents, B):
    """Her kullanici icin tam B kisilik havuz (kume-hizli yakin komsu secimi)."""
    d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    L = d.argmin(1)
    order = np.argsort(d, 1)
    members = [np.flatnonzero(L == c) for c in range(len(cents))]
    pools = []
    for u in range(len(X)):
        pool = []
        for c in order[u]:
            m = members[c]
            m = m[m != u]
            if len(pool) + len(m) <= B:
                pool.extend(m.tolist())
            else:
                need = B - len(pool)
                dd = ((X[m] - X[u]) ** 2).sum(1)
                pool.extend(m[np.argsort(dd)[:need]].tolist())
            if len(pool) >= B:
                break
        pools.append(np.array(pool[:B]))
    return pools


def eval_budget(ctx, S, pm_te, pools):
    p = np.empty(len(ctx.eu)); fb = 0
    memb = [set(x.tolist()) for x in pools]
    for j, (u, i) in enumerate(zip(ctx.eu, ctx.ei)):
        cand = ctx.raters[i]
        cand = np.array([c for c in cand if c in memb[u]])
        pk = np.nan
        if len(cand):
            s = S[u, cand]; t = np.argsort(-s)[:KNN_K]
            s, cc = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                pk = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
        if np.isnan(pk):
            p[j] = pm_te[j]; fb += 1
        else:
            p[j] = BETA * pk + (1 - BETA) * pm_te[j]
    return np.clip(p, 1, 5), 100 * fb / len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        return plot()
    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data)
    X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
    S = build_sims(ctx, iuf=False, sig=False)
    z = np.load(RESULTS / "als_f80_preds.npz"); pm_te = z["pm_te"]

    out = RESULTS / "budget_pool.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(old.config)

    cents_all = {}
    km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
    cents_all["B0"] = km.cluster_centers_
    cls = discover_algorithms(ALGOS)
    for a in ALGOS:
        kisa = a.split(".")[0]
        pos, _, _, _ = solve_meta(cls[a], hard_fitness(ctx, X, K, 2.5 / K),
                                  np.tile(X.min(0), K), np.tile(X.max(0), K),
                                  3000, 30, SEED, max_fe=MAXFE)
        cents_all[kisa] = np.asarray(pos).reshape(K, X.shape[1])

    for B_pct in (15, 25, 35):
        B = int(len(X) * B_pct / 100)
        for yn, c in cents_all.items():
            cfg = f"{yn}_B{B_pct}"
            if cfg in done:
                continue
            pools = budget_pools(X, c, B)
            p, fb = eval_budget(ctx, S, pm_te, pools)
            r = {"config": cfg, "yontem": yn, "B_pct": B_pct, "B": B,
                 **full_metrics(ctx, p, fb, pools)}
            rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
            print(cfg, round(r["mae"], 4), round(r["ndcg10"], 4), flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(RESULTS / "budget_pool.csv")
    renk = {"B0": "tab:gray", "AVOA": "tab:red", "GWO": "tab:blue",
            "NGO": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, t in [(axes[0], "mae", "Test MAE (dusuk iyi)"),
                       (axes[1], "ndcg10", "NDCG@10 (yuksek iyi)")]:
        for y, g in df.groupby("yontem"):
            g = g.sort_values("B_pct")
            ax.plot(g.B_pct, g[col], "o-", color=renk[y], label=y, lw=2)
        ax.set_xlabel("Havuz butcesi (%)"); ax.set_title(t)
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle("Esit butceli havuz kiyasi — tek degisken kumeleme kalitesi (K=20, seed 42)")
    fig.tight_layout()
    fig.savefig(RESULTS / "plots" / "budget_pool.png", dpi=150)
    print("Grafik -> plots/budget_pool.png")


if __name__ == "__main__":
    main()
