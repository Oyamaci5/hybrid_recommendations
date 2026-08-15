"""
REPAIR PROTOKOLUNDE TAM K TARAMASI + EN IYI NOKTALARDA ALGORITMA KIYASI.

Protokol: repair (kapasite = ceil(N/K), kisit ihlali imkansiz) + warm start,
nmf+genre, soft top-2 havuz, kume-MF f=10, kNN k=20, beta ic-val'de.

--exp lowk   : K = 2..10 (Sistem A bandi — mutlak performans)   B0 + AVOA
--exp highk  : K = 10..70 onar (Sistem B bandi — algoritma farki) B0 + AVOA
--exp algos  : verilen K'larda TUM algoritmalar (--klist ile)
Cikti: results/k_tarama_repair.csv   (+ --plot ile grafik)

Ornek:
  python algo_selection_v2/k_tarama_repair.py --exp lowk --resume
  python algo_selection_v2/k_tarama_repair.py --exp highk --resume
  python algo_selection_v2/k_tarama_repair.py --exp algos --klist 5 30 --resume
  python algo_selection_v2/k_tarama_repair.py --plot
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
import tabloB_plus as bp  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402

ALL_ALGOS = ["AVOA.OriginalAVOA", "GWO.OriginalGWO", "HGS.OriginalHGS",
             "NGO.OriginalNGO", "HHO.OriginalHHO"]
OUT = RESULTS / "k_tarama_repair.csv"


def run_point(ctx, X, S, K, yontem, cls=None, seed=42, max_fe=1200):
    bp.K = K; cm.K = K
    km = KMeans(K, init="k-means++", n_init=10, random_state=seed).fit(X)
    c0 = km.cluster_centers_
    if yontem == "B0":
        _, m = bp.evaluate(ctx, X, S, c0)
        return m
    from mealpy import FloatVar
    lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
    rng = np.random.default_rng(seed)
    starts = np.clip(np.vstack(
        [c0.ravel()] + [c0.ravel() + rng.normal(0, 0.08 * X.std(), c0.size)
                        for _ in range(14)]), lb, ub)
    g = cls(epoch=3000, pop_size=15).solve(
        {"obj_func": bp.make_fitness(ctx, X), "bounds": FloatVar(lb=lb, ub=ub),
         "minmax": "min", "log_to": None},
        seed=seed, termination={"max_fe": max_fe}, starting_solutions=starts)
    _, m = bp.evaluate(ctx, X, S, np.asarray(g.solution).reshape(K, X.shape[1]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["lowk", "highk", "algos"])
    ap.add_argument("--klist", type=int, nargs="+", default=None)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fe", type=int, default=1200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        return plot()

    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data, args.fold)
    X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
    S = build_sims(ctx)

    rows, done = [], set()
    if args.resume and OUT.exists() and OUT.stat().st_size > 10:
        old = pd.read_csv(OUT); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K, old.fold, old.seed))

    if args.exp == "lowk":
        ks, algos = list(range(2, 11)), ["AVOA.OriginalAVOA"]
    elif args.exp == "highk":
        ks, algos = list(range(10, 71, 10)), ["AVOA.OriginalAVOA"]
    else:
        ks, algos = (args.klist or [5, 30]), ALL_ALGOS
    cls_map = discover_algorithms(algos)

    for K in ks:
        for yontem in ["B0"] + [a.split(".")[0] for a in algos]:
            if (yontem, K, args.fold, args.seed) in done:
                continue
            cls = cls_map.get(f"{yontem}.Original{yontem}") if yontem != "B0" else None
            m = run_point(ctx, X, S, K, yontem, cls, args.seed, args.max_fe)
            rows.append({"yontem": yontem, "K": K, "fold": args.fold,
                         "seed": args.seed, **m})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"K={K:>2} {yontem:5s} MAE={m['mae']:.4f} NDCG={m['ndcg10']:.4f} "
                  f"P@10={m['prec10']:.4f} havuz={m['havuz']}", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(OUT).sort_values("K")
    renk = {"B0": "tab:gray", "AVOA": "tab:red", "GWO": "tab:blue",
            "HGS": "tab:orange", "NGO": "tab:green", "HHO": "tab:purple"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for y, g in df.groupby("yontem"):
        g = g.sort_values("K")
        ax[0].plot(g.K, g.mae, "o-", color=renk.get(y), label=y, ms=4)
        ax[1].plot(g.K, g.ndcg10, "o-", color=renk.get(y), label=y, ms=4)
        ax[2].plot(g.havuz, g.mae, "o", color=renk.get(y), label=y, ms=5)
    ax[0].axhline(0.7467, ls="--", c="k", lw=1, label="kumesiz kNN")
    ax[1].axhline(0.8352, ls="--", c="k", lw=1)
    ax[0].set_xlabel("K"); ax[0].set_ylabel("MAE"); ax[0].set_title("MAE vs K")
    ax[1].set_xlabel("K"); ax[1].set_ylabel("NDCG@10"); ax[1].set_title("NDCG vs K")
    ax[2].set_xlabel("havuz (kullanici)"); ax[2].set_ylabel("MAE")
    ax[2].set_title("Dogruluk-maliyet cephesi")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle("Repair protokolu — K taramasi (fold 1, seed 42)")
    fig.tight_layout()
    fp = RESULTS / "plots" / "k_tarama_repair.png"
    fig.savefig(fp, dpi=150)
    print("Grafik ->", fp)


if __name__ == "__main__":
    main()
