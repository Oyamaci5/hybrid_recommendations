"""
ML-1M REPAIR ABLASYONU — ayni merkezler, atama serbest vs kapasiteli.

Soru: "repair olmadan MAE ne olur?"
Ayni AVOA/B0 merkezleriyle iki atama: serbest Voronoi ve kapasiteli (repair).
Kume boyutlari, havuz ve tum metrikler yan yana raporlanir.

Kullanim: python algo_selection_v2/ml1m_repair_ablasyon.py [--k 6] [--max-fe 400]
Cikti: results/ml1m_repair_ablasyon.csv
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
import ml1m_run as m  # noqa: E402
from ctx_ml1m import Ctx1M  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402


def free_assign(X, cents, X2=None):
    d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).astype(np.int32), np.argsort(d, 1)[:, :2].astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-fe", type=int, default=400)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    args = ap.parse_args()

    ctx = Ctx1M(Path(args.data_dir), args.fold)
    X = m.build_space(ctx, None, "nmf")
    K = args.k
    cm.K = K
    km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
    c0 = km.cluster_centers_
    cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
    print("AVOA optimizasyonu...", flush=True)
    cav = m.solve_warm(cls, m.make_fitness(ctx, X, K), X, K, args.seed, c0,
                       max_fe=args.max_fe, pop=10)

    orig = m.fast_repair
    rows = []
    for mod in ("repairSIZ", "repairLI"):
        m.fast_repair = free_assign if mod == "repairSIZ" else orig
        for ad, cent in (("B0", c0), ("AVOA", cav)):
            L, _ = m.fast_repair(X, cent)
            sz = np.sort(np.bincount(L, minlength=K))[::-1]
            p, mm = m.evaluate(ctx, X, cent, K)
            rows.append({"mod": mod, "yontem": ad, "K": K,
                         "mae": mm["mae"], "rmse": mm["rmse"],
                         "ndcg10": mm["ndcg10"], "prec10": mm["prec10"],
                         "havuz": mm["havuz"], "maxk": int(sz[0]),
                         "mink": int(sz[-1]), "beta": mm["beta"]})
            pd.DataFrame(rows).to_csv(RESULTS / "ml1m_repair_ablasyon.csv",
                                      index=False)
            print(f"{mod:9s} {ad:4s} MAE={mm['mae']:.4f} NDCG={mm['ndcg10']:.4f} "
                  f"havuz={mm['havuz']} max/min={sz[0]}/{sz[-1]}", flush=True)
    m.fast_repair = orig


if __name__ == "__main__":
    main()
