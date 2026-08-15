"""
E1 — COVERAGE (kapsama) metrikleri. ML-1M, K=40, tek fold/seed.

Olculenler (literaturde standart, bizde yoktu):
  catalog_cov  : Top-10 listelerinde en az bir kez gecen farkli film orani
  gini         : oneri dagiliminin esitsizligi (0=esit, 1=tek film)
  novelty      : ortalama -log2(populerlik) (yuksek = daha az bilinen film)
  user_cov     : tahmin uretilebilen kullanici orani (fallback disi)

Kullanim: python algo_selection_v2/e1_coverage.py [--k 40]
Cikti: results/ml1m_coverage.csv
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
from ctx_ml1m import N_I, N_U, Ctx1M  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402

TOPN = 10


def coverage_metrics(ctx, p, pop):
    """Top-N listelerinden kapsama/cesitlilik metrikleri."""
    sayac = np.zeros(N_I)
    nov = []
    for u in range(N_U):
        idx = ctx.by_user[u]
        if not idx:
            continue
        i = np.array(idx)
        top = i[np.argsort(-p[i])][:TOPN]
        films = ctx.ei[top]
        sayac[films] += 1
        nov.append(np.mean(-np.log2(np.maximum(pop[films], 1) / pop.sum())))
    onerilen = sayac > 0
    q = np.sort(sayac[onerilen])
    n = len(q)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ q / (n * q.sum())) if n else 0
    return {"catalog_cov": float(onerilen.sum() / N_I),
            "farkli_film": int(onerilen.sum()),
            "gini": round(gini, 4),
            "novelty": float(np.mean(nov))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fe", type=int, default=600)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    args = ap.parse_args()

    ctx = Ctx1M(Path(args.data_dir), args.fold)
    X = m.build_space(ctx, None, "nmf")
    K = args.k
    cm.K = K
    pop = np.bincount(ctx.ii, minlength=N_I).astype(float)   # egitim populerligi
    km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
    c0 = km.cluster_centers_
    fitf = m.make_fitness(ctx, X, K)
    algos = discover_algorithms(m.ALGOS)
    rows = []
    for ad, cent in [("B0_repair", c0)] + [
            (a.split(".")[0] + "_warm", None) for a in m.ALGOS]:
        if cent is None:
            cls = algos[[a for a in m.ALGOS if a.startswith(ad.split("_")[0])][0]]
            cent = m.solve_warm(cls, fitf, X, K, args.seed, c0,
                                max_fe=args.max_fe, pop=10)
        p, mm = m.evaluate(ctx, X, cent, K)
        r = {"yontem": ad, "K": K, "mae": mm["mae"], "ndcg10": mm["ndcg10"],
             "fallback_pct": mm["fallback_pct"], **coverage_metrics(ctx, p, pop)}
        rows.append(r)
        pd.DataFrame(rows).to_csv(RESULTS / "ml1m_coverage.csv", index=False)
        print(f"{ad:11s} MAE={r['mae']:.4f} cov={r['catalog_cov']:.3f} "
              f"({r['farkli_film']} film) gini={r['gini']:.3f} "
              f"novelty={r['novelty']:.2f}", flush=True)


if __name__ == "__main__":
    main()
