"""
ML-1M TAHMINCI ABLASYONU — ayni kumeler, dort tahminci.

  cmean : kume-film ortalamasi (literaturun standart tercihi)
  bias  : kullanici ort. + kume-ici film sapmasi
  cknn  : kume-ici kNN (k=20)         [bizim bilesenimiz]
  karisim: beta*cknn + (1-beta)*kumeMF [onerilen]

Amac: "neden kume-ortalamasi degil" sorusunu ML-1M'de de cevaplamak.
Kullanim: python algo_selection_v2/ml1m_tahminci_ablasyon.py --k 40
Cikti: results/ml1m_tahminci.csv
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
from ctx_ml1m import N_I, Ctx1M  # noqa: E402
from pred_v2 import full_metrics  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402


def tahminler(ctx, X, cent, K):
    L, near = m.fast_repair(X, cent)
    flat_tr = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
    # --- cmean: kume-film ortalamasi ---
    s = np.bincount(flat_tr, weights=ctx.ir, minlength=K * N_I)
    n = np.bincount(flat_tr, minlength=K * N_I)
    idx_e = L[ctx.eu].astype(np.int64) * N_I + ctx.ei
    im = np.bincount(ctx.ii, weights=ctx.ir, minlength=N_I) / np.maximum(
        np.bincount(ctx.ii, minlength=N_I), 1)
    p_cmean = np.where(n[idx_e] > 0, s[idx_e] / np.maximum(n[idx_e], 1), im[ctx.ei])
    # --- bias: kullanici ort + kume sapmasi (soft top-2) ---
    sd = np.bincount(flat_tr, weights=ctx.ir - ctx.um[ctx.iu], minlength=K * N_I)
    a = near[ctx.eu, 0].astype(np.int64) * N_I + ctx.ei
    b = near[ctx.eu, 1].astype(np.int64) * N_I + ctx.ei
    num, den = sd[a] + sd[b], n[a] + n[b]
    dev = np.where(den > 0, num / np.maximum(den, 1), ctx.dev_g[ctx.ei])
    p_bias = ctx.um[ctx.eu] + dev
    # --- cknn ---
    p_knn = m.knn_pool(ctx, L, near, ctx.eu, ctx.ei)
    fb = np.isnan(p_knn)
    p_cknn = np.where(fb, p_bias, np.nan_to_num(p_knn))
    # --- karisim (mevcut yontem) ---
    _, mm = m.evaluate(ctx, X, cent, K)
    return ({"cmean": p_cmean, "bias": p_bias, "cknn": p_cknn},
            float(fb.mean()) * 100, mm)


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
    K = args.k; cm.K = K
    km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
    c0 = km.cluster_centers_
    cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
    cav = m.solve_warm(cls, m.make_fitness(ctx, X, K), X, K, args.seed, c0,
                       max_fe=args.max_fe, pop=10)
    rows = []
    for ad, cent in (("B0", c0), ("AVOA", cav)):
        preds, fbp, mm = tahminler(ctx, X, cent, K)
        for tn, p in preds.items():
            p = np.clip(p, 1, 5)
            r = {"yontem": ad, "tahminci": tn,
                 **full_metrics(ctx, p, fbp, [np.arange(1)])}
            r.pop("havuz_ort", None)
            rows.append(r)
            print(f"{ad:5s} {tn:8s} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} "
                  f"NDCG={r['ndcg10']:.4f}", flush=True)
        rows.append({"yontem": ad, "tahminci": "karisim(onerilen)",
                     "mae": mm["mae"], "rmse": mm["rmse"],
                     "ndcg10": mm["ndcg10"], "prec10": mm["prec10"],
                     "rec10": mm["rec10"], "fallback_pct": mm["fallback_pct"]})
        print(f"{ad:5s} {'karisim':8s} MAE={mm['mae']:.4f} RMSE={mm['rmse']:.4f} "
              f"NDCG={mm['ndcg10']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "ml1m_tahminci.csv", index=False)


if __name__ == "__main__":
    main()
