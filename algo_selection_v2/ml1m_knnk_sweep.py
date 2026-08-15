"""
ML-1M kNN KOMSU SAYISI (k) TARAMASI — literaturle kiyaslanabilir aralikta.

Literaturde gorulen k degerleri: 3 (GOA), 5-60 (Katarya PSO), 70 (HSC Sparrow),
90 (Firefly). Bizde k=20 sabitti; burada taranarak dogrulanir.

Taranan: k in {5, 10, 20, 30, 50, 70}
Yontemler: B0 (KMeans++) ve AVOA; tahminci = 0.5*kNN + 0.5*kumeMF (ayarsiz).
Kullanim: python algo_selection_v2/ml1m_knnk_sweep.py --k 40 [--kcluster 6]
Cikti: results/ml1m_knnk.csv
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

KNN_LISTE = [5, 10, 20, 30, 50, 70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40, help="kume sayisi K")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fe", type=int, default=600)
    ap.add_argument("--knn-list", type=int, nargs="+", default=KNN_LISTE)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    ap.add_argument("--resume", action="store_true")
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

    out = RESULTS / "ml1m_knnk.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K, old.knn_k))

    for ad, cent in (("B0", c0), ("AVOA", cav)):
        L, near = m.fast_repair(X, cent)
        models = cm.cluster_mf_models(ctx, L)
        # bias fallback + kume-MF (kNN'den bagimsiz, bir kez)
        flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
        n = np.bincount(flat, minlength=K * N_I)
        sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu],
                         minlength=K * N_I)
        a = near[ctx.eu, 0].astype(np.int64) * N_I + ctx.ei
        b = near[ctx.eu, 1].astype(np.int64) * N_I + ctx.ei
        den = n[a] + n[b]
        dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1),
                       ctx.dev_g[ctx.ei])
        p_bias = ctx.um[ctx.eu] + dev
        p_mf = cm.cmf_predict(models, L, ctx.eu, ctx.ei, p_bias)
        eski_k = m.KNN_K
        for kk in args.knn_list:
            if (ad, K, kk) in done:
                continue
            m.KNN_K = kk
            p_knn = m.knn_pool(ctx, L, near, ctx.eu, ctx.ei)
            fb = np.isnan(p_knn)
            p_knn = np.where(fb, p_bias, np.nan_to_num(p_knn))
            p = np.clip(0.5 * p_knn + 0.5 * p_mf, 1, 5)
            r = full_metrics(ctx, p, 100 * float(fb.mean()), [np.arange(1)])
            r.pop("havuz_ort", None)
            rows.append({"yontem": ad, "K": K, "knn_k": kk, **r})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{ad:4s} K={K} kNN_k={kk:2d}  MAE={r['mae']:.4f} "
                  f"RMSE={r['rmse']:.4f} NDCG={r['ndcg10']:.4f}", flush=True)
        m.KNN_K = eski_k


if __name__ == "__main__":
    main()
