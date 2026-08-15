"""
ML-1M TAHMINCI ABLASYONU-2 — iki soru:
  (a) Yalniz kume-MF (kNN yok, beta=0) ne veriyor?
  (b) Kume-ortalamasi repair'siz (dogal, dengesiz kumeler) daha mi iyi?

Meta optimizasyonu YOK (hizli): B0 = KMeans++ merkezleri; atama iki modda.
Tahminciler: cmean, bias, cknn, MF_tek, karisim(beta ic-val'de).

Kullanim: python algo_selection_v2/ml1m_tahminci2.py --k 40
Cikti: results/ml1m_tahminci2.csv
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


def free_assign(X, cents, X2=None):
    d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).astype(np.int32), np.argsort(d, 1)[:, :2].astype(np.int32)


def calistir(ctx, X, cent, K, mod):
    L, near = (free_assign(X, cent) if mod == "repairSIZ"
               else m.fast_repair(X, cent))
    sz = np.bincount(L, minlength=K)
    havuz = int(np.mean([(np.isin(L, near[u])).sum()
                         for u in range(0, len(X), 20)]))
    flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
    s = np.bincount(flat, weights=ctx.ir, minlength=K * N_I)
    n = np.bincount(flat, minlength=K * N_I)
    sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu], minlength=K * N_I)
    im = np.bincount(ctx.ii, weights=ctx.ir, minlength=N_I) / np.maximum(
        np.bincount(ctx.ii, minlength=N_I), 1)

    def pred_cmean(us, is_):
        j = L[us].astype(np.int64) * N_I + is_
        return np.where(n[j] > 0, s[j] / np.maximum(n[j], 1), im[is_])

    def pred_bias(us, is_):
        a = near[us, 0].astype(np.int64) * N_I + is_
        b = near[us, 1].astype(np.int64) * N_I + is_
        den = n[a] + n[b]
        dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1),
                       ctx.dev_g[is_])
        return ctx.um[us] + dev

    cm.K = K
    models = cm.cluster_mf_models(ctx, L)
    mf_v = cm.cmf_predict(models, L, ctx.vu, ctx.vi, pred_bias(ctx.vu, ctx.vi))
    mf_e = cm.cmf_predict(models, L, ctx.eu, ctx.ei, pred_bias(ctx.eu, ctx.ei))
    knn_v = m.knn_pool(ctx, L, near, ctx.vu, ctx.vi)
    knn_e = m.knn_pool(ctx, L, near, ctx.eu, ctx.ei)
    fbp = 100 * float(np.isnan(knn_e).mean())
    knn_e_dolu = np.where(np.isnan(knn_e), pred_bias(ctx.eu, ctx.ei),
                          np.nan_to_num(knn_e))
    best_b, best = None, np.inf
    for b in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        pv = np.where(np.isnan(knn_v), mf_v,
                      b * np.nan_to_num(knn_v) + (1 - b) * mf_v)
        v = float(np.abs(np.clip(pv, 1, 5) - ctx.vr).mean())
        if v < best:
            best_b, best = b, v
    kar = np.where(np.isnan(knn_e), mf_e,
                   best_b * np.nan_to_num(knn_e) + (1 - best_b) * mf_e)
    out = []
    for ad, p in (("cmean", pred_cmean(ctx.eu, ctx.ei)),
                  ("bias", pred_bias(ctx.eu, ctx.ei)),
                  ("cknn", knn_e_dolu),
                  ("MF_tek", mf_e),
                  (f"karisim(b={best_b})", kar)):
        p = np.clip(p, 1, 5)
        r = full_metrics(ctx, p, fbp, [np.arange(1)])
        r.pop("havuz_ort", None)
        out.append({"mod": mod, "tahminci": ad, "havuz": havuz,
                    "maxk": int(sz.max()), "mink": int(sz.min()), **r})
        print(f"{mod:9s} {ad:14s} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} "
              f"NDCG={r['ndcg10']:.4f} (maxk={sz.max()})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    args = ap.parse_args()
    ctx = Ctx1M(Path(args.data_dir), args.fold)
    X = m.build_space(ctx, None, "nmf")
    K = args.k
    km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
    rows = []
    for mod in ("repairLI", "repairSIZ"):
        rows += calistir(ctx, X, km.cluster_centers_, K, mod)
        pd.DataFrame(rows).to_csv(RESULTS / "ml1m_tahminci2.csv", index=False)


if __name__ == "__main__":
    main()
