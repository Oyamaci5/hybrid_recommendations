"""
TAHMINCI YUKSELTME DENEYI — MAE farki kumelemede degil tahminci katmaninda mi?

Ayni kumeleme, uc tahminci:
  P0 cmean : kume-film ortalamasi (cf_stage'deki pilot tahminci)
  P1 bias  : kullanici ort. + kume-ici film sapmasi  (Resnick'in agirliksiz hali)
  P2 cknn  : kume-ici kullanici-kNN (cosine, ortalama-merkezli, top-k)
Referanslar: item-mean, user-mean, bias(global), kNN(tum kullanicilar, kumesiz).

Protokol cf_stage ile ayni: fold1, ic-train %90 (NMF-20 + istatistikler), test=u1.test.
Seed 42. Cikti: results/predictor_upgrade.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from cf_stage import cluster_item_means, load_fold, to_matrix  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402

N_U, N_I = 943, 1682
K, DIM, SEED, KNN = 7, 20, 42, 30


def mae_rmse(p, r):
    p = np.clip(p, 1, 5)
    return float(np.abs(p - r).mean()), float(np.sqrt(((p - r) ** 2).mean()))


def main():
    data = Path(ROOT.parent / "data" / "ml-100k")
    tu, ti, tr, eu, ei, er = load_fold(data, 1)
    rng = np.random.default_rng(7)
    val_idx = rng.choice(len(tr), len(tr) // 10, replace=False)
    vmask = np.zeros(len(tr), bool); vmask[val_idx] = True
    iu, ii, ir = tu[~vmask], ti[~vmask], tr[~vmask]
    vu, vi, vr = tu[vmask], ti[vmask], tr[vmask]

    R = to_matrix(iu, ii, ir)
    rated = R > 0
    gmean = float(ir.mean())
    with np.errstate(invalid="ignore"):
        um = np.where(rated.sum(1) > 0, R.sum(1) / np.maximum(rated.sum(1), 1), gmean)
        im = np.divide(np.bincount(ii, ir, N_I),
                       np.maximum(np.bincount(ii, None, N_I), 1))
        im = np.where(np.bincount(ii, None, N_I) > 0, im, gmean)
    dev_g = np.zeros(N_I)          # global film sapmasi (bias referansi icin)
    cnt_g = np.maximum(np.bincount(ii, None, N_I), 1)
    np.add.at(dev_g, ii, ir - um[iu])
    dev_g = dev_g / cnt_g

    X = np.ascontiguousarray(
        NMF(DIM, init="nndsvda", max_iter=400, random_state=42).fit_transform(R))

    # --- kullanici-kullanici cosine (ortalama-merkezli, ic-train) — kumeden bagimsiz
    Rc = np.where(rated, R - um[:, None], 0.0)
    nrm = np.linalg.norm(Rc, axis=1); nrm[nrm < 1e-9] = 1.0
    S = (Rc / nrm[:, None]) @ (Rc / nrm[:, None]).T
    np.fill_diagonal(S, 0.0)
    raters = [np.flatnonzero(rated[:, i]) for i in range(N_I)]

    def knn_pred(u, i, members=None):
        cand = raters[i]
        if members is not None:
            cand = cand[np.isin(cand, members)]
        if len(cand) == 0:
            return np.nan
        s = S[u, cand]
        top = np.argsort(-s)[:KNN]
        s, cand = s[top], cand[top]
        w = np.abs(s).sum()
        if w < 1e-9:
            return np.nan
        return um[u] + float((s * (R[cand, i] - um[cand])).sum() / w)

    def eval_knn(labels=None):
        mem = ([np.flatnonzero(labels == c) for c in range(K)]
               if labels is not None else None)
        p = np.array([knn_pred(u, i, mem[labels[u]] if mem else None)
                      for u, i in zip(eu, ei)])
        fb = np.isnan(p)
        p = np.where(fb, um[eu], p)
        return *mae_rmse(p, er), float(fb.mean())

    def eval_cmean(labels):
        M = cluster_item_means(labels, iu, ii, ir, K)
        p = M[labels[eu], ei]
        p = np.where(np.isnan(p), im[ei], p)
        return *mae_rmse(p, er), float(np.isnan(M[labels[eu], ei]).mean())

    def eval_bias(labels):
        dev = np.zeros((K, N_I)); cnt = np.zeros((K, N_I))
        np.add.at(dev, (labels[iu], ii), ir - um[iu])
        np.add.at(cnt, (labels[iu], ii), 1.0)
        with np.errstate(invalid="ignore"):
            dev = np.where(cnt > 0, dev / np.maximum(cnt, 1), np.nan)
        d = dev[labels[eu], ei]
        fb = np.isnan(d)
        d = np.where(fb, dev_g[ei], d)
        return *mae_rmse(um[eu] + d, er), float(fb.mean())

    rows = []

    def add(name, pred, mae, rmse, fb, note=""):
        rows.append({"clustering": name, "predictor": pred, "mae": round(mae, 4),
                     "rmse": round(rmse, 4), "fallback": round(100 * fb, 2),
                     "note": note})
        print(f"{name:18s} {pred:8s} MAE={mae:.4f} RMSE={rmse:.4f} fb={100*fb:.1f}%")

    # --- referanslar (kumesiz) ---
    add("-", "item_mean", *mae_rmse(im[ei], er), 0)
    add("-", "user_mean", *mae_rmse(um[eu], er), 0)
    add("-", "bias_glob", *mae_rmse(um[eu] + dev_g[ei], er), 0)
    t0 = time.time()
    add("-", "knn_all", *eval_knn(None), f"{time.time()-t0:.0f}s")

    # --- kumelemeler (seed 42) ---
    labels_sets = {}
    km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
    labels_sets["B0_KMEANS++"] = km.labels_

    gsse = float(((X - X.mean(0)) ** 2).sum())
    lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)

    def fit_pred_bias(sol):
        c = np.asarray(sol).reshape(K, DIM)
        lab = sse_wcss(X, c)[1]
        dev = np.zeros((K, N_I)); cnt = np.zeros((K, N_I))
        np.add.at(dev, (lab[iu], ii), ir - um[iu])
        np.add.at(cnt, (lab[iu], ii), 1.0)
        with np.errstate(invalid="ignore"):
            dev = np.where(cnt > 0, dev / np.maximum(cnt, 1), np.nan)
        d = dev[lab[vu], vi]
        d = np.where(np.isnan(d), dev_g[vi], d)
        mae = float(np.abs(np.clip(um[vu] + d, 1, 5) - vr).mean())
        return mae + 0.1 * (K - len(np.unique(lab)))

    for name in ["GWO.OriginalGWO", "AVOA.OriginalAVOA"]:
        cls = discover_algorithms([name])[name]
        pos, *_ = solve_meta(cls, fit_pred_bias, lb, ub, 500, 30, SEED, max_fe=2000)
        labels_sets[name.split(".")[0] + "_predfit"] = sse_wcss(
            X, np.asarray(pos).reshape(K, DIM))[1]

    for cname, lab in labels_sets.items():
        add(cname, "cmean", *eval_cmean(lab))
        add(cname, "bias", *eval_bias(lab))
        t0 = time.time()
        add(cname, "cknn", *eval_knn(lab), f"{time.time()-t0:.0f}s")

    pd.DataFrame(rows).to_csv(RESULTS / "predictor_upgrade.csv", index=False)
    print("\nKayit -> results/predictor_upgrade.csv")


if __name__ == "__main__":
    main()
