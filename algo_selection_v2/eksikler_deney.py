"""
EKSIKLER DENEY SETI — seed 42, fold 1, tahminci = kume-ici kNN (cknn).

Deneyler (--exp):
  k_sweep : K in {5,7,10,14,20} x {B0_KMEANS++, AVOA pred-fit} + knn_all referansi.
            "K'yi neden sectik" sorusuna cift kanit: ic kalite (silhouette) +
            dis kalite (test MAE/NDCG).
  nfe     : AVOA pred-fit, K sabit, NFE in {500,1000,2000,5000,10000}.
            "NFE'yi neden artirdik / hangisi yeterli" doyum egrisi.
  lof     : LOF gray-sheep cikarma acik/kapali (kumeleme LOF-temiz veriyle
            fit edilir, herkes tahmin alir). "LOF gerekli mi" cevabi.

Metrikler: MAE, RMSE, Precision@10, Recall@10, NDCG@10 (esik: rating>=4, binary),
fallback%. NDCG/Recall yalnizca >=1 ilgili test filmi olan kullanicilarda.

Kullanim: python algo_selection_v2/eksikler_deney.py --exp k_sweep [--resume]
Cikti:   results/eksikler_<exp>.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from cf_stage import load_fold, to_matrix  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402

N_U, N_I = 943, 1682
SEED, DIM, KNN_K, TOPN, THR = 42, 20, 30, 10, 4.0


class Ctx:
    """Fold-1 baglami: ic-train istatistikleri, ozellikler, benzerlik."""

    def __init__(self, data_dir: Path, fold: int = 1):
        tu, ti, tr, self.eu, self.ei, self.er = load_fold(data_dir, fold)
        rng = np.random.default_rng(7)
        val = rng.choice(len(tr), len(tr) // 10, replace=False)
        m = np.zeros(len(tr), bool); m[val] = True
        self.iu, self.ii, self.ir = tu[~m], ti[~m], tr[~m]
        self.vu, self.vi, self.vr = tu[m], ti[m], tr[m]
        self.R = to_matrix(self.iu, self.ii, self.ir)
        rated = self.R > 0
        self.gmean = float(self.ir.mean())
        cnt_u = rated.sum(1)
        self.um = np.where(cnt_u > 0, self.R.sum(1) / np.maximum(cnt_u, 1), self.gmean)
        self.dev_g = np.zeros(N_I)
        np.add.at(self.dev_g, self.ii, self.ir - self.um[self.iu])
        self.dev_g /= np.maximum(np.bincount(self.ii, None, N_I), 1)
        self.X = np.ascontiguousarray(NMF(DIM, init="nndsvda", max_iter=400,
                                          random_state=42).fit_transform(self.R))
        Rc = np.where(rated, self.R - self.um[:, None], 0.0)
        nrm = np.linalg.norm(Rc, axis=1); nrm[nrm < 1e-9] = 1.0
        self.S = (Rc / nrm[:, None]) @ (Rc / nrm[:, None]).T
        np.fill_diagonal(self.S, 0.0)
        self.raters = [np.flatnonzero(rated[:, i]) for i in range(N_I)]
        self.by_user = [[] for _ in range(N_U)]
        for j, u in enumerate(self.eu):
            self.by_user[u].append(j)

    # ---- tahmin: kume-ici kNN (labels=None -> tum havuz) ----
    def preds_cknn(self, labels):
        mem = ([np.flatnonzero(labels == c) for c in range(int(labels.max()) + 1)]
               if labels is not None else None)
        p = np.empty(len(self.eu)); fb = 0
        for j, (u, i) in enumerate(zip(self.eu, self.ei)):
            cand = self.raters[i]
            if mem is not None:
                cand = cand[np.isin(cand, mem[labels[u]])]
            if len(cand):
                s = self.S[u, cand]
                top = np.argsort(-s)[:KNN_K]
                s, cc = s[top], cand[top]
                w = np.abs(s).sum()
                if w > 1e-9:
                    p[j] = self.um[u] + (s * (self.R[cc, i] - self.um[cc])).sum() / w
                    continue
            p[j] = self.um[u] + self.dev_g[i]; fb += 1
        return np.clip(p, 1, 5), 100 * fb / len(p)

    def metrics(self, labels):
        p, fb = self.preds_cknn(labels)
        e = p - self.er
        out = {"mae": float(np.abs(e).mean()),
               "rmse": float(np.sqrt((e ** 2).mean())), "fallback_pct": round(fb, 2)}
        precs, recs, ndcgs = [], [], []
        for u in range(N_U):
            idxs = np.array(self.by_user[u], dtype=int)
            if len(idxs) == 0:
                continue
            order = idxs[np.argsort(-p[idxs])]
            rel = (self.er[order] >= THR).astype(float)
            n = min(TOPN, len(order))
            precs.append(rel[:n].sum() / TOPN)
            nrel = int((self.er[idxs] >= THR).sum())
            if nrel:
                recs.append(rel[:n].sum() / nrel)
                disc = np.log2(np.arange(2, n + 2))
                ideal = np.sort(rel)[::-1][:n]
                idcg = (ideal / disc).sum()
                ndcgs.append(float((rel[:n] / disc).sum() / idcg) if idcg else 0.0)
        out.update({"prec10": float(np.mean(precs)), "rec10": float(np.mean(recs)),
                    "ndcg10": float(np.mean(ndcgs))})
        return out

    # ---- AVOA pred-fit (bias fitness, ic-val MAE) ----
    def avoa_labels(self, K, max_fe, mask=None):
        Xf = self.X if mask is None else self.X[mask]
        iu, ii, ir = self.iu, self.ii, self.ir
        if mask is not None:
            keep = mask[iu]
            iu, ii, ir = iu[keep], ii[keep], ir[keep]
            remap = -np.ones(N_U, int); remap[np.flatnonzero(mask)] = np.arange(mask.sum())
            iu_f = remap[iu]
        else:
            iu_f = iu
        lb, ub = np.tile(Xf.min(0), K), np.tile(Xf.max(0), K)

        def fit(sol):
            c = np.asarray(sol).reshape(K, DIM)
            lab_f = sse_wcss(Xf, c)[1]
            lab = lab_f if mask is None else sse_wcss(self.X, c)[1]
            dev = np.zeros((K, N_I)); cnt = np.zeros((K, N_I))
            np.add.at(dev, (lab_f[iu_f], ii), ir - self.um[iu])
            np.add.at(cnt, (lab_f[iu_f], ii), 1.0)
            with np.errstate(invalid="ignore"):
                dev = np.where(cnt > 0, dev / np.maximum(cnt, 1), np.nan)
            d = dev[lab[self.vu], self.vi]
            d = np.where(np.isnan(d), self.dev_g[self.vi], d)
            mae = float(np.abs(np.clip(self.um[self.vu] + d, 1, 5) - self.vr).mean())
            return mae + 0.1 * (K - len(np.unique(lab_f)))

        cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
        pos, fv, _, nfe = solve_meta(cls, fit, lb, ub, 2000, 30, SEED, max_fe=max_fe)
        cent = np.asarray(pos).reshape(K, DIM)
        return sse_wcss(self.X, cent)[1], fv, nfe


def run(args):
    ctx = Ctx(Path(args.data_dir))
    out = RESULTS / f"eksikler_{args.exp}.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(old.config)

    def add(config, extra, labels):
        if config in done:
            return
        t0 = time.time()
        r = {"config": config, **extra, **ctx.metrics(labels),
             "eval_s": round(time.time() - t0, 1)}
        if labels is not None and len(np.unique(labels)) > 1:
            r["silhouette"] = round(float(
                silhouette_score(ctx.X, labels, metric="euclidean")), 4)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(config, {k: round(v, 4) for k, v in r.items()
                       if isinstance(v, float)}, flush=True)

    if args.exp == "k_sweep":
        add("knn_all", {"K": 0, "yontem": "kNN(kumesiz)"}, None)
        for K in [5, 7, 10, 14, 20]:
            km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(ctx.X)
            add(f"B0_K{K}", {"K": K, "yontem": "B0"}, km.labels_)
            if f"AVOA_K{K}" not in done:
                lab, fv, nfe = ctx.avoa_labels(K, 2000)
                add(f"AVOA_K{K}", {"K": K, "yontem": "AVOA", "fit_val": fv,
                                   "nfe": nfe}, lab)

    elif args.exp == "nfe":
        K = args.k
        for max_fe in [500, 1000, 2000, 5000, 10000]:
            if f"AVOA_nfe{max_fe}" in done:
                continue
            t0 = time.time()
            lab, fv, nfe = ctx.avoa_labels(K, max_fe)
            add(f"AVOA_nfe{max_fe}", {"K": K, "yontem": "AVOA", "max_fe": max_fe,
                                      "fit_val": fv, "opt_s": round(time.time()-t0, 1)},
                lab)

    elif args.exp == "lof":
        K = args.k
        lof = LocalOutlierFactor(n_neighbors=20).fit_predict(ctx.X)
        inlier = lof == 1
        print(f"LOF: {int((~inlier).sum())} aykiri kullanici ({100*(~inlier).mean():.1f}%)")
        km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(ctx.X)
        add(f"B0_K{K}_lofsuz", {"K": K, "yontem": "B0", "lof": 0}, km.labels_)
        km2 = KMeans(K, init="k-means++", n_init=10,
                     random_state=SEED).fit(ctx.X[inlier])
        lab2 = sse_wcss(ctx.X, km2.cluster_centers_)[1]
        add(f"B0_K{K}_lof", {"K": K, "yontem": "B0", "lof": 1,
                             "aykiri": int((~inlier).sum())}, lab2)
        if f"AVOA_K{K}_lofsuz" not in done:
            lab, fv, nfe = ctx.avoa_labels(K, 2000)
            add(f"AVOA_K{K}_lofsuz", {"K": K, "yontem": "AVOA", "lof": 0,
                                      "fit_val": fv}, lab)
        if f"AVOA_K{K}_lof" not in done:
            lab, fv, nfe = ctx.avoa_labels(K, 2000, mask=inlier)
            add(f"AVOA_K{K}_lof", {"K": K, "yontem": "AVOA", "lof": 1,
                                   "fit_val": fv}, lab)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["k_sweep", "nfe", "lof"], required=True)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-100k"))
    ap.add_argument("--resume", action="store_true")
    run(ap.parse_args())
