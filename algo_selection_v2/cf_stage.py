"""
CF ASAMASI — merkezler tahmine donusuyor (ML-100K fold 1).

Karsilastirma eksenleri:
  1. Algoritma: AVOA / GWO / NGO  vs  B0_KMEANS++ / B0_RANDOM_LLOYD / B0_RANDOM_CENT
  2. Lloyd:     variant=meta (Lloyd YOK, ham meta merkezleri)
                variant=kmref (meta merkezleri -> Lloyd)
  3. Fitness:   wcss      (geometrik hedef — onceki asamalarla ayni)
                pred_mae  (merkez ararken dogrudan ic-validasyon MAE minimize edilir;
                           Lloyd'un OPTIMIZE EDEMEDIGI hedef — tezin ana hipotezi)

Sizinti onlemi: fold1 train %90 ic-train + %10 ic-val'e bolunur. NMF ozellikleri,
kume istatistikleri ve WCSS SADECE ic-train'den; pred_mae fitness'i ic-val'de
olculur; nihai skor u1.test'te. Test hicbir asamada modele girmez.

Tahminci (iki fitness icin de ayni): kullanicinin kumesindeki kullanicilarin o filme
verdigi ortalama puan; fallback: film ort. -> kullanici ort. -> global ort.
(clip 1-5, fallback %'si raporlanir)

Kullanim:  python algo_selection_v2/cf_stage.py [--fitness wcss|pred_mae] [--resume]
Cikti:     results/cf_stage.csv
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

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402

N_U, N_I = 943, 1682


def load_fold(data_dir: Path, fold: int):
    tr = pd.read_csv(data_dir / f"u{fold}.base", sep="\t", names=["u", "i", "r", "t"])
    te = pd.read_csv(data_dir / f"u{fold}.test", sep="\t", names=["u", "i", "r", "t"])
    return (tr.u.values - 1, tr.i.values - 1, tr.r.values.astype(float),
            te.u.values - 1, te.i.values - 1, te.r.values.astype(float))


def to_matrix(u, i, r):
    M = np.zeros((N_U, N_I))
    M[u, i] = r
    return M


def cluster_item_means(labels, u, i, r, K):
    s = np.zeros((K, N_I)); c = np.zeros((K, N_I))
    np.add.at(s, (labels[u], i), r)
    np.add.at(c, (labels[u], i), 1.0)
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def predict(labels, M, item_mean, user_mean, gmean, eu, ei):
    p = M[labels[eu], ei]
    fb = np.isnan(p)
    p = np.where(fb, item_mean[ei], p)
    p = np.where(np.isnan(p), user_mean[eu], p)
    p = np.where(np.isnan(p), gmean, p)
    return np.clip(p, 1, 5), float(fb.mean())


def evaluate(labels, stats, eu, ei, er):
    M, item_mean, user_mean, gmean = stats
    p, fb = predict(labels, M, item_mean, user_mean, gmean, eu, ei)
    return {"mae": float(np.abs(p - er).mean()),
            "rmse": float(np.sqrt(((p - er) ** 2).mean())),
            "fallback_pct": round(100 * fb, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-100k"))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--fitness", choices=["wcss", "pred_mae"], default="wcss")
    ap.add_argument("--max-fe", type=int, default=None,
                    help="vars: wcss=15000, pred_mae=2000")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--algos", nargs="+",
                    default=["AVOA.OriginalAVOA", "GWO.OriginalGWO", "NGO.OriginalNGO"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    max_fe = args.max_fe or (15000 if args.fitness == "wcss" else 2000)

    tu, ti, tr, eu, ei, er = load_fold(Path(args.data_dir), args.fold)

    # ic-train / ic-val (%10) — test degil!
    rng = np.random.default_rng(7)
    val_idx = rng.choice(len(tr), len(tr) // 10, replace=False)
    vmask = np.zeros(len(tr), bool); vmask[val_idx] = True
    iu, ii, ir = tu[~vmask], ti[~vmask], tr[~vmask]     # ic-train
    vu, vi, vr = tu[vmask], ti[vmask], tr[vmask]        # ic-val

    R_inner = to_matrix(iu, ii, ir)
    X = NMF(args.dim, init="nndsvda", max_iter=400,
            random_state=42).fit_transform(R_inner)
    X = np.ascontiguousarray(X)
    gmean = float(ir.mean())
    with np.errstate(invalid="ignore"):
        im = np.divide(np.bincount(ii, ir, N_I), np.bincount(ii, None, N_I))
        um = np.divide(np.bincount(iu, ir, N_U), np.bincount(iu, None, N_U))
    K, D = args.k, args.dim
    lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
    gsse = float(((X - X.mean(0)) ** 2).sum())

    def labels_of(cent):
        return sse_wcss(X, cent)[1]

    def stats_of(labels):
        return (cluster_item_means(labels, iu, ii, ir, K), im, um, gmean)

    def fit_wcss(sol):
        c = np.asarray(sol).reshape(K, D)
        w, lab = sse_wcss(X, c)
        return w + (K - len(np.unique(lab))) * (gsse / K)

    def fit_pred(sol):
        c = np.asarray(sol).reshape(K, D)
        lab = labels_of(c)
        M = cluster_item_means(lab, iu, ii, ir, K)
        p, _ = predict(lab, M, im, um, gmean, vu, vi)
        mae = float(np.abs(p - vr).mean())
        return mae + 0.1 * (K - len(np.unique(lab)))

    obj = fit_wcss if args.fitness == "wcss" else fit_pred

    out = RESULTS / "cf_stage.csv"
    done, rows = set(), []
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out)
        rows = old.to_dict("records")
        done = set(zip(old.fitness, old.algorithm, old.seed, old.variant))

    def save():
        pd.DataFrame(rows).to_csv(out, index=False)

    # --- baseline'lar (fitness'tan bagimsiz; fitness=wcss kolunda bir kez) ---
    if args.fitness == "wcss":
        for seed in args.seeds:
            base = [("B0_KMEANS++", KMeans(K, init="k-means++", n_init=10,
                                           random_state=seed).fit(X), "kmref"),
                    ("B0_RANDOM_LLOYD", KMeans(K, init="random", n_init=1,
                                               random_state=seed).fit(X), "kmref")]
            for bname, km, var in base:
                if ("wcss", bname, seed, var) in done: continue
                lab = km.labels_
                rows.append({"fitness": "wcss", "algorithm": bname, "seed": seed,
                             "variant": var, "wcss": float(km.inertia_), "nfe": 0,
                             **evaluate(lab, stats_of(lab), eu, ei, er)})
            if ("wcss", "B0_RANDOM_CENT", seed, "meta") not in done:
                r2 = np.random.default_rng(seed)
                cent = X[r2.choice(len(X), K, replace=False)]
                w, lab = sse_wcss(X, cent)
                rows.append({"fitness": "wcss", "algorithm": "B0_RANDOM_CENT",
                             "seed": seed, "variant": "meta", "wcss": w, "nfe": 0,
                             **evaluate(lab, stats_of(lab), eu, ei, er)})
            save()

    # --- metalar: iki variant birden (meta ham + Lloyd rafine) ---
    algos = discover_algorithms(args.algos)
    for name, cls in algos.items():
        for seed in args.seeds:
            if all((args.fitness, name, seed, v) in done for v in ("meta", "kmref")):
                continue
            t0 = time.time()
            pos, fit, _, nfe = solve_meta(cls, obj, lb, ub, 500, 30, seed,
                                          max_fe=max_fe)
            cent = np.asarray(pos).reshape(K, D)
            lab_m = labels_of(cent)
            km = KMeans(K, init=cent, n_init=1, random_state=seed).fit(X)
            lab_r = km.labels_
            for var, lab, w in (("meta", lab_m, sse_wcss(X, cent)[0]),
                                ("kmref", lab_r, float(km.inertia_))):
                rows.append({"fitness": args.fitness, "algorithm": name,
                             "seed": seed, "variant": var, "wcss": w,
                             "nfe": nfe, "fit_val": fit,
                             "time_s": round(time.time() - t0, 1),
                             **evaluate(lab, stats_of(lab), eu, ei, er)})
            save()
            print(f"{args.fitness} {name} seed={seed} nfe={nfe} "
                  f"MAE(meta)={rows[-2]['mae']:.4f} MAE(kmref)={rows[-1]['mae']:.4f} "
                  f"({rows[-1]['time_s']}s)", flush=True)

    print("\nBitti ->", out)


if __name__ == "__main__":
    main()
