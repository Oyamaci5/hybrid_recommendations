
"""
HSC (Hybrid Sparrow Clustered) REPLIKASYONU + AVOA IKAMESI.

HSC protokolu (makaleden birebir):
  - Bolme: %80 egitim / %20 test (rastgele)
  - Uzay: HAM 943x1682 (veya 6040x3952) puan matrisi; eksikler 0
  - Merkez boyutu: item sayisi kadar (1682/3952), degerler [1,5] araliginda rastgele
  - Benzerlik: OKLID mesafesi
  - Fitness: WCSS (kume ici kare hatalar toplami)
  - Atama: serbest Voronoi (kisit yok)
  - Tahmin: kullanicinin kumesindeki tum kullanicilarin o filme verdigi ORTALAMA
  - K = 70

Deney kollari:
  HSC_SSA   : orijinal (Sparrow yerine mealpy SSA)
  HSC_AVOA  : ayni protokol, optimizasyon AVOA ile   <- "algoritma ikamesi"
  HSC_KMEANS: ayni protokol, KMeans++ ile            <- baseline
  BIZIM     : ayni bolme, bizim pipeline (NMF20 + repair + kNN/MF karisimi)

Amac: (a) literatur kurulumunda AVOA fark yaratiyor mu, (b) bizim pipeline'in
katkisi ne kadari protokolden ne kadari algoritmadan geliyor.

Kullanim: python algo_selection_v2/hsc_replikasyon.py --dataset 100k --k 70
Cikti: results/hsc_replikasyon.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from recompute_scores import discover_algorithms  # noqa: E402

TOPN, THR = 10, 4.0


def veri_yukle(dataset, data_root, seed=42):
    if dataset == "100k":
        df = pd.read_csv(data_root / "ml-100k" / "u.data", sep="\t",
                         names=["u", "i", "r", "t"])
        NU, NI = 943, 1682
    else:
        df = pd.read_csv(data_root / "ml-1m" / "ratings.dat", sep="::",
                         engine="python", names=["u", "i", "r", "t"],
                         encoding="latin-1")
        NU, NI = 6040, 3952
    u = df.u.values - 1; i = df.i.values - 1; r = df.r.values.astype(float)
    rng = np.random.default_rng(seed)
    te = np.zeros(len(r), bool)
    te[rng.choice(len(r), int(0.2 * len(r)), replace=False)] = True   # %80/%20
    return (u[~te], i[~te], r[~te], u[te], i[te], r[te], NU, NI)


def wcss_fitness(R, K, NI):
    """HSC'nin fitness'i: WCSS (Oklid), serbest atama."""
    R2 = (R ** 2).sum(1)

    def fit(sol):
        C = np.asarray(sol).reshape(K, NI)
        d = R2[:, None] - 2.0 * (R @ C.T) + (C ** 2).sum(1)[None, :]
        return float(np.maximum(d.min(1), 0).sum())
    return fit


def cmean_tahmin(L, K, NI, tu, ti, tr, eu, ei, gmean, im):
    flat = L[tu].astype(np.int64) * NI + ti
    s = np.bincount(flat, weights=tr, minlength=K * NI)
    n = np.bincount(flat, minlength=K * NI)
    j = L[eu].astype(np.int64) * NI + ei
    p = np.where(n[j] > 0, s[j] / np.maximum(n[j], 1), im[ei])
    return np.clip(p, 1, 5), 100 * float((n[j] == 0).mean())


def metrikler(p, er, eu, NU):
    e = p - er
    out = {"mae": float(np.abs(e).mean()),
           "rmse": float(np.sqrt((e ** 2).mean()))}
    by = {}
    for j, uu in enumerate(eu):
        by.setdefault(uu, []).append(j)
    precs, recs, ndcgs = [], [], []
    for uu, idx in by.items():
        i = np.array(idx)
        order = i[np.argsort(-p[i])]
        rel = (er[order] >= THR).astype(float)
        nn = min(TOPN, len(order))
        precs.append(rel[:nn].sum() / TOPN)
        nrel = int((er[i] >= THR).sum())
        if nrel:
            recs.append(rel[:nn].sum() / nrel)
            disc = np.log2(np.arange(2, nn + 2))
            idcg = (np.sort(rel)[::-1][:nn] / disc).sum()
            ndcgs.append(float((rel[:nn] / disc).sum() / idcg) if idcg else 0.0)
    out.update({"prec10": float(np.mean(precs)), "rec10": float(np.mean(recs)),
                "ndcg10": float(np.mean(ndcgs))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    ap.add_argument("--k", type=int, default=70)
    ap.add_argument("--max-fe", type=int, default=600)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data_root = ROOT.parent / "data"
    tu, ti, tr, eu, ei, er, NU, NI = veri_yukle(args.dataset, data_root,
                                                args.seed)
    R = np.zeros((NU, NI)); R[tu, ti] = tr          # eksikler 0 (HSC boyle diyor)
    gmean = float(tr.mean())
    cnt = np.bincount(ti, minlength=NI)
    im = np.where(cnt > 0, np.bincount(ti, weights=tr, minlength=NI)
                  / np.maximum(cnt, 1), gmean)
    K = args.k
    print(f"[HSC protokolu] {args.dataset}: egitim={len(tr)} test={len(er)} "
          f"matris={R.shape} K={K}", flush=True)

    fitf = wcss_fitness(R, K, NI)
    out = RESULTS / "hsc_replikasyon.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K, old.dataset))

    from mealpy import FloatVar
    rng = np.random.default_rng(args.seed)
    lb, ub = np.full(K * NI, 1.0), np.full(K * NI, 5.0)   # HSC: [1,5] rastgele

    kollar = {"HSC_KMEANS": None,
              "HSC_SSA": "SSA.OriginalSSA",
              "HSC_AVOA": "AVOA.OriginalAVOA"}
    for ad, aname in kollar.items():
        if (ad, K, args.dataset) in done:
            continue
        t0 = time.time()
        if aname is None:
            km = KMeans(K, init="k-means++", n_init=3,
                        random_state=args.seed).fit(R)
            C = km.cluster_centers_
        else:
            try:
                cls = discover_algorithms([aname])[aname]
            except KeyError:
                print(f"  {ad}: {aname} bulunamadi, atlaniyor", flush=True)
                continue
            starts = rng.uniform(1, 5, (args.pop, K * NI))   # HSC: rastgele init
            g = cls(epoch=5000, pop_size=args.pop).solve(
                {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub),
                 "minmax": "min", "log_to": None},
                seed=args.seed, termination={"max_fe": args.max_fe},
                starting_solutions=starts)
            C = np.asarray(g.solution).reshape(K, NI)
        R2 = (R ** 2).sum(1)
        d = R2[:, None] - 2.0 * (R @ C.T) + (C ** 2).sum(1)[None, :]
        L = d.argmin(1)
        sz = np.bincount(L, minlength=K)
        p, fbp = cmean_tahmin(L, K, NI, tu, ti, tr, eu, ei, gmean, im)
        r = {"yontem": ad, "K": K, "dataset": args.dataset,
             "sure_s": round(time.time() - t0, 1), "fallback_pct": round(fbp, 2),
             "maxk": int(sz.max()), "mink": int(sz.min()),
             "bos_kume": int((sz == 0).sum()), "ort_kume": int(sz.mean()),
             "std_kume": int(sz.std()), **metrikler(p, er, eu, NU)}
        rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {ad:11s} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} "
              f"NDCG={r['ndcg10']:.4f} P@10={r['prec10']:.4f} | "
              f"kume max={r['maxk']} min={r['mink']} bos={r['bos_kume']} "
              f"({r['sure_s']}s)", flush=True)


if __name__ == "__main__":
    main()
