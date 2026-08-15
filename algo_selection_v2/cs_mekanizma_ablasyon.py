
"""
CS HATTININ LLOYD-DISI MEKANIZMALARI — izole ablasyon.

Lloyd zaten uc kez test edildi (kmref, Lloyd paradoksu, memetik) -> disarida.
Burada CS-Kmeans'in DIGER uc fikri tek tek test ediliyor:

  M1 terk_etme  : her turda Pa olasilikla bir merkez atilir ve VERI NOKTASINDAN
                  yenisi uretilir (CS Adim 3'un ikinci yarisi)
  M2 elitist    : birey bazinda kabul — yeni cozum eskisinden kotuyse geri alinir
                  (CS Adim 5). mealpy global-best tutar ama birey bazinda tutmaz.
  M3 medoid     : merkezler her turda en yakin VERI NOKTASINA cekilir
                  (CS'in "merkezler veri noktasidir" temsili)
  M1+M2, M1+M3  : kombinasyonlar
  duz           : mevcut yontem (referans)

Uygulama: mealpy aramasi butce parcalarina bolunur, parcalar arasinda
ilgili mekanizma uygulanir. Toplam NFE tum kollarda esit.

Kullanim:
  python algo_selection_v2/cs_mekanizma_ablasyon.py --k 40 --seeds 42 43 44 --tum
Cikti: results/cs_mekanizma.csv
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
import cluster_mf as cm  # noqa: E402
import tabloB_plus as bp  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402

ALGOS = ["AVOA.OriginalAVOA", "HGS.OriginalHGS", "HHO.OriginalHHO",
         "NGO.OriginalNGO", "GWO.OriginalGWO"]


def terk_et(C, X, rng, Pa):
    """Pa olasilikla bir merkezi at, veri noktasindan yenisini uret."""
    C = C.copy()
    for c in range(len(C)):
        if rng.random() < Pa:
            C[c] = X[rng.integers(len(X))]
    return C


def medoid_cek(C, X):
    """Her merkezi en yakin veri noktasina tasi (CS'in temsili)."""
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return X[d.argmin(0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-fe", type=int, default=1200)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--tur", type=int, default=4)
    ap.add_argument("--pa", type=float, default=0.15)
    ap.add_argument("--tum", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data, args.fold)
    X = build_space(ctx, genre_profile(ctx, data), "nmf")
    S = build_sims(ctx)
    K = args.k
    bp.K = K; cm.K = K
    fitf = bp.make_fitness(ctx, X)
    from mealpy import FloatVar
    lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)

    out = RESULTS / "cs_mekanizma.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.kol, old.K, old.seed))

    adlar = ALGOS if args.tum else ["AVOA.OriginalAVOA"]
    algos = discover_algorithms(adlar)

    def ara(cls, c0, seed, nfe):
        rng = np.random.default_rng(seed)
        st = np.clip(np.vstack([c0.ravel()] + [
            c0.ravel() + rng.normal(0, .08 * X.std(), c0.size)
            for _ in range(args.pop - 1)]), lb, ub)
        g = cls(epoch=5000, pop_size=args.pop).solve(
            {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub),
             "minmax": "min", "log_to": None}, seed=seed,
            termination={"max_fe": nfe}, starting_solutions=st)
        return np.asarray(g.solution).reshape(K, X.shape[1]), float(g.target.fitness)

    def kol_calistir(cls, c0, seed, kol):
        """Butceyi parcalara bol, parcalar arasi mekanizma uygula."""
        rng = np.random.default_rng(seed + 7)
        pay = max(args.max_fe // args.tur, 1)
        C, F = c0.copy(), fitf(c0.ravel())
        for _ in range(args.tur):
            C_yeni, F_yeni = ara(cls, C, seed, pay)
            # M2 elitist: kotulesirse geri al
            if "elitist" in kol and F_yeni > F:
                C_yeni, F_yeni = C, F
            C, F = C_yeni, F_yeni
            if "terk" in kol:                       # M1
                C_t = terk_et(C, X, rng, args.pa)
                F_t = fitf(C_t.ravel())
                if F_t < F or "elitist" not in kol:
                    C, F = C_t, F_t
            if "medoid" in kol:                     # M3
                C_m = medoid_cek(C, X)
                F_m = fitf(C_m.ravel())
                if F_m < F or "elitist" not in kol:
                    C, F = C_m, F_m
        return C, F

    kollar = ["duz", "terk", "elitist", "medoid", "terk+elitist", "terk+medoid"]

    for seed in args.seeds:
        km = KMeans(K, init="k-means++", n_init=5, random_state=seed).fit(X)
        c0 = km.cluster_centers_
        if ("B0", "duz", K, seed) not in done:
            _, m = bp.evaluate(ctx, X, S, c0)
            rows.append({"yontem": "B0", "kol": "duz", "K": K, "seed": seed, **m})
            print(f"B0    duz          MAE={m['mae']:.4f} NDCG={m['ndcg10']:.4f}",
                  flush=True)
        for an, cls in algos.items():
            kisa = an.split(".")[0]
            for kol in kollar:
                if (kisa, kol, K, seed) in done:
                    continue
                t0 = time.time()
                if kol == "duz":
                    C, F = ara(cls, c0, seed, args.max_fe)
                else:
                    C, F = kol_calistir(cls, c0, seed, kol)
                _, m = bp.evaluate(ctx, X, S, C)
                rows.append({"yontem": kisa, "kol": kol, "K": K, "seed": seed,
                             "fit_val": round(F, 5), "nfe": args.max_fe,
                             "sure_s": round(time.time() - t0, 1), **m})
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{kisa:5s} {kol:12s} MAE={m['mae']:.4f} "
                      f"NDCG={m['ndcg10']:.4f} fit={F:.5f}", flush=True)


if __name__ == "__main__":
    main()
