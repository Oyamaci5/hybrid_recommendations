
"""
MEMETIK LLOYD ABLASYONU — tum algoritmalar + B0, iki K noktasi.

CS-Kmeans sablonundan ogrenilen: dongu icinde Lloyd adimi yuksek K'da faydali,
dusuk K'da zararli. Burada bunu TUM algoritmalarla ve B0 ile test ediyoruz.

Kollar (her algoritma icin):
  duz     : mevcut yontem (warm start + kapasiteli, Lloyd YOK)
  memetik : arama sonunda + periyodik 1 Lloyd adimi (kapasiteli atama ile)
B0 tarafinda:
  B0        : KMeans++ (zaten Lloyd'un kendisi)
  B0_repair : kapasiteli atamayla degerlendirilmis KMeans++

Kapasiteli Lloyd: atama repair ile yapilir, merkez = kume ortalamasi.
Boylece Lloyd adimi da kisitla uyumlu kalir (saf Lloyd kisiti bozardi).

Kullanim:
  python algo_selection_v2/memetik_kiyas.py --k 40 --seeds 42 43 44 --tum
  python algo_selection_v2/memetik_kiyas.py --k 6  --seeds 42 43 44 --tum
  python algo_selection_v2/memetik_kiyas.py --dataset 1m --k 40 --seeds 42 43 44
Cikti: results/memetik_kiyas.csv
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
from recompute_scores import discover_algorithms  # noqa: E402

ALGOS = ["AVOA.OriginalAVOA", "HGS.OriginalHGS", "HHO.OriginalHHO",
         "NGO.OriginalNGO", "GWO.OriginalGWO"]


def kur(dataset, fold):
    if dataset == "100k":
        import tabloB_plus as bp
        from eksikler_deney import Ctx
        from genre_k_deney import build_space, genre_profile
        from pred_v2 import build_sims
        data = ROOT.parent / "data" / "ml-100k"
        ctx = Ctx(data, fold)
        X = build_space(ctx, genre_profile(ctx, data), "nmf")
        return ctx, X, build_sims(ctx), bp, 1682
    import ml1m_run as mm
    from ctx_ml1m import Ctx1M
    ctx = Ctx1M(ROOT.parent / "data" / "ml-1m", fold)
    return ctx, mm.build_space(ctx, None, "nmf"), ctx.S, mm, 3952


def kapasiteli_lloyd(X, C, ata, adim=1):
    """Kisitla uyumlu Lloyd: repair atamasi -> kume ortalamasi."""
    K = len(C)
    for _ in range(adim):
        L, _ = ata(X, C)
        C = np.vstack([X[L == c].mean(0) if (L == c).any() else C[c]
                       for c in range(K)])
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-fe", type=int, default=1200)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--tur", type=int, default=4, help="memetik: kac parca")
    ap.add_argument("--tum", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ctx, X, S, mod, NI = kur(args.dataset, args.fold)
    K = args.k
    cm.K, cm.N_I = K, NI
    if args.dataset == "100k":
        mod.K = K
        fitf = mod.make_fitness(ctx, X)
        ata = mod.repair_assign
        degerlendir = lambda C: mod.evaluate(ctx, X, S, C)[1]          # noqa: E731
        solve = None
    else:
        fitf = mod.make_fitness(ctx, X, K)
        ata = mod.fast_repair
        degerlendir = lambda C: mod.evaluate(ctx, X, C, K)[1]          # noqa: E731
        solve = mod.solve_warm

    from mealpy import FloatVar
    out = RESULTS / "memetik_kiyas.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.dataset, old.yontem, old.kol, old.K, old.seed))

    adlar = ALGOS if args.tum else ["AVOA.OriginalAVOA"]
    algos = discover_algorithms(adlar)

    def ara(cls, c0, seed, nfe):
        lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
        rng = np.random.default_rng(seed)
        st = np.clip(np.vstack([c0.ravel()] + [
            c0.ravel() + rng.normal(0, .08 * X.std(), c0.size)
            for _ in range(args.pop - 1)]), lb, ub)
        g = cls(epoch=5000, pop_size=args.pop).solve(
            {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub),
             "minmax": "min", "log_to": None}, seed=seed,
            termination={"max_fe": nfe}, starting_solutions=st)
        return np.asarray(g.solution).reshape(K, X.shape[1])

    for seed in args.seeds:
        km = KMeans(K, init="k-means++", n_init=5, random_state=seed).fit(X)
        c0 = km.cluster_centers_
        # --- B0 ---
        if (args.dataset, "B0", "duz", K, seed) not in done:
            m = degerlendir(c0)
            rows.append({"dataset": args.dataset, "yontem": "B0", "kol": "duz",
                         "K": K, "seed": seed, "fold": args.fold, **m})
            print(f"B0    duz     K={K} s{seed} MAE={m['mae']:.4f} "
                  f"NDCG={m['ndcg10']:.4f}", flush=True)
        # B0 + kapasiteli Lloyd (kisitla uyumlu hale getirilmis KMeans++)
        if (args.dataset, "B0", "memetik", K, seed) not in done:
            m = degerlendir(kapasiteli_lloyd(X, c0, ata, adim=3))
            rows.append({"dataset": args.dataset, "yontem": "B0",
                         "kol": "memetik", "K": K, "seed": seed,
                         "fold": args.fold, **m})
            print(f"B0    memetik K={K} s{seed} MAE={m['mae']:.4f} "
                  f"NDCG={m['ndcg10']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

        for an, cls in algos.items():
            kisa = an.split(".")[0]
            # --- duz ---
            if (args.dataset, kisa, "duz", K, seed) not in done:
                t0 = time.time()
                C = ara(cls, c0, seed, args.max_fe)
                m = degerlendir(C)
                rows.append({"dataset": args.dataset, "yontem": kisa,
                             "kol": "duz", "K": K, "seed": seed,
                             "fold": args.fold, "nfe": args.max_fe,
                             "sure_s": round(time.time() - t0, 1), **m})
                print(f"{kisa:5s} duz     K={K} s{seed} MAE={m['mae']:.4f} "
                      f"NDCG={m['ndcg10']:.4f}", flush=True)
            # --- memetik: butce parcalara bolunur, aralarda Lloyd ---
            if (args.dataset, kisa, "memetik", K, seed) not in done:
                t0 = time.time()
                C = c0.copy(); pay = max(args.max_fe // args.tur, 1)
                for _ in range(args.tur):
                    C = ara(cls, C, seed, pay)
                    C = kapasiteli_lloyd(X, C, ata, adim=1)
                m = degerlendir(C)
                rows.append({"dataset": args.dataset, "yontem": kisa,
                             "kol": "memetik", "K": K, "seed": seed,
                             "fold": args.fold, "nfe": args.max_fe,
                             "sure_s": round(time.time() - t0, 1), **m})
                print(f"{kisa:5s} memetik K={K} s{seed} MAE={m['mae']:.4f} "
                      f"NDCG={m['ndcg10']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
