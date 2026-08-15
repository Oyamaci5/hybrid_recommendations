
"""
BUYUK K (90, 120) — FCM + KMeans + 5 meta-sezgisel, TAM HAT (regret + soft
top-2 + kume-MF/kNN karisimi), ML-1M.

ml1m_buyukK.py'den farki: (1) regret-kurallli fast_repair (mesafe-kurali
degil), (2) tum 5 meta-sezgisel + FCM + B0 (sadece B0/AVOA + cmean degil),
(3) K=90/120'de "bizim yapimiz" ile klasik yontemleri (FCM/KMeans) ayni
protokolde (kNN+MF, kapasiteli) kiyaslamak icin.

FCM: standart Bezdek fuzzy c-means, elde disi (skfuzzy yok), vektorlestirilmis.
Sonuc merkezler ayni evaluate() hattina (repair+soft top-2+kNN/MF) verilir --
yani FCM de "merkez bulma yontemi" olarak B0/AVOA ile ayni muameleyi gorur.

Kullanim:
  python algo_selection_v2/ml1m_buyukK2.py --klist 90 120 --folds 1 --seeds 42 --resume
Cikti: results/ml1m_buyukK2.csv
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
from ctx_ml1m import N_I, N_U, Ctx1M, genre_profile_1m  # noqa: E402
from ml1m_run import build_space, evaluate, make_fitness, solve_warm  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402
from tam_run import save_user_vec, user_mae_vec  # noqa: E402
from ml1m_run import user_mae_vec_1m  # noqa: E402

cm.N_I = N_I
ALGOS = ["AVOA.OriginalAVOA", "NGO.OriginalNGO", "HHO.OriginalHHO",
         "HGS.OriginalHGS", "GWO.OriginalGWO"]


def fcm_fit(X, K, seed=42, m=2.0, iters=100, tol=1e-5):
    """Standart Bezdek FCM. Doner: merkezler (K x D)."""
    rng = np.random.default_rng(seed)
    N = len(X)
    U = rng.random((N, K))
    U /= U.sum(1, keepdims=True)
    C = None
    for _ in range(iters):
        Um = U ** m
        C = (Um.T @ X) / np.maximum(Um.sum(0)[:, None], 1e-12)
        d = np.sqrt(np.maximum(
            ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1), 1e-20))
        inv = d ** (-2.0 / (m - 1))
        U_new = inv / inv.sum(1, keepdims=True)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klist", type=int, nargs="+", default=[90, 120])
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--uzay", choices=["nmf", "nmf+genre"], default="nmf")
    ap.add_argument("--max-fe", type=int, default=1200)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(args.data_dir)

    out = RESULTS / "ml1m_buyukK2.csv"
    upath = RESULTS / "ml1m_buyukK2_userr.npz"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K, old.fold, old.seed))

    algos = discover_algorithms(ALGOS)

    for fold in args.folds:
        t0 = time.time()
        ctx = Ctx1M(data, fold)
        G = genre_profile_1m(ctx, data) if args.uzay != "nmf" else None
        X = build_space(ctx, G, args.uzay)
        print(f"  baglam+uzay {time.time()-t0:.0f}s", flush=True)
        for K in args.klist:
            cm.K = K
            fitf = make_fitness(ctx, X, K)
            for seed in args.seeds:
                km = KMeans(K, init="k-means++", n_init=5,
                            random_state=seed).fit(X)
                c0 = km.cluster_centers_

                todo = [("KMeans", c0), ("FCM", None)] + \
                    [(a.split(".")[0] + "_warm", a) for a in algos]
                for yn, ref in todo:
                    if (yn, K, fold, seed) in done:
                        continue
                    t1 = time.time()
                    if yn == "KMeans":
                        cent = c0
                    elif yn == "FCM":
                        cent = fcm_fit(X, K, seed=seed)
                    else:
                        cent = solve_warm(algos[ref], fitf, X, K, seed, c0,
                                          args.max_fe)
                    p, m = evaluate(ctx, X, cent, K)
                    rows.append({"yontem": yn, "K": K, "fold": fold,
                                 "seed": seed,
                                 "sure_s": round(time.time() - t1, 1), **m})
                    pd.DataFrame(rows).to_csv(out, index=False)
                    save_user_vec(upath, f"{yn}|K{K}|f{fold}|s{seed}",
                                  user_mae_vec_1m(ctx, p))
                    print(f"  K={K} {yn:11s} f{fold} s{seed} MAE={m['mae']:.4f} "
                          f"NDCG={m['ndcg10']:.4f} havuz={m['havuz']} "
                          f"({round(time.time()-t1,1)}s)", flush=True)


if __name__ == "__main__":
    main()
