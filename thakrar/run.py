"""
Thakrar Algoritma 2 — ML-100K resmi 5-fold sweep runner (sklearn'siz).

Kullanım:
    python -m thakrar.run --quick
    python -m thakrar.run --folds 1 2 3 4 5 --k 9 14 19 --latent 5 10 15
    python -m thakrar.run --init random --mf-epochs 50

Çıktı: results/thakrar_clean_ml100k.csv  (her fold ayrı satır + ortalama).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, pipeline

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "thakrar_clean_ml100k.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Thakrar Alg.2 temiz pipeline (ML-100K)")
    ap.add_argument("--quick", action="store_true", help="fold 1, k=14, L=10 tek koşu")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--k", type=int, nargs="+", default=[9, 14, 19])
    ap.add_argument("--latent", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--init", choices=["kmeans++", "random", "avoa", "hho", "cso"],
                    default="kmeans++",
                    help="centroid başlatma; avoa/hho/cso meta-sezgisel WCSS init")
    ap.add_argument("--meta-epoch", type=int, default=100,
                    help="meta-sezgisel iterasyon (avoa/hho/cso init için)")
    ap.add_argument("--meta-pop", type=int, default=30,
                    help="meta-sezgisel popülasyon (avoa/hho/cso init için)")
    ap.add_argument("--mf-epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.quick:
        folds, k_list, l_list = [1], [14], [10]
    else:
        folds, k_list, l_list = args.folds, args.k, args.latent

    print("Thakrar et al. (2025) Algoritma 2 — temiz pipeline (sklearn'siz)")
    print("MF (Alg.4) -> kendi K-Means (Alg.5) -> kume-ortalamasi (Alg.6)")
    print(f"Dataset: ML-100K resmi fold(lar) {folds}")
    print(f"k={k_list}  L={l_list}  init={args.init}  MF(epochs={args.mf_epochs}, "
          f"lr={args.lr}, reg={args.reg})\n")

    rows = []
    for k in k_list:
        for ld in l_list:
            for fold in folds:
                train, test, n_users, n_items = data.load_official_fold(fold)
                res = pipeline.run_config(
                    train, test, n_users, n_items,
                    k=k, latent_dim=ld,
                    mf_epochs=args.mf_epochs, lr=args.lr, reg=args.reg,
                    init_mode=args.init, meta_epoch=args.meta_epoch,
                    meta_pop=args.meta_pop, seed=args.seed, verbose=args.verbose,
                )
                res["fold"] = fold
                rows.append(res)
                print(
                    f"k={k:2d} L={ld:2d} fold={fold} | "
                    f"MAE={res['mae']:.4f} RMSE={res['rmse']:.4f} | "
                    f"WCSS={res['wcss']:.1f} iters={res['kmeans_iters']} | "
                    f"clmin={res['cluster_min']} clmax={res['cluster_max']} "
                    f"empty={res['n_empty_clusters']} | "
                    f"hit={res['cluster_mean_pct']:.1f}% fb={res['global_fallback_pct']:.1f}% | "
                    f"{res['seconds']:.1f}s",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # fold ortalamaları (k, L bazında)
    agg = (
        df.groupby(["k", "latent_dim"])
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
             rmse_mean=("rmse", "mean"),
             fb_mean=("global_fallback_pct", "mean"))
        .reset_index()
        .sort_values("mae_mean")
    )
    print("\n=== Fold ortalamalari (k, L bazinda, MAE'ye gore sirali) ===")
    print(agg.to_string(index=False))

    best = agg.iloc[0]
    print(
        f"\nEn iyi: k={int(best['k'])} L={int(best['latent_dim'])}  "
        f"MAE={best['mae_mean']:.4f} (+/-{best['mae_std']:.4f})  "
        f"RMSE={best['rmse_mean']:.4f}"
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
