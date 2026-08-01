"""
HA fuzzy K=10 — cluster_avg hiperparametre grid (atama yeniden üretilmez).

Grid: soft_membership_threshold × similarity × min_common

  python experiments/run_ha_k10_cluster_avg_param_grid.py
  python experiments/run_ha_k10_cluster_avg_param_grid.py --summary-only
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))

from wnmf.wnmf_experiment import (  # noqa: E402
    RANDOM_SEED,
    _align_assignment_bundle,
    _cluster_avg_predict_kwargs,
    _nearest_centroid_bundle,
    load_assignment,
    load_memberships,
    load_ratings_100k_all,
    run_cluster_average,
)

ALGO = "HA_AVOAHGS"
K = 10
ASSIGN_SUFFIX = "_fuzzy_imkpp_nogs_trainonly_rand_wnmfep50_none_k10"
ASSIGN_DIR = REPO / "mealpy" / "results" / "assignments" / "ml100k" / f"{ALGO}{ASSIGN_SUFFIX}"
OUT_CSV = REPO / "results" / "ha_k10_cluster_avg_param_grid.csv"

SOFT_LIST = [0.05, 0.1, 0.2, 0.3, 0.5]
SIM_LIST = ["cosine", "pearson"]
MIN_COMMON_LIST = [2, 3, 4, 5]


def run_grid() -> pd.DataFrame:
    data = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1

    assignments, gray_mask = load_assignment(str(ASSIGN_DIR))
    memberships = load_memberships(str(ASSIGN_DIR))
    assignments, gray_mask, memberships, _ = _align_assignment_bundle(
        assignments, gray_mask, memberships, None,
        n_users_expected=n_users, algo_label=ALGO, assign_dir=str(ASSIGN_DIR),
    )
    nc = _nearest_centroid_bundle(None, str(ASSIGN_DIR), assignments)

    combos = list(itertools.product(SOFT_LIST, SIM_LIST, MIN_COMMON_LIST))
    print(f"HA K=10 cluster_avg grid: {len(combos)} kombinasyon", flush=True)

    rows = []
    for i, (soft_t, sim, min_c) in enumerate(combos, 1):
        eval_args = Namespace(
            similarity=sim,
            min_common=min_c,
            soft_membership_threshold=float(soft_t),
        )
        t0 = time.time()
        r = run_cluster_average(
            train, test, assignments, gray_mask, memberships, n_items, ALGO,
            **_cluster_avg_predict_kwargs(eval_args),
            **nc,
            top_n=10,
            relevance_threshold=4.0,
            assign_dir=str(ASSIGN_DIR),
        )
        elapsed = time.time() - t0
        rows.append({
            "algo": ALGO,
            "k": K,
            "assign_suffix": ASSIGN_SUFFIX,
            "predictor": "cluster_avg",
            "soft_threshold": float(soft_t),
            "similarity": sim,
            "min_common": int(min_c),
            "mae": r["mae"],
            "rmse": r["rmse"],
            "ndcg_at_10": r["ndcg_at_10"],
            "precision_at_10": r["precision_at_10"],
            "recall_at_10": r["recall_at_10"],
            "eval_seconds": round(elapsed, 1),
        })
        print(
            f"  [{i}/{len(combos)}] soft={soft_t:g} sim={sim:7s} min_c={min_c} "
            f"MAE={r['mae']:.4f} NDCG={r['ndcg_at_10']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n-> {OUT_CSV}  ({len(df)} satır)", flush=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Sonuç yok.", flush=True)
        return

    baseline = df[
        (df["soft_threshold"] == 0.1)
        & (df["similarity"] == "cosine")
        & (df["min_common"] == 3)
    ]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(
            f"\nBaseline (soft=0.1, cosine, min_c=3): "
            f"MAE={b['mae']:.4f} NDCG={b['ndcg_at_10']:.4f}",
            flush=True,
        )

    best_mae = df.loc[df["mae"].idxmin()]
    best_ndcg = df.loc[df["ndcg_at_10"].idxmax()]
    print("\n=== En iyi MAE ===", flush=True)
    print(best_mae.to_string(), flush=True)
    print("\n=== En iyi NDCG@10 ===", flush=True)
    print(best_ndcg.to_string(), flush=True)

    print("\n=== MAE pivot (soft × min_common), cosine ===", flush=True)
    sub = df[df["similarity"] == "cosine"]
    if not sub.empty:
        print(
            sub.pivot_table(
                index="soft_threshold", columns="min_common", values="mae", aggfunc="first",
            ).to_string(float_format=lambda x: f"{x:.4f}"),
            flush=True,
        )

    print("\n=== NDCG pivot (soft × min_common), cosine ===", flush=True)
    if not sub.empty:
        print(
            sub.pivot_table(
                index="soft_threshold", columns="min_common", values="ndcg_at_10", aggfunc="first",
            ).to_string(float_format=lambda x: f"{x:.4f}"),
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        if not OUT_CSV.is_file():
            print(f"CSV yok: {OUT_CSV}", flush=True)
            return
        print_summary(pd.read_csv(OUT_CSV))
        return

    df = run_grid()
    print_summary(df)


if __name__ == "__main__":
    main()
