"""
FCM m × soft-membership threshold grid (HA_AVOAHGS, K=10).

Atama: mevcut _fuzzy_..._k10 korunur; her m için _m10/_m15/_m20 etiketi eklenir.
Eval : cluster_avg, soft threshold grid.

  python experiments/run_fcm_m_soft_grid.py --phase assign
  python experiments/run_fcm_m_soft_grid.py --phase eval
  python experiments/run_fcm_m_soft_grid.py --phase all
  python experiments/run_fcm_m_soft_grid.py --phase summary
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"
OUT_CSV = REPO / "results" / "fcm_m_soft_grid_ha_k10.csv"

ALGO = "HA_AVOAHGS"
K = 10
WNMF_DIM = 20
M_LIST = [1.0, 1.5, 2.0]
SOFT_LIST = [0.1, 0.3, 0.5, 0.7]

BASE_SUFFIX = (
    f"_fuzzy_imkpp_nogs_trainonly_rand_wnmfep50_none_k{K}"
)
)


def m_tag(fcm_m: float) -> str:
    return f"_m{int(round(float(fcm_m) * 10))}"


def assign_suffix(fcm_m: float) -> str:
    return f"{BASE_SUFFIX}{m_tag(fcm_m)}"


def assign_path(fcm_m: float) -> Path:
    return ASSIGN_ROOT / "ml100k" / f"{ALGO}{assign_suffix(fcm_m)}"


def phase_assign(skip_existing: bool) -> int:
    cmd_base = [
        sys.executable, str(GEN),
        "--dataset", "100k",
        "--algo", ALGO,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--init-mode", "mkpp",
        "--cluster-metric", "fuzzy",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", "random", "--fold", "1",
        "--k", str(K),
        "--fcm-m-suffix",
    ]
    if skip_existing:
        cmd_base.append("--skip-existing")

    rc = 0
    for fcm_m in M_LIST:
        cmd = cmd_base + ["--fcm-m", str(fcm_m)]
        print(" ".join(cmd), flush=True)
        ret = subprocess.run(cmd, cwd=str(REPO)).returncode
        rc = rc or ret
    return rc


def phase_eval() -> pd.DataFrame:
    from wnmf.wnmf_experiment import (
        RANDOM_SEED,
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        load_ratings_100k_all,
        run_cluster_average,
    )

    data = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1

    rows = []
    for fcm_m in M_LIST:
        adir = assign_path(fcm_m)
        if not (adir / "assignments.npy").is_file():
            print(f"SKIP assign missing: {adir}", flush=True)
            continue

        assignments, gray_mask = load_assignment(str(adir))
        memberships = load_memberships(str(adir))
        assignments, gray_mask, memberships, _ = _align_assignment_bundle(
            assignments, gray_mask, memberships, None,
            n_users_expected=n_users, algo_label=ALGO, assign_dir=str(adir),
        )
        nc = _nearest_centroid_bundle(None, str(adir), assignments)

        for soft_t in SOFT_LIST:
            eval_args = Namespace(
                similarity="cosine",
                min_common=3,
                soft_membership_threshold=float(soft_t),
            )
            t0 = time.time()
            r = run_cluster_average(
                train, test, assignments, gray_mask, memberships, n_items, ALGO,
                **_cluster_avg_predict_kwargs(eval_args),
                **nc,
                top_n=10,
                relevance_threshold=4.0,
                assign_dir=str(adir),
            )
            rows.append({
                "protocol": "fcm_m_soft_grid",
                "algo": ALGO,
                "k": K,
                "fcm_m": float(fcm_m),
                "soft_threshold": float(soft_t),
                "predictor": "cluster_avg",
                "assign_suffix": assign_suffix(fcm_m),
                "mae": r["mae"],
                "rmse": r["rmse"],
                "ndcg_at_10": r["ndcg_at_10"],
                "precision_at_10": r["precision_at_10"],
                "recall_at_10": r["recall_at_10"],
                "eval_seconds": round(time.time() - t0, 1),
            })
            print(
                f"  m={fcm_m:g} soft={soft_t:g}: "
                f"MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} "
                f"NDCG={r['ndcg_at_10']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.is_file():
        old = pd.read_csv(OUT_CSV)
        key = ["algo", "k", "fcm_m", "soft_threshold", "predictor"]
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
        merged = merged.sort_values(["fcm_m", "soft_threshold"]).reset_index(drop=True)
        merged.to_csv(OUT_CSV, index=False)
        return merged

    df.to_csv(OUT_CSV, index=False)
    return df


def phase_summary() -> None:
    if not OUT_CSV.is_file():
        print(f"CSV yok: {OUT_CSV}", flush=True)
        return
    df = pd.read_csv(OUT_CSV)
    if df.empty:
        print("CSV boş.", flush=True)
        return

    best_mae = df.loc[df["mae"].idxmin()]
    best_ndcg = df.loc[df["ndcg_at_10"].idxmax()]
    print("\n=== En iyi MAE ===")
    print(best_mae.to_string())
    print("\n=== En iyi NDCG@10 ===")
    print(best_ndcg.to_string())

    pivot = df.pivot_table(
        index="fcm_m", columns="soft_threshold", values="mae", aggfunc="min",
    )
    print("\n=== MAE pivot (m × soft) ===")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    p = argparse.ArgumentParser(description="FCM m × soft threshold grid (HA K=10)")
    p.add_argument(
        "--phase", choices=["assign", "eval", "summary", "all"],
        default="all",
    )
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    rc = 0
    if args.phase in ("assign", "all"):
        rc = phase_assign(skip_existing=args.skip_existing)
    if args.phase in ("eval", "all"):
        phase_eval()
    if args.phase in ("summary", "all"):
        phase_summary()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
