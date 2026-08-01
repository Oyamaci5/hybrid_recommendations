"""
IWO_HHO K=10: ham rating (feature-extraction none) + LOF + fuzzy FCM, cluster_avg eval.

  python -m experiments.run_iwo_hho_k10_lof_no_wnmf --phase all
  python -m experiments.run_iwo_hho_k10_lof_no_wnmf --phase eval
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

from experiments.run_iwo_hho_k10_lof_preprocess_dim_grid import baseline_best_k10_mae
from wnmf.wnmf_experiment import (
    _align_assignment_bundle,
    _cluster_avg_predict_kwargs,
    _nearest_centroid_bundle,
    load_assignment,
    load_memberships,
    run_cluster_average,
)
from wnmf.wnmf_utils import load_ratings_100k

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments_lof_estop" / "ml100k"
OUT_CSV = REPO / "results" / "iwo_hho_k10_lof_no_wnmf.csv"
WNMF_GRID_CSV = REPO / "results" / "iwo_hho_k10_lof_preprocess_wnmf_grid.csv"

ALGO = "IWO_HHO"
K = 10
FOLD = 1
SOFT = 0.1
SIM = "cosine"
# generate_assignments: none FE -> assign_suffix _none_k10 (+ pwcss, m15)
ASSIGN_GLOB = f"{ALGO}*trainonly_official_f{FOLD}_none_k{K}*pwcss*m15"


def find_assign_dir() -> Path | None:
    cands = [
        p
        for p in ASSIGN_ROOT.glob(ASSIGN_GLOB)
        if "wnmf" not in p.name.lower()
        and (p / "assignments.npy").is_file()
        and (p / "memberships.npy").is_file()
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def run_assign(*, jobs: int, skip_existing: bool) -> int:
    cmd = [
        sys.executable,
        str(GEN),
        "--dataset",
        "100k",
        "--algo",
        ALGO,
        "--no-prune",
        "--preprocess",
        "none",
        "--feature-extraction",
        "none",
        "--svd-components",
        "20",
        "--init-mode",
        "mkpp",
        "--cluster-metric",
        "fuzzy",
        "--fitness",
        "wcss",
        "--cluster-objective",
        "wcss",
        "--train-only",
        "--eval-split",
        "official",
        "--fold",
        str(FOLD),
        "--fcm-m",
        "1.5",
        "--fcm-m-suffix",
        "--lof",
        "--k",
        str(K),
        "--jobs",
        str(jobs),
        "--baseline-epoch",
        "40",
        "--pop-size",
        "25",
        "--early-stop",
        "--early-stop-patience",
        "4",
        "--early-stop-block",
        "5",
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def eval_one(adir: Path) -> dict:
    base = str(REPO / "data" / "ml-100k" / "u1.base")
    test = str(REPO / "data" / "ml-100k" / "u1.test")
    train, test_arr = load_ratings_100k(base, test, fold=FOLD)
    n_items = int(max(train[:, 1].max(), test_arr[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test_arr[:, 0].max())) + 1

    assignments, gray_mask = load_assignment(str(adir))
    memberships = load_memberships(str(adir))
    assignments, gray_mask, memberships, _ = _align_assignment_bundle(
        assignments,
        gray_mask,
        memberships,
        None,
        n_users_expected=n_users,
        algo_label=ALGO,
        assign_dir=str(adir),
    )
    nc = _nearest_centroid_bundle(None, str(adir), assignments)
    eval_args = Namespace(similarity=SIM, min_common=3, soft_membership_threshold=SOFT)

    t0 = time.time()
    r = run_cluster_average(
        train,
        test_arr,
        assignments,
        gray_mask,
        memberships,
        n_items,
        ALGO,
        **_cluster_avg_predict_kwargs(eval_args),
        **nc,
        top_n=10,
        relevance_threshold=4.0,
        assign_dir=str(adir),
    )
    return {
        "algo": ALGO,
        "k": K,
        "fold": FOLD,
        "predictor": "cluster_avg_soft",
        "similarity": SIM,
        "soft_threshold": SOFT,
        "preprocess": "none",
        "feature_extraction": "none",
        "lof": True,
        "wnmf_dim": 0,
        "assignment_dir": str(adir),
        "mae": float(r["mae"]),
        "rmse": float(r["rmse"]),
        "ndcg_at_10": float(r["ndcg_at_10"]),
        "precision_at_10": float(r["precision_at_10"]),
        "recall_at_10": float(r["recall_at_10"]),
        "eval_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "eval", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    if args.phase in ("assign", "all"):
        rc = run_assign(jobs=args.jobs, skip_existing=args.skip_existing)
        if rc != 0:
            return rc

    if args.phase in ("eval", "all"):
        adir = find_assign_dir()
        if adir is None:
            print("Atama bulunamadi.", flush=True)
            return 1
        print(f"Eval: {adir}", flush=True)
        row = eval_one(adir)
        df = pd.DataFrame([row])
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)

        base = baseline_best_k10_mae()
        print(f"\nCSV -> {OUT_CSV}")
        print(
            f"no-WNMF+LOF: MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} "
            f"NDCG={row['ndcg_at_10']:.4f}",
            flush=True,
        )
        if WNMF_GRID_CSV.is_file():
            prev = pd.read_csv(WNMF_GRID_CSV)
            best_wnmf = float(prev["mae"].min())
            print(f"Onceki LOF+WNMF en iyi: MAE={best_wnmf:.4f} (wnmf_dim={int(prev.loc[prev['mae'].idxmin(), 'wnmf_dim'])})")
            print(f"Delta vs LOF+WNMF: {row['mae'] - best_wnmf:+.4f}")
        if pd.notna(base):
            print(f"Grid K=10 baseline (fast, no LOF): MAE={base:.4f}")
            print(f"Delta vs baseline: {row['mae'] - base:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
