"""
IWO_HHO K=10 fast grid: preprocess (none/zscore) x WNMF dim (75/100/150), LOF.

Runs assignment generation and evaluates cluster_avg_soft (cosine, soft=0.1)
on official fold-1. Compares against current best K=10 baseline MAE.
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

from experiments.run_fuzzy_official_k3_24_cluster_avg import MERGE_CSVS, OUT_CSV
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
ASSIGN_ROOTS = [
    REPO / "mealpy" / "results" / "assignments_lof_estop" / "ml100k",
    REPO / "mealpy" / "results" / "assignments_estop" / "ml100k",
]
OUT_CSV_PATH = REPO / "results" / "iwo_hho_k10_lof_preprocess_wnmf_grid.csv"

ALGO = "IWO_HHO"
K = 10
DIMS = [75, 100, 150]
PREPROCS = ["none", "zscore"]
SOFT = 0.1
SIM = "cosine"
FOLD = 1


def baseline_best_k10_mae() -> float:
    dfs = []
    for p in [OUT_CSV, *MERGE_CSVS]:
        if p.is_file():
            dfs.append(pd.read_csv(p))
    if not dfs:
        return float("nan")
    df = pd.concat(dfs, ignore_index=True)
    if "fast" not in df.columns:
        df["fast"] = False
    if "prune" not in df.columns:
        df["prune"] = False
    if "knn_k" not in df.columns:
        df["knn_k"] = 0
    sub = df[
        (df.get("predictor", "") == "cluster_avg_soft")
        & (df.get("similarity", "") == SIM)
        & (df.get("fast", False) == True)
        & (df.get("prune", False) == False)
        & (df["k"] == K)
    ].copy()
    key = ["fold", "k", "algo", "predictor", "knn_k", "similarity", "fast", "prune"]
    sub = sub.drop_duplicates(subset=key, keep="last")
    return float(sub["mae"].min()) if not sub.empty else float("nan")


def assignment_dir(preprocess: str, dim: int) -> Path | None:
    pattern = (
        f"{ALGO}*trainonly_official_f{FOLD}_{preprocess}_wnmf{dim}_k{K}*pwcss*m15"
    )
    cands: list[Path] = []
    for root in ASSIGN_ROOTS:
        cands.extend(root.glob(pattern))
    cands = [p for p in cands if (p / "assignments.npy").is_file() and (p / "memberships.npy").is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def run_assign(preprocess: str, dim: int, *, jobs: int, skip_existing: bool) -> int:
    cmd = [
        sys.executable,
        str(GEN),
        "--dataset",
        "100k",
        "--algo",
        ALGO,
        "--no-prune",
        "--preprocess",
        preprocess,
        "--feature-extraction",
        "wnmf",
        "--svd-components",
        str(dim),
        "--wnmf-epochs",
        str(dim),
        "--legacy-wnmf-suffix",
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
    print("\nASSIGN:", preprocess, dim)
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def eval_one(preprocess: str, dim: int) -> dict | None:
    adir = assignment_dir(preprocess, dim)
    if adir is None:
        print(f"SKIP eval ({preprocess}, wnmf{dim}): assignment yok", flush=True)
        return None

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
        "preprocess": preprocess,
        "lof": True,
        "wnmf_dim": dim,
        "assignment_dir": str(adir),
        "mae": float(r["mae"]),
        "rmse": float(r["rmse"]),
        "ndcg_at_10": float(r["ndcg_at_10"]),
        "precision_at_10": float(r["precision_at_10"]),
        "recall_at_10": float(r["recall_at_10"]),
        "eval_seconds": round(time.time() - t0, 1),
    }


def save_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if OUT_CSV_PATH.is_file():
        old = pd.read_csv(OUT_CSV_PATH)
        key = ["algo", "k", "fold", "predictor", "similarity", "soft_threshold", "preprocess", "lof", "wnmf_dim"]
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=key, keep="last")
    df = df.sort_values(["preprocess", "wnmf_dim"]).reset_index(drop=True)
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV_PATH, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "eval", "all"], default="all")
    ap.add_argument(
        "--preprocess",
        nargs="+",
        default=None,
        choices=["none", "zscore"],
        help="Sadece bu preprocess degerleri (default: hepsi)",
    )
    ap.add_argument("--dims", type=int, nargs="+", default=None, help="WNMF boyutlari")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()
    preps = args.preprocess if args.preprocess else PREPROCS
    dims = args.dims if args.dims else DIMS

    rc = 0
    if args.phase in ("assign", "all"):
        for prep in preps:
            for dim in dims:
                rc = rc or run_assign(prep, dim, jobs=args.jobs, skip_existing=args.skip_existing)
        if rc != 0:
            return rc

    if args.phase in ("eval", "all"):
        rows: list[dict] = []
        for prep in preps:
            for dim in dims:
                row = eval_one(prep, dim)
                if row is not None:
                    rows.append(row)
                    print(
                        f"EVAL {prep} w{dim}: MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} "
                        f"NDCG={row['ndcg_at_10']:.4f}",
                        flush=True,
                    )
        if not rows:
            print("Yeni eval sonucu yok.")
            return 0

        df = save_rows(rows)
        best_new = float(df["mae"].min())
        base = baseline_best_k10_mae()
        print(f"\nCSV -> {OUT_CSV_PATH}")
        print(f"Best new MAE: {best_new:.4f}")
        if pd.notna(base):
            print(f"Baseline best K=10 MAE: {base:.4f}")
            print(f"Delta (new - baseline): {best_new - base:+.4f}")
        print(df[["preprocess", "wnmf_dim", "mae", "rmse", "ndcg_at_10"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
