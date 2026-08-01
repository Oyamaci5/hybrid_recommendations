"""
K-means (B0) vs B1_HHO — euclidean WNMF50, nogs, uzun converge.

  python -m experiments.run_kmeans_vs_b1_hho_euc_converge --phase all --jobs 1
  python -m experiments.run_kmeans_vs_b1_hho_euc_converge --phase compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
ROOT = REPO / "mealpy" / "results" / "assignments_estop" / "ml100k"
OUT = REPO / "results" / "kmeans_vs_b1_hho_euc_converge_fold2.csv"

FOLD = 2
K_LIST = [10, 14, 16, 20]
EPOCH = 200
POP = 50
# Uzun converge: blok 10, patience 12 => en fazla ~120 epoch erken durabilir; max 500
ES_MAX = 500
ES_PATIENCE = 12
ES_BLOCK = 10
ES_TOL = 1e-6


def _suffix(k: int) -> str:
    return (
        f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
        f"_none_wnmf50_k{k}_pwcss"
    )


def assign_dir(algo: str, k: int) -> Path:
    return ROOT / f"{algo}{_suffix(k)}"


def run_assign(*, jobs: int, skip_existing: bool) -> int:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", "B0_KMEANS", "B1_HHO",
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", "50", "--wnmf-epochs", "50",
        "--legacy-wnmf-suffix",
        "--init-mode", "mkpp",
        "--cluster-metric", "euclidean",
        "--fitness", "wcss",
        "--cluster-objective", "wcss",
        "--train-only", "--eval-split", "official",
        "--fold", str(FOLD),
        "--k", *[str(k) for k in K_LIST],
        "--jobs", str(jobs),
        "--baseline-epoch", str(EPOCH),
        "--pop-size", str(POP),
        "--early-stop",
        "--early-stop-max-epoch", str(ES_MAX),
        "--early-stop-patience", str(ES_PATIENCE),
        "--early-stop-block", str(ES_BLOCK),
        "--early-stop-tolerance", str(ES_TOL),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def _wcss_from_dir(d: Path, k: int) -> float | None:
    sys.path.insert(0, str(REPO / "mealpy"))
    from mealpy_comparison_v2 import compute_wcss_fast

    sol_path = d / "best_sol.npy"
    uf = d / "user_features.npy"
    if not sol_path.is_file() or not uf.is_file():
        return None
    X = np.load(uf)
    sol = np.load(sol_path)
    wcss, _ = compute_wcss_fast(X, sol, k, metric="euclidean")
    return float(wcss)


def phase_compare() -> None:
    sys.path.insert(0, str(REPO / "mealpy"))
    rows = []
    for k in K_LIST:
        d0 = assign_dir("B0_KMEANS", k)
        d1 = assign_dir("B1_HHO", k)
        w0 = _wcss_from_dir(d0, k)
        w1 = _wcss_from_dir(d1, k)
        ch_path = d1 / "convergence_history.csv"
        b1_init = b1_end = blocks = epochs = None
        gain_pct = gain_abs = None
        if ch_path.is_file():
            h = pd.read_csv(ch_path)
            b1_init = float(h["fitness"].iloc[0])
            b1_end = float(h["fitness"].iloc[-1])
            blocks = len(h)
            epochs = int(h["epoch"].iloc[-1])
            if b1_init and b1_init < 1e5:
                gain_abs = b1_init - b1_end
                gain_pct = gain_abs / b1_init * 100.0
        imp_vs_km = None
        if w0 is not None and w1 is not None and w0 > 0:
            imp_vs_km = (w0 - w1) / w0 * 100.0
        rows.append({
            "k": k,
            "kmeans_wcss": w0,
            "hho_final_wcss": w1,
            "hho_vs_kmeans_pct": imp_vs_km,
            "hho_conv_b1": b1_init,
            "hho_conv_end": b1_end,
            "hho_meta_gain_pct": gain_pct,
            "hho_meta_gain_abs": gain_abs,
            "hho_blocks": blocks,
            "hho_epochs": epochs,
            "kmeans_dir": str(d0),
            "hho_dir": str(d1),
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "compare", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()
    if args.phase in ("assign", "all"):
        rc = run_assign(jobs=args.jobs, skip_existing=args.skip_existing)
        if rc != 0:
            sys.exit(rc)
    if args.phase in ("compare", "all"):
        phase_compare()


if __name__ == "__main__":
    main()
