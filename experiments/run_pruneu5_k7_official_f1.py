"""
Champion pipeline: _pruneu5_i10_zscore_euc_imkpp_none_wnmf30_k7 + official fold 1.

Atama (train-only u1.base) -> WNMF tahmin (u1.base / u1.test).

  python experiments/run_pruneu5_k7_official_f1.py --phase all
  python experiments/run_pruneu5_k7_official_f1.py --phase assign --algo HA_AVOAHGS
  python experiments/run_pruneu5_k7_official_f1.py --phase eval --skip-existing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
WNMF = REPO / "wnmf" / "wnmf_experiment.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments_lof"
FOLD = 1
K = 7
WNMF_DIM = 30
EVAL_SPLIT = "official"

# Champion: prune u5/i10, zscore, euc, imkpp, LOF, wnmf30, k7 (kmref yok)
ASSIGN_SUFFIX = (
    "_pruneu5_i10_zscore_euc_imkpp_trainonly_official_f1_none_wnmf30_k7"
)
DEFAULT_ALGOS = ["HA_AVOAHGS", "B0_KMEANS", "B1_HHO", "IWO_HHO", "LIT_GWO"]


def assign_dir(algo: str) -> Path:
    return ASSIGN_ROOT / "ml100k" / f"{algo}{ASSIGN_SUFFIX}"


def phase_assign(algo: list[str], jobs: int, skip_existing: bool) -> int:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", *algo,
        "--lof",
        "--zscore",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--legacy-wnmf-suffix",
        "--cluster-metric", "euclidean",
        "--init-mode", "mkpp",
        "--k", str(K),
        "--train-only", "--eval-split", EVAL_SPLIT, "--fold", str(FOLD),
        "--jobs", str(jobs),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def phase_eval(algo: list[str], skip_existing: bool) -> int:
    rc = 0
    for a in algo:
        adir = assign_dir(a)
        if not (adir / "assignments.npy").is_file():
            print(f"SKIP eval {a}: atama yok -> {adir}", flush=True)
            rc = 1
            continue
        cmd = [
            sys.executable, "-u", str(WNMF),
            "--dataset", "100k",
            "--eval-split", EVAL_SPLIT,
            "--fold", str(FOLD),
            "--assign-root", str(ASSIGN_ROOT),
            "--assign-suffix", ASSIGN_SUFFIX,
            "--k", str(K),
            "--latent-dim", str(WNMF_DIM),
            "--mode", "baselines",
            "--no-global",
            "--knn", "30",
            "--similarity", "cosine",
            "--min-common", "3",
            "--nearest-centroid",
            "--centroid-metric", "euclidean",
            "--algo", a,
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            rc = r.returncode
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase", choices=["assign", "eval", "all"], default="all",
    )
    ap.add_argument("--algo", nargs="*", default=DEFAULT_ALGOS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("PRUNEU5 K7 — OFFICIAL F1")
    print("=" * 70)
    print(f"  suffix     : {ASSIGN_SUFFIX}")
    print(f"  assign dir : {ASSIGN_ROOT / 'ml100k' / ('<ALGO>' + ASSIGN_SUFFIX)}")
    print(f"  algos      : {args.algo}")
    print(f"  fold       : {FOLD} (u{FOLD}.base train-only, u{FOLD}.base/test eval)")

    if args.phase in ("assign", "all"):
        print("\n>>> PHASE: assign", flush=True)
        rc = phase_assign(args.algo, args.jobs, args.skip_existing)
        if rc != 0:
            sys.exit(rc)

    if args.phase in ("eval", "all"):
        print("\n>>> PHASE: eval", flush=True)
        rc = phase_eval(args.algo, args.skip_existing)
        if rc != 0:
            sys.exit(rc)

    print("\nBitti.")


if __name__ == "__main__":
    main()
