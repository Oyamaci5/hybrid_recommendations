"""
WNMF50, no-prune, no-gray-sheep, none+mkpp, K in {90,120,150},
HA_AVOAHGS + LIT_GWO, kmref on/off.

Suffix: _euc_imkpp_nogs_none_wnmf50_k{K}[_kmref]

Eval (official f1): cluster_avg + cluster_knn, knn = cluster_min (1) and 5.

  python experiments/run_nogs_wnmf50_k90_120_150_grid.py --phase assign
  python experiments/run_nogs_wnmf50_k90_120_150_grid.py --phase assign --no-kmref
  python experiments/run_nogs_wnmf50_k90_120_150_grid.py --phase eval
  python experiments/run_nogs_wnmf50_k90_120_150_grid.py --phase all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
WNMF = REPO / "wnmf" / "wnmf_experiment.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"

ALGOS = ["HA_AVOAHGS", "LIT_GWO"]
KS = [90, 120, 150]
WNMF_DIM = 50
FOLD = 1
EVAL_SPLIT = "official"


def assign_suffix(k: int, *, kmref: bool) -> str:
    base = f"_euc_imkpp_nogs_none_wnmf{WNMF_DIM}_k{k}"
    return base + ("_kmref" if kmref else "")


def phase_assign(kmref: bool, jobs: int, skip_existing: bool) -> int:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", *ALGOS,
        "--no-prune",
        "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--legacy-wnmf-suffix",
        "--cluster-metric", "euclidean",
        "--init-mode", "mkpp",
        "--k", *[str(k) for k in KS],
        "--jobs", str(jobs),
    ]
    if kmref:
        cmd.append("--kmeans-refine-overwrite")
    else:
        cmd.append("--no-kmeans-refine")
    if skip_existing:
        cmd.append("--skip-existing")
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def phase_eval(kmref: bool, skip_existing: bool) -> int:
    rc = 0
    for k in KS:
        suf = assign_suffix(k, kmref=kmref)
        cmd = [
            sys.executable, "-u", str(WNMF),
            "--dataset", "100k",
            "--eval-split", EVAL_SPLIT,
            "--fold", str(FOLD),
            "--assign-root", str(ASSIGN_ROOT),
            "--assign-suffix", suf,
            "--k", str(k),
            "--latent-dim", str(WNMF_DIM),
            "--mode", "baselines",
            "--no-global",
            "--knn", "1", "5",
            "--similarity", "cosine",
            "--min-common", "3",
            "--nearest-centroid",
            "--centroid-metric", "euclidean",
            "--algo", *ALGOS,
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
    ap.add_argument("--phase", choices=["assign", "eval", "all"], default="all")
    ap.add_argument(
        "--kmref-mode",
        choices=["both", "on", "off"],
        default="both",
        help="assign/eval: kmref+no-kmref (both), sadece kmref (on), sadece no-kmref (off)",
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    kmref_flags = [True, False] if args.kmref_mode == "both" else [args.kmref_mode == "on"]

    for kmref in kmref_flags:
        tag = "kmref" if kmref else "no_kmref"
        print(f"\n{'=' * 70}\n  {tag.upper()}  suffix=*_nogs_none_wnmf50_k{{90,120,150}}{'_kmref' if kmref else ''}\n{'=' * 70}")

        if args.phase in ("assign", "all"):
            rc = phase_assign(kmref, args.jobs, args.skip_existing)
            if rc != 0:
                sys.exit(rc)
        if args.phase in ("eval", "all"):
            rc = phase_eval(kmref, args.skip_existing)
            if rc != 0:
                sys.exit(rc)

    print("\nBitti.")


if __name__ == "__main__":
    main()
