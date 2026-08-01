"""
WNMF yok: ham rating matrisi (--preprocess none --feature-extraction none).
Separation ile aynı protokol: euc + imkpp + nogs + trainonly rand f1, no-prune.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"
ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
# Klasör: ..._euc_imkpp_nogs_trainonly_rand_f1_none_none20_k{K}
SUFFIX_TMPL = "_euc_imkpp_nogs_trainonly_rand_f1_none_none20_k{k}"
DEFAULT_KS = [10, 14, 21]


def cluster_report(k: int) -> None:
    print(f"\n{'=' * 72}\nK={k}  (feature=none, preprocess=none)\n{'=' * 72}")
    suf = SUFFIX_TMPL.format(k=k)
    for algo in ALGOS:
        p = ASSIGN_ROOT / f"{algo}{suf}" / "assignments.npy"
        if not p.is_file():
            print(f"  {algo}: MISSING")
            continue
        c = Counter(np.load(p).astype(int).tolist())
        sz = sorted(c.values())
        sing = sum(1 for s in sz if s == 1)
        le3 = sum(1 for s in sz if s <= 3)
        print(
            f"  {algo}: active={len(sz)}/{k} min={min(sz)} max={max(sz)} "
            f"mean={np.mean(sz):.1f} std={np.std(sz):.1f} sing={sing} le3={le3}"
        )
        if len(sz) <= 22:
            print(f"    sizes: {sz}")
        else:
            print(f"    sizes[:8]: {sz[:8]} ... tail: {sz[-5:]}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "report", "all"], default="all")
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if args.phase in ("assign", "all"):
        cmd = [
            sys.executable, str(GEN),
            "--dataset", "100k",
            "--algo", *ALGOS,
            "--no-prune", "--no-gray-sheep",
            "--preprocess", "none",
            "--feature-extraction", "none",
            "--svd-components", "20",
            "--init-mode", "mkpp",
            "--cluster-metric", "euclidean",
            "--fitness", "wcss",
            "--cluster-objective", "multi",
            "--train-only", "--eval-split", "random", "--fold", "1",
            "--k", *[str(k) for k in args.k],
            "--jobs", str(args.jobs),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        print(" ".join(cmd))
        if subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
            sys.exit(1)

    if args.phase in ("report", "all"):
        for k in args.k:
            cluster_report(k)


if __name__ == "__main__":
    main()
