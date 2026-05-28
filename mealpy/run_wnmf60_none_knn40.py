"""wnmf60 + feature-extraction none koşuları; kNN=40 eval."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)
GEN = REPO / "mealpy" / "generate_assignments.py"
EXP = REPO / "wnmf" / "wnmf_experiment.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"

ALGOS = ["B0_KMEANS", "B1_HHO", "B_AVOA", "HA_AVOAHGS", "IWO_HHO"]
KS = [5, 7, 10, 14, 21, 27, 30]
KNN = 40


def run(cmd: list) -> int:
    print("\n" + "=" * 72)
    print(" ".join(str(c) for c in cmd))
    print("=" * 72)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def gen_assign(feature: str, svd: int | None = None) -> int:
    cmd = [
        str(PY), str(GEN),
        "--dataset", "100k",
        "--algo", *ALGOS,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", feature,
        "--init-mode", "mkpp",
        "--kmeans-refine",
        "--k", *[str(k) for k in KS],
        "--jobs", "4",
        "--skip-existing",
    ]
    if feature == "wnmf":
        cmd += ["--svd-components", str(svd)]
    return run(cmd)


def eval_suffix(feature: str, k: int, svd: int = 20) -> str:
    if feature == "wnmf":
        return f"_euc_imkpp_nogs_none_wnmf{svd}_k{k}_kmref"
    return f"_imkpp_nogs_none_none{svd}_k{k}_kmref"


def run_eval(feature: str, svd: int = 20) -> int:
    for k in KS:
        cmd = [
            str(PY), str(EXP),
            "--dataset", "100k",
            "--eval-split", "random",
            "--fold", "1",
            "--mode", "baselines",
            "--no-global", "--no-cluster-avg",
            "--cluster-knn-variant", "baseline",
            "--similarity", "cosine",
            "--knn", str(KNN),
            "--k", str(k),
            "--algo", *ALGOS,
            "--assign-root", str(ASSIGN_ROOT.as_posix()),
            "--assign-suffix", eval_suffix(feature, k, svd),
            "--top-n", "10",
            "--relevance-threshold", "4.0",
        ]
        rc = run(cmd)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    print(f"Python: {PY}")
    if gen_assign("wnmf", 60) != 0:
        return 1
    if gen_assign("none") != 0:
        return 1
    if run_eval("wnmf", 60) != 0:
        return 1
    return run_eval("none", 20)


if __name__ == "__main__":
    raise SystemExit(main())
