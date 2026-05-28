"""
_euc_imkpp_nogs_none_wnmf* assignment'ları ile tam eval:
  - mode sharedV (cluster SharedV WNMF)
  - similarity cosine (cluster KNNBaseline)
  - cluster-avg-hard (calc_avg_rating)
  - cluster-knn-variant baseline

Örnek:
  python mealpy/run_euc_nogs_none_cosine_full_eval.py
  python mealpy/run_euc_nogs_none_cosine_full_eval.py --wnmf 20 --k 27 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "wnmf" / "wnmf_experiment.py"
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"

DEFAULT_ALGOS = ["B0_KMEANS", "B1_HHO", "B_AVOA", "HA_AVOAHGS", "IWO_HHO"]
DEFAULT_K = [5, 7, 10, 14, 21, 27, 30]
DEFAULT_WNMF = [20]


def resolve_python(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if _VENV_PY.is_file():
        return str(_VENV_PY)
    return sys.executable


def assign_suffix(wnmf_dim: int, cluster_k: int) -> str:
    return f"_euc_imkpp_nogs_none_wnmf{int(wnmf_dim)}_k{int(cluster_k)}_kmref"


def run_eval(
    wnmf_dims: list[int],
    ks: list[int],
    knn: int,
    algos: list[str],
    fold: int,
    algo_jobs: int | None,
    dry_run: bool,
    python: str,
) -> int:
    for w in wnmf_dims:
        for k in ks:
            cmd = [
                python,
                str(EXP),
                "--dataset", "100k",
                "--eval-split", "random",
                "--fold", str(fold),
                "--mode", "sharedV",
                "--no-global",
                "--cluster-avg-hard",
                "--cluster-knn-variant", "baseline",
                "--similarity", "cosine",
                "--knn", str(knn),
                "--k", str(k),
                "--algo", *algos,
                "--assign-root", str(ASSIGN_ROOT.as_posix()),
                "--assign-suffix", assign_suffix(w, k),
                "--top-n", "10",
                "--relevance-threshold", "4.0",
            ]
            if algo_jobs is not None:
                cmd.extend(["--algo-jobs", str(algo_jobs)])
            print("\n" + "=" * 72)
            print(" ".join(cmd))
            print("=" * 72)
            if dry_run:
                continue
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode
            if rc != 0:
                print(f"Hata: WNMF={w} K={k} rc={rc}", file=sys.stderr)
                return rc
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="euc/nogs/none/wnmf full eval (sharedV+cosine+hard avg+baseline)")
    p.add_argument("--wnmf", type=int, nargs="+", default=DEFAULT_WNMF)
    p.add_argument("--k", type=int, nargs="+", default=DEFAULT_K)
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--algo", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--algo-jobs", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", default=None)
    ns = p.parse_args()

    python = resolve_python(ns.python)
    t0 = time.time()
    print(f"Python: {python}")
    print(f"WNMF dims: {ns.wnmf}")
    print(f"K values : {ns.k}")
    print(f"kNN      : {ns.knn}")
    print(f"Algos    : {ns.algo}")
    rc = run_eval(
        ns.wnmf, ns.k, ns.knn, ns.algo, ns.fold, ns.algo_jobs, ns.dry_run, python,
    )
    print(f"\nBitti ({(time.time() - t0) / 60:.1f} dk), rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
