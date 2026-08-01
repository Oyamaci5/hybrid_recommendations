"""
Fold 3 eval bitince fold 4-5 atama (eksikse) + eval.

  python -m experiments.run_fuzzy_official_fold45_after_fold3
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import ALL_ALGOS, K3_24_LIST
from experiments.run_fuzzy_official_folds2_5_k3_24 import (
    OUT_ALL,
    assign_ready,
    phase_assign,
    phase_eval_fold,
)

FOLDS_AFTER = [4, 5]
POLL_SEC = 30


def eval_count(fold: int) -> int:
    if not OUT_ALL.is_file():
        return 0
    import pandas as pd

    df = pd.read_csv(OUT_ALL)
    return len(
        df[
            (df["fold"] == fold)
            & (df.get("predictor", "") == "cluster_avg_soft")
            & (df.get("similarity", "") == "cosine")
        ]
    )


def assign_count(fold: int) -> int:
    return sum(1 for k in K3_24_LIST for a in ALL_ALGOS if assign_ready(a, k, fold))


def wait_fold3_eval() -> None:
    need = len(K3_24_LIST) * len(ALL_ALGOS)
    print(f"Fold 3 eval bekleniyor ({need} satir)...", flush=True)
    while True:
        n = eval_count(3)
        print(f"  fold 3 eval: {n}/{need}", flush=True)
        if n >= need:
            print("Fold 3 eval tamam.", flush=True)
            return
        time.sleep(POLL_SEC)


def wait_assign(fold: int) -> None:
    need = len(K3_24_LIST) * len(ALL_ALGOS)
    while assign_count(fold) < need:
        n = assign_count(fold)
        print(f"  fold {fold} atama: {n}/{need}", flush=True)
        if n >= need:
            break
        time.sleep(POLL_SEC)
    if assign_count(fold) < need:
        print(f"=== ASSIGN fold={fold} (eksik) ===", flush=True)
        rc = phase_assign(fold, jobs=3, skip_existing=True)
        if rc != 0:
            sys.exit(rc)


def main() -> None:
    wait_fold3_eval()

    for fold in FOLDS_AFTER:
        wait_assign(fold)
        print(f"\n=== EVAL fold={fold} ===", flush=True)
        phase_eval_fold(fold)

    print("\nBitti.", flush=True)


if __name__ == "__main__":
    main()
