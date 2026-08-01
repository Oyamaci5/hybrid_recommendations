"""
Tamamlanmış separation-protocol atamaları üzerinde wnmf_experiment baselines
(küme avg + küme kNN; global/sharedV/full YOK).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "wnmf" / "wnmf_experiment.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"
ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]

JOBS = [
    ("euc_multi", "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}", [10, 14, 21]),
    ("fuzzy_fcm", "_fuzzy_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}", [10, 14, 21]),
    ("euc_knnmae", "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}_knnmae", [10, 14]),
]


def _complete(protocol: str, suffix_tmpl: str, k: int) -> bool:
    suf = suffix_tmpl.format(k=k)
    for algo in ALGOS:
        if not (ASSIGN_ROOT / "ml100k" / f"{algo}{suf}" / "assignments.npy").is_file():
            return False
    return True


def main() -> None:
    py = sys.executable
    for protocol, suffix_tmpl, ks in JOBS:
        for k in ks:
            if not _complete(protocol, suffix_tmpl, k):
                print(f"SKIP {protocol} K={k} (eksik atama)")
                continue
            suf = suffix_tmpl.format(k=k)
            cmd = [
                py, str(EXP),
                "--dataset", "100k",
                "--mode", "baselines",
                "--no-global",
                "--k", str(k),
                "--algo", *ALGOS,
                "--assign-root", str(ASSIGN_ROOT),
                "--assign-suffix", suf,
                "--eval-split", "random",
                "--fold", "1",
                "--similarity", "cosine",
                "--knn", "20",
                "--min-common", "3",
                "--cluster-knn-backend", "native",
                "--jobs", "4",
            ]
            print("\n" + "=" * 72)
            print(f"{protocol}  K={k}")
            print(" ".join(cmd))
            print("=" * 72)
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode
            if rc != 0:
                sys.exit(rc)
    print("\nBitti. Sonuçlar: results/wnmf/ml100k/k<K>/run*/")


if __name__ == "__main__":
    main()
