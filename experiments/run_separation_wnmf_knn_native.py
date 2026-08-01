"""
Separation-protocol: cluster_knn_native_baseline (--no-cluster-avg), sharedV/global yok.
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
    ("euc_knnmae", "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}_knnmae", [10]),
]


def _complete(suffix_tmpl: str, k: int) -> bool:
    suf = suffix_tmpl.format(k=k)
    return all(
        (ASSIGN_ROOT / "ml100k" / f"{algo}{suf}" / "assignments.npy").is_file()
        for algo in ALGOS
    )


def main() -> None:
    py = sys.executable
    for protocol, suffix_tmpl, ks in JOBS:
        for k in ks:
            if not _complete(suffix_tmpl, k):
                print(f"SKIP {protocol} K={k} (eksik atama)")
                continue
            suf = suffix_tmpl.format(k=k)
            cmd = [
                py, str(EXP),
                "--dataset", "100k",
                "--mode", "baselines",
                "--no-global",
                "--no-cluster-avg",
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
            print(f"{protocol}  K={k}  (cluster_knn_native_baseline)")
            print(" ".join(cmd))
            print("=" * 72)
            if subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
                sys.exit(1)
    print(
        "\nBitti. CSV: results/wnmf/ml100k/k<K>/fold1/run*/ "
        "(Senaryo=cluster_knn_native_baseline)"
    )


if __name__ == "__main__":
    main()
