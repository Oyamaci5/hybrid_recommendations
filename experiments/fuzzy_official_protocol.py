"""FCM official fold-1: ortak suffix / sabitler."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"
ASSIGN_ROOT_FAST = REPO / "mealpy" / "results" / "assignments_estop" / "ml100k"
GEN = REPO / "mealpy" / "generate_assignments.py"

FCM_M = 1.5
WNMF_DIM = 50
FOLD = 1
PRUNE_USER = 5
PRUNE_ITEM = 10

# Hizli K taramasi (official f1, fuzzy, wnmf50, mkpp, nogs, no-prune)
FAST_K_LIST = [3, 5, 7, 9, 11, 13, 15, 17, 22]
FAST_ALGOS = ["HA_AVOAHGS", "LIT_GWO", "B_AVOA"]
# K=4,6,8,10,12,14 ek algo karsilastirmasi (fast estop)
MID_K_LIST = [4, 6, 8, 10, 12, 14]
EXTRA_ALGOS = ["LIT_PSO", "IWO_HHO", "H9_QSA+CDO", "B1_HHO", "B2_HGS"]
K3_24_LIST = list(range(3, 25))
ALL_ALGOS = list(dict.fromkeys([*FAST_ALGOS, *EXTRA_ALGOS]))


def expected_suffix(
    k: int,
    *,
    fold: int = FOLD,
    prune: bool = False,
    fast: bool = False,
) -> str:
    prune_part = f"_pruneu{PRUNE_USER}_i{PRUNE_ITEM}" if prune else ""
    pwcss = "_pwcss" if fast else ""
    return (
        f"{prune_part}_fuzzy_imkpp_nogs_trainonly_official_f{int(fold)}"
        f"_none_wnmf{WNMF_DIM}_k{k}{pwcss}_m15"
    )


def assign_dir(
    algo: str,
    k: int,
    *,
    fold: int = FOLD,
    prune: bool = False,
    fast: bool = False,
) -> Path:
    root = ASSIGN_ROOT_FAST if fast else ASSIGN_ROOT
    return root / f"{algo}{expected_suffix(k, fold=fold, prune=prune, fast=fast)}"


def build_assign_cmd(
    *,
    algos: list[str],
    k_list: list[int],
    prune: bool,
    jobs: int,
    skip_existing: bool,
    fast: bool = False,
    fold: int = FOLD,
) -> list[str]:
    import sys

    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", *algos,
        "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--wnmf-epochs", str(WNMF_DIM),
        "--legacy-wnmf-suffix",
        "--init-mode", "mkpp",
        "--cluster-metric", "fuzzy",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", "official", "--fold", str(int(fold)),
        "--fcm-m", str(FCM_M),
        "--fcm-m-suffix",
        "--k", *[str(k) for k in k_list],
        "--jobs", str(jobs),
    ]
    if prune:
        cmd.extend(["--min-user-ratings", str(PRUNE_USER), "--min-item-ratings", str(PRUNE_ITEM)])
    else:
        cmd.append("--no-prune")
    if skip_existing:
        cmd.append("--skip-existing")
    # Hizli K taramasi: ~4-8x daha kisa (epoch/pop dusuk, wcss-only, early-stop).
    if fast:
        cmd.extend([
            "--baseline-epoch", "40",
            "--pop-size", "25",
            "--early-stop",
            "--early-stop-patience", "4",
            "--early-stop-block", "5",
            "--cluster-objective", "wcss",
        ])
    return cmd
