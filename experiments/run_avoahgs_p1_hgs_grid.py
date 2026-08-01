"""
HA_AVOAHGS p1 × hgs_rate grid — WNMF50 / fuzzy / K=10 / m=1.5.

Atama: tek sefer (fold=1 train-only WNMF+fuzzy); CV eval'de 5 fold'da aynı assignment.
Protokol: _fuzzy_imkpp_nogs_trainonly_rand_wnmfep50_none_k10_m15

  python experiments/run_avoahgs_p1_hgs_grid.py --phase list
  python experiments/run_avoahgs_p1_hgs_grid.py --phase assign --skip-existing
  python experiments/run_avoahgs_p1_hgs_grid.py --phase eval-cv5
  python experiments/run_avoahgs_p1_hgs_grid.py --phase all --skip-existing
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mealpy"))

from generate_assignments import (  # noqa: E402
    HA_AVOAHGS_HGS_DEFAULT,
    HA_AVOAHGS_P1_DEFAULT,
    format_avoahgs_param_folder_suffix,
)

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"
OUT_CSV = REPO / "results" / "avoahgs_p1_hgs_grid_cv5.csv"

ALGO = "HA_AVOAHGS"
K = 10
WNMF_DIM = 50
FCM_M = 1.5
# Kümeleme: fold=1 train (canonical); klasör adında fold/dim yok
ASSIGN_FOLD = 1
FOLDS_CV5 = [1, 2, 3, 4, 5]

PRED = Namespace(similarity="pearson", min_common=5, soft_membership_threshold=0.1)

P1_LIST = [0.2, 0.4, 0.6, 0.8]
HGS_LIST = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]

BASE_SUFFIX = (
    f"_fuzzy_imkpp_nogs_trainonly_rand_wnmfep50_none_k{K}"
    f"_m{int(round(FCM_M * 10))}"
)


def _param_tag(p1: float, hgs_rate: float) -> str:
    if abs(p1 - HA_AVOAHGS_P1_DEFAULT) < 1e-9 and abs(hgs_rate - HA_AVOAHGS_HGS_DEFAULT) < 1e-9:
        return "(default, no extra suffix)"
    return format_avoahgs_param_folder_suffix(p1, hgs_rate).lstrip("_") or "(default)"


def assign_suffix(p1: float, hgs_rate: float) -> str:
    return f"{BASE_SUFFIX}{format_avoahgs_param_folder_suffix(p1, hgs_rate)}"


def assign_path(p1: float, hgs_rate: float) -> Path:
    return ASSIGN_ROOT / "ml100k" / f"{ALGO}{assign_suffix(p1, hgs_rate)}"


def phase_list() -> None:
    combos = list(itertools.product(P1_LIST, HGS_LIST))
    print(f"Grid: {len(combos)} kombinasyon ({len(P1_LIST)} p1 × {len(HGS_LIST)} hgs_rate)")
    print(f"Base: {ALGO}{BASE_SUFFIX}\n")
    for i, (p1, hgs) in enumerate(combos, 1):
        adir = assign_path(p1, hgs)
        exists = (adir / "assignments.npy").is_file()
        tag = _param_tag(p1, hgs)
        status = "EXISTS" if exists else "pending"
        print(f"  [{i:2d}] p1={p1:g} hgs={hgs:g}  {tag:28s}  {status}")


def phase_assign(skip_existing: bool) -> int:
    cmd_base = [
        sys.executable, str(GEN),
        "--dataset", "100k",
        "--algo", ALGO,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--init-mode", "mkpp",
        "--cluster-metric", "fuzzy",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", "random", "--fold", str(ASSIGN_FOLD),
        "--k", str(K),
        "--fcm-m", str(FCM_M),
        "--fcm-m-suffix",
        "--avoahgs-param-suffix",
    ]
    if skip_existing:
        cmd_base.append("--skip-existing")

    combos = list(itertools.product(P1_LIST, HGS_LIST))
    rc = 0
    for i, (p1, hgs) in enumerate(combos, 1):
        adir = assign_path(p1, hgs)
        if skip_existing and (adir / "assignments.npy").is_file():
            print(f"[{i}/{len(combos)}] SKIP existing p1={p1:g} hgs={hgs:g} -> {adir.name}", flush=True)
            continue
        cmd = cmd_base + ["--p1", str(p1), "--hgs-rate", str(hgs)]
        print(f"\n[{i}/{len(combos)}] p1={p1:g} hgs={hgs:g}", flush=True)
        print(" ".join(cmd), flush=True)
        ret = subprocess.run(cmd, cwd=str(REPO)).returncode
        rc = rc or ret
    return rc


def phase_eval_cv5() -> pd.DataFrame:
    from wnmf.wnmf_experiment import (
        RANDOM_SEED,
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        load_ratings_100k_all,
        run_cluster_average,
    )

    data = str(REPO / "data" / "ml-100k" / "u.data")
    rows: list[dict] = []
    combos = list(itertools.product(P1_LIST, HGS_LIST))

    for p1, hgs in combos:
        adir = assign_path(p1, hgs)
        if not (adir / "assignments.npy").is_file():
            print(f"SKIP missing assign: p1={p1:g} hgs={hgs:g}", flush=True)
            continue

        assignments, gray_mask = load_assignment(str(adir))
        memberships = load_memberships(str(adir))
        nc = _nearest_centroid_bundle(None, str(adir), assignments)

        for fold in FOLDS_CV5:
            train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=fold)
            n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
            n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
            a, g, m, _ = _align_assignment_bundle(
                assignments, gray_mask, memberships, None,
                n_users_expected=n_users, algo_label=ALGO, assign_dir=str(adir),
            )
            t0 = time.time()
            r = run_cluster_average(
                train, test, a, g, m, n_items, ALGO,
                **_cluster_avg_predict_kwargs(PRED),
                **nc,
                top_n=10,
                relevance_threshold=4.0,
                assign_dir=str(adir),
            )
            rows.append({
                "p1": p1,
                "hgs_rate": hgs,
                "fold": fold,
                "assign_fold": ASSIGN_FOLD,
                "assign_suffix": assign_suffix(p1, hgs),
                "predictor": "cluster_avg",
                "similarity": PRED.similarity,
                "min_common": PRED.min_common,
                "soft_threshold": PRED.soft_membership_threshold,
                "mae": r["mae"],
                "rmse": r["rmse"],
                "ndcg_at_10": r["ndcg_at_10"],
                "precision_at_10": r["precision_at_10"],
                "recall_at_10": r["recall_at_10"],
                "eval_seconds": round(time.time() - t0, 1),
            })
            print(
                f"  p1={p1:g} hgs={hgs:g} fold={fold}: "
                f"MAE={r['mae']:.4f} NDCG={r['ndcg_at_10']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg = df.groupby(["p1", "hgs_rate"], as_index=False).agg(
        mae=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse=("rmse", "mean"),
        ndcg_at_10=("ndcg_at_10", "mean"),
        ndcg_at_10_std=("ndcg_at_10", "std"),
    )
    for c in ["mae", "rmse", "ndcg_at_10"]:
        agg[c] = agg[c].round(4)
        agg[f"{c}_std"] = agg[f"{c}_std"].round(4)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV.with_name("avoahgs_p1_hgs_grid_cv5_folds.csv"), index=False)
    agg.to_csv(OUT_CSV, index=False)
    print(f"\n-> {OUT_CSV}", flush=True)
    best = agg.loc[agg["mae"].idxmin()]
    print(
        f"En iyi MAE (CV5 ort): p1={best['p1']:g} hgs={best['hgs_rate']:g} "
        f"MAE={best['mae']:.4f}±{best['mae_std']:.4f}",
        flush=True,
    )
    return agg


def main() -> int:
    p = argparse.ArgumentParser(description="HA_AVOAHGS p1 × hgs_rate grid")
    p.add_argument(
        "--phase",
        choices=["assign", "list", "eval-cv5", "all"],
        default="assign",
        help="assign: tek atama üret; eval-cv5: 5-fold CV (aynı assignment)",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="assignments.npy olan klasörleri atla (eski atamalar korunur)",
    )
    args = p.parse_args()

    if args.phase == "list":
        phase_list()
        return 0
    rc = 0
    if args.phase in ("assign", "all"):
        rc = phase_assign(skip_existing=args.skip_existing)
    if args.phase in ("eval-cv5", "all"):
        phase_eval_cv5()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
