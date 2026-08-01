"""
K=3..24, 8 algo, official fold 2-5: atama + cluster_avg_soft eval + fold ortalamasi.

  python -m experiments.run_fuzzy_official_folds2_5_k3_24 --phase status
  python -m experiments.run_fuzzy_official_folds2_5_k3_24 --phase all --jobs 3
  python -m experiments.run_fuzzy_official_folds2_5_k3_24 --phase aggregate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import (
    ALL_ALGOS,
    K3_24_LIST,
    assign_dir,
    build_assign_cmd,
)
from experiments.run_fuzzy_official_f1_eval import DEFAULT_SOFT, eval_k

FOLDS = [2, 3, 4, 5]
OUT_ALL = REPO / "results" / "fuzzy_official_folds2_5_k3_24_cluster_avg_soft.csv"
OUT_AVG = REPO / "results" / "fuzzy_official_folds2_5_avg_k3_24_cluster_avg_soft.csv"
OUT_PIVOT = REPO / "results" / "fuzzy_official_folds2_5_avg_mae_pivot.csv"
EVAL_KEY = ["fold", "k", "algo", "predictor", "knn_k", "similarity", "fast", "prune"]
METRICS = ["mae", "rmse", "ndcg_at_10", "precision_at_10", "recall_at_10"]


def assign_ready(algo: str, k: int, fold: int, *, fast: bool = True) -> bool:
    d = assign_dir(algo, k, fold=fold, fast=fast)
    return (d / "assignments.npy").is_file() and (d / "memberships.npy").is_file()


def load_done_keys() -> set[tuple]:
    if not OUT_ALL.is_file():
        return set()
    df = pd.read_csv(OUT_ALL)
    keys: set[tuple] = set()
    for _, r in df.iterrows():
        keys.add((
            int(r["fold"]), int(r["k"]), str(r["algo"]),
            str(r.get("predictor", "cluster_avg_soft")),
            int(r.get("knn_k", 0)),
            str(r.get("similarity", "cosine")),
            bool(r.get("fast", True)),
            bool(r.get("prune", False)),
        ))
    return keys


def status_report() -> None:
    done = load_done_keys()
    for fold in FOLDS:
        miss_a = miss_e = 0
        for k in K3_24_LIST:
            for algo in ALL_ALGOS:
                if not assign_ready(algo, k, fold):
                    miss_a += 1
                key = (fold, k, algo, "cluster_avg_soft", 0, "cosine", True, False)
                if key not in done:
                    miss_e += 1
        tot = len(K3_24_LIST) * len(ALL_ALGOS)
        print(
            f"Fold {fold}: atama {tot - miss_a}/{tot}, "
            f"eval {tot - miss_e}/{tot}",
            flush=True,
        )


def phase_assign(fold: int, jobs: int, skip_existing: bool) -> int:
    cmd = build_assign_cmd(
        algos=ALL_ALGOS,
        k_list=K3_24_LIST,
        prune=False,
        jobs=jobs,
        skip_existing=skip_existing,
        fast=True,
        fold=fold,
    )
    print(f"\n=== ASSIGN fold={fold} K=3..24 ===", flush=True)
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def phase_eval_fold(fold: int, *, similarity: str = "cosine", soft: float = DEFAULT_SOFT) -> pd.DataFrame:
    done = load_done_keys()
    parts: list[pd.DataFrame] = []
    for k in K3_24_LIST:
        need = [
            a for a in ALL_ALGOS
            if (fold, k, a, "cluster_avg_soft", 0, similarity, True, False) not in done
        ]
        if not need:
            continue
        ready = [a for a in need if assign_ready(a, k, fold)]
        if not ready:
            print(f"fold {fold} K={k}: atama yok, skip", flush=True)
            continue
        print(f"\n--- fold {fold} eval K={k} n={len(ready)} ---", flush=True)
        df = eval_k(
            k,
            soft_threshold=soft,
            similarity=similarity,
            algos=ready,
            prune=False,
            fast=True,
            fold=fold,
            predictors=("cluster_avg_soft",),
        )
        if not df.empty:
            parts.append(df)
            _append_eval_rows(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _append_eval_rows(df: pd.DataFrame) -> None:
    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ALL.is_file():
        old = pd.read_csv(OUT_ALL)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out = out.drop_duplicates(
        subset=EVAL_KEY + ["soft_threshold", "fcm_m", "wnmf_dim"],
        keep="last",
    )
    out = out.sort_values(["fold", "k", "algo"]).reset_index(drop=True)
    out.to_csv(OUT_ALL, index=False)
    print(f"  -> {OUT_ALL} ({len(out)} rows)", flush=True)


def phase_aggregate() -> pd.DataFrame:
    if not OUT_ALL.is_file():
        print(f"CSV yok: {OUT_ALL}", flush=True)
        return pd.DataFrame()
    df = pd.read_csv(OUT_ALL)
    sub = df[
        (df.get("predictor", "") == "cluster_avg_soft")
        & (df.get("similarity", "") == "cosine")
        & (df["fold"].isin(FOLDS))
    ]
    g = sub.groupby(["k", "algo"])
    agg = g[METRICS].mean().add_suffix("_avg_folds2_5")
    for m in METRICS:
        agg[f"{m}_std_folds2_5"] = g[m].std()
    agg["n_folds"] = g["fold"].nunique()
    agg = agg.reset_index().sort_values(["k", "mae_avg_folds2_5"]).reset_index(drop=True)
    agg.to_csv(OUT_AVG, index=False)

    pivot = agg.pivot_table(
        index="algo", columns="k", values="mae_avg_folds2_5", aggfunc="first",
    )
    pivot = pivot.reindex(ALL_ALGOS)
    pivot.to_csv(OUT_PIVOT)
    print(f"All folds -> {OUT_ALL} ({len(sub)} rows)", flush=True)
    print(f"Avg f2-5 -> {OUT_AVG} ({len(agg)} rows)", flush=True)
    print(f"Pivot -> {OUT_PIVOT}", flush=True)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["status", "assign", "eval", "aggregate", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--folds", type=int, nargs="+", default=FOLDS)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()
    folds = [f for f in args.folds if 2 <= f <= 5]

    if args.phase == "status":
        status_report()
        return

    if args.phase in ("assign", "all"):
        for fold in folds:
            rc = phase_assign(fold, args.jobs, args.skip_existing)
            if rc != 0:
                sys.exit(rc)

    if args.phase in ("eval", "all"):
        for fold in folds:
            phase_eval_fold(fold)
        phase_aggregate()

    if args.phase == "aggregate":
        phase_aggregate()


if __name__ == "__main__":
    main()
