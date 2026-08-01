"""
K=3..24, tüm algo: eksik atama + cluster_avg_soft eval (fast estop).

  python -m experiments.run_fuzzy_official_k3_24_cluster_avg --phase status
  python -m experiments.run_fuzzy_official_k3_24_cluster_avg --phase assign --jobs 3
  python -m experiments.run_fuzzy_official_k3_24_cluster_avg --phase eval
  python -m experiments.run_fuzzy_official_k3_24_cluster_avg --phase eval --k-max 18
  python -m experiments.run_fuzzy_official_k3_24_cluster_avg --phase all --jobs 3
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
    EXTRA_ALGOS,
    FAST_ALGOS,
    assign_dir,
    build_assign_cmd,
)
from experiments.run_fuzzy_official_f1_eval import DEFAULT_SOFT, eval_k

K3_24_LIST = list(range(3, 25))
ALL_ALGOS = list(dict.fromkeys([*FAST_ALGOS, *EXTRA_ALGOS]))
OUT_CSV = REPO / "results" / "fuzzy_official_k3_24_cluster_avg_soft.csv"
MERGE_CSVS = [
    REPO / "results" / "fuzzy_official_fast_preds.csv",
    REPO / "results" / "fuzzy_official_k4_14_extra_preds.csv",
    REPO / "results" / "fuzzy_official_f1_cluster_avg_soft.csv",
]
EVAL_KEY = ["fold", "k", "algo", "predictor", "knn_k", "similarity", "fast", "prune"]


def assign_ready(algo: str, k: int, *, fast: bool = True) -> bool:
    d = assign_dir(algo, k, fast=fast)
    return (d / "assignments.npy").is_file() and (d / "memberships.npy").is_file()


def load_done_eval_keys() -> set[tuple]:
    keys: set[tuple] = set()
    paths = [OUT_CSV, *MERGE_CSVS]
    for p in paths:
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "fast" not in df.columns:
            df["fast"] = False
        if "prune" not in df.columns:
            df["prune"] = False
        if "knn_k" not in df.columns:
            df["knn_k"] = 0
        sub = df[
            (df.get("predictor", "") == "cluster_avg_soft")
            & (df.get("similarity", "") == "cosine")
            & (df.get("fast", False) == True)
        ]
        for _, r in sub.iterrows():
            keys.add((
                int(r["fold"]), int(r["k"]), str(r["algo"]),
                "cluster_avg_soft", 0, "cosine", True, False,
            ))
    return keys


def resolve_k_list(
    k_values: list[int] | None,
    *,
    k_min: int | None,
    k_max: int | None,
) -> list[int]:
    if k_values:
        return [k for k in k_values if k in K3_24_LIST]
    lo = k_min if k_min is not None else min(K3_24_LIST)
    hi = k_max if k_max is not None else max(K3_24_LIST)
    return [k for k in K3_24_LIST if lo <= k <= hi]


def status_report(k_list: list[int] | None = None) -> tuple[int, int, int, int]:
    ks = k_list if k_list is not None else K3_24_LIST
    miss_a = miss_e = 0
    n_assign = len(ks) * len(ALL_ALGOS)
    n_eval = n_assign
    done_eval = load_done_eval_keys()
    for k in ks:
        for algo in ALL_ALGOS:
            if not assign_ready(algo, k):
                miss_a += 1
            key = (1, k, algo, "cluster_avg_soft", 0, "cosine", True, False)
            if key not in done_eval:
                miss_e += 1
    return miss_a, n_assign, miss_e, n_eval


def phase_assign(jobs: int, skip_existing: bool) -> int:
    cmd = build_assign_cmd(
        algos=ALL_ALGOS,
        k_list=K3_24_LIST,
        prune=False,
        jobs=jobs,
        skip_existing=skip_existing,
        fast=True,
    )
    print("ASSIGN", "K=3..24", "algos=", ALL_ALGOS, flush=True)
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def phase_eval(
    similarity: str = "cosine",
    soft: float = DEFAULT_SOFT,
    k_list: list[int] | None = None,
) -> int:
    ks = k_list if k_list is not None else K3_24_LIST
    done = load_done_eval_keys()
    parts = []
    for k in ks:
        need = [a for a in ALL_ALGOS if (1, k, a, "cluster_avg_soft", 0, similarity, True, False) not in done]
        if not need:
            print(f"SKIP eval K={k} (tamam)", flush=True)
            continue
        ready = [a for a in need if assign_ready(a, k)]
        skip = [a for a in need if a not in ready]
        if skip:
            print(f"SKIP eval K={k} atama yok: {skip}", flush=True)
        if not ready:
            continue
        print(f"\n--- eval K={k} algos={ready} ---", flush=True)
        df = eval_k(
            k, soft_threshold=soft, similarity=similarity,
            algos=ready, prune=False, fast=True,
            predictors=("cluster_avg_soft",),
        )
        if not df.empty:
            parts.append(df)

    if not parts:
        print("Yeni eval satiri yok.", flush=True)
        return 0

    out = pd.concat(parts, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.is_file():
        old = pd.read_csv(OUT_CSV)
        out = pd.concat([old, out], ignore_index=True)
    out = out.drop_duplicates(subset=EVAL_KEY + ["soft_threshold", "fcm_m", "wnmf_dim"], keep="last")
    out = out.sort_values(["k", "algo"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nCSV -> {OUT_CSV} ({len(out)} rows)", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["status", "assign", "eval", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=None, help="K alt kumesi (or. 3 4 5)")
    ap.add_argument("--k-min", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()
    k_list = resolve_k_list(args.k, k_min=args.k_min, k_max=args.k_max)
    k_label = f"K={k_list[0]}..{k_list[-1]}" if k_list else "K=empty"

    miss_a, tot_a, miss_e, tot_e = status_report(k_list)
    print(
        f"Grid {k_label} x {len(ALL_ALGOS)} algo\n"
        f"  Atama: {tot_a - miss_a}/{tot_a} hazir, eksik={miss_a}\n"
        f"  Eval cluster_avg_soft cosine: {tot_e - miss_e}/{tot_e} hazir, eksik={miss_e}",
        flush=True,
    )

    if args.phase == "status":
        return

    if args.phase in ("assign", "all"):
        if miss_a == 0 and args.skip_existing:
            print("\nTum atamalar mevcut, assign atlaniyor.", flush=True)
        else:
            rc = phase_assign(args.jobs, args.skip_existing)
            if rc != 0:
                sys.exit(rc)

    if args.phase in ("eval", "all"):
        miss_a2, _, miss_e2, _ = status_report(k_list)
        if miss_e2 > 0 and miss_a2 > 0:
            print(f"\nUyari: hala {miss_a2} atama eksik; eval sadece hazir olanlar icin.", flush=True)
        rc = phase_eval(k_list=k_list)
        sys.exit(rc)


if __name__ == "__main__":
    main()
