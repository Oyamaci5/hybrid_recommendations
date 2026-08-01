"""
HA_AVOAHGS official fold-1: tum K=3..14 atama + kume ozeti + cluster_avg tahmin.

  python experiments/test_ha_official_k_sweep.py
  python experiments/test_ha_official_k_sweep.py --skip-assign
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.run_euc_kmref_k_sweep import (
    EVAL_SPLIT,
    GEN,
    K_LIST,
    MIN_COMMON,
    SIM,
    WNMF_EPOCHS,
    assign_dir,
    cluster_stats,
    load_official_fold,
)

ALGO = "HA_AVOAHGS"
DEFAULT_FOLD = 1
DEFAULT_WNMF_DIM = 20
OUT_CSV = REPO / "results" / "ha_official_f1_k_sweep_preview.csv"


def run_assign_all(ks: list[int], fold: int, wnmf_dim: int, jobs: int) -> int:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", ALGO,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(wnmf_dim),
        "--wnmf-epochs", str(WNMF_EPOCHS),
        "--init-mode", "mkpp",
        "--cluster-metric", "euclidean",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", EVAL_SPLIT, "--fold", str(fold),
        "--k", *[str(k) for k in ks],
        "--kmeans-refine-overwrite",
        "--jobs", str(jobs),
        "--skip-existing",
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def eval_one_k(adir: Path, k: int, fold: int, train, test) -> dict:
    from wnmf.wnmf_experiment import (
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        run_cluster_average,
    )

    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    eval_args = Namespace(similarity=SIM, min_common=MIN_COMMON)

    assignments, gray_mask = load_assignment(str(adir))
    memberships = load_memberships(str(adir))
    assignments, gray_mask, memberships, _ = _align_assignment_bundle(
        assignments, gray_mask, memberships, None,
        n_users_expected=n_users, algo_label=ALGO, assign_dir=str(adir),
    )
    st = cluster_stats(assignments)
    nc_avg = _nearest_centroid_bundle(None, str(adir), assignments)
    common = dict(top_n=10, relevance_threshold=4.0, assign_dir=str(adir))

    t0 = time.time()
    r = run_cluster_average(
        train, test, assignments, gray_mask, memberships, n_items, ALGO,
        **_cluster_avg_predict_kwargs(eval_args), **nc_avg, **common,
    )
    sizes = sorted(Counter(assignments.astype(int).tolist()).values(), reverse=True)
    return {
        "algo": ALGO,
        "fold": fold,
        "k": k,
        "n_active_clusters": st["n_active_clusters"],
        "cluster_min": st["cluster_min"],
        "cluster_max": st["cluster_max"],
        "cluster_std": round(st["cluster_std"], 1),
        "singletons": st["singletons"],
        "gray_sheep": int(gray_mask.sum()),
        "cluster_sizes_desc": str(sizes),
        "mae": round(r["mae"], 4),
        "rmse": round(r["rmse"], 4),
        "precision_at_10": round(r["precision_at_10"], 4),
        "recall_at_10": round(r["recall_at_10"], 4),
        "ndcg_at_10": round(r["ndcg_at_10"], 4),
        "coverage_at_10": round(r.get("coverage_at_10", float("nan")), 4),
        "eval_seconds": round(time.time() - t0, 1),
        "assign_dir": adir.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    ap.add_argument("--wnmf-dim", type=int, default=DEFAULT_WNMF_DIM)
    ap.add_argument("--k", type=int, nargs="+", default=None)
    ap.add_argument("--skip-assign", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    ks = args.k if args.k else K_LIST

    print("=" * 70)
    print(f"HA_AVOAHGS OFFICIAL K-SWEEP  fold={args.fold}  K={ks}")
    print("=" * 70)

    if not args.skip_assign:
        rc = run_assign_all(ks, args.fold, args.wnmf_dim, args.jobs)
        if rc != 0:
            sys.exit(rc)

    train, test = load_official_fold(args.fold)
    rows = []
    for k in ks:
        adir = assign_dir(ALGO, k, wnmf_dim=args.wnmf_dim, assign_fold=args.fold)
        if adir is None:
            print(f"  K={k}: ATAMA YOK", flush=True)
            continue
        row = eval_one_k(adir, k, args.fold, train, test)
        rows.append(row)
        print(
            f"  K={k:2d}  clusters={row['n_active_clusters']}  "
            f"sizes=[{row['cluster_min']}-{row['cluster_max']}]  "
            f"MAE={row['mae']:.4f}  P@10={row['precision_at_10']:.4f}  "
            f"NDCG={row['ndcg_at_10']:.4f}  Cov@10={row['coverage_at_10']:.4f}",
            flush=True,
        )

    if not rows:
        sys.exit("Hic sonuc uretilmedi.")

    df = pd.DataFrame(rows).sort_values("k")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("OZET TABLO (cluster_avg)")
    print("=" * 70)
    show = df[["k", "n_active_clusters", "cluster_min", "cluster_max",
               "mae", "precision_at_10", "recall_at_10", "ndcg_at_10", "coverage_at_10"]]
    print(show.to_string(index=False))
    print(f"\nCSV -> {OUT_CSV}")

    best_mae = df.loc[df["mae"].idxmin()]
    best_ndcg = df.loc[df["ndcg_at_10"].idxmax()]
    best_prec = df.loc[df["precision_at_10"].idxmax()]
    print(f"\nEn iyi MAE  : K={int(best_mae['k'])}  MAE={best_mae['mae']:.4f}")
    print(f"En iyi P@10 : K={int(best_prec['k'])}  P@10={best_prec['precision_at_10']:.4f}")
    print(f"En iyi NDCG : K={int(best_ndcg['k'])}  NDCG={best_ndcg['ndcg_at_10']:.4f}")


if __name__ == "__main__":
    main()
