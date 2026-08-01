"""
FCM official fold-1 eval: cluster_avg + soft membership.

  python experiments/run_fuzzy_official_f1_eval.py --k 4 10
  python experiments/run_fuzzy_official_f1_eval.py --k 10 --prune --similarity cosine
"""

from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import (
    FAST_ALGOS,
    FAST_K_LIST,
    FCM_M,
    FOLD,
    WNMF_DIM,
    assign_dir,
    expected_suffix,
)
from experiments.run_fuzzy_official_f1_assign import ALGOS

DEFAULT_SOFT = 0.10
DEFAULT_SIMS = ("pearson", "pearson_iuf", "cosine")
MIN_COMMON = 3
KNN_FIXED = 50
PREDICTORS_ALL = ("cluster_avg_soft", "cluster_knn")


def load_official_fold(fold: int):
    from wnmf.wnmf_utils import load_ratings_100k

    base = str(REPO / "data" / "ml-100k" / "u1.base")
    test = str(REPO / "data" / "ml-100k" / "u1.test")
    return load_ratings_100k(base, test, fold=fold)


def predict_cluster_avg_eval_rows(
    algo: str,
    k: int,
    *,
    similarity: str,
    soft_threshold: float = DEFAULT_SOFT,
    prune: bool = False,
    fast: bool = False,
    fold: int = FOLD,
    quiet: bool = True,
) -> np.ndarray:
    """(n,4) eval_rows: user, item, true, pred — cluster_avg_soft."""
    import contextlib
    import io

    from wnmf.wnmf_experiment import (
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        run_cluster_average,
    )

    train, test = load_official_fold(fold)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    adir = assign_dir(algo, k, prune=prune, fast=fast)
    if not (adir / "assignments.npy").is_file():
        raise FileNotFoundError(f"Atama yok: {adir}")
    if not (adir / "memberships.npy").is_file():
        raise FileNotFoundError(f"memberships.npy yok: {adir}")

    assignments, gray_mask = load_assignment(str(adir))
    memberships = load_memberships(str(adir))
    assignments, gray_mask, memberships, _ = _align_assignment_bundle(
        assignments, gray_mask, memberships, None,
        n_users_expected=n_users, algo_label=algo, assign_dir=str(adir),
    )
    nc = _nearest_centroid_bundle(None, str(adir), assignments)
    eval_args = Namespace(
        similarity=similarity,
        min_common=MIN_COMMON,
        soft_membership_threshold=float(soft_threshold),
    )
    kwargs = dict(
        train=train,
        test=test,
        assignments=assignments,
        gray_mask=gray_mask,
        memberships=memberships,
        n_items=n_items,
        algo_label=algo,
        **_cluster_avg_predict_kwargs(eval_args),
        **nc,
        top_n=10,
        relevance_threshold=4.0,
        assign_dir=str(adir),
        return_eval_rows=True,
    )
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            row = run_cluster_average(**kwargs)
    else:
        row = run_cluster_average(**kwargs)
    return np.asarray(row["eval_rows"], dtype=np.float64)


def _base_result_row(
    *,
    k: int,
    algo: str,
    predictor: str,
    similarity: str,
    soft_threshold: float,
    knn_k: int,
    r: dict,
    sizes: list[int],
    prune: bool,
    fast: bool,
    eval_seconds: float,
    fold: int = FOLD,
) -> dict:
    return {
        "protocol": "fuzzy_imkpp_official",
        "fold": int(fold),
        "k": k,
        "fcm_m": FCM_M,
        "wnmf_dim": WNMF_DIM,
        "prune": prune,
        "fast": fast,
        "algo": algo,
        "predictor": predictor,
        "knn_k": int(knn_k),
        "similarity": similarity,
        "soft_threshold": soft_threshold if predictor == "cluster_avg_soft" else float("nan"),
        "mae": round(r["mae"], 4),
        "rmse": round(r["rmse"], 4),
        "ndcg_at_10": round(r["ndcg_at_10"], 4),
        "precision_at_10": round(r["precision_at_10"], 4),
        "recall_at_10": round(r["recall_at_10"], 4),
        "coverage_at_10": round(r.get("coverage_at_10", float("nan")), 4),
        "jaccard_at_10": round(r.get("jaccard_at_10", float("nan")), 4),
        "n_active_clusters": len(sizes),
        "cluster_min": min(sizes) if sizes else 0,
        "cluster_max": max(sizes) if sizes else 0,
        "cluster_sizes_desc": str(sizes),
        "assign_suffix": expected_suffix(k, fold=fold, prune=prune, fast=fast),
        "eval_seconds": round(eval_seconds, 1),
    }


def eval_k(
    k: int,
    *,
    soft_threshold: float,
    similarity: str,
    algos: list[str],
    prune: bool = False,
    fast: bool = False,
    fold: int = FOLD,
    predictors: tuple[str, ...] = PREDICTORS_ALL,
) -> pd.DataFrame:
    from wnmf.wnmf_experiment import (
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _knn_centroid_bundle,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        load_user_features,
        run_cluster_average,
        run_cluster_knn,
    )

    train, test = load_official_fold(fold)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    eval_args = Namespace(
        similarity=similarity,
        min_common=MIN_COMMON,
        soft_membership_threshold=float(soft_threshold),
    )
    common = dict(top_n=10, relevance_threshold=4.0)

    rows = []
    for algo in algos:
        adir = assign_dir(algo, k, fold=fold, prune=prune, fast=fast)
        if not (adir / "assignments.npy").is_file():
            print(f"SKIP {algo} K={k} fast={fast} prune={prune} (atama yok): {adir.name}", flush=True)
            continue
        if not (adir / "memberships.npy").is_file():
            print(f"SKIP {algo} K={k} (memberships.npy yok)", flush=True)
            continue

        assignments, gray_mask = load_assignment(str(adir))
        memberships = load_memberships(str(adir))
        user_features = load_user_features(str(adir), len(assignments))
        assignments, gray_mask, memberships, user_features = _align_assignment_bundle(
            assignments, gray_mask, memberships, user_features,
            n_users_expected=n_users, algo_label=algo, assign_dir=str(adir),
        )
        sizes = sorted(Counter(assignments.astype(int).tolist()).values(), reverse=True)
        k_min = max(1, min(sizes)) if sizes else 1
        adir_s = str(adir)

        if "cluster_avg_soft" in predictors:
            nc_avg = _nearest_centroid_bundle(None, adir_s, assignments)
            t0 = time.time()
            r = run_cluster_average(
                train, test, assignments, gray_mask, memberships, n_items, algo,
                **_cluster_avg_predict_kwargs(eval_args),
                **nc_avg,
                assign_dir=adir_s,
                **common,
            )
            row = _base_result_row(
                k=k, algo=algo, predictor="cluster_avg_soft",
                similarity=similarity, soft_threshold=soft_threshold, knn_k=0,
                r=r, sizes=sizes, prune=prune, fast=fast, fold=fold,
                eval_seconds=time.time() - t0,
            )
            rows.append(row)
            print(
                f"  {algo} K={k} cluster_avg_soft sim={similarity}: "
                f"MAE={row['mae']:.4f} NDCG={row['ndcg_at_10']:.4f}",
                flush=True,
            )

        if "cluster_knn" in predictors:
            nc_knn = _knn_centroid_bundle(None, adir_s, assignments, knn_mode="cluster")
            for knn_k in dict.fromkeys([k_min, KNN_FIXED]):
                t0 = time.time()
                r = run_cluster_knn(
                    train, test, assignments, gray_mask, memberships, n_items, algo,
                    user_features=user_features,
                    similarity=similarity,
                    min_common=MIN_COMMON,
                    k_neighbors=int(knn_k),
                    cluster_knn_backend="native",
                    **nc_knn,
                    assign_dir=adir_s,
                    **common,
                )
                row = _base_result_row(
                    k=k, algo=algo, predictor="cluster_knn",
                    similarity=similarity, soft_threshold=soft_threshold, knn_k=knn_k,
                    r=r, sizes=sizes, prune=prune, fast=fast, fold=fold,
                    eval_seconds=time.time() - t0,
                )
                rows.append(row)
                print(
                    f"  {algo} K={k} cluster_knn k={knn_k} sim={similarity}: "
                    f"MAE={row['mae']:.4f} NDCG={row['ndcg_at_10']:.4f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[4])
    ap.add_argument("--soft-threshold", type=float, default=DEFAULT_SOFT)
    ap.add_argument("--prune", action="store_true", help="pruneu5_i10 atama seti")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="assignments_estop + _pwcss (hizli K taramasi)",
    )
    ap.add_argument(
        "--similarity",
        nargs="+",
        default=list(DEFAULT_SIMS),
        choices=["pearson", "pearson_iuf", "cosine"],
        help="kNN benzerlik (pearson_iuf = IUF agirlikli pearson)",
    )
    ap.add_argument("--algo", nargs="+", default=None)
    ap.add_argument(
        "--predictor",
        nargs="+",
        default=["all"],
        help="cluster_avg_soft | cluster_knn (k_min+k=50) | all",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "results" / "fuzzy_official_f1_cluster_avg_soft.csv",
    )
    args = ap.parse_args()
    if args.fast and args.algo is None:
        algos = FAST_ALGOS
    else:
        algos = args.algo if args.algo else ALGOS
    if args.fast and args.k == [4]:
        k_list = FAST_K_LIST
    else:
        k_list = args.k

    sims = list(dict.fromkeys(args.similarity))
    if "all" in args.predictor:
        predictors = PREDICTORS_ALL
    elif "cluster_knn" in args.predictor:
        predictors = ("cluster_knn",)
    else:
        predictors = tuple(args.predictor)

    print(
        f"FCM official eval  fold={FOLD}  u{FOLD}.base/u{FOLD}.test  "
        f"predictors={predictors}  soft={args.soft_threshold}  "
        f"fast={args.fast}  prune={args.prune}  similarity={sims}",
        flush=True,
    )

    all_df = []
    for k in k_list:
        for sim in sims:
            print(f"\n--- K={k}  similarity={sim} ---", flush=True)
            df = eval_k(
                k,
                soft_threshold=args.soft_threshold,
                similarity=sim,
                algos=algos,
                prune=args.prune,
                fast=args.fast,
                predictors=predictors,
            )
            if not df.empty:
                all_df.append(df)

    if not all_df:
        sys.exit("Hic sonuc uretilmedi.")

    out = pd.concat(all_df, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.out_csv.is_file():
        old = pd.read_csv(args.out_csv)
        if "prune" not in old.columns:
            old["prune"] = False
        if "fast" not in old.columns:
            old["fast"] = False
        if "knn_k" not in old.columns:
            old["knn_k"] = 0
        key = [
            "fold", "k", "algo", "predictor", "knn_k", "similarity",
            "soft_threshold", "fcm_m", "wnmf_dim", "prune", "fast",
        ]
        out = pd.concat([old, out], ignore_index=True).drop_duplicates(subset=key, keep="last")
    out = out.sort_values(["fast", "prune", "k", "similarity", "algo"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nCSV -> {args.out_csv}", flush=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
