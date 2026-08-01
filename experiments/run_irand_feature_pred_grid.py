"""
SVD/WNMF latent (5,10) × K (3,7,14,27,70) × irand/nogs/euc — küme tahmin yöntemleri (cosine).

Atama: trainonly_rand_f1, kmref yok, --init-mode random
Tahmin (global WNMF / sharedV yok):
  cluster_avg, cluster_avg_hard, cluster_knn_native,
  cluster_knn_surprise_baseline, cluster_knn_with_means, meta_dual

  python experiments/run_irand_feature_pred_grid.py --phase assign
  python experiments/run_irand_feature_pred_grid.py --phase eval
  python experiments/run_irand_feature_pred_grid.py --phase all --jobs 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from argparse import Namespace  # noqa: E402

from wnmf.meta_dual_cf import _load_centroids, predict_meta_dual  # noqa: E402
from wnmf.wnmf_experiment import (  # noqa: E402
    RANDOM_SEED,
    _align_assignment_bundle,
    _cluster_avg_predict_kwargs,
    _knn_centroid_bundle,
    _nearest_centroid_bundle,
    load_assignment,
    load_memberships,
    load_ratings_100k_all,
    load_user_features,
    run_cluster_average,
    run_cluster_knn,
)

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"
DEFAULT_ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
DEFAULT_FEATURES: Tuple[Tuple[str, int], ...] = (
    ("svd", 5),
    ("svd", 10),
    ("wnmf", 5),
    ("wnmf", 10),
)
DEFAULT_K = [3, 7, 14, 27, 70]
KNN_K = 20
SIM = "cosine"
MIN_COMMON = 3
_EVAL_ARGS = Namespace(similarity=SIM, min_common=MIN_COMMON)
OUT_CSV = REPO / "results" / "irand_feature_pred_cluster.csv"
OUT_CSV_KMREF = REPO / "results" / "irand_feature_pred_cluster_kmref.csv"


def out_suffix(feat: str, dim: int, k: int, *, kmref: bool = False) -> str:
    s = (
        f"_euc_irand_nogs_trainonly_rand_f1"
        f"_none_{feat}{int(dim)}_k{int(k)}"
    )
    if kmref:
        s += "_kmref"
    return s


def assign_dir(algo: str, feat: str, dim: int, k: int, *, kmref: bool = False) -> Path:
    return ASSIGN_ROOT / f"{algo}{out_suffix(feat, dim, k, kmref=kmref)}"


def _row_key(feat: str, dim: int, k: int, algo: str, predictor: str) -> tuple:
    return (feat, dim, k, algo, predictor)


def _load_done_keys(csv_path: Path) -> set:
    if not csv_path.is_file():
        return set()
    df = pd.read_csv(csv_path)
    keys = set()
    for _, r in df.iterrows():
        keys.add((
            str(r["feature"]),
            int(r["latent_dim"]),
            int(r["k"]),
            str(r["algo"]),
            str(r["predictor"]),
        ))
    return keys


def _append_rows(rows: List[dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not csv_path.is_file()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def _metrics_row(
    feat: str,
    dim: int,
    k: int,
    algo: str,
    predictor: str,
    r: dict,
    sec: float,
) -> dict:
    return {
        "feature": feat,
        "latent_dim": dim,
        "k": k,
        "algo": algo,
        "predictor": predictor,
        "scenario": r.get("scenario", predictor),
        "mae": r["mae"],
        "rmse": r["rmse"],
        "ndcg_at_10": r["ndcg_at_10"],
        "precision_at_10": r.get("precision_at_10", np.nan),
        "recall_at_10": r.get("recall_at_10", np.nan),
        "time_s": round(sec, 1),
    }


META_ALGOS_KMREF = ["B1_HHO", "HA_AVOAHGS", "IWO_HHO"]


def phase_assign(
    features: Sequence[Tuple[str, int]],
    ks: Sequence[int],
    algos: Sequence[str],
    jobs: int,
    skip_existing: bool,
    dry_run: bool,
    python: str,
    *,
    kmref: bool = False,
) -> int:
    k_str = [str(k) for k in ks]
    assign_algos = list(META_ALGOS_KMREF) if kmref else list(algos)
    if kmref:
        print("kmref: yalnız meta algoritmalar (B0_KMEANS kmref almaz)")
    for feat, dim in features:
        cmd = [
            python,
            str(GEN),
            "--dataset",
            "100k",
            "--algo",
            *assign_algos,
            "--no-prune",
            "--no-gray-sheep",
            "--preprocess",
            "none",
            "--feature-extraction",
            feat,
            "--svd-components",
            str(int(dim)),
            "--init-mode",
            "random",
            "--cluster-metric",
            "euclidean",
            "--train-only",
            "--eval-split",
            "random",
            "--fold",
            "1",
            "--k",
            *k_str,
            "--jobs",
            str(int(jobs)),
        ]
        if kmref:
            cmd.append("--kmeans-refine-overwrite")
        if skip_existing:
            cmd.append("--skip-existing")
        print("\n" + "=" * 72)
        print(" ".join(cmd))
        print("=" * 72)
        if dry_run:
            continue
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            return rc
    return 0


def _eval_predictors(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    gray_mask: np.ndarray,
    memberships: np.ndarray,
    n_items: int,
    algo: str,
    adir: str,
    uf: np.ndarray,
    *,
    feat: str,
    dim: int,
    k: int,
    assign_b0: Optional[np.ndarray] = None,
    b0_dir: Optional[str] = None,
) -> List[Tuple[str, dict, float]]:
    """(predictor_name, result_dict, seconds)"""
    out: List[Tuple[str, dict, float]] = []
    nc_avg = _nearest_centroid_bundle(None, adir, assignments)
    nc_knn = _knn_centroid_bundle(None, adir, assignments, knn_mode="cluster")
    common = dict(top_n=10, relevance_threshold=4.0, assign_dir=adir)

    t0 = time.time()
    r = run_cluster_average(
        train, test, assignments, gray_mask, memberships, n_items, algo,
        **_cluster_avg_predict_kwargs(_EVAL_ARGS),
        **nc_avg,
        **common,
    )
    out.append(("cluster_avg", r, time.time() - t0))

    t0 = time.time()
    r = run_cluster_average(
        train, test, assignments, gray_mask, memberships, n_items, algo,
        cluster_avg_hard=True,
        **nc_avg,
        **common,
    )
    out.append(("cluster_avg_hard", r, time.time() - t0))

    t0 = time.time()
    r = run_cluster_knn(
        train, test, assignments, gray_mask, memberships, n_items, algo,
        user_features=uf,
        similarity=SIM,
        min_common=MIN_COMMON,
        k_neighbors=KNN_K,
        cluster_knn_backend="native",
        **nc_knn,
        **common,
    )
    out.append(("cluster_knn_native", r, time.time() - t0))

    for pred, variant in (
        ("cluster_knn_surprise_baseline", "baseline"),
        ("cluster_knn_with_means", "withmeans"),
    ):
        t0 = time.time()
        r = run_cluster_knn(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            user_features=uf,
            similarity=SIM,
            min_common=MIN_COMMON,
            k_neighbors=KNN_K,
            cluster_knn_backend="surprise",
            surprise_knn_variant=variant,
            **nc_knn,
            **common,
        )
        out.append((pred, r, time.time() - t0))

    if (
        algo != "B0_KMEANS"
        and assign_b0 is not None
        and b0_dir
        and os.path.isfile(os.path.join(adir, "user_features.npy"))
    ):
        centroids = _load_centroids(str(adir), k, uf.shape[1])
        t0 = time.time()
        r = predict_meta_dual(
            train, test, assign_b0, assignments, uf, centroids,
            beta=0.85, tau=None, tune=False,
        )
        r = {
            "scenario": "meta_dual",
            "mae": r["mae"],
            "rmse": r["rmse"],
            "ndcg_at_10": r["ndcg_at_10"],
            "precision_at_10": r.get("precision_at_10", np.nan),
            "recall_at_10": r.get("recall_at_10", np.nan),
        }
        out.append(("meta_dual", r, time.time() - t0))

    return out


def phase_eval(
    features: Sequence[Tuple[str, int]],
    ks: Sequence[int],
    algos: Sequence[str],
    skip_existing: bool,
    csv_path: Path,
    *,
    kmref: bool = False,
) -> int:
    data_path = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data_path, random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1

    eval_algos = list(META_ALGOS_KMREF) if kmref else list(algos)
    if kmref:
        print("kmref eval: B1/HA/IWO (_kmref); meta_dual anchor = B0 (kmref yok)")

    done = _load_done_keys(csv_path) if skip_existing else set()
    total_new = 0

    for feat, dim in features:
        for k in ks:
            b0_path = assign_dir("B0_KMEANS", feat, dim, k, kmref=False)
            if not (b0_path / "assignments.npy").is_file():
                print(f"SKIP B0 yok: {feat}{dim} K={k}")
                continue
            a0, g0 = load_assignment(str(b0_path))
            m0 = load_memberships(str(b0_path))
            uf0 = load_user_features(str(b0_path), len(a0))
            a0, g0, m0, uf0 = _align_assignment_bundle(
                a0, g0, m0, uf0,
                n_users_expected=n_users,
                algo_label="B0_KMEANS",
                assign_dir=str(b0_path),
            )

            for algo in eval_algos:
                adir = assign_dir(algo, feat, dim, k, kmref=kmref)
                if not (adir / "assignments.npy").is_file():
                    tag = "_kmref" if kmref else ""
                    print(f"SKIP {algo} {feat}{dim} K={k}{tag}")
                    continue

                assignments, gray_mask = load_assignment(str(adir))
                memberships = load_memberships(str(adir))
                uf = load_user_features(str(adir), len(assignments))
                assignments, gray_mask, memberships, uf = _align_assignment_bundle(
                    assignments, gray_mask, memberships, uf,
                    n_users_expected=n_users,
                    algo_label=algo,
                    assign_dir=str(adir),
                )

                batch: List[dict] = []
                preds = _eval_predictors(
                    train, test, assignments, gray_mask, memberships, n_items,
                    algo, str(adir), uf,
                    feat=feat, dim=dim, k=k,
                    assign_b0=a0 if algo != "B0_KMEANS" else None,
                    b0_dir=str(b0_path) if algo != "B0_KMEANS" else None,
                )
                for pred_name, r, sec in preds:
                    key = _row_key(feat, dim, k, algo, pred_name)
                    if key in done:
                        continue
                    batch.append(_metrics_row(feat, dim, k, algo, pred_name, r, sec))
                    done.add(key)

                if batch:
                    _append_rows(batch, csv_path)
                    total_new += len(batch)
                    print(
                        f"{feat}{dim} K={k:<2} {algo:<12}  "
                        + "  ".join(
                            f"{b['predictor'][:12]}={b['mae']:.3f}"
                            for b in batch
                        ),
                    )

    print(f"\nYeni satir: {total_new}  ->  {csv_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "eval", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", type=str, default=str(OUT_CSV))
    ap.add_argument("--algo", nargs="+", default=list(DEFAULT_ALGOS))
    ap.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    ap.add_argument(
        "--kmref", action="store_true",
        help="assign: kmref üret; eval: _kmref atamaları + B1/HA/IWO tahmin",
    )
    args = ap.parse_args()

    features = list(DEFAULT_FEATURES)
    ks = list(args.k)
    algos = list(args.algo)
    csv_path = Path(args.csv)
    if args.kmref and str(csv_path) == str(OUT_CSV):
        csv_path = OUT_CSV_KMREF
    python = sys.executable

    if args.phase in ("assign", "all"):
        rc = phase_assign(
            features, ks, algos, args.jobs, args.skip_existing, args.dry_run, python,
            kmref=bool(args.kmref),
        )
        if rc != 0:
            sys.exit(rc)

    if args.phase in ("eval", "all"):
        if args.dry_run:
            print("[dry-run] eval atlandı")
            return
        rc = phase_eval(
            features, ks, algos, args.skip_existing, csv_path,
            kmref=bool(args.kmref),
        )
        sys.exit(rc)


if __name__ == "__main__":
    main()
