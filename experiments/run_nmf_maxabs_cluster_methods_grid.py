"""
NMF (10/20/30/40) + maxabs + trainonly rand f1 — kümeleme yöntemleri (B0 yok).

Kümeleme (compare_clustering_methods.py + sklearn KMeans):
  KMEANS, PCA-KMeans, SOM-Cluster, PCA-SOM

K tahmin yöntemleri (cosine, min_common=3):
  cluster_avg, cluster_avg_hard, cluster_knn_native,
  cluster_knn_surprise_baseline, cluster_knn_with_means

kNN k_neighbors = en küçük küme boyutu (her atama için ayrı).

  python experiments/run_nmf_maxabs_cluster_methods_grid.py --phase cluster
  python experiments/run_nmf_maxabs_cluster_methods_grid.py --phase eval
  python experiments/run_nmf_maxabs_cluster_methods_grid.py --phase all --jobs 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mealpy"))

from baseline_clustering import PCAKMeans, PCASOM, SOMCluster  # noqa: E402
from generate_assignments import (  # noqa: E402
    load_movielens_train_only_100k,
    prepare_matrix_for_clustering,
)
from wnmf.wnmf_experiment import (  # noqa: E402
    RANDOM_SEED,
    _align_assignment_bundle,
    _cluster_avg_predict_kwargs,
    _knn_centroid_bundle,
    _nearest_centroid_bundle,
    load_assignment,
    load_ratings_100k_all,
    load_user_features,
    run_cluster_average,
    run_cluster_knn,
)

ASSIGN_ROOT = REPO / "results" / "cluster_methods" / "ml100k"
OUT_CSV = REPO / "results" / "nmf_maxabs_cluster_methods_preds.csv"

NMF_DIMS = [10, 20, 30, 40]
K_LIST = [6, 10, 14, 21, 30]
CLUSTER_METHODS = ("KMEANS", "PCA_KMEANS", "SOM", "PCA_SOM")
PREDICTORS = (
    "cluster_avg",
    "cluster_avg_hard",
    "cluster_knn_native",
    "cluster_knn_surprise_baseline",
    "cluster_knn_with_means",
)
SIM = "cosine"
MIN_COMMON = 3
_EVAL_ARGS = Namespace(similarity=SIM, min_common=MIN_COMMON)
SEED = 42
_FEATURE_CACHE: Dict[int, np.ndarray] = {}


def assign_dir(method: str, nmf_dim: int, k: int) -> Path:
    return ASSIGN_ROOT / f"{method}_trainonly_rand_f1_maxabs_nmf{nmf_dim}_k{k}"


def min_cluster_size(labels: np.ndarray) -> int:
    c = Counter(labels.astype(int).tolist())
    return int(min(c.values())) if c else 1


def cluster_stats(labels: np.ndarray) -> dict:
    c = Counter(labels.astype(int).tolist())
    sizes = sorted(c.values())
    return {
        "active": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "mean": float(np.mean(sizes)),
        "std": float(np.std(sizes)),
        "sizes": sizes,
        "singletons": sum(1 for s in sizes if s == 1),
    }


def save_bundle(
    out: Path,
    labels: np.ndarray,
    features: np.ndarray,
    centers: np.ndarray,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int32)
    features = np.asarray(features, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float64)
    np.save(out / "assignments.npy", labels)
    np.save(out / "gray_sheep_mask.npy", np.zeros(len(labels), dtype=bool))
    np.save(out / "user_features.npy", features)
    np.save(out / "best_sol.npy", centers.reshape(-1))


def _som_grid_size(k: int) -> int:
    import math
    g = max(2, int(math.ceil(math.sqrt(k * 1.5))))
    while g * g < k:
        g += 1
    return g


def run_clustering(
    method: str,
    X: np.ndarray,
    k: int,
    nmf_dim: int,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, centers) with centers shape (k, n_features)."""
    if method == "KMEANS":
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        return labels, centers

    if method == "PCA_KMEANS":
        pca_k = min(nmf_dim, X.shape[0] - 1, X.shape[1])
        pca_k = max(2, pca_k)
        model = PCAKMeans(n_clusters=k, pca_components=pca_k, random_state=SEED)
        model.fit(X, verbose=verbose)
        return model.get_labels(), model.get_centers()

    if method == "SOM":
        grid = _som_grid_size(k)
        model = SOMCluster(n_clusters=k, grid_size=grid, n_epochs=50, random_state=SEED)
        model.fit(X, verbose=verbose)
        return model.get_labels(), model.get_centers()

    if method == "PCA_SOM":
        pca_k = min(nmf_dim, X.shape[0] - 1, X.shape[1])
        pca_k = max(2, pca_k)
        grid = _som_grid_size(k)
        model = PCASOM(
            n_clusters=k, pca_components=pca_k, grid_size=grid,
            n_epochs=50, random_state=SEED,
        )
        model.fit(X, verbose=verbose)
        return model.get_labels(), model.get_centers()

    raise ValueError(f"Bilinmeyen method: {method}")


def nmf_features(nmf_dim: int) -> np.ndarray:
    if nmf_dim not in _FEATURE_CACHE:
        matrix = load_movielens_train_only_100k(
            eval_split="random", fold=1, random_seed=SEED,
        )
        X = prepare_matrix_for_clustering(
            matrix,
            zscore=False,
            pca_var=None,
            wnmf_k=None,
            preprocess="maxabs",
            feature_extraction="nmf",
            svd_components=nmf_dim,
            min_user_ratings=0,
            min_item_ratings=0,
        )
        _FEATURE_CACHE[nmf_dim] = np.asarray(X, dtype=np.float32)
        print(f"  NMF{nmf_dim} features: {_FEATURE_CACHE[nmf_dim].shape}")
    return _FEATURE_CACHE[nmf_dim]


def phase_cluster(
    nmf_dims: Sequence[int],
    ks: Sequence[int],
    methods: Sequence[str],
    skip_existing: bool,
) -> pd.DataFrame:
    rows: List[dict] = []
    for nmf_dim in nmf_dims:
        X = nmf_features(nmf_dim)
        for k in ks:
            for method in methods:
                out = assign_dir(method, nmf_dim, k)
                if skip_existing and (out / "assignments.npy").is_file():
                    labels = np.load(out / "assignments.npy")
                    st = cluster_stats(labels)
                    rows.append({
                        "method": method, "nmf_dim": nmf_dim, "k": k,
                        "status": "skip", **st,
                    })
                    continue
                t0 = time.time()
                try:
                    labels, centers = run_clustering(method, X, k, nmf_dim)
                    save_bundle(out, labels, X, centers)
                    st = cluster_stats(labels)
                    rows.append({
                        "method": method, "nmf_dim": nmf_dim, "k": k,
                        "status": "ok", "time_s": round(time.time() - t0, 1), **st,
                    })
                    print(
                        f"  {method} nmf{nmf_dim} k={k}: "
                        f"min={st['min']} max={st['max']} std={st['std']:.1f} "
                        f"sing={st['singletons']} ({time.time()-t0:.1f}s)"
                    )
                except Exception as exc:
                    rows.append({
                        "method": method, "nmf_dim": nmf_dim, "k": k,
                        "status": f"err:{exc}", "active": 0,
                    })
                    print(f"  FAIL {method} nmf{nmf_dim} k={k}: {exc}")
    df = pd.DataFrame(rows)
    cluster_csv = REPO / "results" / "nmf_maxabs_cluster_methods_sizes.csv"
    cluster_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cluster_csv, index=False)
    print(f"\nKüme boyutları -> {cluster_csv}")
    return df


def _load_done_keys(csv_path: Path) -> set:
    if not csv_path.is_file():
        return set()
    df = pd.read_csv(csv_path)
    return {
        (str(r["method"]), int(r["nmf_dim"]), int(r["k"]), str(r["predictor"]))
        for _, r in df.iterrows()
    }


def _append_rows(rows: List[dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.is_file()
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=header, index=False)


def _eval_one(
    method: str,
    nmf_dim: int,
    k: int,
    train: np.ndarray,
    test: np.ndarray,
    n_items: int,
    n_users: int,
) -> List[dict]:
    adir = str(assign_dir(method, nmf_dim, k))
    if not os.path.isfile(os.path.join(adir, "assignments.npy")):
        return []

    assignments, gray_mask = load_assignment(adir)
    uf = load_user_features(adir, len(assignments))
    assignments, gray_mask, _, uf = _align_assignment_bundle(
        assignments, gray_mask, None, uf,
        n_users_expected=n_users, algo_label=method, assign_dir=adir,
    )
    knn_k = max(1, min_cluster_size(assignments))
    nc_avg = _nearest_centroid_bundle(None, adir, assignments)
    nc_knn = _knn_centroid_bundle(None, adir, assignments, knn_mode="cluster")
    common = dict(top_n=10, relevance_threshold=4.0, assign_dir=adir)
    rows: List[dict] = []

    def _row(predictor: str, r: dict, sec: float) -> dict:
        return {
            "method": method,
            "nmf_dim": nmf_dim,
            "k": k,
            "predictor": predictor,
            "knn_k": knn_k,
            "min_cluster": min_cluster_size(assignments),
            "scenario": r.get("scenario", predictor),
            "mae": r["mae"],
            "rmse": r["rmse"],
            "ndcg_at_10": r["ndcg_at_10"],
            "precision_at_10": r.get("precision_at_10", np.nan),
            "recall_at_10": r.get("recall_at_10", np.nan),
            "time_s": round(sec, 1),
        }

    t0 = time.time()
    r = run_cluster_average(
        train, test, assignments, gray_mask, None, n_items, method,
        **_cluster_avg_predict_kwargs(_EVAL_ARGS),
        **nc_avg, **common,
    )
    rows.append(_row("cluster_avg", r, time.time() - t0))

    t0 = time.time()
    r = run_cluster_average(
        train, test, assignments, gray_mask, None, n_items, method,
        cluster_avg_hard=True, **nc_avg, **common,
    )
    rows.append(_row("cluster_avg_hard", r, time.time() - t0))

    t0 = time.time()
    r = run_cluster_knn(
        train, test, assignments, gray_mask, None, n_items, method,
        user_features=uf, similarity=SIM, min_common=MIN_COMMON,
        k_neighbors=knn_k, cluster_knn_backend="native",
        **nc_knn, **common,
    )
    rows.append(_row("cluster_knn_native", r, time.time() - t0))

    for pred, variant in (
        ("cluster_knn_surprise_baseline", "baseline"),
        ("cluster_knn_with_means", "withmeans"),
    ):
        t0 = time.time()
        r = run_cluster_knn(
            train, test, assignments, gray_mask, None, n_items, method,
            user_features=uf, similarity=SIM, min_common=MIN_COMMON,
            k_neighbors=knn_k, cluster_knn_backend="surprise",
            surprise_knn_variant=variant,
            **nc_knn, **common,
        )
        rows.append(_row(pred, r, time.time() - t0))

    return rows


def phase_eval(
    nmf_dims: Sequence[int],
    ks: Sequence[int],
    methods: Sequence[str],
    skip_existing: bool,
    csv_path: Path,
) -> None:
    data_path = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data_path, random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    done = _load_done_keys(csv_path) if skip_existing else set()
    batch: List[dict] = []
    total = len(nmf_dims) * len(ks) * len(methods)
    n_done = 0

    for nmf_dim in nmf_dims:
        for k in ks:
            for method in methods:
                n_done += 1
                pending = [
                    p for p in PREDICTORS
                    if (method, nmf_dim, k, p) not in done
                ]
                if skip_existing and not pending:
                    continue
                print(f"[{n_done}/{total}] eval {method} nmf{nmf_dim} k={k}")
                try:
                    rows = _eval_one(method, nmf_dim, k, train, test, n_items, n_users)
                    if skip_existing:
                        rows = [r for r in rows if (method, nmf_dim, k, r["predictor"]) not in done]
                    batch.extend(rows)
                    if len(batch) >= 20:
                        _append_rows(batch, csv_path)
                        batch = []
                except Exception as exc:
                    print(f"  FAIL eval {method} nmf{nmf_dim} k={k}: {exc}")

    _append_rows(batch, csv_path)
    print(f"\nTahmin sonuçları -> {csv_path}")
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        best = df.sort_values("mae").head(15)
        print("\nEn düşük 15 MAE:")
        print(best[["method", "nmf_dim", "k", "predictor", "knn_k", "mae", "ndcg_at_10"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cluster", "eval", "all"], default="all")
    ap.add_argument("--nmf-dim", type=int, nargs="+", default=NMF_DIMS)
    ap.add_argument("--k", type=int, nargs="+", default=K_LIST)
    ap.add_argument(
        "--methods", nargs="+", default=list(CLUSTER_METHODS),
        choices=list(CLUSTER_METHODS),
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args()

    if args.phase in ("cluster", "all"):
        print("=" * 72)
        print("CLUSTER phase")
        print("=" * 72)
        phase_cluster(args.nmf_dim, args.k, args.methods, args.skip_existing)

    if args.phase in ("eval", "all"):
        print("\n" + "=" * 72)
        print("EVAL phase")
        print("=" * 72)
        phase_eval(
            args.nmf_dim, args.k, args.methods,
            args.skip_existing, Path(args.out_csv),
        )


if __name__ == "__main__":
    main()
