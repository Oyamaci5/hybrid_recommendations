"""
3 baseline kümeleyici (baseline_clustering.py) + B0 / AGTO karşılaştırması.

Aynı WNMF U, train/test (fold=1) ve meta_dual cluster_avg / MetaDual metrikleri.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from baseline_clustering import PCAKMeans, PCASOM, SOMCluster
from wnmf.meta_dual_cf import (
    _assign_dir,
    _baseline_cluster_avg_user,
    _load_assignments,
    _load_centroids,
    predict_meta_dual,
)
from wnmf.wnmf_utils import load_ratings_100k_all


def _load_user_features(assign_dir: str) -> np.ndarray:
    for name in ("user_features.npy", "wnmf_user_vectors.npy"):
        path = os.path.join(assign_dir, name)
        if os.path.isfile(path):
            return np.load(path)
    raise FileNotFoundError(f"U matrisi yok: {assign_dir}")


def _cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    return {
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "silhouette": float(silhouette_score(X, labels)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline 3 + AGTO raporu")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--wnmf-dim", type=int, default=20)
    p.add_argument("--pca-components", type=int, default=20)
    p.add_argument("--som-epochs", type=int, default=50)
    p.add_argument("--csv", default="results/baseline_vs_agto_k10.csv")
    args = p.parse_args()

    k = args.k
    data_path = os.path.join(REPO, "data", "ml-100k", "u.data")
    train, test = load_ratings_100k_all(data_path, random_seed=42, fold=args.fold)

    ref_dir = _assign_dir("AGTO", k, wnmf_dim=args.wnmf_dim)
    U = _load_user_features(ref_dir)
    print(f"U shape: {U.shape}  (kaynak: {ref_dir})\n")

    assign_b0 = _load_assignments(_assign_dir("B0_KMEANS", k, wnmf_dim=args.wnmf_dim))
    agto_dir = _assign_dir("AGTO", k, wnmf_dim=args.wnmf_dim)
    assign_agto = _load_assignments(agto_dir)
    centroids_agto = _load_centroids(agto_dir, k, U.shape[1])

    rows: list[dict] = []

    def add_row(method: str, assignments: np.ndarray, fit_sec: float, cluster_only: bool = False):
        m = _baseline_cluster_avg_user(train, test, assignments)
        row = {
            "K": k,
            "method": method,
            "fit_sec": round(fit_sec, 2),
            "mae": m["mae"],
            "rmse": m["rmse"],
            "precision_at_10": m["precision_at_10"],
            "recall_at_10": m["recall_at_10"],
            "ndcg_at_10": m["ndcg_at_10"],
        }
        if not cluster_only:
            row.update(_cluster_metrics(U, assignments))
        rows.append(row)
        return m

    baselines = [
        ("PCA-KMeans", lambda: PCAKMeans(
            n_clusters=k, pca_components=args.pca_components, random_state=42,
        )),
        ("SOM-Cluster", lambda: SOMCluster(
            n_clusters=k,
            grid_size=max(4, int(np.ceil(np.sqrt(k * 2)))),
            n_epochs=args.som_epochs,
            random_state=42,
        )),
        ("PCA-SOM", lambda: PCASOM(
            n_clusters=k,
            pca_components=args.pca_components,
            grid_size=max(4, int(np.ceil(np.sqrt(k * 2)))),
            n_epochs=args.som_epochs,
            random_state=42,
        )),
    ]

    print("=" * 72)
    print(f"BASELINE KÜMELEME (K={k}) — downstream cluster_avg")
    print("=" * 72)
    for name, factory in baselines:
        t0 = time.time()
        model = factory()
        model.fit(U, verbose=False)
        fit_sec = time.time() - t0
        labels = model.get_labels()
        m = add_row(name, labels, fit_sec)
        cm = rows[-1]
        print(
            f"{name:<14}  MAE={m['mae']:.4f}  NDCG={m['ndcg_at_10']:.4f}  "
            f"DB={cm['davies_bouldin']:.4f}  sil={cm['silhouette']:.4f}  "
            f"fit={fit_sec:.1f}s"
        )

    print("\n" + "=" * 72)
    print("REFERANS (meta atama)")
    print("=" * 72)
    for label, assign in [("B0_KMEANS", assign_b0), ("AGTO cluster_avg", assign_agto)]:
        m = add_row(label, assign, 0.0, cluster_only=True)
        print(f"{label:<18}  MAE={m['mae']:.4f}  NDCG={m['ndcg_at_10']:.4f}")

    t0 = time.time()
    dual = predict_meta_dual(
        train, test, assign_b0, assign_agto, U, centroids_agto, beta=0.85,
    )
    dual_sec = time.time() - t0
    rows.append({
        "K": k,
        "method": "AGTO MetaDual",
        "fit_sec": round(dual_sec, 2),
        "mae": dual["mae"],
        "rmse": dual["rmse"],
        "precision_at_10": dual["precision_at_10"],
        "recall_at_10": dual["recall_at_10"],
        "ndcg_at_10": dual["ndcg_at_10"],
        "davies_bouldin": None,
        "silhouette": None,
    })
    print(
        f"{'AGTO MetaDual':<18}  MAE={dual['mae']:.4f}  NDCG={dual['ndcg_at_10']:.4f}  "
        f"(eval {dual_sec:.0f}s)"
    )

    if args.csv:
        import pandas as pd

        out = args.csv if os.path.isabs(args.csv) else os.path.join(REPO, args.csv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
