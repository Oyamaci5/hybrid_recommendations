"""
Meta algoritmalar + baseline: K sweep (WCSS/DB/Sil) ve en iyi K'da tam tablo.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

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

_cmp_path = os.path.join(REPO, "mealpy", "mealpy-algorithms-comparision.py")
_spec = importlib.util.spec_from_file_location("mealpy_cmp", _cmp_path)
_mealpy_cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mealpy_cmp)
compute_wcss_fast = _mealpy_cmp.compute_wcss_fast

ALGOS = [
    "B0_KMEANS",
    "B1_HHO",
    "B2_HGS",
    "H9_QSA+CDO",
    "IWO_HHO",
    "HA_AVOAHGS",
    "LIT_PSO",
    "LIT_GWO",
    "AGTO",
]

ALGO_LABELS = {
    "B0_KMEANS": "B0",
    "B1_HHO": "B1/HHO",
    "B2_HGS": "B2/HGS",
    "H9_QSA+CDO": "H9",
    "IWO_HHO": "IWO",
    "HA_AVOAHGS": "HA",
    "LIT_PSO": "LIT_PSO",
    "LIT_GWO": "LIT_GWO",
    "AGTO": "AGTO",
}

DEFAULT_K_CANDIDATES = [3, 6, 9, 10, 12, 14, 15, 21, 27, 30]


def _exists(algo: str, k: int, wnmf_dim: int) -> bool:
    return os.path.isfile(
        os.path.join(_assign_dir(algo, k, wnmf_dim=wnmf_dim), "assignments.npy"),
    )


def _load_u(adir: str) -> np.ndarray:
    for name in ("user_features.npy", "wnmf_user_vectors.npy"):
        p = os.path.join(adir, name)
        if os.path.isfile(p):
            return np.load(p)
    raise FileNotFoundError(adir)


def _quality(U: np.ndarray, assign: np.ndarray, best_sol: np.ndarray, k: int) -> dict:
    wcss, _ = compute_wcss_fast(U, best_sol, k, metric="euclidean")
    return {
        "wcss": float(wcss),
        "davies_bouldin": float(davies_bouldin_score(U, assign)),
        "silhouette": float(silhouette_score(U, assign)),
    }


def _fit_baselines(U: np.ndarray, k: int, som_epochs: int) -> Dict[str, np.ndarray]:
    grid = max(4, int(np.ceil(np.sqrt(k * 2))))
    out: Dict[str, np.ndarray] = {}
    for name, factory in [
        ("PCA-KMeans", lambda: PCAKMeans(n_clusters=k, pca_components=min(20, U.shape[1]), random_state=42)),
        ("SOM-Cluster", lambda: SOMCluster(n_clusters=k, grid_size=grid, n_epochs=som_epochs, random_state=42)),
        ("PCA-SOM", lambda: PCASOM(
            n_clusters=k, pca_components=min(20, U.shape[1]),
            grid_size=grid, n_epochs=som_epochs, random_state=42,
        )),
    ]:
        m = factory()
        m.fit(U, verbose=False)
        out[name] = m.get_labels()
    return out


def sweep_cluster_metrics(
    k_list: List[int], algos: List[str], wnmf_dim: int, *, baselines: bool, som_epochs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    sep: List[dict] = []

    for k in k_list:
        ref = next((a for a in algos if _exists(a, k, wnmf_dim)), None)
        if ref is None:
            continue
        U = _load_u(_assign_dir(ref, k, wnmf_dim=wnmf_dim))
        meta_rows: List[dict] = []

        for algo in algos:
            if not _exists(algo, k, wnmf_dim):
                continue
            adir = _assign_dir(algo, k, wnmf_dim=wnmf_dim)
            assign = _load_assignments(adir)
            sol = np.load(os.path.join(adir, "best_sol.npy"))
            q = _quality(U, assign, sol, k)
            row = {"K": k, "algo": algo, "label": ALGO_LABELS.get(algo, algo), "type": "meta", **q}
            rows.append(row)
            meta_rows.append(row)

        if baselines:
            for bname, labels in _fit_baselines(U, k, som_epochs).items():
                rows.append({
                    "K": k, "algo": bname, "label": bname, "type": "baseline",
                    "wcss": np.nan,
                    "davies_bouldin": float(davies_bouldin_score(U, labels)),
                    "silhouette": float(silhouette_score(U, labels)),
                })

        if len(meta_rows) >= 2:
            sep.append({
                "K": k,
                "n_algos": len(meta_rows),
                "wcss_std": float(np.std([r["wcss"] for r in meta_rows])),
                "wcss_range": float(np.ptp([r["wcss"] for r in meta_rows])),
                "sil_std": float(np.std([r["silhouette"] for r in meta_rows])),
                "db_std": float(np.std([r["davies_bouldin"] for r in meta_rows])),
            })

    return pd.DataFrame(rows), pd.DataFrame(sep)


def eval_downstream(
    train: np.ndarray,
    test: np.ndarray,
    k: int,
    algos: List[str],
    wnmf_dim: int,
    *,
    meta_dual: bool,
    baselines: bool,
    som_epochs: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    ref = next(a for a in algos if _exists(a, k, wnmf_dim))
    U = _load_u(_assign_dir(ref, k, wnmf_dim=wnmf_dim))
    assign_b0 = _load_assignments(_assign_dir("B0_KMEANS", k, wnmf_dim=wnmf_dim))

    for algo in algos:
        if not _exists(algo, k, wnmf_dim):
            continue
        adir = _assign_dir(algo, k, wnmf_dim=wnmf_dim)
        assign = _load_assignments(adir)
        sol = np.load(os.path.join(adir, "best_sol.npy"))
        q = _quality(U, assign, sol, k)
        pred = _baseline_cluster_avg_user(train, test, assign)
        row = {
            "K": k, "algo": algo, "label": ALGO_LABELS.get(algo, algo), "type": "meta",
            **q,
            "mae": pred["mae"], "rmse": pred["rmse"],
            "ndcg_at_10": pred["ndcg_at_10"],
            "meta_dual_mae": None, "meta_dual_ndcg": None,
        }
        if meta_dual and algo != "B0_KMEANS":
            C = _load_centroids(adir, k, U.shape[1])
            d = predict_meta_dual(train, test, assign_b0, assign, U, C, beta=0.85)
            row["meta_dual_mae"] = d["mae"]
            row["meta_dual_ndcg"] = d["ndcg_at_10"]
        rows.append(row)

    if baselines:
        for bname, labels in _fit_baselines(U, k, som_epochs).items():
            pred = _baseline_cluster_avg_user(train, test, labels)
            rows.append({
                "K": k, "algo": bname, "label": bname, "type": "baseline",
                "wcss": np.nan,
                "davies_bouldin": float(davies_bouldin_score(U, labels)),
                "silhouette": float(silhouette_score(U, labels)),
                "mae": pred["mae"], "rmse": pred["rmse"],
                "ndcg_at_10": pred["ndcg_at_10"],
                "meta_dual_mae": None, "meta_dual_ndcg": None,
            })

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k-list", type=int, nargs="+", default=DEFAULT_K_CANDIDATES)
    p.add_argument("--k-focus", type=int, default=None)
    p.add_argument("--wnmf-dim", type=int, default=20)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--csv", default="results/algo_k_sweep_metrics.csv")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-meta-dual", action="store_true")
    args = p.parse_args()

    train, test = load_ratings_100k_all(
        os.path.join(REPO, "data", "ml-100k", "u.data"), random_seed=42, fold=args.fold,
    )

    print("Faz 1: K sweep — küme metrikleri (hızlı)...")
    clust_df, sep_df = sweep_cluster_metrics(
        args.k_list, ALGOS, args.wnmf_dim,
        baselines=not args.no_baselines,
        som_epochs=50,
    )
    sep_df = sep_df.sort_values("wcss_std", ascending=False)
    print("\nK ayrışması (meta algo WCSS std — yüksek = atamalar daha farklı):")
    print(sep_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    best_k = int(sep_df.iloc[0]["K"]) if len(sep_df) else 10
    print(f"\n>>> Önerilen K (WCSS ayrışması): {best_k}")

  # Downstream ayrışma için NDCG (yalnızca k=10 örnek — hızlı kontrol)
    if 10 in args.k_list and _exists("B1_HHO", 10, args.wnmf_dim):
        sub = eval_downstream(
            train, test, 10, ALGOS, args.wnmf_dim,
            meta_dual=False, baselines=False, som_epochs=50,
        )
        ndcg_std_10 = float(sub["ndcg_at_10"].std())
        print(f"    K=10 downstream NDCG std (cluster_avg): {ndcg_std_10:.4f}")

    focus = args.k_focus or best_k
    print(f"\nFaz 2: K={focus} — cluster_avg + MetaDual...")
    down_df = eval_downstream(
        train, test, focus, ALGOS, args.wnmf_dim,
        meta_dual=not args.no_meta_dual,
        baselines=not args.no_baselines,
        som_epochs=50,
    )

    print(f"\n{'=' * 105}")
    print(f"DETAY — K={focus}")
    print(f"{'Yontem':<12} {'WCSS':>10} {'DB':>8} {'Sil':>8} {'MAE':>8} {'NDCG':>8} {'MdMAE':>8} {'MdNDCG':>8}")
    print("-" * 105)
    for _, r in down_df.sort_values("ndcg_at_10", ascending=False).iterrows():
        print(
            f"{r['label']:<12} {r['wcss']:>10.1f} {r['davies_bouldin']:>8.4f} {r['silhouette']:>8.4f} "
            f"{r['mae']:>8.4f} {r['ndcg_at_10']:>8.4f} "
            f"{(r['meta_dual_mae'] if pd.notna(r.get('meta_dual_mae')) else float('nan')):>8.4f} "
            f"{(r['meta_dual_ndcg'] if pd.notna(r.get('meta_dual_ndcg')) else float('nan')):>8.4f}"
        )

    if args.csv:
        out = args.csv if os.path.isabs(args.csv) else os.path.join(REPO, args.csv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        clust_df.to_csv(out.replace(".csv", "_cluster_all_k.csv"), index=False)
        sep_df.to_csv(out.replace(".csv", "_k_separation.csv"), index=False)
        down_df.to_csv(out, index=False)
        print(f"\nCSV: {out} (+ cluster_all_k, k_separation)")


if __name__ == "__main__":
    main()
