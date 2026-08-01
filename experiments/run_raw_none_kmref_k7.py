"""
Ham matris (preprocess/FE yok) + irand + nogs + no-prune + kmref (meta) — K=7.

  python experiments/run_raw_none_kmref_k7.py --phase assign
  python experiments/run_raw_none_kmref_k7.py --phase compare
  python experiments/run_raw_none_kmref_k7.py --phase eval
  python experiments/run_raw_none_kmref_k7.py --phase all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mealpy"))

from wnmf.meta_dual_cf import _load_centroids as load_meta_centroids, predict_meta_dual  # noqa: E402
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

from compare_cluster_structure import (  # noqa: E402
    _common_mask,
    _load_centroids as load_compare_centroids,
    _load_labels_and_gray,
    _structure_stats,
    mean_centroid_distance,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

GEN = os.path.join(REPO, "mealpy", "generate_assignments.py")
ASSIGN_ROOT = os.path.join(REPO, "mealpy", "results", "assignments", "ml100k")
ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
K = 7

# pearson → metric_suffix yok; init + nogs
OUT_SUFFIX = "_irand_nogs"
ASSIGN_SUFFIX = "_none_none20_k7"
ASSIGN_SUFFIX_KMREF = ASSIGN_SUFFIX + "_kmref"
OUT_CSV = os.path.join(REPO, "results", "raw_none_kmref_k7_cluster_preds.csv")
KNN_K = 20
SIM = "cosine"
MIN_COMMON = 3
_EVAL_ARGS = Namespace(similarity=SIM, min_common=MIN_COMMON)
KNN_PREDICTORS = (
    "cluster_knn_native",
    "cluster_knn_surprise_baseline",
    "cluster_knn_with_means",
)
CLUSTER_PREDICTORS = (
    "cluster_avg",
    "cluster_avg_hard",
    *KNN_PREDICTORS,
    "meta_dual",
)


def folder_suffix(algo: str) -> str:
    suf = ASSIGN_SUFFIX_KMREF if algo != "B0_KMEANS" else ASSIGN_SUFFIX
    return OUT_SUFFIX + suf


def assign_dir(algo: str) -> str:
    return os.path.join(ASSIGN_ROOT, f"{algo}{folder_suffix(algo)}")


def gen_cmd(jobs: int) -> List[str]:
    return [
        sys.executable, GEN,
        "--dataset", "100k",
        "--k", str(K),
        "--algo", *ALGOS,
        "--no-prune",
        "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "none",
        "--init-mode", "random",
        "--cluster-metric", "pearson",
        "--kmeans-refine-overwrite",
        "--jobs", str(jobs),
    ]


def run_compare(csv_dir: str) -> None:
    loaded: Dict[str, dict] = {}
    for algo in ALGOS:
        fp = os.path.join(assign_dir(algo), "assignments.npy")
        if not os.path.isfile(fp):
            print(f"  EKSIK: {assign_dir(algo)}")
            continue
        a, g = _load_labels_and_gray(fp)
        c = load_compare_centroids(fp, K)
        loaded[algo] = {"labels": a, "gray": g, "centroids": c}
        st = _structure_stats(a, K)
        print(
            f"  {algo}: aktif={int(st['n_active'])} bos={int(st['n_empty'])} "
            f"CV={st['size_cv']:.3f} Gini={st['gini_sizes']:.3f}"
        )

    present = [a for a in ALGOS if a in loaded]
    if len(present) < 2:
        return

    os.makedirs(csv_dir, exist_ok=True)
    tag = f"raw_none_kmref_k{K}"
    for metric in ("ari", "nmi", "centroid_l2_mean"):
        n = len(present)
        mat = np.full((n, n), np.nan)
        for i, ai in enumerate(present):
            for j, aj in enumerate(present):
                if i == j:
                    mat[i, j] = 0.0 if metric == "centroid_l2_mean" else 1.0
                    continue
                la, ga = loaded[ai]["labels"], loaded[ai]["gray"]
                lb, gb = loaded[aj]["labels"], loaded[aj]["gray"]
                mask = _common_mask(ga, gb, len(la))
                if int(mask.sum()) < 2:
                    continue
                if metric == "centroid_l2_mean":
                    ca, cb = loaded[ai]["centroids"], loaded[aj]["centroids"]
                    if ca is not None and cb is not None and ca.shape == cb.shape:
                        mat[i, j] = mean_centroid_distance(ca, cb)
                elif metric == "ari":
                    mat[i, j] = adjusted_rand_score(la[mask], lb[mask])
                else:
                    mat[i, j] = normalized_mutual_info_score(
                        la[mask], lb[mask], average_method="arithmetic",
                    )
                mat[j, i] = mat[i, j]
        df = pd.DataFrame(mat, index=present, columns=present)
        path = os.path.join(csv_dir, f"{tag}_{metric}.csv")
        df.round(4).to_csv(path)
        print(f"\n{metric.upper()}:\n{df.round(4).to_string()}\n  -> {path}")


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
    assign_b0: Optional[np.ndarray] = None,
    knn_only: bool = False,
) -> List[dict]:
    rows: List[dict] = []
    nc_avg = _nearest_centroid_bundle(None, adir, assignments)
    nc_knn = _knn_centroid_bundle(None, adir, assignments, knn_mode="cluster")
    common = dict(top_n=10, relevance_threshold=4.0, assign_dir=adir)

    specs: List[tuple] = []
    if not knn_only:
        specs.extend([
            ("cluster_avg", lambda: run_cluster_average(
                train, test, assignments, gray_mask, memberships, n_items, algo,
                **_cluster_avg_predict_kwargs(_EVAL_ARGS), **nc_avg, **common,
            )),
            ("cluster_avg_hard", lambda: run_cluster_average(
                train, test, assignments, gray_mask, memberships, n_items, algo,
                cluster_avg_hard=True, **nc_avg, **common,
            )),
        ])
    specs.extend([
        ("cluster_knn_native", lambda: run_cluster_knn(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            user_features=uf, similarity=SIM, min_common=MIN_COMMON,
            k_neighbors=KNN_K, cluster_knn_backend="native", **nc_knn, **common,
        )),
        ("cluster_knn_surprise_baseline", lambda: run_cluster_knn(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            user_features=uf, similarity=SIM, min_common=MIN_COMMON,
            k_neighbors=KNN_K, cluster_knn_backend="surprise",
            surprise_knn_variant="baseline", **nc_knn, **common,
        )),
        ("cluster_knn_with_means", lambda: run_cluster_knn(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            user_features=uf, similarity=SIM, min_common=MIN_COMMON,
            k_neighbors=KNN_K, cluster_knn_backend="surprise",
            surprise_knn_variant="withmeans", **nc_knn, **common,
        )),
    ])

    for name, fn in specs:
        t0 = time.time()
        r = fn()
        rows.append({
            "k": K,
            "algo": algo,
            "predictor": name,
            "scenario": r.get("scenario", name),
            "mae": r["mae"],
            "rmse": r["rmse"],
            "ndcg_at_10": r["ndcg_at_10"],
            "precision_at_10": r.get("precision_at_10", np.nan),
            "recall_at_10": r.get("recall_at_10", np.nan),
            "time_s": round(time.time() - t0, 1),
        })

    if (
        not knn_only
        and algo != "B0_KMEANS"
        and assign_b0 is not None
        and os.path.isfile(os.path.join(adir, "user_features.npy"))
    ):
        centroids = load_meta_centroids(adir, K, uf.shape[1])
        t0 = time.time()
        r = predict_meta_dual(
            train, test, assign_b0, assignments, uf, centroids,
            beta=0.85, tau=None, tune=False,
        )
        rows.append({
            "k": K,
            "algo": algo,
            "predictor": "meta_dual",
            "scenario": "meta_dual",
            "mae": r["mae"],
            "rmse": r["rmse"],
            "ndcg_at_10": r["ndcg_at_10"],
            "precision_at_10": r.get("precision_at_10", np.nan),
            "recall_at_10": r.get("recall_at_10", np.nan),
            "time_s": round(time.time() - t0, 1),
        })

    return rows


def _print_eval_summary(df: pd.DataFrame, *, knn_only: bool = False) -> None:
    cols = list(KNN_PREDICTORS) if knn_only else [c for c in CLUSTER_PREDICTORS if c in df["predictor"].unique()]
    print("\n=== MAE (düşük iyi) — algo × predictor ===")
    print(df.pivot_table(index="algo", columns="predictor", values="mae", aggfunc="first")[cols].round(4).to_string())
    print("\n=== NDCG@10 (yüksek iyi) ===")
    print(df.pivot_table(index="algo", columns="predictor", values="ndcg_at_10", aggfunc="first")[cols].round(4).to_string())
    print("\n=== En iyi MAE (predictor başına) ===")
    for pred in cols:
        sub = df[df["predictor"] == pred]
        if sub.empty:
            continue
        b = sub.loc[sub["mae"].idxmin()]
        print(f"  [{pred:<28}] {b['algo']:<12} MAE={b['mae']:.4f}  NDCG={b['ndcg_at_10']:.4f}")


def run_eval(csv_path: str, *, skip_existing: bool = False, knn_only: bool = False) -> None:
    data_path = os.path.join(REPO, "data", "ml-100k", "u.data")
    train, test = load_ratings_100k_all(data_path, random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1

    b0_dir = assign_dir("B0_KMEANS")
    if not os.path.isfile(os.path.join(b0_dir, "assignments.npy")):
        raise FileNotFoundError(f"B0 assignment yok: {b0_dir}")

    a0, g0 = load_assignment(b0_dir)
    m0 = load_memberships(b0_dir)
    uf0 = load_user_features(b0_dir, len(a0))
    a0, g0, m0, uf0 = _align_assignment_bundle(
        a0, g0, m0, uf0, n_users_expected=n_users,
        algo_label="B0_KMEANS", assign_dir=b0_dir,
    )

    out = Path(csv_path)
    done: set = set()
    if skip_existing and out.is_file():
        for _, r in pd.read_csv(out).iterrows():
            done.add((str(r["algo"]), str(r["predictor"])))

    all_rows: List[dict] = []
    tag = "kNN-only" if knn_only else "cluster (sharedV yok)"
    print(f"  Eval: {tag}  knn={KNN_K}  sim={SIM}  fold=1")

    for algo in ALGOS:
        adir = assign_dir(algo)
        if not os.path.isfile(os.path.join(adir, "assignments.npy")):
            print(f"SKIP {algo}: {adir}")
            continue

        assignments, gray_mask = load_assignment(adir)
        memberships = load_memberships(adir)
        uf = load_user_features(adir, len(assignments))
        assignments, gray_mask, memberships, uf = _align_assignment_bundle(
            assignments, gray_mask, memberships, uf,
            n_users_expected=n_users, algo_label=algo, assign_dir=adir,
        )

        batch = _eval_predictors(
            train, test, assignments, gray_mask, memberships, n_items,
            algo, adir, uf,
            assign_b0=a0 if algo != "B0_KMEANS" else None,
            knn_only=knn_only,
        )
        new_batch = [r for r in batch if (r["algo"], r["predictor"]) not in done]
        all_rows.extend(new_batch)
        if new_batch:
            print(
                f"{algo:<12}  "
                + "  ".join(f"{r['predictor'][:12]}={r['mae']:.3f}" for r in new_batch),
            )

    if not all_rows and out.is_file():
        df = pd.read_csv(out)
    elif all_rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame(all_rows)
        if out.is_file():
            df = pd.concat([pd.read_csv(out), df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["algo", "predictor"], keep="last")
        else:
            df = df_new
        df.to_csv(out, index=False)
    else:
        print("Yeni eval satırı yok.")
        if out.is_file():
            _print_eval_summary(pd.read_csv(out), knn_only=knn_only)
        return

    print(f"\nKayıt: {out}  ({len(df)} satır)")
    _print_eval_summary(df, knn_only=knn_only)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=("assign", "compare", "eval", "all"),
        default="eval",
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument(
        "--csv-dir",
        default=os.path.join(REPO, "results", "raw_none_kmref_k7_cluster_sim"),
    )
    p.add_argument("--pred-csv", default=OUT_CSV)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--knn-only",
        action="store_true",
        help="Yalnız cluster_knn_* tahmincileri (avg/meta_dual atlanır)",
    )
    args = p.parse_args()

    print("Beklenen klasorler:")
    for algo in ALGOS:
        print(f"  {algo}{folder_suffix(algo)}")

    if args.phase in ("assign", "all"):
        cmd = gen_cmd(args.jobs)
        print("\nKomut:\n ", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO, check=True)

    if args.phase in ("compare", "all"):
        print("\n=== Küme benzerliği (ham matris, kmref meta) ===")
        run_compare(os.path.normpath(args.csv_dir))

    if args.phase in ("eval", "all"):
        print("\n=== Küme tahmin ===")
        run_eval(
            os.path.normpath(args.pred_csv),
            skip_existing=args.skip_existing,
            knn_only=args.knn_only,
        )


if __name__ == "__main__":
    main()
