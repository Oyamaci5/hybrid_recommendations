"""
NMF-50 + maxabs + no-prune + irand + trainonly fold1 + kmref (meta) — K=7.
Atama, küme benzerliği, sharedV hariç küme tahmin eval.

  python experiments/run_nmf50_maxabs_k7_assign_compare.py --phase assign
  python experiments/run_nmf50_maxabs_k7_assign_compare.py --phase compare
  python experiments/run_nmf50_maxabs_k7_assign_compare.py --phase eval
  python experiments/run_nmf50_maxabs_k7_assign_compare.py --phase all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from argparse import Namespace
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

OUT_SUFFIX = "_euc_irand_nogs_trainonly_rand_f1"
ASSIGN_SUFFIX = "_maxabs_nmf50_k7"
ASSIGN_SUFFIX_KMREF = ASSIGN_SUFFIX + "_kmref"
OUT_CSV = os.path.join(REPO, "results", "nmf50_maxabs_k7_cluster_preds.csv")
KNN_K = 20
SIM = "cosine"
MIN_COMMON = 3
_EVAL_ARGS = Namespace(similarity=SIM, min_common=MIN_COMMON)
CLUSTER_PREDICTORS = (
    "cluster_avg",
    "cluster_avg_hard",
    "cluster_knn_native",
    "cluster_knn_surprise_baseline",
    "cluster_knn_with_means",
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
        "--preprocess", "maxabs",
        "--feature-extraction", "nmf",
        "--svd-components", "50",
        "--cluster-metric", "euclidean",
        "--init-mode", "random",
        "--train-only",
        "--eval-split", "random",
        "--fold", "1",
        "--kmeans-refine-overwrite",
        "--jobs", str(jobs),
    ]


def load_algo(algo: str) -> Optional[dict]:
    fp = os.path.join(assign_dir(algo), "assignments.npy")
    if not os.path.isfile(fp):
        return None
    a, g = _load_labels_and_gray(fp)
    c = load_compare_centroids(fp, K)
    return {"labels": a, "gray": g, "centroids": c, "path": fp}


def compare_pair(la, ga, lb, gb, ca, cb) -> dict:
    mask = _common_mask(ga, gb, len(la))
    n_ok = int(mask.sum())
    out = {"n_users": n_ok, "label_agreement": np.nan, "ari": np.nan, "nmi": np.nan, "centroid_l2_mean": np.nan}
    if n_ok < 2:
        return out
    out["label_agreement"] = float((la[mask] == lb[mask]).mean())
    out["ari"] = float(adjusted_rand_score(la[mask], lb[mask]))
    out["nmi"] = float(normalized_mutual_info_score(la[mask], lb[mask], average_method="arithmetic"))
    if ca is not None and cb is not None and ca.shape == cb.shape:
        out["centroid_l2_mean"] = mean_centroid_distance(ca, cb)
    return out


def run_compare(csv_dir: str) -> None:
    loaded: Dict[str, dict] = {}
    for algo in ALGOS:
        rec = load_algo(algo)
        if rec is None:
            print(f"  EKSIK: {assign_dir(algo)}")
            continue
        loaded[algo] = rec
        st = _structure_stats(rec["labels"], K)
        print(
            f"  {algo}: aktif={int(st['n_active'])} bos={int(st['n_empty'])} "
            f"CV={st['size_cv']:.3f} Gini={st['gini_sizes']:.3f}  -> {rec['path']}"
        )

    present = [a for a in ALGOS if a in loaded]
    if len(present) < 2:
        print("Karsilastirma icin en az 2 algo gerekli.")
        return

    os.makedirs(csv_dir, exist_ok=True)
    tag = f"nmf50_maxabs_k{K}"
    metrics = ("ari", "nmi", "label_agreement", "centroid_l2_mean")

    for metric in metrics:
        n = len(present)
        mat = np.full((n, n), np.nan)
        for i, ai in enumerate(present):
            for j, aj in enumerate(present):
                if i == j:
                    mat[i, j] = 0.0 if metric == "centroid_l2_mean" else 1.0
                    continue
                m = compare_pair(
                    loaded[ai]["labels"], loaded[ai]["gray"],
                    loaded[aj]["labels"], loaded[aj]["gray"],
                    loaded[ai]["centroids"], loaded[aj]["centroids"],
                )
                mat[i, j] = m[metric]
                mat[j, i] = mat[i, j]
        df = pd.DataFrame(mat, index=present, columns=present)
        path = os.path.join(csv_dir, f"{tag}_{metric}.csv")
        df.round(4).to_csv(path)
        print(f"\n{metric.upper()}:\n{df.round(4).to_string()}\n  -> {path}")

    rows = []
    for a, b in combinations(present, 2):
        rows.append({"algo_a": a, "algo_b": b, **compare_pair(
            loaded[a]["labels"], loaded[a]["gray"],
            loaded[b]["labels"], loaded[b]["gray"],
            loaded[a]["centroids"], loaded[b]["centroids"],
        )})
    pairs_path = os.path.join(csv_dir, f"{tag}_pairs.csv")
    pd.DataFrame(rows).round(4).to_csv(pairs_path, index=False)
    print(f"  pairs -> {pairs_path}")


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
) -> List[dict]:
    """sharedV yok — yalnız küme tabanlı tahminciler."""
    rows: List[dict] = []
    nc_avg = _nearest_centroid_bundle(None, adir, assignments)
    nc_knn = _knn_centroid_bundle(None, adir, assignments, knn_mode="cluster")
    common = dict(top_n=10, relevance_threshold=4.0, assign_dir=adir)

    specs = [
        ("cluster_avg", lambda: run_cluster_average(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            **_cluster_avg_predict_kwargs(_EVAL_ARGS), **nc_avg, **common,
        )),
        ("cluster_avg_hard", lambda: run_cluster_average(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            cluster_avg_hard=True, **nc_avg, **common,
        )),
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
    ]

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
        algo != "B0_KMEANS"
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


def _print_eval_summary(df: pd.DataFrame) -> None:
    print("\n=== MAE (düşük iyi) — algo × predictor ===")
    p_mae = df.pivot_table(index="algo", columns="predictor", values="mae", aggfunc="first")
    cols = [c for c in CLUSTER_PREDICTORS if c in p_mae.columns]
    print(p_mae[cols].round(4).to_string())

    print("\n=== NDCG@10 (yüksek iyi) ===")
    p_ndcg = df.pivot_table(index="algo", columns="predictor", values="ndcg_at_10", aggfunc="first")
    print(p_ndcg[cols].round(4).to_string())

    print("\n=== En iyi MAE (algo, predictor) ===")
    best = df.loc[df["mae"].idxmin()]
    print(
        f"  {best['algo']} / {best['predictor']}: "
        f"MAE={best['mae']:.4f}  NDCG={best['ndcg_at_10']:.4f}"
    )
    for pred in cols:
        sub = df[df["predictor"] == pred]
        if sub.empty:
            continue
        b = sub.loc[sub["mae"].idxmin()]
        print(f"  [{pred:<28}] {b['algo']:<12} MAE={b['mae']:.4f}  NDCG={b['ndcg_at_10']:.4f}")


def run_eval(csv_path: str, *, skip_existing: bool = False) -> None:
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
        old = pd.read_csv(out)
        for _, r in old.iterrows():
            done.add((str(r["algo"]), str(r["predictor"])))

    all_rows: List[dict] = []
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
        )
        new_batch = [r for r in batch if (r["algo"], r["predictor"]) not in done]
        all_rows.extend(new_batch)
        if new_batch:
            print(
                f"{algo:<12}  "
                + "  ".join(f"{r['predictor'][:10]}={r['mae']:.3f}" for r in new_batch),
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
        return

    print(f"\nKayıt: {out}  ({len(df)} satır)")
    _print_eval_summary(df)


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
        default=os.path.join(REPO, "results", "nmf50_maxabs_k7_cluster_sim"),
    )
    p.add_argument("--pred-csv", default=OUT_CSV)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    print("Beklenen klasor ornekleri:")
    for algo in ALGOS:
        print(f"  {algo}{folder_suffix(algo)}")

    if args.phase in ("assign", "all"):
        cmd = gen_cmd(args.jobs)
        print("\nAtama komutu:\n ", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO, check=True)

    if args.phase in ("compare", "all"):
        print("\n=== Küme benzerliği ===")
        run_compare(os.path.normpath(args.csv_dir))

    if args.phase in ("eval", "all"):
        print("\n=== Küme tahmin (sharedV yok) ===")
        print(f"  knn={KNN_K}  sim={SIM}  min_common={MIN_COMMON}  fold=1")
        run_eval(os.path.normpath(args.pred_csv), skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
