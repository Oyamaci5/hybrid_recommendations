"""
No-kmref: cluster_avg vs cluster_knn (native) — B0, B1, HA, IWO.

  python run_cluster_avg_vs_knn_k30.py --k 7
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from argparse import Namespace

from wnmf.wnmf_experiment import (  # noqa: E402
    RANDOM_SEED,
    _align_assignment_bundle,
    _cluster_avg_predict_kwargs,
    _knn_centroid_bundle,
    _nearest_centroid_bundle,
    _print_assignment_diagnostics,
    load_assignment,
    load_memberships,
    load_ratings_100k_all,
    load_user_features,
    run_cluster_average,
    run_cluster_knn,
)

ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
ASSIGN_ROOT = os.path.join(REPO, "mealpy", "results", "assignments", "ml100k")
KNN_K = 20
SIM = "cosine"
MIN_COMMON = 3

# cluster_avg ve native kNN aynı benzerlik / min_common
_EVAL_ARGS = Namespace(similarity=SIM, min_common=MIN_COMMON)


def assign_dir(algo: str, suffix: str) -> str:
    return os.path.join(ASSIGN_ROOT, f"{algo}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30, help="Küme sayısı (varsayılan: 30)")
    args = ap.parse_args()
    k_used = int(args.k)
    suffix = f"_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k_used}"
    train, test = load_ratings_100k_all(
        os.path.join(REPO, "data", "ml-100k", "u.data"),
        random_seed=RANDOM_SEED,
        fold=1,
    )
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1

    print(f"K={k_used}  suffix={suffix}")
    print(f"train/test fold=1  knn={KNN_K}  sim={SIM}  min_common={MIN_COMMON}")
    print(f"backend=native (küme-içi komşu)\n")

    rows = []
    for algo in ALGOS:
        adir = assign_dir(algo, suffix)
        if not os.path.isfile(os.path.join(adir, "assignments.npy")):
            print(f"SKIP {algo}: klasör yok")
            continue

        assignments, gray_mask = load_assignment(adir)
        memberships = load_memberships(adir)
        uf = load_user_features(adir, len(assignments))
        assignments, gray_mask, memberships, uf = _align_assignment_bundle(
            assignments, gray_mask, memberships, uf,
            n_users_expected=n_users, algo_label=algo, assign_dir=adir,
        )
        _print_assignment_diagnostics(algo, assignments, uf)

        nc_avg = _nearest_centroid_bundle(None, adir, assignments)
        t0 = time.time()
        r_avg = run_cluster_average(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            top_n=10, relevance_threshold=4.0,
            assign_dir=adir,
            **_cluster_avg_predict_kwargs(_EVAL_ARGS),
            **nc_avg,
        )
        t_avg = time.time() - t0

        nc_knn = _knn_centroid_bundle(None, adir, assignments, knn_mode="cluster")
        t0 = time.time()
        r_knn = run_cluster_knn(
            train, test, assignments, gray_mask, memberships, n_items, algo,
            user_features=uf,
            similarity=SIM,
            min_common=MIN_COMMON,
            k_neighbors=KNN_K,
            cluster_knn_backend="native",
            top_n=10,
            relevance_threshold=4.0,
            assign_dir=adir,
            **nc_knn,
        )
        t_knn = time.time() - t0

        for tag, r, sec in (("cluster_avg", r_avg, t_avg), ("cluster_knn_native", r_knn, t_knn)):
            rows.append({
                "algo": algo,
                "scenario": tag,
                "mae": r["mae"],
                "rmse": r["rmse"],
                "ndcg_at_10": r["ndcg_at_10"],
                "precision_at_10": r["precision_at_10"],
                "recall_at_10": r["recall_at_10"],
                "time_s": round(sec, 1),
            })

        print(
            f"{algo:<12}  avg MAE={r_avg['mae']:.4f} NDCG={r_avg['ndcg_at_10']:.4f} ({t_avg:.0f}s)  |  "
            f"kNN MAE={r_knn['mae']:.4f} NDCG={r_knn['ndcg_at_10']:.4f} ({t_knn:.0f}s)  "
            f"dMAE={r_avg['mae']-r_knn['mae']:+.4f}"
        )

    df = pd.DataFrame(rows)
    out = os.path.join(REPO, "results", f"cluster_avg_vs_knn_native_k{k_used}.csv")
    df.to_csv(out, index=False)

    print(f"\n{'=' * 72}")
    print("ÖZET TABLO")
    print(f"{'Algo':<12} {'cluster_avg MAE':>14} {'kNN native MAE':>14} {'dMAE':>8} "
          f"{'avg NDCG':>9} {'kNN NDCG':>9} {'dNDCG':>8}")
    print("-" * 72)
    for algo in ALGOS:
        sub = df[df["algo"] == algo]
        if len(sub) < 2:
            continue
        a = sub[sub["scenario"] == "cluster_avg"].iloc[0]
        k = sub[sub["scenario"] == "cluster_knn_native"].iloc[0]
        print(
            f"{algo:<12} {a['mae']:>14.4f} {k['mae']:>14.4f} {a['mae']-k['mae']:>+8.4f} "
            f"{a['ndcg_at_10']:>9.4f} {k['ndcg_at_10']:>9.4f} {a['ndcg_at_10']-k['ndcg_at_10']:>+8.4f}"
        )

    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
