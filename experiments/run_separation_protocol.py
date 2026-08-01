"""
Meta-algoritma ayrışmasını maksimize eden atama protokolleri + küme tahmin karşılaştırması.

Protokoller (klasör suffix farklı):
  euc_multi   — euclidean + MO WCSS/sil/CH (imkpp)
  fuzzy_fcm   — fuzzy/FCM centroid (makale-tipi AVOA)
  euc_knnmae  — euclidean + --fitness knn_mae (downstream hizalı arama)

K sweep: 10, 14, 21 (küçük K=3,7 ayrıştırmıyor; K=30 downstream çöküyor)

  python experiments/run_separation_protocol.py --phase assign --jobs 4
  python experiments/run_separation_protocol.py --phase separate
  python experiments/run_separation_protocol.py --phase assign-kmref --protocol fuzzy_fcm --k 14
  python experiments/run_separation_protocol.py --phase eval --protocol fuzzy_fcm --k 14
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"

META_ALGOS = ["B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
CORE_ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
K_LIST = [10, 14, 21]
WNMF_DIM = 20
KNN_K = 20
SIM = "cosine"
MIN_COMMON = 3

PROTOCOLS: Dict[str, dict] = {
    "euc_multi": {
        "cluster_metric": "euclidean",
        "fitness": "wcss",
        "cluster_objective": "multi",
        "extra_suffix": "",
    },
    "fuzzy_fcm": {
        "cluster_metric": "fuzzy",
        "fitness": "wcss",
        "cluster_objective": "multi",
        "extra_suffix": "",
    },
    "euc_knnmae": {
        "cluster_metric": "euclidean",
        "fitness": "knn_mae",
        "cluster_objective": "multi",
        "extra_suffix": "_knnmae",
    },
}

BASE_SUFFIX = "_imkpp_nogs_trainonly_rand_f1_none_wnmf{dim}_k{k}"


def suffix(protocol: str, k: int, *, kmref: bool = False, dim: int = WNMF_DIM) -> str:
    p = PROTOCOLS[protocol]
    metric = "_fuzzy" if p["cluster_metric"] == "fuzzy" else "_euc"
    s = f"{metric}{BASE_SUFFIX.format(dim=dim, k=k)}{p['extra_suffix']}"
    if kmref:
        s += "_kmref"
    return s


def assign_dir(algo: str, protocol: str, k: int, *, kmref: bool = False) -> Path:
    return ASSIGN_ROOT / f"{algo}{suffix(protocol, k, kmref=kmref)}"


def _gen_cmd(
    protocol: str,
    ks: Sequence[int],
    algos: Sequence[str],
    jobs: int,
    skip_existing: bool,
    kmref: bool,
    python: str,
) -> List[str]:
    p = PROTOCOLS[protocol]
    cmd = [
        python, str(GEN),
        "--dataset", "100k",
        "--algo", *list(algos),
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--init-mode", "mkpp",
        "--cluster-metric", p["cluster_metric"],
        "--fitness", p["fitness"],
        "--cluster-objective", p["cluster_objective"],
        "--train-only", "--eval-split", "random", "--fold", "1",
        "--k", *[str(k) for k in ks],
        "--jobs", str(int(jobs)),
    ]
    if kmref:
        cmd.append("--kmeans-refine-overwrite")
    if skip_existing:
        cmd.append("--skip-existing")
    return cmd


def phase_assign(
    protocols: Sequence[str],
    ks: Sequence[int],
    jobs: int,
    skip_existing: bool,
    kmref: bool,
    python: str,
) -> int:
    algos = META_ALGOS if kmref else CORE_ALGOS
    for protocol in protocols:
        cmd = _gen_cmd(protocol, ks, algos, jobs, skip_existing, kmref, python)
        print("\n" + "=" * 72)
        print(f"PROTOCOL={protocol}  kmref={kmref}")
        print(" ".join(cmd))
        print("=" * 72)
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            return rc
    return 0


def phase_separation(protocols: Sequence[str], ks: Sequence[int]) -> pd.DataFrame:
    """Meta çiftleri: düşük ARI = yüksek ayrışma."""
    rows = []
    for protocol in protocols:
        for k in ks:
            labels = {}
            for algo in CORE_ALGOS:
                adir = assign_dir(algo, protocol, k, kmref=False)
                p = adir / "assignments.npy"
                if not p.is_file():
                    continue
                labels[algo] = np.load(p)
            if len(labels) < 2:
                continue
            meta = {a: labels[a] for a in META_ALGOS if a in labels}
            pairs_ari, pairs_nmi, pairs_agree = [], [], []
            for a, b in combinations(meta.keys(), 2):
                la, lb = meta[a], meta[b]
                pairs_ari.append(adjusted_rand_score(la, lb))
                pairs_nmi.append(
                    normalized_mutual_info_score(la, lb, average_method="arithmetic"),
                )
                pairs_agree.append(float((la == lb).mean()))
            wcss_vals = []
            for algo, la in labels.items():
                adir = assign_dir(algo, protocol, k)
                bs = adir / "best_sol.npy"
                if bs.is_file():
                    uf = adir / "user_features.npy"
                    if not uf.is_file():
                        uf = adir / "wnmf_user_vectors.npy"
                    if uf.is_file():
                        U = np.load(uf)
                        C = np.load(bs).reshape(k, -1)
                        # quick WCSS proxy
                        d = ((U[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
                        cid = la.astype(int)
                        wcss_vals.append(float(d[np.arange(len(la)), cid].sum()))

            rows.append({
                "protocol": protocol,
                "k": k,
                "n_meta": len(meta),
                "mean_ari": float(np.mean(pairs_ari)) if pairs_ari else np.nan,
                "mean_nmi": float(np.mean(pairs_nmi)) if pairs_nmi else np.nan,
                "mean_agree": float(np.mean(pairs_agree)) if pairs_agree else np.nan,
                "separation_score": float(1.0 - np.mean(pairs_ari)) if pairs_ari else np.nan,
                "wcss_std": float(np.std(wcss_vals)) if len(wcss_vals) >= 2 else np.nan,
            })
    df = pd.DataFrame(rows).sort_values(
        ["separation_score", "wcss_std"], ascending=[False, False],
    )
    out = REPO / "results" / "separation_protocol_scores.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\n-> {out}")
    return df


def phase_eval(protocol: str, k: int, *, kmref: bool = False) -> None:
    from wnmf.meta_dual_cf import _load_centroids, predict_meta_dual
    from wnmf.wnmf_experiment import (
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

    data = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    eval_args = Namespace(similarity=SIM, min_common=MIN_COMMON)

    b0_path = assign_dir("B0_KMEANS", protocol, k, kmref=False)
    a0, g0 = load_assignment(str(b0_path))
    m0 = load_memberships(str(b0_path))
    uf0 = load_user_features(str(b0_path), len(a0))
    a0, g0, m0, uf0 = _align_assignment_bundle(
        a0, g0, m0, uf0, n_users_expected=n_users,
        algo_label="B0_KMEANS", assign_dir=str(b0_path),
    )

    rows = []
    algos = META_ALGOS if kmref else CORE_ALGOS
    for algo in algos:
        adir = assign_dir(algo, protocol, k, kmref=kmref)
        if not (adir / "assignments.npy").is_file():
            print(f"SKIP {algo}")
            continue
        assignments, gray_mask = load_assignment(str(adir))
        memberships = load_memberships(str(adir))
        uf = load_user_features(str(adir), len(assignments))
        assignments, gray_mask, memberships, uf = _align_assignment_bundle(
            assignments, gray_mask, memberships, uf,
            n_users_expected=n_users, algo_label=algo, assign_dir=str(adir),
        )
        nc_avg = _nearest_centroid_bundle(None, str(adir), assignments)
        nc_knn = _knn_centroid_bundle(None, str(adir), assignments, knn_mode="cluster")
        common = dict(top_n=10, relevance_threshold=4.0, assign_dir=str(adir))

        for pred_name, fn in (
            ("cluster_avg", lambda: run_cluster_average(
                train, test, assignments, gray_mask, memberships, n_items, algo,
                **_cluster_avg_predict_kwargs(eval_args), **nc_avg, **common,
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
        ):
            t0 = time.time()
            r = fn()
            rows.append({
                "protocol": protocol, "k": k, "kmref": kmref,
                "algo": algo, "predictor": pred_name,
                "mae": r["mae"], "ndcg_at_10": r["ndcg_at_10"],
                "time_s": round(time.time() - t0, 1),
            })

        if algo in META_ALGOS:
            C = _load_centroids(str(adir), k, uf.shape[1])
            t0 = time.time()
            r = predict_meta_dual(train, test, a0, assignments, uf, C, beta=0.85)
            rows.append({
                "protocol": protocol, "k": k, "kmref": kmref,
                "algo": algo, "predictor": "meta_dual",
                "mae": r["mae"], "ndcg_at_10": r["ndcg_at_10"],
                "time_s": round(time.time() - t0, 1),
            })

    df = pd.DataFrame(rows)
    tag = "_kmref" if kmref else ""
    out = REPO / "results" / f"separation_protocol_eval_{protocol}_k{k}{tag}.csv"
    df.to_csv(out, index=False)
    print(df.pivot_table(index=["algo"], columns="predictor", values="mae").to_string())
    print(f"\n-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["assign", "assign-kmref", "separate", "eval", "all"],
        default="all",
    )
    ap.add_argument("--protocol", default=None, help="eval/assign-kmref için tek protokol")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    protocols = list(PROTOCOLS.keys())
    if args.protocol:
        protocols = [args.protocol]

    if args.phase in ("assign", "all"):
        rc = phase_assign(protocols, K_LIST, args.jobs, args.skip_existing, False, sys.executable)
        if rc:
            sys.exit(rc)

    if args.phase in ("separate", "all"):
        phase_separation(list(PROTOCOLS.keys()), K_LIST)

    if args.phase in ("assign-kmref", "all"):
        pk = [args.k] if args.k else K_LIST
        pprots = [args.protocol] if args.protocol else protocols
        rc = phase_assign(pprots, pk, args.jobs, args.skip_existing, True, sys.executable)
        if rc:
            sys.exit(rc)

    if args.phase in ("eval", "all"):
        if not args.protocol or not args.k:
            # en iyi ayrışma: skor dosyasından oku
            sc = REPO / "results" / "separation_protocol_scores.csv"
            if sc.is_file():
                top = pd.read_csv(sc).iloc[0]
                prot, k = str(top["protocol"]), int(top["k"])
                print(f"eval: en yüksek separation → {prot} K={k}")
            else:
                prot, k = "fuzzy_fcm", 14
                print(f"eval: varsayılan {prot} K={k}")
        else:
            prot, k = args.protocol, int(args.k)
        phase_eval(prot, k, kmref=False)
        if (assign_dir("B1_HHO", prot, k, kmref=True) / "assignments.npy").is_file():
            phase_eval(prot, k, kmref=True)


if __name__ == "__main__":
    main()
