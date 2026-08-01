"""
PCA80% + none20 protokolü: assignment üret, kNN (cluster_min) + cluster_avg eval,
algoritmalar arası küme benzerliği, paired bootstrap CI.

Suffix: _pruneu5_i10_zscore_pca80pct_euc_imkpp_none_none20_k{K}

  python experiments/run_pca80pct_none_k_protocol.py --phase assign --jobs 4
  python experiments/run_pca80pct_none_k_protocol.py --phase sync-db
  python experiments/run_pca80pct_none_k_protocol.py --phase eval
  python experiments/run_pca80pct_none_k_protocol.py --phase cluster-compare
  python experiments/run_pca80pct_none_k_protocol.py --phase bootstrap
  python experiments/run_pca80pct_none_k_protocol.py --phase all --jobs 4
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))
sys.path.insert(0, str(REPO / "mealpy"))

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments_lof" / "ml100k"
DB_PATH = REPO / "results" / "assignment_experiments.sqlite"

ALGOS = ["HA_AVOAHGS", "B2_HGS", "B_AVOA", "LIT_GWO", "LIT_PSO"]
ASSIGN_K = [5, 10, 15, 40, 50, 60, 70]
EVAL_K = [5, 10, 15, 40, 50, 60, 70]
SIMILARITIES = ("cosine", "pearson")
MIN_COMMON = 3
FOLD = 1
FEAT_DIM = 20
BOOT_REFERENCE = "HA_AVOAHGS"

OUT_PREDS = REPO / "results" / "pca80pct_none_k_protocol_preds.csv"
OUT_CLUSTER = REPO / "results" / "pca80pct_none_k_protocol_cluster_pairs.csv"
OUT_BOOT = REPO / "results" / "pca80pct_none_k_protocol_bootstrap.csv"


def suffix(k: int) -> str:
    return f"_pruneu5_i10_zscore_pca80pct_euc_imkpp_none_none20_k{int(k)}"


def assign_dir(algo: str, k: int) -> Path:
    return ASSIGN_ROOT / f"{algo}{suffix(k)}"


def cluster_stats(assignments: np.ndarray) -> dict:
    sizes = sorted(Counter(assignments.astype(int).tolist()).values())
    if not sizes:
        return {
            "n_active_clusters": 0,
            "cluster_min": 0,
            "cluster_max": 0,
            "cluster_std": float("nan"),
            "singletons": 0,
        }
    return {
        "n_active_clusters": len(sizes),
        "cluster_min": min(sizes),
        "cluster_max": max(sizes),
        "cluster_std": float(np.std(sizes)),
        "singletons": sum(1 for s in sizes if s == 1),
    }


def _db_assignment_id(algo: str, k: int) -> Optional[int]:
    if not DB_PATH.is_file():
        return None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        SELECT id FROM assignments
        WHERE dataset='ml100k' AND algo=? AND k=? AND assign_suffix=?
        ORDER BY wcss ASC LIMIT 1
        """,
        (algo, int(k), suffix(k)),
    ).fetchone()
    conn.close()
    return int(row[0]) if row else None


def sync_db_to_disk(algo: str, k: int, *, overwrite: bool = False) -> bool:
    import assignment_db as adb

    aid = _db_assignment_id(algo, k)
    if aid is None:
        return False
    out = assign_dir(algo, k)
    if (out / "assignments.npy").is_file() and not overwrite:
        return True
    record = adb.load_assignment_by_id(aid)
    if record is None:
        return False
    out.mkdir(parents=True, exist_ok=True)
    adb.export_assignment_to_dir(record, str(out), overwrite=True)
    return True


def ensure_on_disk(algo: str, k: int) -> bool:
    if (assign_dir(algo, k) / "assignments.npy").is_file():
        return True
    return sync_db_to_disk(algo, k)


def phase_assign(jobs: int, ks: Sequence[int], skip_existing: bool) -> int:
    cmd = [
        sys.executable, str(GEN),
        "--dataset", "100k",
        "--lof",
        "--zscore",
        "--preprocess", "none",
        "--feature-extraction", "none",
        "--svd-components", str(FEAT_DIM),
        "--pca", "0.8",
        "--cluster-metric", "euclidean",
        "--init-mode", "mkpp",
        "--algo", *ALGOS,
        "--k", *[str(k) for k in ks],
        "--jobs", str(jobs),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    for algo in ALGOS:
        for k in ks:
            sync_db_to_disk(algo, k, overwrite=False)
    return rc


def phase_sync_db(ks: Sequence[int]) -> int:
    n = 0
    for k in ks:
        for algo in ALGOS:
            if sync_db_to_disk(algo, k):
                n += 1
                print(f"  synced {algo} K={k}")
            else:
                print(f"  missing DB {algo} K={k}")
    print(f"sync-db: {n} assignment exported")
    return 0


def _append_csv(rows: List[dict], out_csv: Path, key: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.is_file():
        old = pd.read_csv(out_csv)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=list(key), keep="last")
        merged = merged.sort_values(list(key)).reset_index(drop=True)
        merged.to_csv(out_csv, index=False)
        return merged
    df.to_csv(out_csv, index=False)
    return df


def _eval_key_cols() -> List[str]:
    return ["fold", "k", "algo", "predictor", "similarity", "knn_k"]


def _load_done_eval_keys(out_csv: Path) -> set:
    if not out_csv.is_file():
        return set()
    df = pd.read_csv(out_csv)
    cols = _eval_key_cols()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return set()
    return set(map(tuple, df[cols].itertuples(index=False, name=None)))


def phase_eval(ks: Sequence[int], out_csv: Path, skip_existing: bool = False) -> pd.DataFrame:
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
    train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=FOLD)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    done = _load_done_eval_keys(out_csv) if skip_existing else set()
    key = _eval_key_cols()

    rows: List[dict] = []
    for k in ks:
        for algo in ALGOS:
            if not ensure_on_disk(algo, k):
                print(f"SKIP eval {algo} K={k} (atama yok)")
                continue
            adir = assign_dir(algo, k)
            assignments, gray_mask = load_assignment(str(adir))
            memberships = load_memberships(str(adir))
            uf = load_user_features(str(adir), len(assignments))
            assignments, gray_mask, memberships, uf = _align_assignment_bundle(
                assignments, gray_mask, memberships, uf,
                n_users_expected=n_users, algo_label=algo, assign_dir=str(adir),
            )
            st = cluster_stats(assignments)
            k_min = max(1, int(st["cluster_min"]))
            nc_avg = _nearest_centroid_bundle(None, str(adir), assignments)
            nc_knn = _knn_centroid_bundle(None, str(adir), assignments, knn_mode="cluster")
            common = dict(top_n=10, relevance_threshold=4.0, assign_dir=str(adir))

            def _base_row(predictor: str, sim: str, r: dict, t0: float, *, knn_k: int = 0) -> dict:
                return {
                    "protocol": "pca80pct_none20",
                    "fold": FOLD,
                    "feature_dim": FEAT_DIM,
                    "k": k,
                    "algo": algo,
                    "predictor": predictor,
                    "similarity": sim,
                    "knn_k": int(knn_k),
                    "mae": r["mae"],
                    "rmse": r["rmse"],
                    "ndcg_at_10": r["ndcg_at_10"],
                    "precision_at_10": r["precision_at_10"],
                    "recall_at_10": r["recall_at_10"],
                    "coverage_at_10": r.get("coverage_at_10", float("nan")),
                    "assign_suffix": suffix(k),
                    "eval_seconds": round(time.time() - t0, 1),
                    **st,
                }

            block_rows: List[dict] = []
            for sim in SIMILARITIES:
                eval_args = Namespace(similarity=sim, min_common=MIN_COMMON)

                ek_avg = (FOLD, k, algo, "cluster_avg", sim, 0)
                if skip_existing and ek_avg in done:
                    print(f"  SKIP {algo} K={k} cluster_avg sim={sim} (mevcut)")
                else:
                    t0 = time.time()
                    r = run_cluster_average(
                        train, test, assignments, gray_mask, memberships, n_items, algo,
                        **_cluster_avg_predict_kwargs(eval_args), **nc_avg, **common,
                    )
                    block_rows.append(_base_row("cluster_avg", sim, r, t0))
                    print(f"  {algo} K={k} cluster_avg sim={sim}: MAE={r['mae']:.4f}")

                ek_knn = (FOLD, k, algo, "cluster_knn", sim, k_min)
                if skip_existing and ek_knn in done:
                    print(f"  SKIP {algo} K={k} cluster_knn sim={sim} (mevcut)")
                else:
                    t0 = time.time()
                    r = run_cluster_knn(
                        train, test, assignments, gray_mask, memberships, n_items, algo,
                        user_features=uf,
                        similarity=sim,
                        min_common=MIN_COMMON,
                        k_neighbors=k_min,
                        **nc_knn,
                        **common,
                    )
                    block_rows.append(_base_row("cluster_knn", sim, r, t0, knn_k=k_min))
                    print(
                        f"  {algo} K={k} cluster_knn sim={sim} knn={k_min}: "
                        f"MAE={r['mae']:.4f}"
                    )

            if block_rows:
                _append_csv(block_rows, out_csv, key)
                rows.extend(block_rows)

    if out_csv.is_file():
        return pd.read_csv(out_csv)
    return pd.DataFrame(rows)


def phase_cluster_compare(ks: Sequence[int], out_csv: Path) -> pd.DataFrame:
    from compare_cluster_structure import (
        _common_mask,
        _load_centroids,
        _load_labels_and_gray,
        _structure_stats,
        mean_centroid_distance,
    )
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    rows: List[dict] = []
    for k in ks:
        loaded: Dict[str, tuple] = {}
        for algo in ALGOS:
            if not ensure_on_disk(algo, k):
                continue
            fp = assign_dir(algo, k) / "assignments.npy"
            a, g = _load_labels_and_gray(str(fp))
            c = _load_centroids(str(fp), k)
            loaded[algo] = (a, g, c, _structure_stats(a, k))

        for a1, a2 in combinations(sorted(loaded), 2):
            la, ga, ca, sa = loaded[a1]
            lb, gb, cb, sb = loaded[a2]
            mask = _common_mask(ga, gb, len(la))
            n_ok = int(mask.sum())
            row = {
                "k": k,
                "algo_a": a1,
                "algo_b": a2,
                "assign_suffix": suffix(k),
                "n_users": n_ok,
                "label_agreement": float("nan"),
                "ari": float("nan"),
                "nmi": float("nan"),
                "centroid_l2_mean": float("nan"),
                f"active_{a1}": sa.get("active_clusters"),
                f"active_{a2}": sb.get("active_clusters"),
                f"min_{a1}": sa.get("min_size"),
                f"min_{a2}": sb.get("min_size"),
            }
            if n_ok >= 2:
                row["label_agreement"] = float((la[mask] == lb[mask]).mean())
                row["ari"] = float(adjusted_rand_score(la[mask], lb[mask]))
                row["nmi"] = float(
                    normalized_mutual_info_score(la[mask], lb[mask], average_method="arithmetic")
                )
            if ca is not None and cb is not None and ca.shape == cb.shape:
                row["centroid_l2_mean"] = mean_centroid_distance(ca, cb)
            rows.append(row)
            print(
                f"  K={k} {a1} vs {a2}: ARI={row['ari']:.3f} "
                f"agree={row['label_agreement']:.3f}"
            )

    key = ["k", "algo_a", "algo_b"]
    return _append_csv(rows, out_csv, key)


def phase_bootstrap(
    ks: Sequence[int],
    out_csv: Path,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 95.0,
) -> pd.DataFrame:
    from mealpy.paired_bootstrap_ci import (
        align_eval_rows,
        paired_bootstrap_delta,
        predict_eval_rows,
        resolve_assign_dir,
    )
    from wnmf.wnmf_experiment import RANDOM_SEED, load_ratings_100k_all

    data = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=FOLD)
    assign_root = str(REPO / "mealpy" / "results" / "assignments_lof")

    rows: List[dict] = []
    others = [a for a in ALGOS if a != BOOT_REFERENCE]

    for k in ks:
        ref_dir = assign_dir(BOOT_REFERENCE, k)
        if not ensure_on_disk(BOOT_REFERENCE, k):
            print(f"SKIP bootstrap K={k} (referans yok)")
            continue
        ref_assign = np.load(ref_dir / "assignments.npy")
        k_min_ref = max(1, min(Counter(ref_assign.astype(int).tolist()).values()))

        for sim in SIMILARITIES:
            cache: Dict[str, np.ndarray] = {}
            for algo in [BOOT_REFERENCE] + others:
                if not ensure_on_disk(algo, k):
                    continue
                adir = resolve_assign_dir(assign_root, algo, k, suffix(k))
                knn_k = k_min_ref if algo == BOOT_REFERENCE else max(
                    1, min(Counter(np.load(Path(adir) / "assignments.npy").astype(int).tolist()).values())
                )
                cache[algo] = predict_eval_rows(
                    train, test, algo, adir,
                    similarity=sim, knn=knn_k, min_common=MIN_COMMON,
                )

            for algo_b in others:
                if BOOT_REFERENCE not in cache or algo_b not in cache:
                    continue
                true, pa, pb = align_eval_rows(cache[BOOT_REFERENCE], cache[algo_b])
                for metric in ("mae", "rmse"):
                    stats = paired_bootstrap_delta(
                        true, pa, pb, metric, n_boot=n_boot, seed=seed, ci=ci,
                    )
                    rows.append({
                        "k": k,
                        "reference": BOOT_REFERENCE,
                        "algo_b": algo_b,
                        "predictor": "cluster_knn",
                        "similarity": sim,
                        "metric": metric,
                        "delta": stats["delta"],
                        "ci_lo": stats["ci_lo"],
                        "ci_hi": stats["ci_hi"],
                        "p_ref_better": stats["p_a_better"],
                        "n_pairs": int(stats["n_pairs"]),
                        "assign_suffix": suffix(k),
                    })
                print(
                    f"  bootstrap K={k} sim={sim} {BOOT_REFERENCE} vs {algo_b}: "
                    f"dMAE={rows[-2]['delta']:.4f} "
                    f"[{rows[-2]['ci_lo']:.4f}, {rows[-2]['ci_hi']:.4f}]"
                )

    key = ["k", "reference", "algo_b", "predictor", "similarity", "metric"]
    return _append_csv(rows, out_csv, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["assign", "sync-db", "eval", "cluster-compare", "bootstrap", "all"],
        default="all",
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--assign-k", type=int, nargs="+", default=ASSIGN_K)
    ap.add_argument("--eval-k", type=int, nargs="+", default=EVAL_K)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    if args.phase in ("assign", "all"):
        print("\n=== ASSIGN ===")
        rc = phase_assign(args.jobs, args.assign_k, args.skip_existing)
        if rc != 0 and args.phase == "assign":
            sys.exit(rc)

    if args.phase in ("sync-db", "all"):
        print("\n=== SYNC DB ===")
        phase_sync_db(list(set(args.assign_k) | set(args.eval_k)))

    if args.phase in ("eval", "all"):
        print("\n=== EVAL ===")
        df = phase_eval(args.eval_k, OUT_PREDS, skip_existing=args.skip_existing)
        print(f"Preds -> {OUT_PREDS} ({len(df)} rows)")

    if args.phase in ("cluster-compare", "all"):
        print("\n=== CLUSTER COMPARE ===")
        df = phase_cluster_compare(args.eval_k, OUT_CLUSTER)
        print(f"Cluster pairs -> {OUT_CLUSTER} ({len(df)} rows)")

    if args.phase in ("bootstrap", "all"):
        print("\n=== BOOTSTRAP CI ===")
        df = phase_bootstrap(args.eval_k, OUT_BOOT, n_boot=args.n_bootstrap)
        print(f"Bootstrap -> {OUT_BOOT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
