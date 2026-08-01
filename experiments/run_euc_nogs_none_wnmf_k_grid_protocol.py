"""
_euc_imkpp_nogs_none_wnmf{W}_k{K}_kmref — K × WNMF grid protokolü.

Varsayılan: official 5-fold — her fold için train-only atama (u{N}.base) + aynı fold'da eval (u{N}.test).
`--assign-scope full` → tek atama (tüm u.data). Fold'lar arası ARI: `compare-folds` fazı.

  python experiments/run_euc_nogs_none_wnmf_k_grid_protocol.py --phase eval --skip-existing
  python experiments/run_euc_nogs_none_wnmf_k_grid_protocol.py --phase eval --fold 3  # tek fold
  python experiments/run_euc_nogs_none_wnmf_k_grid_protocol.py --phase assign --jobs 4
  python experiments/run_euc_nogs_none_wnmf_k_grid_protocol.py --phase cluster-compare
  python experiments/run_euc_nogs_none_wnmf_k_grid_protocol.py --phase stats
  python experiments/run_euc_nogs_none_wnmf_k_grid_protocol.py --phase plot
  python experiments/run_euc_nogs_none_wnmf_k_grid_protocol.py --phase all --jobs 4
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))
sys.path.insert(0, str(REPO / "mealpy"))

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"
DB_PATH = REPO / "results" / "assignment_experiments.sqlite"

WNMF_DIMS = [25, 50, 75, 100, 125, 150]
K_LIST = [3, 5, 7, 10, 12, 14]
ALGOS = ["LIT_GWO", "IWO_HHO", "HA_AVOAHGS", "LIT_PSO", "B_AVOA", "B1_HHO", "B2_HGS"]
CV_FOLDS = [1, 2, 3, 4, 5]
SIM = "cosine"
KNN_FIXED = 50
MIN_COMMON = 3
BOOT_REFERENCE = "HA_AVOAHGS"
HOPKINS_TARGET = 0.85
CV_RAW_REF = 0.497
CV_NORM_REF = 0.283

OUT_PREDS = REPO / "results" / "euc_nogs_none_wnmf_k_grid_preds.csv"
OUT_PREDS_CV5_MEAN = REPO / "results" / "euc_nogs_none_wnmf_k_grid_preds_cv5_mean.csv"
PRED_NUM_COLS = [
    "mae", "rmse", "ndcg_at_10", "precision_at_10", "recall_at_10",
    "jaccard_at_10", "coverage_at_10", "eval_seconds",
]
OUT_CLUSTER = REPO / "results" / "euc_nogs_none_wnmf_k_grid_cluster_pairs.csv"
OUT_ASSIGN = REPO / "results" / "euc_nogs_none_wnmf_k_grid_assignment_metrics.csv"
OUT_HOPKINS = REPO / "results" / "euc_nogs_none_wnmf_k_grid_hopkins_cv.csv"
OUT_STATS = REPO / "results" / "euc_nogs_none_wnmf_k_grid_bootstrap_wilcoxon.csv"
OUT_FOLD_ASSIGN = REPO / "results" / "euc_nogs_none_wnmf_k_grid_fold_assign_pairs.csv"
ASSIGN_SCOPE_OFFICIAL_CV = "official_cv"
ASSIGN_SCOPE_FULL = "full"

ML100K_DIR = REPO / "data" / "ml-100k"
DATA_100K_ALL = ML100K_DIR / "u.data"
DATA_100K_U1_BASE = ML100K_DIR / "u1.base"
DATA_100K_U1_TEST = ML100K_DIR / "u1.test"


def load_eval_train_test(
    eval_fold: int,
    *,
    eval_split: str = "official",
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ML-100K train/test yükle.

    official: GroupLens u{N}.base / u{N}.test (kullanıcı bazlı %%80/%%20, literatür standardı).
    random: u.data üzerinde KFold(5); fold N = splits[N-1].
    """
    from wnmf.wnmf_utils import load_ratings_100k, load_ratings_100k_all

    split = (eval_split or "official").strip().lower()
    if split == "official":
        base = DATA_100K_U1_BASE
        test = DATA_100K_U1_TEST
        if not base.is_file() or not test.is_file():
            raise FileNotFoundError(
                f"Official ML-100K fold dosyaları yok: {base}, {test}. "
                "GroupLens ml-100k.zip içinden u1.base … u5.test kopyalayın."
            )
        return load_ratings_100k(str(base), str(test), fold=int(eval_fold))
    if split == "random":
        if not DATA_100K_ALL.is_file():
            raise FileNotFoundError(f"u.data bulunamadı: {DATA_100K_ALL}")
        return load_ratings_100k_all(
            str(DATA_100K_ALL), random_seed=random_seed, fold=int(eval_fold),
        )
    raise ValueError(f"eval_split bilinmiyor: {eval_split!r} (official | random)")


def assign_suffix(
    wnmf_dim: int,
    k: int,
    *,
    fold: Optional[int] = None,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
) -> str:
    """generate_assignments klasör adı ile uyumlu suffix."""
    if assign_scope == ASSIGN_SCOPE_OFFICIAL_CV and fold is not None:
        return (
            f"_euc_imkpp_nogs_trainonly_official_f{int(fold)}"
            f"_none_wnmf{int(wnmf_dim)}_k{int(k)}_kmref"
        )
    return f"_euc_imkpp_nogs_none_wnmf{int(wnmf_dim)}_k{int(k)}_kmref"


def assign_dir(
    algo: str,
    wnmf_dim: int,
    k: int,
    *,
    fold: Optional[int] = None,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
) -> Path:
    return ASSIGN_ROOT / f"{algo}{assign_suffix(wnmf_dim, k, fold=fold, assign_scope=assign_scope)}"


def _assign_folds(assign_scope: str) -> List[Optional[int]]:
    if assign_scope == ASSIGN_SCOPE_OFFICIAL_CV:
        return list(CV_FOLDS)
    return [None]


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


def _db_assignment_id(
    algo: str, k: int, suffix: str, *, fold: Optional[int] = None,
) -> Optional[int]:
    if not DB_PATH.is_file():
        return None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        SELECT id FROM assignments
        WHERE dataset='ml100k' AND algo=? AND k=? AND assign_suffix=?
        ORDER BY wcss ASC LIMIT 1
        """,
        (algo, int(k), suffix),
    ).fetchone()
    conn.close()
    return int(row[0]) if row else None


def sync_db_to_disk(
    algo: str,
    wnmf_dim: int,
    k: int,
    *,
    fold: Optional[int] = None,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
    overwrite: bool = False,
) -> bool:
    import assignment_db as adb

    suf = assign_suffix(wnmf_dim, k, fold=fold, assign_scope=assign_scope)
    aid = _db_assignment_id(algo, k, suf)
    if aid is None:
        return False
    out = assign_dir(algo, wnmf_dim, k, fold=fold, assign_scope=assign_scope)
    if (out / "assignments.npy").is_file() and not overwrite:
        return True
    record = adb.load_assignment_by_id(aid)
    if record is None:
        return False
    out.mkdir(parents=True, exist_ok=True)
    adb.export_assignment_to_dir(record, str(out), overwrite=True)
    return True


def ensure_on_disk(
    algo: str,
    wnmf_dim: int,
    k: int,
    *,
    fold: Optional[int] = None,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
) -> bool:
    adir = assign_dir(algo, wnmf_dim, k, fold=fold, assign_scope=assign_scope)
    if (adir / "assignments.npy").is_file():
        return True
    return sync_db_to_disk(
        algo, wnmf_dim, k, fold=fold, assign_scope=assign_scope,
    )


def phase_status(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    *,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
) -> int:
    folds = _assign_folds(assign_scope)
    total = len(wnmf_dims) * len(ks) * len(algos) * len(folds)
    disk = db = 0
    missing: List[str] = []
    for fold in folds:
        for w in wnmf_dims:
            for k in ks:
                for algo in algos:
                    on_disk = (
                        assign_dir(algo, w, k, fold=fold, assign_scope=assign_scope)
                        / "assignments.npy"
                    ).is_file()
                    in_db = _db_assignment_id(
                        algo, k, assign_suffix(w, k, fold=fold, assign_scope=assign_scope),
                    ) is not None
                    if on_disk:
                        disk += 1
                    if in_db:
                        db += 1
                    if not on_disk and not in_db:
                        tag = f"f{fold}_" if fold else ""
                        missing.append(f"{tag}wnmf{w}_k{k}_{algo}")
    print(
        f"Grid ({assign_scope}): {total} hücre | DB={db} | disk={disk} | "
        f"eksik={total - max(db, disk)}"
    )
    if missing:
        print("İlk 20 eksik:", ", ".join(missing[:20]))
    return 0


def phase_assign(
    jobs: int,
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    skip_existing: bool,
    *,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
    assign_folds: Optional[Sequence[int]] = None,
) -> int:
    k_str = [str(k) for k in ks]
    if assign_scope == ASSIGN_SCOPE_OFFICIAL_CV:
        folds = list(assign_folds) if assign_folds is not None else _assign_folds(assign_scope)
    else:
        folds = _assign_folds(assign_scope)
    for fold in folds:
        fold_label = f"official fold {fold}" if fold else "full u.data"
        print(f"\n=== ASSIGN {fold_label} ===")
        for w in wnmf_dims:
            if skip_existing and all(
                (
                    assign_dir(algos[0], w, k, fold=fold, assign_scope=assign_scope)
                    / "assignments.npy"
                ).is_file()
                for k in ks
            ):
                print(f"[skip-existing] WNMF={w} fold={fold} tüm K diskte")
                continue
            cmd = [
                sys.executable, str(GEN),
                "--dataset", "100k",
                "--algo", *algos,
                "--no-prune", "--no-gray-sheep",
                "--preprocess", "none",
                "--feature-extraction", "wnmf",
                "--svd-components", str(w),
                "--init-mode", "mkpp",
                "--cluster-metric", "euclidean",
                "--fitness", "wcss",
                "--cluster-objective", "multi",
                "--k", *k_str,
                "--kmeans-refine-overwrite",
                "--jobs", str(jobs),
            ]
            if assign_scope == ASSIGN_SCOPE_OFFICIAL_CV:
                cmd.extend([
                    "--train-only", "--eval-split", "official",
                    "--fold", str(int(fold)),
                ])
            if skip_existing:
                cmd.append("--skip-existing")
            print(" ".join(cmd))
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode
            if rc != 0:
                return rc
            for algo in algos:
                for k in ks:
                    sync_db_to_disk(
                        algo, w, k, fold=fold, assign_scope=assign_scope, overwrite=False,
                    )
    return 0


def phase_sync_db(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    *,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
) -> int:
    n = 0
    for fold in _assign_folds(assign_scope):
        for w in wnmf_dims:
            for k in ks:
                for algo in algos:
                    if sync_db_to_disk(
                        algo, w, k, fold=fold, assign_scope=assign_scope,
                    ):
                        n += 1
                    else:
                        print(f"  missing DB {algo} f={fold} wnmf{w} k{k}")
    print(f"sync-db: {n} exported")
    return 0


def hopkins_statistic(X: np.ndarray, sample_size: int = 100, seed: int = 42) -> float:
    n, d = X.shape
    rng = np.random.default_rng(seed)
    m = min(sample_size, n)
    idx = rng.choice(n, m, replace=False)
    X_sample = X[idx]
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    w = nbrs.kneighbors(X_sample)[0][:, 1]
    X_uniform = rng.uniform(X.min(axis=0), X.max(axis=0), (m, d))
    u = nbrs.kneighbors(X_uniform)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


def _rating_matrix() -> np.ndarray:
    from mealpy_comparison_v2 import load_movielens

    data = REPO / "data" / "ml-100k" / "u.data"
    return load_movielens(str(data))


def phase_hopkins_cv(wnmf_dims: Sequence[int], out_csv: Path) -> pd.DataFrame:
    R = _rating_matrix().astype(np.float64)
    rows: List[dict] = []
    h_rating = hopkins_statistic(R)
    cv_rating = float(np.std(R) / (np.mean(np.abs(R)) + 1e-12))
    rows.append({
        "wnmf_dim": 0,
        "hopkins_raw": h_rating,
        "hopkins_norm": float("nan"),
        "hopkins_target": HOPKINS_TARGET,
        "cv_raw": cv_rating,
        "cv_norm": float("nan"),
        "cv_raw_ref": CV_RAW_REF,
        "cv_norm_ref": CV_NORM_REF,
        "note": "ham_rating_matrix",
    })
    print(f"  Ham rating: H={h_rating:.3f} CV={cv_rating:.3f} (ref raw={CV_RAW_REF})")
    for w in wnmf_dims:
        W_raw = TruncatedSVD(n_components=int(w), random_state=42).fit_transform(R)
        W_norm = normalize(W_raw)
        cv_raw = float(np.std(W_raw) / (np.mean(np.abs(W_raw)) + 1e-12))
        cv_norm = float(np.std(W_norm) / (np.mean(np.abs(W_norm)) + 1e-12))
        h_raw = hopkins_statistic(W_raw.astype(np.float64))
        h_norm = hopkins_statistic(W_norm.astype(np.float64))
        rows.append({
            "wnmf_dim": w,
            "hopkins_raw": h_raw,
            "hopkins_norm": h_norm,
            "hopkins_target": HOPKINS_TARGET,
            "cv_raw": cv_raw,
            "cv_norm": cv_norm,
            "cv_raw_ref": CV_RAW_REF,
            "cv_norm_ref": CV_NORM_REF,
            "note": "truncated_svd_latent",
        })
        print(
            f"  WNMF={w}: H_raw={h_raw:.3f} H_norm={h_norm:.3f} "
            f"CV_raw={cv_raw:.3f} CV_norm={cv_norm:.3f}"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


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
    return ["fold", "wnmf_dim", "k", "algo", "predictor", "knn_k", "eval_split"]


def _load_done_eval_keys(out_csv: Path, eval_split: str) -> set:
    if not out_csv.is_file():
        return set()
    df = pd.read_csv(out_csv)
    cols = _eval_key_cols()
    if "eval_split" not in df.columns:
        df = df.assign(eval_split="random")
    if any(c not in df.columns for c in cols):
        return set()
    sub = df[df["eval_split"] == eval_split] if "eval_split" in df.columns else df
    return set(map(tuple, sub[cols].itertuples(index=False, name=None)))


def aggregate_preds_cv5_mean(preds_csv: Path, mean_csv: Path) -> pd.DataFrame:
    """5 fold satırlarını (wnmf, k, algo, predictor, knn_k) başına ortala."""
    if not preds_csv.is_file():
        return pd.DataFrame()
    df = pd.read_csv(preds_csv)
    if "fold" not in df.columns or df["fold"].nunique() <= 1:
        return df
    keys = ["wnmf_dim", "k", "algo", "predictor", "knn_k"]
    for c in ("similarity", "protocol", "eval_split"):
        if c in df.columns:
            keys.append(c)
    num_cols = [c for c in PRED_NUM_COLS if c in df.columns]
    agg = df.groupby(keys, as_index=False)[num_cols].mean()
    for c in ("assign_suffix", "n_active_clusters", "cluster_min", "cluster_max", "cluster_std", "singletons"):
        if c in df.columns:
            first = df.groupby(keys, as_index=False)[c].first()
            agg = agg.merge(first, on=keys, how="left")
    agg["fold"] = 0
    agg["cv_folds"] = int(df["fold"].nunique())
    mean_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(mean_csv, index=False)
    return agg


def _assign_fold_for_eval(eval_fold: int, assign_scope: str) -> Optional[int]:
    if assign_scope == ASSIGN_SCOPE_OFFICIAL_CV:
        return int(eval_fold)
    return None


def phase_eval(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    out_csv: Path,
    *,
    eval_fold: int,
    eval_split: str = "official",
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
    skip_existing: bool = False,
) -> pd.DataFrame:
    from wnmf.wnmf_experiment import (
        RANDOM_SEED,
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

    train, test = load_eval_train_test(
        eval_fold, eval_split=eval_split, random_seed=RANDOM_SEED,
    )
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    done = _load_done_eval_keys(out_csv, eval_split) if skip_existing else set()
    key = _eval_key_cols()
    eval_args = Namespace(similarity=SIM, min_common=MIN_COMMON)
    af = _assign_fold_for_eval(eval_fold, assign_scope)

    for w in wnmf_dims:
        for k in ks:
            for algo in algos:
                if not ensure_on_disk(
                    algo, w, k, fold=af, assign_scope=assign_scope,
                ):
                    print(f"SKIP eval {algo} wnmf{w} K={k} fold={af} (atama yok)")
                    continue
                adir = assign_dir(algo, w, k, fold=af, assign_scope=assign_scope)
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

                def _base_row(predictor: str, r: dict, t0: float, *, knn_k: int = 0) -> dict:
                    return {
                        "protocol": f"euc_nogs_none_wnmf_kmref_{eval_split}",
                        "eval_split": eval_split,
                        "fold": int(eval_fold),
                        "wnmf_dim": w,
                        "k": k,
                        "algo": algo,
                        "predictor": predictor,
                        "similarity": SIM,
                        "knn_k": int(knn_k),
                        "mae": r["mae"],
                        "rmse": r["rmse"],
                        "ndcg_at_10": r["ndcg_at_10"],
                        "precision_at_10": r["precision_at_10"],
                        "recall_at_10": r["recall_at_10"],
                        "jaccard_at_10": r.get("jaccard_at_10", float("nan")),
                        "coverage_at_10": r.get("coverage_at_10", float("nan")),
                        "assign_suffix": assign_suffix(
                            w, k, fold=af, assign_scope=assign_scope,
                        ),
                        "assign_scope": assign_scope,
                        "eval_seconds": round(time.time() - t0, 1),
                        **st,
                    }

                block: List[dict] = []

                ek_avg = (eval_fold, w, k, algo, "cluster_avg", 0, eval_split)
                if not (skip_existing and ek_avg in done):
                    t0 = time.time()
                    r = run_cluster_average(
                        train, test, assignments, gray_mask, memberships, n_items, algo,
                        **_cluster_avg_predict_kwargs(eval_args), **nc_avg, **common,
                    )
                    block.append(_base_row("cluster_avg", r, t0))
                    print(
                        f"  fold={eval_fold} {algo} wnmf{w} K={k} cluster_avg "
                        f"MAE={r['mae']:.4f}"
                    )

                for knn_k in (k_min, KNN_FIXED):
                    pred = "cluster_knn_native"
                    ek = (eval_fold, w, k, algo, pred, knn_k, eval_split)
                    if skip_existing and ek in done:
                        continue
                    t0 = time.time()
                    r = run_cluster_knn(
                        train, test, assignments, gray_mask, memberships, n_items, algo,
                        user_features=uf,
                        similarity=SIM,
                        min_common=MIN_COMMON,
                        k_neighbors=knn_k,
                        cluster_knn_backend="native",
                        **nc_knn,
                        **common,
                    )
                    block.append(_base_row(pred, r, t0, knn_k=knn_k))
                    print(
                        f"  fold={eval_fold} {algo} wnmf{w} K={k} {pred} knn={knn_k} "
                        f"MAE={r['mae']:.4f} J@10={r.get('jaccard_at_10', float('nan')):.4f}"
                    )

                if block:
                    _append_csv(block, out_csv, key)

    if out_csv.is_file():
        return pd.read_csv(out_csv)
    return pd.DataFrame()


def _read_wcss_silhouette(adir: Path, assignments: np.ndarray, uf: Optional[np.ndarray]) -> dict:
    out = {"wcss": float("nan"), "silhouette_euclidean": float("nan"), "silhouette_cosine": float("nan")}
    cm = adir / "cluster_metrics.csv"
    if cm.is_file():
        with cm.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f), None)
        if row:
            out["wcss"] = float(row["wcss"]) if row.get("wcss") else float("nan")
            out["silhouette_euclidean"] = float(row.get("silhouette_euclidean") or "nan")
            out["silhouette_cosine"] = float(row.get("silhouette_cosine") or "nan")
            return out
    meta = adir / "db_export_meta.csv"
    if meta.is_file():
        row = pd.read_csv(meta).iloc[0]
        if pd.notna(row.get("wcss")):
            out["wcss"] = float(row["wcss"])
    if uf is not None and uf.ndim == 2 and len(np.unique(assignments)) > 1:
        try:
            out["silhouette_euclidean"] = float(
                silhouette_score(uf, assignments, metric="euclidean")
            )
            out["silhouette_cosine"] = float(
                silhouette_score(normalize(uf), assignments, metric="euclidean")
            )
        except Exception:
            pass
    return out


def phase_assignment_metrics(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    out_csv: Path,
    *,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
) -> pd.DataFrame:
    from wnmf.wnmf_experiment import load_assignment, load_user_features

    rows: List[dict] = []
    for fold in _assign_folds(assign_scope):
        for w in wnmf_dims:
            for k in ks:
                for algo in algos:
                    if not ensure_on_disk(
                        algo, w, k, fold=fold, assign_scope=assign_scope,
                    ):
                        continue
                    adir = assign_dir(algo, w, k, fold=fold, assign_scope=assign_scope)
                    a, _ = load_assignment(str(adir))
                    try:
                        uf = load_user_features(str(adir), len(a))
                    except Exception:
                        uf = None
                    met = _read_wcss_silhouette(adir, a, uf)
                    st = cluster_stats(a)
                    rows.append({
                        "assign_fold": fold if fold is not None else 0,
                        "wnmf_dim": w,
                        "k": k,
                        "algo": algo,
                        "assign_suffix": assign_suffix(
                            w, k, fold=fold, assign_scope=assign_scope,
                        ),
                        **met,
                        **st,
                    })
    key = ["assign_fold", "wnmf_dim", "k", "algo"]
    return _append_csv(rows, out_csv, key)


def phase_cluster_compare(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    out_csv: Path,
    *,
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
) -> pd.DataFrame:
    from compare_cluster_structure import (
        _common_mask,
        _load_centroids,
        _load_labels_and_gray,
        _structure_stats,
        mean_centroid_distance,
    )
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    rows: List[dict] = []
    for fold in _assign_folds(assign_scope):
        for w in wnmf_dims:
            for k in ks:
                loaded: Dict[str, tuple] = {}
                for algo in algos:
                    if not ensure_on_disk(
                        algo, w, k, fold=fold, assign_scope=assign_scope,
                    ):
                        continue
                    fp = assign_dir(algo, w, k, fold=fold, assign_scope=assign_scope) / "assignments.npy"
                    la, g = _load_labels_and_gray(str(fp))
                    c = _load_centroids(str(fp), k)
                    loaded[algo] = (la, g, c, _structure_stats(la, k))

                for a1, a2 in combinations(sorted(loaded), 2):
                    la, ga, ca, sa = loaded[a1]
                    lb, gb, cb, sb = loaded[a2]
                    mask = _common_mask(ga, gb, len(la))
                    n_ok = int(mask.sum())
                    row = {
                        "assign_fold": fold if fold is not None else 0,
                        "wnmf_dim": w,
                        "k": k,
                        "algo_a": a1,
                        "algo_b": a2,
                        "assign_suffix": assign_suffix(
                            w, k, fold=fold, assign_scope=assign_scope,
                        ),
                        "n_users": n_ok,
                        "label_agreement": float("nan"),
                        "ari": float("nan"),
                        "nmi": float("nan"),
                        "centroid_l2_mean": float("nan"),
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

    key = ["assign_fold", "wnmf_dim", "k", "algo_a", "algo_b"]
    return _append_csv(rows, out_csv, key)


def phase_compare_fold_assignments(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    out_csv: Path,
) -> pd.DataFrame:
    """Aynı (algo, wnmf, k) için official fold'lar arası küme etiketi benzerliği."""
    from compare_cluster_structure import _load_labels_and_gray
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    rows: List[dict] = []
    for w in wnmf_dims:
        for k in ks:
            for algo in algos:
                by_fold: Dict[int, np.ndarray] = {}
                for fold in CV_FOLDS:
                    fp = assign_dir(
                        algo, w, k, fold=fold, assign_scope=ASSIGN_SCOPE_OFFICIAL_CV,
                    ) / "assignments.npy"
                    if not fp.is_file():
                        continue
                    la, _ = _load_labels_and_gray(str(fp))
                    by_fold[int(fold)] = la
                if len(by_fold) < 2:
                    continue
                for f1, f2 in combinations(sorted(by_fold), 2):
                    la, lb = by_fold[f1], by_fold[f2]
                    n = min(len(la), len(lb))
                    if n < 2:
                        continue
                    la, lb = la[:n], lb[:n]
                    rows.append({
                        "wnmf_dim": w,
                        "k": k,
                        "algo": algo,
                        "fold_a": f1,
                        "fold_b": f2,
                        "n_users": n,
                        "label_agreement": float((la == lb).mean()),
                        "ari": float(adjusted_rand_score(la, lb)),
                        "nmi": float(
                            normalized_mutual_info_score(la, lb, average_method="arithmetic")
                        ),
                    })
    key = ["wnmf_dim", "k", "algo", "fold_a", "fold_b"]
    df = _append_csv(rows, out_csv, key)
    if not df.empty:
        summ = (
            df.groupby(["wnmf_dim", "k", "algo"], as_index=False)["ari"]
            .mean()
            .rename(columns={"ari": "mean_ari_across_folds"})
        )
        print("\nFold'lar arası ortalama ARI (algo başına, ilk 10):")
        print(summ.head(10).to_string(index=False))
    return df


def _predict_native_rows(
    train: np.ndarray,
    test: np.ndarray,
    algo: str,
    assign_dir_path: str,
    *,
    knn: int,
) -> np.ndarray:
    from wnmf.wnmf_utils import load_assignment, load_memberships
    from wnmf.wnmf_experiment import run_cluster_knn

    assignments, gray_mask = load_assignment(assign_dir_path)
    memberships = load_memberships(assign_dir_path)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    row = run_cluster_knn(
        train,
        test,
        assignments,
        gray_mask,
        memberships,
        n_items,
        algo,
        similarity=SIM,
        min_common=MIN_COMMON,
        k_neighbors=knn,
        cluster_knn_backend="native",
        assign_dir=assign_dir_path,
        return_eval_rows=True,
    )
    return np.asarray(row["eval_rows"], dtype=np.float64)


def phase_stats(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    out_csv: Path,
    *,
    eval_fold: int,
    eval_split: str = "official",
    assign_scope: str = ASSIGN_SCOPE_OFFICIAL_CV,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 95.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    from mealpy.paired_bootstrap_ci import (
        align_eval_rows,
        paired_bootstrap_delta,
        paired_wilcoxon,
        resolve_assign_dir,
    )
    from wnmf.wnmf_experiment import RANDOM_SEED

    train, test = load_eval_train_test(
        eval_fold, eval_split=eval_split, random_seed=RANDOM_SEED,
    )
    assign_root = str(REPO / "mealpy" / "results" / "assignments")
    others = [a for a in algos if a != BOOT_REFERENCE]
    rows: List[dict] = []

    af = _assign_fold_for_eval(eval_fold, assign_scope)
    for w in wnmf_dims:
        for k in ks:
            if not ensure_on_disk(
                BOOT_REFERENCE, w, k, fold=af, assign_scope=assign_scope,
            ):
                continue
            suf = assign_suffix(w, k, fold=af, assign_scope=assign_scope)
            ref_dir = resolve_assign_dir(assign_root, BOOT_REFERENCE, k, suf)
            ref_assign = np.load(Path(ref_dir) / "assignments.npy")
            k_min_ref = max(1, min(Counter(ref_assign.astype(int).tolist()).values()))

            pred = "cluster_knn_native"
            knn_k = k_min_ref
            cache: Dict[str, np.ndarray] = {}
            for algo in [BOOT_REFERENCE] + others:
                if not ensure_on_disk(
                    algo, w, k, fold=af, assign_scope=assign_scope,
                ):
                    continue
                adir = resolve_assign_dir(assign_root, algo, k, suf)
                cache[algo] = _predict_native_rows(train, test, algo, adir, knn=knn_k)

            for algo_b in others:
                if BOOT_REFERENCE not in cache or algo_b not in cache:
                    continue
                true, pa, pb = align_eval_rows(cache[BOOT_REFERENCE], cache[algo_b])
                for metric in ("mae", "rmse"):
                    stats = paired_bootstrap_delta(
                        true, pa, pb, metric, n_boot=n_boot, seed=seed, ci=ci,
                    )
                    wx = paired_wilcoxon(
                        true, pa, pb, metric, alpha=alpha, alternative="two-sided",
                    )
                    rows.append({
                        "fold": eval_fold,
                        "wnmf_dim": w,
                        "k": k,
                        "reference": BOOT_REFERENCE,
                        "algo_b": algo_b,
                        "predictor": pred,
                        "knn_k": knn_k if pred != "cluster_avg" else 0,
                        "metric": metric,
                        "delta": stats["delta"],
                        "ci_lo": stats["ci_lo"],
                        "ci_hi": stats["ci_hi"],
                        "p_ref_better": stats["p_a_better"],
                        "wilcoxon_p": wx["p_value"],
                        "wilcoxon_sig": wx["significant"],
                        "n_pairs": int(stats["n_pairs"]),
                        "assign_suffix": suf,
                    })

    key = ["fold", "wnmf_dim", "k", "reference", "algo_b", "predictor", "knn_k", "metric"]
    return _append_csv(rows, out_csv, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=[
            "status", "assign", "sync-db", "hopkins-cv", "eval",
            "assignment-metrics", "cluster-compare", "compare-folds",
            "stats", "plot", "all",
        ],
        default="status",
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--wnmf", type=int, nargs="+", default=WNMF_DIMS)
    ap.add_argument("--k", type=int, nargs="+", default=K_LIST)
    ap.add_argument("--algo", nargs="+", default=ALGOS)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument(
        "--assign-scope",
        choices=[ASSIGN_SCOPE_OFFICIAL_CV, ASSIGN_SCOPE_FULL],
        default=ASSIGN_SCOPE_OFFICIAL_CV,
        help="official_cv: her fold train-only atama (uN.base); full: tek atama u.data",
    )
    ap.add_argument(
        "--eval-split",
        choices=["official", "random"],
        default="official",
        help="Eval test bölmesi: official=uN.test; random=u.data KFold",
    )
    ap.add_argument(
        "--no-cv5", action="store_true",
        help="Tek fold eval (varsayılan: 5-fold; atama da 5 fold)",
    )
    ap.add_argument("--fold", type=int, default=None, help="Tek eval fold (1–5), --no-cv5 ile")
    ap.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Assign/eval fold alt kümesi (örn. 5 veya 1 3 5); yoksa 1–5",
    )
    args = ap.parse_args()

    wnmf_dims = args.wnmf
    ks = args.k
    algos = args.algo

    if args.no_cv5:
        eval_folds = [args.fold if args.fold is not None else 1]
        use_cv5 = False
    elif args.folds:
        eval_folds = list(args.folds)
        use_cv5 = len(eval_folds) > 1
    elif args.fold is not None:
        eval_folds = [args.fold]
        use_cv5 = False
    else:
        eval_folds = list(CV_FOLDS)
        use_cv5 = True

    if args.phase in ("status", "all"):
        print("\n=== STATUS ===")
        phase_status(wnmf_dims, ks, algos, assign_scope=args.assign_scope)

    if args.phase in ("assign", "all"):
        print(f"\n=== ASSIGN (scope={args.assign_scope}) ===")
        assign_folds = (
            eval_folds if args.assign_scope == ASSIGN_SCOPE_OFFICIAL_CV else None
        )
        rc = phase_assign(
            args.jobs, wnmf_dims, ks, algos, args.skip_existing,
            assign_scope=args.assign_scope,
            assign_folds=assign_folds,
        )
        if rc != 0 and args.phase == "assign":
            sys.exit(rc)

    if args.phase in ("sync-db", "all"):
        print("\n=== SYNC DB ===")
        phase_sync_db(wnmf_dims, ks, algos, assign_scope=args.assign_scope)

    if args.phase in ("hopkins-cv", "all"):
        print("\n=== HOPKINS / CV ===")
        phase_hopkins_cv(wnmf_dims, OUT_HOPKINS)

    if args.phase in ("eval", "all"):
        print(
            f"\n=== EVAL (split={args.eval_split}, folds={eval_folds}, cv5={use_cv5}) ==="
        )
        for eval_fold in eval_folds:
            print(f"\n--- eval fold {eval_fold} ({args.eval_split}) ---")
            phase_eval(
                wnmf_dims, ks, algos, OUT_PREDS,
                eval_fold=eval_fold,
                eval_split=args.eval_split,
                assign_scope=args.assign_scope,
                skip_existing=args.skip_existing,
            )
        df = pd.read_csv(OUT_PREDS) if OUT_PREDS.is_file() else pd.DataFrame()
        if use_cv5 and not df.empty:
            df_mean = aggregate_preds_cv5_mean(OUT_PREDS, OUT_PREDS_CV5_MEAN)
            print(f"CV5 ortalama -> {OUT_PREDS_CV5_MEAN} ({len(df_mean)} rows)")
        print(f"Preds (fold başına) -> {OUT_PREDS} ({len(df)} rows)")
        if use_cv5 and "fold" in df.columns:
            print(df.groupby("fold").size().to_string())

    if args.phase in ("assignment-metrics", "all"):
        print("\n=== ASSIGNMENT METRICS ===")
        df = phase_assignment_metrics(
            wnmf_dims, ks, algos, OUT_ASSIGN, assign_scope=args.assign_scope,
        )
        print(f"Metrics -> {OUT_ASSIGN} ({len(df)} rows)")

    if args.phase in ("cluster-compare", "all"):
        print("\n=== CLUSTER COMPARE (algo çiftleri) ===")
        df = phase_cluster_compare(
            wnmf_dims, ks, algos, OUT_CLUSTER, assign_scope=args.assign_scope,
        )
        print(f"Pairs -> {OUT_CLUSTER} ({len(df)} rows)")

    if args.phase in ("compare-folds", "all"):
        if args.assign_scope != ASSIGN_SCOPE_OFFICIAL_CV:
            print("compare-folds yalnızca --assign-scope official_cv ile anlamlı")
        else:
            print("\n=== FOLD ASSIGNMENT COMPARE ===")
            df = phase_compare_fold_assignments(wnmf_dims, ks, algos, OUT_FOLD_ASSIGN)
            print(f"Fold pairs -> {OUT_FOLD_ASSIGN} ({len(df)} rows)")

    if args.phase in ("stats", "all"):
        print(
            f"\n=== BOOTSTRAP / WILCOXON "
            f"(split={args.eval_split}, folds={eval_folds}) ==="
        )
        for eval_fold in eval_folds:
            print(f"\n--- stats fold {eval_fold} ---")
            phase_stats(
                wnmf_dims, ks, algos, OUT_STATS,
                eval_fold=eval_fold,
                eval_split=args.eval_split,
                assign_scope=args.assign_scope,
                n_boot=args.n_bootstrap,
            )
        df = pd.read_csv(OUT_STATS) if OUT_STATS.is_file() else pd.DataFrame()
        print(f"Stats -> {OUT_STATS} ({len(df)} rows)")

    if args.phase in ("plot", "all"):
        print("\n=== PLOT ===")
        plot_script = REPO / "experiments" / "plot_euc_nogs_none_wnmf_k_grid.py"
        plot_args = [sys.executable, str(plot_script)]
        if use_cv5:
            plot_args.append("--cv5-mean")
        rc = subprocess.run(plot_args, cwd=str(REPO)).returncode
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
