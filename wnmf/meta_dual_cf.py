"""
MetaDual-CF: B0 kalibrasyon + meta küme sıralama sinyali.

Amaç
----
- B0 (KMeans) MAE/NDCG tabanını korur.
- Meta atama latent uzayda optimize edildiği için, meta küme-içi item sapması
  (z-score) NDCG'de ayrışma sağlar.
- Kullanıcının meta centroid'e yakınlığı (user_features × best_sol) ile meta
  ağırlığı artar → meta algoritmanın gücü merkezi kullanıcılarda görünür.

Tahmin (test user u, item i):
    pred = anchor(u) + beta * blend_rank(u, i)

    anchor(u)  = B0 kullanıcı ortalaması (MAE stabil)
    blend_rank   = w(u) * z_meta(u,i) + (1-w(u)) * z_b0(u,i)
    z_meta       = (R_meta[c_m,i] - mu_meta[c_m]) / std_meta[c_m]
    w(u)         = sigmoid(-dist(U[u], centroid_meta[c_m]) / tau)

Kullanım:
    python wnmf/meta_dual_cf.py --k 10 --meta-algo B1_HHO --fold 1
    python wnmf/meta_dual_cf.py --k 10 --meta-algo B1_HHO --fusion-ab
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple, List

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "wnmf"))

from wnmf.wnmf_experiment import (  # noqa: E402
    _compute_metrics,
    _compute_topn_jaccard,
    _compute_topn_metrics,
)
from wnmf.wnmf_utils import load_ratings_100k_all  # noqa: E402

ASSIGN_ROOT = os.path.join(REPO, "mealpy", "results", "assignments", "ml100k")
SUFFIX_TMPL = "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf{W}_k{K}"

RANK_ZSCORE = "zscore"
RANK_RAW_DELTA = "raw_delta"
RANK_MODES = (RANK_ZSCORE, RANK_RAW_DELTA)


def _assign_suffix(k: int, wnmf_dim: int = 20) -> str:
    return SUFFIX_TMPL.format(W=int(wnmf_dim), K=k)


def _assign_dir(
    algo: str, k: int, kmref_meta: bool = False, wnmf_dim: int = 20,
) -> str:
    """kmref_meta=True yalnizca eski --kmeans-refine-overwrite kosulari icin."""
    suf = _assign_suffix(k, wnmf_dim)
    if algo != "B0_KMEANS" and kmref_meta:
        suf += "_kmref"
    return os.path.join(ASSIGN_ROOT, f"{algo}{suf}")


def _load_assignments(path: str) -> np.ndarray:
    return np.load(os.path.join(path, "assignments.npy"))


def _cluster_item_stats(
    train: np.ndarray,
    assignments: np.ndarray,
    n_items: int,
    global_mean: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """cluster_item_means, cluster_item_counts, cluster_means, cluster_stds."""
    n_clusters = int(assignments.max()) + 1
    sums = np.zeros((n_clusters, n_items), dtype=np.float64)
    counts = np.zeros((n_clusters, n_items), dtype=np.int32)
    for u, i, r in train:
        cid = int(assignments[int(u)])
        sums[cid, int(i)] += float(r)
        counts[cid, int(i)] += 1

    means = np.full((n_clusters, n_items), global_mean, dtype=np.float64)
    mask = counts > 0
    means[mask] = sums[mask] / counts[mask]

    cluster_means = np.zeros(n_clusters, dtype=np.float64)
    cluster_stds = np.ones(n_clusters, dtype=np.float64)
    for cid in range(n_clusters):
        users = np.where(assignments == cid)[0]
        if users.size == 0:
            cluster_means[cid] = global_mean
            continue
        ratings = []
        for row in train:
            if int(row[0]) in users:
                ratings.append(float(row[2]))
        if ratings:
            cluster_means[cid] = float(np.mean(ratings))
            std = float(np.std(ratings))
            cluster_stds[cid] = std if std > 1e-6 else 1.0

    return means, counts, cluster_means, cluster_stds


def _user_means(train: np.ndarray, n_users: int, global_mean: float) -> np.ndarray:
    sums = np.zeros(n_users, dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.int32)
    for u, _, r in train:
        ui = int(u)
        sums[ui] += float(r)
        counts[ui] += 1
    out = np.full(n_users, global_mean, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def _load_centroids(assign_dir: str, k: int, dim: int) -> np.ndarray:
    raw = np.load(os.path.join(assign_dir, "best_sol.npy"))
    return raw.reshape(k, dim)


def _meta_confidence(
    u: int,
    cid: int,
    U: np.ndarray,
    centroids: np.ndarray,
    tau: float,
) -> float:
    d = float(np.linalg.norm(U[u] - centroids[cid]))
    return float(1.0 / (1.0 + np.exp(d / max(tau, 1e-6) - 1.0)))


def _item_rank_signal(
    cid: int,
    i: int,
    item_means: np.ndarray,
    item_counts: np.ndarray,
    cluster_means: np.ndarray,
    cluster_stds: np.ndarray,
    *,
    rank_mode: str = RANK_ZSCORE,
) -> float:
    """Küme-içi item sinyali: zscore veya normalize edilmemiş sapma."""
    if item_counts[cid, i] <= 0:
        return 0.0
    delta = float(item_means[cid, i] - cluster_means[cid])
    if rank_mode == RANK_RAW_DELTA:
        return delta
    if rank_mode == RANK_ZSCORE:
        return float(delta / cluster_stds[cid])
    raise ValueError(f"rank_mode must be one of {RANK_MODES}, got {rank_mode!r}")


def _default_tau(U: np.ndarray, assign_meta: np.ndarray, centroids_meta: np.ndarray) -> float:
    dists = []
    for u in range(min(len(assign_meta), U.shape[0])):
        cid = int(assign_meta[u])
        dists.append(float(np.linalg.norm(U[u] - centroids_meta[cid])))
    return float(np.median(dists)) if dists else 1.0


def _build_dual_context(
    train: np.ndarray,
    assign_b0: np.ndarray,
    assign_meta: np.ndarray,
    U: np.ndarray,
    centroids_meta: np.ndarray,
    n_users: Optional[int] = None,
    n_items: Optional[int] = None,
) -> dict:
    n_users = n_users or int(train[:, 0].max()) + 1
    n_items = n_items or int(train[:, 1].max()) + 1
    global_mean = float(train[:, 2].mean())
    meta_means, meta_counts, meta_cmean, meta_cstd = _cluster_item_stats(
        train, assign_meta, n_items, global_mean,
    )
    b0_means, b0_counts, b0_cmean, b0_cstd = _cluster_item_stats(
        train, assign_b0, n_items, global_mean,
    )
    anchor = _user_means(train, n_users, global_mean)
    tau_base = _default_tau(U, assign_meta, centroids_meta)
    return {
        "assign_b0": assign_b0,
        "assign_meta": assign_meta,
        "U": U,
        "centroids_meta": centroids_meta,
        "meta_means": meta_means,
        "meta_counts": meta_counts,
        "meta_cmean": meta_cmean,
        "meta_cstd": meta_cstd,
        "b0_means": b0_means,
        "b0_counts": b0_counts,
        "b0_cmean": b0_cmean,
        "b0_cstd": b0_cstd,
        "anchor": anchor,
        "global_mean": global_mean,
        "tau_base": tau_base,
    }


def _predict_rows(
    rows: np.ndarray,
    ctx: dict,
    *,
    beta: float,
    tau: float,
    rank_mode: str = RANK_ZSCORE,
) -> Tuple[list, list, float]:
    assign_b0 = ctx["assign_b0"]
    assign_meta = ctx["assign_meta"]
    U = ctx["U"]
    centroids_meta = ctx["centroids_meta"]
    anchor = ctx["anchor"]

    true_vals, pred_vals = [], []
    w_meta_sum = 0.0
    for u, i, r in rows:
        u, i, r = int(u), int(i), float(r)
        cid_m = int(assign_meta[u])
        cid_b = int(assign_b0[u])
        sig_m = _item_rank_signal(
            cid_m, i, ctx["meta_means"], ctx["meta_counts"],
            ctx["meta_cmean"], ctx["meta_cstd"], rank_mode=rank_mode,
        )
        sig_b = _item_rank_signal(
            cid_b, i, ctx["b0_means"], ctx["b0_counts"],
            ctx["b0_cmean"], ctx["b0_cstd"], rank_mode=rank_mode,
        )
        w = _meta_confidence(u, cid_m, U, centroids_meta, tau)
        w_meta_sum += w
        rank = w * sig_m + (1.0 - w) * sig_b
        pred = float(np.clip(anchor[u] + beta * rank, 1.0, 5.0))
        true_vals.append(r)
        pred_vals.append(pred)
    return true_vals, pred_vals, w_meta_sum / max(len(rows), 1)


def _eval_rows(
    rows: np.ndarray,
    ctx: dict,
    train: np.ndarray,
    *,
    beta: float,
    tau: float,
    rank_mode: str = RANK_ZSCORE,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
) -> dict:
    true_vals, pred_vals, w_mean = _predict_rows(
        rows, ctx, beta=beta, tau=tau, rank_mode=rank_mode,
    )
    eval_rows = [
        (int(rows[j, 0]), int(rows[j, 1]), float(rows[j, 2]), float(pred_vals[j]))
        for j in range(len(rows))
    ]
    eval_rows_arr = np.array(eval_rows, dtype=np.float32)
    mae, rmse = _compute_metrics(true_vals, pred_vals)
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train,
        assignments=ctx["assign_meta"],
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "precision_at_10": precision,
        "recall_at_10": recall,
        "f1_at_10": f1,
        "ndcg_at_10": ndcg,
        "mean_meta_weight": w_mean,
        "beta": beta,
        "tau": tau,
        "rank_mode": rank_mode,
    }


def tune_meta_dual(
    train: np.ndarray,
    ctx: dict,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    beta_grid: Optional[list] = None,
    tau_scale_grid: Optional[list] = None,
    optimize: str = "ndcg",
    rank_mode: str = RANK_ZSCORE,
) -> Tuple[float, float, dict]:
    """Train icinden val ornegi ile algo-ozel beta/tau sec."""
    rng = np.random.default_rng(seed)
    n = len(train)
    idx = rng.permutation(n)
    n_val = max(500, int(n * val_frac))
    val_rows = train[idx[:n_val]]
    fit_rows = train[idx[n_val:]]
    n_users = int(train[:, 0].max()) + 1
    n_items = int(train[:, 1].max()) + 1

    fit_ctx = _build_dual_context(
        fit_rows, ctx["assign_b0"], ctx["assign_meta"],
        ctx["U"], ctx["centroids_meta"],
        n_users=n_users, n_items=n_items,
    )
    if beta_grid is None:
        betas = (
            [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
            if rank_mode == RANK_RAW_DELTA
            else [0.4, 0.6, 0.85, 1.0, 1.2]
        )
    else:
        betas = beta_grid
    tau_scales = tau_scale_grid or [0.5, 0.75, 1.0, 1.25, 1.5]
    tau_base = fit_ctx["tau_base"]

    best = None
    best_key = None
    best_beta, best_tau = betas[0], tau_base
    for beta in betas:
        for ts in tau_scales:
            tau = tau_base * ts
            m = _eval_rows(
                val_rows, fit_ctx, fit_rows, beta=beta, tau=tau, rank_mode=rank_mode,
            )
            key = m["ndcg_at_10"] if optimize == "ndcg" else -m["mae"]
            if best is None or key > best_key:
                best_key = key
                best = m
                best_beta, best_tau = beta, tau
    assert best is not None
    return best_beta, best_tau, best


def predict_meta_dual(
    train: np.ndarray,
    test: np.ndarray,
    assign_b0: np.ndarray,
    assign_meta: np.ndarray,
    U: np.ndarray,
    centroids_meta: np.ndarray,
    *,
    beta: float = 0.85,
    tau: Optional[float] = None,
    rank_mode: str = RANK_ZSCORE,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
    tune: bool = False,
) -> dict:
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    ctx = _build_dual_context(
        train, assign_b0, assign_meta, U, centroids_meta,
        n_users=n_users, n_items=n_items,
    )
    if tau is None:
        tau = ctx["tau_base"]
    tune_info = {}
    if tune:
        beta, tau, tune_info = tune_meta_dual(
            train, ctx, optimize="ndcg", rank_mode=rank_mode,
        )

    out = _eval_rows(
        test, ctx, train, beta=beta, tau=tau, rank_mode=rank_mode,
        top_n=top_n, relevance_threshold=relevance_threshold,
    )
    jaccard_rows = [
        (int(test[j, 0]), int(test[j, 1]), float(test[j, 2]), 0.0)
        for j in range(len(test))
    ]
    _, pred_vals, _ = _predict_rows(
        test, ctx, beta=beta, tau=tau, rank_mode=rank_mode,
    )
    eval_rows_arr = np.array(
        [
            (int(test[j, 0]), int(test[j, 1]), float(test[j, 2]), float(pred_vals[j]))
            for j in range(len(test))
        ],
        dtype=np.float32,
    )
    out["jaccard_at_10"] = _compute_topn_jaccard(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train,
        assignments=assign_meta,
    )
    out["tau_base"] = ctx["tau_base"]
    if tune_info:
        out["tune_val_ndcg"] = tune_info.get("ndcg_at_10")
    return out


META_ALGOS_DEFAULT = [
    "HA_AVOAHGS", "B1_HHO", "IWO_HHO", "B_AVOA",
    "H4_MFO+HHO", "H9_QSA+CDO", "LIT_PSO", "LIT_GWO",
]


def _baseline_cluster_avg_user(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
) -> dict:
    """Basit küme×item ortalaması + user_mean fallback (referans)."""
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    global_mean = float(train[:, 2].mean())
    means, counts, _, _ = _cluster_item_stats(train, assignments, n_items, global_mean)
    user_m = _user_means(train, n_users, global_mean)

    true_vals, pred_vals = [], []
    eval_rows = []
    for u, i, r in test:
        u, i, r = int(u), int(i), float(r)
        cid = int(assignments[u])
        if counts[cid, i] > 0:
            pred = float(means[cid, i])
        else:
            pred = float(user_m[u])
        pred = float(np.clip(pred, 1.0, 5.0))
        true_vals.append(r)
        pred_vals.append(pred)
        eval_rows.append((u, i, r, pred))

    eval_rows_arr = np.array(eval_rows, dtype=np.float32)
    mae, rmse = _compute_metrics(true_vals, pred_vals)
    p, rec, f1, ndcg = _compute_topn_metrics(
        eval_rows_arr, top_n=top_n, threshold=relevance_threshold,
        train=train, assignments=assignments,
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "precision_at_10": p,
        "recall_at_10": rec,
        "f1_at_10": f1,
        "ndcg_at_10": ndcg,
    }


def _fmt_row(label: str, row: dict) -> str:
    return (
        f"{label:<26} {row['mae']:>7.4f} {row['rmse']:>7.4f} "
        f"{row['precision_at_10']:>7.4f} {row['recall_at_10']:>7.4f} "
        f"{row['ndcg_at_10']:>7.4f}"
    )


def _delta_row(label: str, row: dict, ref: dict) -> str:
    return (
        f"{label:<26} "
        f"{row['mae'] - ref['mae']:>+7.4f} "
        f"{row['rmse'] - ref['rmse']:>+7.4f} "
        f"{row['precision_at_10'] - ref['precision_at_10']:>+7.4f} "
        f"{row['recall_at_10'] - ref['recall_at_10']:>+7.4f} "
        f"{row['ndcg_at_10'] - ref['ndcg_at_10']:>+7.4f}"
    )


def run_comparison(
    k: int,
    meta_algos: list,
    base_algo: str = "B0_KMEANS",
    fold: int = 1,
    beta: float = 0.85,
    tau: Optional[float] = None,
    kmref: bool = False,
    tune: bool = False,
    wnmf_dim: int = 20,
) -> dict:
    data_path = os.path.join(REPO, "data", "ml-100k", "u.data")
    train, test = load_ratings_100k_all(data_path, random_seed=42, fold=fold)

    b0_dir = _assign_dir(base_algo, k, kmref_meta=False, wnmf_dim=wnmf_dim)
    assign_b0 = _load_assignments(b0_dir)
    b0 = _baseline_cluster_avg_user(train, test, assign_b0)

    results = {"B0 cluster_avg": b0, "meta": {}}
    for meta_algo in meta_algos:
        meta_dir = _assign_dir(meta_algo, k, kmref_meta=kmref, wnmf_dim=wnmf_dim)
        assign_meta = _load_assignments(meta_dir)
        U = np.load(os.path.join(meta_dir, "user_features.npy"))
        centroids = _load_centroids(meta_dir, k, U.shape[1])
        meta_only = _baseline_cluster_avg_user(train, test, assign_meta)
        dual = predict_meta_dual(
            train, test, assign_b0, assign_meta, U, centroids,
            beta=beta, tau=tau, tune=tune,
        )
        results["meta"][meta_algo] = {
            "cluster_avg": meta_only,
            "meta_dual": dual,
        }
    return results


def run_fusion_ab(
    k: int,
    meta_algos: list,
    base_algo: str = "B0_KMEANS",
    fold: int = 1,
    beta: float = 0.85,
    tau: Optional[float] = None,
    tune: bool = False,
    wnmf_dim: int = 20,
) -> dict:
    """
    kmref (dis birlesim) vs MetaDual karsilastirmasi.

    Her meta algo icin:
      - cluster_avg: ham meta atama
      - kmref_cluster_avg: _kmref atama + cluster_avg (Lloyd sonrasi dis tahmin)
      - meta_dual: z-score rank (meta atama + B0 dual)
      - meta_dual_raw: z olmadan (R_ci - mu_c) sapma
    """
    data_path = os.path.join(REPO, "data", "ml-100k", "u.data")
    train, test = load_ratings_100k_all(data_path, random_seed=42, fold=fold)

    b0_dir = _assign_dir(base_algo, k, kmref_meta=False, wnmf_dim=wnmf_dim)
    assign_b0 = _load_assignments(b0_dir)
    b0 = _baseline_cluster_avg_user(train, test, assign_b0)

    out: dict = {"B0 cluster_avg": b0, "meta": {}}
    for meta_algo in meta_algos:
        meta_dir = _assign_dir(meta_algo, k, kmref_meta=False, wnmf_dim=wnmf_dim)
        kmref_dir = _assign_dir(meta_algo, k, kmref_meta=True, wnmf_dim=wnmf_dim)

        assign_meta = _load_assignments(meta_dir)
        U = np.load(os.path.join(meta_dir, "user_features.npy"))
        centroids = _load_centroids(meta_dir, k, U.shape[1])

        cluster_avg = _baseline_cluster_avg_user(train, test, assign_meta)
        if os.path.isdir(kmref_dir) and os.path.isfile(
            os.path.join(kmref_dir, "assignments.npy")
        ):
            kmref_avg = _baseline_cluster_avg_user(
                train, test, _load_assignments(kmref_dir),
            )
        else:
            kmref_avg = None

        out["meta"][meta_algo] = {
            "cluster_avg": cluster_avg,
            "kmref_cluster_avg": kmref_avg,
            "meta_dual": predict_meta_dual(
                train, test, assign_b0, assign_meta, U, centroids,
                beta=beta, tau=tau, rank_mode=RANK_ZSCORE, tune=tune,
            ),
            "meta_dual_raw": predict_meta_dual(
                train, test, assign_b0, assign_meta, U, centroids,
                beta=beta, tau=tau, rank_mode=RANK_RAW_DELTA, tune=tune,
            ),
        }
    return out


def _results_to_rows(k: int, results: dict, meta_algos: list) -> List[dict]:
    rows = []
    b0 = results["B0 cluster_avg"]
    rows.append({
        "K": k, "algo": "B0_KMEANS", "method": "cluster_avg",
        "mae": b0["mae"], "rmse": b0["rmse"],
        "precision_at_10": b0["precision_at_10"],
        "recall_at_10": b0["recall_at_10"],
        "ndcg_at_10": b0["ndcg_at_10"],
        "beta": None, "tau": None,
    })
    for algo in meta_algos:
        m = results["meta"][algo]
        rows.append({
            "K": k, "algo": algo, "method": "cluster_avg",
            "mae": m["cluster_avg"]["mae"], "rmse": m["cluster_avg"]["rmse"],
            "precision_at_10": m["cluster_avg"]["precision_at_10"],
            "recall_at_10": m["cluster_avg"]["recall_at_10"],
            "ndcg_at_10": m["cluster_avg"]["ndcg_at_10"],
            "beta": None, "tau": None,
        })
        d = m["meta_dual"]
        rows.append({
            "K": k, "algo": algo, "method": "meta_dual",
            "mae": d["mae"], "rmse": d["rmse"],
            "precision_at_10": d["precision_at_10"],
            "recall_at_10": d["recall_at_10"],
            "ndcg_at_10": d["ndcg_at_10"],
            "beta": d.get("beta"), "tau": d.get("tau"),
        })
    return rows


def _print_k_sweep_summary(all_rows: List[dict]) -> None:
    """K basina en iyi MetaDual NDCG / MAE ozeti."""
    import pandas as pd

    df = pd.DataFrame(all_rows)
    dual = df[df["method"] == "meta_dual"]
    b0 = df[(df["algo"] == "B0_KMEANS") & (df["method"] == "cluster_avg")]
    print("\n" + "=" * 72)
    print("K SWEEP OZET")
    print("=" * 72)
    print(f"{'K':>4} {'B0 MAE':>8} {'B0 NDCG':>8} "
          f"{'best dual':>14} {'MAE':>8} {'NDCG':>8} {'dNDCG':>8}")
    print("-" * 72)
    for k in sorted(df["K"].unique()):
        b0r = b0[b0["K"] == k].iloc[0]
        dk = dual[dual["K"] == k]
        best = dk.loc[dk["ndcg_at_10"].idxmax()]
        print(
            f"{k:>4} {b0r['mae']:>8.4f} {b0r['ndcg_at_10']:>8.4f} "
            f"{best['algo']:>14} {best['mae']:>8.4f} {best['ndcg_at_10']:>8.4f} "
            f"{best['ndcg_at_10'] - b0r['ndcg_at_10']:>+8.4f}"
        )


def _print_comparison(k: int, results: dict, meta_algos: list, tuned: bool = False) -> None:
    b0 = results["B0 cluster_avg"]
    hdr = f"{'Yontem':<26} {'MAE':>7} {'RMSE':>7} {'P@10':>7} {'R@10':>7} {'NDCG':>7}"
    if tuned:
        hdr += f" {'beta':>5} {'tau':>6}"
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f"K={k}  — mutlak metrikler")
    print(hdr)
    print(sep)
    print(_fmt_row("B0 cluster_avg", b0))
    for algo in meta_algos:
        m = results["meta"][algo]
        print(_fmt_row(f"{algo} cluster_avg", m["cluster_avg"]))
        row = m["meta_dual"]
        line = _fmt_row(f"MetaDual-CF ({algo})", row)
        if tuned:
            line += f" {row.get('beta', 0):>5.2f} {row.get('tau', 0):>6.3f}"
        print(line)

    dual_ndcg = [results["meta"][a]["meta_dual"]["ndcg_at_10"] for a in meta_algos]
    dual_mae = [results["meta"][a]["meta_dual"]["mae"] for a in meta_algos]
    print(
        f"\n  MetaDual spread: MAE {min(dual_mae):.4f}-{max(dual_mae):.4f} "
        f"(range {max(dual_mae)-min(dual_mae):.4f}), "
        f"NDCG {min(dual_ndcg):.4f}-{max(dual_ndcg):.4f} "
        f"(range {max(dual_ndcg)-min(dual_ndcg):.4f})"
    )

    print(f"\nK={k}  — B0 cluster_avg farki (d, negatif MAE = daha iyi)")
    print(f"{'Yontem':<26} {'dMAE':>7} {'dRMSE':>7} {'dP@10':>7} {'dR@10':>7} {'dNDCG':>7}")
    print(sep)
    for algo in meta_algos:
        m = results["meta"][algo]
        print(_delta_row(f"{algo} cluster_avg", m["cluster_avg"], b0))
        print(_delta_row(f"MetaDual-CF ({algo})", m["meta_dual"], b0))

    print(f"\nK={k}  — MetaDual vs ayni algo cluster_avg (d)")
    print(f"{'Yontem':<26} {'dMAE':>7} {'dRMSE':>7} {'dP@10':>7} {'dR@10':>7} {'dNDCG':>7}")
    print(sep)
    for algo in meta_algos:
        m = results["meta"][algo]
        print(_delta_row(f"MetaDual ({algo})", m["meta_dual"], m["cluster_avg"]))


def _print_fusion_ab(k: int, results: dict, meta_algos: list, tuned: bool = False) -> None:
    """kmref+cluster_avg vs MetaDual (z / raw) ozet tablosu."""
    b0 = results["B0 cluster_avg"]
    hdr = f"{'Yontem':<32} {'MAE':>7} {'RMSE':>7} {'P@10':>7} {'NDCG':>7}"
    if tuned:
        hdr += f" {'beta':>5} {'rank':>10}"
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f"K={k}  FUSION A/B — kmref (dis) vs MetaDual")
    print("  kmref_cluster_avg = meta centroid + Lloyd atama, duz cluster_avg tahmin")
    print("  meta_dual_raw     = z yok; sinyal = R_ci - mu_cluster (normalize yok)")
    print(hdr)
    print(sep)
    print(_fmt_row("B0 cluster_avg", b0))

    for algo in meta_algos:
        m = results["meta"][algo]
        rows = [
            ("cluster_avg (meta)", m["cluster_avg"]),
            ("MetaDual zscore", m["meta_dual"]),
            ("MetaDual raw_delta", m["meta_dual_raw"]),
        ]
        if m.get("kmref_cluster_avg") is not None:
            rows.insert(1, ("kmref + cluster_avg", m["kmref_cluster_avg"]))
        for label, row in rows:
            line = _fmt_row(f"{algo} {label}", row)
            if tuned and "meta_dual" in label:
                line += (
                    f" {row.get('beta', 0):>5.2f} "
                    f"{row.get('rank_mode', ''):>10}"
                )
            print(line)

    print(f"\nK={k}  — kmref+cluster_avg referansina gore delta (d)")
    print(f"{'Yontem':<32} {'dMAE':>7} {'dNDCG':>7}")
    print(sep)
    for algo in meta_algos:
        m = results["meta"][algo]
        ref = m.get("kmref_cluster_avg") or m["cluster_avg"]
        ref_name = "kmref" if m.get("kmref_cluster_avg") else "cluster_avg"
        for label, row in [
            ("cluster_avg", m["cluster_avg"]),
            ("MetaDual z", m["meta_dual"]),
            ("MetaDual raw", m["meta_dual_raw"]),
        ]:
            print(
                f"{algo + ' ' + label:<32} "
                f"{row['mae'] - ref['mae']:>+7.4f} "
                f"{row['ndcg_at_10'] - ref['ndcg_at_10']:>+7.4f}  (vs {ref_name})"
            )


def _fusion_results_to_rows(k: int, results: dict, meta_algos: list) -> List[dict]:
    rows = []
    b0 = results["B0 cluster_avg"]
    rows.append({
        "K": k, "algo": "B0_KMEANS", "method": "cluster_avg",
        "mae": b0["mae"], "ndcg_at_10": b0["ndcg_at_10"],
        "beta": None, "rank_mode": None,
    })
    method_keys = (
        ("cluster_avg", "cluster_avg"),
        ("kmref_cluster_avg", "kmref_cluster_avg"),
        ("meta_dual", "meta_dual"),
        ("meta_dual_raw", "meta_dual_raw"),
    )
    for algo in meta_algos:
        m = results["meta"][algo]
        for key, method in method_keys:
            r = m.get(key)
            if r is None:
                continue
            rows.append({
                "K": k, "algo": algo, "method": method,
                "mae": r["mae"], "ndcg_at_10": r["ndcg_at_10"],
                "beta": r.get("beta"), "rank_mode": r.get("rank_mode"),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="MetaDual-CF değerlendirme")
    p.add_argument("--k", type=int, nargs="+", default=[10])
    p.add_argument(
        "--meta-algo", nargs="+", default=None,
        help=f"Meta algoritmalar (varsayılan: {' '.join(META_ALGOS_DEFAULT)})",
    )
    p.add_argument("--base-algo", default="B0_KMEANS")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument(
        "--kmref", action="store_true",
        help="Meta atamayi _kmref klasorunden yukle (eski overwrite kosulari)",
    )
    p.add_argument(
        "--fusion-ab", action="store_true",
        help="kmref+cluster_avg vs MetaDual (z / raw_delta) karsilastirmasi",
    )
    p.add_argument(
        "--rank-mode",
        choices=list(RANK_MODES),
        default=RANK_ZSCORE,
        help="Tek-kosuda MetaDual rank sinyali (fusion-ab disinda)",
    )
    p.add_argument(
        "--tune", action="store_true",
        help="Algo-ozel beta/tau: train icinden val ornegi, NDCG ile sec",
    )
    p.add_argument(
        "--csv", default=None,
        help="Tum K sonuclarini CSV olarak kaydet",
    )
    p.add_argument(
        "--wnmf-dim", type=int, default=20,
        help="Assignment suffix wnmf boyutu (varsayilan 20; sweep icin 40)",
    )
    args = p.parse_args()

    meta_algos = args.meta_algo or list(META_ALGOS_DEFAULT)
    t0 = time.time()
    all_rows: List[dict] = []

    for k in args.k:
        if args.fusion_ab:
            results = run_fusion_ab(
                k, meta_algos,
                base_algo=args.base_algo,
                fold=args.fold,
                beta=args.beta,
                tau=args.tau,
                tune=args.tune,
                wnmf_dim=args.wnmf_dim,
            )
            all_rows.extend(_fusion_results_to_rows(k, results, meta_algos))
            _print_fusion_ab(k, results, meta_algos, tuned=args.tune)
        else:
            results = run_comparison(
                k, meta_algos,
                base_algo=args.base_algo,
                fold=args.fold,
                beta=args.beta,
                tau=args.tau,
                kmref=args.kmref,
                tune=args.tune,
                wnmf_dim=args.wnmf_dim,
            )
            all_rows.extend(_results_to_rows(k, results, meta_algos))
            _print_comparison(k, results, meta_algos, tuned=args.tune)
            if args.rank_mode != RANK_ZSCORE:
                print(
                    f"\nNot: --rank-mode {args.rank_mode} yalnizca "
                    "predict_meta_dual API ile; standart karsilastirmada zscore kullanilir."
                )

    if len(args.k) > 1:
        _print_k_sweep_summary(all_rows)

    if args.csv:
        import pandas as pd
        out = os.path.join(REPO, args.csv) if not os.path.isabs(args.csv) else args.csv
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nCSV kaydedildi: {out}")

    print(f"\nToplam süre: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
