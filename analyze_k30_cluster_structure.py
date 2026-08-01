"""
K=30: B0 + B1/HA/IWO — küme düzeyinde neden tahmin ayrışmıyor?

1) Küme boyutu / doluluk
2) Hungarian eşleştirilmiş kümeler: kullanıcı Jaccard, item-ortalama vektör korelasyonu
3) cluster_avg tahminleri test üzerinde korelasyon
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wnmf.meta_dual_cf import (
    _assign_dir,
    _baseline_cluster_avg_user,
    _cluster_item_stats,
    _load_assignments,
)
from wnmf.wnmf_utils import load_ratings_100k_all

ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
K = 30
VARIANTS = ("no_kmref", "kmref")


def _suffix(kmref: bool) -> str:
    s = f"_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{K}"
    return s + ("_kmref" if kmref else "")


def _load_assign(kmref: bool, algo: str) -> np.ndarray | None:
    d = _assign_dir(algo, K, kmref_meta=kmref)
    p = os.path.join(d, "assignments.npy")
    return np.load(p) if os.path.isfile(p) else None


def cluster_structure(assign: np.ndarray, k: int) -> dict:
    bc = np.bincount(assign, minlength=k)
    active = bc[bc > 0]
    return {
        "n_active": int((bc > 0).sum()),
        "n_empty": int((bc == 0).sum()),
        "size_min": int(active.min()) if active.size else 0,
        "size_max": int(active.max()) if active.size else 0,
        "size_mean": float(active.mean()) if active.size else 0.0,
        "size_cv": float(active.std() / active.mean()) if active.size and active.mean() > 0 else 0.0,
    }


def _item_mean_corr(
    means_a: np.ndarray,
    counts_a: np.ndarray,
    means_b: np.ndarray,
    counts_b: np.ndarray,
    min_count: int = 3,
) -> float:
    """Eşleşmiş kümelerde ortak destekli item ortalamaları korelasyonu."""
    mask = (counts_a >= min_count) & (counts_b >= min_count)
    if mask.sum() < 5:
        return float("nan")
    va = means_a[mask]
    vb = means_b[mask]
    if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
        return float("nan")
    return float(pearsonr(va, vb)[0])


def match_clusters(
    assign_a: np.ndarray,
    assign_b: np.ndarray,
    means_a: np.ndarray,
    counts_a: np.ndarray,
    means_b: np.ndarray,
    counts_b: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[dict]]:
    """Centroid = küme item-ortalama vektörü (global_mean ile doldurulmuş)."""
    Ca = means_a.copy()
    Cb = means_b.copy()
    cost = cdist(Ca, Cb, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for ia, ib in zip(row_ind, col_ind):
        users_a = set(np.where(assign_a == ia)[0].tolist())
        users_b = set(np.where(assign_b == ib)[0].tolist())
        inter = users_a & users_b
        union = users_a | users_b
        jacc = len(inter) / len(union) if union else 0.0
        ic = _item_mean_corr(
            means_a[ia], counts_a[ia], means_b[ib], counts_b[ib],
        )
        pairs.append({
            "cluster_a": int(ia),
            "cluster_b": int(ib),
            "users_a": len(users_a),
            "users_b": len(users_b),
            "user_jaccard": float(jacc),
            "item_mean_corr": ic,
            "centroid_l2": float(cost[ia, ib]),
        })
    return np.column_stack([row_ind, col_ind]), pairs


def predict_cluster_avg_batch(
    test: np.ndarray,
    train: np.ndarray,
    assign: np.ndarray,
    n_items: int,
    global_mean: float,
) -> np.ndarray:
    means, counts, _, _ = _cluster_item_stats(train, assign, n_items, global_mean)
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    user_m = np.full(n_users, global_mean)
    for u, _, r in train:
        user_m[int(u)] = user_m[int(u)]  # placeholder
    sums = np.zeros(n_users)
    cnts = np.zeros(n_users, dtype=np.int32)
    for u, _, r in train:
        ui = int(u)
        sums[ui] += float(r)
        cnts[ui] += 1
    ok = cnts > 0
    user_m[ok] = sums[ok] / cnts[ok]

    preds = []
    for u, i, r in test:
        u, i = int(u), int(i)
        cid = int(assign[u])
        if counts[cid, i] > 0:
            preds.append(float(means[cid, i]))
        else:
            preds.append(float(user_m[u]))
    return np.array(preds, dtype=np.float64)


def analyze_variant(kmref: bool, train: np.ndarray, test: np.ndarray) -> None:
    tag = "kmref" if kmref else "no_kmref"
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    gm = float(train[:, 2].mean())

    data: dict[str, np.ndarray] = {}
    stats_rows = []
    item_stats: dict[str, tuple] = {}

    for algo in ALGOS:
        a = _load_assign(kmref, algo)
        if a is None:
            print(f"  [{tag}] {algo}: atama yok")
            continue
        data[algo] = a
        st = cluster_structure(a, K)
        means, counts, cm, cs = _cluster_item_stats(train, a, n_items, gm)
        item_stats[algo] = (means, counts)
        n_cells = int((counts > 0).sum())
        sparsity = 1.0 - n_cells / (K * n_items)
        stats_rows.append({
            "variant": tag,
            "algo": algo,
            **st,
            "item_cells_filled": n_cells,
            "item_sparsity": sparsity,
        })

    if len(data) < 2:
        return

    print(f"\n{'#' * 70}\n# K={K}  [{tag}]  küme yapısı\n{'#' * 70}")
    df_st = pd.DataFrame(stats_rows)
    print(df_st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # User-level ARI matrix
    names = list(data.keys())
    print(f"\nKullanıcı ARI ({tag}):")
    for i, a in enumerate(names):
        row = []
        for j, b in enumerate(names):
            row.append(f"{adjusted_rand_score(data[a], data[b]):.3f}")
        print(f"  {a:<12} " + "  ".join(f"{names[j][:4]:>6}" for j in range(len(names))))

    # Matched cluster analysis
    match_rows = []
    for a, b in combinations(names, 2):
        assign_a, assign_b = data[a], data[b]
        ma, ca = item_stats[a]
        mb, cb = item_stats[b]
        _, pairs = match_clusters(assign_a, assign_b, ma, ca, mb, cb, K)
        uj = [p["user_jaccard"] for p in pairs]
        ic = [p["item_mean_corr"] for p in pairs if np.isfinite(p["item_mean_corr"])]
        match_rows.append({
            "variant": tag,
            "pair": f"{a} vs {b}",
            "mean_user_jaccard": float(np.mean(uj)),
            "mean_item_corr": float(np.mean(ic)) if ic else float("nan"),
            "min_item_corr": float(np.min(ic)) if ic else float("nan"),
            "max_item_corr": float(np.max(ic)) if ic else float("nan"),
        })

    print(f"\nHungarian küme eşleştirme ({tag}) — ortalama:")
    print(pd.DataFrame(match_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Prediction correlation
    print(f"\ncluster_avg tahmin korelasyonu — test ({tag}):")
    preds: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for algo in names:
        m = _baseline_cluster_avg_user(train, test, data[algo])
        metrics[algo] = m
        preds[algo] = predict_cluster_avg_batch(test, train, data[algo], n_items, gm)

    print(f"{'Algo':<12} {'MAE':>8} {'NDCG':>8}")
    for algo in names:
        print(f"{algo:<12} {metrics[algo]['mae']:>8.4f} {metrics[algo]['ndcg_at_10']:>8.4f}")

    pred_rows = []
    for a, b in combinations(names, 2):
        pa, pb = preds[a], preds[b]
        r = float(pearsonr(pa, pb)[0])
        same = float((np.abs(pa - pb) < 1e-6).mean())
        mae_diff = float(np.abs(pa - pb).mean())
        pred_rows.append({
            "variant": tag,
            "pair": f"{a} vs {b}",
            "pred_pearson": r,
            "pred_identical_frac": same,
            "pred_mean_abs_diff": mae_diff,
        })
    print(pd.DataFrame(pred_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Fallback analizi: kaç test (u,i) küme item ortalaması yerine user_mean kullanıyor?
    print(f"\nTahmin kaynağı — test ({tag}, fallback=user_mean):")
    fb_rows = []
    for algo in names:
        assign = data[algo]
        _, counts = item_stats[algo]
        n_fb = 0
        for u, i, _ in test:
            if counts[int(assign[int(u)]), int(i)] == 0:
                n_fb += 1
        fb_rows.append({
            "algo": algo,
            "fallback_frac": n_fb / len(test),
        })
    print(pd.DataFrame(fb_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    train, test = load_ratings_100k_all(
        os.path.join(REPO, "data", "ml-100k", "u.data"), random_seed=42, fold=1,
    )
    print(f"K={K}  train={len(train)}  test={len(test)}  users=943  items=1682")

    analyze_variant(False, train, test)
    analyze_variant(True, train, test)

    print("\n" + "=" * 70)
    print("Yorum ipuçları:")
    print("  - Düşük user_jaccard + yüksek item_mean_corr → farklı kullanıcılar,")
    print("    benzer küme×item profili → cluster_avg tahminleri yakın.")
    print("  - Yüksek fallback_frac → çoğu (u,i) user_mean; atama farkı azalır.")
    print("  - pred_pearson≈1 → downstream'de algo ayrışması yok (beklenen).")


if __name__ == "__main__":
    main()
