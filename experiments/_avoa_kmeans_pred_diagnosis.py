"""
AVOA-K-MEANS vs B0: neden ARI farklı ama MAE yakın? Çift doğrulama.

Fold 1 üzerinde:
  1) Assignment dosyaları gerçekten farklı mı?
  2) CalcAvgRating tahminleri çift yönlü bağımsız hesaplanınca aynı mı?
  3) Test çifti bazında tahmin korelasyonu / özdeşlik oranı
  4) Küme-item ortalama matrisleri ne kadar benzer?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))

from wnmf.wnmf_utils import load_ratings_100k, load_assignment  # noqa: E402
from wnmf.wnmf_experiment import run_cluster_average  # noqa: E402

FOLD = 1
K = 10
LATENT = 50
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"

DIRS = {
    "B0_KMEANS": ASSIGN_ROOT
    / f"B0_KMEANS_euc_imkpp_nogs_trainonly_official_f{FOLD}_none_wnmf{LATENT}_k{K}_pwcss",
    "B_AVOA": ASSIGN_ROOT
    / f"B_AVOA_euc_imkpp_nogs_trainonly_official_f{FOLD}_none_wnmf{LATENT}_k{K}_pwcss_kmref",
}


def calc_avg_rating_manual(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    n_items: int,
) -> tuple[np.ndarray, dict]:
    """Thakrar Alg.6: cluster ortalaması; yoksa kullanıcı ortalaması."""
    n_users = len(assignments)
    n_clusters = int(assignments.max()) + 1
    cluster_users: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
    for u in range(n_users):
        cluster_users[int(assignments[u])].append(u)

    train_map: dict[tuple[int, int], float] = {}
    user_sum = np.zeros(n_users, dtype=np.float64)
    user_cnt = np.zeros(n_users, dtype=np.int32)
    for u, i, r in train:
        u, i = int(u), int(i)
        train_map[(u, i)] = float(r)
        user_sum[u] += float(r)
        user_cnt[u] += 1
    user_mean = np.where(user_cnt > 0, user_sum / np.maximum(user_cnt, 1), train[:, 2].mean())

    preds = np.zeros(len(test), dtype=np.float64)
    src = {"cluster_mean": 0, "user_mean": 0, "global_mean": 0}
    global_mean = float(train[:, 2].mean())

    for idx, (u, i, _r) in enumerate(test):
        u, i = int(u), int(i)
        cid = int(assignments[u])
        vals = [
            train_map[(uu, i)]
            for uu in cluster_users[cid]
            if (uu, i) in train_map
        ]
        if vals:
            preds[idx] = float(np.mean(vals))
            src["cluster_mean"] += 1
        elif user_cnt[u] > 0:
            preds[idx] = float(user_mean[u])
            src["user_mean"] += 1
        else:
            preds[idx] = global_mean
            src["global_mean"] += 1
    return preds, src


def build_cluster_item_means(
    train: np.ndarray,
    assignments: np.ndarray,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = int(assignments.max()) + 1
    means = np.full((n_clusters, n_items), np.nan, dtype=np.float64)
    counts = np.zeros((n_clusters, n_items), dtype=np.int32)
    for u, i, r in train:
        cid = int(assignments[int(u)])
        i = int(i)
        if np.isnan(means[cid, i]):
            means[cid, i] = 0.0
        means[cid, i] += float(r)
        counts[cid, i] += 1
    mask = counts > 0
    means[mask] /= counts[mask]
    return means, counts


def main() -> None:
    base = REPO / "data" / "ml-100k" / f"u{FOLD}.base"
    test_p = REPO / "data" / "ml-100k" / f"u{FOLD}.test"
    train, test = load_ratings_100k(str(base), str(test_p), fold=FOLD)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1

    print("=" * 72)
    print(f"AVOA-K-MEANS tahmin teşhisi — fold {FOLD}, K={K}, L={LATENT}")
    print("=" * 72)

    # --- 1) Assignment farkı ---
    assigns = {}
    for algo, d in DIRS.items():
        a, g = load_assignment(str(d))
        assigns[algo] = a
        print(f"\n[{algo}] dir={d.name}")
        print(f"  assignments: shape={a.shape}, unique={len(np.unique(a))}")
        print(f"  cluster sizes: {sorted(np.bincount(a.astype(int), minlength=K).tolist())}")

    ari = adjusted_rand_score(assigns["B0_KMEANS"], assigns["B_AVOA"])
    agree = float(np.mean(assigns["B0_KMEANS"] == assigns["B_AVOA"]))
    print(f"\nAssignment overlap: ARI={ari:.4f}, same_label_pct={agree*100:.1f}%")

    # --- 2) run_cluster_average vs manual (paper-mode) ---
    rows = {}
    for algo, d in DIRS.items():
        assignments, gray_mask = load_assignment(str(d))
        r_exp = run_cluster_average(
            train, test, assignments, gray_mask, None, n_items, algo,
            top_n=10, relevance_threshold=4.0,
            cluster_avg_hard=True,
            cluster_avg_global_fallback=False,
            assign_dir=str(d),
            return_eval_rows=True,
        )
        preds_man, src_man = calc_avg_rating_manual(train, test, assignments, n_items)
        preds_exp = r_exp["eval_rows"][:, 3].astype(np.float64)
        max_diff = float(np.max(np.abs(preds_exp - preds_man)))
        print(f"\n[{algo}] run_cluster_average vs manual Alg.6")
        print(f"  MAE exp={r_exp['mae']:.6f}  manual={np.mean(np.abs(test[:,2]-preds_man)):.6f}")
        print(f"  max|pred_exp - pred_man| = {max_diff:.8f}")
        print(f"  manual src: {src_man}")
        rows[algo] = {
            "mae": r_exp["mae"],
            "preds": preds_exp,
            "assignments": assignments,
        }

    # --- 3) Tahmin karşılaştırması ---
    p0 = rows["B0_KMEANS"]["preds"]
    p1 = rows["B_AVOA"]["preds"]
    truths = test[:, 2].astype(np.float64)
    diff = p1 - p0
    identical = float(np.mean(np.isclose(p0, p1, atol=1e-6)))
    corr = float(np.corrcoef(p0, p1)[0, 1])
    mae_b0 = float(np.mean(np.abs(truths - p0)))
    mae_avoa = float(np.mean(np.abs(truths - p1)))
    print("\n--- Test çifti bazında B0 vs AVOA tahminleri ---")
    print(f"  MAE B0={mae_b0:.6f}  AVOA={mae_avoa:.6f}  delta={mae_avoa-mae_b0:+.6f}")
    print(f"  pred correlation: {corr:.6f}")
    print(f"  identical preds (atol=1e-6): {identical*100:.2f}%")
    print(f"  |pred_diff| mean={np.mean(np.abs(diff)):.6f}  median={np.median(np.abs(diff)):.6f}")
    print(f"  |pred_diff| p90={np.percentile(np.abs(diff),90):.6f}  max={np.max(np.abs(diff)):.6f}")

    # Kullanıcı bazında farklı küme → tahmin farkı
    u_diff_cluster = assigns["B0_KMEANS"] != assigns["B_AVOA"]
    test_u = test[:, 0].astype(int)
    mask_diff_u = u_diff_cluster[test_u]
    if np.any(mask_diff_u):
        d_sub = np.abs(p1[mask_diff_u] - p0[mask_diff_u])
        print(f"\n  Sadece farklı kümedeki kullanıcıların test çiftleri ({mask_diff_u.sum()}/{len(test)}):")
        print(f"    |pred_diff| mean={d_sub.mean():.6f}  identical={np.mean(np.isclose(p0[mask_diff_u],p1[mask_diff_u],atol=1e-6))*100:.1f}%")
    same_u = ~mask_diff_u
    if np.any(same_u):
        print(f"  Aynı kümedeki kullanıcılar ({same_u.sum()} çift): identical={np.mean(np.isclose(p0[same_u],p1[same_u],atol=1e-6))*100:.1f}%")

    # --- 4) Küme-item ortalama matrisi benzerliği ---
    m0, c0 = build_cluster_item_means(train, assigns["B0_KMEANS"], n_items)
    m1, c1 = build_cluster_item_means(train, assigns["B_AVOA"], n_items)
    both = (c0 > 0) & (c1 > 0)
    if np.any(both):
        v0 = m0[both]
        v1 = m1[both]
        print("\n--- Train küme×item ortalama matrisleri (farklı partition) ---")
        print(f"  Hücreler her iki partition'da da dolu: {both.sum()} / {n_items*K*2}")
        print(f"  Ortalama |mean_B0 - mean_AVOA| (aynı (cid,i) indeksi): N/A (farklı partition)")
    # Item bazında: tüm kümelerdeki ortalama
    item_mean_b0 = np.nanmean(m0, axis=0)
    item_mean_avoa = np.nanmean(m1, axis=0)
    valid_items = ~(np.isnan(item_mean_b0) | np.isnan(item_mean_avoa))
    print(f"  Item başına K-küme ort. korelasyonu: {np.corrcoef(item_mean_b0[valid_items], item_mean_avoa[valid_items])[0,1]:.4f}")

    # Aynı test (u,i): farklı cid → farklı cluster_mean
    cid0 = assigns["B0_KMEANS"][test_u]
    cid1 = assigns["B_AVOA"][test_u]
    items = test[:, 1].astype(int)
    cm0 = np.array([
        m0[c, i] if c0[c, i] > 0 else np.nan for c, i in zip(cid0, items)
    ])
    cm1 = np.array([
        m1[c, i] if c1[c, i] > 0 else np.nan for c, i in zip(cid1, items)
    ])
    both_cm = ~(np.isnan(cm0) | np.isnan(cm1))
    if np.any(both_cm):
        cm_corr = float(np.corrcoef(cm0[both_cm], cm1[both_cm])[0, 1])
        cm_mae = float(np.mean(np.abs(cm0[both_cm] - cm1[both_cm])))
        print(f"\n  Test çiftlerinde kullanılan cluster_mean değerleri:")
        print(f"    her ikisi de cluster_mean: {both_cm.sum()} / {len(test)}")
        print(f"    cluster_mean korelasyonu: {cm_corr:.6f}")
        print(f"    |cluster_mean_B0 - cluster_mean_AVOA| ort={cm_mae:.6f}")

    # Popüler item: küme değişse bile ortalama benzer mi?
    item_freq = np.bincount(train[:, 1].astype(int), minlength=n_items)
    top_items = np.argsort(-item_freq)[:20]
    print("\n  Top-20 popüler item: test çiftlerinde |pred_diff| ort")
    for it in top_items:
        m = items == it
        if m.sum() == 0:
            continue
        print(f"    item {it:4d} (train_cnt={item_freq[it]:4d}): "
              f"n_test={m.sum():3d}  mean|diff|={np.mean(np.abs(diff[m])):.4f}")

    # CSV özet
    out = REPO / "results" / "avoa_kmeans_pred_diagnosis_fold1.csv"
    summary = pd.DataFrame([{
        "fold": FOLD, "ari": ari, "same_label_pct": agree,
        "pred_corr": corr, "identical_pred_pct": identical,
        "mae_b0": mae_b0, "mae_avoa": mae_avoa,
        "mae_delta": mae_avoa - mae_b0,
        "mean_abs_pred_diff": float(np.mean(np.abs(diff))),
        "cluster_mean_corr": cm_corr if both_cm.any() else float("nan"),
    }])
    summary.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
