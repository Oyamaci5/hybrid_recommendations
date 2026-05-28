"""
Aynı (u,i) test çiftleri için meta-algoritma tahminleri arası Pearson korelasyonu.
ClusterAvg (ağırlıklı küme-içi) ile uyumlu; wnmf_experiment.run_cluster_average ile aynı train/test.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wnmf.wnmf_experiment import (  # noqa: E402
    DATA_100K_ALL,
    RANDOM_SEED,
    _build_knn_sim_index,
    _compute_cluster_rating_means,
    _compute_item_mean_offsets,
    _predict_weighted_cluster_avg,
    build_item_popularity,
    load_ratings_100k_all,
)
from wnmf.wnmf_utils import load_assignment  # noqa: E402

ALGOS = ['B0_KMEANS', 'B1_HHO', 'B_AVOA', 'HA_AVOAHGS', 'IWO_HHO']
ASSIGN_ROOT = os.path.join(ROOT, 'mealpy', 'results', 'assignments', 'ml100k')


def resolve_assign_dir(algo: str, assign_suffix: str) -> str:
    d = os.path.join(ASSIGN_ROOT, f'{algo}{assign_suffix}')
    if os.path.isdir(d):
        return d
    if algo == 'B0_KMEANS' and assign_suffix.endswith('_kmref'):
        alt = assign_suffix.replace('_kmref', '')
        d2 = os.path.join(ASSIGN_ROOT, f'{algo}{alt}')
        if os.path.isdir(d2):
            return d2
    raise FileNotFoundError(f'Assignment yok: {d}')


def build_weighted_cluster_context(train: np.ndarray, assignments: np.ndarray, similarity: str, min_common: int):
    n_users = len(assignments)
    user_ratings: dict = {}
    user_sums = np.zeros(n_users, dtype=np.float64)
    user_counts = np.zeros(n_users, dtype=np.int32)
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        user_ratings.setdefault(u, {})[i] = r
        if u < n_users:
            user_sums[u] += r
            user_counts[u] += 1
    global_mean = float(train[:, 2].mean())
    user_means = {
        u: float(user_sums[u] / user_counts[u])
        for u in range(n_users)
        if user_counts[u] > 0
    }
    cluster_users: dict = {}
    for u in range(n_users):
        cid = int(assignments[u])
        if cid < 0:
            continue
        cluster_users.setdefault(cid, []).append(u)
    cluster_mean_arr = _compute_cluster_rating_means(train, assignments)
    cluster_means = {
        int(cid): float(cluster_mean_arr[cid]) for cid in range(len(cluster_mean_arr))
    }
    item_mean_offsets = _compute_item_mean_offsets(train, global_mean)
    item_popularity = build_item_popularity(user_ratings=user_ratings)
    sim_index = _build_knn_sim_index(
        user_ratings,
        user_means,
        item_popularity,
        similarity=similarity,
        min_common=min_common,
    )
    return (
        user_ratings,
        user_means,
        cluster_means,
        item_mean_offsets,
        global_mean,
        cluster_users,
        item_popularity,
        sim_index,
    )


def predict_test_matrix(
    test: np.ndarray,
    assignments: np.ndarray,
    ctx: tuple,
    similarity: str,
    min_common: int,
) -> np.ndarray:
    (
        user_ratings,
        user_means,
        cluster_means,
        item_mean_offsets,
        global_mean,
        cluster_users,
        item_popularity,
        sim_index,
    ) = ctx
    preds = np.zeros(len(test), dtype=np.float32)
    for idx, row in enumerate(test):
        u, i = int(row[0]), int(row[1])
        cid = int(assignments[u]) if u < len(assignments) else -1
        if cid < 0:
            preds[idx] = float(user_means.get(u, global_mean))
            continue
        preds[idx] = _predict_weighted_cluster_avg(
            u,
            i,
            cid,
            cluster_users,
            user_ratings,
            user_means,
            cluster_means,
            item_mean_offsets,
            global_mean,
            sim_index,
            similarity=similarity,
            item_popularity=item_popularity,
            min_common=min_common,
        )
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--wnmf-dim', type=int, default=30, help='wnmf latent suffix: wnmf{N}')
    p.add_argument('--assign-suffix', type=str, default=None)
    p.add_argument('--similarity', type=str, default='cosine')
    p.add_argument('--min-common', type=int, default=3)
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    if args.assign_suffix:
        suffix = args.assign_suffix
    else:
        suffix = f'_euc_imkpp_nogs_none_wnmf{args.wnmf_dim}_k{args.k}_kmref'

    train, test = load_ratings_100k_all(
        DATA_100K_ALL, random_seed=RANDOM_SEED, fold=args.fold,
    )
    true = test[:, 2].astype(np.float32)

    pred_cols = {}
    mae_by_algo = {}
    for algo in ALGOS:
        adir = resolve_assign_dir(algo, suffix)
        assignments, _gray = load_assignment(adir)
        ctx = build_weighted_cluster_context(
            train, assignments, args.similarity, args.min_common,
        )
        pred = predict_test_matrix(test, assignments, ctx, args.similarity, args.min_common)
        pred_cols[algo] = pred
        mae_by_algo[algo] = float(np.mean(np.abs(true - pred)))

    P = np.column_stack([pred_cols[a] for a in ALGOS])
    corr = np.corrcoef(P, rowvar=False)
    corr_df = pd.DataFrame(corr, index=ALGOS, columns=ALGOS)

    off_diag = corr[np.triu_indices(len(ALGOS), k=1)]
    spread = max(mae_by_algo.values()) - min(mae_by_algo.values())

    out = args.out or os.path.join(
        ROOT, 'mealpy', 'results',
        f'k{args.k}_wnmf{args.wnmf_dim}_pred_correlation_{args.similarity}.csv',
    )
    corr_df.to_csv(out, float_format='%.6f')

    print(f'K={args.k} wnmf{args.wnmf_dim} | suffix={suffix} | sim={args.similarity}')
    print(f'Test pairs: {len(test)}')
    print('\nMAE by algo:')
    for a in ALGOS:
        print(f'  {a}: {mae_by_algo[a]:.4f}')
    print(f'MAE spread (max-min): {spread:.4f} ({100*spread/np.mean(list(mae_by_algo.values())):.2f}%)')
    print('\nPearson correlation (same u,i predictions):')
    print(corr_df.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'\nOff-diagonal: mean={off_diag.mean():.4f} min={off_diag.min():.4f} max={off_diag.max():.4f}')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
