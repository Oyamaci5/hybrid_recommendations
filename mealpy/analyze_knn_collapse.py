"""Neden farkli assignment'lar benzer MAE veriyor? kNN=60 + expand-knn analizi."""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'wnmf'))
sys.path.insert(0, os.path.join(REPO, 'mealpy'))

from wnmf_experiment import run_cluster_knn  # noqa: E402
from generate_assignments import load_movielens  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

SEED = 42
SUFFIX = '_pruneu5_i10_zscore_euc_imkpp_none_wnmf30_k20'
ROOT = os.path.join(REPO, 'mealpy', 'results', 'assignments_lof', 'ml100k')
ALGOS = ['B0_KMEANS', 'HA_AVOAHGS', 'LIT_PSO', 'H4_MFO+HHO']


def load_fold5():
    ratings = load_movielens(os.path.join(REPO, 'data', 'ml-100k', 'u.data'))
    n_users, n_items = ratings.shape
    rows = []
    for u in range(n_users):
        for i in range(n_items):
            r = ratings[u, i]
            if r > 0:
                rows.append([u, i, r])
    rows = np.array(rows, dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_i, (tr, te) in enumerate(kf.split(rows), start=1):
        if fold_i == 5:
            return rows[tr], rows[te], n_users, n_items


def load_algo(algo):
    d = os.path.join(ROOT, f'{algo}{SUFFIX}')
    assignments = np.load(os.path.join(d, 'assignments.npy'))
    summary = pd.read_csv(os.path.join(d, 'assignment_summary.csv'))
    gray = summary['is_gray_sheep'].values.astype(bool)
    return assignments, gray


def expand_rate(train, test, assignments, gray, k_neighbors=60):
    """Test tahminlerinde kac oranda expand-knn devreye giriyor?"""
    user_ratings = {}
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        user_ratings.setdefault(u, {})[i] = r

    cluster_users = {}
    for u in range(len(assignments)):
        cluster_users.setdefault(int(assignments[u]), []).append(u)

    n_intra_only = n_expanded = n_no_neighbors = 0
    n_total = 0
    for row in test:
        u, i = int(row[0]), int(row[1])
        if gray[u]:
            continue
        cid = int(assignments[u])
        intra = 0
        for v in cluster_users.get(cid, []):
            if v == u:
                continue
            if i in user_ratings.get(v, {}):
                intra += 1
        n_total += 1
        if intra == 0:
            n_no_neighbors += 1
        elif intra < k_neighbors:
            n_expanded += 1
        else:
            n_intra_only += 1
    return {
        'intra_sufficient': n_intra_only,
        'needs_expand': n_expanded,
        'no_intra_neighbor': n_no_neighbors,
        'total': n_total,
    }


def main():
    train, test, n_users, n_items = load_fold5()
    print(f'Train={len(train)} Test={len(test)} users={n_users} items={n_items}\n')

    print('=== Assignment farki (B0_KMEANS referans) ===')
    ref, _ = load_algo('B0_KMEANS')
    for a in ALGOS[1:]:
        ass, _ = load_algo(a)
        agree = (ref == ass).mean()
        print(f'  B0_KMEANS vs {a}: label agreement = {agree:.1%}')

    print('\n=== expand-knn devreye girme orani (white test hucreleri) ===')
    for a in ALGOS:
        ass, gray = load_algo(a)
        stats = expand_rate(train, test, ass, gray, k_neighbors=60)
        t = stats['total']
        print(
            f"  {a:12s}  "
            f"kume icinde yeterli (>={60}): {stats['intra_sufficient']/t:.1%}  "
            f"expand gerekli (<60): {stats['needs_expand']/t:.1%}  "
            f"kume icinde 0 komşu: {stats['no_intra_neighbor']/t:.1%}"
        )

    print('\n=== Tam test MAE (run_cluster_knn) ===')
    rows = {}
    for a in ALGOS:
        ass, gray = load_algo(a)
        row = run_cluster_knn(
            train, test, ass, gray, None, n_items, a,
            similarity='pearson', k_neighbors=60, expand_knn=True, sig_weight=10,
        )
        rows[a] = row
        print(f"  {a:12s}  MAE={row['mae']:.6f}  NDCG={row['ndcg_at_10']:.6f}")

    print('\n=== kNN=10 ile karsilastirma (expand-knn kapali) ===')
    for a in ['B0_KMEANS', 'HA_AVOAHGS', 'LIT_PSO']:
        ass, gray = load_algo(a)
        row = run_cluster_knn(
            train, test, ass, gray, None, n_items, a,
            similarity='pearson', k_neighbors=10, expand_knn=False, sig_weight=10,
        )
        print(f"  {a:12s}  MAE={row['mae']:.6f}  NDCG={row['ndcg_at_10']:.6f}")


if __name__ == '__main__':
    main()
