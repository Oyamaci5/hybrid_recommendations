# check_pca_variance.py
import argparse
import os

import numpy as np
from sklearn.decomposition import PCA

from mealpy_comparison_v2 import load_movielens, load_movielens_1m

_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(os.path.dirname(_BASE), 'data')
DATA_100K = os.path.join(_DATA_ROOT, 'ml-100k', 'u.data')
DATA_1M = os.path.join(_DATA_ROOT, 'ml-1m', 'ratings.dat')

N_LIST = [10, 20, 30, 50, 75, 100, 150, 200]


def pca_report(title: str, matrix: np.ndarray) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)
    pca = PCA()
    pca.fit(matrix)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    max_n = len(cumvar)
    for n in N_LIST:
        if n > max_n:
            print(f"PCA {n:3d}: (n > max bileşen {max_n}, atlandı)")
            continue
        print(f"PCA {n:3d}: %{cumvar[n - 1] * 100:.1f} varyans açıklanıyor")


def parse_args():
    p = argparse.ArgumentParser(description='ML-100K / ML-1M üzerinde PCA kümülatif varyans özeti')
    p.add_argument(
        '--dataset',
        choices=['100k', '1m', 'both'],
        default='both',
        help='Hangi veri seti (varsayılan: both)',
    )
    p.add_argument('--data-100k', default=DATA_100K, help='ML-100K u.data yolu')
    p.add_argument('--data-1m', default=DATA_1M, help='ML-1M ratings.dat yolu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dataset in ('100k', 'both'):
        m = load_movielens(args.data_100k)
        pca_report('ML-100K', m)
    if args.dataset in ('1m', 'both'):
        m1 = load_movielens_1m(args.data_1m)
        pca_report('ML-1M', m1)

# Çıktıya göre karar ver. Genellikle %80 varyans için gereken bileşen → optimal PCA boyutu
