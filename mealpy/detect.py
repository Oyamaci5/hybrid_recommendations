import numpy as np
import os

# Mevcut assignment sonuçlarına bak
base = 'results/assignments_lof'
for dataset in ['ml100k']:
    for algo in ['B1_HHO', 'B2_HGS', 'H1_HHO+HGS', 'H4_MFO+HHO']:
        path = os.path.join(base, dataset, algo)
        if not os.path.exists(path):
            path = os.path.join(base, dataset, f'{algo}_k70')
        if os.path.exists(path):
            # best_sol'un WCSS'ini hesapla
            from mealpy_comparison_v2 import load_movielens, compute_wcss_fast
            matrix = load_movielens('../data/ml-100k/u.data')
            best_sol = np.load(os.path.join(path, 'best_sol.npy'))
            K = int(best_sol.shape[0] / matrix.shape[1])
            wcss, _ = compute_wcss_fast(matrix, best_sol, K)
            print(f"{algo:<20} K={K}  WCSS={wcss:.2f}")