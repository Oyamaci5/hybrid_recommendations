"""Quick B0_KMEANS assignment generation."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_assignments import load_movielens, prepare_matrix_for_clustering

DATA_100K = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ml-100k', 'u.data')

print('Loading ML-100K...')
matrix = load_movielens(DATA_100K)
print(f'Matrix: {matrix.shape}')

matrix = prepare_matrix_for_clustering(matrix, zscore=True, pca_var=None, wnmf_k=None,
    preprocess='minmax', feature_extraction='none', svd_components=0,
    min_user_ratings=5, min_item_ratings=10)
print(f'After prep: {matrix.shape}')

from sklearn.cluster import KMeans
km = KMeans(n_clusters=7, n_init=10, max_iter=500, random_state=42)
labels = km.fit_predict(matrix)
print(f'KMeans WCSS: {km.inertia_:.2f}')
print(f'Cluster sizes: {np.bincount(labels)}')

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'assignments_lof', 'ml100k', 'B0_KMEANS_pruneu5_i10_zscore_k7')
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'assignments.npy'), labels)
gray_mask = np.zeros(len(labels), dtype=bool)
np.save(os.path.join(save_dir, 'gray_mask.npy'), gray_mask)
centroids = km.cluster_centers_.flatten()
np.save(os.path.join(save_dir, 'best_sol.npy'), centroids)
print(f'Saved to {save_dir}')
