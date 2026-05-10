import numpy as np
import os
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
import sys

sys.path.insert(0, "mealpy")
from wnmf.wnmf_utils import load_ratings_100k

train, _ = load_ratings_100k("data/ml-100k/u1.base", "data/ml-100k/u1.test")
n_users = int(train[:, 0].max()) + 1
n_items = int(train[:, 1].max()) + 1
R = np.zeros((n_users, n_items), dtype=np.float32)
for u, i, r in train:
    R[int(u), int(i)] = r

R_zs = StandardScaler().fit_transform(R)
R_nn = np.maximum(R_zs, 0)
X = normalize(NMF(20, random_state=42, max_iter=300).fit_transform(R_nn))

base = "mealpy/results/assignments_lof/ml100k"
algos = {
    "H4_MFO+HHO": "_pruneu5_i10_zscore_nmf20_k7",
    "H9_QSA+CDO": "_pruneu5_i10_zscore_nmf20_k7",
    "B3_MFO": "_pruneu5_i10_zscore_nmf20_k7",
    "LIT_GOA": "_pruneu5_i10_zscore_nmf20_k7",
    "HA_AVOAHGS": "_pruneu5_i10_minmax_svd20_k7",
}
maes = {
    "H4_MFO+HHO": 0.7295,
    "H9_QSA+CDO": 0.7295,
    "B3_MFO": 0.7298,
    "LIT_GOA": 0.7296,
    "HA_AVOAHGS": np.nan,
}

print(f"{'Algo':<15} {'WCSS':>10} {'Silhouette':>12} {'MAE':>8}")
print("-" * 50)
results = []
for algo, suffix in algos.items():
    a = np.load(os.path.join(base, f"{algo}{suffix}", "assignments.npy"))
    g = np.load(os.path.join(base, f"{algo}{suffix}", "gray_sheep_mask.npy"))
    X_c = X[~g]
    a_c = a[~g]
    wcss = sum(
        np.sum((X_c[a_c == c] - X_c[a_c == c].mean(0)) ** 2)
        for c in np.unique(a_c)
        if (a_c == c).sum() > 0
    )
    sil = silhouette_score(X_c, a_c)
    print(f"{algo:<15} {wcss:>10.4f} {sil:>12.4f} {maes[algo]:>8.4f}")
    results.append((algo, wcss, sil, maes[algo]))

wcss_vals = [r[1] for r in results]
mae_vals = [r[3] for r in results]
corr = np.corrcoef(wcss_vals, mae_vals)[0, 1]
print(f"\nWCSS-MAE Pearson: {corr:.4f}")
