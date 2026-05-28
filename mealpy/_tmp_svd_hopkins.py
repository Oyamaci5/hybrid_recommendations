"""TruncatedSVD / ham rating: Hopkins (raw vs L2-normalized)."""
import os
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(__file__))
from mealpy_comparison_v2 import load_movielens


def hopkins(X, sample_size=100, seed=42):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, sample_size, replace=False)
    X_sample = X[idx]
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    w = nbrs.kneighbors(X_sample)[0][:, 1]
    X_uniform = rng.uniform(X.min(axis=0), X.max(axis=0), (sample_size, d))
    u = nbrs.kneighbors(X_uniform)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


DATA = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
R = load_movielens(DATA)

print(f"\n{'dim':<12} {'raw H':>10} {'norm H':>10}")
print("-" * 34)
for n in [10, 20, 30]:
    W_raw = TruncatedSVD(n_components=n, random_state=42).fit_transform(R)
    W_norm = normalize(W_raw)

    h_raw = hopkins(W_raw)
    h_norm = hopkins(W_norm)

    print(f"SVD-{n:<8} {h_raw:10.3f} {h_norm:10.3f}")

h_raw_rating = hopkins(R.astype(float))
print(f"\nHam rating: H={h_raw_rating:.3f}")
