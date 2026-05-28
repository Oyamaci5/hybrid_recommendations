"""TruncatedSVD: raw vs L2-normalized pairwise distance CV (ml-100k)."""
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(__file__))
from mealpy_comparison_v2 import load_movielens

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
R = load_movielens(DATA)

print(f"\n{'dim':<8} {'raw CV':>10} {'norm CV':>10}")
print("-" * 32)
for n in [10, 20, 30]:
    W_raw = TruncatedSVD(n_components=n, random_state=42).fit_transform(R)
    d = pdist(W_raw, metric="euclidean")
    cv_raw = d.std() / d.mean()

    W_norm = normalize(W_raw)
    d = pdist(W_norm, metric="euclidean")
    cv_norm = d.std() / d.mean()

    print(f"SVD-{n:<3} {cv_raw:10.3f} {cv_norm:10.3f}")
