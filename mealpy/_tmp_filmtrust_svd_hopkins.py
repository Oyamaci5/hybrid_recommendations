"""FilmTrust: Hopkins + TruncatedSVD raw/norm (CV ve H)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "filmtrust", "ratings.txt"
)


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


def load_filmtrust(path):
    df = pd.read_csv(path, sep=" ", names=["user_id", "item_id", "rating"])
    matrix = (
        df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        .values.astype(np.float32)
    )
    total = matrix.size
    nonzero = np.count_nonzero(matrix)
    print(f"Shape    : {matrix.shape}")
    print(f"Sparsity : {1 - nonzero / total:.3f}")
    print(f"Rating   : {matrix[matrix > 0].min():.1f} - {matrix.max():.1f}")
    return matrix


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA
    if not os.path.isfile(path):
        print(f"Dosya bulunamadi: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Kaynak: {os.path.abspath(path)}\n")
    R = load_filmtrust(path)
    print()

    h = hopkins(R.astype(float))
    print(f"Ham rating Hopkins: {h:.3f}\n")

    print(f"{'':6} {'CV raw':>8} {'H raw':>8} {'CV norm':>8} {'H norm':>8}")
    print("-" * 42)
    for n in [10, 20]:
        W_raw = TruncatedSVD(n_components=n, random_state=42).fit_transform(R)
        W_norm = normalize(W_raw)
        d_raw = pdist(W_raw, metric="euclidean")
        d_norm = pdist(W_norm, metric="euclidean")
        cv_raw = d_raw.std() / d_raw.mean()
        cv_norm = d_norm.std() / d_norm.mean()
        h_raw = hopkins(W_raw)
        h_norm = hopkins(W_norm)
        print(
            f"SVD-{n:<3} {cv_raw:8.3f} {h_raw:8.3f} "
            f"{cv_norm:8.3f} {h_norm:8.3f}"
        )


if __name__ == "__main__":
    main()
