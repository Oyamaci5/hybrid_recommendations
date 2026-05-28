"""
FilmTrust: TruncatedSVD-10 ile rating tahmini (80/20 holdout).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "filmtrust", "ratings.txt"
)
N_COMPONENTS = 10
TEST_SIZE = 0.2
SEED = 42
RATING_MIN, RATING_MAX = 0.5, 4.0


def load_triplets(path):
    df = pd.read_csv(path, sep=" ", names=["user_id", "item_id", "rating"])
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["item_id"].unique())
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {it: j for j, it in enumerate(items)}
    rows = np.array(
        [[u_map[int(u)], i_map[int(i)], float(r)] for u, i, r in df.values],
        dtype=np.float64,
    )
    return rows, len(users), len(items)


def triplets_to_matrix(rows, n_users, n_items):
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r in rows:
        R[int(u), int(i)] = float(r)
    return R


def metrics(y_true, y_pred):
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def predict_test(R_hat, test_rows):
    preds = []
    for u, i, r in test_rows:
        preds.append(float(np.clip(R_hat[int(u), int(i)], RATING_MIN, RATING_MAX)))
    return np.array(preds, dtype=np.float64)


def svd_reconstruct(R_train, n_components, center="none"):
    R = np.asarray(R_train, dtype=np.float64)
    offset = np.zeros(R.shape[0], dtype=np.float64)
    if center == "user":
        counts = (R > 0).sum(axis=1)
        sums = np.where(R > 0, R, 0.0).sum(axis=1)
        offset = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        mask = R > 0
        R_fit = np.where(mask, R - offset[:, None], 0.0)
    else:
        R_fit = R

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    U = svd.fit_transform(R_fit)
    R_hat = svd.inverse_transform(U)
    if center == "user":
        R_hat = R_hat + offset[:, None]
    return R_hat.astype(np.float32), float(svd.explained_variance_ratio_.sum())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA
    if not os.path.isfile(path):
        print(f"Dosya bulunamadi: {path}", file=sys.stderr)
        sys.exit(1)

    all_rows, n_users, n_items = load_triplets(path)
    train_rows, test_rows = train_test_split(
        all_rows, test_size=TEST_SIZE, random_state=SEED,
    )

    R_train = triplets_to_matrix(train_rows, n_users, n_items)
    sparsity = 1.0 - np.count_nonzero(R_train) / R_train.size
    y_true = test_rows[:, 2]

    print(f"Kaynak     : {os.path.abspath(path)}")
    print(f"Kullanici  : {n_users}  Item: {n_items}")
    print(f"Train/Test : {len(train_rows)} / {len(test_rows)}")
    print(f"Train seyreklik: {sparsity:.3f}")
    print(f"SVD boyutu : {N_COMPONENTS}\n")

    for center in ("none", "user"):
        R_hat, evr = svd_reconstruct(R_train, N_COMPONENTS, center=center)
        y_pred = predict_test(R_hat, test_rows)
        mae, rmse = metrics(y_true, y_pred)
        tag = "kullanici-ort. cikarilmis" if center == "user" else "ham 0 doldurma"
        print(
            f"[{tag}] cum. varyans aciklanan: {evr:.3f}  "
            f"MAE={mae:.4f}  RMSE={rmse:.4f}"
        )


if __name__ == "__main__":
    main()
