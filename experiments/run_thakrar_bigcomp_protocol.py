"""
Thakrar et al. (2025 BigComp) — Algorithm 2 reproduction on ml-latest-small.

Paper dataset: 610 users, ~9742 movies, 100836 ratings (same as ml-latest-small).
Split: 80% train / 20% test (random, seed=42).
MF objective (Step 1): min sum (R - P@Q)^2 on observed entries (Alg. 4).
Meta fitness (CSO / HHO init, Step 2): WCSS = sum_i min_c ||x_i - mu_c||^2 (Alg. 5).
Prediction (Step 3): Algorithm 6 cluster average; global mean if no cluster rating.

  python -m experiments.run_thakrar_bigcomp_protocol
  python -m experiments.run_thakrar_bigcomp_protocol --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
MEALPY = REPO / "mealpy"
sys.path.insert(0, str(MEALPY))

from mealpy_comparison_v2 import compute_wcss_fast, get_all_algorithms_v3, mkmeans_plus_plus_init  # noqa: E402
from generate_assignments import run_single  # noqa: E402

RATINGS = REPO / "data" / "ml-latest-small" / "ratings.csv"
OUT = REPO / "results" / "thakrar_bigcomp_protocol.csv"
SEED = 42


def load_latest_small(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    df = pd.read_csv(path)
    u_raw = df["userId"].astype(np.int64).values
    i_raw = df["movieId"].astype(np.int64).values
    r = df["rating"].astype(np.float32).values
    u_map = {u: idx for idx, u in enumerate(sorted(df["userId"].unique()))}
    i_map = {m: idx for idx, m in enumerate(sorted(df["movieId"].unique()))}
    u = np.array([u_map[x] for x in u_raw], dtype=np.int32)
    i = np.array([i_map[x] for x in i_raw], dtype=np.int32)
    n_users, n_items = len(u_map), len(i_map)
    return np.column_stack([u, i, r]), np.zeros((n_users, n_items), dtype=np.float32), n_users, n_items


def split_80_20(rows: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    train, test = train_test_split(rows, test_size=0.2, random_state=seed, shuffle=True)
    return train.astype(np.float32), test.astype(np.float32)


def matrix_factorization(
    train: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    latent_dim: int,
    n_epochs: int,
    lr: float,
    reg: float,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 4 style MF on observed entries only."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, (n_users, latent_dim)).astype(np.float64)
    Q = rng.normal(0, 0.1, (n_items, latent_dim)).astype(np.float64)
    for _ in range(n_epochs):
        for u, i, r in train:
            u, i = int(u), int(i)
            pred = float(np.dot(P[u], Q[i]))
            err = float(r) - pred
            pu = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * pu - reg * Q[i])
    return P.astype(np.float32), Q.astype(np.float32)


def kmeans_lloyd_from_init(
    X: np.ndarray,
    init_centroids: np.ndarray,
    k: int,
    max_iter: int = 300,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 5: Lloyd k-means with fixed init (CSO/HHO output)."""
    km = KMeans(
        n_clusters=k,
        init=np.asarray(init_centroids, dtype=np.float64).reshape(k, X.shape[1]),
        n_init=1,
        max_iter=max_iter,
        random_state=seed,
    )
    km.fit(X)
    return km.labels_.astype(np.int32), km.cluster_centers_.astype(np.float32)


def random_kmeans_init(X: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].astype(np.float64)


def hho_wcss_init(
    X: np.ndarray,
    k: int,
    *,
    epoch: int = 100,
    pop_size: int = 30,
    seed: int = SEED,
) -> np.ndarray:
    """CSO substitute: HHO minimizes WCSS on user-latent matrix P."""
    algo_map = {a["full_name"]: a for a in get_all_algorithms_v3()}
    hho = algo_map["HHO.OriginalHHO"]
    init = mkmeans_plus_plus_init(X, K=k, n_solutions=pop_size, seed=seed, metric="euclidean")
    best_sol, _ = run_single(
        hho, X, k, init, epoch, pop_size,
        metric="euclidean", cluster_objective="wcss",
        fitness_config={"objective": "wcss"},
    )
    return np.asarray(best_sol, dtype=np.float64).reshape(k, X.shape[1])


def calc_avg_rating(
    train: np.ndarray,
    assignments: np.ndarray,
    test: np.ndarray,
) -> tuple[float, float, dict]:
    """Algorithm 6: cluster average; global mean fallback."""
    n_users = len(assignments)
    global_mean = float(train[:, 2].mean())
    cluster_users: dict[int, list[int]] = {}
    for u in range(n_users):
        cluster_users.setdefault(int(assignments[u]), []).append(u)

    train_map: dict[tuple[int, int], float] = {}
    for u, i, r in train:
        train_map[(int(u), int(i))] = float(r)

    preds, truths = [], []
    src = {"cluster_mean": 0, "global_mean": 0}

    for u, i, r in test:
        u, i = int(u), int(i)
        cid = int(assignments[u])
        vals = [
            train_map[(uu, i)]
            for uu in cluster_users.get(cid, [])
            if (uu, i) in train_map
        ]
        if vals:
            pred = float(np.mean(vals))
            src["cluster_mean"] += 1
        else:
            pred = global_mean
            src["global_mean"] += 1
        preds.append(pred)
        truths.append(float(r))

    err = np.array(truths) - np.array(preds)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse, src


def run_one_config(
    train: np.ndarray,
    test: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    k: int,
    latent_dim: int,
    mf_epochs: int,
    lr: float,
    reg: float,
    init_mode: str,
    hho_epoch: int,
    hho_pop: int,
) -> dict:
    t0 = time.time()
    P, Q = matrix_factorization(
        train, n_users, n_items,
        latent_dim=latent_dim, n_epochs=mf_epochs, lr=lr, reg=reg,
    )
    if init_mode == "random":
        centroids = random_kmeans_init(P, k)
    elif init_mode == "mkpp":
        centroids = np.asarray(
            mkmeans_plus_plus_init(P, K=k, n_solutions=1, seed=SEED, metric="euclidean")[0],
            dtype=np.float64,
        ).reshape(k, P.shape[1])
    elif init_mode == "hho":
        centroids = hho_wcss_init(P, k, epoch=hho_epoch, pop_size=hho_pop)
    else:
        raise ValueError(init_mode)

    wcss_pre, _ = compute_wcss_fast(P, centroids.flatten(), k, metric="euclidean")
    assignments, refined = kmeans_lloyd_from_init(P, centroids, k)
    wcss_post, _ = compute_wcss_fast(P, refined.flatten(), k, metric="euclidean")
    mae, rmse, src = calc_avg_rating(train, assignments, test)
    n_test = len(test)
    return {
        "k": k,
        "latent_dim": latent_dim,
        "mf_epochs": mf_epochs,
        "init_mode": init_mode,
        "wcss_after_meta": float(wcss_pre),
        "wcss_after_kmeans": float(wcss_post),
        "mae": mae,
        "rmse": rmse,
        "cluster_mean_pct": 100.0 * src["cluster_mean"] / max(n_test, 1),
        "global_fallback_pct": 100.0 * src["global_mean"] / max(n_test, 1),
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="K=14,L=10 only")
    ap.add_argument("--mf-lr", type=float, default=0.01)
    ap.add_argument("--mf-reg", type=float, default=0.01)
    ap.add_argument("--mf-epochs", type=int, default=50)
    ap.add_argument("--hho-epoch", type=int, default=100)
    ap.add_argument("--hho-pop", type=int, default=30)
    args = ap.parse_args()

    rows_raw, _, n_users, n_items = load_latest_small(RATINGS)
    train, test = split_80_20(rows_raw)
    print(f"Dataset: ml-latest-small  users={n_users}  items={n_items}")
    print(f"Train={len(train):,}  Test={len(test):,}  (80/20, seed={SEED})")
    print(f"MF: L sweep, T={args.mf_epochs}, lr={args.mf_lr}, reg={args.mf_reg}")
    print("Cluster fitness (CSO/HHO): WCSS  |  Prediction: Algorithm 6 cluster avg\n")

    if args.quick:
        k_list = [14]
        l_list = [10]
    else:
        k_list = [9, 14, 19]
        l_list = [5, 10, 15]

    init_modes = ["random", "mkpp", "hho"]
    results = []
    for k in k_list:
        for ld in l_list:
            for mode in init_modes:
                print(f"--- k={k} L={ld} init={mode} ---", flush=True)
                row = run_one_config(
                    train, test, n_users, n_items,
                    k=k, latent_dim=ld, mf_epochs=args.mf_epochs,
                    lr=args.mf_lr, reg=args.mf_reg,
                    init_mode=mode, hho_epoch=args.hho_epoch, hho_pop=args.hho_pop,
                )
                print(
                    f"  MAE={row['mae']:.4f} RMSE={row['rmse']:.4f}  "
                    f"WCSS meta={row['wcss_after_meta']:.1f} km={row['wcss_after_kmeans']:.1f}  "
                    f"cluster_hit={row['cluster_mean_pct']:.1f}%  "
                    f"global_fb={row['global_fallback_pct']:.1f}%",
                    flush=True,
                )
                results.append(row)

    df = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n-> {OUT}")
    best = df.loc[df["mae"].idxmin()]
    print(
        f"\nBest MAE={best['mae']:.4f}  (k={int(best['k'])}, L={int(best['latent_dim'])}, "
        f"init={best['init_mode']})"
    )


if __name__ == "__main__":
    main()
