from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from core.data_loader import load_dataset
from core.metrics import evaluate_cf

K_CANDIDATES = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]


def _default_ratings_path(dataset: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset == "1m":
        return os.path.join(root, "data", "ml-1m", "ratings.dat")
    return os.path.join(root, "data", "ml-100k", "u.data")


def _print_table(rows: List[Tuple[int, float, float, float | None, float]]) -> None:
    headers = ["K", "WCSS", "Silhouette", "Delta_WCSS", "MAE"]
    w = [len(h) for h in headers]
    rendered = []
    for k, wcss, sil, delta, mae_val in rows:
        delta_str = "-" if delta is None else f"{delta:.2f}"
        row = [str(k), f"{wcss:.2f}", f"{sil:.6f}", delta_str, f"{mae_val:.6f}"]
        rendered.append(row)
        for i, cell in enumerate(row):
            w[i] = max(w[i], len(cell))
    sep = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    print(sep)
    print("| " + " | ".join(headers[i].ljust(w[i]) for i in range(len(headers))) + " |")
    print(sep)
    for row in rendered:
        print("| " + " | ".join(row[i].ljust(w[i]) for i in range(len(row))) + " |")
    print(sep)


def main() -> int:
    p = argparse.ArgumentParser(description="Find optimal K with Elbow + Silhouette")
    p.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    p.add_argument("--ratings-path", default=None, help="Optional custom ratings file path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", choices=["none", "user-mean"], default="none")
    p.add_argument("--test-ratio", type=float, default=0.2)
    args = p.parse_args()

    ratings_path = args.ratings_path or _default_ratings_path(args.dataset)
    train_matrix, test_ratings, _ = load_dataset(
        ratings_path=ratings_path,
        test_ratio=args.test_ratio,
        seed=args.seed,
        normalize=args.normalize,
    )
    rows: List[Tuple[int, float, float, float | None, float]] = []
    deltas: List[Tuple[int, float]] = []
    sils: List[Tuple[int, float]] = []
    maes: List[Tuple[int, float]] = []
    prev_wcss = None
    sample_size = min(200, train_matrix.shape[0])
    unique_test_users = np.unique(test_ratings[:, 0].astype(int))
    sampled_users = set(unique_test_users[: min(100, unique_test_users.size)])
    sampled_test = np.asarray([row for row in test_ratings if int(row[0]) in sampled_users], dtype=np.float64)
    for k in K_CANDIDATES:
        model = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=args.seed, max_iter=300)
        labels = model.fit_predict(train_matrix)
        wcss = float(model.inertia_)
        sil = float(
            silhouette_score(train_matrix, labels, sample_size=sample_size, random_state=args.seed)
        )
        metrics = evaluate_cf(
            test_ratings=sampled_test,
            train_matrix=train_matrix,
            cluster_labels=labels,
            centroids=model.cluster_centers_,
            top_k=30,
            N=10,
            relevance_threshold=3.5,
        )
        mae_val = float(metrics["MAE"])
        delta = None
        if prev_wcss is not None:
            delta = prev_wcss - wcss
            deltas.append((k, float(delta)))
        sils.append((k, sil))
        maes.append((k, mae_val))
        rows.append((k, wcss, sil, delta, mae_val))
        prev_wcss = wcss
    _print_table(rows)
    elbow_k = max(deltas, key=lambda x: x[1])[0] if deltas else K_CANDIDATES[0]
    silhouette_k = max(sils, key=lambda x: x[1])[0]
    optimal_k = min(maes, key=lambda x: x[1])[0]
    suggested_k = silhouette_k if silhouette_k == elbow_k else int(round((elbow_k + silhouette_k) / 2))
    print(f"Elbow K: {elbow_k}")
    print(f"Silhouette K: {silhouette_k}")
    print(f"Optimal K (MAE): {optimal_k}")
    print(f"Elbow K: {elbow_k}, Silhouette K: {silhouette_k} -> Onerilen K: {suggested_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
