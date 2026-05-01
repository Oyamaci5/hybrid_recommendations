import time

import numpy as np

from core.metrics import (
    compute_binary_accuracy,
    compute_metrics,
    compute_topn_metrics,
)


def run_cluster_average(train, test, assignments, gray_mask,
                        memberships,
                        n_items, algo_label, top_n: int = 10,
                        relevance_threshold: float = 4.0):
    t0 = time.time()
    n_clusters = int(assignments.max()) + 1
    use_soft = (
        memberships is not None
        and memberships.shape[0] >= len(assignments)
        and memberships.shape[1] == n_clusters
    )

    cluster_item_means = np.zeros((n_clusters, n_items), dtype=np.float32)
    cluster_item_counts = np.zeros((n_clusters, n_items), dtype=np.int32)

    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        cid = int(assignments[u])
        cluster_item_means[cid, i] += r
        cluster_item_counts[cid, i] += 1

    global_mean = float(train[:, 2].mean())
    for cid in range(n_clusters):
        mask = cluster_item_counts[cid] > 0
        cluster_item_means[cid, mask] /= cluster_item_counts[cid, mask]
        cluster_item_means[cid, ~mask] = global_mean

    true_vals, pred_vals = [], []
    gray_true, gray_pred = [], []
    eval_rows = []

    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if use_soft:
            w = memberships[u]
            pred = float(np.clip(np.dot(w, cluster_item_means[:, i]), 1.0, 5.0))
        else:
            cid = int(assignments[u])
            pred = float(np.clip(cluster_item_means[cid, i], 1.0, 5.0))

        if gray_mask[u]:
            gray_true.append(r)
            gray_pred.append(pred)
        else:
            true_vals.append(r)
            pred_vals.append(pred)
        eval_rows.append((u, i, r, pred))

    all_true = true_vals + gray_true
    all_pred = pred_vals + gray_pred
    mae, rmse = compute_metrics(all_true, all_pred)
    accuracy = compute_binary_accuracy(
        all_true, all_pred, threshold=relevance_threshold
    )
    precision, recall, f1, ndcg = compute_topn_metrics(
        np.array(eval_rows, dtype=np.float32),
        top_n=top_n,
        threshold=relevance_threshold,
    )

    gray_mae, gray_rmse = float('nan'), float('nan')
    if gray_true:
        gray_errors = np.array(gray_true) - np.array(gray_pred)
        gray_mae = float(np.mean(np.abs(gray_errors)))
        gray_rmse = float(np.sqrt(np.mean(gray_errors ** 2)))

    white_mae, white_rmse = compute_metrics(true_vals, pred_vals)

    elapsed = time.time() - t0
    print(f"  [{algo_label} | ClusterAvg] MAE={mae:.4f} "
          f"RMSE={rmse:.4f} | Gray MAE={gray_mae:.4f} ({elapsed:.1f}s)")

    return {
        'scenario': 'cluster_avg',
        'algo_label': algo_label,
        'mae': mae,
        'rmse': rmse,
        'gray_mae': gray_mae,
        'gray_rmse': gray_rmse,
        'white_mae': white_mae,
        'white_rmse': white_rmse,
        'n_clusters': n_clusters,
        'n_train': len(train),
        'n_test': len(test),
        'time_seconds': elapsed,
        'cluster_mae_std': float('nan'),
        'cluster_mae_mean': float('nan'),
        'cluster_mae_min': float('nan'),
        'cluster_mae_max': float('nan'),
        'accuracy': accuracy,
        'precision_at_10': precision,
        'recall_at_10': recall,
        'f1_at_10': f1,
        'ndcg_at_10': ndcg,
    }


def run_cluster_knn(train, test, assignments, gray_mask, memberships,
                    n_items, algo_label,
                    similarity: str = 'pearson',
                    min_common: int = 3,
                    k_neighbors: int = 30,
                    top_n: int = 10,
                    relevance_threshold: float = 4.0):
    t0 = time.time()
    k_neighbors = max(1, int(k_neighbors))

    global_mean = float(train[:, 2].mean())
    user_ratings = {}
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if u not in user_ratings:
            user_ratings[u] = {}
        user_ratings[u][i] = r
    user_means = {u: float(np.mean(list(d.values()))) for u, d in user_ratings.items()}

    cluster_users = {}
    for u in range(len(assignments)):
        cid = int(assignments[u])
        cluster_users.setdefault(cid, []).append(u)

    def pearson_sim(u, v, min_common=3):
        common = list(set(user_ratings.get(u, {}).keys()) & set(user_ratings.get(v, {}).keys()))
        if len(common) < min_common:
            return 0.0
        u_c = np.array([user_ratings[u][i] - user_means[u] for i in common], dtype=np.float32)
        v_c = np.array([user_ratings[v][i] - user_means[v] for i in common], dtype=np.float32)
        norm_u = np.sqrt((u_c ** 2).sum())
        norm_v = np.sqrt((v_c ** 2).sum())
        if norm_u < 1e-8 or norm_v < 1e-8:
            return 0.0
        return float(np.clip(np.dot(u_c, v_c) / (norm_u * norm_v), -1.0, 1.0))

    def cosine_sim(u, v, min_common=3):
        common = list(set(user_ratings.get(u, {}).keys()) & set(user_ratings.get(v, {}).keys()))
        if len(common) < min_common:
            return 0.0
        u_r = np.array([user_ratings[u][i] for i in common], dtype=np.float32)
        v_r = np.array([user_ratings[v][i] for i in common], dtype=np.float32)
        denom = np.sqrt((u_r ** 2).sum()) * np.sqrt((v_r ** 2).sum())
        if denom < 1e-8:
            return 0.0
        return float(np.clip(np.dot(u_r, v_r) / denom, 0.0, 1.0))

    use_soft = (
        memberships is not None
        and memberships.shape[0] >= len(assignments)
        and memberships.shape[1] >= (int(assignments.max()) + 1)
    )

    def _predict_for_cluster(u, i, cid, similarity='pearson', min_common=3):
        sims = []
        for v in cluster_users.get(cid, []):
            if v == u or i not in user_ratings.get(v, {}):
                continue
            s = pearson_sim(u, v, min_common) if similarity == 'pearson' else cosine_sim(u, v, min_common)
            if abs(s) > 0.0:
                sims.append((s, v))
        if not sims:
            return float(user_means.get(u, global_mean))
        sims.sort(key=lambda x: -abs(x[0]))
        top_k = sims[:k_neighbors]
        if similarity == 'pearson':
            weighted = [(max(0.0, float(s)), v) for s, v in top_k]
            den = sum(w for w, _ in weighted)
            if den < 1e-8:
                return float(user_means.get(u, global_mean))
            num = sum(w * float(user_ratings[v][i]) for w, v in weighted)
            return float(np.clip(num / den, 1.0, 5.0))
        num = sum(s * (user_ratings[v][i] - user_means.get(v, global_mean)) for s, v in top_k)
        den = sum(abs(s) for s, _ in top_k)
        if den < 1e-8:
            return float(user_means.get(u, global_mean))
        return float(np.clip(user_means.get(u, global_mean) + num / den, 1.0, 5.0))

    def predict(u, i, similarity='pearson', min_common=3):
        if not use_soft:
            return _predict_for_cluster(u, i, int(assignments[u]), similarity=similarity, min_common=min_common)
        pred = 0.0
        for cid, w in enumerate(memberships[u]):
            if w <= 0:
                continue
            pred += float(w) * _predict_for_cluster(u, i, cid, similarity=similarity, min_common=min_common)
        return float(np.clip(pred, 1.0, 5.0))

    true_vals, pred_vals = [], []
    gray_true, gray_pred = [], []
    eval_rows = []
    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        pr = predict(u, i, similarity=similarity, min_common=min_common)
        if gray_mask[u]:
            gray_true.append(r)
            gray_pred.append(pr)
        else:
            true_vals.append(r)
            pred_vals.append(pr)
        eval_rows.append((u, i, r, pr))

    all_true = true_vals + gray_true
    all_pred = pred_vals + gray_pred
    mae, rmse = compute_metrics(all_true, all_pred)
    gray_mae, gray_rmse = float('nan'), float('nan')
    if gray_true:
        gray_errors = np.array(gray_true) - np.array(gray_pred)
        gray_mae = float(np.mean(np.abs(gray_errors)))
        gray_rmse = float(np.sqrt(np.mean(gray_errors ** 2)))

    white_mae, white_rmse = compute_metrics(true_vals, pred_vals)
    precision, recall, f1, ndcg = compute_topn_metrics(
        np.array(eval_rows, dtype=np.float32),
        top_n=top_n,
        threshold=relevance_threshold,
    )

    elapsed = time.time() - t0
    print(f"  [{algo_label} | ClusterKNN|{similarity}|k={k_neighbors}] MAE={mae:.4f} "
          f"RMSE={rmse:.4f} | Gray MAE={gray_mae:.4f} ({elapsed:.1f}s)")

    return {
        'scenario': 'cluster_knn',
        'algo_label': algo_label,
        'mae': mae,
        'rmse': rmse,
        'gray_mae': gray_mae,
        'gray_rmse': gray_rmse,
        'white_mae': white_mae,
        'white_rmse': white_rmse,
        'n_clusters': int(assignments.max()) + 1,
        'n_train': len(train),
        'n_test': len(test),
        'time_seconds': elapsed,
        'cluster_mae_std': float('nan'),
        'cluster_mae_mean': float('nan'),
        'cluster_mae_min': float('nan'),
        'cluster_mae_max': float('nan'),
        'accuracy': float('nan'),
        'precision_at_10': precision,
        'recall_at_10': recall,
        'f1_at_10': f1,
        'ndcg_at_10': ndcg,
        'similarity': similarity,
        'k_neighbors': k_neighbors,
    }
