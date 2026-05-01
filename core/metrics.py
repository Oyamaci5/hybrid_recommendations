"""
Evaluation metrics for recommender systems.
"""

import numpy as np
from typing import Tuple


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True ratings
        y_pred: Predicted ratings
        
    Returns:
        RMSE value
    """
    if len(y_true) == 0:
        return 0.0
    
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True ratings
        y_pred: Predicted ratings
        
    Returns:
        MAE value
    """
    if len(y_true) == 0:
        return 0.0
    
    return np.mean(np.abs(y_true - y_pred))


def evaluate_predictions(
    true_ratings: np.ndarray,
    pred_ratings: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate predictions using both RMSE and MAE.
    
    Args:
        true_ratings: True ratings array
        pred_ratings: Predicted ratings array
        
    Returns:
        Tuple of (RMSE, MAE)
    """
    rmse_value = rmse(true_ratings, pred_ratings)
    mae_value = mae(true_ratings, pred_ratings)
    return rmse_value, mae_value


def pearson_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Compute Pearson-based distance between two user vectors.

    Returns:
        0.0 -> highly similar trend
        2.0 -> highly dissimilar/opposite trend
    """
    common_indices = np.where((vector_a > 0) & (vector_b > 0))[0]
    if len(common_indices) < 2:
        return 2.0

    a_vals = vector_a[common_indices]
    b_vals = vector_b[common_indices]

    a_diff = a_vals - np.mean(a_vals)
    b_diff = b_vals - np.mean(b_vals)

    numerator = np.sum(a_diff * b_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2) * np.sum(b_diff ** 2))
    if denominator == 0:
        return 2.0

    correlation = numerator / denominator
    return float(1.0 - correlation)


def pearson_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Pearson Correlation Coefficient — 1.0 tam benzer, -1.0 tam zıt.

    Sadece her iki vektörde de sıfır olmayan ortak itemler kullanılır.
    Ortak item < 2 ise 0.0 döner.
    """
    return float(1.0 - pearson_distance(vector_a, vector_b))


# ---------------------------------------------------------------------------
# Küme tabanlı CF tahmin ve değerlendirme
# ---------------------------------------------------------------------------

def _user_mean(user_vec: np.ndarray) -> float:
    """Sıfır olmayan ratinglerin ortalaması."""
    rated = user_vec[user_vec != 0]
    return float(rated.mean()) if len(rated) > 0 else 0.0


def predict_rating(
    active_user_vec: np.ndarray,
    cluster_labels: np.ndarray,
    train_matrix: np.ndarray,
    centroids: np.ndarray,
    item_id: int,
    top_k: int = 30,
    distance_metric: str = "pearson",
) -> float:
    """
    Küme tabanlı CF ile tek bir (kullanıcı, item) rating tahmini.

    Formül: r̂(u,i) = r̄_u + Σ[sim(u,v)*(r_vi - r̄_v)] / Σ|sim(u,v)|

    Adımlar:
      1. active_user_vec'i train_matrix'te bul → küme ID'sini al
      2. Küme üyelerinden item_id'yi oylayan komşuları seç
      3. PCC benzerliği hesapla, top-K'yı al
      4. Ağırlıklı ortalama

    Parameters
    ----------
    active_user_vec : np.ndarray, shape (n_items,)
    cluster_labels  : np.ndarray, shape (n_users,)
    train_matrix    : np.ndarray, shape (n_users, n_items)
    centroids       : np.ndarray, shape (K, n_items)
    item_id         : int
    top_k           : int

    Returns
    -------
    float — tahmin edilen rating ∈ [1.0, 5.0]
    """
    u_mean = _user_mean(active_user_vec)

    # Küme bul — train_matrix'te tam eşleşme ara
    matches = np.where(np.all(train_matrix == active_user_vec, axis=1))[0]
    if len(matches) > 0:
        cluster_id = int(cluster_labels[matches[0]])
    else:
        # Yoksa en yakın centroid'e ata (PCC veya Euclidean)
        if distance_metric == "euclidean":
            idx = np.where(active_user_vec != 0)[0]
            if len(idx) == 0:
                dists = np.full(len(centroids), fill_value=np.inf)
            else:
                dists = np.array(
                    [np.linalg.norm(active_user_vec[idx] - c[idx]) for c in centroids]
                )
        else:
            dists = np.array([pearson_distance(active_user_vec, c) for c in centroids])
        cluster_id = int(np.argmin(dists))

    members = np.where(cluster_labels == cluster_id)[0]
    if len(members) == 0:
        return float(np.clip(u_mean, 1.0, 5.0))

    # Sadece item_id'yi oylayan üyeler
    rated_mask = train_matrix[members, item_id] != 0
    candidates = members[rated_mask]
    if len(candidates) == 0:
        return float(np.clip(u_mean, 1.0, 5.0))

    sims = np.array([
        pearson_similarity(active_user_vec, train_matrix[v]) for v in candidates
    ])

    k = min(top_k, len(candidates))
    top_idx = np.argpartition(np.abs(sims), -k)[-k:]
    top_idx = top_idx[np.argsort(np.abs(sims[top_idx]))[::-1]]

    nbr_ids  = candidates[top_idx]
    nbr_sims = sims[top_idx]

    v_ratings = train_matrix[nbr_ids, item_id].astype(float)
    v_means   = np.array([_user_mean(train_matrix[v]) for v in nbr_ids])

    numer = np.dot(nbr_sims, v_ratings - v_means)
    denom = np.abs(nbr_sims).sum()

    if denom < 1e-10:
        return float(np.clip(u_mean, 1.0, 5.0))

    return float(np.clip(u_mean + numer / denom, 1.0, 5.0))


def precision_at_n(
    recommended: list[int],
    relevant: set[int],
    N: int,
) -> float:
    """Precision@N = |relevant ∩ recommended[:N]| / N"""
    hits = len(set(recommended[:N]) & relevant)
    return hits / N if N > 0 else 0.0


def recall_at_n(
    recommended: list[int],
    relevant: set[int],
    N: int,
) -> float:
    """Recall@N = |relevant ∩ recommended[:N]| / |relevant|"""
    if not relevant:
        return 0.0
    hits = len(set(recommended[:N]) & relevant)
    return hits / len(relevant)


def evaluate_cf(
    test_ratings: np.ndarray,
    train_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    top_k: int = 30,
    N: int = 10,
    relevance_threshold: float = 3.5,
    distance_metric: str = "pearson",
) -> dict:
    """
    Test seti üzerinde MAE, RMSE, Precision@N, Recall@N hesaplar.

    Parameters
    ----------
    test_ratings : np.ndarray, shape (n_test, 3) — [user_id, item_id, rating]
    train_matrix : np.ndarray, shape (n_users, n_items)
    cluster_labels : np.ndarray, shape (n_users,)
    centroids : np.ndarray, shape (K, n_items)
    top_k : int — CF komşu sayısı
    N : int — Precision@N / Recall@N için liste uzunluğu
    relevance_threshold : float

    Returns
    -------
    dict — MAE, RMSE, Precision@N, Recall@N
    """
    preds, actuals = [], []
    for row in test_ratings:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        pred = predict_rating(train_matrix[u], cluster_labels, train_matrix,
                              centroids, i, top_k, distance_metric=distance_metric)
        preds.append(pred)
        actuals.append(r)

    p_arr = np.asarray(preds)
    a_arr = np.asarray(actuals)
    mae_val  = float(np.abs(p_arr - a_arr).mean())
    rmse_val = float(np.sqrt(((p_arr - a_arr) ** 2).mean()))

    # Precision@N / Recall@N — kullanıcı bazında grupla
    user_test: dict[int, dict[int, float]] = {}
    for row in test_ratings:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        user_test.setdefault(u, {})[i] = r

    precisions, recalls = [], []
    for u, item_ratings in user_test.items():
        relevant = {i for i, r in item_ratings.items() if r >= relevance_threshold}
        if not relevant:
            continue
        unseen = np.where(train_matrix[u] == 0)[0]
        item_preds = [
            (int(i), predict_rating(train_matrix[u], cluster_labels,
                                    train_matrix, centroids, int(i), top_k,
                                    distance_metric=distance_metric))
            for i in unseen
        ]
        item_preds.sort(key=lambda x: x[1], reverse=True)
        recs = [i for i, _ in item_preds[:N]]
        precisions.append(precision_at_n(recs, relevant, N))
        recalls.append(recall_at_n(recs, relevant, N))

    return {
        "MAE":            round(mae_val, 4),
        "RMSE":           round(rmse_val, 4),
        f"Precision@{N}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"Recall@{N}":    round(float(np.mean(recalls))    if recalls    else 0.0, 4),
    }


def evaluate_by_group(
    test_ratings,
    train_matrix,
    cluster_labels,
    centroids,
    gray_sheep_mask,
    top_k=30,
    N=10,
    relevance_threshold=3.5,
) -> dict:
    """
    test_ratings'i gray_sheep_mask kullanarak 3 gruba bol:
    total / white / gray_sheep
    Her grup icin evaluate_cf() cagir ve sonuclari dondur.
    gray_sheep_mask: shape (n_users,) bool array,
                     True = gray sheep kullanici
    """
    gray_sheep_mask = np.asarray(gray_sheep_mask).astype(bool)
    white_users = set(np.where(~gray_sheep_mask)[0])
    gray_users = set(np.where(gray_sheep_mask)[0])

    mask_white = np.array([int(row[0]) in white_users for row in test_ratings], dtype=bool)
    mask_gray = np.array([int(row[0]) in gray_users for row in test_ratings], dtype=bool)

    results = {}

    results["total"] = evaluate_cf(
        test_ratings,
        train_matrix,
        cluster_labels,
        centroids,
        top_k,
        N,
        relevance_threshold,
    )

    if mask_white.any():
        results["white"] = evaluate_cf(
            test_ratings[mask_white],
            train_matrix,
            cluster_labels,
            centroids,
            top_k,
            N,
            relevance_threshold,
        )
    else:
        results["white"] = {}

    if mask_gray.any():
        results["gray_sheep"] = evaluate_cf(
            test_ratings[mask_gray],
            train_matrix,
            cluster_labels,
            centroids,
            top_k,
            N,
            relevance_threshold,
        )
    else:
        results["gray_sheep"] = {}

    return results


def compute_binary_accuracy(true_vals, pred_vals, threshold=3.5):
    if not true_vals:
        return float("nan")
    true_bin = np.array(true_vals, dtype=np.float32) >= float(threshold)
    pred_bin = np.array(pred_vals, dtype=np.float32) >= float(threshold)
    return float(np.mean(true_bin == pred_bin))


def compute_topn_metrics(eval_rows: np.ndarray, top_n: int = 10, threshold: float = 4.0):
    if eval_rows is None or len(eval_rows) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    by_user = {}
    for row in eval_rows:
        u = int(row[0])
        i = int(row[1])
        r_true = float(row[2])
        r_pred = float(row[3])
        by_user.setdefault(u, []).append((i, r_true, r_pred))
    precisions, recalls, f1s, ndcgs = [], [], [], []
    k = max(1, int(top_n))
    for _, items in by_user.items():
        relevant = {i for i, r_true, _ in items if r_true >= threshold}
        if not relevant:
            continue
        ranked = sorted(items, key=lambda x: x[2], reverse=True)[:k]
        top_items = [i for i, _, _ in ranked]
        hits = len(set(top_items) & relevant)
        p = hits / k
        r = hits / len(relevant) if relevant else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        dcg = 0.0
        for rank_idx, item_id in enumerate(top_items):
            rel = 1.0 if item_id in relevant else 0.0
            dcg += (2.0**rel - 1.0) / np.log2(rank_idx + 2.0)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(ndcg)
    if not precisions:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        float(np.mean(ndcgs)),
    )


def compute_metrics(true_list, pred_list):
    if not true_list:
        return float("nan"), float("nan")
    true_arr = np.array(true_list, dtype=np.float32)
    pred_arr = np.array(pred_list, dtype=np.float32)
    errors = true_arr - pred_arr
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors**2)))

