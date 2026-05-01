"""
Generate predicted ratings from precomputed assignments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.data_loader import load_dataset
from models.cluster_manager import ClusterManager


def _load_assignments_and_gray(
    assignment_path: str,
    gray_mask_path: str | None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    if assignment_path.endswith(".npy") and os.path.exists(assignment_path):
        assignments = np.load(assignment_path)
        gray_mask = np.load(gray_mask_path) if gray_mask_path and os.path.exists(gray_mask_path) else None
        return assignments, gray_mask

    # CSV path support: assignment_summary.csv
    if assignment_path.endswith(".csv") and os.path.exists(assignment_path):
        rows = np.genfromtxt(assignment_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        assignments = np.asarray(rows["cluster_id"], dtype=np.int32)
        if "is_gray_sheep" in rows.dtype.names:
            gray = np.asarray(rows["is_gray_sheep"], dtype=np.int32).astype(bool)
        else:
            gray = None
        if gray_mask_path and os.path.exists(gray_mask_path):
            gray = np.load(gray_mask_path).astype(bool)
        return assignments, gray

    # Last chance: if extension is omitted, try npy first then csv parser.
    if os.path.exists(assignment_path):
        try:
            assignments = np.load(assignment_path)
            gray_mask = np.load(gray_mask_path) if gray_mask_path and os.path.exists(gray_mask_path) else None
            return assignments, gray_mask
        except Exception:
            rows = np.genfromtxt(assignment_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if rows.dtype.names and "cluster_id" in rows.dtype.names:
                assignments = np.asarray(rows["cluster_id"], dtype=np.int32)
                gray = (
                    np.asarray(rows["is_gray_sheep"], dtype=np.int32).astype(bool)
                    if "is_gray_sheep" in rows.dtype.names
                    else None
                )
                if gray_mask_path and os.path.exists(gray_mask_path):
                    gray = np.load(gray_mask_path).astype(bool)
                return assignments, gray

    raise FileNotFoundError(f"Assignment input not found: {assignment_path}")


def _user_mean(user_vec: np.ndarray) -> float:
    rated = user_vec[user_vec != 0]
    return float(rated.mean()) if rated.size > 0 else 3.0


def _pearson_similarity(user_a: np.ndarray, user_b: np.ndarray) -> float:
    mask = (user_a != 0) & (user_b != 0)
    if np.count_nonzero(mask) < 2:
        return 0.0
    a = user_a[mask].astype(np.float64)
    b = user_b[mask].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _predict_with_neighbors(
    train_matrix: np.ndarray,
    user_id: int,
    item_id: int,
    neighbor_ids: np.ndarray,
    top_k: int,
) -> float:
    u_vec = train_matrix[user_id]
    u_mean = _user_mean(u_vec)
    if neighbor_ids.size == 0:
        return float(np.clip(u_mean, 1.0, 5.0))

    rated_neighbors = neighbor_ids[train_matrix[neighbor_ids, item_id] != 0]
    if rated_neighbors.size == 0:
        return float(np.clip(u_mean, 1.0, 5.0))

    sims = np.array(
        [_pearson_similarity(u_vec, train_matrix[v]) for v in rated_neighbors],
        dtype=np.float64,
    )
    if sims.size == 0:
        return float(np.clip(u_mean, 1.0, 5.0))

    k = min(top_k, sims.size)
    idx = np.argpartition(np.abs(sims), -k)[-k:]
    idx = idx[np.argsort(np.abs(sims[idx]))[::-1]]
    nbr = rated_neighbors[idx]
    nbr_sims = sims[idx]
    nbr_r = train_matrix[nbr, item_id].astype(np.float64)
    nbr_m = np.array([_user_mean(train_matrix[v]) for v in nbr], dtype=np.float64)
    numer = float(np.dot(nbr_sims, nbr_r - nbr_m))
    denom = float(np.abs(nbr_sims).sum())
    if denom <= 1e-12:
        return float(np.clip(u_mean, 1.0, 5.0))
    return float(np.clip(u_mean + numer / denom, 1.0, 5.0))


def _predict_fallback(train_matrix: np.ndarray, user_id: int, item_id: int) -> float:
    item_vals = train_matrix[:, item_id]
    item_vals = item_vals[item_vals != 0]
    if item_vals.size > 0:
        return float(np.clip(item_vals.mean(), 1.0, 5.0))
    return float(np.clip(_user_mean(train_matrix[user_id]), 1.0, 5.0))


def _topn_for_user(
    train_matrix: np.ndarray,
    user_id: int,
    predictor,
    n_reco: int,
) -> np.ndarray:
    unseen = np.where(train_matrix[user_id] == 0)[0]
    if unseen.size == 0:
        return np.array([], dtype=np.int32)
    preds = [(int(i), float(predictor(user_id, int(i)))) for i in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)
    return np.array([i for i, _ in preds[:n_reco]], dtype=np.int32)


def run(
    dataset: str,
    ratings_path: str,
    assignment_path: str,
    gray_mask_path: str | None,
    out_dir: str,
    K: int,
    top_k: int,
    n_reco: int,
    test_ratio: float,
    seed: int,
    normalize: str,
    max_users: int | None = None,
) -> Dict[str, str]:
    train_matrix, test_ratings, _ = load_dataset(
        ratings_path=ratings_path,
        test_ratio=test_ratio,
        seed=seed,
        normalize=normalize,
    )
    assignments, gray_mask = _load_assignments_and_gray(assignment_path, gray_mask_path)

    cm = ClusterManager(K=K)
    cm.load_assignments(assignments=assignments, gray_sheep_mask=gray_mask)
    selected_users = set(range(train_matrix.shape[0]))
    if max_users is not None and max_users > 0:
        selected_users = set(range(min(max_users, train_matrix.shape[0])))

    os.makedirs(out_dir, exist_ok=True)
    mixed_rows = []
    gray_same_rows = []
    gray_fallback_rows = []
    rec_white: Dict[int, np.ndarray] = {}
    rec_gray_same: Dict[int, np.ndarray] = {}
    rec_gray_fallback: Dict[int, np.ndarray] = {}

    def white_predict(uid: int, iid: int) -> float:
        cid = cm.get_user_cluster(uid)
        pool = cm.get_white_members(cid)
        pool = pool[pool != uid]
        return _predict_with_neighbors(train_matrix, uid, iid, pool, top_k=top_k)

    def gray_same_predict(uid: int, iid: int) -> float:
        cid = cm.get_user_cluster(uid)
        pool = cm.get_cluster_members(cid)
        pool = pool[pool != uid]
        return _predict_with_neighbors(train_matrix, uid, iid, pool, top_k=top_k)

    def gray_fallback_predict(uid: int, iid: int) -> float:
        return _predict_fallback(train_matrix, uid, iid)

    for row in test_ratings:
        uid, iid, true_r = int(row[0]), int(row[1]), float(row[2])
        if uid not in selected_users:
            continue
        is_gray = bool(cm.gray_mask[uid])
        if is_gray:
            pred_same = gray_same_predict(uid, iid)
            pred_fb = gray_fallback_predict(uid, iid)
            gray_same_rows.append([uid, iid, true_r, pred_same])
            gray_fallback_rows.append([uid, iid, true_r, pred_fb])
            mixed_rows.append([uid, iid, true_r, pred_same, 1])
        else:
            pred = white_predict(uid, iid)
            mixed_rows.append([uid, iid, true_r, pred, 0])

    for uid in sorted(selected_users):
        if bool(cm.gray_mask[uid]):
            rec_gray_same[uid] = _topn_for_user(train_matrix, uid, gray_same_predict, n_reco)
            rec_gray_fallback[uid] = _topn_for_user(train_matrix, uid, gray_fallback_predict, n_reco)
        else:
            rec_white[uid] = _topn_for_user(train_matrix, uid, white_predict, n_reco)

    mixed_path = os.path.join(out_dir, "predicted_ratings.npy")
    same_path = os.path.join(out_dir, "predicted_ratings_gray_same.npy")
    fb_path = os.path.join(out_dir, "predicted_ratings_gray_fallback.npy")
    np.save(mixed_path, np.asarray(mixed_rows, dtype=np.float64))
    np.save(same_path, np.asarray(gray_same_rows, dtype=np.float64))
    np.save(fb_path, np.asarray(gray_fallback_rows, dtype=np.float64))
    np.save(os.path.join(out_dir, "recommendations_white.npy"), rec_white, allow_pickle=True)
    np.save(os.path.join(out_dir, "recommendations_gray_same.npy"), rec_gray_same, allow_pickle=True)
    np.save(os.path.join(out_dir, "recommendations_gray_fallback.npy"), rec_gray_fallback, allow_pickle=True)

    manifest = {
        "dataset": dataset,
        "assignment_path": assignment_path,
        "gray_mask_path": gray_mask_path,
        "mixed_predictions": mixed_path,
        "gray_same_predictions": same_path,
        "gray_fallback_predictions": fb_path,
        "top_k": top_k,
        "n_reco": n_reco,
        "seed": seed,
        "max_users": max_users,
    }
    manifest_path = os.path.join(out_dir, "recommendation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _default_ratings_path(dataset: str) -> str:
    if dataset == "1m":
        return str(_PROJECT_ROOT / "data" / "ml-1m" / "ratings.dat")
    return str(_PROJECT_ROOT / "data" / "ml-100k" / "u.data")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions from assignments")
    p.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    p.add_argument("--ratings-path", default=None)
    p.add_argument("--assignment-path", required=True)
    p.add_argument("--gray-mask-path", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--n-reco", type=int, default=10)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", choices=["none", "user-mean"], default="none")
    p.add_argument("--max-users", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ratings_path = args.ratings_path or _default_ratings_path(args.dataset)
    result = run(
        dataset=args.dataset,
        ratings_path=ratings_path,
        assignment_path=args.assignment_path,
        gray_mask_path=args.gray_mask_path,
        out_dir=args.out_dir,
        K=args.K,
        top_k=args.top_k,
        n_reco=args.n_reco,
        test_ratio=args.test_ratio,
        seed=args.seed,
        normalize=args.normalize,
        max_users=args.max_users,
    )
    print(json.dumps(result, indent=2))
