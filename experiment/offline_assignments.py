"""Offline-assignment pipeline (pipeline_mode='offline_assignments').

Önceden ``mealpy/generate_assignments.py`` ile üretilmiş kullanıcı
atama (.npy) dosyasını tüketen kanonik akış:

  assignment.npy + gray_mask.npy
        ↓
  ClusterManager (load_assignments)
        ↓
  CFRecommender (gray-aware)
        ↓
  predicted_ratings*.npy + recommendations_*.npy
        ↓
  Evaluator.from_prediction_artifacts → evaluation_summary.{json,csv}

Bu modül ``experiments/generate_recommendations.py`` + ``experiments/
evaluate_assignments.py`` paralel akışının kanonik karşılığıdır;
runner ``pipeline_mode='offline_assignments'`` ile çağırır.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from core.data_loader import load_dataset
from models.cluster_manager import ClusterManager
from recommender.cf_recommender import CFRecommender
from recommender.evaluator import Evaluator
from utils.config import Config


def _load_assignments_and_gray(
    assignment_path: str,
    gray_mask_path: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not assignment_path or not os.path.exists(assignment_path):
        raise FileNotFoundError(f"Assignment input not found: {assignment_path!r}")

    if assignment_path.endswith(".csv"):
        rows = np.genfromtxt(assignment_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        assignments = np.asarray(rows["cluster_id"], dtype=np.int32)
        gray = (
            np.asarray(rows["is_gray_sheep"], dtype=np.int32).astype(bool)
            if "is_gray_sheep" in (rows.dtype.names or ())
            else None
        )
    else:
        assignments = np.load(assignment_path)
        gray = None

    if gray_mask_path and os.path.exists(gray_mask_path):
        gray = np.load(gray_mask_path).astype(bool)

    return assignments, gray


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


def _default_ratings_path(dataset: str) -> str:
    root = Path(__file__).resolve().parent.parent
    if dataset.lower() in ("ml1m", "1m", "ml-1m"):
        return str(root / "data" / "ml-1m" / "ratings.dat")
    return str(root / "data" / "ml-100k" / "u.data")


def run(cfg: Config) -> dict:
    """Offline-assignment pipeline'ını çalıştırır.

    Returns
    -------
    dict — {"prediction_dir", "evaluation_dir", "summary"}.
    """
    oa = cfg.offline_assignments

    ratings_path = oa.ratings_path or _default_ratings_path(cfg.data.dataset)
    train_matrix, test_ratings, _ = load_dataset(
        ratings_path=ratings_path,
        test_ratio=cfg.data.test_size,
        seed=cfg.data.random_state,
        normalize=cfg.preprocess.normalization if cfg.preprocess.normalization in ("none", "user-mean") else "none",
    )

    assignments, gray_mask = _load_assignments_and_gray(oa.assignment_path, oa.gray_mask_path)

    cm = ClusterManager(K=int(cfg.clustering.n_clusters))
    cm.load_assignments(assignments=assignments, gray_sheep_mask=gray_mask)

    rec = CFRecommender(
        train_matrix=train_matrix,
        cluster_labels=cm.labels,
        gray_mask=cm.gray_mask,
        top_k=int(cfg.recommender.n_neighbors),
    )

    n_users = train_matrix.shape[0]
    if oa.max_users and oa.max_users > 0:
        selected = set(range(min(int(oa.max_users), n_users)))
    else:
        selected = set(range(n_users))

    mixed_rows: list[list[float]] = []
    gray_same_rows: list[list[float]] = []
    gray_fallback_rows: list[list[float]] = []
    rec_white: Dict[int, np.ndarray] = {}
    rec_gray_same: Dict[int, np.ndarray] = {}
    rec_gray_fallback: Dict[int, np.ndarray] = {}

    for row in test_ratings:
        uid, iid, true_r = int(row[0]), int(row[1]), float(row[2])
        if uid not in selected:
            continue
        if bool(cm.gray_mask[uid]):
            pred_same = rec.predict_gray_same(uid, iid)
            pred_fb = rec.predict_gray_fallback(uid, iid)
            gray_same_rows.append([uid, iid, true_r, pred_same])
            gray_fallback_rows.append([uid, iid, true_r, pred_fb])
            mixed_rows.append([uid, iid, true_r, pred_same, 1])
        else:
            pred = rec.predict_white(uid, iid)
            mixed_rows.append([uid, iid, true_r, pred, 0])

    for uid in sorted(selected):
        if bool(cm.gray_mask[uid]):
            rec_gray_same[uid] = _topn_for_user(train_matrix, uid, rec.predict_gray_same, oa.n_reco)
            rec_gray_fallback[uid] = _topn_for_user(train_matrix, uid, rec.predict_gray_fallback, oa.n_reco)
        else:
            rec_white[uid] = _topn_for_user(train_matrix, uid, rec.predict_white, oa.n_reco)

    pred_dir = oa.prediction_dir
    eval_dir = oa.evaluation_dir
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    np.save(os.path.join(pred_dir, "predicted_ratings.npy"), np.asarray(mixed_rows, dtype=np.float64))
    np.save(os.path.join(pred_dir, "predicted_ratings_gray_same.npy"), np.asarray(gray_same_rows, dtype=np.float64))
    np.save(os.path.join(pred_dir, "predicted_ratings_gray_fallback.npy"), np.asarray(gray_fallback_rows, dtype=np.float64))
    np.save(os.path.join(pred_dir, "recommendations_white.npy"), rec_white, allow_pickle=True)
    np.save(os.path.join(pred_dir, "recommendations_gray_same.npy"), rec_gray_same, allow_pickle=True)
    np.save(os.path.join(pred_dir, "recommendations_gray_fallback.npy"), rec_gray_fallback, allow_pickle=True)

    evaluator = Evaluator(
        relevance_threshold=oa.relevance_threshold,
        top_neighbors=cfg.recommender.n_neighbors,
        at_n=oa.n_reco,
    )
    summary = evaluator.from_prediction_artifacts(
        prediction_dir=pred_dir,
        test_ratings=test_ratings,
        n_reco=oa.n_reco,
        relevance_threshold=oa.relevance_threshold,
    )

    json_path = os.path.join(eval_dir, "evaluation_summary.json")
    csv_path = os.path.join(eval_dir, "evaluation_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = ["segment,MAE,RMSE,Precision@N,Recall@N,n_predictions"]
    for seg, vals in summary.items():
        lines.append(
            "{},{},{},{},{},{}".format(
                seg,
                vals.get("MAE", ""),
                vals.get("RMSE", ""),
                vals.get("Precision@N", ""),
                vals.get("Recall@N", ""),
                vals.get("n_predictions", ""),
            )
        )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "prediction_dir": pred_dir,
        "evaluation_dir": eval_dir,
        "summary": summary,
        "json": json_path,
        "csv": csv_path,
    }
