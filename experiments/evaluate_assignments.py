"""
Evaluate assignment-based recommendation outputs (workflow; not core.metrics).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.data_loader import load_dataset
from core.metrics import mae, rmse


def _precision_recall_at_n(
    rec_map: Dict[int, np.ndarray],
    test_ratings: np.ndarray,
    n_reco: int,
    relevance_threshold: float,
    allowed_users: Iterable[int] | None = None,
) -> tuple[float, float]:
    allowed = set(allowed_users) if allowed_users is not None else None
    user_rel: Dict[int, set] = {}
    for row in test_ratings:
        uid, iid, r = int(row[0]), int(row[1]), float(row[2])
        if allowed is not None and uid not in allowed:
            continue
        if r >= relevance_threshold:
            user_rel.setdefault(uid, set()).add(iid)

    precisions = []
    recalls = []
    for uid, rel in user_rel.items():
        if uid not in rec_map or len(rel) == 0:
            continue
        rec = list(rec_map[uid][:n_reco])
        hits = len(set(rec) & rel)
        precisions.append(hits / max(n_reco, 1))
        recalls.append(hits / len(rel))
    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))


def _metrics_from_array(pred_arr: np.ndarray) -> dict:
    if pred_arr.size == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "n_predictions": 0}
    true = pred_arr[:, 2]
    pred = pred_arr[:, 3]
    return {
        "MAE": round(float(mae(true, pred)), 4),
        "RMSE": round(float(rmse(true, pred)), 4),
        "n_predictions": int(len(pred_arr)),
    }


def run(
    ratings_path: str,
    prediction_dir: str,
    output_dir: str,
    test_ratio: float,
    seed: int,
    normalize: str,
    n_reco: int,
    relevance_threshold: float,
) -> dict:
    _, test_ratings, _ = load_dataset(
        ratings_path=ratings_path,
        test_ratio=test_ratio,
        seed=seed,
        normalize=normalize,
    )

    mixed = np.load(os.path.join(prediction_dir, "predicted_ratings.npy"))
    gray_same = np.load(os.path.join(prediction_dir, "predicted_ratings_gray_same.npy"))
    gray_fallback = np.load(os.path.join(prediction_dir, "predicted_ratings_gray_fallback.npy"))
    rec_white = np.load(os.path.join(prediction_dir, "recommendations_white.npy"), allow_pickle=True).item()
    rec_gray_same = np.load(os.path.join(prediction_dir, "recommendations_gray_same.npy"), allow_pickle=True).item()
    rec_gray_fallback = np.load(os.path.join(prediction_dir, "recommendations_gray_fallback.npy"), allow_pickle=True).item()

    white_users = set(int(u) for u in rec_white.keys())
    gray_users = set(int(u) for u in rec_gray_same.keys()) | set(int(u) for u in rec_gray_fallback.keys())

    out = {
        "mixed": _metrics_from_array(mixed),
        "gray_same_formula": _metrics_from_array(gray_same),
        "gray_fallback": _metrics_from_array(gray_fallback),
    }

    p_white, r_white = _precision_recall_at_n(
        rec_map=rec_white,
        test_ratings=test_ratings,
        n_reco=n_reco,
        relevance_threshold=relevance_threshold,
        allowed_users=white_users,
    )
    p_gray_same, r_gray_same = _precision_recall_at_n(
        rec_map=rec_gray_same,
        test_ratings=test_ratings,
        n_reco=n_reco,
        relevance_threshold=relevance_threshold,
        allowed_users=gray_users,
    )
    p_gray_fb, r_gray_fb = _precision_recall_at_n(
        rec_map=rec_gray_fallback,
        test_ratings=test_ratings,
        n_reco=n_reco,
        relevance_threshold=relevance_threshold,
        allowed_users=gray_users,
    )

    out["white"] = {"Precision@N": round(p_white, 4), "Recall@N": round(r_white, 4)}
    out["gray_same_formula"].update(
        {"Precision@N": round(p_gray_same, 4), "Recall@N": round(r_gray_same, 4)}
    )
    out["gray_fallback"].update(
        {"Precision@N": round(p_gray_fb, 4), "Recall@N": round(r_gray_fb, 4)}
    )

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "evaluation_summary.json")
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = ["segment,MAE,RMSE,Precision@N,Recall@N,n_predictions"]
    for seg, vals in out.items():
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

    return {"json": json_path, "csv": csv_path, "summary": out}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate assignment-based predictions")
    p.add_argument("--ratings-path", required=True)
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", choices=["none", "user-mean"], default="none")
    p.add_argument("--n-reco", type=int, default=10)
    p.add_argument("--relevance-threshold", type=float, default=3.5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(
        ratings_path=args.ratings_path,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        normalize=args.normalize,
        n_reco=args.n_reco,
        relevance_threshold=args.relevance_threshold,
    )
    print(json.dumps(result, indent=2))
