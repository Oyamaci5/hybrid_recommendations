"""Tek noktadan değerlendirme (MAE, RMSE, P@N, R@N vb.)."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable

import numpy as np

from core.metrics import compute_topn_metrics, evaluate_cf, evaluate_predictions, mae, predict_rating, rmse
from utils.config import EvaluationConfig
from utils.metrics import f1_at_n, ndcg_at_n, precision_at_n, recall_at_n


class Evaluator:
    def __init__(
        self,
        cfg: EvaluationConfig | None = None,
        relevance_threshold: float = 3.5,
        top_neighbors: int = 30,
        at_n: int = 10,
    ):
        if isinstance(cfg, EvaluationConfig):
            self.cfg = cfg
            self.relevance_threshold = float(relevance_threshold)
            self.top_neighbors = int(top_neighbors)
            self.at_n = int(cfg.at_n[0]) if cfg.at_n else int(at_n)
        else:
            self.cfg = EvaluationConfig(at_n=[int(at_n)])
            self.relevance_threshold = float(relevance_threshold)
            self.top_neighbors = int(top_neighbors)
            self.at_n = int(at_n)

    def rating_errors(
        self,
        test_ratings: np.ndarray,
        train_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
        distance_metric: str = "pearson",
    ) -> tuple[np.ndarray, np.ndarray]:
        preds, acts = [], []
        for row in test_ratings:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            pred = predict_rating(
                train_matrix[u],
                cluster_labels,
                train_matrix,
                centroids,
                i,
                top_k=self.top_neighbors,
                distance_metric=distance_metric,
            )
            preds.append(pred)
            acts.append(r)
        return np.asarray(acts), np.asarray(preds)

    def cluster_cf_summary(
        self,
        test_ratings: np.ndarray,
        train_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
        distance_metric: str = "pearson",
    ) -> dict[str, Any]:
        return evaluate_cf(
            test_ratings,
            train_matrix,
            cluster_labels,
            centroids,
            top_k=self.top_neighbors,
            N=self.at_n,
            relevance_threshold=self.relevance_threshold,
            distance_metric=distance_metric,
        )

    def from_pred_rows(self, eval_rows: np.ndarray, top_n: int | None = None) -> tuple[float, float, float, float]:
        """Satırlar [user, item, rating_true, rating_pred]."""
        tn = int(top_n) if top_n is not None else self.at_n
        return compute_topn_metrics(eval_rows, top_n=tn, threshold=self.relevance_threshold)

    @staticmethod
    def regression_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        return evaluate_predictions(np.asarray(y_true), np.asarray(y_pred))

    def evaluate_fold(
        self,
        recommender,
        R_test: np.ndarray,
        mask_test: np.ndarray,
        preprocessor,
        bias,
    ) -> dict:
        test_pairs = np.argwhere(mask_test > 0)
        y_true, y_pred = [], []
        for u, i in test_pairs:
            pred = recommender.predict_rating(int(u), int(i))
            pred_orig = preprocessor.inverse_transform_single(pred, bias, int(u), int(i))
            true_orig = preprocessor.inverse_transform_single(float(R_test[u, i]), bias, int(u), int(i))
            pred_orig = float(np.clip(pred_orig, 1.0, 5.0))
            true_orig = float(np.clip(true_orig, 1.0, 5.0))
            y_true.append(true_orig)
            y_pred.append(pred_orig)

        metrics: dict[str, float] = {
            "mae": mae(np.asarray(y_true), np.asarray(y_pred)) if y_true else 0.0,
            "rmse": rmse(np.asarray(y_true), np.asarray(y_pred)) if y_true else 0.0,
        }

        ns = self.cfg.at_n if isinstance(self.cfg.at_n, list) else [self.at_n]
        if len(test_pairs) == 0:
            for n in ns:
                metrics[f"precision@{n}"] = 0.0
                metrics[f"recall@{n}"] = 0.0
                metrics[f"f1@{n}"] = 0.0
                metrics[f"ndcg@{n}"] = 0.0
            return metrics

        test_users = np.unique(test_pairs[:, 0])
        for n in ns:
            p_list, r_list, f_list, ndcg_list = [], [], [], []
            for u in test_users:
                relevant = {
                    int(i)
                    for i in range(R_test.shape[1])
                    if mask_test[int(u), int(i)] > 0 and R_test[int(u), int(i)] >= self.relevance_threshold
                }
                if not relevant:
                    continue
                recs = recommender.recommend(int(u), top_n=int(n))
                rec_items = [int(item) for item, _ in recs[: int(n)]]
                p_list.append(precision_at_n(rec_items, relevant, int(n)))
                r_list.append(recall_at_n(rec_items, relevant, int(n)))
                f_list.append(f1_at_n(rec_items, relevant, int(n)))
                ndcg_list.append(ndcg_at_n(rec_items, relevant, int(n)))
            metrics[f"precision@{n}"] = float(np.mean(p_list)) if p_list else 0.0
            metrics[f"recall@{n}"] = float(np.mean(r_list)) if r_list else 0.0
            metrics[f"f1@{n}"] = float(np.mean(f_list)) if f_list else 0.0
            metrics[f"ndcg@{n}"] = float(np.mean(ndcg_list)) if ndcg_list else 0.0
        return metrics

    def aggregate(self, fold_results: list[dict]) -> dict:
        if not fold_results:
            return {}
        summary: dict[str, float] = {}
        keys = fold_results[0].keys()
        for k in keys:
            vals = [float(f[k]) for f in fold_results if k in f]
            summary[f"{k}_mean"] = float(np.mean(vals)) if vals else 0.0
            summary[f"{k}_std"] = float(np.std(vals)) if vals else 0.0
        return summary

    # ------------------------------------------------------------------
    # Offline-assignment artefakt değerlendirmesi
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_from_array(pred_arr: np.ndarray) -> dict:
        if pred_arr.size == 0:
            return {"MAE": 0.0, "RMSE": 0.0, "n_predictions": 0}
        true = pred_arr[:, 2]
        pred = pred_arr[:, 3]
        return {
            "MAE":           round(float(mae(true, pred)), 4),
            "RMSE":          round(float(rmse(true, pred)), 4),
            "n_predictions": int(len(pred_arr)),
        }

    @staticmethod
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

        precisions, recalls = [], []
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

    def from_prediction_artifacts(
        self,
        prediction_dir: str,
        test_ratings: np.ndarray,
        n_reco: int | None = None,
        relevance_threshold: float | None = None,
    ) -> dict:
        """Offline-assignment akışının ürettiği .npy dosyalarından metrik üretir.

        Beklenen dosyalar (``generate_recommendations`` çıktısı):
          - predicted_ratings.npy            (mixed: white + gray)
          - predicted_ratings_gray_same.npy  (gray, same-cluster formülü)
          - predicted_ratings_gray_fallback.npy
          - recommendations_white.npy        (uid → top-N item array)
          - recommendations_gray_same.npy
          - recommendations_gray_fallback.npy

        Returns
        -------
        dict — segment bazlı MAE, RMSE, Precision@N, Recall@N.
        """
        n = int(n_reco) if n_reco is not None else self.at_n
        thr = float(relevance_threshold) if relevance_threshold is not None else self.relevance_threshold

        mixed         = np.load(os.path.join(prediction_dir, "predicted_ratings.npy"))
        gray_same     = np.load(os.path.join(prediction_dir, "predicted_ratings_gray_same.npy"))
        gray_fallback = np.load(os.path.join(prediction_dir, "predicted_ratings_gray_fallback.npy"))
        rec_white         = np.load(os.path.join(prediction_dir, "recommendations_white.npy"), allow_pickle=True).item()
        rec_gray_same     = np.load(os.path.join(prediction_dir, "recommendations_gray_same.npy"), allow_pickle=True).item()
        rec_gray_fallback = np.load(os.path.join(prediction_dir, "recommendations_gray_fallback.npy"), allow_pickle=True).item()

        white_users = {int(u) for u in rec_white}
        gray_users  = {int(u) for u in rec_gray_same} | {int(u) for u in rec_gray_fallback}

        out = {
            "mixed":             self._metrics_from_array(mixed),
            "gray_same_formula": self._metrics_from_array(gray_same),
            "gray_fallback":     self._metrics_from_array(gray_fallback),
        }

        p_w, r_w = self._precision_recall_at_n(rec_white, test_ratings, n, thr, white_users)
        p_g, r_g = self._precision_recall_at_n(rec_gray_same, test_ratings, n, thr, gray_users)
        p_f, r_f = self._precision_recall_at_n(rec_gray_fallback, test_ratings, n, thr, gray_users)

        out["white"] = {"Precision@N": round(p_w, 4), "Recall@N": round(r_w, 4)}
        out["gray_same_formula"].update({"Precision@N": round(p_g, 4), "Recall@N": round(r_g, 4)})
        out["gray_fallback"].update({"Precision@N": round(p_f, 4), "Recall@N": round(r_f, 4)})
        return out
