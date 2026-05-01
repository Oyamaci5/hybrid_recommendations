"""Kümeli CF öneri katmanı (Yol 1-3).

Doc'taki temel CF tahmini (centroid + label tabanlı) yanı sıra
gray-sheep akışını da destekler:
  - predict_white          → cluster içinde sadece white üyeler
  - predict_gray_same      → cluster içinde tüm üyeler (gray dahil)
  - predict_gray_fallback  → item-ortalaması fallback
  - predict                → gray_mask varsa otomatik yönlendirir
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from core.metrics import pearson_similarity, predict_rating, _user_mean


GrayStrategy = Literal["white_only", "same_cluster", "fallback"]


class CFRecommender:
    """Yol 1-3 CF tahmin katmanı.

    İki çalışma modu:
      1. Centroid + label modu: ``centroids`` verilir, ``predict()`` ``core.metrics.predict_rating``
         üzerinden çalışır (geriye uyumlu).
      2. Assignment modu: ``cluster_labels`` ve opsiyonel ``gray_mask`` verilir,
         ``predict_white``/``predict_gray_same``/``predict_gray_fallback`` metodları ile
         offline-assignment akışı desteklenir.
    """

    def __init__(
        self,
        train_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: np.ndarray | None = None,
        top_k: int = 30,
        distance_metric: str = "pearson",
        gray_mask: np.ndarray | None = None,
        gray_strategy: GrayStrategy = "same_cluster",
    ) -> None:
        self.train_matrix = np.asarray(train_matrix, dtype=np.float32)
        self.cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        self.centroids = np.asarray(centroids, dtype=np.float32) if centroids is not None else None
        self.top_k = int(top_k)
        self.distance_metric = distance_metric
        self.gray_strategy = gray_strategy
        if gray_mask is not None:
            gm = np.asarray(gray_mask).reshape(-1).astype(bool)
            if gm.shape[0] != self.cluster_labels.shape[0]:
                raise ValueError("gray_mask length must match cluster_labels")
            self.gray_mask = gm
        else:
            self.gray_mask = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, user_id: int, item_id: int) -> float:
        """Otomatik yönlendirme.

        - ``gray_mask`` yoksa ve ``centroids`` varsa: legacy ``predict_rating`` çağrısı.
        - ``gray_mask`` varsa: kullanıcı gray ise ``gray_strategy`` uygulanır,
          değilse ``predict_white``.
        """
        u = int(user_id)
        i = int(item_id)

        if self.gray_mask is None:
            if self.centroids is None:
                return self.predict_white(u, i)
            return float(
                predict_rating(
                    self.train_matrix[u],
                    self.cluster_labels,
                    self.train_matrix,
                    self.centroids,
                    i,
                    top_k=self.top_k,
                    distance_metric=self.distance_metric,
                )
            )

        if bool(self.gray_mask[u]):
            if self.gray_strategy == "fallback":
                return self.predict_gray_fallback(u, i)
            return self.predict_gray_same(u, i)
        return self.predict_white(u, i)

    def predict_white(self, user_id: int, item_id: int) -> float:
        """Cluster içinde sadece white (gray olmayan) üyelerden komşu seç."""
        u = int(user_id)
        i = int(item_id)
        cid = int(self.cluster_labels[u])
        members = np.where(self.cluster_labels == cid)[0]
        if self.gray_mask is not None:
            members = members[~self.gray_mask[members]]
        members = members[members != u]
        return self._predict_with_neighbors(u, i, members)

    def predict_gray_same(self, user_id: int, item_id: int) -> float:
        """Cluster içinde tüm üyelerden komşu seç (gray dahil)."""
        u = int(user_id)
        i = int(item_id)
        cid = int(self.cluster_labels[u])
        members = np.where(self.cluster_labels == cid)[0]
        members = members[members != u]
        return self._predict_with_neighbors(u, i, members)

    def predict_gray_fallback(self, user_id: int, item_id: int) -> float:
        """Item-ortalaması fallback; item hiç oylanmamışsa kullanıcı ortalaması."""
        u = int(user_id)
        i = int(item_id)
        item_vals = self.train_matrix[:, i]
        item_vals = item_vals[item_vals != 0]
        if item_vals.size > 0:
            return float(np.clip(item_vals.mean(), 1.0, 5.0))
        return float(np.clip(_user_mean(self.train_matrix[u]), 1.0, 5.0))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _predict_with_neighbors(
        self,
        user_id: int,
        item_id: int,
        neighbor_ids: np.ndarray,
    ) -> float:
        u_vec = self.train_matrix[user_id]
        u_mean = _user_mean(u_vec)
        if neighbor_ids.size == 0:
            return float(np.clip(u_mean, 1.0, 5.0))

        rated = neighbor_ids[self.train_matrix[neighbor_ids, item_id] != 0]
        if rated.size == 0:
            return float(np.clip(u_mean, 1.0, 5.0))

        sims = np.array(
            [pearson_similarity(u_vec, self.train_matrix[v]) for v in rated],
            dtype=np.float64,
        )

        k = min(self.top_k, sims.size)
        idx = np.argpartition(np.abs(sims), -k)[-k:]
        idx = idx[np.argsort(np.abs(sims[idx]))[::-1]]
        nbr = rated[idx]
        nbr_sims = sims[idx]
        nbr_r = self.train_matrix[nbr, item_id].astype(np.float64)
        nbr_m = np.array([_user_mean(self.train_matrix[v]) for v in nbr], dtype=np.float64)

        denom = float(np.abs(nbr_sims).sum())
        if denom <= 1e-12:
            return float(np.clip(u_mean, 1.0, 5.0))
        numer = float(np.dot(nbr_sims, nbr_r - nbr_m))
        return float(np.clip(u_mean + numer / denom, 1.0, 5.0))
