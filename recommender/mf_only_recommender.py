"""Yalnızca matris faktörizasyonu ile doğrudan rating tahmini."""

from __future__ import annotations

import numpy as np

from models.base import MatrixFactorizationBase


class MFOnlyRecommender:
    """Yol 4: önerilen hiperparametrelerle MF ve `predict_all`."""

    def __init__(self, model: MatrixFactorizationBase) -> None:
        self.model = model

    def predict(self, user_id: int, item_id: int) -> float:
        return float(self.model.predict(int(user_id), int(item_id)))

    def predict_matrix(self) -> np.ndarray:
        return self.model.predict_all()
