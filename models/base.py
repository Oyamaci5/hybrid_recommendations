from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class MatrixFactorizationBase(ABC):
    @abstractmethod
    def fit(self, R: np.ndarray, mask: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict(self, u: int, i: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def predict_all(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_user_factors(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_item_factors(self) -> np.ndarray:
        raise NotImplementedError
