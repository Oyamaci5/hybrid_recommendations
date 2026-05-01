"""Metaheuristic / MF tuner optimizer tabanı (rs_meta uyum katmanı)."""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class OptimizeResult:
    best_vector: np.ndarray
    best_score: float
    history: list[float] = field(default_factory=list)


@dataclass
class MFTuneResult(OptimizeResult):
    best_params: dict = field(default_factory=dict)


class OptimizerBase(ABC):
    """Genel tek hedefli sürekli optimizer arayüzü."""

    @abstractmethod
    def minimize(self, objective, dim: int, lb: np.ndarray, ub: np.ndarray) -> OptimizeResult:
        raise NotImplementedError
