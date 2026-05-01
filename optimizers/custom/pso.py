"""MealPy PSO’nun doğrudan sarmalayıcısı (aynı MealPyWrapper davranışı)."""

from __future__ import annotations

import numpy as np

from optimizers.mealpy_wrapper import MealPyWrapper


class CustomPSO(MealPyWrapper):
    """PSO özelleştirmesi yapılacaksa burada genişletilebilir."""

    def __init__(self, n_agents: int = 30, n_iter: int = 100, seed: int = 42):
        super().__init__(algo_name="pso", n_agents=n_agents, n_iter=n_iter, seed=seed)
