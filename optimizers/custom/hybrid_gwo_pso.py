"""Basit iki-aşama hibrit: GWO sonra PSO iyileştirme."""

from __future__ import annotations

from typing import Callable

import numpy as np

from optimizers.base import OptimizeResult


class HybridGWOPSO:
    """MealPy sınıfları yüklüyse sırayla GWO→PSO küçük bütçe ile çalıştırılır."""

    def __init__(self, gwo_agents: int = 20, gwo_iter: int = 30, pso_agents: int = 25, pso_iter: int = 40):
        self.gwo_agents, self.gwo_iter = int(gwo_agents), int(gwo_iter)
        self.pso_agents, self.pso_iter = int(pso_agents), int(pso_iter)

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        lb: float | np.ndarray = 1.0,
        ub: float | np.ndarray = 5.0,
    ) -> OptimizeResult:
        from optimizers.mealpy_wrapper import MealPyWrapper

        gwo = MealPyWrapper("gwo", n_agents=self.gwo_agents, n_iter=self.gwo_iter)
        r1 = gwo.minimize(objective, dim, np.ones(dim) * float(lb), np.ones(dim) * float(ub))
        shifted = lambda z: objective(z) + 1e-9 * np.linalg.norm(z - r1.best_vector)
        pso = MealPyWrapper("pso", n_agents=self.pso_agents, n_iter=self.pso_iter)
        r2 = pso.minimize(shifted, dim, np.ones(dim) * float(lb), np.ones(dim) * float(ub))
        return r2 if r2.best_score <= r1.best_score else r1
