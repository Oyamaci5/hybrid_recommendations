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

        lb_arr = np.broadcast_to(np.asarray(lb, dtype=np.float64), (int(dim),))
        ub_arr = np.broadcast_to(np.asarray(ub, dtype=np.float64), (int(dim),))

        gwo = MealPyWrapper("gwo", n_agents=self.gwo_agents, n_iter=self.gwo_iter)
        r1 = gwo.minimize(objective, dim, lb_arr, ub_arr)
        shifted = lambda z: objective(z) + 1e-9 * np.linalg.norm(z - r1.best_vector)
        pso = MealPyWrapper("pso", n_agents=self.pso_agents, n_iter=self.pso_iter)
        r2 = pso.minimize(shifted, dim, lb_arr, ub_arr)
        return r2 if r2.best_score <= r1.best_score else r1
