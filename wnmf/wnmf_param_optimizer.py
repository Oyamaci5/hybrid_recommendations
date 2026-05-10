"""
wnmf_param_optimizer.py
=======================
WNMF hiperparametre araması: lr, reg, latent_dim, n_epochs.
Fitness: validasyon MAE (düşük = iyi). Optimizasyon: MFO (mealpy MFO.OriginalMFO, B3_MFO ile aynı çekirdek).
"""

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from mealpy import FloatVar
from mealpy.swarm_based.MFO import OriginalMFO

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from wnmf_model import WNMFModel


class WNMFParamOptimizer:
    def __init__(self, train, val, n_agents=20, n_iter=30):
        self.train = train
        self.val = val
        self.n_agents = n_agents
        self.n_iter = n_iter

        # Arama uzayı
        self.bounds = {
            "lr": (0.001, 0.05),
            "reg": (0.001, 0.05),
            "latent_dim": (10, 40),  # int
            "n_epochs": (50, 150),  # int
        }

    def _decode(self, individual: np.ndarray) -> Tuple[float, float, int, int]:
        ind = np.asarray(individual, dtype=np.float64).ravel()
        lr = float(ind[0])
        reg = float(ind[1])
        ld_lo, ld_hi = self.bounds["latent_dim"]
        ep_lo, ep_hi = self.bounds["n_epochs"]
        ld = int(round(float(np.clip(ind[2], ld_lo, ld_hi))))
        ep = int(round(float(np.clip(ind[3], ep_lo, ep_hi))))
        return lr, reg, ld, ep

    def fitness(self, individual):
        lr, reg, ld, ep = self._decode(individual)

        n_users = int(self.train[:, 0].max()) + 1
        n_items = int(self.train[:, 1].max()) + 1

        model = WNMFModel(
            n_users=n_users,
            n_items=n_items,
            latent_dim=ld,
            learning_rate=lr,
            regularization=reg,
            n_epochs=ep,
            random_seed=42,
            use_bias=True,
        )
        model.fit(self.train)
        mae, _ = model.evaluate(self.val)
        return float(mae)

    def optimize(self) -> Dict[str, Any]:
        # DOA veya B3_MFO: burada MFO (OriginalMFO) kullanılıyor.
        # Her birey: [lr, reg, latent_dim, n_epochs] — son iki boyut sürekli aranıp tam sayıya yuvarlanır.
        # Fitness: MAE (minimize)
        b = self.bounds
        lb: List[float] = [b["lr"][0], b["reg"][0], float(b["latent_dim"][0]), float(b["n_epochs"][0])]
        ub: List[float] = [b["lr"][1], b["reg"][1], float(b["latent_dim"][1]), float(b["n_epochs"][1])]

        problem = {
            "obj_func": self.fitness,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": "min",
            "log_to": None,
        }

        pop_size = max(5, int(self.n_agents))
        model = OriginalMFO(epoch=int(self.n_iter), pop_size=pop_size)
        model.solve(problem, seed=42)

        sol = np.asarray(model.g_best.solution, dtype=np.float64)
        lr, reg, ld, ep = self._decode(sol)
        return {
            "lr": lr,
            "reg": reg,
            "latent_dim": ld,
            "n_epochs": ep,
        }


# Çalıştırma:
# opt = WNMFParamOptimizer(train, val, n_agents=20, n_iter=30)
# best = opt.optimize()
# print(best)  # {'lr': 0.008, 'reg': 0.003, 'latent_dim': 28, 'n_epochs': 120}
