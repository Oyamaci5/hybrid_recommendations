"""Optimizer fabrikası."""

from __future__ import annotations

from typing import Callable

from optimizers.base import OptimizerBase
from utils.config import OptimizerConfig


def build_optimizer(cfg: OptimizerConfig) -> OptimizerBase | object | Callable:
    """
    MealPy swarm optimizerları veya DOA küme merkezi optimizasyonu.
    MealPy yüklü değil veya algo çözülmezse açıklayıcı hata fırlatılır (DOA dışında).
    """
    name = cfg.name.strip().lower()
    src = (cfg.source or "mealpy").lower()

    if name == "doa":
        from optimizers.doa import DOA

        # DOA dim is problem-dependent; return factory callable to instantiate later.
        return lambda dim, _lb, _ub: DOA(
            pop_size=int(cfg.n_agents),
            max_iter=int(cfg.n_iter),
            dim=int(dim),
            lb=float(_lb[0]),
            ub=float(_ub[0]),
            seed=int(cfg.seed),
            verbose=False,
        )

    if name == "hybrid_gwo_pso" and src == "custom":
        from optimizers.custom.hybrid_gwo_pso import HybridGWOPSO

        half_iter = max(int(cfg.n_iter) // 2, 1)
        return HybridGWOPSO(
            gwo_agents=int(cfg.n_agents),
            gwo_iter=half_iter,
            pso_agents=int(cfg.n_agents),
            pso_iter=max(int(cfg.n_iter) - half_iter, 1),
        )

    if src != "mealpy" and src != "":
        raise ValueError(f"Unsupported optimizer source '{src}' for optimizer '{name}'.")

    from optimizers.mealpy_wrapper import MealPyWrapper

    return MealPyWrapper(
        algo_name=name,
        n_agents=int(cfg.n_agents),
        n_iter=int(cfg.n_iter),
        seed=int(cfg.seed),
        ctor_kwargs=dict(cfg.algo_params or {}),
    )
