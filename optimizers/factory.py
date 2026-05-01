"""Optimizer fabrikası."""

from __future__ import annotations

from optimizers.base import OptimizerBase
from utils.config import OptimizerConfig


def build_optimizer(cfg: OptimizerConfig) -> OptimizerBase | object:
    """
    MealPy swarm optimizerları veya DOA küme merkezi optimizasyonu.
    MealPy yüklü değil veya algo çözülmezse açıklayıcı hata fırlatılır (DOA dışında).
    """
    name = cfg.name.strip().lower()
    src = (cfg.source or "mealpy").lower()

    if name == "doa":
        from optimizers.doa import DOA

        return DOA(
            pop_size=int(cfg.n_agents),
            max_iter=int(cfg.n_iter),
            seed=int(cfg.seed),
        )

    if src != "mealpy" and src != "":
        pass

    from optimizers.mealpy_wrapper import MealPyWrapper

    return MealPyWrapper(
        algo_name=name,
        n_agents=int(cfg.n_agents),
        n_iter=int(cfg.n_iter),
        seed=int(cfg.seed),
        ctor_kwargs=dict(cfg.algo_params or {}),
    )
