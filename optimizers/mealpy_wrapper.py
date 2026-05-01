"""MealPy sarmalayıcı ve kayıtlı birkaç yaygın swarm ismi."""

from __future__ import annotations

from functools import lru_cache
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from optimizers.base import OptimizerBase, OptimizeResult

_MEALPY_SHIM = Path(__file__).resolve().parent.parent / "mealpy"


def _ensure_mealpy_path() -> None:
    if _MEALPY_SHIM.is_dir():
        mp = str(_MEALPY_SHIM)
        if mp not in sys.path:
            sys.path.insert(0, mp)


@lru_cache(maxsize=1)
def _all_mealpy_algorithms():
    try:
        _ensure_mealpy_path()
        from mealpy_comparison_v2 import get_all_algorithms_v3

        return get_all_algorithms_v3()
    except Exception:
        return []


def mealpy_resolve(name: str) -> type | None:
    """Örn. ``pso`` · ``PSS`` olmayan doğru başlıkta `PSO.OriginalPSO` eşlemesi yapılır."""
    key = name.strip().lower().replace("-", "")
    lst = _all_mealpy_algorithms()
    for a in lst:
        pref = a["full_name"].split(".", 1)[0].lower()
        pref = pref.replace("_", "")
        if pref == key:
            return a["class"]
    return None


class MealPyWrapper(OptimizerBase):
    def __init__(
        self,
        algo_name: str,
        n_agents: int = 30,
        n_iter: int = 100,
        seed: int = 42,
        ctor_kwargs: dict | None = None,
    ):
        self._algo_name = algo_name
        cls = mealpy_resolve(algo_name)
        if cls is None:
            raise ImportError(
                f"MealPy algoritması çözülmedi `{algo_name}`. MealPy yüklü ve "
                "`mealpy_comparison_v2.get_all_algorithms_v3` erişilebilir mi kontrol edin."
            )
        self._Ctor = cls
        self.epoch = int(n_iter)
        self.pop_size = int(n_agents)
        self.seed = int(seed)
        self._ctor_extra = dict(ctor_kwargs or {})

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        lb: np.ndarray,
        ub: np.ndarray,
        **_kwargs,
    ) -> OptimizeResult:
        try:
            _ensure_mealpy_path()
            from mealpy_comparison_v2 import get_special_params
            from mealpy import FloatVar
        except ImportError as e:
            raise ImportError("mealpy ve mealpy_comparison_v2 gerekli.") from e

        lbs = np.broadcast_to(lb, dim).astype(float).ravel()
        ubs = np.broadcast_to(ub, dim).astype(float).ravel()

        fname = getattr(self._Ctor, "__name__", "")
        algo_key_hint = fname or self._algo_name
        algo_full = None
        for a in _all_mealpy_algorithms():
            if self._Ctor is a["class"]:
                algo_full = a["full_name"]
                break
        sp_call = algo_full if algo_full else algo_key_hint
        try:
            sp = get_special_params(sp_call, self.epoch, self.pop_size)
        except Exception:
            sp = {"epoch": self.epoch, "pop_size": self.pop_size}
        ctor_args = dict(sp or {})
        ctor_args.update(self._ctor_extra)
        model = self._Ctor(**ctor_args)

        def obj_func(vec) -> float:
            x = np.asarray(vec, dtype=np.float64).ravel()
            return float(objective(x))

        problem = {
            "obj_func": obj_func,
            "bounds": FloatVar(lb=lbs.tolist(), ub=ubs.tolist(), name="solution"),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

        np.random.seed(self.seed)

        try:
            model.solve(problem)
        except TypeError:
            model.solve(problem, {})

        vec = np.asarray(model.g_best.solution, dtype=float).ravel()
        fitness = float(model.g_best.target.fitness)
        return OptimizeResult(best_vector=vec, best_score=fitness)


def registry_short_names() -> dict[str, str]:
    """Tüm MealPy algoları yüklendiğinde {kisa_isim -> tam_isim}."""
    return {
        entry["full_name"].split(".")[0].lower(): entry["full_name"] for entry in _all_mealpy_algorithms()
    }
