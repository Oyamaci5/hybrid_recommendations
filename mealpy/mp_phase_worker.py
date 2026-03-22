"""
Aşama çalıştırma için ProcessPoolExecutor worker'ı (import edilebilir modül).
Windows spawn + ana script'te tire (-) olduğu için pickle burada tutulur.
"""

import importlib.util
import os

_MEALPY_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_PATH = os.path.join(_MEALPY_DIR, "mealpy-algorithms-comparision.py")
_shared = None


def pool_init(script_dir):
    import sys
    d = script_dir or _MEALPY_DIR
    if d not in sys.path:
        sys.path.insert(0, d)
    _ensure_shared()


def _ensure_shared():
    global _shared
    if _shared is None:
        spec = importlib.util.spec_from_file_location(
            "mealpy_alg_cmp", _SHARED_PATH
        )
        _shared = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_shared)
    return _shared


def run_algo_v3_task(task):
    """
    task: (algo_module, class_name, algo_name, matrix, K,
           initial_solutions, epoch, pop_size)
    """
    mod = _ensure_shared()
    (
        algo_module,
        class_name,
        algo_name,
        matrix,
        K,
        initial_solutions,
        epoch,
        pop_size,
    ) = task
    return mod._run_algo_v3_serialized(
        algo_module,
        class_name,
        algo_name,
        matrix,
        K,
        initial_solutions,
        epoch,
        pop_size,
    )
