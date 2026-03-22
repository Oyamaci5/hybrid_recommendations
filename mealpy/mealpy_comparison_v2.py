"""
Shim for imports: shared helpers live in mealpy-algorithms-comparision.py
(hyphenated name is not a valid Python package module name).
"""
import importlib.util
import os

_impl_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mealpy-algorithms-comparision.py",
)
_spec = importlib.util.spec_from_file_location(
    "_mealpy_algorithms_comparision_impl", _impl_path
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

load_movielens = _mod.load_movielens
mkmeans_plus_plus_init = _mod.mkmeans_plus_plus_init
compute_wcss_fast = _mod.compute_wcss_fast
detect_gray_sheep = _mod.detect_gray_sheep
make_fitness_function = _mod.make_fitness_function
pearson_distance_batch = _mod.pearson_distance_batch
_compute_metrics = _mod._compute_metrics

__all__ = [
    "load_movielens",
    "mkmeans_plus_plus_init",
    "compute_wcss_fast",
    "detect_gray_sheep",
    "make_fitness_function",
    "pearson_distance_batch",
    "_compute_metrics",
]
