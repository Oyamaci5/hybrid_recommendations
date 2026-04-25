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
load_movielens_1m = _mod.load_movielens_1m
mkmeans_plus_plus_init = _mod.mkmeans_plus_plus_init
compute_wcss_fast = _mod.compute_wcss_fast
detect_gray_sheep = _mod.detect_gray_sheep
make_fitness_function = _mod.make_fitness_function
compute_fcm_objective = _mod.compute_fcm_objective
pearson_distance_batch = _mod.pearson_distance_batch
euclidean_distance_batch = _mod.euclidean_distance_batch
_compute_metrics = _mod._compute_metrics
get_all_algorithms_v3 = _mod.get_all_algorithms_v3
get_special_params = _mod.get_special_params

__all__ = [
    "load_movielens",
    "load_movielens_1m",
    "mkmeans_plus_plus_init",
    "compute_wcss_fast",
    "detect_gray_sheep",
    "make_fitness_function",
    "compute_fcm_objective",
    "pearson_distance_batch",
    "euclidean_distance_batch",
    "_compute_metrics",
    "get_all_algorithms_v3",
    "get_special_params",
]
