"""
Expose OriginalSFOA from the repo's mealpy/sfoa_optimizer.py (avoid self-import).
"""
import importlib.util
import os

_MEALPY_SFOA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mealpy", "sfoa_optimizer.py")
)
_spec = importlib.util.spec_from_file_location("_mealpy_sfoa_impl", _MEALPY_SFOA)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
OriginalSFOA = _mod.OriginalSFOA

__all__ = ["OriginalSFOA"]
