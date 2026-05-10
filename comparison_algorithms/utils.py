"""
Helpers: mealpy algorithm discovery, CSV export, summaries.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type, Tuple

import numpy as np
import pandas as pd

import mealpy

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MEALPY_LOCAL = os.path.join(_REPO_ROOT, "mealpy")
_COMP_DIR = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_ALGOS = os.path.join(_COMP_DIR, "custom_algos")


def short_algo_column(algo_str: str) -> str:
    if algo_str == "SA.GaussianSA":
        return "GaussianSA"
    return algo_str.split(".", 1)[0]


def build_mealpy_name_index() -> Dict[str, Tuple[str, str]]:
    """
    Map 'HGS.OriginalHGS' -> ('mealpy.swarm_based.HGS', 'OriginalHGS').
    Includes Original* and GaussianSA classes with solve().
    """
    index: Dict[str, Tuple[str, str]] = {}
    for modname in _iter_mealpy_modules():
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        short = modname.split(".")[-1]
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not (name.startswith("Original") or name == "GaussianSA"):
                continue
            if getattr(obj, "__module__", "") != modname:
                continue
            if not (hasattr(obj, "solve") and hasattr(obj, "generate_population")):
                continue
            key = f"{short}.{name}"
            index[key] = (modname, name)
    return index


def _iter_mealpy_modules() -> List[str]:
    import pkgutil

    out: List[str] = []
    for finder, modname, ispkg in pkgutil.walk_packages(
        mealpy.__path__, mealpy.__name__ + ".", onerror=lambda x: None
    ):
        out.append(modname)
    return out


_MEALPY_INDEX: Optional[Dict[str, Tuple[str, str]]] = None


def get_mealpy_index() -> Dict[str, Tuple[str, str]]:
    global _MEALPY_INDEX
    if _MEALPY_INDEX is None:
        _MEALPY_INDEX = build_mealpy_name_index()
    return _MEALPY_INDEX


def load_algorithm(algo_str: str) -> Optional[Type[Any]]:
    """Return optimizer class or None if unavailable."""
    if algo_str == "DOA.OriginalDOA":
        if _MEALPY_LOCAL not in sys.path:
            sys.path.insert(0, _MEALPY_LOCAL)
        try:
            from doa_optimizer import OriginalDOA

            return OriginalDOA
        except Exception:
            return None

    if algo_str == "SFOA.OriginalSFOA":
        if _CUSTOM_ALGOS not in sys.path:
            sys.path.insert(0, _CUSTOM_ALGOS)
        try:
            from sfoa_optimizer import OriginalSFOA

            return OriginalSFOA
        except Exception:
            return None

    idx = get_mealpy_index()
    if algo_str not in idx:
        return None
    modname, clsname = idx[algo_str]
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, clsname)
    except Exception:
        return None


def format_mean_std(values: List[float]) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "nan ± nan"
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=0))
    return f"{_fmt_sci(m)} ± {_fmt_sci(s)}"


def _fmt_sci(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    s = f"{x:.10E}"
    if "E" in s:
        mant, exp = s.split("E")
        return f"{mant}E{exp}"
    return s.replace("e", "E")


def _fmt_cell_mean(x: float) -> str:
    if not np.isfinite(x):
        return "NaN"
    s = f"{x:.10E}".replace("e", "E")
    return s


def _fmt_cell_std_paren(x: float) -> str:
    if not np.isfinite(x):
        return "NaN"
    inner = f"{x:.10E}".replace("e", "E")
    return f"({inner})"


@contextmanager
def silence_mealpy_logging():
    """Suppress mealpy tqdm (log_to='console') and INFO logs during solve."""
    # Blocks new mealpy loggers created mid-solve (loggerDict snapshot is not enough).
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def build_summary_table(
    raw_df: pd.DataFrame,
    algo_columns: List[str],
    bench_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Wide table: Function, Metric, Type, <algos...>."""
    key_to_name = {b["key"]: b["name"] for b in bench_rows}
    key_to_type = {b["key"]: b["type"] for b in bench_rows}
    from benchmark_functions import type_display

    means = raw_df.pivot_table(
        index="function", columns="algorithm", values="fitness", aggfunc="mean"
    )
    stds = raw_df.pivot_table(
        index="function", columns="algorithm", values="fitness", aggfunc="std"
    )

    rows_out: List[Dict[str, Any]] = []
    for key in [b["key"] for b in bench_rows]:
        fname = key_to_name[key]
        t = type_display(key_to_type[key])
        mean_row: Dict[str, Any] = {
            "Function": fname,
            "Metric": "Mean",
            "Type": t,
        }
        std_row: Dict[str, Any] = {
            "Function": fname,
            "Metric": "Std",
            "Type": t,
        }
        for a in algo_columns:
            mv = means.loc[key, a] if key in means.index and a in means.columns else np.nan
            sv = stds.loc[key, a] if key in stds.index and a in stds.columns else np.nan
            mean_row[a] = _fmt_cell_mean(float(mv)) if pd.notna(mv) else "NaN"
            std_row[a] = _fmt_cell_std_paren(float(sv)) if pd.notna(sv) else "NaN"
        rows_out.append(mean_row)
        rows_out.append(std_row)
    return pd.DataFrame(rows_out)


def _rank_row_lowest_best(means: pd.Series) -> pd.Series:
    """Rank 1 = smallest mean; NaN -> worst rank + 1."""
    r = means.rank(method="dense", ascending=True)
    if r.notna().any():
        worst = float(r.max(skipna=True))
        r = r.fillna(worst + 1.0)
    return r.astype(float)


def build_summary_ranked(
    raw_df: pd.DataFrame,
    algo_columns: List[str],
    bench_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    from benchmark_functions import type_display

    key_to_name = {b["key"]: b["name"] for b in bench_rows}

    means = raw_df.pivot_table(
        index="function", columns="algorithm", values="fitness", aggfunc="mean"
    )
    # ensure all algo columns
    for a in algo_columns:
        if a not in means.columns:
            means[a] = np.nan

    body_rows: List[Dict[str, Any]] = []
    for b in bench_rows:
        key = b["key"]
        row: Dict[str, Any] = {
            "Function": b["name"],
            "Type": type_display(b["type"]),
        }
        mrow = means.loc[key] if key in means.index else pd.Series({a: np.nan for a in algo_columns})
        rk = _rank_row_lowest_best(mrow[algo_columns])
        for a in algo_columns:
            row[f"{a}_rank"] = int(rk[a]) if np.isfinite(rk[a]) else "NaN"
        body_rows.append(row)

    df_body = pd.DataFrame(body_rows)

    def best_counts_for_keys(keys: List[str]) -> Dict[str, Any]:
        names = {key_to_name[k] for k in keys}
        sub = df_body[df_body["Function"].isin(names)]
        counts = {a: 0 for a in algo_columns}
        for _, r in sub.iterrows():
            for a in algo_columns:
                col = f"{a}_rank"
                if col in r and r[col] == 1:
                    counts[a] += 1
        out: Dict[str, Any] = {}
        for a in algo_columns:
            out[f"{a}_rank"] = counts[a]
        return out

    def footer_row(label: str, keys: List[str]) -> Dict[str, Any]:
        bc = best_counts_for_keys(keys)
        fr: Dict[str, Any] = {"Function": label, "Type": "-"}
        for a in algo_columns:
            fr[f"{a}_rank"] = bc[f"{a}_rank"]
        return fr

    keys_all = [b["key"] for b in bench_rows]
    keys_u = [b["key"] for b in bench_rows if b["type"] == "unimodal"]
    keys_m = [b["key"] for b in bench_rows if b["type"] == "multimodal"]
    keys_f = [b["key"] for b in bench_rows if b["type"] == "fixed_multimodal"]

    footers = [
        footer_row("Best_Count_Unimodal_1_7", keys_u),
        footer_row("Best_Count_Multimodal_8_13", keys_m),
        footer_row("Best_Count_Fixed_Multimodal_14_23", keys_f),
        footer_row("Best_Count_TOTAL", keys_all),
    ]
    return pd.concat([df_body, pd.DataFrame(footers)], ignore_index=True)


def save_results(
    raw_df: pd.DataFrame,
    algo_columns: List[str],
    bench_rows: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_results.csv")
    raw_df.to_csv(raw_path, index=False)

    summary = build_summary_table(raw_df, algo_columns, bench_rows)
    summary.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)

    ranked = build_summary_ranked(raw_df, algo_columns, bench_rows)
    ranked.to_csv(os.path.join(output_dir, "summary_ranked.csv"), index=False)
