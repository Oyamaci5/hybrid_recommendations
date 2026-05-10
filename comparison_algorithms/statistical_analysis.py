"""
Statistical analysis utilities for HA-NGOAVOA benchmark comparison.

Provides:
    - Wilcoxon signed-rank test (HA_NGOAVOA vs each rival per function)
    - Friedman test (across all algorithms over all functions)
    - CSV writers in literature-standard format
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


SIGNIF_ALPHA = 0.05


def _clean_pair(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def wilcoxon_pair(
    ha_values: Sequence[float],
    rival_values: Sequence[float],
) -> Tuple[float, str]:
    """
    Wilcoxon signed-rank between HA_NGOAVOA and one rival.

    Returns
    -------
    (p_value, sig)
        sig = '+' if HA_NGOAVOA significantly better (lower mean & p<alpha)
              '-' if HA_NGOAVOA significantly worse  (higher mean & p<alpha)
              '=' otherwise (or both arrays equal)
    """
    a, b = _clean_pair(ha_values, rival_values)
    if a.size < 3 or b.size < 3:
        return 1.0, "="

    diff = a - b
    if np.allclose(diff, 0.0):
        return 1.0, "="

    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return 1.0, "="

    p = float(p)
    if not np.isfinite(p):
        return 1.0, "="

    if p < SIGNIF_ALPHA:
        return p, "+" if np.mean(a) < np.mean(b) else "-"
    return p, "="


def build_wilcoxon_table(
    raw_results: Dict[str, Dict[str, List[float]]],
    bench_rows: List[Dict[str, object]],
    ha_name: str,
    rivals: List[str],
) -> pd.DataFrame:
    """
    Long-format CSV: one row per function with <rival>_p / <rival>_sig columns.
    Footer row counts +/-/= per rival.
    """
    body: List[Dict[str, object]] = []
    counts = {r: {"+": 0, "-": 0, "=": 0} for r in rivals}

    for b in bench_rows:
        key = b["key"]
        row: Dict[str, object] = {"Function": b["name"]}
        ha_vals = raw_results.get(ha_name, {}).get(key, [])
        for rival in rivals:
            r_vals = raw_results.get(rival, {}).get(key, [])
            p, sig = wilcoxon_pair(ha_vals, r_vals)
            row[f"{rival}_p"] = f"{p:.2E}"
            row[f"{rival}_sig"] = sig
            counts[rival][sig] += 1
        body.append(row)

    footer: Dict[str, object] = {"Function": "Summary (+/-/=)"}
    for rival in rivals:
        c = counts[rival]
        footer[f"{rival}_p"] = "-"
        footer[f"{rival}_sig"] = f"+{c['+']}/-{c['-']}/={c['=']}"
    body.append(footer)
    return pd.DataFrame(body), counts


def _per_func_aggregate(
    raw_results: Dict[str, Dict[str, List[float]]],
    bench_keys: List[str],
    algorithms: List[str],
    aggregator: str = "median",
) -> np.ndarray:
    """
    Build (n_funcs x n_algos) matrix using `aggregator` over the 30 runs.
    NaN-tolerant.
    """
    rows = []
    for key in bench_keys:
        row = []
        for algo in algorithms:
            vals = np.asarray(raw_results.get(algo, {}).get(key, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row.append(np.nan)
            elif aggregator == "median":
                row.append(float(np.median(vals)))
            elif aggregator == "mean":
                row.append(float(np.mean(vals)))
            else:
                raise ValueError(aggregator)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def friedman_analysis(
    raw_results: Dict[str, Dict[str, List[float]]],
    bench_rows: List[Dict[str, object]],
    algorithms: List[str],
    aggregator: str = "median",
) -> Dict[str, object]:
    """
    Run Friedman test on (n_funcs x n_algos) median matrix.

    Returns
    -------
    dict with keys:
        mean_ranks   : {algo: float}
        final_rank   : {algo: int}   (1 = best, lower mean rank wins)
        chi2         : float
        p_value      : float
        matrix       : np.ndarray
    """
    bench_keys = [b["key"] for b in bench_rows]
    matrix = _per_func_aggregate(raw_results, bench_keys, algorithms, aggregator)

    nan_mask = ~np.any(np.isnan(matrix), axis=1)
    clean = matrix[nan_mask]
    if clean.shape[0] < 2:
        chi2, p = float("nan"), float("nan")
    else:
        try:
            chi2, p = friedmanchisquare(*[clean[:, i] for i in range(clean.shape[1])])
            chi2, p = float(chi2), float(p)
        except Exception:
            chi2, p = float("nan"), float("nan")

    ranks_per_func = np.apply_along_axis(rankdata, 1, clean)
    mean_ranks_arr = ranks_per_func.mean(axis=0)
    mean_ranks = {algo: float(mean_ranks_arr[i]) for i, algo in enumerate(algorithms)}

    order = sorted(algorithms, key=lambda a: mean_ranks[a])
    final_rank = {algo: i + 1 for i, algo in enumerate(order)}

    return {
        "mean_ranks": mean_ranks,
        "final_rank": final_rank,
        "chi2": chi2,
        "p_value": p,
        "matrix": matrix,
    }


def build_friedman_table(friedman: Dict[str, object], algorithms: List[str]) -> pd.DataFrame:
    rows = []
    order = sorted(algorithms, key=lambda a: friedman["mean_ranks"][a])
    for algo in order:
        rows.append(
            {
                "Algorithm": algo,
                "Mean_Rank": f"{friedman['mean_ranks'][algo]:.4f}",
                "Final_Rank": friedman["final_rank"][algo],
            }
        )
    rows.append(
        {
            "Algorithm": "Friedman_chi2",
            "Mean_Rank": f"{friedman['chi2']:.4f}" if np.isfinite(friedman["chi2"]) else "nan",
            "Final_Rank": "-",
        }
    )
    rows.append(
        {
            "Algorithm": "p_value",
            "Mean_Rank": f"{friedman['p_value']:.2E}" if np.isfinite(friedman["p_value"]) else "nan",
            "Final_Rank": "-",
        }
    )
    return pd.DataFrame(rows)


def write_wilcoxon_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def write_friedman_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
