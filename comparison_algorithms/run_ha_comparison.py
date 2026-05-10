"""
HA-NGOAVOA benchmark comparison.

Runs HA_NGOAVOA against AVOA, NGO, GWO, OOA, HHO on the 23 classical
benchmark functions (30 runs each) and writes:

    results/ha_raw_results.pkl
    results/ha_summary_table.csv
    results/ha_wilcoxon_table.csv
    results/ha_friedman_table.csv
    results/ha_convergence_plots/{F1,F2,F9,F10,F15,F23}_convergence.png

Usage
-----
    python run_ha_comparison.py              # full run (~30-60 min)
    python run_ha_comparison.py --skip-run   # reuse existing pickle, only analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OPTIMIZERS_DIR = os.path.join(_REPO_ROOT, "optimizers")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based.AVOA import OriginalAVOA
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.swarm_based.NGO import OriginalNGO
from mealpy.swarm_based.OOA import OriginalOOA
from tqdm import tqdm

from benchmark_functions import benchmark_map, get_all_benchmarks, type_display
from statistical_analysis import (
    build_friedman_table,
    build_wilcoxon_table,
    friedman_analysis,
    write_friedman_csv,
    write_wilcoxon_csv,
)
from utils import silence_mealpy_logging


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POP_SIZE = 30
ITERATIONS = 500
NUM_RUNS = 30
SEED_BASE = 42

ALGO_ORDER = ["HA_AVOAHGS", "AVOA", "NGO", "GWO", "OOA", "HHO"]
# ALGO_ORDER = ["HA_NGOAVOA", "AVOA", "NGO", "GWO", "OOA", "HHO"]
HA_NAME = "HA_AVOAHGS"
RIVALS = [a for a in ALGO_ORDER if a != HA_NAME]

CONVERGENCE_KEYS = ["F1", "F2", "F9", "F10", "F15", "F23"]

PLOT_COLORS = {
    "HA_AVOAHGS": "#E63946",
    # "HA_NGOAVOA": "#E63946",
    "AVOA":       "#457B9D",
    "NGO":        "#2A9D8F",
    "GWO":        "#E9C46A",
    "OOA":        "#F4A261",
    "HHO":        "#A8DADC",
}
PLOT_LINESTYLES = {
    "HA_AVOAHGS": "-",
    # "HA_NGOAVOA": "-",
    "AVOA":       "--",
    "NGO":        "-.",
    "GWO":        ":",
    "OOA":        (0, (3, 1, 1, 1)),
    "HHO":        (0, (5, 2)),
}


# ---------------------------------------------------------------------------
# Algorithm loader
# ---------------------------------------------------------------------------
def _load_ha_ngoavoa_class():
    """Dynamically import HA_NGOAVOA from optimizers/HA-NGOAVOA.py."""
    path = os.path.join(_OPTIMIZERS_DIR, "HA-NGOAVOA.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HA-NGOAVOA implementation not found at {path}")
    spec = importlib.util.spec_from_file_location("ha_ngoavoa_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HA_NGOAVOA


def _load_ha_avoahgs_class():
    """Dynamically import HA_AVOAHGS from comparison_algorithms/HA_AVOAHGS.py."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HA_AVOAHGS.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HA_AVOAHGS implementation not found at {path}")
    spec = importlib.util.spec_from_file_location("ha_avoahgs_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HA_AVOAHGS


def build_competitors() -> Dict[str, Callable[..., Any]]:
    HA_AVOAHGS = _load_ha_avoahgs_class()
    return {
        "HA_AVOAHGS": HA_AVOAHGS,
        # "HA_NGOAVOA": _load_ha_ngoavoa_class(),
        "AVOA": OriginalAVOA,
        "NGO": OriginalNGO,
        "GWO": OriginalGWO,
        "OOA": OriginalOOA,
        "HHO": OriginalHHO,
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def _extract_convergence(model) -> List[float]:
    """Per-epoch best fitness from mealpy's history.list_current_best."""
    try:
        return [float(a.target.fitness) for a in model.history.list_current_best]
    except Exception:
        return []


def _pad_curve(curve: List[float], target_len: int) -> List[float]:
    """Pad a convergence curve so all curves share a common length."""
    if not curve:
        return [float("nan")] * target_len
    if len(curve) >= target_len:
        return curve[:target_len]
    last = curve[-1]
    return list(curve) + [last] * (target_len - len(curve))


def run_single(
    AlgoCls: Callable[..., Any],
    bench: Dict[str, Any],
    seed: int,
) -> Tuple[float, List[float]]:
    bounds = FloatVar(lb=list(bench["lb"]), ub=list(bench["ub"]))
    problem = {
        "obj_func": bench["func"],
        "bounds": bounds,
        "minmax": "min",
        "log_to": "console",
    }
    model = AlgoCls(epoch=ITERATIONS, pop_size=POP_SIZE)
    with silence_mealpy_logging():
        model.solve(problem, seed=int(seed))
    fit = float(model.g_best.target.fitness)
    curve = _extract_convergence(model)
    return fit, curve


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmarks() -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, List[List[float]]]],
]:
    competitors = build_competitors()
    bench_rows = get_all_benchmarks(0.0)
    base_map = benchmark_map(0.0)

    raw_results: Dict[str, Dict[str, List[float]]] = {
        algo: {b["key"]: [] for b in bench_rows} for algo in ALGO_ORDER
    }
    convergence_data: Dict[str, Dict[str, List[List[float]]]] = {
        algo: {b["key"]: [] for b in bench_rows} for algo in ALGO_ORDER
    }

    rng_for_f7 = np.random.default_rng(SEED_BASE)

    total_steps = len(ALGO_ORDER) * len(bench_rows) * NUM_RUNS
    with tqdm(total=total_steps, desc="HA benchmark", unit="run") as pbar:
        for run in range(NUM_RUNS):
            np.random.seed(SEED_BASE + run)
            f7_noise = float(rng_for_f7.random())
            f7_map = benchmark_map(f7_noise)

            for bench in bench_rows:
                key = bench["key"]
                bench_obj = f7_map[key] if key == "F7" else base_map[key]
                for algo_name in ALGO_ORDER:
                    AlgoCls = competitors[algo_name]
                    pbar.set_postfix_str(f"{algo_name}/{key}/run{run+1}")
                    try:
                        fit, curve = run_single(AlgoCls, bench_obj, SEED_BASE + run)
                    except Exception as exc:
                        fit = float("nan")
                        curve = []
                    raw_results[algo_name][key].append(fit)
                    convergence_data[algo_name][key].append(
                        _pad_curve(curve, ITERATIONS)
                    )
                    pbar.update(1)

    return raw_results, convergence_data


# ---------------------------------------------------------------------------
# Summary table (literature format)
# ---------------------------------------------------------------------------
def build_summary_table(
    raw_results: Dict[str, Dict[str, List[float]]],
    bench_rows: List[Dict[str, Any]],
    algorithms: List[str],
    friedman_ranks: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Per function: two rows (Mean, Std), '*' marks best mean per row.
    Footers: best-counts per group + Friedman ranks.
    Returns (DataFrame, group_counts).
    """
    cols = ["Function", "Type", "Metric"] + algorithms
    body: List[Dict[str, Any]] = []

    type_to_keys = defaultdict(list)
    for b in bench_rows:
        type_to_keys[b["type"]].append(b["key"])

    group_counts: Dict[str, Dict[str, int]] = {
        "unimodal": {a: 0 for a in algorithms},
        "multimodal": {a: 0 for a in algorithms},
        "fixed_multimodal": {a: 0 for a in algorithms},
        "total": {a: 0 for a in algorithms},
    }

    def fmt(x: float) -> str:
        if not np.isfinite(x):
            return "NaN"
        return f"{x:.2E}"

    for b in bench_rows:
        key = b["key"]
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for a in algorithms:
            vals = np.asarray(raw_results.get(a, {}).get(key, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            means[a] = float(np.mean(vals)) if vals.size else float("nan")
            stds[a] = float(np.std(vals, ddof=0)) if vals.size else float("nan")

        finite_means = {a: m for a, m in means.items() if np.isfinite(m)}
        best_algo = min(finite_means, key=finite_means.get) if finite_means else None
        if best_algo is not None:
            group_counts[b["type"]][best_algo] += 1
            group_counts["total"][best_algo] += 1

        mean_row = {"Function": b["name"], "Type": type_display(b["type"]), "Metric": "Mean"}
        std_row = {"Function": b["name"], "Type": type_display(b["type"]), "Metric": "Std"}
        for a in algorithms:
            cell_mean = fmt(means[a])
            if best_algo is not None and a == best_algo and np.isfinite(means[a]):
                cell_mean = f"{cell_mean}*"
            mean_row[a] = cell_mean
            std_row[a] = f"({fmt(stds[a])})"
        body.append(mean_row)
        body.append(std_row)

    def footer(label: str, counts: Dict[str, int]) -> Dict[str, Any]:
        row: Dict[str, Any] = {"Function": label, "Type": "-", "Metric": "Count"}
        for a in algorithms:
            row[a] = counts[a]
        return row

    body.append(footer("Best Count (Unimodal F1-7)", group_counts["unimodal"]))
    body.append(footer("Best Count (Multimodal F8-13)", group_counts["multimodal"]))
    body.append(footer("Best Count (Fixed F14-23)", group_counts["fixed_multimodal"]))
    body.append(footer("Best Count (TOTAL)", group_counts["total"]))

    fr_row: Dict[str, Any] = {
        "Function": "Friedman Rank",
        "Type": "-",
        "Metric": "Rank",
    }
    for a in algorithms:
        v = friedman_ranks.get(a, float("nan"))
        fr_row[a] = f"{v:.4f}" if np.isfinite(v) else "nan"
    body.append(fr_row)

    return pd.DataFrame(body, columns=cols), group_counts


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------
def plot_convergence_curves(
    convergence_data: Dict[str, Dict[str, List[List[float]]]],
    bench_rows: List[Dict[str, Any]],
    output_dir: str,
    keys: List[str] = CONVERGENCE_KEYS,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    name_by_key = {b["key"]: b["name"] for b in bench_rows}

    for key in keys:
        if key not in name_by_key:
            continue
        title = name_by_key[key]
        any_negative = False
        algo_curves: Dict[str, np.ndarray] = {}
        for algo in ALGO_ORDER:
            runs = convergence_data.get(algo, {}).get(key, [])
            if not runs:
                continue
            arr = np.asarray(runs, dtype=float)
            with np.errstate(invalid="ignore"):
                mean_curve = np.nanmean(arr, axis=0)
            algo_curves[algo] = mean_curve
            if np.any(mean_curve <= 0):
                any_negative = True

        if not algo_curves:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        plot_fn = ax.plot if any_negative else ax.semilogy
        for algo, curve in algo_curves.items():
            plot_fn(
                np.arange(1, len(curve) + 1),
                curve,
                label=algo,
                color=PLOT_COLORS[algo],
                linestyle=PLOT_LINESTYLES[algo],
                linewidth=1.8,
            )
        ax.set_xlabel("Iteration", fontsize=12)
        ylabel = "Best Fitness" if any_negative else "Best Fitness (log scale)"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Convergence Curve - {title}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{key}_convergence.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_console_summary(
    elapsed: float,
    bench_rows: List[Dict[str, Any]],
    group_counts: Dict[str, Dict[str, int]],
    friedman: Dict[str, Any],
    wilcoxon_counts: Dict[str, Dict[str, int]],
) -> None:
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print()
    print("========================================")
    print("HA Benchmark Sonuclari")
    print("========================================")
    print(f"Toplam sure      : {minutes} dakika {seconds} saniye")
    print(f"Fonksiyon sayisi : {len(bench_rows)}")
    print(f"Run sayisi       : {NUM_RUNS} x {len(ALGO_ORDER)} algoritma "
          f"= {NUM_RUNS * len(ALGO_ORDER)} run/fonksiyon")
    print()
    print(f"Best Count (TOTAL):")
    for a in ALGO_ORDER:
        print(f"  {a:<12}: {group_counts['total'][a]:>2} / {len(bench_rows)} fonksiyonda en iyi")
    print()
    print("Friedman Mean Rank:")
    order = sorted(ALGO_ORDER, key=lambda x: friedman["mean_ranks"][x])
    for a in order:
        print(f"  {a:<12}: {friedman['mean_ranks'][a]:.4f}")
    if np.isfinite(friedman["chi2"]) and np.isfinite(friedman["p_value"]):
        print(f"  Friedman chi2 = {friedman['chi2']:.4f}, p = {friedman['p_value']:.2E}")
    print()
    print(f"Wilcoxon ({HA_NAME} kazandigi fonksiyon sayisi):")
    for r in RIVALS:
        c = wilcoxon_counts[r]
        print(f"  vs {r:<5}: +{c['+']:>2} / -{c['-']:>2} / ={c['=']:>2}")
    print("========================================")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="HA-NGOAVOA benchmark comparison")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip benchmark runs; load existing ha_raw_results.pkl and only analyze.",
    )
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "ha_convergence_plots")

    raw_pkl = os.path.join(out_dir, "ha_raw_results.pkl")
    conv_pkl = os.path.join(out_dir, "ha_convergence_data.pkl")

    bench_rows = get_all_benchmarks(0.0)

    t0 = time.perf_counter()
    if args.skip_run and os.path.isfile(raw_pkl):
        print(f"[skip-run] Loading cached results from {raw_pkl}")
        raw_results = load_pickle(raw_pkl)
        convergence_data = (
            load_pickle(conv_pkl) if os.path.isfile(conv_pkl) else {}
        )
        elapsed = 0.0
    else:
        raw_results, convergence_data = run_benchmarks()
        save_pickle(raw_results, raw_pkl)
        save_pickle(convergence_data, conv_pkl)
        elapsed = time.perf_counter() - t0

    friedman = friedman_analysis(raw_results, bench_rows, ALGO_ORDER, aggregator="median")

    summary_df, group_counts = build_summary_table(
        raw_results, bench_rows, ALGO_ORDER, friedman["mean_ranks"]
    )
    summary_df.to_csv(os.path.join(out_dir, "ha_summary_table.csv"), index=False)

    wilcoxon_df, wilcoxon_counts = build_wilcoxon_table(
        raw_results, bench_rows, HA_NAME, RIVALS
    )
    write_wilcoxon_csv(wilcoxon_df, os.path.join(out_dir, "ha_wilcoxon_table.csv"))

    friedman_df = build_friedman_table(friedman, ALGO_ORDER)
    write_friedman_csv(friedman_df, os.path.join(out_dir, "ha_friedman_table.csv"))

    if convergence_data:
        plot_convergence_curves(convergence_data, bench_rows, plot_dir)

    print_console_summary(
        elapsed, bench_rows, group_counts, friedman, wilcoxon_counts
    )
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
