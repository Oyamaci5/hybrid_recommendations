"""
Run TWOA-style benchmark comparison: 23 functions x algorithms x 30 runs.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MEALPY_LOCAL = os.path.join(_REPO_ROOT, "mealpy")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _MEALPY_LOCAL not in sys.path:
    sys.path.insert(0, _MEALPY_LOCAL)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from mealpy import FloatVar
from tqdm import tqdm

from benchmark_functions import benchmark_map, get_all_benchmarks
from utils import load_algorithm, save_results, short_algo_column, silence_mealpy_logging

POP_SIZE = 30
ITERATIONS = 500
NUM_RUNS = 30

TERMINATION = {"max_fe": POP_SIZE * ITERATIONS}

ALGORITHMS = [
    "HGS.OriginalHGS",
    "HHO.OriginalHHO",
    "AEO.OriginalAEO",
    "CircleSA.OriginalCircleSA",
    "MFO.OriginalMFO",
    "SquirrelSA.OriginalSquirrelSA",
    "INFO.OriginalINFO",
    "NGO.OriginalNGO",
    "BWO.OriginalBWO",
    "WOA.OriginalWOA",
    "GWO.OriginalGWO",
    "PSO.OriginalPSO",
    "DE.OriginalDE",
    "GA.OriginalGA",
    "OOA.OriginalOOA",
    "SFOA.OriginalSFOA",
    "DOA.OriginalDOA",
    "BeesA.OriginalBeesA",
    "HBA.OriginalHBA",
    "MPA.OriginalMPA",
    "AGTO.OriginalAGTO",
    "SA.OriginalSA",
    "CoatiOA.OriginalCoatiOA",
    "GBO.OriginalGBO",
    "ASO.OriginalASO",
    "AVOA.OriginalAVOA",
    "BA.OriginalBA",
    "SA.GaussianSA",
]


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    bench_meta = get_all_benchmarks(0.0)
    bench_keys = [b["key"] for b in bench_meta]
    base_map = benchmark_map(0.0)

    algo_columns = [short_algo_column(s) for s in ALGORITHMS]
    classes: list = []
    for a in ALGORITHMS:
        classes.append(load_algorithm(a))

    total_steps = len(bench_keys) * len(ALGORITHMS) * NUM_RUNS
    rows: list = []
    t0 = time.perf_counter()

    rng = np.random.default_rng(42)

    with tqdm(total=total_steps, desc="Benchmark runs", unit="run") as pbar:
        for run_id in range(NUM_RUNS):
            f7_noise = float(rng.random())
            f7_map = benchmark_map(f7_noise)
            for key in bench_keys:
                bench = f7_map[key] if key == "F7" else base_map[key]
                bounds = FloatVar(lb=list(bench["lb"]), ub=list(bench["ub"]))
                for algo_idx, (algo_str, AlgoCls, short_name) in enumerate(
                    zip(ALGORITHMS, classes, algo_columns)
                ):
                    fit = np.nan
                    if AlgoCls is None:
                        rows.append(
                            {
                                "algorithm": short_name,
                                "function": key,
                                "run_id": run_id,
                                "fitness": fit,
                            }
                        )
                        pbar.update(1)
                        continue
                    problem = {
                        "obj_func": bench["func"],
                        "bounds": bounds,
                        "minmax": "min",
                        "log_to": "console",
                    }
                    try:
                        model = AlgoCls(epoch=ITERATIONS, pop_size=POP_SIZE)
                        with silence_mealpy_logging():
                            model.solve(
                                problem,
                                termination=TERMINATION,
                                seed=int(10_000 + run_id * 1_000 + algo_idx * 17),
                            )
                        fit = float(model.g_best.target.fitness)
                    except Exception:
                        fit = np.nan
                    rows.append(
                        {
                            "algorithm": short_name,
                            "function": key,
                            "run_id": run_id,
                            "fitness": fit,
                        }
                    )
                    pbar.update(1)

    elapsed = time.perf_counter() - t0
    raw_df = pd.DataFrame(rows)
    save_results(raw_df, algo_columns, bench_meta, out_dir)

    n_fail = total_steps - int(raw_df["fitness"].notna().sum())
    print(
        f"Done in {elapsed:.1f}s. Successful finite runs: {int(raw_df['fitness'].notna().sum())} / {total_steps} "
        f"(failed/NaN: {n_fail})."
    )
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
