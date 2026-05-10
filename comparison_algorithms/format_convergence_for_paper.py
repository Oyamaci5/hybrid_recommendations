from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FUNCS = ["F1", "F2", "F9", "F10", "F15", "F23"]
PLOT_COLORS = {
    "HA_AVOAHGS": "#E63946",
    "AVOA": "#457B9D",
    "NGO": "#2A9D8F",
    "GWO": "#E9C46A",
    "OOA": "#F4A261",
    "HHO": "#1D3557",
}


def _load_conv_data(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _mean_curve(curves: List[List[float]]) -> np.ndarray:
    arr = np.asarray(curves, dtype=float)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.array([], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def _get_algo_order(conv_data: Dict[str, Dict[str, List[List[float]]]]) -> List[str]:
    preferred = ["HA_AVOAHGS", "AVOA", "NGO", "GWO", "OOA", "HHO"]
    existing = [a for a in preferred if a in conv_data]
    for a in conv_data.keys():
        if a not in existing:
            existing.append(a)
    return existing


def create_panel_figure(
    conv_data: Dict[str, Dict[str, List[List[float]]]],
    function_keys: List[str],
    out_png: str,
) -> None:
    algo_order = _get_algo_order(conv_data)
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=True)
    axes = axes.ravel()
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.22)

    handles = None
    labels = None

    for idx, key in enumerate(function_keys):
        ax = axes[idx]
        has_nonpositive = False
        curves_by_algo: Dict[str, np.ndarray] = {}
        for algo in algo_order:
            algo_map = conv_data.get(algo, {})
            runs = algo_map.get(key, [])
            curve = _mean_curve(runs)
            if curve.size == 0:
                continue
            curves_by_algo[algo] = curve
            if np.any(curve <= 0):
                has_nonpositive = True

        if not curves_by_algo:
            ax.set_title(f"{key} (No data)")
            ax.axis("off")
            continue

        plot_fn = ax.plot if has_nonpositive else ax.semilogy
        for algo, curve in curves_by_algo.items():
            color = PLOT_COLORS.get(algo, None)
            plot_fn(
                np.arange(1, len(curve) + 1),
                curve,
                label=algo,
                linewidth=1.8,
                color=color,
            )

        ax.grid(True, alpha=0.25)
        ax.set_title(f"({chr(97 + idx)}) {key}", fontsize=11)
        ax.set_xlabel("Number of Iterations", fontsize=10)
        ax.set_ylabel("Best Fitness Value", fontsize=10)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    for j in range(len(function_keys), len(axes)):
        axes[j].axis("off")

    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=True)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_latex_figure(out_tex: str, panel_png_path: str, keys: List[str]) -> None:
    rel_path = panel_png_path.replace("\\", "/")
    caption = (
        "Convergence trends of HA_AVOAHGS against baseline metaheuristics "
        f"on {', '.join(keys)} benchmark functions."
    )
    lines = [
        "\\begin{figure*}[t]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth]{{{rel_path}}}",
        f"\\caption{{{caption}}}",
        "\\label{fig:ha-convergence-panel}",
        "\\end{figure*}",
        "",
    ]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-ready convergence panel image and LaTeX snippet."
    )
    parser.add_argument(
        "--input-pkl",
        default="comparison_algorithms/results/ha_convergence_data.pkl",
        help="Path to convergence pickle file.",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        default=DEFAULT_FUNCS,
        help="Function keys to include (e.g., F1 F8 F15).",
    )
    parser.add_argument(
        "--out-png",
        default="comparison_algorithms/results/ha_convergence_panel.png",
        help="Output panel image path.",
    )
    parser.add_argument(
        "--out-tex",
        default="comparison_algorithms/results/ha_convergence_paper.tex",
        help="Output LaTeX figure path.",
    )
    args = parser.parse_args()

    conv_data = _load_conv_data(args.input_pkl)
    create_panel_figure(conv_data, args.functions, args.out_png)
    write_latex_figure(args.out_tex, args.out_png, args.functions)
    print(f"Wrote: {args.out_png}")
    print(f"Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
