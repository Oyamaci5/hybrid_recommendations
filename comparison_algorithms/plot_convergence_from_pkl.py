import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


PKL_PATH = "comparison_algorithms/results/ha_convergence_data.pkl"
OUT_DIR = "comparison_algorithms/results"
OUT_PATH = os.path.join(OUT_DIR, "ha_convergence_overall.png")
OUT_PATH_GAP = os.path.join(OUT_DIR, "ha_convergence_relative_gap.png")
OUT_PATH_GAP_ZOOM = os.path.join(OUT_DIR, "ha_convergence_relative_gap_zoom.png")
OUT_PATH_GAP_LOG = os.path.join(OUT_DIR, "ha_convergence_relative_gap_log.png")


def main():
    with open(PKL_PATH, "rb") as f:
        conv_data = pickle.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    stacked_curves = []
    labels = []
    for algo, func_map in conv_data.items():
        all_runs = []
        for _, runs in func_map.items():
            arr = np.asarray(runs, dtype=float)  # (30, 500)
            if arr.ndim == 2 and arr.shape[1] > 0:
                all_runs.append(arr)
        if not all_runs:
            continue

        merged = np.concatenate(all_runs, axis=0)  # (23*30, 500)
        mean_curve = np.nanmean(merged, axis=0)
        stacked_curves.append(mean_curve)
        labels.append(algo)

    if not stacked_curves:
        print("No convergence data found.")
        return

    curves = np.asarray(stacked_curves, dtype=float)
    finite_vals = curves[np.isfinite(curves) & (curves > 0)]
    if finite_vals.size > 0:
        y_low = np.percentile(finite_vals, 1)
        y_high = np.percentile(finite_vals, 99.5)
    else:
        y_low, y_high = 1e-12, 1.0

    for algo, mean_curve in zip(labels, curves):
        safe_curve = np.clip(mean_curve, y_low, y_high)
        ax.semilogy(safe_curve, label=algo, linewidth=1.7)

    ax.set_title("Overall Mean Convergence Curves (All Functions)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (log, clipped)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

    # Relative gap plot: emphasizes small differences between algorithms.
    # gap(%) = ((curve - best_at_iter) / (|best_at_iter| + eps)) * 100
    eps = 1e-12
    best_at_iter = np.nanmin(curves, axis=0)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    gap_curves = []
    for algo, mean_curve in zip(labels, curves):
        gap = ((mean_curve - best_at_iter) / (np.abs(best_at_iter) + eps)) * 100.0
        gap = np.nan_to_num(gap, nan=np.nan, posinf=np.nan, neginf=np.nan)
        gap_curves.append(gap)
        ax2.plot(gap, label=algo, linewidth=1.8)
    ax2.set_title("Relative Convergence Gap vs Best (All Functions)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Gap to Best (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH_GAP, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {OUT_PATH_GAP}")

    # Zoomed gap plot: skip early spike-heavy zone and clip y-range
    gap_arr = np.asarray(gap_curves, dtype=float)
    start_iter = 10
    tail = gap_arr[:, start_iter:]
    finite_tail = tail[np.isfinite(tail)]
    if finite_tail.size > 0:
        y_upper = np.percentile(finite_tail, 95)
    else:
        y_upper = 1.0
    y_upper = max(float(y_upper), 1e-6)

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    for algo, gap in zip(labels, gap_arr):
        ax3.plot(np.arange(start_iter, len(gap)), gap[start_iter:], label=algo, linewidth=1.8)
    ax3.set_ylim(0, y_upper)
    ax3.set_title("Relative Convergence Gap (Zoomed, Iter>=10)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Gap to Best (%)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH_GAP_ZOOM, dpi=160, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {OUT_PATH_GAP_ZOOM}")

    # Log-gap plot: log10(1 + gap) for robust visibility
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    for algo, gap in zip(labels, gap_arr):
        log_gap = np.log10(1.0 + np.clip(gap, 0, None))
        ax4.plot(log_gap, label=algo, linewidth=1.8)
    ax4.set_title("Relative Convergence Gap (log10(1+gap))")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("log10(1 + Gap%)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH_GAP_LOG, dpi=160, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {OUT_PATH_GAP_LOG}")


if __name__ == "__main__":
    main()
