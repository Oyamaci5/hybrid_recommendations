"""Quick comparison of paper-mode runs."""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.join(os.path.dirname(__file__), "..", "mealpy", "results", "assignments", "ml100k")


def load_metrics(pattern, label):
    rows = []
    dirs = {}
    for d in sorted(glob.glob(os.path.join(ROOT, pattern))):
        cm_path = os.path.join(d, "cluster_metrics.csv")
        if not os.path.isfile(cm_path):
            continue
        algo = os.path.basename(d).split("_colzscore")[0].split("_zscore")[0]
        cm = pd.read_csv(os.path.join(d, "cluster_metrics.csv")).iloc[0]
        assign = np.load(os.path.join(d, "assignments.npy"))
        rows.append({
            "pipe": label,
            "algo": algo,
            "wcss": float(cm["wcss"]),
            "sil": float(cm.get("silhouette_euclidean", np.nan)),
            "min_cl": int(np.bincount(assign).min()),
            "max_cl": int(np.bincount(assign).max()),
        })
        dirs[algo] = assign
    return pd.DataFrame(rows), dirs


new_df, new_assign = load_metrics("*colzscore*pca50*k30*pwcss*", "paper_pca50")
old_df, _ = load_metrics("*_zscore_euc_irand_paper*wnmf50*k30*pwcss*", "old_wnmf50")

print("=== WCSS (lower better) ===")
for label, df in [("paper_pca50", new_df), ("old_wnmf50", old_df)]:
    if df.empty:
        continue
    spread = df["wcss"].max() - df["wcss"].min()
    rel = 100 * (df["wcss"].max() / df["wcss"].min() - 1)
    mae_spread = None
    print(f"\n{label}: WCSS spread = {spread:.0f} ({rel:.1f}% above best)")
    print(df.sort_values("wcss")[["algo", "wcss", "sil"]].to_string(index=False))

# ARI between algos (same partition => similar MAE)
algos = sorted(new_assign)
if len(algos) >= 2:
    print("\n=== Adjusted Rand Index vs B0_KMEANS (1=identical) ===")
    ref = new_assign.get("B0_KMEANS")
    if ref is not None:
        for a in algos:
            if a == "B0_KMEANS":
                continue
            ari = adjusted_rand_score(ref, new_assign[a])
            print(f"  {a:14s} ARI={ari:.4f}")

# Eval CSV
for tag, run in [("paper_pca50", "run5"), ("old_wnmf50", "run3")]:
    files = glob.glob(
        os.path.join("results", "wnmf", "ml100k", "k30", "fold1", run, "wnmf_results*.csv")
    )
    if not files:
        continue
    df = pd.read_csv(files[0], comment="#")
    sub = df[df["scenario"].str.contains("calc_avg|cluster_avg", na=False)]
    spread = sub["mae"].max() - sub["mae"].min()
    print(f"\n=== Eval MAE ({tag}): spread = {spread:.4f} ===")
    print(sub.sort_values("mae")[["algo_label", "mae", "rmse", "scenario"]].to_string(index=False))
