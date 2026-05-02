import os
import numpy as np
import pandas as pd

base = "mealpy/results/assignments_lof/ml100k"
folders = {
    "B3_MFO": "B3_MFO_k2_pruneu5_i10_minmax_svd20_k2",
    "H4_MFO+HHO": "H4_MFO+HHO_k2_pruneu5_i10_minmax_svd20_k2",
    "H9_QSA+CDO": "H9_QSA+CDO_k2_pruneu5_i10_minmax_svd20_k2",
    "H12_MFO+CDO": "H12_MFO+CDO_k2_pruneu5_i10_minmax_svd20_k2",
    "LIT_GOA": "LIT_GOA_k2_pruneu5_i10_minmax_svd20_k2",
}

print("{:<15} {:>12} {:>12} {:>10} {:>8}".format(
    "Algoritma", "WCSS", "Silhouette", "GS_count", "GS_pct"
))
print("-" * 60)

for algo, folder in folders.items():
    path = os.path.join(base, folder)
    g = np.load(os.path.join(path, "gray_sheep_mask.npy"))
    sol = np.load(os.path.join(path, "best_sol.npy"))
    wcss = float(sol) if sol.ndim == 0 else float(sol.min())

    csv_path = os.path.join(path, "assignment_summary.csv")
    sil = "—"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "silhouette" in df.columns:
            sil = "{:.4f}".format(float(df["silhouette"].iloc[0]))

    gs_count = int(g.sum())
    gs_pct = gs_count / len(g) * 100.0

    print("{:<15} {:12.2f} {:>12} {:10d} {:7.1f}%".format(
        algo, wcss, sil, gs_count, gs_pct
    ))