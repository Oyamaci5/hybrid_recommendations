import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
df = pd.read_csv(REPO / "results/fuzzy_official_fast_preds.csv")
avg = df[(df["predictor"] == "cluster_avg_soft") & (df["similarity"] == "cosine")]
knn50 = df[(df["predictor"] == "cluster_knn") & (df["knn_k"] == 50) & (df["similarity"] == "cosine")]

print("=== cluster_avg_soft (cosine) ===")
for k in sorted(avg["k"].unique()):
    s = avg[avg["k"] == k].sort_values("mae")
    parts = ", ".join(f"{r.algo.split('_')[0]}={r.mae:.4f}" for _, r in s.iterrows())
    print(f"K={int(k):2d}: {parts}")

print("\n=== cluster_knn k=50 (cosine) ===")
for k in sorted(knn50["k"].unique()):
    s = knn50[knn50["k"] == k].sort_values("mae")
    parts = ", ".join(f"{r.algo.split('_')[0]}={r.mae:.4f}" for _, r in s.iterrows())
    print(f"K={int(k):2d}: {parts}")

from experiments.run_fuzzy_official_fast_compare import build_summary, cluster_structure_rows

struct = cluster_structure_rows()
summ = build_summary(df, struct, soft=0.1)
summ.to_csv(REPO / "results/fuzzy_official_fast_k_sweep_summary.csv", index=False)
print(f"\nSummary -> {REPO / 'results/fuzzy_official_fast_k_sweep_summary.csv'}")
