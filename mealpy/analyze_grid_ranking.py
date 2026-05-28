"""Grid summary: NDCG / Prec / Rec karsilastirma."""
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "results/grid/k_wnmf_knn_grid_summary.csv")
for c in ["ndcg_at_10", "precision_at_10", "recall_at_10", "f1_at_10", "mae", "rmse", "wcss", "silhouette_euclidean"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(f"Rows: {len(df)}\n")


def fmt(row) -> str:
    return (
        f"wnmf={int(row.wnmf_dim)} K={int(row.cluster_k)} knn={int(row.knn_k)} {row.algo} | "
        f"NDCG={row.ndcg_at_10:.4f} P={row.precision_at_10:.4f} R={row.recall_at_10:.4f} | "
        f"MAE={row.mae:.4f} sil={row.silhouette_euclidean:.4f}"
    )


print("=== GLOBAL BEST (ranking metrikleri) ===")
for m, name in [
    ("ndcg_at_10", "NDCG@10"),
    ("precision_at_10", "Prec@10"),
    ("recall_at_10", "Rec@10"),
    ("f1_at_10", "F1@10"),
]:
    r = df.loc[df[m].idxmax()]
    print(f"{name:8} {r[m]:.4f}  {fmt(r)}")

print("\n=== Algo bazinda en iyi NDCG / Prec / Rec ===")
cols = ["algo", "wnmf_dim", "cluster_k", "knn_k", "ndcg_at_10", "precision_at_10", "recall_at_10", "mae"]
for m in ["ndcg_at_10", "precision_at_10", "recall_at_10"]:
    b = df.loc[df.groupby("algo")[m].idxmax()].sort_values("algo")
    print(f"--- max {m} ---")
    print(b[cols].round(4).to_string(index=False))
    print()

print("=== cluster_k ortalamalari (NDCG/P/R) ===")
print(df.groupby("cluster_k")[["ndcg_at_10", "precision_at_10", "recall_at_10"]].mean().round(4).sort_values("ndcg_at_10", ascending=False).to_string())

print("\n=== wnmf_dim ortalamalari ===")
print(df.groupby("wnmf_dim")[["ndcg_at_10", "precision_at_10", "recall_at_10"]].mean().round(4).sort_values("ndcg_at_10", ascending=False).to_string())

print("\n=== knn_k ortalamalari ===")
print(df.groupby("knn_k")[["ndcg_at_10", "precision_at_10", "recall_at_10"]].mean().round(4).sort_values("ndcg_at_10", ascending=False).to_string())

print("\n=== TOP 8 NDCG@10 ===")
for _, r in df.nlargest(8, "ndcg_at_10").iterrows():
    print(fmt(r))

print("\n=== TOP 8 Prec@10 ===")
for _, r in df.nlargest(8, "precision_at_10").iterrows():
    print(fmt(r))

print("\n=== TOP 8 Rec@10 ===")
for _, r in df.nlargest(8, "recall_at_10").iterrows():
    print(fmt(r))

print("\n=== NDCG>=0.840 en dusuk MAE (trade-off) ===")
for _, r in df[df.ndcg_at_10 >= 0.840].nsmallest(8, "mae").iterrows():
    print(fmt(r))
