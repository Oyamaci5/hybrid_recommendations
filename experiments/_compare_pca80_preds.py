"""Quick MAE/NDCG comparison for pca80pct_none_k_protocol_preds.csv."""
import pandas as pd

df = pd.read_csv("results/pca80pct_none_k_protocol_preds.csv")
ALGOS = ["HA_AVOAHGS", "B2_HGS", "B_AVOA", "LIT_GWO", "LIT_PSO"]

for pred, sim, label in [
    ("cluster_knn", "pearson", "cluster_knn / pearson (k=cluster_min)"),
    ("cluster_knn", "cosine", "cluster_knn / cosine"),
    ("cluster_avg", "cosine", "cluster_avg / cosine"),
    ("cluster_avg", "pearson", "cluster_avg / pearson"),
]:
    sub = df[(df.predictor == pred) & (df.similarity == sim)]
    if sub.empty:
        continue
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    mae = sub.pivot(index="k", columns="algo", values="mae").reindex(columns=ALGOS).round(4)
    ndcg = sub.pivot(index="k", columns="algo", values="ndcg_at_10").reindex(columns=ALGOS).round(4)
    print("\nMAE (dusuk iyi)")
    print(mae.to_string())
    print("\nNDCG@10 (yuksek iyi)")
    print(ndcg.to_string())
    print("\nMAE en iyi algo:")
    for k in mae.index:
        print(f"  K={k:2d}: {mae.loc[k].idxmin()} ({mae.loc[k].min():.4f})")
    print("NDCG en iyi algo:")
    for k in ndcg.index:
        print(f"  K={k:2d}: {ndcg.loc[k].idxmax()} ({ndcg.loc[k].max():.4f})")

print("\n" + "=" * 72)
print("HA_AVOAHGS — tüm koşullarda ortalama sıra (1=en iyi)")
print("=" * 72)
for metric, asc in [("mae", True), ("ndcg_at_10", False)]:
    ranks = []
    for (pred, sim), g in df.groupby(["predictor", "similarity"]):
        p = g.pivot(index="k", columns="algo", values=metric).reindex(columns=ALGOS)
        r = p.rank(axis=1, ascending=asc)
        ranks.append(r["HA_AVOAHGS"])
    avg_rank = pd.concat(ranks, axis=1).mean(axis=1)
    print(f"\n{metric}:")
    for k, v in avg_rank.items():
        print(f"  K={k:2d}: ort. sıra {v:.2f}/5")
