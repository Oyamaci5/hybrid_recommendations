import pandas as pd

df = pd.read_csv("results/fuzzy_official_k4_14_extra_preds.csv")
avg = df[(df["predictor"] == "cluster_avg_soft") & (df["similarity"] == "cosine")]
knn = df[(df["predictor"] == "cluster_knn") & (df["knn_k"] == 50)]

def short(a):
    return a.replace("H9_QSA+CDO", "H9").replace("IWO_HHO", "IWO").replace("LIT_PSO", "PSO")

print(f"rows={len(df)}  avg={len(avg)}  knn50={len(knn)}\n")
for title, sub in [("cluster_avg_soft", avg), ("cluster_knn k=50", knn)]:
    print(f"=== {title} ===")
    for k in sorted(sub["k"].unique()):
        s = sub[sub["k"] == k].sort_values("mae")
        parts = [f"{short(r.algo)}={r.mae:.4f}" for _, r in s.iterrows()]
        print(f"K={int(k):2d}: " + " | ".join(parts))
