import sqlite3

c = sqlite3.connect("results/assignment_experiments.sqlite")
c.row_factory = sqlite3.Row
rows = c.execute(
    """
    SELECT algo, scenario, mae, ndcg_at_10, knn, assign_suffix
    FROM wnmf_results WHERE run_id = 3360 AND k = 70
    ORDER BY scenario, mae
    """
).fetchall()

print("RUN 3360 | K=70 | suffix:", rows[0]["assign_suffix"] if rows else "?")
print()
for scen in ("calc_avg_rating", "cluster_avg", "cluster_knn_weighted"):
    sub = [dict(r) for r in rows if r["scenario"] == scen]
    if not sub:
        continue
    print(f"=== {scen} ===")
    for r in sorted(sub, key=lambda x: x["mae"]):
        print(f"  {r['algo']:<14} MAE={r['mae']:.4f}  NDCG={r['ndcg_at_10']:.4f}")

knn = sorted([dict(r) for r in rows if r["scenario"] == "cluster_knn_weighted"], key=lambda x: x["mae"])
print("\ncluster_knn_weighted sıralama (MAE):")
for i, r in enumerate(knn, 1):
    mark = " <-- B0" if r["algo"] == "B0_KMEANS" else ""
    print(f"  {i}. {r['algo']:<14} {r['mae']:.4f}{mark}")

print("\ncalc_avg -> cluster_knn iyileşme (MAE düşüşü):")
by_algo = {}
for r in rows:
    by_algo.setdefault(r["algo"], {})[r["scenario"]] = dict(r)
print(f"{'Algo':<14} {'calc_MAE':>9} {'knn_MAE':>9} {'dMAE':>8} {'knn_NDCG':>8}")
for algo in sorted(by_algo):
    ca = by_algo[algo].get("calc_avg_rating")
    kn = by_algo[algo].get("cluster_knn_weighted")
    if not ca or not kn:
        continue
    print(
        f"{algo:<14} {ca['mae']:>9.4f} {kn['mae']:>9.4f} "
        f"{ca['mae'] - kn['mae']:>8.4f} {kn['ndcg_at_10']:>8.4f}"
    )
