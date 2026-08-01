"""K=10 fuzzy FCM: HA vs others cluster structure + separation."""
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

root = Path("mealpy/results/assignments/ml100k")
suffix = "_fuzzy_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k10"
algos = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
labels = {}
stats = {}

for a in algos:
    assign = np.load(root / f"{a}{suffix}" / "assignments.npy")
    c = Counter(assign.astype(int).tolist())
    sizes = sorted(c.values())
    labels[a] = assign
    stats[a] = {
        "active": len(c),
        "min": min(sizes),
        "max": max(sizes),
        "mean": float(np.mean(sizes)),
        "std": float(np.std(sizes)),
        "sizes": sizes,
        "counts": dict(sorted(c.items())),
        "singletons": sum(1 for s in sizes if s == 1),
        "le3": sum(1 for s in sizes if s <= 3),
    }

print("=== K=10 fuzzy FCM cluster sizes ===")
for a in algos:
    s = stats[a]
    print(
        f"{a}: active={s['active']} min={s['min']} max={s['max']} "
        f"mean={s['mean']:.1f} std={s['std']:.1f} "
        f"singletons={s['singletons']} n<=3={s['le3']}"
    )
    print(f"  sorted: {s['sizes']}")
    by_id = [s["counts"].get(i, 0) for i in range(max(s["counts"]) + 1)]
    print(f"  C0..C{len(by_id)-1}: {by_id}")

print("\n=== Pairwise vs HA_AVOAHGS ===")
ha = labels["HA_AVOAHGS"]
for a in algos:
    if a == "HA_AVOAHGS":
        continue
    la = labels[a]
    ari = adjusted_rand_score(ha, la)
    nmi = normalized_mutual_info_score(ha, la, average_method="arithmetic")
    agree = float((ha == la).mean())
    print(
        f"HA vs {a:12s}: ARI={ari:.4f}  NMI={nmi:.4f}  "
        f"agree={agree:.4f} ({int(agree * 943)}/943)"
    )

print("\n=== All pairs ===")
for i, a in enumerate(algos):
    for b in algos[i + 1 :]:
        ari = adjusted_rand_score(labels[a], labels[b])
        nmi = normalized_mutual_info_score(labels[a], labels[b], average_method="arithmetic")
        print(f"{a:12s} vs {b:12s}: ARI={ari:.4f}  NMI={nmi:.4f}")

# WCSS from DB if available
try:
    import sqlite3
    conn = sqlite3.connect("results/assignment_experiments.sqlite")
    rows = conn.execute(
        """
        SELECT algo, wcss FROM assignments
        WHERE assign_suffix=? AND k=10
        ORDER BY algo
        """,
        (suffix,),
    ).fetchall()
    print("\n=== WCSS ===")
    for algo, wcss in rows:
        print(f"  {algo}: {wcss:.2f}")
except Exception as e:
    print(f"\nWCSS skip: {e}")
