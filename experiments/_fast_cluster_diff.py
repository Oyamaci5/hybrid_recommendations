import numpy as np
from collections import Counter
from sklearn.metrics import adjusted_rand_score

from experiments.fuzzy_official_protocol import FAST_K_LIST, FAST_ALGOS, assign_dir

print("K  exact%_HA-LIT  exact%_HA-B  exact%_LIT-B  ARI_HA-LIT  ARI_HA-B  ARI_LIT-B")
for k in FAST_K_LIST:
    labels = {}
    sizes = {}
    for algo in FAST_ALGOS:
        a = np.load(assign_dir(algo, k, fast=True) / "assignments.npy")
        labels[algo] = a.astype(int)
        sizes[algo] = sorted(Counter(a.tolist()).values(), reverse=True)
    ha, lit, b = labels["HA_AVOAHGS"], labels["LIT_GWO"], labels["B_AVOA"]
    print(
        f"{k:2d}  "
        f"{np.mean(ha == lit) * 100:8.1f}  "
        f"{np.mean(ha == b) * 100:8.1f}  "
        f"{np.mean(lit == b) * 100:8.1f}  "
        f"{adjusted_rand_score(ha, lit):8.3f}  "
        f"{adjusted_rand_score(ha, b):8.3f}  "
        f"{adjusted_rand_score(lit, b):8.3f}"
    )

print("\nKume boyutlari (desc, ilk 6):")
for k in FAST_K_LIST:
    parts = []
    for algo in FAST_ALGOS:
        a = np.load(assign_dir(algo, k, fast=True) / "assignments.npy")
        s = sorted(Counter(a.tolist()).values(), reverse=True)
        parts.append(f"{algo[:3]}={s[:6]}")
    print(f"K={k:2d}: " + " | ".join(parts))
