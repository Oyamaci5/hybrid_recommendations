import numpy as np
from collections import Counter
from sklearn.metrics import adjusted_rand_score
from experiments.fuzzy_official_protocol import FAST_ALGOS, assign_dir

for k in [24, 26, 28, 30]:
    labels = {
        a: np.load(assign_dir(a, k, fast=True) / "assignments.npy").astype(int)
        for a in FAST_ALGOS
    }
    ha, lit, b = labels["HA_AVOAHGS"], labels["LIT_GWO"], labels["B_AVOA"]
    print(
        f"K={k}: exact% HA-LIT={np.mean(ha==lit)*100:.1f} "
        f"ARI={adjusted_rand_score(ha,lit):.3f} | "
        f"sizes HA={sorted(Counter(ha.tolist()).values(), reverse=True)[:4]}"
    )
