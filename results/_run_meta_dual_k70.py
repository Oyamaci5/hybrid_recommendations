"""Quick k=70 MetaDual eval (beta=0.85, no tune)."""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import wnmf.meta_dual_cf as m
from wnmf.wnmf_utils import load_ratings_100k_all

K = 70
ALGOS = ["B1_HHO", "IWO_HHO", "HA_AVOAHGS", "AGTO", "B3_MFO"]

train, test = load_ratings_100k_all(
    os.path.join(REPO, "data", "ml-100k", "u.data"), random_seed=42, fold=1,
)
assign_b0 = m._load_assignments(m._assign_dir("B0_KMEANS", K))
b0 = m._baseline_cluster_avg_user(train, test, assign_b0)
print(f"K={K}  B0 cluster_avg  MAE={b0['mae']:.4f}  NDCG={b0['ndcg_at_10']:.4f}\n")
print(f"{'Algo':<14} {'c_avg MAE':>10} {'c_avg NDCG':>10} {'dual MAE':>10} {'dual NDCG':>10}")
print("-" * 58)

for algo in ALGOS:
    t0 = time.time()
    ad = m._assign_dir(algo, K)
    am = m._load_assignments(ad)
    U = np.load(os.path.join(ad, "user_features.npy"))
    C = m._load_centroids(ad, K, U.shape[1])
    ca = m._baseline_cluster_avg_user(train, test, am)
    du = m.predict_meta_dual(train, test, assign_b0, am, U, C, beta=0.85)
    print(
        f"{algo:<14} {ca['mae']:>10.4f} {ca['ndcg_at_10']:>10.4f} "
        f"{du['mae']:>10.4f} {du['ndcg_at_10']:>10.4f}  ({time.time() - t0:.0f}s)"
    )
