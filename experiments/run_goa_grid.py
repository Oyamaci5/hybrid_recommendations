# experiments/run_goa_grid.py
from itertools import product
from run_goa_kmeans_paper import main as goa_main
import sys

grids = {
    "k": [3, 6, 9, 12, 15],
    "agents": [20, 40],
    "iters": [50, 100],
    "pca_components": [20, 50],
}

for k, agents, iters, pca in product(grids["k"], grids["agents"], grids["iters"], grids["pca_components"]):
    sys.argv = [
        "run_goa_kmeans_paper.py",
        "--k", str(k),
        "--agents", str(agents),
        "--iters", str(iters),
        "--pca_components", str(pca),
    ]
    print(f"\n=== k={k}, agents={agents}, iters={iters}, pca={pca} ===")
    goa_main()