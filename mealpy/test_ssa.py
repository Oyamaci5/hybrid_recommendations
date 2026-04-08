# test_new_assignments.py
import numpy as np
import os

base = 'results/assignments_lof/ml1m'
algos = ['H9_QSA+CDO_k70', 'LIT_GOA_k70', 'LIT_GWO_k70', 
         'LIT_SSA_k70', 'H12_MFO+CDO_k70', 'H4_MFO+HHO_k70']

for algo in algos:
    path = os.path.join(base, algo)
    if not os.path.exists(path):
        print(f"{algo}: YOK")
        continue
    assignments = np.load(os.path.join(path, 'assignments.npy'))
    best_sol    = np.load(os.path.join(path, 'best_sol.npy'))
    print(f"{algo}:")
    print(f"  assignments hash: {hash(assignments.tobytes())}")
    print(f"  best_sol hash   : {hash(best_sol.tobytes())}")
    print(f"  assignments[:3] : {assignments[:3]}")