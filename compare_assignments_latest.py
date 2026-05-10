"""
Her algoritma için assignment klasörünü seç ve atamaları karşılaştır.

--suffix verilirse:  {base}/{algo}{suffix}/assignments.npy
verilmezse:          {base}/{algo}_* altında assignments.npy en yeni olan

Örnek:
  python compare_assignments_latest.py \\
    --dataset-root mealpy/results/assignments_lof/ml100k \\
    --algos B0_KMEANS B3_MFO H4_MFO+HHO HA_AVOAHGS LIT_GOA LIT_PSO \\
    --suffix _pruneu5_i10_minmax_nmf50_k100
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np


def resolve_path(base: str, algo: str, suffix: Optional[str]) -> Optional[str]:
    if suffix is not None:
        d = os.path.join(base, f"{algo}{suffix}")
        fp = os.path.join(d, "assignments.npy")
        return fp if os.path.isfile(fp) else None

    pattern = os.path.join(base, f"{algo}_*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    best_fp: Optional[str] = None
    best_t = -1.0
    for d in dirs:
        fp = os.path.join(d, "assignments.npy")
        if os.path.isfile(fp):
            t = os.path.getmtime(fp)
            if t > best_t:
                best_t = t
                best_fp = fp
    return best_fp


def main() -> None:
    p = argparse.ArgumentParser(description="Compare assignments.npy across algorithms.")
    p.add_argument(
        "--dataset-root",
        default="mealpy/results/assignments_lof/ml100k",
        help="Örn: mealpy/results/assignments_lof/ml100k veya .../ml1m",
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=[
            "B0_KMEANS",
            "B3_MFO",
            "H4_MFO+HHO",
            "HA_AVOAHGS",
            "LIT_GOA",
            "LIT_PSO",
        ],
    )
    p.add_argument(
        "--suffix",
        default=None,
        help="Tam klasör eki: örn _pruneu5_i10_minmax_nmf50_k100",
    )
    args = p.parse_args()

    base = os.path.normpath(args.dataset_root)
    algos: List[str] = args.algos

    paths: Dict[str, str] = {}
    for algo in algos:
        fp = resolve_path(base, algo, args.suffix)
        if fp is None:
            print(f"{algo}: DOSYA YOK (base={base})")
            continue
        paths[algo] = fp
        arr = np.load(fp)
        rel = os.path.relpath(fp, start=os.getcwd())
        print(
            f"{algo}: {rel} | shape={arr.shape}, "
            f"unique_first5={np.unique(arr)[:5]}, hash={hash(arr.tobytes())}"
        )

    algos_ok = list(paths.keys())
    for i in range(len(algos_ok)):
        for j in range(i + 1, len(algos_ok)):
            a, b = algos_ok[i], algos_ok[j]
            xa, xb = np.load(paths[a]), np.load(paths[b])
            if xa.shape != xb.shape:
                print(f"{a} vs {b}: ŞEKİL FARKLI {xa.shape} vs {xb.shape}")
                continue
            same = np.array_equal(xa, xb)
            diff_count = int(np.sum(xa != xb))
            print(f"{a} vs {b}: {'AYNI!' if same else f'{diff_count} farklı atama'}")


if __name__ == "__main__":
    main()
