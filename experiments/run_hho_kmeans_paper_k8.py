"""
K=8 fold2: B0_KMEANS vs B1_HHO — makale (HHO + KMeans Lloyd) vs varsayilan kmref.

Makale (IJISAE): HHO ile centroid bul -> KMeans Lloyd (init=HHO centroids).
Kodda makale adimi: --kmeans-refine-overwrite
Varsayilan grid: sadece bos kume onarimi (_repair_empty_clusters), Lloyd YOK.

  python -m experiments.run_hho_kmeans_paper_k8 --phase all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"
OUT = REPO / "results" / "hho_kmeans_paper_k8_fold2.csv"

FOLD = 2
K = 8
EPOCH = 200
POP = 50

BASE = [
    "--dataset", "100k",
    "--no-prune", "--no-gray-sheep",
    "--preprocess", "none",
    "--feature-extraction", "wnmf",
    "--svd-components", "50", "--wnmf-epochs", "50",
    "--legacy-wnmf-suffix",
    "--init-mode", "mkpp",
    "--cluster-metric", "euclidean",
    "--fitness", "wcss",
    "--cluster-objective", "wcss",
    "--train-only", "--eval-split", "official",
    "--fold", str(FOLD),
    "--k", str(K),
    "--baseline-epoch", str(EPOCH),
    "--pop-size", str(POP),
]

VARIANTS = [
    {
        "variant": "B0_KMEANS",
        "algo": ["B0_KMEANS"],
        "extra": [],
        "suffix": f"B0_KMEANS_euc_imkpp_nogs_trainonly_official_f{FOLD}_none_wnmf50_k{K}_pwcss",
    },
    {
        "variant": "B1_HHO_hho_only",
        "algo": ["B1_HHO"],
        "extra": ["--no-kmeans-refine"],
        "suffix": f"B1_HHO_euc_imkpp_nogs_trainonly_official_f{FOLD}_none_wnmf50_k{K}_pwcss",
    },
    {
        "variant": "B1_HHO_default_kmref",
        "algo": ["B1_HHO"],
        "extra": [],  # bos kume onarimi; Lloyd yok
        "suffix": f"B1_HHO_euc_imkpp_nogs_trainonly_official_f{FOLD}_none_wnmf50_k{K}_pwcss_kmref",
    },
    {
        "variant": "B1_HHO_paper_kmeans_overwrite",
        "algo": ["B1_HHO"],
        "extra": ["--kmeans-refine-overwrite", "--kmeans-refine-iter", "300"],
        "suffix": f"B1_HHO_euc_imkpp_nogs_trainonly_official_f{FOLD}_none_wnmf50_k{K}_pwcss_kmref",
    },
]


def run_assign(*, skip_existing: bool) -> int:
    for v in VARIANTS:
        cmd = [sys.executable, "-u", str(GEN), *BASE, "--algo", *v["algo"], *v["extra"]]
        if skip_existing:
            cmd.append("--skip-existing")
        print(f"\n=== {v['variant']} ===", flush=True)
        print(" ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            return rc
    return 0


def _wcss(d: Path) -> float | None:
    sys.path.insert(0, str(REPO / "mealpy"))
    from mealpy_comparison_v2 import compute_wcss_fast

    sol_p, uf = d / "best_sol.npy", d / "user_features.npy"
    if not sol_p.is_file() or not uf.is_file():
        return None
    X = np.load(uf)
    sol = np.load(sol_p)
    w, _ = compute_wcss_fast(X, sol, K, metric="euclidean")
    return float(w)


def phase_compare() -> None:
    rows = []
    b0_wcss = None
    for v in VARIANTS:
        d = ROOT / v["suffix"]
        w = _wcss(d)
        if v["variant"] == "B0_KMEANS":
            b0_wcss = w
        hho_pre = hho_post = meta_gain = None
        ch = d / "convergence_history.csv"
        if ch.is_file():
            h = pd.read_csv(ch)
            hho_pre = float(h["fitness"].iloc[0])
            hho_post = float(h["fitness"].iloc[-1])
            if hho_pre and hho_pre < 1e5:
                meta_gain = (hho_pre - hho_post) / hho_pre * 100.0
        vs_km = None
        if b0_wcss and w and b0_wcss > 0 and v["variant"] != "B0_KMEANS":
            vs_km = (b0_wcss - w) / b0_wcss * 100.0
        rows.append({
            "variant": v["variant"],
            "k": K,
            "wcss": w,
            "vs_B0_kmeans_pct": vs_km if v["variant"] != "B0_KMEANS" else 0.0,
            "hho_conv_start": hho_pre,
            "hho_conv_end": hho_post,
            "hho_meta_gain_pct": meta_gain,
            "dir": str(d),
            "exists": d.is_dir(),
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "compare", "all"], default="all")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()
    if args.phase in ("assign", "all"):
        if run_assign(skip_existing=args.skip_existing) != 0:
            sys.exit(1)
    if args.phase in ("compare", "all"):
        phase_compare()


if __name__ == "__main__":
    main()
