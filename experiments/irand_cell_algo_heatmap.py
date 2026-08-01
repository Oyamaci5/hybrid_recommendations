"""
Tek hücre (örn. svd5, K=14): algo×algo ARI / NMI / agreement / centroid L2 heatmap + CSV.

  python experiments/irand_cell_algo_heatmap.py --feature svd --dim 5 --k 14
  python experiments/irand_cell_algo_heatmap.py --feature-tag wnmf10 --k 27
  python experiments/irand_cell_algo_heatmap.py --feature svd --dim 5 --k 14 --no-plot
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mealpy"))

from compare_cluster_structure import (  # noqa: E402
    _common_mask,
    _load_centroids,
    _load_labels_and_gray,
    mean_centroid_distance,
)
from compare_irand_feature_cluster_similarity import (  # noqa: E402
    ASSIGN_ROOT,
    DEFAULT_ALGOS,
    compare_pair,
    load_one,
    suffix,
)

METRICS = ("ari", "nmi", "label_agreement", "centroid_l2_mean")


def parse_feature_tag(tag: str) -> Tuple[str, int]:
    m = re.fullmatch(r"(svd|wnmf)(\d+)", tag.strip().lower())
    if not m:
        raise ValueError(f"feature-tag beklenen form: svd5, wnmf10, … aldı: {tag!r}")
    return m.group(1), int(m.group(2))


def load_cell(
    algos: List[str], feat: str, dim: int, k: int,
) -> Dict[str, dict]:
    loaded: Dict[str, dict] = {}
    for algo in algos:
        out = load_one(algo, feat, dim, k)
        if out is None:
            print(f"  SKIP {algo}: yok")
            continue
        a, g, c, _ = out
        loaded[algo] = {"labels": a, "gray": g, "centroids": c}
    return loaded


def symmetric_matrix(
    algos: List[str],
    loaded: Dict[str, dict],
    metric: str,
) -> pd.DataFrame:
    n = len(algos)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, ai in enumerate(algos):
        for j, aj in enumerate(algos):
            if i == j:
                if metric == "centroid_l2_mean":
                    mat[i, j] = 0.0
                elif metric == "label_agreement":
                    mat[i, j] = 1.0
                else:
                    mat[i, j] = 1.0
                continue
            if ai not in loaded or aj not in loaded:
                continue
            da, db = loaded[ai], loaded[aj]
            m = compare_pair(
                da["labels"], da["gray"], db["labels"], db["gray"],
                da["centroids"], db["centroids"],
            )
            mat[i, j] = m[metric]
            mat[j, i] = mat[i, j]
    return pd.DataFrame(mat, index=algos, columns=algos)


def long_pairs(algos: List[str], loaded: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(algos, 2):
        if a not in loaded or b not in loaded:
            continue
        da, db = loaded[a], loaded[b]
        rows.append({
            "algo_a": a,
            "algo_b": b,
            **compare_pair(
                da["labels"], da["gray"], db["labels"], db["gray"],
                da["centroids"], db["centroids"],
            ),
        })
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, title: str, out_png: str, *, cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    data = df.astype(float).values
    im = ax.imshow(data, cmap=cmap, vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(df.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def export_cell(
    feat: str,
    dim: int,
    k: int,
    algos: List[str],
    out_dir: str,
    *,
    do_plot: bool = True,
) -> None:
    tag = f"{feat}{dim}_k{k}"
    suf = suffix(feat, dim, k)
    loaded = load_cell(algos, feat, dim, k)
    present = [a for a in algos if a in loaded]
    if len(present) < 2:
        print(f"[{tag}] En az 2 algo gerekli; bulunan: {present}")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {tag}  suffix={suf}  algos={present} ===")

    matrices: Dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        df = symmetric_matrix(present, loaded, metric)
        matrices[metric] = df
        path = os.path.join(out_dir, f"{tag}_{metric}.csv")
        df.round(4).to_csv(path)
        print(f"\n{metric.upper()}:")
        print(df.round(4).to_string())
        print(f"  -> {path}")

    pairs_path = os.path.join(out_dir, f"{tag}_pairs.csv")
    long_pairs(present, loaded).round(4).to_csv(pairs_path, index=False)
    print(f"\n  pairs -> {pairs_path}")

    if do_plot:
        try:
            plot_heatmap(
                matrices["ari"],
                f"ARI — {tag}",
                os.path.join(out_dir, f"{tag}_ari.png"),
                cmap="YlOrRd",
            )
            plot_heatmap(
                matrices["nmi"],
                f"NMI — {tag}",
                os.path.join(out_dir, f"{tag}_nmi.png"),
                cmap="Blues",
            )
            plot_heatmap(
                matrices["centroid_l2_mean"],
                f"Centroid L2 (Hungarian) — {tag}",
                os.path.join(out_dir, f"{tag}_centroid_l2.png"),
                cmap="magma_r",
            )
            print(f"  PNG -> {out_dir}/{tag}_*.png")
        except ImportError:
            print("  matplotlib yok; PNG atlandı.")


def main() -> None:
    p = argparse.ArgumentParser(description="Tek hücre algo×algo heatmap / CSV")
    p.add_argument("--feature", choices=("svd", "wnmf"), default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--feature-tag", default=None, help="svd5, wnmf10, …")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument(
        "--out-dir",
        default=os.path.join(REPO, "results", "irand_cell_heatmaps"),
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--all-features",
        action="store_true",
        help="svd5/10, wnmf5/10 hepsini bu K için üret",
    )
    args = p.parse_args()

    cells: List[Tuple[str, int, int]] = []
    if args.all_features:
        for feat, dim in (("svd", 5), ("svd", 10), ("wnmf", 5), ("wnmf", 10)):
            cells.append((feat, dim, args.k))
    elif args.feature_tag:
        f, d = parse_feature_tag(args.feature_tag)
        cells.append((f, d, args.k))
    else:
        if args.feature is None or args.dim is None:
            p.error("--feature ve --dim veya --feature-tag gerekli")
        cells.append((args.feature, args.dim, args.k))

    for feat, dim, k in cells:
        export_cell(
            feat, dim, k, args.algos, os.path.normpath(args.out_dir),
            do_plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
