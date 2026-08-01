"""Algo ayrismasi tanisi: neden cizgiler ust uste? ARI + delta heatmap."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.run_euc_kmref_k_sweep import ALGOS, K_LIST, assign_dir

CSV = REPO / "results" / "euc_kmref_k_sweep_preds_w20_cv5_official.csv"
OUT = REPO / "results" / "plots"


def ari_matrix(fold: int, k: int) -> np.ndarray:
    assigns = {}
    for algo in ALGOS:
        d = assign_dir(algo, k, wnmf_dim=20, assign_fold=fold)
        if d:
            assigns[algo] = np.load(d / "assignments.npy")
    n = len(ALGOS)
    m = np.eye(n)
    for i, a1 in enumerate(ALGOS):
        for j, a2 in enumerate(ALGOS):
            if i < j and a1 in assigns and a2 in assigns:
                v = adjusted_rand_score(assigns[a1], assigns[a2])
                m[i, j] = m[j, i] = v
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    ca = df[(df["fold"].isin([1, 2])) & (df["predictor"] == "cluster_avg") & (df["knn_k"] == 0)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Ayrisma tanisi: algo metrikleri birbirine cok yakin (kmref + ayni WNMF uzayi)",
        fontsize=12,
    )

    # 1) MAE delta from K-mean across algos
    for fi, fold in enumerate([1, 2]):
        ax = axes[fi, 0]
        sub = ca[ca["fold"] == fold]
        pivot = sub.pivot(index="k", columns="algo", values="mae")
        delta = pivot.sub(pivot.mean(axis=1), axis=0)
        im = ax.imshow(delta.values.T, aspect="auto", cmap="RdBu_r", vmin=-0.015, vmax=0.015)
        ax.set_yticks(range(len(ALGOS)))
        ax.set_yticklabels(ALGOS, fontsize=8)
        ax.set_xticks(range(len(K_LIST)))
        ax.set_xticklabels(K_LIST)
        ax.set_xlabel("K")
        ax.set_title(f"Fold {fold}: MAE - ort(algo) (max ~0.01)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 2) K effect (large separation)
    ax = axes[0, 1]
    for fold, ls in [(1, "-"), (2, "--")]:
        sub = ca[ca["fold"] == fold].groupby("k")["mae"].mean()
        ax.plot(sub.index, sub.values, ls, lw=2, label=f"fold {fold} ort(algo)")
    ax.set_xlabel("K")
    ax.set_ylabel("MAE")
    ax.set_title("K etkisi BUYUK (0.76 -> 0.80)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Predictor effect at K=3
    ax = axes[1, 1]
    sub = df[(df["fold"] == 2) & (df["k"] == 3)]
    preds = ["cluster_avg", "cluster_avg_hard", "cluster_knn_native",
             "cluster_knn_surprise_baseline", "cluster_knn_with_means"]
    vals = []
    for p in preds:
        s = sub[sub["predictor"] == p]
        if p.startswith("cluster_knn"):
            s = s[s["knn_k"] == 30]
        else:
            s = s[s["knn_k"] == 0]
        vals.append(s["mae"].mean() if not s.empty else np.nan)
    ax.barh(preds, vals, color=["#457B9D", "#E9C46A", "#2A9D8F", "#E63946", "#8338EC"])
    ax.set_xlabel("MAE (fold2, K=3, ort algo)")
    ax.set_title("Predictor ayrismasi VAR (~0.07 fark)")
    ax.grid(True, axis="x", alpha=0.3)

    # 4) ARI heatmaps fold1/f2 K=7
    for fi, (fold, k) in enumerate([(1, 7), (2, 7)]):
        ax = axes[fi, 2]
        m = ari_matrix(fold, k)
        im = ax.imshow(m, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(len(ALGOS)))
        ax.set_yticks(range(len(ALGOS)))
        ax.set_xticklabels(ALGOS, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ALGOS, fontsize=7)
        ax.set_title(f"Atama ARI K={k} fold{fold}\n(0.4-0.6: farkli ama benzer tahmin)")
        for i in range(len(ALGOS)):
            for j in range(len(ALGOS)):
                ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    p = OUT / "euc_kmref_official_separation_diagnosis.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
