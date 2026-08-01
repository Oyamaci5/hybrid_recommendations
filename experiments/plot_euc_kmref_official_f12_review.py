"""
Official CV5 fold 1-2: en iyi sonuclar, 5 metrik, kume dagilimi, fold karsilastirma.

  python experiments/plot_euc_kmref_official_f12_review.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.run_euc_kmref_k_sweep import ALGOS, K_LIST, assign_dir

CSV_IN = REPO / "results" / "euc_kmref_k_sweep_preds_w20_cv5_official.csv"
OUT_DIR = REPO / "results" / "plots"
SUMMARY_CSV = REPO / "results" / "euc_kmref_official_f12_best_summary.csv"
CLUSTER_CSV = REPO / "results" / "euc_kmref_official_f12_cluster_compare.csv"

FOLDS = [1, 2]
METRICS = [
    ("mae", "MAE", "min"),
    ("rmse", "RMSE", "min"),
    ("ndcg_at_10", "NDCG@10", "max"),
    ("precision_at_10", "P@10", "max"),
    ("recall_at_10", "R@10", "max"),
]
COLORS = {
    "HA_AVOAHGS": "#E63946",
    "B1_HHO": "#457B9D",
    "B_AVOA": "#2A9D8F",
    "LIT_PSO": "#E9C46A",
    "LIT_GWO": "#8338EC",
}


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_IN)
    return df[df["fold"].isin(FOLDS)].copy()


def best_overall(df: pd.DataFrame, fold: int) -> pd.DataFrame:
    sub = df[df["fold"] == fold]
    rows = []
    for col, label, how in METRICS:
        idx = sub[col].idxmin() if how == "min" else sub[col].idxmax()
        r = sub.loc[idx]
        rows.append({
            "fold": fold,
            "metric": label,
            "value": r[col],
            "algo": r["algo"],
            "k": int(r["k"]),
            "predictor": r["predictor"],
            "knn_k": int(r["knn_k"]),
        })
    return pd.DataFrame(rows)


def best_per_algo_cluster_avg(df: pd.DataFrame, fold: int) -> pd.DataFrame:
    sub = df[(df["fold"] == fold) & (df["predictor"] == "cluster_avg") & (df["knn_k"] == 0)]
    rows = []
    for algo in ALGOS:
        a = sub[sub["algo"] == algo]
        if a.empty:
            continue
        row = {"fold": fold, "algo": algo}
        for col, label, how in METRICS:
            idx = a[col].idxmin() if how == "min" else a[col].idxmax()
            r = a.loc[idx]
            row[f"best_{col}"] = r[col]
            row[f"best_k_{col}"] = int(r["k"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_cluster_sizes(algo: str, k: int, fold: int) -> list[int]:
    adir = assign_dir(algo, k, wnmf_dim=20, assign_fold=fold)
    if adir is None:
        return []
    arr = np.load(adir / "assignments.npy")
    return sorted(Counter(arr.astype(int).tolist()).values(), reverse=True)


def plot_5metrics_per_algo(df: pd.DataFrame) -> Path:
    """cluster_avg: K sweep, 5 metrik, fold1 vs fold2."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex="col")
    fig.suptitle("Official CV5 — cluster_avg (5 metrik, fold 1 vs 2)", fontsize=14, y=1.02)

    for fi, fold in enumerate(FOLDS):
        sub = df[(df["fold"] == fold) & (df["predictor"] == "cluster_avg") & (df["knn_k"] == 0)]
        for mi, (col, label, how) in enumerate(METRICS):
            ax = axes[fi, mi]
            for algo in ALGOS:
                a = sub[sub["algo"] == algo].sort_values("k")
                if a.empty:
                    continue
                ax.plot(a["k"], a[col], "o-", label=algo, color=COLORS[algo], lw=1.8, ms=5)
            ax.set_title(f"Fold {fold} — {label}")
            ax.set_xlabel("K")
            ax.grid(True, alpha=0.3)
            if mi == 0:
                ax.set_ylabel(label)
            if fi == 0 and mi == 4:
                ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    out = OUT_DIR / "euc_kmref_official_f12_5metrics_cluster_avg.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_best_envelope(df: pd.DataFrame) -> Path:
    """Algo basina en iyi K (cluster_avg MAE) uzerinden 5 metrik karsilastirma."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("En iyi K (cluster_avg, min MAE) — 5 metrik karsilastirma", fontsize=13)

    metric_cols = [m[0] for m in METRICS]
    metric_labels = [m[1] for m in METRICS]

    for ax_i, fold in enumerate(FOLDS):
        ax = axes[ax_i]
        sub = df[(df["fold"] == fold) & (df["predictor"] == "cluster_avg") & (df["knn_k"] == 0)]
        rows = []
        for algo in ALGOS:
            a = sub[sub["algo"] == algo]
            if a.empty:
                continue
            best = a.loc[a["mae"].idxmin()]
            rows.append([best[c] for c in metric_cols])
        if not rows:
            continue
        data = np.array(rows)
        x = np.arange(len(ALGOS))
        w = 0.15
        for mi, (col, lab, how) in enumerate(METRICS):
            vals = data[:, mi]
            if how == "min" and col in ("mae", "rmse"):
                pass
            ax.bar(x + mi * w, vals, width=w, label=lab)
        ax.set_xticks(x + w * 2)
        ax.set_xticklabels(ALGOS, rotation=25, ha="right")
        ax.set_title(f"Fold {fold}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "euc_kmref_official_f12_best_k_mae_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cluster_stats(df: pd.DataFrame) -> Path:
    """cluster_min / max / std vs K — fold1 vs fold2 (HA or best algo)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle("Kume boyutu istatistikleri (cluster_avg hucreleri)", fontsize=13)
    stat_cols = [
        ("cluster_min", "Min kume boyutu"),
        ("cluster_max", "Max kume boyutu"),
        ("cluster_std", "Std kume boyutu"),
    ]
    for col_i, (col, title) in enumerate(stat_cols):
        for fi, fold in enumerate(FOLDS):
            ax = axes[col_i, fi]
            sub = df[(df["fold"] == fold) & (df["predictor"] == "cluster_avg") & (df["knn_k"] == 0)]
            for algo in ALGOS:
                a = sub[sub["algo"] == algo].sort_values("k")
                if a.empty:
                    continue
                ax.plot(a["k"], a[col], "o-", label=algo, color=COLORS[algo], lw=1.5, ms=4)
            ax.set_title(f"{title} — Fold {fold}")
            ax.grid(True, alpha=0.3)
            if col_i == 2:
                ax.set_xlabel("K")
            if fi == 0 and col_i == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    out = OUT_DIR / "euc_kmref_official_f12_cluster_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cluster_size_bars(fold: int, k: int = 7) -> Path | None:
    """Sabit K=7: algo basina kume boyut dagilimi (bar groups)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.15
    x_base = np.arange(k)
    any_data = False
    for ai, algo in enumerate(ALGOS):
        sizes = load_cluster_sizes(algo, k, fold)
        if not sizes or len(sizes) < k:
            continue
        any_data = True
        xs = x_base + ai * width
        ax.bar(xs, sizes, width=width, label=algo, color=COLORS[algo])
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("Cluster rank (0=largest)")
    ax.set_ylabel("Users in cluster")
    ax.set_title(f"Fold {fold} — K={k} kume boyutlari (algo karsilastirma)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"euc_kmref_official_f{fold}_k{k}_cluster_sizes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fold_cluster_compare(algo: str = "HA_AVOAHGS", k: int = 7) -> Path | None:
    """Ayni algo/K: fold1 vs fold2 kume boyut dagilimi."""
    s1 = load_cluster_sizes(algo, k, 1)
    s2 = load_cluster_sizes(algo, k, 2)
    if not s1 or not s2:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(max(len(s1), len(s2)))
    w = 0.35
    ax.bar(x - w / 2, s1 + [0] * (len(x) - len(s1)), width=w, label="Fold 1 (u1.base)", color="#457B9D")
    ax.bar(x + w / 2, s2 + [0] * (len(x) - len(s2)), width=w, label="Fold 2 (u2.base)", color="#E63946")
    ax.set_xlabel("Cluster index (sorted by size desc)")
    ax.set_ylabel("Users")
    ax.set_title(f"{algo} K={k} — Fold1 vs Fold2 kume dagilimi")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"euc_kmref_official_{algo}_k{k}_fold12_cluster_sizes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def cluster_compare_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = df[(df["predictor"] == "cluster_avg") & (df["knn_k"] == 0)]
    for algo in ALGOS:
        for k in K_LIST:
            for fold in FOLDS:
                r = sub[(sub["algo"] == algo) & (sub["k"] == k) & (sub["fold"] == fold)]
                if r.empty:
                    continue
                r = r.iloc[0]
                sizes = load_cluster_sizes(algo, k, fold)
                rows.append({
                    "algo": algo,
                    "k": k,
                    "fold": fold,
                    "n_active": int(r["n_active_clusters"]),
                    "cluster_min": int(r["cluster_min"]),
                    "cluster_max": int(r["cluster_max"]),
                    "cluster_std": round(float(r["cluster_std"]), 1),
                    "singletons": int(r["singletons"]),
                    "sizes_desc": str(sizes),
                    "mae": round(float(r["mae"]), 4),
                    "ndcg_at_10": round(float(r["ndcg_at_10"]), 4),
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    print(f"Yuklendi: {len(df)} satir (fold {FOLDS})")

    all_best = []
    all_algo = []
    for fold in FOLDS:
        n = len(df[df["fold"] == fold])
        print(f"\nFold {fold}: {n} satir (hedef 480)")
        bo = best_overall(df, fold)
        all_best.append(bo)
        print(bo.to_string(index=False))
        ba = best_per_algo_cluster_avg(df, fold)
        all_algo.append(ba)

    pd.concat(all_best).to_csv(SUMMARY_CSV, index=False)
    cluster_compare_table(df).to_csv(CLUSTER_CSV, index=False)

    paths = [
        plot_5metrics_per_algo(df),
        plot_best_envelope(df),
        plot_cluster_stats(df),
        plot_cluster_size_bars(1, 7),
        plot_cluster_size_bars(2, 7),
        plot_fold_cluster_compare("HA_AVOAHGS", 7),
        plot_fold_cluster_compare("HA_AVOAHGS", 3),
    ]
    print("\nGrafikler:")
    for p in paths:
        if p:
            print(f"  {p.relative_to(REPO)}")
    print(f"\nCSV: {SUMMARY_CSV.name}, {CLUSTER_CSV.name}")


if __name__ == "__main__":
    main()
