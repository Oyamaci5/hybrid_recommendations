"""
euc_nogs_none WNMF×K grid CSV'lerinden çok panelli grafikler.

  python experiments/plot_euc_nogs_none_wnmf_k_grid.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PREDS = REPO / "results" / "euc_nogs_none_wnmf_k_grid_preds.csv"
PREDS_CV5_MEAN = REPO / "results" / "euc_nogs_none_wnmf_k_grid_preds_cv5_mean.csv"
CLUSTER = REPO / "results" / "euc_nogs_none_wnmf_k_grid_cluster_pairs.csv"
ASSIGN = REPO / "results" / "euc_nogs_none_wnmf_k_grid_assignment_metrics.csv"
HOPKINS = REPO / "results" / "euc_nogs_none_wnmf_k_grid_hopkins_cv.csv"
STATS = REPO / "results" / "euc_nogs_none_wnmf_k_grid_bootstrap_wilcoxon.csv"
PLOT_DIR = REPO / "results" / "plots" / "euc_nogs_none_wnmf_k_grid"

ALGOS = ["LIT_GWO", "IWO_HHO", "HA_AVOAHGS", "LIT_PSO", "B_AVOA", "B1_HHO", "B2_HGS"]
COLORS = {
    "HA_AVOAHGS": "#E63946",
    "B1_HHO": "#457B9D",
    "B_AVOA": "#2A9D8F",
    "LIT_PSO": "#E9C46A",
    "LIT_GWO": "#8338EC",
    "IWO_HHO": "#F4A261",
    "B2_HGS": "#1D3557",
}

PRED_METRICS = [
    ("mae", "MAE", "min"),
    ("rmse", "RMSE", "min"),
    ("precision_at_10", "Precision@10", "max"),
    ("recall_at_10", "Recall@10", "max"),
    ("ndcg_at_10", "NDCG@10", "max"),
    ("jaccard_at_10", "Jaccard@10", "max"),
    ("coverage_at_10", "Coverage@10", "max"),
]

def aggregate_cv5(df: pd.DataFrame) -> pd.DataFrame:
    """Fold başına satırları (wnmf, k, algo, predictor, knn_k) için ortala."""
    if "fold" not in df.columns or df["fold"].nunique() <= 1:
        return df
    num_cols = [
        c for c in (
            "mae", "rmse", "ndcg_at_10", "precision_at_10", "recall_at_10",
            "jaccard_at_10", "coverage_at_10",
        )
        if c in df.columns
    ]
    keys = ["wnmf_dim", "k", "algo", "predictor", "knn_k"]
    for c in ("similarity", "protocol"):
        if c in df.columns:
            keys.append(c)
    agg = df.groupby(keys, as_index=False)[num_cols].mean()
    agg["fold"] = 0
    return agg


STRUCT_METRICS = [
    ("wcss", "WCSS", "min"),
    ("silhouette_euclidean", "Silhouette (eucl)", "max"),
    ("silhouette_cosine", "Silhouette (cos)", "max"),
    ("ari", "ARI (pair mean)", "max"),
    ("label_agreement", "Label agreement", "max"),
]


def _best_per_group(df: pd.DataFrame, group_cols: list, col: str, how: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if sub.empty or col not in sub.columns:
            continue
        if isinstance(keys, tuple):
            key_dict = dict(zip(group_cols, keys))
        else:
            key_dict = {group_cols[0]: keys}
        idx = sub[col].idxmin() if how == "min" else sub[col].idxmax()
        rows.append({**key_dict, **sub.loc[idx].to_dict(), "best_metric": col})
    return pd.DataFrame(rows)


def plot_pred_envelope(
    df: pd.DataFrame,
    predictor: str,
    knn_k: int | None,
    out_path: Path,
    *,
    knn_fixed: int = 50,
) -> None:
    if predictor == "cluster_avg":
        sub = df[df["predictor"] == predictor]
    elif knn_k is not None:
        sub = df[(df["predictor"] == predictor) & (df["knn_k"] == knn_k)]
    else:
        sub = df[(df["predictor"] == predictor) & (df["knn_k"] != knn_fixed)]
    if sub.empty:
        return
    n_met = len(PRED_METRICS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    wnmf_vals = sorted(sub["wnmf_dim"].unique())

    for ax, (col, label, how) in zip(axes, PRED_METRICS):
        for algo in ALGOS:
            a_sub = sub[sub["algo"] == algo]
            if a_sub.empty:
                continue
            best_vals = []
            for w in wnmf_vals:
                cell = a_sub[a_sub["wnmf_dim"] == w]
                if cell.empty:
                    best_vals.append(np.nan)
                    continue
                idx = cell[col].idxmin() if how == "min" else cell[col].idxmax()
                best_vals.append(float(cell.loc[idx, col]))
            ax.plot(wnmf_vals, best_vals, "o-", label=algo, color=COLORS.get(algo), linewidth=1.5)
        ax.set_xlabel("WNMF dim")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for j in range(len(PRED_METRICS), len(axes)):
        axes[j].set_visible(False)

    knn_title = knn_k if knn_k is not None else "cluster_min"
    fig.suptitle(f"{predictor} knn={knn_title} — en iyi değer (K içinde)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_heatmap_k_wnmf(df: pd.DataFrame, predictor: str, knn_k: int, algo: str, out_path: Path) -> None:
    sub = df[(df["predictor"] == predictor) & (df["knn_k"] == knn_k) & (df["algo"] == algo)]
    if sub.empty:
        return
    n_met = min(4, len(PRED_METRICS))
    fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 4))
    if n_met == 1:
        axes = [axes]
    ks = sorted(sub["k"].unique())
    ws = sorted(sub["wnmf_dim"].unique())

    for ax, (col, label, _) in zip(axes, PRED_METRICS[:n_met]):
        mat = np.full((len(ws), len(ks)), np.nan)
        for i, w in enumerate(ws):
            for j, k in enumerate(ks):
                cell = sub[(sub["wnmf_dim"] == w) & (sub["k"] == k)]
                if not cell.empty:
                    mat[i, j] = float(cell[col].iloc[0])
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_yticks(range(len(ws)))
        ax.set_yticklabels(ws)
        ax.set_xlabel("K")
        ax.set_ylabel("WNMF")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{algo} | {predictor} knn={knn_k}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_assignment_metrics(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label, how in zip(
        axes,
        ["wcss", "silhouette_euclidean", "cluster_min"],
        ["WCSS", "Silhouette", "Cluster min size"],
        ["min", "max", "max"],
    ):
        if col not in df.columns:
            continue
        for algo in ALGOS:
            a_sub = df[df["algo"] == algo]
            if a_sub.empty:
                continue
            by_k = []
            for k in sorted(a_sub["k"].unique()):
                cell = a_sub[a_sub["k"] == k]
                if cell.empty:
                    continue
                idx = cell[col].idxmin() if how == "min" else cell[col].idxmax()
                by_k.append((k, float(cell.loc[idx, col])))
            if by_k:
                xs, ys = zip(*by_k)
                ax.plot(xs, ys, "o-", label=algo, color=COLORS.get(algo))
        ax.set_xlabel("K")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Assignment metrics (WNMF içinde en iyi algo / K)")
    fig.legend(loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cluster_similarity(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in zip(axes, ["ari", "label_agreement"], ["ARI", "Label agreement"]):
        if col not in df.columns:
            continue
        for w in sorted(df["wnmf_dim"].unique())[:3]:
            sub = df[df["wnmf_dim"] == w]
            by_k = sub.groupby("k")[col].mean()
            ax.plot(by_k.index, by_k.values, "o-", label=f"wnmf{w}")
        ax.set_xlabel("K")
        ax.set_ylabel(label)
        ax.set_title(f"Ortalama {label} (algo çiftleri)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Algoritma çiftleri — küme benzerliği")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_hopkins_cv(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = df["wnmf_dim"].values
    axes[0].plot(x, df["hopkins_raw"], "o-", label="raw")
    axes[0].plot(x, df["hopkins_norm"], "s-", label="norm")
    if "hopkins_target" in df.columns:
        axes[0].axhline(float(df["hopkins_target"].iloc[0]), color="gray", ls="--", label="H=0.85")
    axes[0].set_xlabel("WNMF dim")
    axes[0].set_ylabel("Hopkins")
    axes[0].set_title("Hopkins")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, df["cv_raw"], "o-", label="CV raw")
    axes[1].plot(x, df["cv_norm"], "s-", label="CV norm")
    if "cv_raw_ref" in df.columns:
        axes[1].axhline(float(df["cv_raw_ref"].iloc[0]), color="C0", ls=":", alpha=0.7)
    if "cv_norm_ref" in df.columns:
        axes[1].axhline(float(df["cv_norm_ref"].iloc[0]), color="C1", ls=":", alpha=0.7)
    axes[1].set_xlabel("WNMF dim")
    axes[1].set_ylabel("CV")
    axes[1].set_title("CV (raw vs norm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_wilcoxon_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "wilcoxon_p" not in df.columns:
        return
    sub = df[(df["metric"] == "mae") & (df["predictor"] == "cluster_knn_native")]
    if sub.empty:
        return
    pivot = sub.pivot_table(
        index=["wnmf_dim", "k"],
        columns="algo_b",
        values="wilcoxon_p",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(pivot))))
    im = ax.imshow(-np.log10(pivot.values.clip(1e-300)), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ylabels = [f"w{w}_k{k}" for w, k in pivot.index]
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(f"−log10(p) Wilcoxon MAE vs {BOOT_REFERENCE}")
    fig.colorbar(im, ax=ax, label="−log10(p)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


BOOT_REFERENCE = "HA_AVOAHGS"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=PREDS)
    ap.add_argument(
        "--cv5-mean", action="store_true",
        help="Grafiklerde 5-fold ortalaması (preds_cv5_mean.csv veya preds üzerinden)",
    )
    ap.add_argument("--out-dir", type=Path, default=PLOT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0

    if args.preds.is_file():
        if args.cv5_mean and PREDS_CV5_MEAN.is_file():
            preds = pd.read_csv(PREDS_CV5_MEAN)
        else:
            preds = pd.read_csv(args.preds)
            if args.cv5_mean:
                preds = aggregate_cv5(preds)
        for predictor, knn_k, tag in (
            ("cluster_avg", 0, "cluster_avg"),
            ("cluster_knn_native", None, "cluster_knn_native_kmin"),
            ("cluster_knn_native", 50, "cluster_knn_native_k50"),
        ):
            if predictor == "cluster_avg":
                mask = preds["predictor"] == predictor
            elif knn_k is None:
                mask = (preds["predictor"] == predictor) & (preds["knn_k"] != 50)
            else:
                mask = (preds["predictor"] == predictor) & (preds["knn_k"] == knn_k)
            if not preds[mask].empty:
                plot_pred_envelope(
                    preds, predictor, knn_k if predictor != "cluster_avg" else 0,
                    args.out_dir / f"envelope_{tag}.png",
                )
                n_plots += 1
        ref_algo = "HA_AVOAHGS"
        plot_heatmap_k_wnmf(
            preds, "cluster_avg", 0, ref_algo,
            args.out_dir / f"heatmap_{ref_algo}_cluster_avg.png",
        )
        n_plots += 1

    if ASSIGN.is_file():
        plot_assignment_metrics(pd.read_csv(ASSIGN), args.out_dir / "assignment_metrics.png")
        n_plots += 1

    if CLUSTER.is_file():
        plot_cluster_similarity(pd.read_csv(CLUSTER), args.out_dir / "cluster_similarity.png")
        n_plots += 1

    if HOPKINS.is_file():
        plot_hopkins_cv(pd.read_csv(HOPKINS), args.out_dir / "hopkins_cv.png")
        n_plots += 1

    if STATS.is_file():
        plot_wilcoxon_heatmap(pd.read_csv(STATS), args.out_dir / "wilcoxon_mae_heatmap.png")
        n_plots += 1

    print(f"Grafikler -> {args.out_dir} ({n_plots} dosya grubu)")


if __name__ == "__main__":
    main()
