"""
euc_kmref sweep CSV — en iyi MAE/NDCG/RMSE/P/R + grafikler (tam algo adları).

  python experiments/plot_euc_kmref_k_sweep_best.py
  python experiments/plot_euc_kmref_k_sweep_best.py --csv results/euc_kmref_k_sweep_preds_w50_f5.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "results" / "euc_kmref_k_sweep_preds_w20_cv5.csv"

ALGOS = ["HA_AVOAHGS", "B1_HHO", "B_AVOA", "LIT_PSO", "LIT_GWO"]
K_LIST = list(range(3, 15))

PLOT_COLORS = {
    "HA_AVOAHGS": "#E63946",
    "B1_HHO": "#457B9D",
    "B_AVOA": "#2A9D8F",
    "LIT_PSO": "#E9C46A",
    "LIT_GWO": "#8338EC",
}

METRICS = [
    ("mae", "MAE", "min"),
    ("rmse", "RMSE", "min"),
    ("ndcg_at_10", "NDCG@10", "max"),
    ("precision_at_10", "Precision@10", "max"),
    ("recall_at_10", "Recall@10", "max"),
    ("coverage_at_10", "Coverage@10", "max"),
]


def _paths_for_csv(in_csv: Path) -> tuple[Path, Path, Path, Path]:
    stem = in_csv.stem.replace("euc_kmref_k_sweep_preds_", "")
    tag = stem if stem else "default"
    res = REPO / "results"
    return (
        res / f"euc_kmref_k_sweep_best_summary_{tag}.csv",
        res / f"euc_kmref_k_sweep_best_overall_{tag}.csv",
        res / "plots" / f"euc_kmref_k_sweep_best_per_algo_{tag}.png",
        res / "plots" / f"euc_kmref_k_sweep_best_envelope_{tag}.png",
    )


def _aggregate_cv5(df: pd.DataFrame) -> pd.DataFrame:
    """Çoklu fold: (k, algo, predictor, knn_k) başına metrik ortalaması."""
    if "fold" not in df.columns or df["fold"].nunique() <= 1:
        return df
    num_cols = [
        c for c in (
            "mae", "rmse", "ndcg_at_10", "precision_at_10",
            "recall_at_10", "coverage_at_10",
        )
        if c in df.columns
    ]
    keys = ["k", "algo", "predictor", "knn_k"]
    for c in ("wnmf_dim", "kmref", "protocol", "label"):
        if c in df.columns:
            keys.append(c)
    agg = df.groupby(keys, as_index=False)[num_cols].mean()
    if "assign_suffix" in df.columns:
        agg["assign_suffix"] = "cv5_mean"
    return agg


def _best_row(df: pd.DataFrame, col: str, how: str) -> pd.Series:
    idx = df[col].idxmin() if how == "min" else df[col].idxmax()
    return df.loc[idx]


def best_per_k(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k]
        for col, label, how in METRICS:
            r = _best_row(sub, col, how)
            rows.append({
                "k": k,
                "metric": col,
                "metric_label": label,
                "direction": how,
                "value": r[col],
                "algo": r["algo"],
                "label": r.get("label", r["algo"]),
                "predictor": r["predictor"],
                "knn_k": int(r["knn_k"]),
            })
    return pd.DataFrame(rows)


def best_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label, how in METRICS:
        r = _best_row(df, col, how)
        rows.append({
            "metric": col,
            "metric_label": label,
            "direction": how,
            "value": r[col],
            "k": int(r["k"]),
            "algo": r["algo"],
            "label": r.get("label", r["algo"]),
            "predictor": r["predictor"],
            "knn_k": int(r["knn_k"]),
            "mae": r["mae"],
            "rmse": r["rmse"],
            "ndcg_at_10": r["ndcg_at_10"],
            "precision_at_10": r["precision_at_10"],
            "recall_at_10": r["recall_at_10"],
        })
    return pd.DataFrame(rows)


def best_per_algo_per_k(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        for k in K_LIST:
            sub = df[(df["algo"] == algo) & (df["k"] == k)]
            if sub.empty:
                continue
            rows.append(_best_row(sub, "mae", "min"))
    return pd.DataFrame(rows)


def _plot_panel(ax, sub: pd.DataFrame, col: str, ylabel: str, title: str, marker: str) -> None:
    for algo in ALGOS:
        s = sub[sub["algo"] == algo].sort_values("k")
        if s.empty:
            continue
        ax.plot(
            s["k"], s[col], f"{marker}-",
            label=algo, color=PLOT_COLORS.get(algo), linewidth=2, markersize=7,
        )
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)


def plot_best_per_algo(df_best: pd.DataFrame, out_plot: Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"EUC + kmref — K başına en iyi MAE satırı ({title_suffix})",
        fontsize=12, y=1.01,
    )
    panels = [
        ("mae", "MAE", "MAE", "o"),
        ("rmse", "RMSE", "RMSE", "o"),
        ("ndcg_at_10", "NDCG@10", "NDCG@10", "s"),
        ("precision_at_10", "Precision@10", "P@10", "D"),
        ("recall_at_10", "Recall@10", "R@10", "D"),
        ("coverage_at_10", "Coverage@10", "Cov@10", "^"),
    ]
    for ax, (col, ylabel, ttl, mk) in zip(axes.flat, panels):
        if col not in df_best.columns:
            ax.axis("off")
            continue
        _plot_panel(ax, df_best, col, ylabel, ttl, mk)
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_envelope(per_k: pd.DataFrame, out_plot: Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"EUC + kmref — her K için global en iyi ({title_suffix})",
        fontsize=12, y=1.01,
    )
    for ax, (col, label, how) in zip(axes.flat, METRICS):
        sub = per_k[per_k["metric"] == col].sort_values("k")
        ax.plot(sub["k"], sub["value"], "k-o", linewidth=2, markersize=7)
        for _, r in sub.iterrows():
            ax.annotate(
                r["algo"],
                (r["k"], r["value"]),
                textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=6, rotation=45,
            )
        ax.set_xlabel("K")
        ax.set_ylabel(label)
        ax.set_title(f"{'min' if how == 'min' else 'max'} {label}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(K_LIST)
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_IN)
    args = ap.parse_args()
    in_csv = args.csv
    if not in_csv.is_file():
        raise SystemExit(f"CSV yok: {in_csv}")

    out_sum, out_all, out_p1, out_p2 = _paths_for_csv(in_csv)
    title_suffix = in_csv.stem.replace("euc_kmref_k_sweep_preds_", "")

    df = pd.read_csv(in_csv)
    if "label" in df.columns:
        df["label"] = df["algo"]
    n_folds = df["fold"].nunique() if "fold" in df.columns else 1
    if n_folds > 1:
        print(f"CV5: {n_folds} fold ortalaması alınıyor")
        df = _aggregate_cv5(df)

    per_k = best_per_k(df)
    overall = best_overall(df)
    df_best_algo = best_per_algo_per_k(df)

    per_k.to_csv(out_sum, index=False)
    overall.to_csv(out_all, index=False)
    plot_best_per_algo(df_best_algo, out_p1, title_suffix)
    plot_envelope(per_k, out_p2, title_suffix)

    fold = df["fold"].iloc[0] if "fold" in df.columns else "?"
    wdim = df["wnmf_dim"].iloc[0] if "wnmf_dim" in df.columns else "?"

    print(f"folds={n_folds}  wnmf_dim={wdim}  ({len(df)} satır)")
    print("=== Genel en iyi ===")
    for _, r in overall.iterrows():
        pred = r["predictor"]
        if int(r["knn_k"]) > 0:
            pred += f" knn_k={int(r['knn_k'])}"
        print(f"  {r['metric_label']:14s} {r['value']:.4f}  K={int(r['k'])} {r['algo']} {pred}")

    print(f"\nÖzet: {out_sum}")
    print(f"Grafik: {out_p1}")
    print(f"Grafik: {out_p2}")


if __name__ == "__main__":
    main()
