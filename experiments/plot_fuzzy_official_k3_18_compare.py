"""
FCM official fold-1, K=3..18, cluster_avg_soft: 8 algo x 5 metrik.

  python -m experiments.plot_fuzzy_official_k3_18_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import ALL_ALGOS
from experiments.run_fuzzy_official_k3_24_cluster_avg import MERGE_CSVS, OUT_CSV

DEFAULT_K_MIN = 3
DEFAULT_K_MAX = 18
OUT_DIR = REPO / "results" / "plots"
SUMMARY_CSV = REPO / "results" / "fuzzy_official_k3_18_compare_summary.csv"

METRICS = [
    ("mae", "MAE", "min"),
    ("rmse", "RMSE", "min"),
    ("ndcg_at_10", "NDCG@10", "max"),
    ("precision_at_10", "P@10", "max"),
    ("recall_at_10", "R@10", "max"),
]

COLORS = {
    "HA_AVOAHGS": "#E63946",
    "LIT_GWO": "#8338EC",
    "B_AVOA": "#2A9D8F",
    "LIT_PSO": "#E9C46A",
    "IWO_HHO": "#F4A261",
    "H9_QSA+CDO": "#264653",
    "B1_HHO": "#457B9D",
    "B2_HGS": "#9B5DE5",
}


def load_eval_df(
    k_min: int,
    k_max: int,
    *,
    fold: int | None = None,
    in_csv: Path | None = None,
) -> pd.DataFrame:
    paths = [in_csv] if in_csv else [OUT_CSV, *MERGE_CSVS]
    paths = [p for p in paths if p is not None and p.is_file()]
    parts = []
    for p in paths:
        if p.is_file():
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError("Eval CSV bulunamadi")
    df = pd.concat(parts, ignore_index=True)
    if "fast" not in df.columns:
        df["fast"] = False
    if "prune" not in df.columns:
        df["prune"] = False
    if "knn_k" not in df.columns:
        df["knn_k"] = 0
    sub = df[
        (df.get("predictor", "") == "cluster_avg_soft")
        & (df.get("similarity", "") == "cosine")
        & (df["fast"] == True)
        & (df["prune"] == False)
        & (df["k"].between(k_min, k_max))
    ]
    if fold is not None:
        sub = sub[df["fold"] == int(fold)]
    sub = sub.copy()
    key = ["fold", "k", "algo", "predictor", "knn_k", "similarity", "fast", "prune"]
    sub = sub.drop_duplicates(subset=key, keep="last")
    return sub.sort_values(["k", "algo"]).reset_index(drop=True)


def best_per_algo_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo in ALL_ALGOS:
        a = df[df["algo"] == algo]
        if a.empty:
            continue
        row = {"algo": algo}
        for col, label, how in METRICS:
            idx = a[col].idxmin() if how == "min" else a[col].idxmax()
            r = a.loc[idx]
            row[f"best_{col}"] = r[col]
            row[f"best_k_{col}"] = int(r["k"])
        row["mean_mae"] = a["mae"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values("mean_mae")
    return out


def plot_5metrics_k_sweep(
    df: pd.DataFrame,
    out_png: Path,
    *,
    k_list: list[int],
    fold: int | None = None,
    title: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    if title:
        fig.suptitle(title, fontsize=12, y=1.08)
    else:
        fold_lbl = f"fold-{fold}" if fold is not None else "fold-1"
        fig.suptitle(
            f"FCM official {fold_lbl} — fuzzy/imkpp/wnmf50/m1.5, cluster_avg_soft (cosine)\n"
            f"K={k_list[0]}..{k_list[-1]}, fast estop, 8 algoritma",
        fontsize=12,
        y=1.08,
    )
    for mi, (col, label, _) in enumerate(METRICS):
        ax = axes[mi]
        for algo in ALL_ALGOS:
            a = df[df["algo"] == algo].sort_values("k")
            if a.empty:
                continue
            ax.plot(
                a["k"],
                a[col],
                "o-",
                label=algo,
                color=COLORS.get(algo, None),
                lw=1.8,
                ms=5,
            )
        ax.set_title(label)
        ax.set_xlabel("K")
        ax.set_xticks(k_list)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.set_ylabel(label)
        if mi == 4:
            ax.legend(fontsize=6, loc="best", ncol=1)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mae_heatmap(df: pd.DataFrame, out_png: Path, *, k_list: list[int]) -> None:
    pivot = df.pivot(index="algo", columns="k", values="mae")
    pivot = pivot.reindex(ALL_ALGOS)[k_list]
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(k_list)))
    ax.set_xticklabels(k_list)
    ax.set_yticks(range(len(ALL_ALGOS)))
    ax.set_yticklabels(ALL_ALGOS)
    ax.set_xlabel("K")
    ax.set_title("MAE — algo x K (dusuk = iyi)")
    for i, algo in enumerate(pivot.index):
        for j, k in enumerate(pivot.columns):
            v = pivot.loc[algo, k]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_k_bar(df: pd.DataFrame, out_png: Path) -> None:
    """Algo basina min-MAE K satirindaki 5 metrik."""
    rows = []
    for algo in ALL_ALGOS:
        a = df[df["algo"] == algo]
        if a.empty:
            continue
        best = a.loc[a["mae"].idxmin()]
        rows.append([algo] + [best[c] for c, _, _ in METRICS])
    if not rows:
        return
    data = pd.DataFrame(rows, columns=["algo"] + [m[0] for m in METRICS])
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("En iyi K (min MAE) uzerinden 5 metrik — algo karsilastirma", fontsize=12)
    algos = data["algo"].tolist()
    x = range(len(algos))
    for ax, (col, label, _) in zip(axes, METRICS):
        ax.bar(x, data[col], color=[COLORS.get(a, "#888") for a in algos])
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=35, ha="right", fontsize=7)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    ap.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--in-csv", type=Path, default=None)
    args = ap.parse_args()
    k_list = list(range(args.k_min, args.k_max + 1))

    df = load_eval_df(args.k_min, args.k_max, fold=args.fold, in_csv=args.in_csv)
    n_expected = len(k_list) * len(ALL_ALGOS)
    if len(df) < n_expected:
        print(f"Uyari: {len(df)}/{n_expected} satir (eksik kombinasyon olabilir)")

    summary = best_per_algo_summary(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"summary -> {SUMMARY_CSV}")
    print(summary[["algo", "mean_mae", "best_mae", "best_k_mae", "best_ndcg_at_10"]].to_string(index=False))

    fold_tag = f"f{args.fold}_" if args.fold is not None else ""
    tag = f"{fold_tag}k{args.k_min}_{args.k_max}"
    p1 = OUT_DIR / f"fuzzy_official_{tag}_5metrics_cluster_avg_soft.png"
    p2 = OUT_DIR / f"fuzzy_official_{tag}_mae_heatmap.png"
    p3 = OUT_DIR / f"fuzzy_official_{tag}_bestk_5metrics_bar.png"
    plot_5metrics_k_sweep(df, p1, k_list=k_list, fold=args.fold)
    plot_mae_heatmap(df, p2, k_list=k_list)
    plot_best_k_bar(df, p3)
    print(f"plot -> {p1}")
    print(f"plot -> {p2}")
    print(f"plot -> {p3}")


if __name__ == "__main__":
    main()
