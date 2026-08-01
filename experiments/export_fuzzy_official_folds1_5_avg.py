"""
Fold 1-5 birlestir + K=3..24 x 8 algo ortalama tablolari.

  python -m experiments.export_fuzzy_official_folds1_5_avg
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import ALL_ALGOS, K3_24_LIST

SOURCE_FILES = [
    REPO / "results" / "fuzzy_official_folds2_5_k3_24_cluster_avg_soft.csv",
    REPO / "results" / "fuzzy_official_k3_24_cluster_avg_soft.csv",
    REPO / "results" / "fuzzy_official_fast_preds.csv",
    REPO / "results" / "fuzzy_official_k4_14_extra_preds.csv",
]

OUT_ALL = REPO / "results" / "fuzzy_official_folds1_5_k3_24_cluster_avg_soft.csv"
OUT_AVG = REPO / "results" / "fuzzy_official_folds1_5_avg_k3_24_cluster_avg_soft.csv"
OUT_PIVOT_MAE = REPO / "results" / "fuzzy_official_folds1_5_avg_mae_pivot.csv"
OUT_PIVOT_NDCG = REPO / "results" / "fuzzy_official_folds1_5_avg_ndcg_pivot.csv"

METRICS = ["mae", "rmse", "ndcg_at_10", "precision_at_10", "recall_at_10"]
DEDUP_KEY = ["fold", "k", "algo"]


def load_all_folds() -> pd.DataFrame:
    parts = []
    for p in SOURCE_FILES:
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "fold" not in df.columns:
            df["fold"] = 1
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
            & (df["k"].between(min(K3_24_LIST), max(K3_24_LIST)))
        ]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        raise FileNotFoundError("Kaynak eval CSV bulunamadi")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=DEDUP_KEY, keep="last")
    return out.sort_values(["fold", "k", "algo"]).reset_index(drop=True)


def build_avg_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["k", "algo"])
    agg = g[METRICS].mean().add_suffix("_avg_f1_5")
    for m in METRICS:
        agg[f"{m}_std_f1_5"] = g[m].std()
    agg["n_folds"] = g["fold"].nunique()
    return agg.reset_index().sort_values(["k", "mae_avg_f1_5"]).reset_index(drop=True)


def main() -> None:
    df = load_all_folds()
    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ALL, index=False)

    avg = build_avg_table(df)
    avg.to_csv(OUT_AVG, index=False)

    pivot_mae = avg.pivot_table(
        index="algo", columns="k", values="mae_avg_f1_5", aggfunc="first",
    ).reindex(ALL_ALGOS)
    pivot_mae.to_csv(OUT_PIVOT_MAE)

    pivot_ndcg = avg.pivot_table(
        index="algo", columns="k", values="ndcg_at_10_avg_f1_5", aggfunc="first",
    ).reindex(ALL_ALGOS)
    pivot_ndcg.to_csv(OUT_PIVOT_NDCG)

    print(f"Per-fold -> {OUT_ALL} ({len(df)} rows)")
    for f in sorted(df["fold"].unique()):
        print(f"  fold {int(f)}: {(df['fold']==f).sum()} satir")
    print(f"Avg f1-5 -> {OUT_AVG} ({len(avg)} rows)")
    print(f"MAE pivot -> {OUT_PIVOT_MAE}")
    print(f"NDCG pivot -> {OUT_PIVOT_NDCG}")

    best = avg.loc[avg.groupby("k")["mae_avg_f1_5"].idxmin()]
    print("\nEn iyi MAE (avg f1-5), ilk 10 K:")
    print(best[["k", "algo", "mae_avg_f1_5", "ndcg_at_10_avg_f1_5"]].head(10).to_string(index=False))

    _plot_avg(avg)


def _plot_avg(avg: pd.DataFrame) -> None:
    from experiments.plot_fuzzy_official_k3_18_compare import (
        OUT_DIR,
        plot_5metrics_k_sweep,
        plot_best_k_bar,
        plot_mae_heatmap,
    )

    plot_df = avg.rename(columns={
        "mae_avg_f1_5": "mae",
        "rmse_avg_f1_5": "rmse",
        "ndcg_at_10_avg_f1_5": "ndcg_at_10",
        "precision_at_10_avg_f1_5": "precision_at_10",
        "recall_at_10_avg_f1_5": "recall_at_10",
    })
    k_list = K3_24_LIST
    tag = "f1_5avg_k3_24"
    p1 = OUT_DIR / f"fuzzy_official_{tag}_5metrics_cluster_avg_soft.png"
    p2 = OUT_DIR / f"fuzzy_official_{tag}_mae_heatmap.png"
    p3 = OUT_DIR / f"fuzzy_official_{tag}_bestk_5metrics_bar.png"
    plot_5metrics_k_sweep(
        plot_df, p1, k_list=k_list,
        title="FCM official fold 1-5 ort. — cluster_avg_soft (cosine)\n"
        f"K={k_list[0]}..{k_list[-1]}, fast estop, 8 algoritma",
    )
    plot_mae_heatmap(plot_df, p2, k_list=k_list)
    plot_best_k_bar(plot_df, p3)
    print(f"plot -> {p1}")
    print(f"plot -> {p2}")
    print(f"plot -> {p3}")


if __name__ == "__main__":
    main()
