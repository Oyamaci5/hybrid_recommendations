"""
Tüm fuzzy official cluster_avg_soft sonuçlarını tek CSV'de birleştir.

  python -m experiments.export_fuzzy_official_master_table
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import ALL_ALGOS

OUT = REPO / "results" / "fuzzy_official_master_cluster_avg_soft.csv"
OUT_PIVOT = REPO / "results" / "fuzzy_official_master_mae_pivot.csv"

# Öncelik: tam grid en sona (keep=last ile üzerine yazar)
SOURCE_FILES = [
    REPO / "results" / "fuzzy_official_f1_cluster_avg_soft.csv",
    REPO / "results" / "fuzzy_k_sweep_cluster_avg_w20.csv",
    REPO / "results" / "fuzzy_official_k4_14_extra_preds.csv",
    REPO / "results" / "fuzzy_official_fast_preds.csv",
    REPO / "results" / "fuzzy_official_k3_24_cluster_avg_soft.csv",
]

METRIC_COLS = [
    "mae", "rmse", "ndcg_at_10", "precision_at_10", "recall_at_10",
    "coverage_at_10", "jaccard_at_10",
]
# Ayni K+algo icin tek satir (kaynak onceligi SOURCE_FILES sirasi + keep=last)
DEDUP_KEY = ["fold", "k", "algo", "predictor", "knn_k", "similarity"]


def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "fast" not in df.columns:
        df["fast"] = False
    if "prune" not in df.columns:
        df["prune"] = False
    if "knn_k" not in df.columns:
        df["knn_k"] = 0
    if "fold" not in df.columns:
        df["fold"] = 1
    df["source_file"] = source
    return df


def load_master(*, cluster_avg_only: bool = True, cosine_only: bool = True) -> pd.DataFrame:
    parts = []
    for p in SOURCE_FILES:
        if not p.is_file():
            continue
        df = _normalize(pd.read_csv(p), p.name)
        if cluster_avg_only and "predictor" in df.columns:
            df = df[df["predictor"].astype(str) == "cluster_avg_soft"]
        if cosine_only and "similarity" in df.columns:
            df = df[df["similarity"].astype(str) == "cosine"]
        if not df.empty:
            parts.append(df)
    if not parts:
        raise FileNotFoundError("Kaynak CSV bulunamadi")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=DEDUP_KEY, keep="last")
    out = out.sort_values(["k", "algo"]).reset_index(drop=True)
    return out


def main() -> None:
    df = load_master()
    cols = [
        "source_file", "protocol", "fold", "k", "algo", "predictor",
        "similarity", "soft_threshold", "fcm_m", "wnmf_dim", "fast", "prune",
        *METRIC_COLS,
        "n_active_clusters", "cluster_min", "cluster_max", "eval_seconds",
        "assign_suffix",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    pivot = df.pivot_table(index="algo", columns="k", values="mae", aggfunc="first")
    pivot = pivot.reindex(ALL_ALGOS)
    pivot.to_csv(OUT_PIVOT)

    ks = sorted(df["k"].unique())
    k3_24 = [k for k in ks if 3 <= k <= 24]
    extra = [k for k in ks if k < 3 or k > 24]
    full_3_24 = len(df[(df["k"].between(3, 24)) & (df["algo"].isin(ALL_ALGOS))])
    expected = len(range(3, 25)) * len(ALL_ALGOS)

    print(f"Master -> {OUT}")
    print(f"  Satir: {len(df)}")
    print(f"  K (3-24): {len(k3_24)} deger, tam grid satiri: {full_3_24}/{expected}")
    if extra:
        print(f"  Ek K (3-24 disi): {extra}")
    print(f"MAE pivot -> {OUT_PIVOT}")
    print(f"\nK basina satir sayisi:\n{df.groupby('k').size().to_string()}")


if __name__ == "__main__":
    main()
