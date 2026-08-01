"""
irand_feature_pred_cluster(.csv) + _kmref: en iyi MAE/NDCG ve B0'dan en farklı meta koşular.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NO_KMREF = REPO / "results" / "irand_feature_pred_cluster.csv"
KMREF = REPO / "results" / "irand_feature_pred_cluster_kmref.csv"
META_ALGOS = {"B1_HHO", "HA_AVOAHGS", "IWO_HHO"}
KEY = ["feature", "latent_dim", "k", "algo", "predictor"]


def _load(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = tag
    for c in ("mae", "ndcg_at_10", "rmse"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["latent_dim"] = df["latent_dim"].astype(int)
    df["k"] = df["k"].astype(int)
    n0 = len(df)
    df = df.drop_duplicates(subset=KEY, keep="last")
    if len(df) < n0:
        print(f"  [{tag}] dedupe: {n0} -> {len(df)}")
    return df


def _top_table(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    rows = []
    for label, col, asc in (
        ("best_mae", "mae", True),
        ("best_ndcg", "ndcg_at_10", False),
    ):
        sub = df.nsmallest(n, "mae") if asc else df.nlargest(n, "ndcg_at_10")
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            rows.append({
                "rank_type": label,
                "rank": rank,
                "source": r["source"],
                "feature": r["feature"],
                "latent_dim": int(r["latent_dim"]),
                "k": int(r["k"]),
                "algo": r["algo"],
                "predictor": r["predictor"],
                "mae": round(float(r["mae"]), 4),
                "ndcg_at_10": round(float(r["ndcg_at_10"]), 4),
            })
    return pd.DataFrame(rows)


def _best_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Her (source, feature, latent_dim, k, predictor) için en iyi MAE ve NDCG algo."""
    rows = []
    gcols = ["source", "feature", "latent_dim", "k", "predictor"]
    for keys, g in df.groupby(gcols):
        src, feat, dim, k, pred = keys
        i_mae = g["mae"].idxmin()
        i_ndcg = g["ndcg_at_10"].idxmax()
        r_mae = g.loc[i_mae]
        r_ndcg = g.loc[i_ndcg]
        rows.append({
            "source": src,
            "feature": feat,
            "latent_dim": dim,
            "k": k,
            "predictor": pred,
            "best_mae_algo": r_mae["algo"],
            "best_mae": round(float(r_mae["mae"]), 4),
            "best_ndcg_algo": r_ndcg["algo"],
            "best_ndcg": round(float(r_ndcg["ndcg_at_10"]), 4),
            "same_algo_both": r_mae["algo"] == r_ndcg["algo"],
        })
    return pd.DataFrame(rows)


def _b0_diff(df: pd.DataFrame, b0_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Meta satırları: aynı (source, feature, dim, k, predictor) için B0'a uzaklık."""
    b0 = df[df["algo"] == "B0_KMEANS"].copy()
    if b0_ref is not None:
        b0 = b0_ref
    meta = df[df["algo"].isin(META_ALGOS)].copy()
    if b0.empty or meta.empty:
        return pd.DataFrame()

    merge_on = ["source", "feature", "latent_dim", "k", "predictor"]
    b0_slim = b0[merge_on + ["mae", "ndcg_at_10"]].rename(
        columns={"mae": "b0_mae", "ndcg_at_10": "b0_ndcg"},
    )
    m = meta.merge(b0_slim, on=merge_on, how="inner")
    m["d_mae"] = m["mae"] - m["b0_mae"]
    m["d_ndcg"] = m["ndcg_at_10"] - m["b0_ndcg"]
    # Ölçeklenmiş L2: MAE ~0.8, NDCG ~0.8
    m["dist_b0"] = np.sqrt((m["d_mae"] / 0.1) ** 2 + (m["d_ndcg"] / 0.05) ** 2)
    m["abs_d_mae"] = m["d_mae"].abs()
    cols = merge_on + [
        "algo", "mae", "ndcg_at_10", "b0_mae", "b0_ndcg",
        "d_mae", "d_ndcg", "dist_b0", "abs_d_mae",
    ]
    return m[cols].sort_values("dist_b0", ascending=False)


def main() -> None:
    out_dir = REPO / "results"
    df_nk = _load(NO_KMREF, "no_kmref")
    df_km = _load(KMREF, "kmref")

    # kmref için B0 referansı no_kmref'ten (aynı feat/k/predictor)
    b0_ref = df_nk[df_nk["algo"] == "B0_KMEANS"][
        ["feature", "latent_dim", "k", "predictor", "mae", "ndcg_at_10"]
    ].copy()
    b0_ref["source"] = "kmref"

    all_df = pd.concat([df_nk, df_km], ignore_index=True)

    top = _top_table(all_df, n=20)
    top.to_csv(out_dir / "irand_pred_top_mae_ndcg.csv", index=False)

    per_cell = _best_per_cell(all_df)
    per_cell.to_csv(out_dir / "irand_pred_best_per_cell.csv", index=False)

    diff_nk = _b0_diff(df_nk)
    diff_km = _b0_diff(df_km, b0_ref=b0_ref)
    diff = pd.concat([diff_nk, diff_km], ignore_index=True)
    diff_top = diff.head(40)
    diff.to_csv(out_dir / "irand_pred_vs_b0_all.csv", index=False)
    diff_top.to_csv(out_dir / "irand_pred_vs_b0_top40.csv", index=False)

    print("=" * 72)
    print("GENEL — en düşük MAE (top 10)")
    print("=" * 72)
    print(
        all_df.nsmallest(10, "mae")[
            ["source", "feature", "latent_dim", "k", "algo", "predictor", "mae", "ndcg_at_10"]
        ].to_string(index=False),
    )

    print("\n" + "=" * 72)
    print("GENEL — en yüksek NDCG@10 (top 10)")
    print("=" * 72)
    print(
        all_df.nlargest(10, "ndcg_at_10")[
            ["source", "feature", "latent_dim", "k", "algo", "predictor", "mae", "ndcg_at_10"]
        ].to_string(index=False),
    )

    for src, label in (("no_kmref", "NO-KMREF"), ("kmref", "KMREF")):
        sub = diff[diff["source"] == src].head(12)
        print(f"\n{'=' * 72}")
        print(f"B0'dan en farklı meta — {label} (top 12, dist_b0)")
        print("=" * 72)
        if sub.empty:
            print("  (veri yok)")
            continue
        print(
            sub[
                ["feature", "latent_dim", "k", "algo", "predictor",
                 "mae", "b0_mae", "d_mae", "ndcg_at_10", "b0_ndcg", "d_ndcg"]
            ].to_string(index=False, float_format="%.4f"),
        )

    print(f"\nCSV:")
    print(f"  {out_dir / 'irand_pred_top_mae_ndcg.csv'}")
    print(f"  {out_dir / 'irand_pred_best_per_cell.csv'}")
    print(f"  {out_dir / 'irand_pred_vs_b0_top40.csv'}")


if __name__ == "__main__":
    main()
