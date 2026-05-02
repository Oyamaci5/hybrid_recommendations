"""
Grid araması özet CSV'sinden (top5_manual_no_gray_random_grid.csv) algoritma davranışı tabloları üretir.

Çıktılar (--out-dir, default results/grid_search):
  - algorithms_behavior_summary.csv
      Her algoritma için: MAE/RMSE/NDCG (min, max, mean, std), en iyi ve en kötü senaryo (k, prune, zscore, feature).
  - algorithms_scenario_win_counts.csv
      Her grid senaryosunda (84) hangi algoritma MAE'de / NDCG'de birinci — özet sayımlar.

Örnek:
  python scripts/summarize_grid_per_algorithm.py
  python scripts/summarize_grid_per_algorithm.py --grid results/grid_search/top5_manual_no_gray_random_grid.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SCENARIO_KEYS = ["k", "prune", "zscore", "feature"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sonuçlarını algoritma bazında özetle")
    p.add_argument(
        "--grid",
        type=Path,
        default=REPO_ROOT / "results" / "grid_search" / "top5_manual_no_gray_random_grid.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "grid_search",
    )
    return p.parse_args()


def _scenario_str(row: pd.Series) -> str:
    return f"k{row['k']}_{row['prune']}_{row['zscore']}_{row['feature']}"


def main() -> int:
    args = parse_args()
    if not args.grid.is_file():
        raise SystemExit(f"Grid CSV yok: {args.grid}")

    df = pd.read_csv(args.grid)
    for c in ["mae", "rmse", "precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["scenario_id"] = df.apply(_scenario_str, axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Özet istatistik + en iyi / en kötü senaryo (MAE ve NDCG) ---
    rows_out = []
    for algo, g in df.groupby("algo_label", sort=True):
        r = {"algo_label": algo, "n_grid_scenarios": g["scenario_id"].nunique()}
        for col, label in [
            ("mae", "mae"),
            ("rmse", "rmse"),
            ("ndcg_at_10", "ndcg_at_10"),
            ("precision_at_10", "precision_at_10"),
            ("recall_at_10", "recall_at_10"),
            ("f1_at_10", "f1_at_10"),
        ]:
            s = g[col]
            r[f"{label}_min"] = s.min()
            r[f"{label}_max"] = s.max()
            r[f"{label}_mean"] = s.mean()
            r[f"{label}_std"] = s.std(ddof=0)

        i_best_mae = g["mae"].idxmin()
        i_worst_mae = g["mae"].idxmax()
        i_best_ndcg = g["ndcg_at_10"].idxmax()
        i_worst_ndcg = g["ndcg_at_10"].idxmin()

        for prefix, idx in [
            ("mae_best", i_best_mae),
            ("mae_worst", i_worst_mae),
            ("ndcg_best", i_best_ndcg),
            ("ndcg_worst", i_worst_ndcg),
        ]:
            row = df.loc[idx]
            r[f"{prefix}_k"] = int(row["k"])
            r[f"{prefix}_prune"] = row["prune"]
            r[f"{prefix}_zscore"] = row["zscore"]
            r[f"{prefix}_feature"] = row["feature"]
            r[f"{prefix}_mae"] = float(row["mae"])
            r[f"{prefix}_ndcg_at_10"] = float(row["ndcg_at_10"])

        rows_out.append(r)

    summary = pd.DataFrame(rows_out)
    summary_path = args.out_dir / "algorithms_behavior_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    # --- Senaryo bazında kazanan sayıları ---
    mae_winners = df.loc[df.groupby("scenario_id")["mae"].idxmin()]["algo_label"]
    ndcg_winners = df.loc[df.groupby("scenario_id")["ndcg_at_10"].idxmax()]["algo_label"]
    rmse_winners = df.loc[df.groupby("scenario_id")["rmse"].idxmin()]["algo_label"]

    algos = sorted(df["algo_label"].unique())
    win_rows = []
    for a in algos:
        win_rows.append({
            "algo_label": a,
            "mae_first_place_count": int((mae_winners == a).sum()),
            "ndcg_first_place_count": int((ndcg_winners == a).sum()),
            "rmse_first_place_count": int((rmse_winners == a).sum()),
            "n_scenarios": int(df["scenario_id"].nunique()),
        })
    wins_df = pd.DataFrame(win_rows)
    wins_path = args.out_dir / "algorithms_scenario_win_counts.csv"
    wins_df.to_csv(wins_path, index=False, encoding="utf-8")

    # --- İnsan okuması için dar tablo (rapor) ---
    slim = summary[
        [
            "algo_label",
            "n_grid_scenarios",
            "mae_min",
            "mae_mean",
            "mae_max",
            "mae_best_k",
            "mae_best_prune",
            "mae_best_zscore",
            "mae_best_feature",
            "ndcg_at_10_max",
            "ndcg_at_10_mean",
            "ndcg_best_k",
            "ndcg_best_feature",
        ]
    ].copy()
    slim_path = args.out_dir / "algorithms_behavior_table.csv"
    slim.to_csv(slim_path, index=False, encoding="utf-8")

    print(f"Yazıldı: {summary_path}")
    print(f"Yazıldı: {wins_path}")
    print(f"Yazıldı: {slim_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
