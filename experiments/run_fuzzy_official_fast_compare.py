"""
Fast K sweep: bootstrap -> eval -> yapisal ozet.

  python -m experiments.run_fuzzy_official_fast_compare
  python -m experiments.run_fuzzy_official_fast_compare --phase bootstrap
  python -m experiments.run_fuzzy_official_fast_compare --phase eval --similarity cosine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from collections import Counter

from experiments.fuzzy_official_protocol import FAST_ALGOS, FAST_K_LIST, assign_dir
from experiments.run_fuzzy_official_f1_eval import DEFAULT_SOFT

STRUCT_CSV = REPO / "results" / "fuzzy_official_fast_cluster_structure.csv"
EVAL_CSV = REPO / "results" / "fuzzy_official_fast_preds.csv"
BOOT_CSV = REPO / "results" / "fuzzy_official_fast_bootstrap.csv"
SUMMARY_CSV = REPO / "results" / "fuzzy_official_fast_k_sweep_summary.csv"


def cluster_structure_rows() -> pd.DataFrame:
    rows = []
    for k in FAST_K_LIST:
        labels = {}
        for algo in FAST_ALGOS:
            labels[algo] = np.load(assign_dir(algo, k, fast=True) / "assignments.npy").astype(int)
        ha, lit, b = labels["HA_AVOAHGS"], labels["LIT_GWO"], labels["B_AVOA"]
        rows.append({
            "k": k,
            "exact_pct_HA_LIT": round(float(np.mean(ha == lit)) * 100, 1),
            "exact_pct_HA_B": round(float(np.mean(ha == b)) * 100, 1),
            "exact_pct_LIT_B": round(float(np.mean(lit == b)) * 100, 1),
            "ari_HA_LIT": round(adjusted_rand_score(ha, lit), 3),
            "ari_HA_B": round(adjusted_rand_score(ha, b), 3),
            "ari_LIT_B": round(adjusted_rand_score(lit, b), 3),
        })
    return pd.DataFrame(rows)


def run_eval(sims: list[str], soft: float) -> pd.DataFrame:
    from experiments.run_fuzzy_official_f1_eval import PREDICTORS_ALL, eval_k

    parts = []
    for k in FAST_K_LIST:
        for sim in sims:
            print(f"\n--- eval K={k} sim={sim} (fast, cluster_avg + cluster_knn) ---", flush=True)
            df = eval_k(
                k, soft_threshold=soft, similarity=sim,
                algos=FAST_ALGOS, prune=False, fast=True,
                predictors=PREDICTORS_ALL,
            )
            if not df.empty:
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    if EVAL_CSV.is_file():
        old = pd.read_csv(EVAL_CSV)
        for col, default in [("fast", False), ("prune", False), ("knn_k", 0)]:
            if col not in old.columns:
                old[col] = default
        key = [
            "fold", "k", "algo", "predictor", "knn_k", "similarity",
            "soft_threshold", "fcm_m", "wnmf_dim", "prune", "fast",
        ]
        out = pd.concat([old, out], ignore_index=True).drop_duplicates(subset=key, keep="last")
    out = out.sort_values(["fast", "k", "similarity", "predictor", "knn_k", "algo"]).reset_index(drop=True)
    out.to_csv(EVAL_CSV, index=False)
    print(f"\nEval CSV -> {EVAL_CSV} ({len(out)} rows)", flush=True)
    return out


def build_summary(eval_df: pd.DataFrame, struct_df: pd.DataFrame, *, soft: float) -> pd.DataFrame:
    rows = []
    for predictor in sorted(eval_df["predictor"].unique()):
        pred_df = eval_df[eval_df["predictor"] == predictor]
        for sim in sorted(pred_df["similarity"].unique()):
            sub = pred_df[pred_df["similarity"] == sim]
            if predictor == "cluster_avg_soft":
                sub = sub[sub["soft_threshold"] == soft]
            for k in FAST_K_LIST:
                ks = sub[sub["k"] == k]
                if ks.empty:
                    continue
                st = struct_df[struct_df["k"] == k].iloc[0]
                best = ks.loc[ks["mae"].idxmin()]
                spread = float(ks["mae"].max() - ks["mae"].min())
                rows.append({
                    "predictor": predictor,
                    "similarity": sim,
                    "k": k,
                    "mae_HA": ks.loc[ks["algo"] == "HA_AVOAHGS", "mae"].iloc[0] if (ks["algo"] == "HA_AVOAHGS").any() else None,
                    "mae_LIT": ks.loc[ks["algo"] == "LIT_GWO", "mae"].iloc[0] if (ks["algo"] == "LIT_GWO").any() else None,
                    "mae_B": ks.loc[ks["algo"] == "B_AVOA", "mae"].iloc[0] if (ks["algo"] == "B_AVOA").any() else None,
                    "mae_spread": round(spread, 4),
                    "best_algo": best["algo"],
                    "best_mae": best["mae"],
                    "ari_HA_LIT": st["ari_HA_LIT"],
                    "exact_pct_HA_LIT": st["exact_pct_HA_LIT"],
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["bootstrap", "eval", "structure", "all"],
        default="all",
    )
    ap.add_argument("--similarity", nargs="+", default=["cosine"])
    ap.add_argument("--soft-threshold", type=float, default=DEFAULT_SOFT)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--skip-eval", action="store_true", help="eval atla (CSV varsa)")
    args = ap.parse_args()

    if args.phase in ("bootstrap", "all"):
        from experiments.run_fuzzy_official_f1_bootstrap import run_bootstrap
        from experiments.run_fuzzy_official_f1_bootstrap import DEFAULT_PAIRS

        print("\n=== PAIRED BOOTSTRAP (cosine) ===", flush=True)
        boot_df = run_bootstrap(
            FAST_K_LIST,
            DEFAULT_PAIRS,
            similarity="cosine",
            soft=args.soft_threshold,
            fast=True,
            prune=False,
            n_boot=args.n_bootstrap,
            seed=42,
            ci=95.0,
            alpha=0.05,
        )
        boot_df.to_csv(BOOT_CSV, index=False)
        print(f"Bootstrap CSV -> {BOOT_CSV}", flush=True)

    if args.phase == "structure":
        struct_df = cluster_structure_rows()
        struct_df.to_csv(STRUCT_CSV, index=False)
        print(struct_df.to_string(index=False), flush=True)
        return

    struct_df = cluster_structure_rows()
    struct_df.to_csv(STRUCT_CSV, index=False)
    print(f"Yapisal CSV -> {STRUCT_CSV}", flush=True)
    print(struct_df.to_string(index=False), flush=True)

    if args.phase in ("eval", "all"):
        if args.skip_eval and EVAL_CSV.is_file():
            eval_df = pd.read_csv(EVAL_CSV)
        else:
            eval_df = run_eval(list(dict.fromkeys(args.similarity)), args.soft_threshold)
    else:
        eval_df = pd.read_csv(EVAL_CSV) if EVAL_CSV.is_file() else pd.DataFrame()

    if eval_df.empty:
        if args.phase == "bootstrap":
            return
        sys.exit("Eval sonucu yok.")

    summary = build_summary(eval_df, struct_df, soft=args.soft_threshold)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nOzet CSV -> {SUMMARY_CSV}", flush=True)
    for pred in summary["predictor"].unique():
        for sim in summary[summary["predictor"] == pred]["similarity"].unique():
            print(f"\n=== {pred} | {sim} (MAE) ===", flush=True)
            piv = summary[(summary["predictor"] == pred) & (summary["similarity"] == sim)][
                ["k", "mae_HA", "mae_LIT", "mae_B", "mae_spread", "best_algo", "ari_HA_LIT"]
            ]
            print(piv.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
