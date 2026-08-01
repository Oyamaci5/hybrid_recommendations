"""
FCM official fold-1: iki algoritma arasi paired bootstrap + Wilcoxon (cluster_avg_soft).

Delta = MAE(A) - MAE(B); negatif -> A daha iyi. CI 0'i icermiyorsa bootstrap anlamli.

  python -m experiments.run_fuzzy_official_f1_bootstrap --fast
  python -m experiments.run_fuzzy_official_f1_bootstrap --fast --k 10 --algo-a HA_AVOAHGS --algo-b LIT_GWO
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import (
    FAST_ALGOS,
    FAST_K_LIST,
    FCM_M,
    FOLD,
    WNMF_DIM,
    expected_suffix,
)
from experiments.run_fuzzy_official_f1_eval import (
    DEFAULT_SOFT,
    predict_cluster_avg_eval_rows,
)
import importlib.util

_pb_path = REPO / "mealpy" / "paired_bootstrap_ci.py"
_spec = importlib.util.spec_from_file_location("paired_bootstrap_ci", _pb_path)
_pb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pb)
align_eval_rows = _pb.align_eval_rows
paired_bootstrap_delta = _pb.paired_bootstrap_delta
paired_wilcoxon = _pb.paired_wilcoxon
_bootstrap_note = _pb._bootstrap_note
_wilcoxon_note = _pb._wilcoxon_note

OUT_CSV = REPO / "results" / "fuzzy_official_f1_bootstrap.csv"
OUT_FAST_CSV = REPO / "results" / "fuzzy_official_fast_bootstrap.csv"

DEFAULT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("HA_AVOAHGS", "LIT_GWO"),
    ("HA_AVOAHGS", "B_AVOA"),
    ("LIT_GWO", "B_AVOA"),
)


def _bootstrap_significant(ci_lo: float, ci_hi: float) -> bool:
    return ci_hi < 0.0 or ci_lo > 0.0


def compare_pair_rows(
    algo_a: str,
    algo_b: str,
    k: int,
    *,
    similarity: str,
    soft: float,
    fast: bool,
    prune: bool,
    n_boot: int,
    seed: int,
    ci: float,
    alpha: float,
    cache: Dict[Tuple[str, int], np.ndarray],
) -> List[dict]:
    key_a, key_b = (algo_a, k), (algo_b, k)
    for algo, key in ((algo_a, key_a), (algo_b, key_b)):
        if key not in cache:
            print(f"  [{algo}] K={k} tahmin…", flush=True)
            cache[key] = predict_cluster_avg_eval_rows(
                algo, k,
                similarity=similarity,
                soft_threshold=soft,
                prune=prune,
                fast=fast,
                quiet=True,
            )

    true, pa, pb = align_eval_rows(cache[key_a], cache[key_b])
    rows = []
    for metric in ("mae", "rmse"):
        stats = paired_bootstrap_delta(
            true, pa, pb, metric, n_boot=n_boot, seed=seed, ci=ci,
        )
        wx = paired_wilcoxon(
            true, pa, pb, metric, alpha=alpha, alternative="two-sided",
        )
        lo, hi = stats["ci_lo"], stats["ci_hi"]
        rows.append({
            "protocol": "fuzzy_imkpp_official",
            "fold": FOLD,
            "k": k,
            "fcm_m": FCM_M,
            "wnmf_dim": WNMF_DIM,
            "fast": fast,
            "prune": prune,
            "algo_a": algo_a,
            "algo_b": algo_b,
            "predictor": "cluster_avg_soft",
            "similarity": similarity,
            "soft_threshold": soft,
            "metric": metric,
            "n_pairs": int(stats["n_pairs"]),
            "mae_a": stats["mae_a"] if metric == "mae" else np.nan,
            "mae_b": stats["mae_b"] if metric == "mae" else np.nan,
            "delta": stats["delta"],
            "ci_lo": lo,
            "ci_hi": hi,
            "p_a_better": stats["p_a_better"],
            "bootstrap_sig": _bootstrap_significant(lo, hi),
            "bootstrap_note": _bootstrap_note(lo, hi, algo_a, algo_b),
            "wilcoxon_p": wx["p_value"],
            "wilcoxon_sig": bool(wx["significant"]) if not np.isnan(wx["p_value"]) else False,
            "wilcoxon_note": _wilcoxon_note(
                wx["median_diff"], wx["p_value"], alpha, algo_a, algo_b,
            ),
            "assign_suffix": expected_suffix(k, prune=prune, fast=fast),
        })
    return rows


def run_bootstrap(
    k_list: Sequence[int],
    pairs: Sequence[Tuple[str, str]],
    *,
    similarity: str,
    soft: float,
    fast: bool,
    prune: bool,
    n_boot: int,
    seed: int,
    ci: float,
    alpha: float,
) -> pd.DataFrame:
    cache: Dict[Tuple[str, int], np.ndarray] = {}
    all_rows: List[dict] = []
    for k in k_list:
        print(f"\n=== K={k} sim={similarity} soft={soft} fast={fast} ===", flush=True)
        for algo_a, algo_b in pairs:
            rows = compare_pair_rows(
                algo_a, algo_b, k,
                similarity=similarity,
                soft=soft,
                fast=fast,
                prune=prune,
                n_boot=n_boot,
                seed=seed,
                ci=ci,
                alpha=alpha,
                cache=cache,
            )
            all_rows.extend(rows)
            mae_row = rows[0]
            print(
                f"  {algo_a} vs {algo_b}: dMAE={mae_row['delta']:.4f} "
                f"[{mae_row['ci_lo']:.4f}, {mae_row['ci_hi']:.4f}] "
                f"P({algo_a} better)={mae_row['p_a_better']:.1%} "
                f"-> {mae_row['bootstrap_note']}",
                flush=True,
            )
    return pd.DataFrame(all_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", default=None)
    ap.add_argument("--algo-a", default=None)
    ap.add_argument("--algo-b", default=None)
    ap.add_argument("--similarity", default="cosine")
    ap.add_argument("--soft-threshold", type=float, default=DEFAULT_SOFT)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    if args.fast:
        k_list = args.k if args.k else FAST_K_LIST
        algos = FAST_ALGOS
        out_csv = args.out_csv or OUT_FAST_CSV
    else:
        k_list = args.k if args.k else [4, 10]
        algos = FAST_ALGOS  # override if single pair given
        out_csv = args.out_csv or OUT_CSV

    if args.algo_a and args.algo_b:
        pairs = [(args.algo_a, args.algo_b)]
    else:
        pairs = list(combinations(algos, 2)) if len(algos) >= 2 else DEFAULT_PAIRS

    print(
        f"Paired bootstrap  fold={FOLD}  predictor=cluster_avg_soft  "
        f"fast={args.fast}  K={list(k_list)}  pairs={pairs}",
        flush=True,
    )

    df = run_bootstrap(
        k_list,
        pairs,
        similarity=args.similarity,
        soft=args.soft_threshold,
        fast=args.fast,
        prune=args.prune,
        n_boot=args.n_bootstrap,
        seed=args.seed,
        ci=args.ci,
        alpha=args.alpha,
    )
    if df.empty:
        sys.exit("Sonuc yok.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.is_file():
        old = pd.read_csv(out_csv)
        if "fast" not in old.columns:
            old["fast"] = False
        key = [
            "fold", "k", "algo_a", "algo_b", "predictor", "similarity",
            "soft_threshold", "metric", "fast", "prune",
        ]
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=key, keep="last")
    df = df.sort_values(["fast", "k", "algo_a", "algo_b", "metric"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV -> {out_csv}", flush=True)

    mae = df[df["metric"] == "mae"]
    print("\nOzet (MAE, bootstrap anlamli mi):", flush=True)
    for _, r in mae.iterrows():
        sig = "EVET" if r["bootstrap_sig"] else "hayir"
        print(
            f"  K={int(r['k']):2d} {r['algo_a']} vs {r['algo_b']}: "
            f"d={r['delta']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] sig={sig}",
            flush=True,
        )


if __name__ == "__main__":
    main()
