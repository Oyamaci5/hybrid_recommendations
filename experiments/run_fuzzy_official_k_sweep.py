"""
FCM official fold-1: prune test (K=10) + K sweep (HA/LIT_GWO).

  1) K=10 prune atama + cosine eval -> no-prune ile karsilastir
  2) prune ise yararliysa K=15,20,25,30,70 prune; degilse no-prune
  3) tum K icin cosine cluster_avg eval

  python experiments/run_fuzzy_official_k_sweep.py --phase k10-test
  python experiments/run_fuzzy_official_k_sweep.py --phase assign --use-prune
  python experiments/run_fuzzy_official_k_sweep.py --phase eval --use-prune
  python experiments/run_fuzzy_official_k_sweep.py --phase all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import (
    FOLD,
    assign_dir,
    build_assign_cmd,
    expected_suffix,
)
from experiments.run_fuzzy_official_f1_eval import eval_k

SWEEP_ALGOS = ["HA_AVOAHGS", "LIT_GWO"]
SWEEP_K = [15, 20, 25, 30, 70]
K10_TEST_K = [10]
DEFAULT_OUT = REPO / "results" / "fuzzy_official_f1_cluster_avg_soft.csv"
K10_BASELINE = {
    "HA_AVOAHGS": {"mae": 0.7600, "ndcg_at_10": 0.8364},
    "LIT_GWO": {"mae": 0.7600, "ndcg_at_10": 0.8368},
}


def run_assign(algos: list[str], k_list: list[int], *, prune: bool, jobs: int, skip: bool) -> int:
    cmd = build_assign_cmd(
        algos=algos, k_list=k_list, prune=prune, jobs=jobs, skip_existing=skip,
    )
    print("ASSIGN", "prune" if prune else "no-prune", "K=", k_list, "algos=", algos, flush=True)
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def run_eval_rows(
    k_list: list[int],
    *,
    prune: bool,
    algos: list[str],
    out_csv: Path,
    soft: float,
) -> pd.DataFrame:
    rows = []
    for k in k_list:
        print(f"\n--- eval K={k} prune={prune} cosine soft={soft} ---", flush=True)
        df = eval_k(k, soft_threshold=soft, similarity="cosine", algos=algos, prune=prune)
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.is_file():
        old = pd.read_csv(out_csv)
        key = [
            "fold", "k", "algo", "predictor", "similarity",
            "soft_threshold", "fcm_m", "wnmf_dim", "prune",
        ]
        out = pd.concat([old, out], ignore_index=True).drop_duplicates(subset=key, keep="last")
    out = out.sort_values(["prune", "k", "algo"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"\nCSV -> {out_csv}", flush=True)
    print(out[out["prune"] == prune].to_string(index=False), flush=True)
    return out


def prune_beats_baseline(df: pd.DataFrame) -> bool:
    sub = df[(df["k"] == 10) & (df["prune"] == True) & (df["similarity"] == "cosine")]
    if sub.empty:
        print("K=10 prune eval sonucu yok.", flush=True)
        return False
    wins = 0
    for _, row in sub.iterrows():
        base = K10_BASELINE.get(row["algo"], {})
        mae_ok = row["mae"] <= base.get("mae", 999) + 0.003
        ndcg_ok = row["ndcg_at_10"] >= base.get("ndcg_at_10", 0) - 0.005
        print(
            f"  {row['algo']}: MAE={row['mae']:.4f} (base {base.get('mae')}) "
            f"NDCG={row['ndcg_at_10']:.4f} ok={mae_ok and ndcg_ok}",
            flush=True,
        )
        if mae_ok and ndcg_ok:
            wins += 1
    mean_mae = float(sub["mae"].mean())
    base_mean = sum(v["mae"] for v in K10_BASELINE.values()) / len(K10_BASELINE)
    algo_spread = float(sub["mae"].max() - sub["mae"].min())
    ok = wins >= 1 and mean_mae <= base_mean + 0.002 and algo_spread >= 0.003
    print(
        f"Prune karar: wins={wins}/2 mean_mae={mean_mae:.4f} base_mean={base_mean:.4f} "
        f"spread={algo_spread:.4f} -> {'PRUNE' if ok else 'NO_PRUNE'}",
        flush=True,
    )
    return ok


def status(algos: list[str], k_list: list[int], *, prune: bool) -> None:
    for k in k_list:
        for algo in algos:
            p = assign_dir(algo, k, prune=prune)
            ok = (p / "assignments.npy").is_file()
            mem = (p / "memberships.npy").is_file()
            print(f"  {'OK' if ok else 'MISSING'}  {p.name}  mem={mem}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["k10-test", "assign", "eval", "all"],
        default="all",
    )
    ap.add_argument("--use-prune", action="store_true", help="assign/eval icin prune zorla")
    ap.add_argument("--no-prune", action="store_true", help="assign/eval icin prune kapali")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--soft-threshold", type=float, default=0.10)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df10 = pd.DataFrame()
    if args.phase in ("k10-test", "all"):
        rc = run_assign(SWEEP_ALGOS, K10_TEST_K, prune=True, jobs=args.jobs, skip=args.skip_existing)
        if rc != 0:
            sys.exit(rc)
        status(SWEEP_ALGOS, K10_TEST_K, prune=True)
        df10 = run_eval_rows(
            K10_TEST_K, prune=True, algos=SWEEP_ALGOS,
            out_csv=args.out_csv, soft=args.soft_threshold,
        )
        if args.phase == "k10-test":
            prune_beats_baseline(df10)
            return

    use_prune = args.use_prune
    if args.no_prune:
        use_prune = False
    elif args.phase == "all" and not args.use_prune:
        use_prune = prune_beats_baseline(df10)

    if args.phase in ("assign", "all"):
        rc = run_assign(
            SWEEP_ALGOS, SWEEP_K, prune=use_prune,
            jobs=args.jobs, skip=args.skip_existing,
        )
        if rc != 0:
            sys.exit(rc)
        status(SWEEP_ALGOS, SWEEP_K, prune=use_prune)

    if args.phase in ("eval", "all"):
        run_eval_rows(
            SWEEP_K, prune=use_prune, algos=SWEEP_ALGOS,
            out_csv=args.out_csv, soft=args.soft_threshold,
        )
        if args.phase == "eval":
            run_eval_rows(
                K10_TEST_K, prune=use_prune, algos=SWEEP_ALGOS,
                out_csv=args.out_csv, soft=args.soft_threshold,
            )


if __name__ == "__main__":
    main()
