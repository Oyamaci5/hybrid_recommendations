"""
IWO_HHO fold 2: mkpp vs random init — convergence (euclidean WCSS).

  python -m experiments.run_iwo_hho_irand_vs_mkpp_fold2 --phase assign --jobs 2
  python -m experiments.run_iwo_hho_irand_vs_mkpp_fold2 --phase compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
ROOT = REPO / "mealpy" / "results" / "assignments_estop" / "ml100k"
OUT = REPO / "results" / "iwo_hho_fold2_euc_irand_vs_mkpp_convergence.csv"

ALGO = "IWO_HHO"
FOLD = 2
K_LIST = [10, 16, 20]
CLUSTER_METRIC = "euclidean"
INITS = [
    ("mkpp", "imkpp"),
    ("random", "irand"),
]


def _assign_dir(init_tag: str, k: int) -> Path:
    return ROOT / (
        f"{ALGO}_euc_{init_tag}_nogs_trainonly_official_f{FOLD}"
        f"_none_wnmf50_k{k}_pwcss"
    )


def _build_cmd(*, init_mode: str, jobs: int, skip_existing: bool) -> list[str]:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", ALGO,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", "50", "--wnmf-epochs", "50",
        "--legacy-wnmf-suffix",
        "--init-mode", init_mode,
        "--cluster-metric", CLUSTER_METRIC,
        "--fitness", "wcss",
        "--cluster-objective", "wcss",
        "--train-only", "--eval-split", "official",
        "--fold", str(FOLD),
        "--k", *[str(k) for k in K_LIST],
        "--jobs", str(jobs),
        "--baseline-epoch", "40", "--pop-size", "25",
        "--early-stop", "--early-stop-patience", "4",
        "--early-stop-block", "5",
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    return cmd


def run_assign(*, init_mode: str, jobs: int, skip_existing: bool) -> int:
    cmd = _build_cmd(init_mode=init_mode, jobs=jobs, skip_existing=skip_existing)
    print(f"ASSIGN init={init_mode} metric={CLUSTER_METRIC}:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def summarize_conv(path: Path) -> dict | None:
    ch = path / "convergence_history.csv"
    if not ch.is_file():
        return None
    h = pd.read_csv(ch)
    f0, fL = float(h["fitness"].iloc[0]), float(h["fitness"].iloc[-1])
    if f0 >= 1e5:
        return {
            "b1": f0, "end": fL, "gain_pct": 0.0, "gain_abs": 0.0,
            "blocks": len(h), "status": "STUCK_1e6",
        }
    gain_abs = f0 - fL
    gain_pct = gain_abs / f0 * 100.0
    if gain_pct > 0.01 or gain_abs > 0.5:
        status = "iyilesme"
    elif gain_pct > 0:
        status = "cok_az"
    else:
        status = "sabit"
    return {
        "b1": f0, "end": fL, "gain_pct": gain_pct, "gain_abs": gain_abs,
        "blocks": len(h), "status": status,
    }


def phase_compare() -> None:
    rows = []
    for init_mode, init_tag in INITS:
        for k in K_LIST:
            d = _assign_dir(init_tag, k)
            s = summarize_conv(d)
            row = {
                "cluster_metric": CLUSTER_METRIC,
                "init_mode": init_mode,
                "k": k,
                "dir": str(d),
                "exists": d.is_dir(),
            }
            if s:
                row.update(s)
            else:
                row.update(
                    b1=None, end=None, gain_pct=None, gain_abs=None,
                    blocks=None, status="no_conv_csv",
                )
            rows.append(row)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "compare", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument(
        "--inits",
        default="mkpp,random",
        help="Virgulle: mkpp,random",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.inits.split(",") if s.strip()}

    if args.phase in ("assign", "all"):
        for init_mode, _ in INITS:
            if init_mode not in only:
                continue
            rc = run_assign(
                init_mode=init_mode, jobs=args.jobs, skip_existing=args.skip_existing,
            )
            if rc != 0:
                sys.exit(rc)
    if args.phase in ("compare", "all"):
        phase_compare()


if __name__ == "__main__":
    main()
