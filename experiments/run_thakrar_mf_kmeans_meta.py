"""
Thakrar et al. (2025) — Algorithm 2: MF + k-means(meta init) + CalculateAverageRating.

Paper pipeline (CSO → our 5 meta algorithms):

  Step 1  P,Q ← MatrixFactorization(R, L, α, λ, T)
          → generate_assignments: --feature-extraction wnmf --svd-components L
            (sparse-aware WNMF; paper Alg.4 SGD MF ile eşdeğer rol)

  Step 2a centroids ← MetaSearch(P, k)     → --algo B0_KMEANS B1_HHO …
  Step 2b clusters  ← KMeans(P, init)      → --kmeans-refine-overwrite (_kmref)

  Step 3  r̂ ← CalculateAverageRating       → wnmf_experiment --paper-mode
  Eval    80/20 holdout MAE                 → --train-only --eval-split random --fold 1

Usage:
  python experiments/run_thakrar_mf_kmeans_meta.py --dry-run
  python experiments/run_thakrar_mf_kmeans_meta.py --phase assign
  python experiments/run_thakrar_mf_kmeans_meta.py --phase eval
  python experiments/run_thakrar_mf_kmeans_meta.py --phase all --k 14 --latent 10
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
EXP = REPO / "wnmf" / "wnmf_experiment.py"
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"

# Paper Fig. 2–5: k∈{9,14,19}, L∈{5,10,15}. CSO → bizim 5 algo.
THAKRAR_ALGOS = ["B0_KMEANS", "B1_HHO", "B_AVOA", "HA_AVOAHGS", "IWO_HHO"]
DEFAULT_K = [9, 14, 19]
DEFAULT_LATENT = [5, 10, 15]
PAPER_SCENARIOS = frozenset({"calc_avg_rating", "calc_avg_rating_nc"})


def resolve_python(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if _VENV_PY.is_file():
        return str(_VENV_PY)
    return sys.executable


def assign_suffix(latent: int, cluster_k: int, with_kmref: bool = True) -> str:
    """wnmf_experiment --assign-suffix (label hariç; B0 için _kmref otomatik strip)."""
    base = (
        f"_euc_imkpp_nogs_trainonly_rand_f1"
        f"_none_wnmf{int(latent)}_k{int(cluster_k)}"
    )
    return f"{base}_kmref" if with_kmref else base


def assign_dir(algo: str, latent: int, cluster_k: int) -> Path:
    suf = assign_suffix(latent, cluster_k, with_kmref=(algo != "B0_KMEANS"))
    return ASSIGN_ROOT / "ml100k" / f"{algo}{suf}"


def run_cmd(cmd: List[str], dry_run: bool, python: str) -> int:
    if cmd and cmd[0] == sys.executable:
        cmd = [python] + cmd[1:]
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def phase_assignments(
    latent_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    jobs: int,
    skip_existing: bool,
    dry_run: bool,
    python: str,
    mo_weights: Optional[str],
    repulsion_lambda: float,
) -> int:
    k_str = [str(k) for k in ks]
    for latent in latent_dims:
        if skip_existing and all(
            assign_dir(algos[0], latent, k).joinpath("assignments.npy").is_file()
            for k in ks
        ):
            print(f"[skip-existing] L={latent} tüm K mevcut, atlandı.")
            continue
        cmd = [
            sys.executable,
            str(GEN),
            "--dataset",
            "100k",
            "--algo",
            *list(algos),
            "--no-prune",
            "--no-gray-sheep",
            "--preprocess",
            "none",
            "--feature-extraction",
            "wnmf",
            "--svd-components",
            str(int(latent)),
            "--init-mode",
            "mkpp",
            "--cluster-metric",
            "euclidean",
            "--fitness",
            "wcss",
            "--cluster-objective",
            "multi",
            "--kmeans-refine-overwrite",
            "--train-only",
            "--fold",
            "1",
            "--k",
            *k_str,
            "--jobs",
            str(jobs),
        ]
        if mo_weights:
            cmd.extend(["--mo-weights", mo_weights])
        if repulsion_lambda > 0.0:
            cmd.extend(["--centroid-repulsion-lambda", str(repulsion_lambda)])
        if skip_existing:
            cmd.append("--skip-existing")
        rc = run_cmd(cmd, dry_run, python)
        if rc != 0:
            return rc
    return 0


def phase_eval(
    latent_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    skip_existing: bool,
    dry_run: bool,
    python: str,
) -> int:
    for latent in latent_dims:
        for k in ks:
            suffix = assign_suffix(latent, k, with_kmref=True)
            missing = [
                a
                for a in algos
                if not assign_dir(a, latent, k).joinpath("assignments.npy").is_file()
            ]
            if missing and not dry_run:
                print(
                    f"[uyarı] L={latent} K={k} eksik assignment: {missing}; "
                    "eval yine denenecek."
                )
            cmd = [
                sys.executable,
                str(EXP),
                "--dataset",
                "100k",
                "--eval-split",
                "random",
                "--fold",
                "1",
                "--mode",
                "baselines",
                "--no-global",
                "--paper-mode",
                "--k",
                str(k),
                "--algo",
                *list(algos),
                "--assign-root",
                str(ASSIGN_ROOT.as_posix()),
                "--assign-suffix",
                suffix,
                "--top-n",
                "10",
                "--relevance-threshold",
                "4.0",
            ]
            if skip_existing:
                cmd.append("--skip-existing")
            rc = run_cmd(cmd, dry_run, python)
            if rc != 0:
                print(f"Hata: eval L={latent} K={k} rc={rc}", file=sys.stderr)
                return rc
    return 0


def _iter_paper_result_csvs() -> Iterable[Path]:
    base = REPO / "results" / "wnmf" / "ml100k"
    if not base.is_dir():
        return
    for p in sorted(base.rglob("wnmf_results_ml100k_k*_baselines.csv")):
        yield p


def _parse_latent_k_from_cmd(cmd_line: str) -> Tuple[Optional[int], Optional[int]]:
    mw = re.search(r"_wnmf(\d+)", cmd_line)
    mk = re.search(r"_k(\d+)(?:_kmref|_pwcss)?(?:\s|$|\"|,)", cmd_line)
    latent = int(mw.group(1)) if mw else None
    k = int(mk.group(1)) if mk else None
    return latent, k


def aggregate_summary(
    out_csv: Path,
    latent_dims: Sequence[int],
    ks: Sequence[int],
) -> None:
    best: dict = {}
    tag = "_none_wnmf"
    for csv_path in _iter_paper_result_csvs():
        lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
        cmd_line = next((ln for ln in lines if ln.startswith("# command:")), "")
        if tag not in cmd_line or "--paper-mode" not in cmd_line:
            continue
        latent, k_cl = _parse_latent_k_from_cmd(cmd_line)
        if latent is not None and latent not in latent_dims:
            continue
        if k_cl is not None and k_cl not in ks:
            continue
        data_lines = [ln for ln in lines if not ln.startswith("#")]
        if not data_lines:
            continue
        for row in csv.DictReader(data_lines):
            if row.get("scenario") not in PAPER_SCENARIOS:
                continue
            algo = row.get("algo_label", "")
            key = (latent, k_cl, algo)
            mtime = csv_path.stat().st_mtime
            prev = best.get(key)
            if prev is not None and prev["_mtime"] >= mtime:
                continue
            best[key] = {
                "_mtime": mtime,
                "latent_L": latent if latent is not None else "",
                "cluster_k": k_cl if k_cl is not None else row.get("assignment_k", ""),
                "algo": algo,
                "mae": row.get("mae", ""),
                "rmse": row.get("rmse", ""),
                "precision_at_10": row.get("precision_at_10", ""),
                "recall_at_10": row.get("recall_at_10", ""),
                "ndcg_at_10": row.get("ndcg_at_10", ""),
                "assign_dir": str(
                    assign_dir(algo, int(latent), int(k_cl)).relative_to(REPO)
                )
                if latent is not None and k_cl is not None and algo
                else "",
                "result_csv": str(csv_path.relative_to(REPO)),
            }

    rows_out = [{k: v for k, v in r.items() if k != "_mtime"} for r in best.values()]
    rows_out.sort(
        key=lambda r: (
            int(r["cluster_k"]) if str(r["cluster_k"]).isdigit() else 999,
            float(r["mae"]) if r["mae"] else 999,
        )
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "latent_L",
        "cluster_k",
        "algo",
        "mae",
        "rmse",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "assign_dir",
        "result_csv",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nThakrar özet: {len(rows_out)} satır -> {out_csv}")
    print("\nMAE (CalcAvgRating, paper-mode) — en iyi 15:")
    by_mae = sorted(
        rows_out,
        key=lambda r: float(r["mae"]) if r["mae"] else 999,
    )
    for i, r in enumerate(by_mae[:15], 1):
        print(
            f"{i:2d}) MAE={float(r['mae']):.4f} "
            f"L={r['latent_L']} K={r['cluster_k']} {r['algo']}"
        )


def parse_int_list(values: Sequence[str]) -> List[int]:
    return [int(v) for v in values]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Thakrar (2025) Alg.2 — MF+meta-kmeans+CalcAvgRating, 5 meta algo.",
    )
    ap.add_argument(
        "--phase",
        choices=["assign", "eval", "aggregate", "all"],
        default="all",
        help="assign / eval / aggregate / all (varsayılan: all)",
    )
    ap.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=DEFAULT_K,
        metavar="K",
        help=f"küme sayısı (paper: 9,14,19; varsayılan: {DEFAULT_K})",
    )
    ap.add_argument(
        "--latent",
        nargs="+",
        type=int,
        default=DEFAULT_LATENT,
        metavar="L",
        help=f"MF latent boyutu (paper: 5,10,15; varsayılan: {DEFAULT_LATENT})",
    )
    ap.add_argument(
        "--algo",
        nargs="+",
        default=THAKRAR_ALGOS,
        help="meta algoritmalar (CSO yerine)",
    )
    ap.add_argument("--jobs", type=int, default=2, help="assignment paralellik")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", type=str, default=None, help=".venv python yolu")
    ap.add_argument(
        "--mo-weights",
        type=str,
        default=None,
        help="meta MO preset (örn. spread); verilmezse generate_assignments default",
    )
    ap.add_argument(
        "--centroid-repulsion-lambda",
        type=float,
        default=0.0,
        help="centroid repulsion (0=kapalı; paper replikasyonu için 0 önerilir)",
    )
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO / "results" / "thakrar_mf_kmeans_meta_summary.csv",
    )
    args = ap.parse_args()
    python = resolve_python(args.python)

    print("Thakrar (2025) Algorithm 2 — meta pipeline")
    print(f"  Algolar : {list(args.algo)}")
    print(f"  K       : {list(args.k)}")
    print(f"  L       : {list(args.latent)}")
    print(f"  Suffix  : {assign_suffix(args.latent[0], args.k[0])} (örnek)")

    if args.phase in ("assign", "all"):
        rc = phase_assignments(
            args.latent,
            args.k,
            args.algo,
            args.jobs,
            args.skip_existing,
            args.dry_run,
            python,
            args.mo_weights,
            args.centroid_repulsion_lambda,
        )
        if rc != 0:
            return rc

    if args.phase in ("eval", "all"):
        rc = phase_eval(
            args.latent,
            args.k,
            args.algo,
            args.skip_existing,
            args.dry_run,
            python,
        )
        if rc != 0:
            return rc

    if args.phase in ("aggregate", "all") and not args.dry_run:
        aggregate_summary(args.summary_csv, args.latent, args.k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
