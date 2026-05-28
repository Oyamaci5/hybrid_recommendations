"""
K × WNMF-latent × kNN grid: assignment üret + cluster KNN değerlendir.

Varsayılan konfig (önceki cosine koşusu ile uyumlu):
  no-prune, no-gray-sheep, preprocess=none, WNMF feature, mkpp, kmeans-refine, cosine kNN

Örnek:
  python mealpy/run_k_wnmf_knn_grid.py
  python mealpy/run_k_wnmf_knn_grid.py --phase eval --skip-existing
  python mealpy/run_k_wnmf_knn_grid.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
EXP = REPO / "wnmf" / "wnmf_experiment.py"
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"


def resolve_python(explicit: Optional[str] = None) -> str:
    """Repo .venv varsa onu kullan (scikit-surprise için); yoksa sys.executable."""
    if explicit:
        return explicit
    if _VENV_PY.is_file():
        return str(_VENV_PY)
    return sys.executable

DEFAULT_ALGOS = ["B0_KMEANS", "B1_HHO", "B_AVOA", "HA_AVOAHGS", "IWO_HHO"]
DEFAULT_K = [5, 7, 10, 14, 21, 27, 30]
DEFAULT_WNMF = [10, 20, 30, 40, 50]
DEFAULT_KNN = [20, 30, 40, 50]
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"
_GRID_ASSIGN_TAG = "_euc_imkpp_nogs_none_wnmf"
_CLUSTER_KNN_SCENARIOS = frozenset({
    "cluster_knn",
    "cluster_knn_baseline",
    "cluster_knn_baseline_gs_split",
    "cluster_knn_native_baseline",
    "cluster_knn_native_baseline_gs_split",
    "cluster_knn_with_means",
    "cluster_knn_with_means_gs_split",
    "cluster_knn_gs_split",
})


def assign_suffix(wnmf_dim: int, cluster_k: int, with_kmref: bool = True) -> str:
    base = f"_euc_imkpp_nogs_none_wnmf{int(wnmf_dim)}_k{int(cluster_k)}"
    return f"{base}_kmref" if with_kmref else base


def assign_dir(algo: str, wnmf_dim: int, cluster_k: int) -> Path:
    kmref = "_kmref" if algo != "B0_KMEANS" else ""
    name = f"{algo}_euc_imkpp_nogs_none_wnmf{wnmf_dim}_k{cluster_k}{kmref}"
    return ASSIGN_ROOT / "ml100k" / name


def load_assignment_metrics(algo: str, wnmf_dim: int, cluster_k: int) -> dict:
    """Assignment klasöründen WCSS, silhouette ve küme boyutu istatistikleri."""
    d = assign_dir(algo, wnmf_dim, cluster_k)
    out = {
        "wcss": "",
        "silhouette_euclidean": "",
        "silhouette_cosine": "",
        "n_gray_sheep": "",
        "cluster_size_min": "",
        "cluster_size_max": "",
        "cluster_size_mean": "",
        "assign_dir": str(d.relative_to(REPO)) if d.is_dir() else "",
    }
    cm = d / "cluster_metrics.csv"
    if cm.is_file():
        with cm.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f), None)
        if row:
            out["wcss"] = row.get("wcss", "")
            out["silhouette_euclidean"] = row.get("silhouette_euclidean", "")
            out["silhouette_cosine"] = row.get("silhouette_cosine", "")
            out["n_gray_sheep"] = row.get("n_gray_sheep", "")

    summary = d / "assignment_summary.csv"
    if summary.is_file():
        from collections import Counter
        counts: Counter = Counter()
        with summary.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("is_gray_sheep", "0")).strip() in ("0", "False", "false"):
                    counts[int(row["cluster_id"])] += 1
        if counts:
            sizes = list(counts.values())
            out["cluster_size_min"] = min(sizes)
            out["cluster_size_max"] = max(sizes)
            out["cluster_size_mean"] = round(sum(sizes) / len(sizes), 2)
    return out


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool, python: str) -> int:
    if cmd and cmd[0] == sys.executable:
        cmd = [python] + cmd[1:]
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def phase_assignments(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    algos: Sequence[str],
    jobs: int,
    skip_existing: bool,
    dry_run: bool,
    python: str,
) -> int:
    k_str = [str(k) for k in ks]
    algo_str = list(algos)
    for w in wnmf_dims:
        if skip_existing and all(
            assign_dir(algo_str[0], w, k).joinpath("assignments.npy").is_file()
            for k in ks
        ):
            print(f"[skip-existing] WNMF={w} tüm K mevcut, atlandı.")
            continue
        cmd = [
            sys.executable, str(GEN),
            "--dataset", "100k",
            "--algo", *algo_str,
            "--no-prune", "--no-gray-sheep",
            "--preprocess", "none",
            "--feature-extraction", "wnmf",
            "--svd-components", str(w),
            "--init-mode", "mkpp",
            "--kmeans-refine",
            "--k", *k_str,
            "--jobs", str(jobs),
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        rc = run_cmd(cmd, REPO, dry_run, python)
        if rc != 0:
            return rc
    return 0


def phase_eval(
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    knn_vals: Sequence[int],
    algos: Sequence[str],
    fold: int,
    skip_existing: bool,
    dry_run: bool,
    python: str,
) -> int:
    knn_str = [str(k) for k in knn_vals]
    for w in wnmf_dims:
        for k in ks:
            suffix = assign_suffix(w, k, with_kmref=True)
            if skip_existing and not dry_run:
                missing = [
                    a for a in algos
                    if not assign_dir(a, w, k).is_dir()
                ]
                if missing:
                    print(
                        f"[uyarı] WNMF={w} K={k} eksik assignment: {missing}; "
                        "eval yine de denenecek."
                    )
            cmd = [
                sys.executable, str(EXP),
                "--dataset", "100k",
                "--eval-split", "random",
                "--fold", str(fold),
                "--mode", "baselines",
                "--no-global", "--no-cluster-avg",
                "--cluster-knn-variant", "baseline",
                "--similarity", "cosine",
                "--knn", *knn_str,
                "--k", str(k),
                "--algo", *list(algos),
                "--assign-root", str(ASSIGN_ROOT.as_posix()),
                "--assign-suffix", suffix,
                "--top-n", "10",
                "--relevance-threshold", "4.0",
            ]
            rc = run_cmd(cmd, REPO, dry_run, python)
            if rc != 0:
                print(f"Hata: eval WNMF={w} K={k} rc={rc}", file=sys.stderr)
                return rc
    return 0


def _iter_result_csvs() -> Iterable[Path]:
    base = REPO / "results" / "wnmf" / "ml100k"
    if not base.is_dir():
        return
    for p in sorted(base.rglob("wnmf_results_ml100k_k*_baselines.csv")):
        yield p


def _parse_suffix_meta(path: Path, assign_suffix_col: str = "") -> Tuple[Optional[int], Optional[int]]:
    text = assign_suffix_col or path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
    mw = re.search(r"_wnmf(\d+)", text)
    mk = re.search(r"_k(\d+)(?:_kmref|_pwcss)?(?:\s|$|\"|,)", text)
    w = int(mw.group(1)) if mw else None
    k = int(mk.group(1)) if mk else None
    if k is None:
        m2 = re.search(r"\\k(\d+)\\", str(path))
        if m2:
            k = int(m2.group(1))
    return w, k


def aggregate_summary(
    out_csv: Path,
    wnmf_dims: Sequence[int],
    ks: Sequence[int],
    knn_vals: Sequence[int],
) -> None:
    best: dict = {}
    for csv_path in _iter_result_csvs():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            lines = f.readlines()
        if not lines:
            continue
        header_line = next((ln for ln in lines if not ln.startswith("#")), None)
        cmd_line = next((ln for ln in lines if ln.startswith("# command:")), "")
        if _GRID_ASSIGN_TAG not in cmd_line:
            continue
        w_dim, k_cl = _parse_suffix_meta(csv_path, cmd_line)
        data_lines = [ln for ln in lines if not ln.startswith("#")]
        reader = csv.DictReader(data_lines)
        for row in reader:
            if row.get("scenario") not in _CLUSTER_KNN_SCENARIOS:
                continue
            if w_dim is not None and w_dim not in wnmf_dims:
                continue
            if k_cl is not None and k_cl not in ks:
                continue
            knn_k = row.get("k_neighbors", "")
            try:
                if int(float(knn_k)) not in knn_vals:
                    continue
            except (TypeError, ValueError):
                continue
            if row.get("similarity") != "cosine":
                continue
            key = (w_dim, k_cl, knn_k, row.get("algo_label", ""))
            mtime = csv_path.stat().st_mtime
            prev = best.get(key)
            if prev is not None and prev["_mtime"] >= mtime:
                continue
            algo = row.get("algo_label", "")
            assign_meta = {}
            if w_dim is not None and k_cl is not None and algo:
                assign_meta = load_assignment_metrics(algo, int(w_dim), int(k_cl))
            best[key] = {
                "_mtime": mtime,
                "wnmf_dim": w_dim if w_dim is not None else "",
                "cluster_k": k_cl if k_cl is not None else row.get("assignment_k", ""),
                "knn_k": knn_k,
                "algo": algo,
                "similarity": row.get("similarity", ""),
                "wcss": assign_meta.get("wcss", ""),
                "silhouette_euclidean": assign_meta.get("silhouette_euclidean", ""),
                "silhouette_cosine": assign_meta.get("silhouette_cosine", ""),
                "n_gray_sheep": assign_meta.get("n_gray_sheep", ""),
                "cluster_size_min": assign_meta.get("cluster_size_min", ""),
                "cluster_size_max": assign_meta.get("cluster_size_max", ""),
                "cluster_size_mean": assign_meta.get("cluster_size_mean", ""),
                "mae": row.get("mae", ""),
                "rmse": row.get("rmse", ""),
                "accuracy": row.get("accuracy", ""),
                "precision_at_10": row.get("precision_at_10", ""),
                "recall_at_10": row.get("recall_at_10", ""),
                "f1_at_10": row.get("f1_at_10", ""),
                "ndcg_at_10": row.get("ndcg_at_10", ""),
                "assign_dir": assign_meta.get("assign_dir", ""),
                "result_csv": str(csv_path.relative_to(REPO)),
            }

    rows_out = [{k: v for k, v in r.items() if k != "_mtime"} for r in best.values()]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "wnmf_dim", "cluster_k", "knn_k", "algo", "similarity",
        "wcss", "silhouette_euclidean", "silhouette_cosine",
        "n_gray_sheep", "cluster_size_min", "cluster_size_max", "cluster_size_mean",
        "mae", "rmse", "accuracy",
        "precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10",
        "assign_dir", "result_csv",
    ]
    rows_out.sort(key=lambda r: (
        float(r["mae"]) if r["mae"] else 999,
    ))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    leaders_csv = out_csv.with_name(out_csv.stem + "_leaders.csv")
    _write_metric_leaders(rows_out, leaders_csv, fields)

    print(f"\nÖzet: {len(rows_out)} satır -> {out_csv}")
    print(f"Lider tablosu -> {leaders_csv}")
    print("\nEn iyi 15 (MAE):")
    for i, r in enumerate(rows_out[:15], 1):
        sil = r.get("silhouette_euclidean", "")
        sil_s = f" sil={float(sil):.4f}" if sil else ""
        print(
            f"{i:2d}) MAE={float(r['mae']):.4f} "
            f"P@10={float(r['precision_at_10']):.4f} R@10={float(r['recall_at_10']):.4f} "
            f"NDCG={float(r['ndcg_at_10']):.4f}{sil_s} | "
            f"wnmf={r['wnmf_dim']} K={r['cluster_k']} knn={r['knn_k']} {r['algo']}"
        )


def _write_metric_leaders(rows: List[dict], out_path: Path, all_fields: List[str]) -> None:
    """Her metrik için en iyi satırı (min/max yönüne göre) tek CSV'de topla."""
    specs = [
        ("mae", "min", "En düşük MAE"),
        ("rmse", "min", "En düşük RMSE"),
        ("precision_at_10", "max", "En yüksek Prec@10"),
        ("recall_at_10", "max", "En yüksek Rec@10"),
        ("f1_at_10", "max", "En yüksek F1@10"),
        ("ndcg_at_10", "max", "En yüksek NDCG@10"),
        ("accuracy", "max", "En yüksek Accuracy"),
        ("silhouette_euclidean", "max", "En yüksek Silhouette (eucl)"),
        ("silhouette_cosine", "max", "En yüksek Silhouette (cosine)"),
        ("wcss", "min", "En düşük WCSS"),
    ]
    leaders: List[dict] = []
    for col, direction, label in specs:
        cand = [r for r in rows if r.get(col) not in ("", None)]
        if not cand:
            continue
        key_fn = lambda r, c=col: float(r[c])
        best = min(cand, key=key_fn) if direction == "min" else max(cand, key=key_fn)
        leaders.append({
            "metric": label,
            "metric_col": col,
            "value": best[col],
            "wnmf_dim": best.get("wnmf_dim", ""),
            "cluster_k": best.get("cluster_k", ""),
            "knn_k": best.get("knn_k", ""),
            "algo": best.get("algo", ""),
            "wcss": best.get("wcss", ""),
            "silhouette_euclidean": best.get("silhouette_euclidean", ""),
            "mae": best.get("mae", ""),
            "precision_at_10": best.get("precision_at_10", ""),
            "recall_at_10": best.get("recall_at_10", ""),
            "ndcg_at_10": best.get("ndcg_at_10", ""),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    leader_fields = list(leaders[0].keys()) if leaders else ["metric"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=leader_fields)
        w.writeheader()
        w.writerows(leaders)
    print("\nMetrik liderleri:")
    for row in leaders:
        print(
            f"  {row['metric']:28} {float(row['value']):.4f}  "
            f"wnmf={row['wnmf_dim']} K={row['cluster_k']} knn={row['knn_k']} {row['algo']}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K × WNMF × kNN grid koşusu")
    p.add_argument(
        "--phase", choices=["all", "assign", "eval", "summary"], default="all",
    )
    p.add_argument("--wnmf", type=int, nargs="+", default=DEFAULT_WNMF)
    p.add_argument("--k", type=int, nargs="+", default=DEFAULT_K)
    p.add_argument("--knn", type=int, nargs="+", default=DEFAULT_KNN)
    p.add_argument("--algo", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--python", type=str, default=None,
        help="Python yorumlayıcı (varsayılan: repo/.venv/Scripts/python.exe varsa o)",
    )
    p.add_argument(
        "--summary-csv",
        default=str(REPO / "results" / "grid" / "k_wnmf_knn_grid_summary.csv"),
    )
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    python = resolve_python(ns.python)
    t0 = time.time()
    print("Grid boyutu:")
    print(f"  Python    : {python}")
    print(f"  WNMF dims : {ns.wnmf}")
    print(f"  K         : {ns.k}")
    print(f"  kNN       : {ns.knn}")
    print(f"  Algo      : {ns.algo}")
    print(f"  Assign    : {len(ns.wnmf) * len(ns.k) * len(ns.algo)} klasör hedefi")
    print(f"  Eval runs : {len(ns.wnmf) * len(ns.k)} × {len(ns.knn)} kNN")

    if ns.phase in ("all", "assign"):
        rc = phase_assignments(
            ns.wnmf, ns.k, ns.algo, ns.jobs, ns.skip_existing, ns.dry_run, python,
        )
        if rc != 0:
            return rc

    if ns.phase in ("all", "eval"):
        rc = phase_eval(
            ns.wnmf, ns.k, ns.knn, ns.algo, ns.fold, ns.skip_existing, ns.dry_run, python,
        )
        if rc != 0:
            return rc

    if ns.phase in ("all", "summary") and not ns.dry_run:
        aggregate_summary(Path(ns.summary_csv), ns.wnmf, ns.k, ns.knn)

    print(f"\nBitti ({(time.time() - t0) / 60:.1f} dk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
