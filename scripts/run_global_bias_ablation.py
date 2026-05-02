"""
Global WNMF açıkken --bias vs --no-bias karşılaştırması (aynı fold, assignment, algoritmalar).

Örnek:
  python scripts/run_global_bias_ablation.py
  python scripts/run_global_bias_ablation.py --fold 1 --k 6 --epochs-global 100
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WNMF_SCRIPT = REPO_ROOT / "wnmf" / "wnmf_experiment.py"

DEFAULT_ALGOS = [
    "H9_QSA+CDO",
    "H12_MFO+CDO",
    "B3_MFO",
    "H4_MFO+HHO",
    "H13_HHO+GAop",
    "LIT_GOA",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global + bias açık/kapalı iki koşu")
    p.add_argument("--dataset", default="100k", choices=["100k", "1m", "both"])
    p.add_argument("--fold", type=int, default=1, help="ML-100K u{fold}.base/test (1-5)")
    p.add_argument("--k", type=int, default=6)
    p.add_argument(
        "--assign-root",
        type=Path,
        default=REPO_ROOT / "mealpy" / "results" / "assignments",
    )
    p.add_argument(
        "--assign-suffix",
        default="_pruneu5_i10_zscore_pca95pct_euc_paper",
    )
    p.add_argument("--algo", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--epochs-global", type=int, default=50)
    p.add_argument("--epochs-cluster", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(tag: str, bias_flag: list[str], ns: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(WNMF_SCRIPT),
        "--dataset",
        ns.dataset,
        "--fold",
        str(ns.fold),
        "--mode",
        "baselines",
        "--k",
        str(ns.k),
        "--assign-root",
        str(ns.assign_root.as_posix()),
        "--assign-suffix",
        ns.assign_suffix,
        "--algo",
        *ns.algo,
        "--epochs-global",
        str(ns.epochs_global),
        "--epochs-cluster",
        str(ns.epochs_cluster),
        "--note",
        tag,
    ] + bias_flag
    print("\n" + "=" * 72)
    print("CMD:", " ".join(cmd))
    print("=" * 72 + "\n")
    if ns.dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def _latest_baselines_csv(k: int, fold: int | None) -> Path | None:
    if fold is None:
        base = REPO_ROOT / "results" / "wnmf" / "ml100k" / f"k{k}"
    else:
        base = REPO_ROOT / "results" / "wnmf" / "ml100k" / f"k{k}" / f"fold{fold}"
    if not base.is_dir():
        return None
    cand = sorted(
        base.glob(f"run*/wnmf_results_ml100k_k{k}_baselines.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cand[0] if cand else None


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def main() -> int:
    ns = parse_args()
    if ns.dataset != "100k":
        print("Bu script şimdilik ml100k özetini yazdırır.")

    rc = _run("global_bias_on_fold%d" % ns.fold, ["--bias"], ns)
    if rc != 0:
        return rc
    path_on = _latest_baselines_csv(ns.k, ns.fold)
    if not path_on:
        print("bias_on sonrası CSV bulunamadı.")
        return 1
    print(f"[bias ON]  -> {path_on}")

    rc = _run("global_bias_off_fold%d" % ns.fold, ["--no-bias"], ns)
    if rc != 0:
        return rc
    path_off = _latest_baselines_csv(ns.k, ns.fold)
    if not path_off or path_off == path_on:
        print("bias_off sonrası yeni CSV bulunamadı.")
        return 1
    print(f"[bias OFF] -> {path_off}")

    rows_on = { (r.get("scenario"), r.get("algo_label")): r for r in _load_rows(path_on) }
    rows_off = { (r.get("scenario"), r.get("algo_label")): r for r in _load_rows(path_off) }
    keys = sorted(set(rows_off.keys()) | set(rows_on.keys()))

    out_dir = REPO_ROOT / "results" / "wnmf" / "ml100k" / f"k{ns.k}" / f"fold{ns.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_ablation_compare.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["scenario", "algo_label", "mae_bias_on", "mae_bias_off", "delta_off_minus_on"],
        )
        for key in keys:
            ro = rows_on.get(key, {})
            rf = rows_off.get(key, {})
            try:
                m_on = float(ro.get("mae", "nan"))
            except (TypeError, ValueError):
                m_on = float("nan")
            try:
                m_off = float(rf.get("mae", "nan"))
            except (TypeError, ValueError):
                m_off = float("nan")
            d = m_off - m_on if m_on == m_on and m_off == m_off else float("nan")
            w.writerow([
                key[0],
                key[1],
                f"{m_on:.6f}" if m_on == m_on else "",
                f"{m_off:.6f}" if m_off == m_off else "",
                f"{d:.6f}" if d == d else "",
            ])

    print(f"\nÖzet: {out_path}")
    print("(delta_off_minus_on = mae_off - mae_on; pozitif → bias ON daha düşük MAE, yani bias ON daha iyi)")

    # cluster_knn için en iyi algo iki koşuda
    def best_knn(rows: dict) -> tuple[str, float] | None:
        cand = [
            (k[1], float(r["mae"]))
            for k, r in rows.items()
            if k[0] == "cluster_knn" and r.get("mae") not in ("", None)
        ]
        if not cand:
            return None
        return min(cand, key=lambda x: x[1])

    b_on = best_knn(rows_on)
    b_off = best_knn(rows_off)
    print("\ncluster_knn en düşük MAE:")
    print(f"  bias ON : {b_on}")
    print(f"  bias OFF: {b_off}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
