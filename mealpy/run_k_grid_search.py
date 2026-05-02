"""
K-grid + preprocess/grid koşuları için assignment + cluster_knn tarayıcı.

İstenen boyutlar:
- K: 2 4 10 14 20 30 70 (varsayılan)
- algoritma: H9_QSA+CDO, H12_MFO+CDO, B3_MFO, H4_MFO+HHO, H13_HHO+GAop
- mod: manual (varsayılan; paper mode yok)
- gray: no_gray (LOF/gray-sheep kapalı)
- init: random (MkMeans++ yok)
- prune: prune / no_prune
- norm: zscore / no_zscore
- feature: pca / wnmf / none

Notlar:
- Paper mode'da zorunlu preset nedeniyle sadece şu kombinasyonlar anlamlıdır:
  gray=no_gray, norm=zscore, feature=pca (init mkpp/random serbest).
- Geçersiz veya anlamsız kombinasyonlar otomatik atlanır.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
GEN_SCRIPT = REPO_ROOT / "mealpy" / "generate_assignments.py"
WNMF_SCRIPT = REPO_ROOT / "wnmf" / "wnmf_experiment.py"

DEFAULT_KS = [2, 4, 10, 14, 20, 30, 70]
DEFAULT_ALGOS = ["H9_QSA+CDO", "H12_MFO+CDO", "B3_MFO", "H4_MFO+HHO", "H13_HHO+GAop"]


@dataclass(frozen=True)
class Combo:
    mode: str          # paper | manual
    gray: str          # lof | no_gray
    init_mode: str     # mkpp | random
    prune: str         # prune | no_prune
    zscore: str        # zscore | no_zscore
    feature: str       # pca | wnmf | none
    k: int


def _iter_combos(ns: argparse.Namespace) -> Iterable[Combo]:
    for mode, gray, init_mode, prune, zscore, feature, k in itertools.product(
        ns.modes, ns.gray_modes, ns.init_modes, ns.prune_modes, ns.zscore_modes, ns.features, ns.k
    ):
        c = Combo(
            mode=mode,
            gray=gray,
            init_mode=init_mode,
            prune=prune,
            zscore=zscore,
            feature=feature,
            k=k,
        )
        if not _is_valid(c):
            continue
        yield c


def _is_valid(c: Combo) -> bool:
    if c.mode == "paper":
        # Paper mode preset: zscore+pca+no_gray zorunlu; wnmf yok.
        if c.gray != "no_gray":
            return False
        if c.zscore != "zscore":
            return False
        if c.feature != "pca":
            return False
        return True
    # manual mod: hepsi serbest
    return True


def _prune_values(c: Combo, ns: argparse.Namespace) -> tuple[int, int]:
    if c.prune == "no_prune":
        return 0, 0
    return ns.min_user_ratings, ns.min_item_ratings


def _metric_suffix(metric: str) -> str:
    return {"euclidean": "_euc", "fuzzy": "_fuzzy"}.get(metric, "")


def _assign_suffix(c: Combo, ns: argparse.Namespace) -> str:
    min_user, min_item = _prune_values(c, ns)
    parts = [f"_pruneu{min_user}_i{min_item}"]
    if c.zscore == "zscore":
        parts.append("_zscore")
    if c.mode == "paper" or c.feature == "pca":
        parts.append(f"_pca{int(round(ns.pca * 100))}pct")
    if c.feature == "wnmf":
        parts.append(f"_wnmf{ns.wnmf_features}_{ns.wnmf_init}_trim{ns.inmed_trim_low:g}_{ns.inmed_trim_high:g}")
    parts.append(_metric_suffix(ns.cluster_metric))
    if c.mode == "paper":
        parts.append("_paper")
    elif c.gray == "no_gray":
        parts.append("_nogs")
    return "".join(parts)


def _assign_root(c: Combo) -> Path:
    root_name = "assignments_lof" if c.gray == "lof" else "assignments"
    return REPO_ROOT / "mealpy" / "results" / root_name


def _build_generate_cmd(c: Combo, ns: argparse.Namespace) -> List[str]:
    min_user, min_item = _prune_values(c, ns)
    cmd = [
        sys.executable, str(GEN_SCRIPT),
        "--dataset", ns.dataset,
        "--k", str(c.k),
        "--jobs", str(ns.jobs),
        "--cluster-metric", ns.cluster_metric,
        "--min-user-ratings", str(min_user),
        "--min-item-ratings", str(min_item),
        "--init-mode", c.init_mode,
        "--algo", *ns.algos,
    ]
    if c.mode == "paper":
        cmd.append("--paper-mode")
    else:
        if c.zscore == "zscore":
            cmd.append("--zscore")
        if c.feature == "pca":
            cmd += ["--pca", str(ns.pca)]
        elif c.feature == "wnmf":
            cmd += [
                "--wnmf-features", str(ns.wnmf_features),
                "--wnmf-init", ns.wnmf_init,
                "--inmed-trim-low", str(ns.inmed_trim_low),
                "--inmed-trim-high", str(ns.inmed_trim_high),
            ]

    if c.gray == "lof":
        cmd.append("--lof")
    else:
        cmd.append("--no-gray-sheep")
    return cmd


def _build_wnmf_cmd(c: Combo, ns: argparse.Namespace) -> List[str]:
    return [
        sys.executable, str(WNMF_SCRIPT),
        "--dataset", ns.dataset,
        "--mode", "baselines",
        "--algo", *ns.algos,
        "--k", str(c.k),
        "--assign-root", str(_assign_root(c).as_posix()),
        "--assign-suffix", _assign_suffix(c, ns),
        "--no-global",
        "--no-cluster-avg",
        "--similarity", ns.similarity,
        "--knn", str(ns.knn),
        "--min-common", str(ns.min_common),
    ]


def _latest_result_csv(dataset: str, k: int) -> Path | None:
    base = REPO_ROOT / "results" / "wnmf" / ("ml100k" if dataset == "100k" else "ml1m") / f"k{k}"
    if not base.exists():
        return None
    cand = sorted(
        base.glob(f"**/wnmf_results_{'ml100k' if dataset == '100k' else 'ml1m'}_k{k}_baselines*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cand[0] if cand else None


def _collect_cluster_knn_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            if row.get("scenario") == "cluster_knn":
                rows.append(row)
    return rows


def _append_summary(summary_csv: Path, combo: Combo, assign_suffix: str, rows: List[dict]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "mode", "gray", "init_mode", "prune", "zscore", "feature", "k", "assign_suffix",
            "algo_label", "mae", "rmse", "precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({
                "mode": combo.mode,
                "gray": combo.gray,
                "init_mode": combo.init_mode,
                "prune": combo.prune,
                "zscore": combo.zscore,
                "feature": combo.feature,
                "k": combo.k,
                "assign_suffix": assign_suffix,
                "algo_label": r.get("algo_label", ""),
                "mae": r.get("mae", ""),
                "rmse": r.get("rmse", ""),
                "precision_at_10": r.get("precision_at_10", ""),
                "recall_at_10": r.get("recall_at_10", ""),
                "f1_at_10": r.get("f1_at_10", ""),
                "ndcg_at_10": r.get("ndcg_at_10", ""),
            })


def _print_best(summary_csv: Path, topn: int = 10) -> None:
    if not summary_csv.exists():
        print("Özet bulunamadı.")
        return
    data = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                mae = float(row["mae"])
            except Exception:
                continue
            row["_mae"] = mae
            data.append(row)
    data.sort(key=lambda x: x["_mae"])
    print("\nEn iyi sonuçlar (MAE küçük daha iyi):")
    for i, r in enumerate(data[:topn], 1):
        print(
            f"{i:2d}) MAE={r['_mae']:.4f} | algo={r['algo_label']} | "
            f"k={r['k']} | mode={r['mode']} | gray={r['gray']} | init={r['init_mode']} | "
            f"prune={r.get('prune', '')} | "
            f"zscore={r['zscore']} | feat={r['feature']}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-grid: assignment + cluster_knn tarayıcı")
    p.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    p.add_argument("--algo", dest="algos", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--k", nargs="+", type=int, default=DEFAULT_KS)
    p.add_argument("--modes", nargs="+", choices=["paper", "manual"], default=["manual"])
    p.add_argument("--gray-modes", nargs="+", choices=["lof", "no_gray"], default=["no_gray"])
    p.add_argument("--init-modes", nargs="+", choices=["mkpp", "random"], default=["random"])
    p.add_argument("--prune-modes", nargs="+", choices=["prune", "no_prune"], default=["prune", "no_prune"])
    p.add_argument("--zscore-modes", nargs="+", choices=["zscore", "no_zscore"], default=["zscore", "no_zscore"])
    p.add_argument("--features", nargs="+", choices=["pca", "wnmf", "none"], default=["pca", "wnmf", "none"])
    p.add_argument("--pca", type=float, default=0.95)
    p.add_argument("--wnmf-features", type=int, default=20)
    p.add_argument("--wnmf-init", choices=["random", "inmed"], default="random")
    p.add_argument("--inmed-trim-low", type=float, default=5.0)
    p.add_argument("--inmed-trim-high", type=float, default=95.0)
    p.add_argument("--cluster-metric", choices=["pearson", "euclidean", "fuzzy"], default="euclidean")
    p.add_argument("--min-user-ratings", type=int, default=5)
    p.add_argument("--min-item-ratings", type=int, default=10)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--similarity", choices=["pearson", "cosine"], default="pearson")
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--min-common", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-runs", type=int, default=0, help="0=limitsiz")
    p.add_argument(
        "--summary-csv",
        default=str((REPO_ROOT / "results" / "grid_search" / "k_grid_summary.csv").as_posix()),
    )
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    summary_csv = Path(ns.summary_csv)
    combos = list(_iter_combos(ns))
    print(f"Toplam geçerli kombinasyon: {len(combos)}")
    if ns.max_runs > 0:
        combos = combos[:ns.max_runs]
        print(f"--max-runs uygulandı: {len(combos)}")

    for idx, c in enumerate(combos, 1):
        assign_suffix = _assign_suffix(c, ns)
        gen_cmd = _build_generate_cmd(c, ns)
        exp_cmd = _build_wnmf_cmd(c, ns)
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(combos)}] mode={c.mode} gray={c.gray} init={c.init_mode} prune={c.prune} "
              f"z={c.zscore} feat={c.feature} k={c.k}")
        print("GEN :", " ".join(gen_cmd))
        print("EXP :", " ".join(exp_cmd))
        if ns.dry_run:
            continue

        r1 = subprocess.run(gen_cmd, cwd=str(REPO_ROOT))
        if r1.returncode != 0:
            print(f"generate_assignments hata kodu: {r1.returncode}")
            continue
        r2 = subprocess.run(exp_cmd, cwd=str(REPO_ROOT))
        if r2.returncode != 0:
            print(f"wnmf_experiment hata kodu: {r2.returncode}")
            continue

        csv_path = _latest_result_csv(ns.dataset, c.k)
        if not csv_path:
            print("Sonuç CSV bulunamadı, atlandı.")
            continue
        rows = _collect_cluster_knn_rows(csv_path)
        if not rows:
            print("cluster_knn satırı yok.")
            continue
        _append_summary(summary_csv, c, assign_suffix, rows)

    if not ns.dry_run:
        print(f"\nÖzet CSV: {summary_csv}")
        _print_best(summary_csv, topn=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

