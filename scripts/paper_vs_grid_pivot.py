"""
Paper (k=6, literatür preset) cv5_mean sonuçlarını grid araması CSV'si ile birleştirip pivot tablolar üretir.

Çıktılar (varsayılan --out-dir):
  - paper_vs_grid_mae_pivot.csv, paper_vs_grid_mae_delta_pivot.csv
  - paper_vs_grid_rmse_pivot.csv
  - paper_vs_grid_precision_at_10_pivot.csv (+ _delta_: grid − paper, pozitif = grid daha iyi)
  - paper_vs_grid_recall_at_10_pivot.csv (+ delta)
  - paper_vs_grid_f1_at_10_pivot.csv (+ delta)
  - paper_vs_grid_ndcg_at_10_pivot.csv (+ delta)
  - paper_vs_grid_long.csv : tüm metrikler uzun formatta

Örnek:
  python scripts/paper_vs_grid_pivot.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TOP5_ALGOS = [
    "H9_QSA+CDO",
    "H12_MFO+CDO",
    "B3_MFO",
    "H4_MFO+HHO",
    "H13_HHO+GAop",
]

METRICS = [
    "mae",
    "rmse",
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "ndcg_at_10",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper k=6 vs grid pivot CSV üret")
    p.add_argument(
        "--grid",
        type=Path,
        default=REPO_ROOT / "results" / "grid_search" / "top5_manual_no_gray_random_grid.csv",
        help="run_k_grid_search özet CSV",
    )
    p.add_argument(
        "--paper-root",
        type=Path,
        default=REPO_ROOT / "results" / "wnmf" / "ml100k" / "k6" / "cv5_mean",
        help="Paper cv5_mean run klasörleri kökü (otomatik arama için)",
    )
    p.add_argument(
        "--paper",
        type=Path,
        default=None,
        help="Paper sonuç CSV (verilmezse --paper-root altında en güncel paper satırı aranır)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "grid_search",
        help="Pivot çıktı dizini",
    )
    return p.parse_args()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def _is_paper_result_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            head = f.read(800).lower()
    except OSError:
        return False
    return "paper" in head and "baselines" in path.name.lower()


def find_latest_paper_csv(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    best: Optional[Path] = None
    best_mtime = -1.0
    for path in root.rglob("wnmf_results_*_baselines_cv5_mean.csv"):
        if not _is_paper_result_csv(path):
            continue
        try:
            m = path.stat().st_mtime
        except OSError:
            continue
        if m > best_mtime:
            best_mtime = m
            best = path
    return best


def _grid_scenario_key(row: Dict[str, str]) -> str:
    return "g_k{}_{}_{}_{}".format(
        row["k"],
        row["prune"],
        row["zscore"],
        row["feature"],
    )


def _float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key) or "nan")


def build_paper_metrics(
    rows: List[Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, float]], int]:
    """metric -> algo -> value; assignment_k."""
    out: Dict[str, Dict[str, float]] = {m: {} for m in METRICS}
    k_val = 0
    for row in rows:
        if row.get("scenario") != "cluster_knn":
            continue
        label = row.get("algo_label", "")
        if label not in TOP5_ALGOS:
            continue
        for m in METRICS:
            out[m][label] = _float(row, m)
        if not k_val:
            try:
                k_val = int(row.get("assignment_k") or row.get("k") or 0)
            except ValueError:
                k_val = 0
    return out, k_val


def _grid_sort_key(s: str) -> Tuple[int, str]:
    if not s.startswith("g_k"):
        return (0, s)
    rest = s[3:]
    k_str, _, tail = rest.partition("_")
    try:
        return (int(k_str), tail)
    except ValueError:
        return (0, s)


def write_pivot(
    out_dir: Path,
    fname: str,
    paper_col: str,
    paper_vals: Dict[str, float],
    grid_block: Dict[str, Dict[str, float]],
    grid_keys: List[str],
    *,
    delta: bool,
    metric: str,
) -> None:
    if delta:
        cols = [f"paper_ref_{metric}"] + grid_keys
    else:
        cols = [paper_col] + grid_keys
    out = out_dir / fname
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo_label"] + cols)
        ref_header = f"paper_ref_{metric}"
        for algo in TOP5_ALGOS:
            row_out = [algo]
            p = paper_vals.get(algo, float("nan"))
            for c in cols:
                if delta and c == ref_header:
                    row_out.append(f"{p:.10g}" if p == p else "")
                    continue
                if not delta and c == paper_col:
                    row_out.append(f"{p:.10g}" if p == p else "")
                    continue
                g = grid_block[c].get(algo, float("nan"))
                if delta:
                    d = g - p if (g == g and p == p) else float("nan")
                    row_out.append(f"{d:.10g}" if d == d else "")
                else:
                    row_out.append(f"{g:.10g}" if g == g else "")
            writer.writerow(row_out)


def main() -> int:
    args = parse_args()
    grid_path = args.grid
    if not grid_path.is_file():
        raise SystemExit(f"Grid CSV yok: {grid_path}")

    paper_path = args.paper
    if paper_path is None:
        paper_path = find_latest_paper_csv(args.paper_root)
    if paper_path is None or not paper_path.is_file():
        raise SystemExit(
            f"Paper cv5_mean CSV bulunamadı. --paper verin veya {args.paper_root} altında "
            "yorum satırında 'paper' geçen bir wnmf_results_*_baselines_cv5_mean.csv oluşturun."
        )

    grid_rows = _read_csv_rows(grid_path)
    paper_rows = _read_csv_rows(paper_path)
    paper_by_metric, paper_k = build_paper_metrics(paper_rows)

    paper_col = f"paper_k{paper_k}_zscore_pca95pct_euc" if paper_k else "paper_k6_baseline"

    grid_keys: List[str] = []
    seen = set()
    # metric -> sk -> algo -> float
    grid_by_metric: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in METRICS}
    for row in grid_rows:
        if row.get("algo_label") not in TOP5_ALGOS:
            continue
        sk = _grid_scenario_key(row)
        if sk not in seen:
            seen.add(sk)
            grid_keys.append(sk)
        algo = row["algo_label"]
        for m in METRICS:
            grid_by_metric[m].setdefault(sk, {})[algo] = _float(row, m)

    grid_keys.sort(key=_grid_sort_key)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_path = args.out_dir / "paper_vs_grid_long.csv"
    with long_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["algo_label", "scenario_key", "source", "k_or_note"]
            + METRICS
        )
        for algo in TOP5_ALGOS:
            w.writerow(
                [algo, paper_col, "paper", str(paper_k)]
                + [f"{paper_by_metric[m].get(algo, float('nan')):.10g}" for m in METRICS]
            )
        for sk in grid_keys:
            for algo in TOP5_ALGOS:
                w.writerow(
                    [algo, sk, "grid", ""]
                    + [
                        f"{grid_by_metric[m][sk].get(algo, float('nan')):.10g}"
                        for m in METRICS
                    ]
                )

    for m in METRICS:
        stem = m if m in ("mae", "rmse") else m
        write_pivot(
            args.out_dir,
            f"paper_vs_grid_{stem}_pivot.csv",
            paper_col,
            paper_by_metric[m],
            grid_by_metric[m],
            grid_keys,
            delta=False,
            metric=m,
        )
        write_pivot(
            args.out_dir,
            f"paper_vs_grid_{stem}_delta_pivot.csv",
            paper_col,
            paper_by_metric[m],
            grid_by_metric[m],
            grid_keys,
            delta=True,
            metric=m,
        )

    # Eski tek dosya adıyla uyum: mae_delta zaten paper_vs_grid_mae_delta_pivot.csv
    meta_path = args.out_dir / "paper_vs_grid_meta.txt"
    meta_path.write_text(
        f"paper_csv: {paper_path.as_posix()}\n"
        f"grid_csv: {grid_path.as_posix()}\n"
        f"paper_k: {paper_k}\n"
        f"metrics: {', '.join(METRICS)}\n"
        f"algos: {', '.join(TOP5_ALGOS)}\n"
        f"grid_scenarios: {len(grid_keys)}\n"
        f"delta: grid - paper (mae/rmse: negatif iyi; precision/recall/f1/ndcg: pozitif iyi)\n",
        encoding="utf-8",
    )

    print(f"Paper CSV: {paper_path}")
    print(f"Grid CSV : {grid_path}")
    print(f"Yazıldı  : {args.out_dir} (pivot + delta + paper_vs_grid_long.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
