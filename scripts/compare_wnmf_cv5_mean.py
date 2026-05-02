"""
Compare WNMF cv5_mean result CSV files.

Default usage:
    python scripts/compare_wnmf_cv5_mean.py

Custom root / output:
    python scripts/compare_wnmf_cv5_mean.py --root results/wnmf/ml100k/k6/cv5_mean --out results/wnmf/ml100k/k6/cv5_mean/comparison.csv

Columns:
    tag, Algoritma, Senaryo, MAE, RMSE, ClStd, GS MAE, Wh MAE,
    Acc, Prec, Rec, F1, NDCG
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_ROOT = Path("results") / "wnmf" / "ml100k" / "k6" / "cv5_mean"

OUTPUT_COLUMNS = [
    "tag",
    "Algoritma",
    "Senaryo",
    "MAE",
    "RMSE",
    "ClStd",
    "GS MAE",
    "Wh MAE",
    "Acc",
    "Prec",
    "Rec",
    "F1",
    "NDCG",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WNMF cv5_mean klasoru altindaki CSV sonuclarini karsilastirir."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Sonuc kok klasoru (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Istege bagli CSV cikti dosyasi.",
    )
    parser.add_argument(
        "--sort",
        default="MAE",
        choices=["tag", "Algoritma", "Senaryo", "MAE", "RMSE", "ClStd", "GS MAE", "Wh MAE", "Acc", "Prec", "Rec", "F1", "NDCG"],
        help="Tablo siralama kolonu (default: MAE).",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Siralamayi buyukten kucuge yapar.",
    )
    parser.add_argument(
        "--tag-source",
        default="run",
        choices=["run", "hyperparam_tag", "file"],
        help="tag kolonu icin kullanilacak kaynak (default: run).",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Ekrana yazdirirken kullanilacak ondalik hane sayisi (default: 4).",
    )
    return parser.parse_args()


def iter_result_csvs(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.csv"), key=natural_sort_key)


def natural_sort_key(path: Path) -> List[object]:
    parts: List[object] = []
    text = str(path).lower()
    token = ""
    for char in text:
        if char.isdigit():
            token += char
            continue
        if token:
            parts.append(int(token))
            token = ""
        parts.append(char)
    if token:
        parts.append(int(token))
    return parts


def read_result_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        data_lines = [
            line
            for line in file_obj
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not data_lines:
        return []
    return list(csv.DictReader(data_lines))


def pick_tag(row: Dict[str, str], path: Path, root: Path, tag_source: str) -> str:
    if tag_source == "hyperparam_tag":
        return row.get("hyperparam_tag") or path.parent.name
    if tag_source == "file":
        return str(path.relative_to(root))
    return path.parent.name


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_value(value: Optional[float], digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def normalize_row(row: Dict[str, str], path: Path, root: Path, tag_source: str) -> Dict[str, object]:
    return {
        "tag": pick_tag(row, path, root, tag_source),
        "Algoritma": row.get("algo_label", ""),
        "Senaryo": row.get("scenario", ""),
        "MAE": to_float(row.get("mean_mae")) or to_float(row.get("mae")),
        "RMSE": to_float(row.get("mean_rmse")) or to_float(row.get("rmse")),
        "ClStd": to_float(row.get("cluster_mae_std")),
        "GS MAE": to_float(row.get("gray_mae")),
        "Wh MAE": to_float(row.get("white_mae")),
        "Acc": to_float(row.get("accuracy")),
        "Prec": to_float(row.get("precision_at_10")),
        "Rec": to_float(row.get("recall_at_10")),
        "F1": to_float(row.get("f1_at_10")),
        "NDCG": to_float(row.get("ndcg_at_10")),
    }


def sort_rows(rows: List[Dict[str, object]], sort_col: str, descending: bool) -> List[Dict[str, object]]:
    def key(row: Dict[str, object]) -> object:
        value = row.get(sort_col)
        if isinstance(value, float):
            return (0, value)
        if value is None:
            return (1, "")
        return (0, str(value))

    return sorted(rows, key=key, reverse=descending)


def printable_rows(rows: List[Dict[str, object]], digits: int) -> List[Dict[str, str]]:
    printable: List[Dict[str, str]] = []
    for row in rows:
        printable.append(
            {
                col: format_value(row[col], digits)
                if isinstance(row.get(col), float)
                else str(row.get(col) or "")
                for col in OUTPUT_COLUMNS
            }
        )
    return printable


def print_table(rows: List[Dict[str, str]]) -> None:
    widths = {
        col: max(len(col), *(len(row[col]) for row in rows))
        for col in OUTPUT_COLUMNS
    }
    header = "  ".join(col.ljust(widths[col]) for col in OUTPUT_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(row[col].ljust(widths[col]) for col in OUTPUT_COLUMNS))


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Klasor bulunamadi: {root}")

    rows: List[Dict[str, object]] = []
    for csv_path in iter_result_csvs(root):
        for raw_row in read_result_rows(csv_path):
            rows.append(normalize_row(raw_row, csv_path, root, args.tag_source))

    if not rows:
        raise SystemExit(f"CSV sonucu bulunamadi: {root}")

    rows = sort_rows(rows, args.sort, args.desc)
    display_rows = printable_rows(rows, args.digits)
    print_table(display_rows)

    if args.out:
        write_csv(display_rows, Path(args.out))
        print(f"\nCSV kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
