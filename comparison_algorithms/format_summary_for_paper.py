from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


def _extract_fn(function_name: str) -> str:
    match = re.match(r"(F\d+)", str(function_name).strip())
    return match.group(1) if match else str(function_name).strip()


def _parse_mean(value: str) -> Tuple[str, bool]:
    text = str(value).strip()
    starred = text.endswith("*")
    if starred:
        text = text[:-1].strip()
    return text, starred


def _parse_std(value: str) -> str:
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _to_float_or_none(value: str):
    try:
        return float(value)
    except Exception:
        return None


def _build_paper_rows(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, str]]]:
    algo_cols = [c for c in df.columns if c not in ("Function", "Type", "Metric")]
    body = df[df["Metric"].isin(["Mean", "Std"])].copy()
    ordered_functions = []
    for fn in body["Function"]:
        if fn not in ordered_functions:
            ordered_functions.append(fn)

    rows: List[Dict[str, str]] = []
    for fn_name in ordered_functions:
        fn_df = body[body["Function"] == fn_name]
        mean_row = fn_df[fn_df["Metric"] == "Mean"].iloc[0]
        std_row = fn_df[fn_df["Metric"] == "Std"].iloc[0]

        mean_values: Dict[str, str] = {}
        std_values: Dict[str, str] = {}
        starred_algos: List[str] = []
        numeric_means: Dict[str, float] = {}
        for algo in algo_cols:
            mean_text, is_starred = _parse_mean(mean_row[algo])
            std_text = _parse_std(std_row[algo])
            mean_values[algo] = mean_text
            std_values[algo] = std_text
            if is_starred:
                starred_algos.append(algo)
            parsed = _to_float_or_none(mean_text)
            if parsed is not None:
                numeric_means[algo] = parsed

        if starred_algos:
            best_algo = starred_algos[0]
        elif numeric_means:
            best_algo = min(numeric_means, key=numeric_means.get)
        else:
            best_algo = None

        row: Dict[str, str] = {"Fn.": _extract_fn(fn_name)}
        for algo in algo_cols:
            mean_text = mean_values[algo]
            std_text = std_values[algo]
            if best_algo == algo:
                mean_text = f"\\textbf{{{mean_text}}}"
                std_text = f"\\textbf{{{std_text}}}"
            row[f"{algo}_MEAN"] = mean_text
            row[f"{algo}_STD"] = std_text
        rows.append(row)

    return algo_cols, rows


def _write_clean_csv(path: str, algo_cols: List[str], rows: List[Dict[str, str]]) -> None:
    columns = ["Fn."]
    for algo in algo_cols:
        columns.extend([f"{algo}_MEAN", f"{algo}_STD"])
    out_df = pd.DataFrame(rows, columns=columns)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out_df.to_csv(path, index=False)


def _write_latex(path: str, algo_cols: List[str], rows: List[Dict[str, str]]) -> None:
    col_spec = "l" + "cc" * len(algo_cols)
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Mean and standard deviation of the fitness value over 30 runs}")
    lines.append("\\label{tab:ha-summary}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    top_header = ["Fn."]
    for algo in algo_cols:
        top_header.append(f"\\multicolumn{{2}}{{c}}{{{algo}}}")
    lines.append(" & ".join(top_header) + " \\\\")

    sub_header = [""]
    for _ in algo_cols:
        sub_header.extend(["MEAN", "STD"])
    lines.append(" & ".join(sub_header) + " \\\\")
    lines.append("\\hline")

    for row in rows:
        cells = [row["Fn."]]
        for algo in algo_cols:
            cells.append(row[f"{algo}_MEAN"])
            cells.append(row[f"{algo}_STD"])
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\\\[2pt]")
    lines.append("\\small{Bold indicates the best value.}")
    lines.append("\\end{table*}")
    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ha_summary_table.csv into paper-ready CSV and LaTeX table."
    )
    parser.add_argument(
        "--input",
        default="comparison_algorithms/results/ha_summary_table.csv",
        help="Input summary CSV path.",
    )
    parser.add_argument(
        "--out-csv",
        default="comparison_algorithms/results/ha_summary_table_paper.csv",
        help="Output wide-format CSV path.",
    )
    parser.add_argument(
        "--out-tex",
        default="comparison_algorithms/results/ha_summary_table_paper.tex",
        help="Output LaTeX table path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    algo_cols, rows = _build_paper_rows(df)
    _write_clean_csv(args.out_csv, algo_cols, rows)
    _write_latex(args.out_tex, algo_cols, rows)
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
