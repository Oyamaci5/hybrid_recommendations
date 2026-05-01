"""
Compare baseline evaluation summaries in one table.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_for_compare(algorithm: str, summary: dict) -> List[dict]:
    rows = []
    for segment, vals in summary.items():
        rows.append(
            {
                "algorithm": algorithm,
                "segment": segment,
                "MAE": vals.get("MAE", ""),
                "RMSE": vals.get("RMSE", ""),
                "Precision@N": vals.get("Precision@N", ""),
                "Recall@N": vals.get("Recall@N", ""),
                "n_predictions": vals.get("n_predictions", ""),
            }
        )
    return rows


def run(input_map: Dict[str, str], output_path: str) -> str:
    rows = []
    for algo, summary_path in input_map.items():
        if not os.path.exists(summary_path):
            continue
        summary = _load_summary(summary_path)
        rows.extend(_flatten_for_compare(algo, summary))

    headers = [
        "algorithm",
        "segment",
        "MAE",
        "RMSE",
        "Precision@N",
        "Recall@N",
        "n_predictions",
    ]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[h]) for h in headers))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def _parse_pairs(pairs: list[str]) -> Dict[str, str]:
    parsed = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid input pair: {p}. Expected algo=path.")
        k, v = p.split("=", 1)
        parsed[k.strip()] = v.strip()
    return parsed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline summaries")
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Pairs: ALGO=path/to/evaluation_summary.json",
    )
    p.add_argument("--output", required=True, help="Output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mapping = _parse_pairs(args.input)
    out = run(mapping, args.output)
    print(out)
