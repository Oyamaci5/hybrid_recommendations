"""wnmf60 + none (no WNMF) knn=40 eval özeti."""
from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "results" / "wnmf" / "ml100k"
KS = [5, 7, 10, 14, 21, 27, 30]


def parse_csv(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    cmd = lines[0]
    if "wnmf60" in cmd:
        feat = "wnmf60"
    elif "none_none20" in cmd:
        feat = "none"
    else:
        return []
    out = []
    for row in csv.DictReader(lines[1:]):
        if row.get("scenario") != "cluster_knn_baseline":
            continue
        if int(float(row.get("k_neighbors", 0))) != 40:
            continue
        out.append({
            "feature": feat,
            "k": int(row["assignment_k"]),
            "algo": row["algo_label"],
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "ndcg": float(row["ndcg_at_10"]),
            "prec": float(row["precision_at_10"]),
            "rec": float(row["recall_at_10"]),
            "file": str(path.relative_to(REPO)),
        })
    return out


def main() -> None:
    rows: list[dict] = []
    for k in KS:
        fold_dir = BASE / f"k{k}" / "fold1"
        if not fold_dir.is_dir():
            continue
        runs = sorted(
            fold_dir.glob("run*/wnmf_results_*_baselines.csv"),
            key=lambda p: int(p.parent.name[3:]),
            reverse=True,
        )
        seen: set[str] = set()
        for path in runs:
            cmd = path.read_text(encoding="utf-8").splitlines()[0]
            if f"--k {k} " not in cmd and f"--k {k}\"" not in cmd:
                continue
            for tag in ("wnmf60", "none_none20"):
                if tag in cmd and tag not in seen:
                    rows.extend(parse_csv(path))
                    seen.add(tag)
                    break

    rows.sort(key=lambda r: (r["feature"], r["k"], r["algo"]))
    out = REPO / "results" / "grid" / "wnmf60_none_knn40_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["feature", "k", "algo", "mae", "rmse", "ndcg", "prec", "rec", "file"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Rows: {len(rows)} -> {out.relative_to(REPO)}")
    for feat in ("wnmf60", "none"):
        sub = [r for r in rows if r["feature"] == feat]
        if not sub:
            print(f"\n{feat}: veri yok")
            continue
        best_ndcg = max(sub, key=lambda r: r["ndcg"])
        best_mae = min(sub, key=lambda r: r["mae"])
        print(f"\n{feat} | en iyi NDCG@10: {best_ndcg['ndcg']:.4f} "
              f"(K={best_ndcg['k']}, {best_ndcg['algo']})")
        print(f"{feat} | en iyi MAE: {best_mae['mae']:.4f} "
              f"(K={best_mae['k']}, {best_mae['algo']})")
        print("K\tAlgo\t\tMAE\tNDCG\tPrec\tRec")
        for k in KS:
            kr = [r for r in sub if r["k"] == k]
            if not kr:
                continue
            b = max(kr, key=lambda r: r["ndcg"])
            print(f"{k}\t{b['algo']:<12}\t{b['mae']:.4f}\t{b['ndcg']:.4f}\t"
                  f"{b['prec']:.4f}\t{b['rec']:.4f}")


if __name__ == "__main__":
    main()
