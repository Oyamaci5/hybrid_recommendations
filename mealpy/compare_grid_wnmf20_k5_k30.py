"""WNMF=20 grid sonuçları için K=5 ve K=30 küme benzerlik karşılaştırması."""
from __future__ import annotations

import csv
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "mealpy" / "results" / "assignments" / "ml100k"
OUT_DIR = REPO / "results" / "grid"

ALGOS = [
    ("B0_KMEANS", False),
    ("B_AVOA", True),
    ("HA_AVOAHGS", True),
    ("IWO_HHO", True),
    ("B1_HHO", True),
]
SHORT = {
    "B0_KMEANS": "B0",
    "B_AVOA": "AVOA",
    "HA_AVOAHGS": "HA_AVOA",
    "IWO_HHO": "IWO",
    "B1_HHO": "B1",
}


def assign_dir(algo: str, kmref: bool, wnmf: int, k: int) -> Path:
    suffix = f"_euc_imkpp_nogs_none_wnmf{wnmf}_k{k}"
    if kmref:
        suffix += "_kmref"
    return BASE / f"{algo}{suffix}"


def load(algo: str, kmref: bool, wnmf: int, k: int):
    d = assign_dir(algo, kmref, wnmf, k)
    a = np.load(d / "assignments.npy").astype(np.int64)
    gpath = d / "gray_sheep_mask.npy"
    g = np.load(gpath).astype(bool) if gpath.is_file() else None
    bpath = d / "best_sol.npy"
    C = None
    if bpath.is_file():
        flat = np.load(bpath).astype(np.float64).ravel()
        if flat.size % k == 0:
            C = flat.reshape(k, flat.size // k)
    return a, g, C


def common_mask(ga, gb, n: int) -> np.ndarray:
    if ga is None and gb is None:
        return np.ones(n, dtype=bool)
    if ga is not None and gb is not None:
        return (~ga) & (~gb)
    g = ga if ga is not None else gb
    return ~g


def cent_dist(Ca: np.ndarray, Cb: np.ndarray) -> float:
    cost = cdist(Ca, Cb, metric="euclidean")
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def compare_k(k: int, wnmf: int = 20) -> tuple[list, list]:
    print("=" * 72)
    print(f"WNMF={wnmf}, K={k} — küme benzerlik karşılaştırması (referans: B0)")
    print("=" * 72)

    data = {}
    structure_rows = []
    for algo, kmref in ALGOS:
        a, g, C = load(algo, kmref, wnmf, k)
        data[algo] = (a, g, C)
        labels = a[~g] if g is not None else a
        sizes = np.bincount(labels, minlength=k)[:k]
        active_sizes = sizes[sizes > 0]
        structure_rows.append(
            {
                "wnmf": wnmf,
                "k": k,
                "algo": SHORT[algo],
                "active": int((sizes > 0).sum()),
                "size_min": int(active_sizes.min()),
                "size_max": int(active_sizes.max()),
                "size_mean": round(float(active_sizes.mean()), 1),
            }
        )
        print(
            f"  {SHORT[algo]:<8} aktif={structure_rows[-1]['active']}/{k}, "
            f"boyut min={structure_rows[-1]['size_min']}, "
            f"max={structure_rows[-1]['size_max']}, "
            f"ort={structure_rows[-1]['size_mean']}"
        )
    print()

    names = [a for a, _ in ALGOS]
    ref = "B0_KMEANS"
    lr, gr, Cr = data[ref]

    ref_rows = []
    print(f"{'Algo':<10} {'Agree':>8} {'ARI':>8} {'NMI':>8} {'CentDist':>10}")
    print("-" * 48)
    for algo in names:
        la, ga, Ca = data[algo]
        mask = common_mask(gr, ga, len(lr))
        if algo == ref:
            print(f"{SHORT[algo]:<10} {'100.0%':>8} {'1.0000':>8} {'1.0000':>8} {'0.0000':>10}")
            ref_rows.append(
                {
                    "wnmf": wnmf,
                    "k": k,
                    "algo": SHORT[algo],
                    "agree": 1.0,
                    "ari": 1.0,
                    "nmi": 1.0,
                    "cent_dist": 0.0,
                }
            )
            continue
        agree = float((lr[mask] == la[mask]).mean())
        ari = float(adjusted_rand_score(lr[mask], la[mask]))
        nmi = float(
            normalized_mutual_info_score(lr[mask], la[mask], average_method="arithmetic")
        )
        cd = cent_dist(Cr, Ca) if Cr is not None and Ca is not None else float("nan")
        print(f"{SHORT[algo]:<10} {agree:>7.1%} {ari:>8.4f} {nmi:>8.4f} {cd:>10.4f}")
        ref_rows.append(
            {
                "wnmf": wnmf,
                "k": k,
                "algo": SHORT[algo],
                "agree": round(agree, 4),
                "ari": round(ari, 4),
                "nmi": round(nmi, 4),
                "cent_dist": round(cd, 4) if not np.isnan(cd) else "",
            }
        )
    print()

    pair_rows = []
    print("Çiftler arası ARI / NMI:")
    print(f"{'Çift':<20} {'Agree':>8} {'ARI':>8} {'NMI':>8} {'CentDist':>10}")
    print("-" * 58)
    for na, nb in combinations(names, 2):
        la, ga, Ca = data[na]
        lb, gb, Cb = data[nb]
        mask = common_mask(ga, gb, len(la))
        agree = float((la[mask] == lb[mask]).mean())
        ari = float(adjusted_rand_score(la[mask], lb[mask]))
        nmi = float(
            normalized_mutual_info_score(la[mask], lb[mask], average_method="arithmetic")
        )
        cd = cent_dist(Ca, Cb) if Ca is not None and Cb is not None else float("nan")
        pair = f"{SHORT[na]} vs {SHORT[nb]}"
        print(f"{pair:<20} {agree:>7.1%} {ari:>8.4f} {nmi:>8.4f} {cd:>10.4f}")
        pair_rows.append(
            {
                "wnmf": wnmf,
                "k": k,
                "pair": pair,
                "agree": round(agree, 4),
                "ari": round(ari, 4),
                "nmi": round(nmi, 4),
                "cent_dist": round(cd, 4) if not np.isnan(cd) else "",
            }
        )
    print()
    return structure_rows, ref_rows, pair_rows


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Kaydedildi: {path}")


def main() -> None:
    all_structure = []
    all_ref = []
    all_pairs = []
    for k in (5, 30):
        structure, ref, pairs = compare_k(k, wnmf=20)
        all_structure.extend(structure)
        all_ref.extend(ref)
        all_pairs.extend(pairs)

    write_csv(
        OUT_DIR / "cluster_compare_wnmf20_structure.csv",
        all_structure,
        ["wnmf", "k", "algo", "active", "size_min", "size_max", "size_mean"],
    )
    write_csv(
        OUT_DIR / "cluster_compare_wnmf20_vs_b0.csv",
        all_ref,
        ["wnmf", "k", "algo", "agree", "ari", "nmi", "cent_dist"],
    )
    write_csv(
        OUT_DIR / "cluster_compare_wnmf20_pairs.csv",
        all_pairs,
        ["wnmf", "k", "pair", "agree", "ari", "nmi", "cent_dist"],
    )


if __name__ == "__main__":
    main()
