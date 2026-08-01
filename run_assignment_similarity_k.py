"""
Küme ataması benzerliği: no-kmref vs kmref (aynı wnmf20 trainonly protokol).
K = 6, 14, 30 — ARI / NMI / label agreement.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "mealpy"))

from compare_cluster_structure import (  # noqa: E402
    _common_mask,
    _load_centroids,
    _load_labels_and_gray,
    _resolve_path,
    mean_centroid_distance,
)

DEFAULT_ALGOS = [
    "B0_KMEANS",
    "B1_HHO",
    "B_AVOA",
    "HA_AVOAHGS",
    "IWO_HHO",
]

ASSIGN_ROOT = os.path.join(REPO, "mealpy", "results", "assignments", "ml100k")


def suffix_for_k(k: int, wnmf_dim: int = 20, *, kmref: bool = False) -> str:
    base = f"_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf{wnmf_dim}_k{k}"
    return base + ("_kmref" if kmref else "")


def _compare_labels(
    la: np.ndarray,
    ga: np.ndarray | None,
    lb: np.ndarray,
    gb: np.ndarray | None,
) -> dict:
    mask = _common_mask(ga, gb, len(la))
    n_ok = int(mask.sum())
    if n_ok < 2:
        return {"n_users": n_ok, "label_agreement": np.nan, "ari": np.nan, "nmi": np.nan}
    return {
        "n_users": n_ok,
        "label_agreement": float((la[mask] == lb[mask]).mean()),
        "ari": float(adjusted_rand_score(la[mask], lb[mask])),
        "nmi": float(
            normalized_mutual_info_score(la[mask], lb[mask], average_method="arithmetic")
        ),
    }


def _load_algo(
    algo: str,
    k: int,
    wnmf_dim: int,
    kmref: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    suf = suffix_for_k(k, wnmf_dim, kmref=kmref)
    fp = _resolve_path(ASSIGN_ROOT, algo, suf)
    if fp is None:
        return None
    a, g = _load_labels_and_gray(fp)
    C = _load_centroids(fp, k)
    return a, g, C


def analyze_k(
    k: int,
    algos: list[str],
    wnmf_dim: int,
    *,
    kmref: bool,
    reference: str = "B0_KMEANS",
    ref_kmref: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """ref_kmref: referans B0 yüklerken kmref kullan (B0'da genelde yok → False)."""
    tag = "kmref" if kmref else "no_kmref"
    available: list[str] = []
    loaded: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    centroids: dict[str, np.ndarray] = {}

    for algo in algos:
        out = _load_algo(algo, k, wnmf_dim, kmref)
        if out is None:
            continue
        a, g, C = out
        loaded[algo] = (a, g)
        if C is not None:
            centroids[algo] = C
        available.append(algo)

    pair_rows = []
    for na, nb in combinations(available, 2):
        la, ga = loaded[na]
        lb, gb = loaded[nb]
        m = _compare_labels(la, ga, lb, gb)
        cd = np.nan
        if na in centroids and nb in centroids:
            cd = mean_centroid_distance(centroids[na], centroids[nb])
        pair_rows.append({
            "K": k, "variant": tag, "algo_a": na, "algo_b": nb, **m,
            "centroid_l2_mean": cd,
        })

    ref_rows = []
    use_ref_kmref = kmref if ref_kmref is None else ref_kmref
    ref_out = _load_algo(reference, k, wnmf_dim, use_ref_kmref)
    if ref_out is not None and reference in loaded:
        lr, gr, _ = ref_out
        for algo in available:
            la, ga = loaded[algo]
            if algo == reference:
                ref_rows.append({
                    "K": k, "variant": tag, "algo": algo,
                    "reference": f"{reference}({('kmref' if use_ref_kmref else 'no_kmref')})",
                    "label_agreement": 1.0, "ari": 1.0, "nmi": 1.0,
                    "centroid_l2_mean": 0.0,
                })
                continue
            m = _compare_labels(lr, gr, la, ga)
            cd = np.nan
            if reference in centroids and algo in centroids:
                cd = mean_centroid_distance(centroids[reference], centroids[algo])
            ref_rows.append({
                "K": k, "variant": tag, "algo": algo,
                "reference": f"{reference}({('kmref' if use_ref_kmref else 'no_kmref')})",
                **m, "centroid_l2_mean": cd,
            })
    elif ref_out is not None:
        lr, gr, Cref = ref_out
        for algo in available:
            if algo == reference:
                continue
            la, ga = loaded[algo]
            m = _compare_labels(lr, gr, la, ga)
            cd = np.nan
            if Cref is not None and algo in centroids:
                cd = mean_centroid_distance(Cref, centroids[algo])
            ref_rows.append({
                "K": k, "variant": tag, "algo": algo,
                "reference": f"{reference}({('kmref' if use_ref_kmref else 'no_kmref')})",
                **m, "centroid_l2_mean": cd,
            })

    return pd.DataFrame(pair_rows), pd.DataFrame(ref_rows), available


def compare_kmref_toggle(
    k: int,
    algos: list[str],
    wnmf_dim: int,
) -> pd.DataFrame:
    """Aynı algo: no_kmref atama vs kmref atama."""
    rows = []
    for algo in algos:
        if algo == "B0_KMEANS":
            continue
        a0 = _load_algo(algo, k, wnmf_dim, kmref=False)
        a1 = _load_algo(algo, k, wnmf_dim, kmref=True)
        if a0 is None or a1 is None:
            continue
        la, ga, C0 = a0
        lb, gb, C1 = a1
        m = _compare_labels(la, ga, lb, gb)
        cd = np.nan
        if C0 is not None and C1 is not None:
            cd = mean_centroid_distance(C0, C1)
        rows.append({"K": k, "algo": algo, **m, "centroid_l2_mean": cd})
    return pd.DataFrame(rows)


def _print_block(k: int, tag: str, pairs: pd.DataFrame, ref: pd.DataFrame, avail: list[str]) -> None:
    print(f"\n{'=' * 60}\nK = {k}  [{tag}]  (n={len(avail)} algo)\n{'=' * 60}")
    if not avail:
        print("  Atama klasörü yok — bu K için kmref trainonly wnmf20 üretilmemiş.")
        return
    print(f"  Mevcut: {', '.join(avail)}")
    if len(ref):
        print("\nReferansa göre:")
        print(ref.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(pairs):
        print(
            f"\nMeta çift ort.: ARI={pairs['ari'].mean():.4f}  "
            f"Agree={pairs['label_agreement'].mean():.1%}  "
            f"NMI={pairs['nmi'].mean():.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k-list", type=int, nargs="+", default=[6, 14, 30])
    p.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--wnmf-dim", type=int, default=20)
    p.add_argument("--csv-prefix", default="results/assignment_similarity")
    p.add_argument(
        "--mode",
        choices=("both", "no_kmref", "kmref"),
        default="both",
        help="both: no_kmref + kmref + kmref vs no_kmref karşılaştırması",
    )
    args = p.parse_args()

    print("Protokol: none + wnmf20 + nogs + trainonly fold1 (kmref meta-only, B0 kmref yok)")
    print(f"Algolar: {', '.join(args.algos)}\n")

    all_pairs, all_ref, all_toggle = [], [], []

    if args.mode in ("both", "no_kmref"):
        print("\n" + "#" * 60 + "\n# NO-KMREF\n" + "#" * 60)
        for k in args.k_list:
            pairs, ref, avail = analyze_k(
                k, args.algos, args.wnmf_dim, kmref=False, ref_kmref=False,
            )
            _print_block(k, "no_kmref", pairs, ref, avail)
            pairs["variant"] = "no_kmref"
            ref["variant"] = "no_kmref"
            all_pairs.append(pairs)
            all_ref.append(ref)

    if args.mode in ("both", "kmref"):
        print("\n" + "#" * 60 + "\n# KMREF (meta)\n" + "#" * 60)
        meta_algos = [a for a in args.algos if a != "B0_KMEANS"]
        for k in args.k_list:
            pairs, ref, avail = analyze_k(
                k, meta_algos, args.wnmf_dim, kmref=True, ref_kmref=False,
            )
            _print_block(k, "kmref", pairs, ref, avail)
            if len(ref):
                print("  (Referans: B0 no-kmref — B0 için kmref klasörü yok)")
            pairs["variant"] = "kmref"
            ref["variant"] = "kmref"
            all_pairs.append(pairs)
            all_ref.append(ref)

    if args.mode == "both":
        print("\n" + "#" * 60 + "\n# AYNI ALGO: no_kmref vs kmref\n" + "#" * 60)
        for k in args.k_list:
            tdf = compare_kmref_toggle(k, args.algos, args.wnmf_dim)
            if tdf.empty:
                print(f"\nK={k}: karşılaştırılacak çift yok")
                continue
            print(f"\nK={k}")
            print(tdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            all_toggle.append(tdf)

    prefix = args.csv_prefix
    if not os.path.isabs(prefix):
        prefix = os.path.join(REPO, prefix)
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(f"{prefix}_pairs.csv", index=False)
    if all_ref:
        ref_df = pd.concat(all_ref, ignore_index=True)
        ref_df.to_csv(f"{prefix}_vs_ref.csv", index=False)
        print(f"\n{'=' * 60}\nB0 referansına göre ARI pivot (variant x K)")
        sub = ref_df[ref_df["algo"] != "B0_KMEANS"]
        if len(sub):
            print(
                sub.pivot_table(index=["variant", "algo"], columns="K", values="ari")
                .to_string(float_format=lambda x: f"{x:.4f}")
            )
    if all_toggle:
        pd.concat(all_toggle, ignore_index=True).to_csv(
            f"{prefix}_kmref_vs_nokmref.csv", index=False,
        )
        print(f"\nkmref vs no_kmref pivot (ARI):")
        tdf = pd.concat(all_toggle, ignore_index=True)
        print(
            tdf.pivot_table(index="algo", columns="K", values="ari")
            .to_string(float_format=lambda x: f"{x:.4f}")
        )

    print(f"\nCSV önek: {prefix}_*.csv")


if __name__ == "__main__":
    main()
