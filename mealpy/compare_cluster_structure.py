"""
Aynı veri önkoşuluyla (aynı kullanıcı sırası, aynı suffix) assignment klasörleri arasında
küme yapısı farklarını özetler: aktif küme, boş küme, denge, çiftler arası ARI/NMI.

Örnek (aynı deney ekiyle tüm algoritmalar):
  python mealpy/compare_cluster_structure.py \\
    --dataset-root mealpy/results/assignments/ml100k \\
    --suffix _pruneu5_i10_zscore_euc_nogs_none_wnmf25_k30 \\
    --algos B0_KMEANS B2_HGS HA_AVOAHGS H4_MFO+HHO LIT_PSO LIT_GWO LIT_GOA

Örnek (referansa göre ARI — ilk algoritma referans):
  python mealpy/compare_cluster_structure.py \\
    --dataset-root mealpy/results/assignments/ml100k \\
    --suffix _pruneu5_i10_zscore_euc_nogs_none_wnmf25_k30 \\
    --algos B0_KMEANS LIT_GOA --reference B0_KMEANS
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _gini_positive_sizes(sizes: np.ndarray) -> float:
    """Aktif küme boyutları üzerinde Gini (0=dengeli, 1=çok dengesiz)."""
    x = np.asarray(sizes, dtype=np.float64)
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = int(x.shape[0])
    s = float(x.sum())
    if s <= 0:
        return float("nan")
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(((np.sum((2.0 * idx - n - 1.0) * x)) / (n * s)))


def _parse_k_expected(folder_name: str, fallback_k: int) -> int:
    m = re.search(r"_k(\d+)$", folder_name)
    if m:
        return int(m.group(1))
    return fallback_k


def _resolve_path(base: str, algo: str, suffix: Optional[str]) -> Optional[str]:
    if suffix is not None:
        d = os.path.join(base, f"{algo}{suffix}")
        fp = os.path.join(d, "assignments.npy")
        return fp if os.path.isfile(fp) else None
    pattern = os.path.join(base, f"{algo}_*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    best_fp: Optional[str] = None
    best_t = -1.0
    for d in dirs:
        fp = os.path.join(d, "assignments.npy")
        if os.path.isfile(fp):
            t = os.path.getmtime(fp)
            if t > best_t:
                best_t = t
                best_fp = fp
    return best_fp


def _load_labels_and_gray(fp: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    d = os.path.dirname(fp)
    a = np.load(fp).astype(np.int64, copy=False)
    gpath = os.path.join(d, "gray_sheep_mask.npy")
    g = np.load(gpath).astype(bool, copy=False) if os.path.isfile(gpath) else None
    return a, g


def _structure_stats(
    labels: np.ndarray, k_expected: int
) -> Dict[str, float]:
    k_cap = max(k_expected, int(labels.max()) + 1)
    sizes = np.bincount(labels, minlength=k_cap)[:k_expected]
    if sizes.size < k_expected:
        sizes = np.pad(sizes, (0, k_expected - sizes.size))
    active = sizes[sizes > 0]
    n_active = int((sizes > 0).sum())
    n_empty = int((sizes == 0).sum())
    if active.size == 0:
        return {
            "n_active": 0.0,
            "n_empty": float(n_empty),
            "size_min": float("nan"),
            "size_max": float("nan"),
            "size_mean": float("nan"),
            "size_std": float("nan"),
            "size_cv": float("nan"),
            "gini_sizes": float("nan"),
        }
    mean_s = float(active.mean())
    std_s = float(active.std(ddof=0)) if active.size > 1 else 0.0
    cv = float(std_s / mean_s) if mean_s > 1e-9 else 0.0
    return {
        "n_active": float(n_active),
        "n_empty": float(n_empty),
        "size_min": float(active.min()),
        "size_max": float(active.max()),
        "size_mean": mean_s,
        "size_std": std_s,
        "size_cv": cv,
        "gini_sizes": _gini_positive_sizes(active),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Assignment klasörleri arasında küme yapısı ve ARI/NMI karşılaştırması."
    )
    p.add_argument(
        "--dataset-root",
        default="mealpy/results/assignments/ml100k",
        help="ml100k / ml1m altındaki algoritma klasörlerinin kökü",
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=[
            "B0_KMEANS",
            "B2_HGS",
            "HA_AVOAHGS",
            "H4_MFO+HHO",
            "LIT_PSO",
            "LIT_GWO",
            "LIT_GOA",
        ],
    )
    p.add_argument(
        "--suffix",
        default=None,
        help="Klasör eki: örn _pruneu5_i10_zscore_euc_nogs_none_wnmf25_k30",
    )
    p.add_argument(
        "--reference",
        default=None,
        help="ARI/NMI özetinde referans algoritma (varsayılan: --algos içindeki ilki)",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="Yapı tablosunu bu CSV dosyasına da yaz",
    )
    args = p.parse_args()

    base = os.path.normpath(args.dataset_root)
    ref_name = args.reference or (args.algos[0] if args.algos else None)

    paths: Dict[str, str] = {}
    folder_meta: Dict[str, Tuple[str, int]] = {}
    for algo in args.algos:
        fp = _resolve_path(base, algo, args.suffix)
        if fp is None:
            print(f"[{algo}] assignments.npy bulunamadı (base={base})")
            continue
        paths[algo] = fp
        folder = os.path.basename(os.path.dirname(fp))
        a, _ = _load_labels_and_gray(fp)
        k_fb = int(a.max()) + 1
        folder_meta[algo] = (folder, _parse_k_expected(folder, k_fb))

    if len(paths) < 1:
        print("Yüklenecek assignment yok.")
        return

    # Ortak K_expected: aynı suffix ise hepsi aynı olmalı; yoksa max al
    k_expected = max(meta[1] for meta in folder_meta.values())
    print(f"Kök: {base}")
    print(f"Beklenen K (klasör adından veya max+1): {k_expected}")
    print()

    rows = []
    loaded: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    for algo, fp in paths.items():
        a, g = _load_labels_and_gray(fp)
        loaded[algo] = (a, g)
        folder, k_i = folder_meta[algo]
        if k_i != k_expected:
            print(f"  Not: {algo} klasörü _k={k_i}, tablo için K={k_expected} kullanılıyor.")

        st_all = _structure_stats(a, k_expected)
        if g is not None:
            aw = a[~g]
            st_w = _structure_stats(aw, k_expected)
        else:
            st_w = st_all

        rows.append(
            {
                "algo": algo,
                "folder": folder,
                "n_users": len(a),
                "K_expected": k_expected,
                "active_all": int(st_all["n_active"]),
                "empty_all": int(st_all["n_empty"]),
                "active_white": int(st_w["n_active"]),
                "empty_white": int(st_w["n_empty"]),
                "white_min": st_w["size_min"],
                "white_max": st_w["size_max"],
                "white_mean": st_w["size_mean"],
                "white_cv": st_w["size_cv"],
                "gini_white": st_w["gini_sizes"],
                "gray_n": int(g.sum()) if g is not None else 0,
            }
        )

    hdr = (
        f"{'Algo':<14} {'Aktif':>6} {'Boş':>5} "
        f"{'Aktif*':>6} {'Boş*':>5} {'min':>5} {'max':>5} "
        f"{'ort':>7} {'CV':>6} {'Gini':>6} {'GS':>5}"
    )
    print("Küme yapısı (* = gray sheep hariç, tipik raporla uyumlu):")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['algo']:<14} {r['active_all']:>6d} {r['empty_all']:>5d} "
            f"{r['active_white']:>6d} {r['empty_white']:>5d} "
            f"{int(r['white_min']):>5d} {int(r['white_max']):>5d} "
            f"{r['white_mean']:>7.1f} {r['white_cv']:>6.3f} {r['gini_white']:>6.3f} "
            f"{r['gray_n']:>5d}"
        )
    print()
    print(
        "Aktif/Boş: bincount ile K_expected diliminde. "
        "CV düşük = boyutlar birbirine yakın. Gini düşük = daha dengeli."
    )
    print()

    # Pairwise ARI / NMI (ortak non-gray)
    names = list(paths.keys())
    if len(names) >= 2:
        print("Çiftler — ortak non-gray kullanıcılar üzerinde ARI / NMI:")
        print(f"{'Çift':<36} {'n_ortak':>8} {'ARI':>8} {'NMI':>8}")
        print("-" * 64)
        for na, nb in combinations(names, 2):
            la, ga = loaded[na]
            lb, gb = loaded[nb]
            if ga is None and gb is None:
                mask = np.ones(len(la), dtype=bool)
            elif ga is not None and gb is not None:
                mask = (~ga) & (~gb)
            else:
                g = ga if ga is not None else gb
                mask = ~g
            n_ok = int(mask.sum())
            if n_ok < 2:
                print(f"{na + ' vs ' + nb:<36} {n_ok:>8d} {'—':>8} {'—':>8}")
                continue
            ari = adjusted_rand_score(la[mask], lb[mask])
            nmi = normalized_mutual_info_score(
                la[mask], lb[mask], average_method="arithmetic"
            )
            print(f"{na + ' vs ' + nb:<36} {n_ok:>8d} {ari:>8.4f} {nmi:>8.4f}")
        print("  ARI ~0 rastgele; 1 ayni bolusum (yeniden numaralandirmaya dayanikli).")
        print("  NMI 0..1, yüksek = daha uyumlu bilgi paylaşımı.")
        print()

    if ref_name and ref_name in loaded:
        print(f"Referansa göre ({ref_name}) — ARI / NMI:")
        print(f"{'Algo':<14} {'ARI':>8} {'NMI':>8}")
        print("-" * 32)
        lr, gr = loaded[ref_name]
        for algo in names:
            if algo == ref_name:
                print(f"{algo:<14} {'1.0000':>8} {'1.0000':>8}")
                continue
            la, ga = loaded[algo]
            if gr is not None and ga is not None:
                mask = (~gr) & (~ga)
            elif gr is not None:
                mask = ~gr
            elif ga is not None:
                mask = ~ga
            else:
                mask = np.ones(len(lr), dtype=bool)
            ari = adjusted_rand_score(lr[mask], la[mask])
            nmi = normalized_mutual_info_score(
                lr[mask], la[mask], average_method="arithmetic"
            )
            print(f"{algo:<14} {ari:>8.4f} {nmi:>8.4f}")
        print()

    if args.csv and rows:
        out = os.path.normpath(args.csv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        keys = list(rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"CSV yazıldı: {out}")


if __name__ == "__main__":
    main()
