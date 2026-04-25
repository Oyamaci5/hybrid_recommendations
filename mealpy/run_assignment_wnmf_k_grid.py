"""
WNMF boyutu (latent dim) ve küme sayısı K için assignment üretimini toplu çalıştırır.

generate_assignments.py tek seferde tek --wnmf-features aldığından, her WNMF boyutu için
dış döngüyü bu script uygular.

Örnek:

  python run_assignment_wnmf_k_grid.py --dataset both --lof --zscore --cluster-metric fuzzy \\
    --wnmf 10 20 40 --k 20 30 70 --jobs 4 --skip-existing

--skip-existing: İlk --algo klasöründe (yoksa B0_KMEANS) assignments.npy varsa o K atlanır.
--dataset both ise ml100k ve ml1m için ikisinde de dosya varsa atlanır.

Ek argümanlar generate_assignments.py'ye aynen iletilir.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Sequence, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_SCRIPT = os.path.join(BASE_DIR, "generate_assignments.py")

# generate_assignments.py ile aynı (klasör _k{K} eki için)
K_100K_DEFAULT = 30
K_1M_DEFAULT = 70
DEFAULT_MARKER_LABEL = "B0_KMEANS"


def _parse_child_flags(extra: Sequence[str]) -> dict:
    """generate_assignments ile uyumlu path için gerekli bayraklar (varsayılanlar dahil)."""
    o = {
        "dataset": "both",
        "lof": False,
        "zscore": False,
        "min_user_ratings": 5,
        "min_item_ratings": 10,
        "wnmf_init": "inmed",
        "inmed_trim_low": 5.0,
        "inmed_trim_high": 95.0,
        "pca_variance": None,  # float or None
        "cluster_metric": "auto",
        "algo": None,  # list of str or None = tüm algoritmalar
    }
    it = list(extra)
    i = 0
    while i < len(it):
        a = it[i]
        if a in ("--dataset",) and i + 1 < len(it):
            o["dataset"] = it[i + 1]
            i += 2
            continue
        if a == "--lof":
            o["lof"] = True
            i += 1
            continue
        if a == "--zscore":
            o["zscore"] = True
            i += 1
            continue
        if a in ("-j", "--jobs") and i + 1 < len(it):
            i += 2
            continue
        if a == "--min-user-ratings" and i + 1 < len(it):
            o["min_user_ratings"] = int(it[i + 1])
            i += 2
            continue
        if a == "--min-item-ratings" and i + 1 < len(it):
            o["min_item_ratings"] = int(it[i + 1])
            i += 2
            continue
        if a == "--wnmf-init" and i + 1 < len(it):
            o["wnmf_init"] = it[i + 1]
            i += 2
            continue
        if a == "--inmed-trim-low" and i + 1 < len(it):
            o["inmed_trim_low"] = float(it[i + 1])
            i += 2
            continue
        if a == "--inmed-trim-high" and i + 1 < len(it):
            o["inmed_trim_high"] = float(it[i + 1])
            i += 2
            continue
        if a in ("--pca",) and i + 1 < len(it):
            o["pca_variance"] = float(it[i + 1])
            i += 2
            continue
        if a == "--cluster-metric" and i + 1 < len(it):
            o["cluster_metric"] = it[i + 1]
            i += 2
            continue
        if a == "--algo":
            algs: List[str] = []
            i += 1
            while i < len(it) and not it[i].startswith("-"):
                algs.append(it[i])
                i += 1
            o["algo"] = algs if algs else None
            continue
        if a in ("--data-100k", "--data-1m", "--contamination", "--n-neighbors",
                 "--save-wnmf-u"):
            i += 2 if i + 1 < len(it) else 1
            continue
        if a in ("--last-only",):
            i += 1
            continue
        i += 1
    return o


def _resolve_cluster_metric(raw: str, wnmf_dim: int) -> str:
    if raw == "auto":
        return "euclidean" if wnmf_dim else "pearson"
    return raw


def _build_out_suffix(wnmf_dim: int, fl: dict) -> str:
    """generate_assignments.py out_suffix ile aynı (satır 1329–1344)."""
    m = _resolve_cluster_metric(fl["cluster_metric"], wnmf_dim)
    prune = f"_pruneu{fl['min_user_ratings']}_i{fl['min_item_ratings']}"
    z = "_zscore" if fl["zscore"] else ""
    pca = fl["pca_variance"]
    pca_s = (
        f"_pca{int(round(pca * 100))}pct" if pca is not None else ""
    )
    wnmf_s = (
        f"_wnmf{wnmf_dim}_{fl['wnmf_init']}_trim{fl['inmed_trim_low']:g}_"
        f"{fl['inmed_trim_high']:g}"
    )
    metric_map = {"euclidean": "_euc", "fuzzy": "_fuzzy"}
    met_s = metric_map.get(m, "")
    return prune + z + pca_s + wnmf_s + met_s


def _k_folder_suffix(k: int, default_k: int) -> str:
    return "" if k == default_k else f"_k{k}"


def _out_root(fl: dict) -> str:
    name = "assignments_lof" if fl["lof"] else "assignments"
    return os.path.join(BASE_DIR, "results", name)


def _marker_label(fl: dict) -> str:
    algs = fl.get("algo")
    if algs:
        return algs[0]
    return DEFAULT_MARKER_LABEL


def _marker_ok_for_dataset(
    out_root: str,
    dataset_sub: str,  # 'ml100k' or 'ml1m'
    label: str,
    k: int,
    default_k: int,
    out_suffix: str,
) -> bool:
    k_s = _k_folder_suffix(k, default_k)
    folder = f"{label}{k_s}{out_suffix}"
    path = os.path.join(out_root, dataset_sub, folder, "assignments.npy")
    return os.path.isfile(path)


def _filter_k_skip_existing(
    wnmf_dim: int, k_list: List[int], extra: List[str]
) -> Tuple[List[int], List[int]]:
    """
    Döner: (çalıştırılacak K'ler, atlanan K'ler).
    dataset=both iken: her iki veri seti için de marker varsa atla.
    """
    fl = _parse_child_flags(extra)
    out_suffix = _build_out_suffix(wnmf_dim, fl)
    out_root = _out_root(fl)
    label = _marker_label(fl)
    run: List[int] = []
    skipped: List[int] = []

    ds = fl["dataset"]
    checks: List[Tuple[str, int]] = []
    if ds in ("100k", "both"):
        checks.append(("ml100k", K_100K_DEFAULT))
    if ds in ("1m", "both"):
        checks.append(("ml1m", K_1M_DEFAULT))

    for k in k_list:
        ok_all = True
        for sub, def_k in checks:
            if not _marker_ok_for_dataset(
                out_root, sub, label, k, def_k, out_suffix
            ):
                ok_all = False
                break
        if ok_all:
            skipped.append(k)
        else:
            run.append(k)
    return run, skipped


def main() -> int:
    p = argparse.ArgumentParser(
        description="WNMF x K ızgarası: generate_assignments.py'yi tekrarlı çağırır."
    )
    p.add_argument(
        "--wnmf",
        type=int,
        nargs="+",
        required=True,
        metavar="DIM",
        help="WNMF latent boyutları (her biri için ayrı matris + K listesi).",
    )
    p.add_argument(
        "--k",
        type=int,
        nargs="+",
        required=True,
        metavar="K",
        help="Küme sayıları (ör. 20 30 70).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Sadece komutları yazdır, çalıştırma.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="assignments.npy (referans algoritma klasörü) mevcutsa o K'yi atla.",
    )
    ns, extra = p.parse_known_args()

    wnmf_dims = [int(x) for x in ns.wnmf]
    k_list = [int(x) for x in ns.k]
    if any(x <= 0 for x in wnmf_dims + k_list):
        print("Hata: --wnmf ve --k değerleri pozitif olmalı.", file=sys.stderr)
        return 2

    for w in wnmf_dims:
        to_run, skipped = (k_list, [])
        if ns.skip_existing:
            to_run, skipped = _filter_k_skip_existing(w, k_list, extra)
            if skipped:
                print(
                    f"[skip-existing] WNMF={w} atlandı: K={skipped} "
                    f"(zaten assignments.npy mevcut)"
                )
        if not to_run:
            print(
                f"[skip-existing] WNMF={w} için tüm K değerleri mevcut, çağrı yok."
            )
            continue

        k_str = " ".join(str(x) for x in to_run)
        cmd = [
            sys.executable,
            GEN_SCRIPT,
            f"--wnmf-features",
            str(w),
            "--k",
            *map(str, to_run),
        ] + list(extra)
        print("=" * 60)
        print(f"--- WNMF={w}  |  K={k_str} ---")
        if ns.skip_existing and skipped:
            print(f"    (atlanan K: {skipped})")
        print(" ".join(cmd))
        print("=" * 60)
        if not ns.dry_run:
            r = subprocess.run(cmd, cwd=BASE_DIR)
            if r.returncode != 0:
                print(
                    f"Hata: generate_assignments wnmf={w} için çıkış kodu {r.returncode}",
                    file=sys.stderr,
                )
                return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
