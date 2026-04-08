"""
Assignment çıktı klasörlerinde özdeş dosyaları bulur.

Kullanım (repo kökünden):
  python mealpy/check_assignment_duplicates.py
  python mealpy/check_assignment_duplicates.py mealpy/results/assignments_lof/ml1m

Aynı assignments.npy veya best_sol.npy SHA256 özeti → farklı algoritma etiketleriyle
aynı sonuç: ya dosya kopyası / manuel taşıma, ya da (daha seyrek) aynı sayısal optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mtime_utc(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _collect(root: Path, name: str) -> dict[str, list[Path]]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        f = p / name
        if not f.is_file():
            continue
        by_hash[_sha256_file(f)].append(f)
    return by_hash


def _print_groups(title: str, by_hash: dict[str, list[Path]]) -> int:
    dups = [(h, paths) for h, paths in by_hash.items() if len(paths) > 1]
    if not dups:
        print(f"{title}: no duplicates.")
        return 0
    print(f"\n{title}: {len(dups)} duplicate group(s)")
    for h, paths in sorted(dups, key=lambda x: -len(x[1])):
        print(f"  sha256={h[:16]}... ({len(paths)} dosya)")
        for f in paths:
            print(f"    {f.parent.name}  |  {_mtime_utc(f)}")
    return len(dups)


def main() -> None:
    ap = argparse.ArgumentParser(description="Özdeş assignment / best_sol dosyalarını listele.")
    ap.add_argument(
        "root",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "results", "assignments_lof", "ml1m"),
        help="Dataset alt klasörü (ör. .../assignments_lof/ml1m)",
    )
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Klasör yok: {root.resolve()}")

    print(f"Root: {root.resolve()}")
    n_a = _print_groups("assignments.npy", _collect(root, "assignments.npy"))
    n_b = _print_groups("best_sol.npy", _collect(root, "best_sol.npy"))
    if n_a or n_b:
        print(
            "\nNot: Farkli yazim zamanlari + ayni ozet -> genelde klasor kopyasi veya"
            "\n     ayni cozumun tekrar kaydi. Paralel havuz hatasi olsaydi, genelde"
            "\n     farkli klasorlere tutarsiz veri yazilir; burada 3 dosya da birebir ayni."
        )


if __name__ == "__main__":
    main()
