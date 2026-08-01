"""
FCM official fold-1 atama: fuzzy/imkpp/nogs/none/wnmf50/m1.5.

  python experiments/run_fuzzy_official_f1_assign.py --fast
  python experiments/run_fuzzy_official_f1_assign.py --fast --k 10 --algo HA_AVOAHGS LIT_GWO B_AVOA
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import (
    ASSIGN_ROOT_FAST,
    FCM_M,
    FAST_ALGOS,
    FAST_K_LIST,
    FOLD,
    WNMF_DIM,
    assign_dir,
    build_assign_cmd,
    expected_suffix,
)

ALGOS = ["HA_AVOAHGS", "LIT_GWO", "B1_HHO"]
K_LIST = [4, 10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=None)
    ap.add_argument("--algo", nargs="+", default=None)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Hizli atama: epoch=40 pop=25 early-stop wcss-only (~5-10 dk/K/algo)",
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if args.fast and args.algo is None:
        algos = FAST_ALGOS
    else:
        algos = args.algo if args.algo else ALGOS
    if args.fast and args.k is None:
        k_list = FAST_K_LIST
    else:
        k_list = args.k if args.k else K_LIST

    print("FCM OFFICIAL FOLD 1")
    print(f"  algos: {algos}")
    print(f"  K: {k_list}")
    print(f"  prune: {args.prune}  (default: no-prune)")
    print(f"  fast: {args.fast}")
    if args.fast:
        print("  fast cfg: epoch=40 pop=25 early-stop wcss-only")
    print(f"  gray sheep: off  preprocess: none  init: mkpp")
    print(f"  wnmf_dim={WNMF_DIM}  fcm_m={FCM_M}")
    print(f"  train: u{FOLD}.base  eval-split=official")
    for k in k_list:
        print(f"  K={k} suffix: {expected_suffix(k, prune=args.prune, fast=args.fast)}")
        if args.fast:
            print(f"    root: {ASSIGN_ROOT_FAST}")

    cmd = build_assign_cmd(
        algos=algos,
        k_list=k_list,
        prune=args.prune,
        jobs=args.jobs,
        skip_existing=args.skip_existing,
        fast=args.fast,
    )
    print("\n" + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        sys.exit(rc)

    print("\nAtama durumu:")
    for k in k_list:
        for algo in algos:
            p = assign_dir(algo, k, prune=args.prune, fast=args.fast)
            ok = (p / "assignments.npy").is_file()
            mem = (p / "memberships.npy").is_file()
            print(f"  {'OK' if ok else 'MISSING'}  {p.name}  memberships={mem}")


if __name__ == "__main__":
    main()
