"""Genel CLI giriş noktası (config + bileşen listesi veya smoke)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.runner import ExperimentRunner


def main() -> None:
    ap = argparse.ArgumentParser(description="rs_meta tabanlı deney iskeleti")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true", help="rastgele matris ile hızlı duman testi")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    runner = ExperimentRunner(config_path=args.config, overrides=args.override or None)
    if args.smoke:
        out = runner.run_dummy_smoke()
        print(out)
        return
    c = runner.components()
    print("Pipeline bileşenleri hazır:", list(c.keys()))


if __name__ == "__main__":
    main()
