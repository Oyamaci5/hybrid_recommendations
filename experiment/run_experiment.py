"""Genel CLI giriş noktası (config + pipeline_mode dispatch)."""

from __future__ import annotations

import argparse
import json
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

    cfg = runner.load()
    mode = (cfg.pipeline_mode or "cluster").strip().lower()

    if mode == "offline_assignments":
        out = runner.run_offline_assignments()
        print(json.dumps({"prediction_dir": out["prediction_dir"],
                          "evaluation_dir": out["evaluation_dir"],
                          "summary": out["summary"]}, indent=2, ensure_ascii=False))
        return

    # cluster / meta_mf_tune: bileşenleri kur, build_pipeline çalıştırılır
    c = runner.components()
    print("Pipeline bileşenleri hazır:", list(c.keys()))


if __name__ == "__main__":
    main()
