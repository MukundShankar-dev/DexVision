"""Freeze/re-render the 5.6 pixel-center data amendment without test reuse."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p = commands.add_parser("collect")
    p.add_argument("--protocol", required=True)
    p.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(f"DexVision 5.6 pixel-center amendment: {args.command}", flush=True)
    try:
        from dexvision.logging.perception_raster import freeze_development, collect_dataset
        result = (freeze_development(args.config, args.output_dir) if args.command == "freeze" else
                  collect_dataset(args.protocol, args.output_dir))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"Render amendment failed: {exc}")
        return 1
    print(json.dumps({k: result[k] for k in ("case_count", "frame_count", "historical_pixel_collisions") if k in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
