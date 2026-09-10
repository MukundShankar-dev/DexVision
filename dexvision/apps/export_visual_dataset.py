"""Export Level 4.7 fixed-camera visual supervision from saved episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.visual_stream import export_visual_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level4_visual_dataset.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/demos/level4"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/visual/level4"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("DexVision Level 4.7 fixed-camera visual export", flush=True)
    try:
        report = export_visual_dataset(config_path=args.config, dataset_dir=args.dataset_dir,
                                       output_dir=args.output_dir,
                                       progress=lambda message: print(message, flush=True))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Use the dexvision environment and an available offscreen OpenGL context.", file=sys.stderr)
        return 2
    print(f"Frames: {report['frame_count']}; payload: {report['payload_bytes']:,} bytes")
    print(f"Automated checks: {'PASS' if report['automated_passed'] else 'FAIL'}")
    print(f"Contact sheets and report: {args.output_dir}")
    print("Manual contact-sheet confirmation is required; Level 4.7 remains incomplete.")
    return 0 if report["automated_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
