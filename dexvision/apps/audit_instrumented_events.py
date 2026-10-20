"""Audit saved 5.7 pick/drop and place/release events without rerunning control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/level5/perception_event_audit_v1.yaml"),
    )
    args = parser.parse_args(argv)
    print(f"DexVision 5.7 saved-state event audit: {args.evaluation_dir}", flush=True)
    try:
        from dexvision.evaluation.instrumented_events import audit_instrumented_events

        result = audit_instrumented_events(
            args.evaluation_dir, args.output_dir, args.config
        )
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}. Use the dexvision Conda environment.", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
