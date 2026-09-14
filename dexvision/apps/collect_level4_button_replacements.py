"""Collect only the eight frozen Level 4.8 button replacements."""

import argparse
import subprocess
import sys
from pathlib import Path

from dexvision.logging.button_amendment import collect_button_replacements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/demos/level4"))
    args = parser.parse_args(argv)
    print(f"DexVision Level 4.8 button amendment: {args.plan}", flush=True)
    print(f"Append-only dataset: {args.dataset_dir}", flush=True)
    try:
        receipt = collect_button_replacements(args.plan, args.dataset_dir)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Qualified {len(receipt['episodes'])} replacements; rerun the complete dataset audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
