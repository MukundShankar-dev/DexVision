"""Review immutable 5.7 clips with a recorded-action HUD; never grant acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"DexVision 5.7 saved-result review: {args.evaluation_dir}", flush=True)
    try:
        from dexvision.evaluation.instrumented_review_overlay import (
            review_instrumented_results,
        )

        value = review_instrumented_results(args.evaluation_dir)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}. Use the dexvision Conda environment.", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
