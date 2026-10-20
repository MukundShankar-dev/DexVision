"""Continue only unstarted 5.7 requests; preserve consumed control attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"DexVision 5.7 preserve-and-continue recovery: {args.config}", flush=True)
    try:
        from dexvision.evaluation.instrumented_recovery import recover_instrumented

        result = recover_instrumented(args.config)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(
            f"ERROR: {exc}. Preserve evidence and use the dexvision environment.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
