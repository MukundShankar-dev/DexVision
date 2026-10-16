"""Restore byte-identical dataset archives from checksum-bound Git LFS parts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dexvision.logging.multipart_release import restore_multipart_releases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("datasets/level5-perception-lfs-transport-v1.json"))
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify all parts without creating missing original archives")
    args = parser.parse_args(argv)
    print(f"DexVision multipart release integrity: {args.manifest}", flush=True)
    try:
        report = restore_multipart_releases(args.manifest, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
