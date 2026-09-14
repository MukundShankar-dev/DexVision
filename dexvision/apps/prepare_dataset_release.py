"""Package one local immutable candidate from the passing Level 4.8 audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.dataset_release import prepare_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--audit-dir", default="outputs/level4/audit_v4_final")
    parser.add_argument("--release-dir", type=Path, default=Path("datasets/level4-v1"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-config", type=Path,
                        help="Owner-approved license and documented hosting configuration")
    args = parser.parse_args(argv)
    print(f"DexVision Level 4.9 local candidate: {args.release_dir}", flush=True)
    try:
        result = prepare_release(root=args.root, audit_relative=args.audit_dir,
                                 release_dir=args.release_dir, archive=args.archive,
                                 source_commit=args.source_commit,
                                 release_config=args.release_config)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Verified {result['file_count']} files; archive {result['archive']['size_bytes']} bytes")
    print("Pending gates: " + ", ".join(result["publication_blockers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
