"""Verify a Level 4 release and optionally restore into a new directory."""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

from dexvision.logging.dataset_release import retrieve_lfs_payload, verify_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=Path("datasets/level4-v1"))
    parser.add_argument("--archive", type=Path, help="Downloaded/local payload archive")
    parser.add_argument("--download-dir", type=Path,
                        help="New directory for retrieval through an empty Git LFS cache")
    parser.add_argument("--restore-dir", type=Path, help="New, nonexistent restoration directory")
    parser.add_argument("--require-ready", action="store_true",
                        help="Also require resolved publication and manual acceptance gates")
    args = parser.parse_args(argv)
    print(f"DexVision Level 4.9 release verification: {args.release_dir}", flush=True)
    try:
        if args.archive is not None and args.download_dir is not None:
            raise ValueError("Choose --archive or --download-dir, not both")
        archive = (retrieve_lfs_payload(args.release_dir, args.download_dir)
                   if args.download_dir is not None else args.archive)
        result = verify_release(args.release_dir, archive=archive,
                                restore_dir=args.restore_dir, require_ready=args.require_ready)
    except (OSError, ValueError, KeyError, TypeError, EOFError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    print("Integrity PASS; " + ("publication ready" if result["publication_ready"]
                              else "candidate only; publication gates remain pending"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
