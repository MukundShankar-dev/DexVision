"""Train the checksum-locked Level 5.4A V19 all-cell place source."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    print(f'DexVision place V19 train seed {args.seed}: {args.config}', flush=True)
    try:
        from dexvision.learning.place_full_source_policy import train_full_source

        result = train_full_source(args.config, seed=args.seed,
            output_dir=args.output_dir, dry_run=args.dry_run, resume=args.resume)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision environment and verified V3 release.',
              file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
