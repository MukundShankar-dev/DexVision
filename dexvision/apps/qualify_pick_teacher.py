"""Qualify a versioned deterministic pick teacher without object-pose assistance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    print('DexVision 5.3A unassisted pick teacher feasibility', flush=True)
    try:
        from dexvision.evaluation.pick_teacher import qualify_teacher
        result = qualify_teacher(args.config, args.output_dir)
        print(json.dumps(result, indent=2))
    except (ImportError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision Conda environment.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
