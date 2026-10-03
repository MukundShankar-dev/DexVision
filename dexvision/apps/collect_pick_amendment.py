"""Freeze, collect and package the owner-authorized unassisted pick dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'collect', 'release', 'verify'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'))
    parser.add_argument('--release-dir', type=Path, default=Path('datasets'))
    args = parser.parse_args(argv)
    print(f'DexVision 5.3A additive pick data: {args.command}', flush=True)
    try:
        from dexvision.logging.pick_amendment import collect_data_split, freeze_data_plan, publish_data
        if args.command == 'verify':
            from dexvision.learning.pick_data import audit_amendment
            result = audit_amendment(args.config, args.output_dir)
        elif args.command == 'freeze':
            result = freeze_data_plan(args.config, args.output_dir)
        elif args.command == 'collect':
            if args.split is None:
                raise ValueError('Collection requires --split')
            result = collect_data_split(args.config, args.output_dir, args.split)
        else:
            result = publish_data(args.config, args.output_dir, args.release_dir)
        print(json.dumps(result, indent=2))
        if args.command == 'verify' and not result['passed']:
            return 1
    except (ImportError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision Conda environment.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
