"""Freeze, collect and package the 5.4A additive ordinary-physics place source."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'collect', 'verify', 'release'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'))
    parser.add_argument('--release-dir', type=Path, default=Path('datasets'))
    args = parser.parse_args(argv)
    print(f'DexVision 5.4A additive place data: {args.command}', flush=True)
    try:
        from dexvision.logging.place_amendment import (
            collect_place_data_split,
            freeze_place_data_plan,
            publish_place_data,
        )
        if args.command == 'freeze':
            result = freeze_place_data_plan(args.config, args.output_dir)
        elif args.command == 'collect':
            if args.split is None:
                raise ValueError('Collection requires --split')
            result = collect_place_data_split(args.config, args.output_dir, args.split)
        elif args.command == 'verify':
            from dexvision.evaluation.place_source_audit import audit_place_data
            result = audit_place_data(args.config, args.output_dir)
        else:
            result = publish_place_data(args.config, args.output_dir, args.release_dir)
        print(json.dumps(result, indent=2))
        if args.command == 'verify' and not result['passed']:
            return 1
    except (ImportError, KeyError, ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision Conda environment.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
