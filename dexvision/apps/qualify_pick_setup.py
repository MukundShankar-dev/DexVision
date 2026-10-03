"""Evaluate a corrected scripted approach protocol with frozen pick policy weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'validate', 'confirm', 'audit', 'review'))
    parser.add_argument('--config', type=Path, default=Path('configs/level5/pick_setup_v1.yaml'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reset-dir', type=Path)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--release-root', type=Path)
    parser.add_argument('--seed', type=int, default=5001)
    args = parser.parse_args(argv)
    print(f'DexVision 5.3A fixed-weight setup study: {args.command}', flush=True)
    try:
        from dexvision.evaluation.pick_setup import audit_setup, evaluate_setup, freeze_setup
        if args.command != 'review' and args.output_dir is None:
            raise ValueError('This command requires --output-dir')
        if args.command == 'freeze':
            result = freeze_setup(args.config, args.output_dir, args.release_root)
        else:
            if args.reset_dir is None:
                raise ValueError('This command requires --reset-dir')
            if args.command == 'review':
                from dexvision.evaluation.pick_setup_review import review_setup
                result = review_setup(args.config, args.reset_dir, args.seed)
            elif args.command == 'audit':
                if args.report_dir is None:
                    raise ValueError('Audit requires --report-dir')
                result = audit_setup(args.config, args.reset_dir, args.report_dir, args.output_dir, args.seed)
            else:
                result = evaluate_setup(args.config, args.reset_dir, args.output_dir, seed=args.seed,
                    split='validation' if args.command == 'validate' else 'confirmation')
        print(json.dumps(result, indent=2))
    except (ImportError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision Conda environment.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
