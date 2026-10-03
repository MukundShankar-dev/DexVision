"""Freeze and evaluate fresh state pick confirmation after validation selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze-confirmation', 'confirm', 'audit', 'review'))
    parser.add_argument('--config', type=Path, default=Path('configs/level5/pick_policy_v3.yaml'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--release-root', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--confirmation-dir', type=Path)
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args(argv)
    print(f'DexVision 5.3A pick repair: {args.command}', flush=True)
    try:
        from dexvision.evaluation.pick_repair import (
            audit_pick_confirmation, confirm_pick, freeze_pick_confirmation, review_pick_confirmation,
        )
        if args.command == 'freeze-confirmation':
            result = freeze_pick_confirmation(args.config, args.output_dir, args.release_root)
        else:
            if args.run_dir is None or args.confirmation_dir is None:
                raise ValueError('Confirmation requires --run-dir and --confirmation-dir')
            if args.command in {'audit', 'review'}:
                if args.report_dir is None:
                    raise ValueError('Saved-state audit/review requires --report-dir')
                result = (audit_pick_confirmation(args.config, args.run_dir, args.confirmation_dir,
                    args.report_dir, args.output_dir) if args.command == 'audit' else
                    review_pick_confirmation(args.config, args.run_dir, args.confirmation_dir, args.report_dir))
            else:
                result = confirm_pick(args.config, args.run_dir, args.confirmation_dir, args.output_dir)
        print(json.dumps(result, indent=2))
    except (ImportError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision Conda environment.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
