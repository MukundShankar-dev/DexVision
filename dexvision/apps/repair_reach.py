"""Diagnose and repair reach using the versioned training/validation protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'screen', 'validate', 'freeze-confirmation', 'select-method', 'confirm', 'audit', 'review'))
    parser.add_argument('--config', type=Path, default=Path('configs/level5/reach_repair_v1.yaml'))
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--release-root', type=Path)
    parser.add_argument('--arm', choices=('physical_scale', 'physical_scale_hold', 'relative_goal_hold', 'goal_plane_hold', 'goal_plane_calibrated_hold', 'goal_plane_height_hold'))
    parser.add_argument('--seed', type=int, default=5001)
    parser.add_argument('--screen-dirs', type=Path, nargs='+')
    parser.add_argument('--confirmation-dir', type=Path)
    parser.add_argument('--decision-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    print(f'DexVision 5.2A reach repair: {args.command}', flush=True)
    try:
        from dexvision.learning.reach_repair import prepare_repair_cache, train_repair
        from dexvision.evaluation.reach_repair import evaluate_repair_run, freeze_confirmation, select_repair_method, confirm_repair

        if args.command in {'audit', 'review'}:
            from dexvision.evaluation.reach_repair_review import audit_repair, review_repair

            if args.run_dir is None or args.confirmation_dir is None:
                raise ValueError('Saved-state audit/review requires --run-dir and --confirmation-dir')
            result = (audit_repair(args.config, args.run_dir, args.confirmation_dir, args.output_dir)
                      if args.command == 'audit' else review_repair(args.config, args.run_dir, args.confirmation_dir))
        elif args.command == 'select-method':
            if not args.screen_dirs or args.confirmation_dir is None:
                raise ValueError('Method selection requires --screen-dirs and --confirmation-dir')
            result = select_repair_method(args.config, args.screen_dirs, args.output_dir, reset_dir=args.confirmation_dir)
        elif args.command == 'confirm':
            if any(v is None for v in (args.run_dir, args.confirmation_dir, args.decision_dir)):
                raise ValueError('Confirmation requires --run-dir, --confirmation-dir and --decision-dir')
            result = confirm_repair(args.config, args.run_dir, args.confirmation_dir, args.decision_dir, args.output_dir)
        elif args.command == 'freeze-confirmation':
            result = freeze_confirmation(args.config, args.output_dir, release_root=args.release_root)
        elif args.command == 'prepare':
            result = prepare_repair_cache(args.config, args.output_dir, args.release_root)
        elif args.command == 'train':
            if args.cache_dir is None or args.arm is None:
                raise ValueError('Training requires --cache-dir and --arm')
            result = train_repair(args.config, args.cache_dir, args.output_dir, arm=args.arm, seed=args.seed, resume=args.resume)
        else:
            if args.run_dir is None:
                raise ValueError('Evaluation requires --run-dir')
            result = evaluate_repair_run(args.config, args.run_dir, args.output_dir, screen=args.command == 'screen', release_root=args.release_root)
        print(json.dumps({k:v for k,v in result.items() if k not in {'reports', 'history', 'candidates', 'evidence', 'results', 'input_digests', 'execution_code_sha256'}}, indent=2))
    except (ImportError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision Conda environment.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
