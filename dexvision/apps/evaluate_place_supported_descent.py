"""Validate, screen, select or confirm the frozen Level 5.4A V22 place policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'screen', 'select', 'confirm'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    print(f'DexVision place V22 {args.action}: {args.config}', flush=True)
    try:
        if args.action in ('validate', 'confirm'):
            if args.checkpoint is None or args.output_dir is None:
                raise ValueError('Validation and confirmation require --checkpoint and --output-dir')
            from dexvision.evaluation.place_supported_descent_policy import evaluate_supported_descent

            report = evaluate_supported_descent(args.config, args.checkpoint, args.output_dir,
                split='validation' if args.action == 'validate' else 'confirmation')
        else:
            if args.seed is None:
                raise ValueError('Selection screen and bundle require --seed')
            from dexvision.evaluation.place_supported_descent_selection import (
                screen_candidate, select_screened,
            )

            report = (screen_candidate(args.config, args.seed) if args.action == 'screen'
                      else select_screened(args.config, args.seed))
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision environment and verified V3 release.',
              file=sys.stderr)
        return 2
    compact = {key: value for key, value in report.items() if key != 'results'}
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
