"""Evaluate a frozen state-grounded reach or versioned pick policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--track", choices=("state", "perception"), default="state")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--select-candidates", type=Path, help="Training run whose candidates need frozen validation rollouts")
    parser.add_argument("--compare-runs", nargs="+", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path, default=Path("outputs/level3/reach_bc_v2/policy_best.pt"))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--scenario-set", choices=("manual_stratified",))
    args = parser.parse_args(argv)
    print(f"DexVision skill evaluation: {args.config} ({args.track})", flush=True)
    try:
        import yaml

        version = yaml.safe_load(args.config.read_text()).get("version")
        if version in {'level5/push-policy-v1', 'level5/press-policy-v1'}:
            from dexvision.evaluation.contact_policy import run_contact_evaluation as run_evaluation_command
            if args.compare_runs:
                raise ValueError('Reach comparison is not supported for contact recipes')
        elif version == 'level5/place-policy-v1':
            from dexvision.evaluation.place_policy import run_place_evaluation as run_evaluation_command
            if args.compare_runs:
                raise ValueError('Reach comparison is not supported for place recipes')
        elif version in {"level5/pick-policy-v1", "level5/pick-policy-v2", "level5/pick-policy-v3", "level5/pick-policy-v4", "level5/pick-policy-v5", "level5/pick-policy-v6", "level5/pick-policy-v7"}:
            from dexvision.evaluation.pick_policy import run_pick_evaluation as run_evaluation_command
            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for pick recipes")
        else:
            from dexvision.evaluation.evaluate_skill import run_evaluation_command

        if args.compare_runs:
            from dexvision.evaluation.evaluate_skill import compare_reach_runs

            if args.output_dir is None or args.checkpoint or args.select_candidates or args.viewer or args.track != "state":
                raise ValueError("Comparison requires --compare-runs and a new --output-dir")
            report = compare_reach_runs(args.config, args.compare_runs, args.output_dir, args.legacy_checkpoint)
        else:
            options = vars(args).copy()
            options.pop("compare_runs")
            options.pop("legacy_checkpoint")
            report = run_evaluation_command(**options)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}. Use dexvision; macOS viewer requires mjpython.", file=sys.stderr)
        return 2
    compact = {k: v for k, v in report.items() if k not in {"results", "reports", "input_digests", "training_history", "breakdowns", "runs", "legacy"}}
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
