"""Evaluate frozen state policies or the 5.7 inferred-control boundary."""

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
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument(
        "--select-candidates",
        type=Path,
        help="Training run whose candidates need frozen validation rollouts",
    )
    parser.add_argument("--compare-runs", nargs="+", type=Path)
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        default=Path("outputs/level3/reach_bc_v2/policy_best.pt"),
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--all-required-skills", action="store_true")
    parser.add_argument(
        "--review-dir",
        type=Path,
        help="Saved 5.7 rejection evidence; requires --viewer",
    )
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Freeze the 5.7 instrumented entry matrices without running policies",
    )
    parser.add_argument("--scenario-set", choices=("manual_stratified",))
    args = parser.parse_args(argv)
    print(f"DexVision skill evaluation: {args.config} ({args.track})", flush=True)
    try:
        import yaml

        version = yaml.safe_load(args.config.read_text()).get("version")
        if version == "level5/perception-rollouts-v2":
            from dexvision.evaluation.instrumented_rollouts import (
                run_instrumented_evaluation,
            )

            if (
                not args.all_required_skills
                or args.track != "perception"
                or args.checkpoint
                or args.select_candidates
                or args.compare_runs
                or args.release_root
                or args.scenario_set
            ):
                raise ValueError(
                    "Instrumented 5.7 requires --all-required-skills --track perception"
                )
            if args.freeze_only:
                from dexvision.evaluation.instrumented_rollouts import (
                    freeze_instrumented_matrices,
                )

                if args.viewer or args.output_dir or args.review_dir:
                    raise ValueError(
                        "--freeze-only uses the locked protocol's case directory"
                    )
                report = freeze_instrumented_matrices(args.config)
            elif args.viewer:
                from dexvision.evaluation.instrumented_review import review_instrumented

                if args.review_dir is None or args.output_dir is not None:
                    raise ValueError("Use --viewer --review-dir for saved 5.7 rollouts")
                report = review_instrumented(args.review_dir)
            else:
                if args.output_dir is None or args.review_dir:
                    raise ValueError("Instrumented 5.7 requires a new --output-dir")
                report = run_instrumented_evaluation(
                    args.config, args.output_dir, split=args.split or "test"
                )
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.freeze_only:
            raise ValueError("--freeze-only requires the instrumented 5.7 protocol")
        if version == "level5/perception-rollouts-v1":
            from dexvision.evaluation.perception_rollouts import (
                run_perception_evaluation,
            )

            if (
                not args.all_required_skills
                or args.track != "perception"
                or args.checkpoint
                or args.select_candidates
                or args.compare_runs
                or args.scenario_set
                or args.release_root
                or args.split not in (None, "validation")
            ):
                raise ValueError(
                    "5.7 requires --all-required-skills --track perception and its locked protocol"
                )
            report = run_perception_evaluation(
                args.config,
                output_dir=args.output_dir,
                viewer=args.viewer,
                review_dir=args.review_dir,
            )
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.all_required_skills or args.review_dir:
            raise ValueError(
                "--all-required-skills/--review-dir require the 5.7 perception protocol"
            )
        args.split = args.split or "test"
        if version == "level5/push-policy-v8":
            from dexvision.evaluation.push_guarded import (
                run_distance_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for guarded push")
        elif version == "level5/push-policy-v7":
            from dexvision.evaluation.push_predicates import (
                run_distance_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for predicate push")
        elif version == "level5/push-policy-v6":
            from dexvision.evaluation.push_latched import (
                run_distance_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for latched push")
        elif version == "level5/push-policy-v5":
            from dexvision.evaluation.push_canonical import (
                run_distance_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for canonical push")
        elif version == "level5/push-policy-v4":
            from dexvision.evaluation.push_distance import (
                run_distance_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError(
                    "Reach comparison is not supported for push distance repair"
                )
        elif version == "level5/push-policy-v3":
            from dexvision.evaluation.push_settled import (
                run_settled_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for settled push")
        elif version == "level5/push-policy-v2":
            from dexvision.evaluation.push_repair import (
                run_push_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for push repair")
        elif version in {"level5/push-policy-v1", "level5/press-policy-v1"}:
            from dexvision.evaluation.contact_policy import (
                run_contact_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError(
                    "Reach comparison is not supported for contact recipes"
                )
        elif version == "level5/place-policy-v1":
            from dexvision.evaluation.place_policy import (
                run_place_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for place recipes")
        elif version in {
            "level5/pick-policy-v1",
            "level5/pick-policy-v2",
            "level5/pick-policy-v3",
            "level5/pick-policy-v4",
            "level5/pick-policy-v5",
            "level5/pick-policy-v6",
            "level5/pick-policy-v7",
        }:
            from dexvision.evaluation.pick_policy import (
                run_pick_evaluation as run_evaluation_command,
            )

            if args.compare_runs:
                raise ValueError("Reach comparison is not supported for pick recipes")
        else:
            from dexvision.evaluation.evaluate_skill import run_evaluation_command

        if args.compare_runs:
            from dexvision.evaluation.evaluate_skill import compare_reach_runs

            if (
                args.output_dir is None
                or args.checkpoint
                or args.select_candidates
                or args.viewer
                or args.track != "state"
            ):
                raise ValueError(
                    "Comparison requires --compare-runs and a new --output-dir"
                )
            report = compare_reach_runs(
                args.config, args.compare_runs, args.output_dir, args.legacy_checkpoint
            )
        else:
            options = vars(args).copy()
            options.pop("compare_runs")
            options.pop("legacy_checkpoint")
            options.pop("all_required_skills")
            options.pop("review_dir")
            options.pop("freeze_only")
            report = run_evaluation_command(**options)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(
            f"ERROR: {exc}. Use dexvision; macOS viewer requires mjpython.",
            file=sys.stderr,
        )
        return 2
    compact = {
        k: v
        for k, v in report.items()
        if k
        not in {
            "results",
            "reports",
            "input_digests",
            "training_history",
            "breakdowns",
            "runs",
            "legacy",
        }
    }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
