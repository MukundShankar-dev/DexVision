"""Print a balanced-random command for the next reach-touch recording."""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path

from dexvision.logging.collection_planner import (
    DEFAULT_REACH_TOUCH_DATASET,
    CollectionPlannerError,
    ReachTouchCollectionPlan,
    format_recording_command,
    plan_reach_touch_collection,
    recording_arguments,
)
from dexvision.logging.quality_filters import (
    QualityFilterError,
    evaluate_episode_quality,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Choose randomly among the reach-touch targets with the fewest clean "
            "successes and print the next non-overwriting recorder command."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_REACH_TOUCH_DATASET,
        help=f"Reach-touch raw dataset directory. Defaults to {DEFAULT_REACH_TOUCH_DATASET}.",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Episode date in YYYY-MM-DD form. Defaults to today.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic selector seed for testing or reproducibility.",
    )
    parser.add_argument(
        "--python-command",
        default="mjpython",
        help="Recorder Python command. Defaults to mjpython; use python on Windows.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Launch through a quality gate: accepted episodes move into raw/ and "
            "failed attempts move into rejected/."
        ),
    )
    return parser


def run_selector(args: argparse.Namespace) -> int:
    plan = plan_reach_touch_collection(
        args.dataset,
        collection_date=args.date,
        seed=args.seed,
    )
    raw_counts = ", ".join(
        f"{target}={count}" for target, count in plan.target_counts
    )
    clean_counts = ", ".join(
        f"{target}={count}" for target, count in plan.clean_target_counts
    )
    print("DexVision Level 2.7C reach-touch target selector")
    print(f"Current raw counts: {raw_counts}")
    print(f"Current clean counts: {clean_counts}")
    print(f"Selected target: {plan.target_site}")
    if not args.run:
        print(f"Next output: {plan.output_directory}")
        print("Run without an automatic quality gate:")
        print(format_recording_command(plan, python_command=args.python_command))
        print("No dataset files were created or modified.")
        return 0

    return _run_quality_gated_recording(
        plan,
        python_command=args.python_command,
    )


def _run_quality_gated_recording(
    plan: ReachTouchCollectionPlan,
    *,
    python_command: str,
) -> int:
    final_output = plan.output_directory
    staging_output = _staging_output(final_output)
    staged_plan = replace(plan, output_directory=staging_output)
    print(f"Accepted output: {final_output}")
    print(f"Temporary staging output: {staging_output}")
    print("Launching recorder with automatic quality gate:")
    print(format_recording_command(staged_plan, python_command=python_command))
    completed = subprocess.run(
        recording_arguments(staged_plan, python_command=python_command),
        check=False,
    )
    if completed.returncode != 0:
        if staging_output.exists():
            rejected = _move_to_rejected(staging_output, final_output)
            print(
                f"Recorder exited with status {completed.returncode}; "
                f"partial attempt preserved at: {rejected}",
                file=sys.stderr,
            )
        return completed.returncode
    if not staging_output.is_dir():
        raise CollectionPlannerError(
            f"Recorder completed without creating the staged episode: {staging_output}"
        )

    try:
        quality = evaluate_episode_quality(staging_output)
    except QualityFilterError as exc:
        rejected = _move_to_rejected(staging_output, final_output)
        raise CollectionPlannerError(
            f"Recorded attempt could not pass quality validation and was preserved "
            f"at {rejected}: {exc}"
        ) from exc

    if not quality.passed:
        rejected = _move_to_rejected(staging_output, final_output)
        failures = ", ".join(quality.failed_filters)
        print(f"REJECTED: {failures}", file=sys.stderr)
        print(f"Failed attempt preserved outside raw data: {rejected}", file=sys.stderr)
        print("Run this selector again to record a replacement.", file=sys.stderr)
        return 1

    final_output.parent.mkdir(parents=True, exist_ok=True)
    if final_output.exists():
        rejected = _move_to_rejected(staging_output, final_output)
        raise CollectionPlannerError(
            f"Refusing to overwrite existing raw episode {final_output}; "
            f"new clean attempt was preserved at {rejected}."
        )
    staging_output.replace(final_output)
    print(f"ACCEPTED: clean successful episode saved to {final_output}")
    return 0


def _staging_output(final_output: Path) -> Path:
    data_root = _data_root(final_output.parent)
    staging_root = data_root / "staging" / final_output.parent.name
    token = uuid.uuid4().hex[:8]
    return staging_root / f"{final_output.name}_{token}"


def _move_to_rejected(staging_output: Path, final_output: Path) -> Path:
    data_root = _data_root(final_output.parent)
    rejected_root = data_root / "rejected" / final_output.parent.name
    rejected_root.mkdir(parents=True, exist_ok=True)
    candidate = rejected_root / final_output.name
    suffix = 2
    while candidate.exists():
        candidate = rejected_root / f"{final_output.name}_failed_{suffix:02d}"
        suffix += 1
    staging_output.replace(candidate)
    return candidate


def _data_root(dataset: Path) -> Path:
    if dataset.parent.name == "raw":
        return dataset.parent.parent
    return dataset.parent


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_selector(args)
    except (CollectionPlannerError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
