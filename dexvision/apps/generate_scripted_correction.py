"""Generate or audit frozen Level 4.6 scripted failures and corrections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.evaluation.correction_summary import (
    DEFAULT_REPORT_NAME,
    save_correction_summary,
    summarize_corrections,
)
from dexvision.logging.corrective_demos import (
    CorrectiveDemoError,
    append_level4_6_session,
    build_level4_6_plan,
    generate_level4_6_dataset,
    generate_scripted_correction,
    generate_scripted_failure,
)
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import DEFAULT_LEVEL4_CONFIG


DEFAULT_DATASET_DIR = Path("data/demos/level4")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append deterministic Level 4.6 failure/correction evidence. "
            "Existing episodes are never overwritten."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_LEVEL4_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--source-rollout",
        type=Path,
        help=(
            "Triggering scripted rollout. A Level 4.6 failure automatically creates "
            "its matching correction; a nominal rollout requires --cell-id."
        ),
    )
    parser.add_argument("--cell-id", help="Frozen failure/correction cell for one episode.")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Append-only output directory for single-episode generation.",
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Append every missing assignment in the frozen 12-cell/120-episode plan.",
    )
    parser.add_argument(
        "--max-assignments",
        type=int,
        default=0,
        help="Optional smoke-test cap for --generate-all; zero means all missing slots.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Summary output path; defaults to dataset-dir/reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        print("DexVision Level 4.6 scripted failures and corrections")
        print(f"Config: {args.config}")
        print(f"Dataset: {args.dataset_dir}")
        if args.generate_all:
            generated = generate_level4_6_dataset(
                config_path=args.config,
                dataset_dir=args.dataset_dir,
                max_assignments=args.max_assignments,
            )
            print(f"Generated missing episodes: {len(generated)}")
            for path in generated[-3:]:
                print(f"  {path}")
        else:
            output = _generate_one(args)
            print(f"Generated: {output}")
        report = summarize_corrections(
            config_path=args.config,
            dataset_dir=args.dataset_dir,
        )
        report_path = args.report or args.dataset_dir / "reports" / DEFAULT_REPORT_NAME
        save_correction_summary(report, report_path)
        print(
            "Coverage: "
            f"{report['episode_count']}/{report['required_episode_count']} episodes, "
            f"{report['complete_cell_count']}/{report['required_cell_count']} cells"
        )
        print(f"Automated requirements: {'PASS' if report['automated_requirements_passed'] else 'INCOMPLETE'}")
        print(f"Report: {report_path}")
        return 0
    except (CorrectiveDemoError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _generate_one(args: argparse.Namespace) -> Path:
    assert args.source_rollout is not None
    source = load_logged_demo(args.source_rollout)
    plan = build_level4_6_plan(args.config)
    payload = source.metadata.get("level4_6")
    if isinstance(payload, dict):
        failure_class = source.metadata.get("failure_class")
        candidates = [
            item
            for item in plan
            if item.is_correction
            and item.failure_class == failure_class
            and item.repetition == int(payload.get("repetition", args.repetition))
        ]
    else:
        candidates = [
            item
            for item in plan
            if item.coverage_cell_id == args.cell_id
            and item.repetition == args.repetition
        ]
    if len(candidates) != 1:
        raise CorrectiveDemoError(
            "could not resolve one frozen assignment; supply a valid --cell-id and repetition."
        )
    assignment = candidates[0]
    output = args.output_dir or (
        args.dataset_dir / assignment.session_id / "episode_000001"
    )
    append_level4_6_session(
        dataset_dir=args.dataset_dir,
        assignment=assignment,
        source_dir=args.source_rollout,
    )
    if assignment.is_correction:
        return generate_scripted_correction(
            source_rollout=args.source_rollout,
            output_dir=output,
            assignment=assignment,
            dataset_dir=args.dataset_dir,
        )
    return generate_scripted_failure(
        source_rollout=args.source_rollout,
        output_dir=output,
        assignment=assignment,
        dataset_dir=args.dataset_dir,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_assignments < 0:
        raise ValueError("--max-assignments must be non-negative.")
    if args.repetition < 1 or args.repetition > 10:
        raise ValueError("--repetition must be between 1 and 10.")
    if args.generate_all and args.source_rollout is not None:
        raise ValueError("--generate-all and --source-rollout are mutually exclusive.")
    if not args.generate_all and args.source_rollout is None:
        raise ValueError("provide --source-rollout or --generate-all.")
    if args.cell_id is not None and args.source_rollout is None:
        raise ValueError("--cell-id requires --source-rollout.")


if __name__ == "__main__":
    raise SystemExit(main())
