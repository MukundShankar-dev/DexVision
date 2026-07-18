"""Summarize saved Level 2 demonstrations by skill and task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.dataset_summary import (
    DatasetSummaryError,
    default_summary_paths,
    save_dataset_summary,
    summarize_demo_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize saved demonstrations, schema versions, relabel results, "
            "and quality-filter results without modifying raw episodes."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset root containing episode directories.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <dataset>/reports/summaries/dataset_summary.json.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <dataset>/reports/summaries/dataset_summary.csv.",
    )
    return parser


def run_summary(args: argparse.Namespace) -> int:
    """Create and save the requested dataset summary."""

    default_json, default_csv = default_summary_paths(args.dataset)
    json_path = args.json_output or default_json
    csv_path = args.csv_output or default_csv
    print("DexVision Level 2 dataset summary")
    print(f"Dataset: {args.dataset}")
    print(f"JSON output: {json_path}")
    print(f"CSV output: {csv_path}")
    print("Raw episodes: immutable")

    report = summarize_demo_dataset(args.dataset)
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    saved_json, saved_csv = save_dataset_summary(
        report,
        json_path=json_path,
        csv_path=csv_path,
    )
    print(
        "Summary complete: "
        f"groups={report.num_groups}, episodes={report.num_episodes}"
    )
    for group in report.groups:
        success_rate = (
            "n/a" if group.success_rate is None else f"{group.success_rate:.3f}"
        )
        print(
            f"  {group.skill_name}/{group.task_id}: "
            f"episodes={group.num_episodes}, success_rate={success_rate}, "
            f"quality_pass={group.quality_pass_count}, "
            f"quality_fail={group.quality_fail_count}, "
            f"relabel_disagreements={group.relabel_disagreement_count}"
        )
    print(f"Saved JSON: {saved_json}")
    print(f"Saved CSV: {saved_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_summary(args)
    except (DatasetSummaryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
