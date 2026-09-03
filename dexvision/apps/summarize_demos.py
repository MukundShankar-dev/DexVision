"""Summarize saved Level 2 demonstrations by skill and task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.dataset_summary import (
    DEFAULT_BUTTON_PRESS_CONFIG,
    DEFAULT_REACH_TOUCH_CONFIG,
    DatasetSummaryError,
    default_summary_paths,
    load_button_press_dataset_config,
    load_reach_touch_dataset_config,
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
    parser.add_argument(
        "--reach-touch-config",
        type=Path,
        default=DEFAULT_REACH_TOUCH_CONFIG,
        help=(
            "Versioned reach-touch train/held-out split and readiness thresholds. "
            f"Defaults to {DEFAULT_REACH_TOUCH_CONFIG}."
        ),
    )
    parser.add_argument(
        "--button-press-config",
        type=Path,
        default=DEFAULT_BUTTON_PRESS_CONFIG,
        help=(
            "Versioned button-goal train/held-out split and readiness thresholds. "
            f"Defaults to {DEFAULT_BUTTON_PRESS_CONFIG}."
        ),
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
    print(f"Reach-touch readiness config: {args.reach_touch_config}")
    print(f"Button-press readiness config: {args.button_press_config}")
    print("Raw episodes: immutable")

    reach_touch_config = load_reach_touch_dataset_config(args.reach_touch_config)
    button_press_config = load_button_press_dataset_config(args.button_press_config)
    report = summarize_demo_dataset(
        args.dataset,
        reach_touch_config=reach_touch_config,
        button_press_config=button_press_config,
    )
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
            f"relabel_disagreements={group.relabel_disagreement_count}, "
            f"clean_success={group.clean_success_count}, "
            f"level3_ready={group.level3_ready}"
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
