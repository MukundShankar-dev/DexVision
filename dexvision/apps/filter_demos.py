"""Generate a read-only quality report for saved Level 2 pilot demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.quality_filters import (
    DEFAULT_REPORT_NAME,
    QualityFilterError,
    QualityThresholds,
    filter_demo_dataset,
    save_quality_report,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = QualityThresholds()
    parser = argparse.ArgumentParser(
        description="Flag low-quality saved supported Level 2 pilot demonstrations."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Single-task dataset directory containing saved episode directories.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"Output JSON report. Defaults to <dataset>/{DEFAULT_REPORT_NAME}.",
    )
    parser.add_argument(
        "--min-mean-tracking-confidence",
        type=float,
        default=defaults.min_mean_tracking_confidence,
    )
    parser.add_argument(
        "--min-mean-feature-confidence",
        type=float,
        default=defaults.min_mean_feature_confidence,
    )
    parser.add_argument(
        "--max-missing-frame-fraction",
        type=float,
        default=defaults.max_missing_frame_fraction,
    )
    parser.add_argument(
        "--max-feature-jitter-p95",
        type=float,
        default=defaults.max_feature_jitter_p95,
    )
    parser.add_argument(
        "--max-action-jerk-p95",
        type=float,
        default=defaults.max_action_jerk_p95,
    )
    parser.add_argument(
        "--max-joint-limit-hit-fraction",
        type=float,
        default=defaults.max_joint_limit_hit_fraction,
    )
    parser.add_argument(
        "--max-workspace-limit-hit-fraction",
        type=float,
        default=defaults.max_workspace_limit_hit_fraction,
    )
    parser.add_argument(
        "--limit-margin-fraction",
        type=float,
        default=defaults.limit_margin_fraction,
    )
    return parser


def run_filtering(args: argparse.Namespace) -> int:
    """Evaluate the requested dataset and save its quality report."""

    report_path = args.report or args.dataset / DEFAULT_REPORT_NAME
    thresholds = QualityThresholds(
        min_mean_tracking_confidence=args.min_mean_tracking_confidence,
        min_mean_feature_confidence=args.min_mean_feature_confidence,
        max_missing_frame_fraction=args.max_missing_frame_fraction,
        max_feature_jitter_p95=args.max_feature_jitter_p95,
        max_action_jerk_p95=args.max_action_jerk_p95,
        max_joint_limit_hit_fraction=args.max_joint_limit_hit_fraction,
        max_workspace_limit_hit_fraction=args.max_workspace_limit_hit_fraction,
        limit_margin_fraction=args.limit_margin_fraction,
    )
    print("DexVision Level 2 pilot quality filtering")
    print(f"Dataset: {args.dataset}")
    print(f"Report: {report_path}")
    print(f"Thresholds: {thresholds.version}")
    print("Raw episodes: immutable")

    report = filter_demo_dataset(args.dataset, thresholds=thresholds)
    saved_path = save_quality_report(report, report_path)
    print(
        "Quality filtering complete: "
        f"episodes={report.episode_count}, "
        f"passed={report.pass_count}, "
        f"failed={report.fail_count}, "
        f"groups={len(report.groups)}"
    )
    print(f"Saved report: {saved_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_filtering(args)
    except (QualityFilterError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
