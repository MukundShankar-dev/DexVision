"""Recompute success labels for saved Level 2 task demonstrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.relabel_success import (
    DEFAULT_REPORT_NAME,
    SuccessRelabelError,
    relabel_demo_dataset,
    save_relabel_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute supported Level 2 task success from saved metric inputs."
        )
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
        help=(
            f"Output JSON report. Defaults to <dataset>/{DEFAULT_REPORT_NAME}; "
            "raw episode metadata and arrays are never rewritten."
        ),
    )
    return parser


def run_relabeling(args: argparse.Namespace) -> int:
    """Relabel the requested dataset and save its audit report."""

    report_path = args.report or args.dataset / DEFAULT_REPORT_NAME
    print("DexVision Level 2 task success relabeling")
    print(f"Dataset: {args.dataset}")
    print(f"Report: {report_path}")
    print("Raw episodes: immutable")

    report = relabel_demo_dataset(args.dataset)
    saved_path = save_relabel_report(report, report_path)
    print(
        "Relabeling complete: "
        f"episodes={report.episode_count}, "
        f"recomputed_success={report.recomputed_success_count}, "
        f"label_disagreements={report.label_disagreement_count}"
    )
    print(f"Saved report: {saved_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_relabeling(args)
    except (SuccessRelabelError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
