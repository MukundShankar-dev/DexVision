"""CLI for read-only Level 4 pilot and core-collection coverage reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.evaluation.dataset_coverage import (
    DEFAULT_REPORT_NAME,
    DatasetCoverageError,
    save_coverage_report,
    summarize_level4_coverage,
)
from dexvision.logging.level4_collection import (
    DEFAULT_LEVEL4_CONFIG,
    DEFAULT_PILOT_DATASET_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Level 4 pilot coverage without modifying episodes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_LEVEL4_CONFIG,
        help=f"Frozen Level 4 dataset config. Defaults to {DEFAULT_LEVEL4_CONFIG}.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_PILOT_DATASET_DIR,
        help=f"Pilot dataset root. Defaults to {DEFAULT_PILOT_DATASET_DIR}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON report path. Defaults to <dataset-dir>/reports/"
            f"{DEFAULT_REPORT_NAME}."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return exit status 1 when the pilot or manual replay gate is incomplete.",
    )
    parser.add_argument(
        "--require-level4-4-complete",
        action="store_true",
        help="Return exit status 1 unless the Level 4.4 core haul is complete.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.dataset_dir / "reports" / DEFAULT_REPORT_NAME
    try:
        report = summarize_level4_coverage(
            config_path=args.config,
            dataset_dir=args.dataset_dir,
        )
        save_coverage_report(report, output)
    except (DatasetCoverageError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("DexVision Level 4 pilot coverage summary")
    print(f"Dataset: {args.dataset_dir}")
    print(
        "Episodes: "
        f"attempts={report['attempt_episode_count']}, "
        f"expert_accepted={report['expert_accepted_episode_count']}, "
        f"ordinary_failures={report['ordinary_failure_episode_count']}"
    )
    for group, counts in report["episode_counts_by_group"].items():
        print(
            f"  {group}: {counts['accepted']}/{counts['minimum']} "
            f"({'PASS' if counts['passed'] else 'INCOMPLETE'})"
        )
    matrix = report["coverage_matrix"]
    print(
        "Final matrix: "
        f"{matrix['minimum_episode_total']} required episodes across "
        f"{matrix['cell_count']} cells, planning envelope "
        f"{matrix['required_envelope'][0]}-{matrix['required_envelope'][1]}"
    )
    print("Required source mix:")
    for source, counts in report["source_mix"]["sources"].items():
        print(f"  {source}: {counts['observed']}/{counts['minimum']}")
    print(
        "Sessions: "
        f"{report['genuine_session_requirement']['observed']}/"
        f"{report['genuine_session_requirement']['minimum']}"
    )
    print(f"Dial decision: {report['optional_dial_decision']}")
    print(f"Payload handling: {report['storage']['payload_handling']}")
    print(f"Pilot status: {report['pilot_status']}")
    core = report["level4_4_core_collection"]
    print(
        "Level 4.4 core haul: "
        f"{core['accepted_episode_count']}/{core['required_accepted_episodes']} "
        f"accepted, cells={core['coverage_matrix']['complete_cell_count']}/"
        f"{core['coverage_matrix']['cell_count']}, status={core['status']}"
    )
    print(f"Report: {output}")
    if args.require_level4_4_complete and not core["checkpoint_complete"]:
        return 1
    if args.require_complete and not report["checkpoint_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
