"""CLI for the controlled Level 3.6 diagnostic matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CONFIG = Path("configs/level3_diagnostics.yaml")
DEFAULT_MODEL = Path("assets/mujoco/task_board_scene.xml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the Level 3.6 data, action-space, and goal-input "
            "diagnostic matrix."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser


def run(args: argparse.Namespace) -> int:
    from dexvision.evaluation.level3_diagnostics import (
        load_level3_diagnostics_config,
        run_level3_diagnostics,
    )

    config = load_level3_diagnostics_config(args.config)
    print("DexVision Level 3.6 data and action-space diagnostics")
    print(f"Config: {config.source_path} ({config.source_digest})")
    print(f"Dataset root: {config.dataset_root}")
    print(f"Output: {config.output_directory}")
    report = run_level3_diagnostics(config, model_path=args.model)
    print(f"Experiments: {len(report['summary_table'])}")
    print(f"JSON report: {config.output_directory / 'report.json'}")
    print(f"CSV table: {config.output_directory / 'summary.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
