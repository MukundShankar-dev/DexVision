"""CLI for the frozen Level 4.5B validation-only scaling probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.evaluation.level4_scaling_probe import (
    Level4ScalingProbeError,
    run_level4_scaling_probe,
    save_scaling_probe_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen 4/8/16 Level 4.5B train/validation probe."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workcell-config", type=Path, default=Path("configs/workcell.yaml")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_level4_scaling_probe(
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            workcell_config=args.workcell_config,
        )
        output = save_scaling_probe_report(
            report, args.output_dir / "scaling_probe.json"
        )
    except (Level4ScalingProbeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("DexVision Level 4.5B scaling probe")
    for tranche in report["tranches"]:
        print(
            f"  {tranche['episodes_per_training_cell']}/cell: "
            f"validation={tranche['aggregate_validation_success']:.3f}, "
            f"worst_cell={tranche['worst_cell_validation_success']:.3f}, "
            f"safety={tranche['safety_violation_count']}, "
            f"invalid={tranche['invalid_action_count']}"
        )
    print(f"Decision: {report['status']}")
    print(f"Report: {output}")
    return 0 if report["dataset_sufficient"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
