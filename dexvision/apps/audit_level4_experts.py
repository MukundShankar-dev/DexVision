"""Audit repeated Level 4 scripted experts by replaying saved episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.evaluation.level4_expert_audit import (
    Level4ExpertAuditError,
    audit_expert_architecture,
    save_expert_audit_report,
)
from dexvision.logging.level4_collection import DEFAULT_LEVEL4_CONFIG
from dexvision.logging.replay_demo import DemoReplayError
from dexvision.sim.mujoco_env import MujocoError
from dexvision.sim.workcell import DEFAULT_WORKCELL_CONFIG, WorkcellError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and recompute a repeated cross-skill Level 4 expert audit."
        )
    )
    parser.add_argument(
        "--episode",
        type=Path,
        action="append",
        required=True,
        help="Episode directory to audit; repeat for every success or failure.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_LEVEL4_CONFIG)
    parser.add_argument("--workcell-config", type=Path, default=DEFAULT_WORKCELL_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Generated JSON audit report; episode directories remain immutable.",
    )
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Return status 1 unless the repeated expert architecture qualifies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_expert_architecture(
            args.episode,
            config_path=args.config,
            workcell_config=args.workcell_config,
        )
        save_expert_audit_report(report, args.output)
    except (
        DemoReplayError,
        Level4ExpertAuditError,
        MujocoError,
        OSError,
        ValueError,
        WorkcellError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("DexVision Level 4 expert/replay qualification audit")
    print(f"Episodes: {report['episode_count']}")
    print(f"Accepted: {report['accepted_episode_count']}")
    print(f"Ordinary failures retained: {report['ordinary_failure_count']}")
    for skill in report["required_source_skills"]:
        observed = report["accepted_source_skill_counts"].get(skill, 0)
        print(f"  {skill}: {observed}/{report['minimum_repeats_per_source_skill']}")
    print(f"Qualified: {report['qualified']}")
    print(f"Report: {args.output}")
    if args.require_qualified and not report["qualified"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
