"""Generate the frozen Level 4.5B append-only procedural expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dexvision.apps import record_demo
from dexvision.evaluation.level4_expert_audit import audit_scripted_episode
from dexvision.logging.level4_collection import (
    DEFAULT_LEVEL4_CONFIG,
    PilotReview,
    build_level4_procedural_expansion_plan,
    discover_pilot_episodes,
    save_pilot_review,
)
from dexvision.logging.session_manifest import (
    load_session_manifest,
    next_episode_directory,
)
from dexvision.sim.workcell import DEFAULT_WORKCELL_CONFIG


DEFAULT_DATASET_DIR = Path("data/demos/level4")
DEFAULT_SCALING_REPORT = Path("outputs/level4/scaling_probe_v1/scaling_probe.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append missing Level 4.5B scripted episodes in frozen plan order. "
            "Existing accepted episodes are never overwritten."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_LEVEL4_CONFIG)
    parser.add_argument("--workcell-config", type=Path, default=DEFAULT_WORKCELL_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
        help="Generate only these frozen splits. Defaults to train and validation.",
    )
    parser.add_argument(
        "--scaling-report",
        type=Path,
        default=DEFAULT_SCALING_REPORT,
        help="Frozen sufficient train/validation decision required before test generation.",
    )
    parser.add_argument(
        "--max-assignments",
        type=int,
        default=0,
        help="Optional positive smoke-test limit; zero processes every missing assignment.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=25,
        help="Print aggregate progress every N processed assignments.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_assignments < 0:
        print("ERROR: --max-assignments must be nonnegative.", file=sys.stderr)
        return 2
    if args.print_interval <= 0:
        print("ERROR: --print-interval must be positive.", file=sys.stderr)
        return 2
    splits = tuple(dict.fromkeys(args.splits))
    if "test" in splits:
        try:
            _require_sufficient_scaling_decision(
                args.scaling_report, config_path=args.config
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    plan = tuple(
        assignment
        for assignment in build_level4_procedural_expansion_plan(args.config)
        if assignment.split in splits
    )
    existing_episodes = tuple(discover_pilot_episodes(args.dataset_dir))
    accepted_episodes = tuple(
        episode
        for episode in existing_episodes
        if _procedural_repetition(episode.metadata) is not None
        and episode.expert_accepted
    )
    manifest_path = args.dataset_dir / "session_manifest.json"
    existing_session_ids = (
        {
            session.recording_session_id
            for session in load_session_manifest(manifest_path).sessions
        }
        if manifest_path.exists()
        else set()
    )
    missing = [
        assignment
        for assignment in plan
        if not any(
            episode.goal_condition_id == assignment.coverage_cell_id
            and _procedural_repetition(episode.metadata) == assignment.repetition
            and episode.session_id == assignment.session_id
            and episode.metadata.get("random_seed") == assignment.seed
            for episode in accepted_episodes
        )
    ]
    if args.max_assignments:
        missing = missing[: args.max_assignments]
    print("DexVision Level 4.5B procedural expansion")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Frozen assignments in selection: {len(plan)}")
    print(f"Missing assignments to generate: {len(missing)}")

    accepted = 0
    rejected = 0
    for index, assignment in enumerate(missing, start=1):
        episode_dir = next_episode_directory(
            args.dataset_dir,
            recording_session_id=assignment.session_id,
        )
        attempt = int(episode_dir.name.rsplit("_", maxsplit=1)[1])
        record_args = [
                "--task",
                "level4_workcell",
                "--skill",
                assignment.skill_name,
                "--source",
                assignment.source,
                "--goal-condition-id",
                assignment.coverage_cell_id,
                "--task-seed",
                str(assignment.seed),
                "--level4-procedural-repetition",
                str(assignment.repetition),
                "--session-id",
                assignment.session_id,
                "--operator-id",
                "scripted_procedural_v1",
                "--session-split",
                assignment.split,
                "--level4-dataset-config",
                str(args.config),
                "--level4-dataset-dir",
                str(args.dataset_dir),
                "--workcell-config",
                str(args.workcell_config),
                "--episode-id",
                (
                    f"{assignment.episode_id_prefix}_{assignment.sequence:06d}"
                    if attempt == 1
                    else (
                        f"{assignment.episode_id_prefix}_{assignment.sequence:06d}"
                        f"_retry_{attempt:03d}"
                    )
                ),
                "--enforce-frozen-cell-owner",
                "--print-interval",
                "1000",
        ]
        if assignment.session_id in existing_session_ids:
            record_args.append("--resume-existing-session")
        result = record_demo.main(record_args)
        if not episode_dir.exists():
            print(
                f"ERROR: assignment {assignment.sequence} did not create {episode_dir}",
                file=sys.stderr,
            )
            return result or 2
        audit = audit_scripted_episode(
            episode_dir,
            config_path=args.config,
            workcell_config=args.workcell_config,
        )
        save_pilot_review(
            episode_dir,
            PilotReview(
                episode_id=audit.episode_id,
                schema_validation=audit.schema_validation,
                timestamp_alignment=audit.timestamp_alignment,
                headless_replay=audit.headless_replay,
                terminal_metric_recomputation=audit.terminal_metric_recomputation,
                recomputed_success=audit.recomputed_success,
                operator_label_agreement=audit.operator_label_agreement,
                quality_thresholds=(
                    audit.safety_violation_count == 0
                    and audit.maximum_neighbor_disturbance_m
                    <= audit.neighbor_disturbance_limit_m
                ),
                coverage_assignment=audit.coverage_assignment,
                split_session_leakage_audit=True,
                expert_accepted=audit.accepted,
                rejection_reasons=audit.rejection_reasons,
            ),
        )
        if audit.accepted:
            accepted += 1
        else:
            rejected += 1
            print(
                f"ERROR: assignment {assignment.sequence} failed audit: "
                f"{', '.join(audit.rejection_reasons)}",
                file=sys.stderr,
            )
            return 1
        if index == 1 or index % args.print_interval == 0 or index == len(missing):
            print(
                f"Progress: processed={index}/{len(missing)} "
                f"accepted={accepted} rejected={rejected}"
            )
    return 0


def _procedural_repetition(metadata: object) -> int | None:
    if not isinstance(metadata, dict):
        return None
    expansion = metadata.get("procedural_expansion")
    if not isinstance(expansion, dict):
        return None
    value = expansion.get("repetition")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _require_sufficient_scaling_decision(path: Path, *, config_path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse scaling report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"scaling report {path} must contain a JSON object.")
    if payload.get("dataset_sufficient") is not True:
        raise ValueError(
            "test generation is locked until the frozen train/validation scaling "
            "report declares dataset_sufficient=true."
        )
    if payload.get("decision_uses_test_data") is not False:
        raise ValueError("scaling decision must explicitly exclude test data.")
    digest = "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
    if payload.get("config_digest") != digest:
        raise ValueError(
            "scaling report does not match the active frozen dataset config; "
            "rerun the train/validation probe before test generation."
        )


if __name__ == "__main__":
    raise SystemExit(main())
