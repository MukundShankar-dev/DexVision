"""Export a Level 2 task's policy-free skill metadata stub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.skill_card_metadata import (
    DEFAULT_DATASET_SUMMARY,
    DEFAULT_SKILL_VERSION,
    SkillMetadataError,
    build_skill_metadata,
    save_skill_metadata,
)
from dexvision.sim.mujoco_env import MujocoError
from dexvision.sim.tasks import (
    BUTTON_PRESS_TASK_ID,
    DEFAULT_TASK_BOARD_MODEL,
    PUSH_CUBE_TASK_ID,
    REACH_TOUCH_TARGET_TASK_ID,
    ButtonPressTask,
    PushCubeTask,
    ReachTouchTargetTask,
    TaskError,
    TaskSpec,
)


SUPPORTED_TASKS = (
    REACH_TOUCH_TARGET_TASK_ID,
    BUTTON_PRESS_TASK_ID,
    PUSH_CUBE_TASK_ID,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a policy-free Level 2 skill metadata stub from a task spec "
            "and dataset summary."
        )
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        required=True,
        help="Level 2 task/skill to export.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_TASK_BOARD_MODEL,
        help=f"Task-board MuJoCo XML. Defaults to {DEFAULT_TASK_BOARD_MODEL}.",
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=DEFAULT_DATASET_SUMMARY,
        help=f"Dataset summary JSON. Defaults to {DEFAULT_DATASET_SUMMARY}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to data/skill_metadata/<task>.json.",
    )
    parser.add_argument(
        "--skill-version",
        default=DEFAULT_SKILL_VERSION,
        help=f"Metadata stub skill version. Defaults to {DEFAULT_SKILL_VERSION}.",
    )
    return parser


def load_task_spec(task_id: str, model_path: Path) -> TaskSpec:
    """Load a supported task long enough to extract its immutable spec."""

    task_types = {
        REACH_TOUCH_TARGET_TASK_ID: ReachTouchTargetTask,
        BUTTON_PRESS_TASK_ID: ButtonPressTask,
        PUSH_CUBE_TASK_ID: PushCubeTask,
    }
    task_type = task_types.get(task_id)
    if task_type is None:
        raise ValueError(f"Unsupported task id: {task_id}")
    with task_type(model_path) as task:
        return task.spec


def run_export(args: argparse.Namespace) -> int:
    """Build and save the requested metadata stub."""

    output = args.output or Path("data") / "skill_metadata" / f"{args.task}.json"
    print("DexVision Level 2 skill metadata export")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Dataset summary: {args.dataset_summary}")
    print(f"Output: {output}")
    print("Policy checkpoint: not required")

    task_spec = load_task_spec(args.task, args.model)
    metadata = build_skill_metadata(
        task_spec,
        dataset_summary_path=args.dataset_summary,
        skill_version=args.skill_version,
    )
    saved_path = save_skill_metadata(metadata, output)
    print(
        "Export complete: "
        f"action={metadata.action_schema_version}, "
        f"observation={metadata.observation_schema_version}, "
        f"level3_ready={metadata.dataset_summary['level3_ready']}"
    )
    print(f"Saved metadata: {saved_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_export(args)
    except (MujocoError, TaskError, SkillMetadataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
