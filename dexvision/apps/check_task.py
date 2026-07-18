"""Headless smoke check for one DexVision Level 2 task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dexvision.sim.mujoco_env import MujocoError
from dexvision.sim.tasks import (
    DEFAULT_TASK_BOARD_MODEL,
    REACH_TOUCH_TARGET_TASK_ID,
    ReachTouchTargetParameters,
    ReachTouchTargetTask,
    TaskError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load, reset, and inspect a DexVision task headlessly."
    )
    parser.add_argument(
        "--task",
        choices=(REACH_TOUCH_TARGET_TASK_ID,),
        default=REACH_TOUCH_TARGET_TASK_ID,
        help="Task id to check.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_TASK_BOARD_MODEL,
        help=f"Task-board MuJoCo XML. Defaults to {DEFAULT_TASK_BOARD_MODEL}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic reset seed.",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-site",
        help="Use one configured named target site instead of seeded sampling.",
    )
    target_group.add_argument(
        "--target-pose",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Use an explicit target position in MuJoCo world metres.",
    )
    return parser


def run_task_check(
    *,
    task_id: str,
    model_path: Path,
    seed: int,
    parameters: ReachTouchTargetParameters,
) -> int:
    if task_id != REACH_TOUCH_TARGET_TASK_ID:
        raise ValueError(f"Unsupported task id: {task_id}")

    print("DexVision task check")
    print(f"Task: {task_id}")
    print(f"Model: {model_path}")
    print(f"Seed: {seed}")
    print("Viewer: off")
    print("No camera or MediaPipe is used.")

    with ReachTouchTargetTask(model_path) as task:
        first = task.reset(seed=seed, parameters=parameters)
        first_vector = first.as_task_state()
        second = task.reset(seed=seed, parameters=parameters)
        if first.target_source != second.target_source or not np.array_equal(
            first.target_position,
            second.target_position,
        ):
            raise TaskError("deterministic reset check failed for identical seeds.")
        stepped = task.step(n_steps=1)

        print(f"Target source: {second.target_source}")
        print(f"Target position: {second.target_position.tolist()}")
        print(f"Touch position: {second.touch_position.tolist()}")
        print(f"Distance: {second.distance_to_target:.6f} m")
        print(
            "Success rule: "
            f"{task.spec.success_condition}; current_success={stepped.success}"
        )
        print(
            f"State vector: shape={first_vector.shape}, "
            f"finite={bool(np.all(np.isfinite(first_vector)))}"
        )
        print(
            "Schemas: "
            f"action={task.spec.action_schema.version}, "
            f"observation={task.spec.observation_schema.version}"
        )

    print("Task board scene load and deterministic reset: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parameters = ReachTouchTargetParameters(
            target_pose=tuple(args.target_pose) if args.target_pose is not None else None,
            target_site=args.target_site,
        )
        return run_task_check(
            task_id=args.task,
            model_path=args.model,
            seed=args.seed,
            parameters=parameters,
        )
    except (MujocoError, TaskError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Task check closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
