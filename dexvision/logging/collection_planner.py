"""Planning helpers for quality-gated scaled task data collection."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from dexvision.logging.quality_filters import (
    QualityFilterError,
    evaluate_episode_quality,
)
from dexvision.logging.dataset_summary import (
    DEFAULT_BUTTON_PRESS_CONFIG,
    DEFAULT_PUSH_CUBE_CONFIG,
    ButtonGoalDefinition,
    ButtonPressDatasetConfig,
    CubeGoalDefinition,
    PushCubeDatasetConfig,
    load_button_press_dataset_config,
    load_push_cube_dataset_config,
)
from dexvision.sim.tasks import ReachTouchTargetConfig


DEFAULT_REACH_TOUCH_DATASET = Path("data/demos/raw/reach_touch_target")
DEFAULT_BUTTON_PRESS_DATASET = Path("data/demos/raw/button_press")
DEFAULT_PUSH_CUBE_DATASET = Path("data/demos/raw/push_cube_to_target")
REACH_TOUCH_TARGET_SITES = ReachTouchTargetConfig().target_sites


class CollectionPlannerError(RuntimeError):
    """Raised when a collection plan cannot be created safely."""


@dataclass(frozen=True)
class ReachTouchCollectionPlan:
    """One balanced-random reach-touch recording suggestion."""

    target_site: str
    output_directory: Path
    target_counts: tuple[tuple[str, int], ...]
    clean_target_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ButtonPressCollectionPlan:
    """One balanced-random button/depth recording suggestion."""

    goal: ButtonGoalDefinition
    output_directory: Path
    goal_counts: tuple[tuple[str, int], ...]
    clean_goal_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PushCubeCollectionPlan:
    """One balanced-random cube start/target recording suggestion."""

    goal: CubeGoalDefinition
    output_directory: Path
    goal_counts: tuple[tuple[str, int], ...]
    clean_goal_counts: tuple[tuple[str, int], ...]


def plan_reach_touch_collection(
    dataset_dir: str | Path = DEFAULT_REACH_TOUCH_DATASET,
    *,
    collection_date: date | None = None,
    seed: int | None = None,
) -> ReachTouchCollectionPlan:
    """Choose randomly among targets with the fewest clean successful demos."""

    dataset = Path(dataset_dir)
    counts, clean_counts = _target_counts(dataset)
    minimum_count = min(clean_counts.values())
    candidates = tuple(
        target
        for target in REACH_TOUCH_TARGET_SITES
        if clean_counts[target] == minimum_count
    )
    generator: random.Random = (
        random.SystemRandom() if seed is None else random.Random(seed)
    )
    target_site = generator.choice(candidates)
    output_directory = _next_episode_directory(
        dataset,
        collection_date=collection_date or date.today(),
    )
    return ReachTouchCollectionPlan(
        target_site=target_site,
        output_directory=output_directory,
        target_counts=tuple(
            (target, counts[target]) for target in REACH_TOUCH_TARGET_SITES
        ),
        clean_target_counts=tuple(
            (target, clean_counts[target]) for target in REACH_TOUCH_TARGET_SITES
        ),
    )


def plan_button_press_collection(
    dataset_dir: str | Path = DEFAULT_BUTTON_PRESS_DATASET,
    *,
    config: ButtonPressDatasetConfig | None = None,
    config_path: str | Path = DEFAULT_BUTTON_PRESS_CONFIG,
    collection_date: date | None = None,
    seed: int | None = None,
) -> ButtonPressCollectionPlan:
    """Choose randomly among button/depth goals with the fewest clean demos."""

    dataset = Path(dataset_dir)
    resolved_config = config or load_button_press_dataset_config(config_path)
    counts, clean_counts = _button_goal_counts(dataset, config=resolved_config)
    minimum_count = min(clean_counts.values())
    candidates = tuple(
        goal
        for goal in resolved_config.training_goals
        if clean_counts[goal.goal_id] == minimum_count
    )
    generator: random.Random = (
        random.SystemRandom() if seed is None else random.Random(seed)
    )
    goal = generator.choice(candidates)
    output_directory = _next_episode_directory(
        dataset,
        collection_date=collection_date or date.today(),
    )
    return ButtonPressCollectionPlan(
        goal=goal,
        output_directory=output_directory,
        goal_counts=tuple(
            (item.goal_id, counts[item.goal_id])
            for item in resolved_config.training_goals
        ),
        clean_goal_counts=tuple(
            (item.goal_id, clean_counts[item.goal_id])
            for item in resolved_config.training_goals
        ),
    )


def plan_push_cube_collection(
    dataset_dir: str | Path = DEFAULT_PUSH_CUBE_DATASET,
    *,
    config: PushCubeDatasetConfig | None = None,
    config_path: str | Path = DEFAULT_PUSH_CUBE_CONFIG,
    collection_date: date | None = None,
    seed: int | None = None,
) -> PushCubeCollectionPlan:
    """Choose randomly among cube goals with the fewest clean demos."""

    dataset = Path(dataset_dir)
    resolved_config = config or load_push_cube_dataset_config(config_path)
    counts, clean_counts = _push_cube_goal_counts(dataset, config=resolved_config)
    minimum_count = min(clean_counts.values())
    candidates = tuple(
        goal
        for goal in resolved_config.training_goals
        if clean_counts[goal.goal_id] == minimum_count
    )
    generator: random.Random = (
        random.SystemRandom() if seed is None else random.Random(seed)
    )
    goal = generator.choice(candidates)
    output_directory = _next_episode_directory(
        dataset,
        collection_date=collection_date or date.today(),
    )
    return PushCubeCollectionPlan(
        goal=goal,
        output_directory=output_directory,
        goal_counts=tuple(
            (item.goal_id, counts[item.goal_id])
            for item in resolved_config.training_goals
        ),
        clean_goal_counts=tuple(
            (item.goal_id, clean_counts[item.goal_id])
            for item in resolved_config.training_goals
        ),
    )


def format_recording_command(
    plan: ReachTouchCollectionPlan,
    *,
    python_command: str = "mjpython",
) -> str:
    """Format the recorder command for the current operating system."""

    arguments = recording_arguments(plan, python_command=python_command)
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def recording_arguments(
    plan: ReachTouchCollectionPlan,
    *,
    python_command: str = "mjpython",
) -> tuple[str, ...]:
    """Return recorder arguments suitable for a shell-free subprocess call."""

    if not python_command:
        raise CollectionPlannerError("python_command must be non-empty.")
    return (
        python_command,
        "-m",
        "dexvision.apps.record_demo",
        "--task",
        "reach_touch_target",
        "--retargeter",
        "curl",
        "--target-site",
        plan.target_site,
        "--output",
        str(plan.output_directory),
        "--level1-13-full",
    )


def format_button_recording_command(
    plan: ButtonPressCollectionPlan,
    *,
    python_command: str = "mjpython",
) -> str:
    """Format the button recorder command for the current operating system."""

    arguments = button_recording_arguments(plan, python_command=python_command)
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def button_recording_arguments(
    plan: ButtonPressCollectionPlan,
    *,
    python_command: str = "mjpython",
) -> tuple[str, ...]:
    """Return button recorder arguments for a shell-free subprocess call."""

    if not python_command:
        raise CollectionPlannerError("python_command must be non-empty.")
    return (
        python_command,
        "-m",
        "dexvision.apps.record_demo",
        "--task",
        "button_press",
        "--retargeter",
        "curl",
        "--button-id",
        plan.goal.button_id,
        "--target-press-depth",
        str(plan.goal.target_press_depth),
        "--output",
        str(plan.output_directory),
        "--level1-13-full",
    )


def format_push_cube_recording_command(
    plan: PushCubeCollectionPlan,
    *,
    python_command: str = "mjpython",
) -> str:
    """Format the push-cube recorder command for the current operating system."""

    arguments = push_cube_recording_arguments(plan, python_command=python_command)
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def push_cube_recording_arguments(
    plan: PushCubeCollectionPlan,
    *,
    python_command: str = "mjpython",
) -> tuple[str, ...]:
    """Return push-cube recorder arguments for a shell-free subprocess call."""

    if not python_command:
        raise CollectionPlannerError("python_command must be non-empty.")
    if plan.goal.target_source == "target_pose":
        target_arguments = (
            "--target-pose",
            *(str(value) for value in plan.goal.target_position),
        )
    else:
        target_arguments = ("--target-zone-id", plan.goal.target_source)
    return (
        python_command,
        "-m",
        "dexvision.apps.record_demo",
        "--task",
        "push_cube_to_target",
        "--retargeter",
        "curl",
        "--object-id",
        plan.goal.object_id,
        *target_arguments,
        "--approach-side",
        plan.goal.approach_side,
        "--output",
        str(plan.output_directory),
        "--level1-13-full",
    )


def _target_counts(dataset: Path) -> tuple[dict[str, int], dict[str, int]]:
    counts = dict.fromkeys(REACH_TOUCH_TARGET_SITES, 0)
    clean_counts = dict.fromkeys(REACH_TOUCH_TARGET_SITES, 0)
    if not dataset.exists():
        return counts, clean_counts
    if not dataset.is_dir():
        raise CollectionPlannerError(f"Dataset path is not a directory: {dataset}")

    for metadata_path in sorted(dataset.glob("*/metadata.json")):
        metadata = _load_metadata(metadata_path)
        if metadata.get("task_id") != "reach_touch_target":
            raise CollectionPlannerError(
                f"{metadata_path} is not a reach_touch_target episode."
            )
        task_config = metadata.get("task_config")
        if not isinstance(task_config, dict):
            raise CollectionPlannerError(
                f"{metadata_path} must declare task_config as an object."
            )
        target_source = task_config.get("resolved_target_source")
        if target_source not in counts:
            allowed = ", ".join(REACH_TOUCH_TARGET_SITES)
            raise CollectionPlannerError(
                f"{metadata_path} has unsupported resolved_target_source "
                f"{target_source!r}; expected one of: {allowed}."
            )
        counts[target_source] += 1
        if _episode_is_clean(metadata_path.parent):
            clean_counts[target_source] += 1
    return counts, clean_counts


def _button_goal_counts(
    dataset: Path,
    *,
    config: ButtonPressDatasetConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    counts = {goal.goal_id: 0 for goal in config.training_goals}
    clean_counts = counts.copy()
    if not dataset.exists():
        return counts, clean_counts
    if not dataset.is_dir():
        raise CollectionPlannerError(f"Dataset path is not a directory: {dataset}")

    goals_by_state = {
        (goal.button_id, goal.target_press_depth): goal
        for goal in config.training_goals
    }
    for metadata_path in sorted(dataset.glob("*/metadata.json")):
        metadata = _load_metadata(metadata_path)
        if metadata.get("task_id") != "button_press":
            raise CollectionPlannerError(
                f"{metadata_path} is not a button_press episode."
            )
        task_config = metadata.get("task_config")
        if not isinstance(task_config, dict):
            raise CollectionPlannerError(
                f"{metadata_path} must declare task_config as an object."
            )
        button_id = task_config.get("resolved_button_id")
        depth_value = task_config.get("target_press_depth")
        if isinstance(depth_value, bool):
            depth_value = None
        try:
            target_depth = float(depth_value)
        except (TypeError, ValueError):
            target_depth = math.nan
        goal = next(
            (
                candidate
                for (
                    candidate_button,
                    candidate_depth,
                ), candidate in goals_by_state.items()
                if button_id == candidate_button
                and math.isclose(
                    target_depth,
                    candidate_depth,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ),
            None,
        )
        if goal is None:
            raise CollectionPlannerError(
                f"{metadata_path} has button/depth goal "
                f"({button_id!r}, {depth_value!r}) outside the training split."
            )
        counts[goal.goal_id] += 1
        if _episode_is_clean(metadata_path.parent):
            clean_counts[goal.goal_id] += 1
    return counts, clean_counts


def _push_cube_goal_counts(
    dataset: Path,
    *,
    config: PushCubeDatasetConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    counts = {goal.goal_id: 0 for goal in config.training_goals}
    clean_counts = counts.copy()
    if not dataset.exists():
        return counts, clean_counts
    if not dataset.is_dir():
        raise CollectionPlannerError(f"Dataset path is not a directory: {dataset}")

    for metadata_path in sorted(dataset.glob("*/metadata.json")):
        metadata = _load_metadata(metadata_path)
        if metadata.get("task_id") != "push_cube_to_target":
            raise CollectionPlannerError(
                f"{metadata_path} is not a push_cube_to_target episode."
            )
        task_config = metadata.get("task_config")
        if not isinstance(task_config, dict):
            raise CollectionPlannerError(
                f"{metadata_path} must declare task_config as an object."
            )
        goal = _matching_cube_goal(task_config, config=config)
        if goal is None:
            raise CollectionPlannerError(
                f"{metadata_path} has a cube start/target goal outside the training split."
            )
        counts[goal.goal_id] += 1
        if _episode_is_clean(metadata_path.parent):
            clean_counts[goal.goal_id] += 1
    return counts, clean_counts


def _matching_cube_goal(
    task_config: dict[str, Any],
    *,
    config: PushCubeDatasetConfig,
) -> CubeGoalDefinition | None:
    object_id = task_config.get("resolved_object_id")
    target_source = task_config.get("resolved_target_source")
    approach_side = task_config.get("resolved_approach_side")
    try:
        initial_position = tuple(
            float(value) for value in task_config["initial_object_position"]
        )
        target_position = tuple(
            float(value) for value in task_config["target_position"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if len(initial_position) != 3 or len(target_position) != 3:
        return None
    return next(
        (
            goal
            for goal in config.training_goals
            if object_id == goal.object_id
            and target_source == goal.target_source
            and approach_side == goal.approach_side
            and all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
                for actual, expected in zip(
                    initial_position,
                    goal.initial_object_position,
                    strict=True,
                )
            )
            and all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
                for actual, expected in zip(
                    target_position,
                    goal.target_position,
                    strict=True,
                )
            )
        ),
        None,
    )


def _episode_is_clean(episode_dir: Path) -> bool:
    try:
        return evaluate_episode_quality(episode_dir).passed
    except QualityFilterError as exc:
        raise CollectionPlannerError(
            f"Could not evaluate existing episode quality for {episode_dir}: {exc}"
        ) from exc


def _next_episode_directory(dataset: Path, *, collection_date: date) -> Path:
    prefix = collection_date.isoformat()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    used_numbers = (
        {
            int(match.group(1))
            for path in dataset.iterdir()
            if path.is_dir() and (match := pattern.fullmatch(path.name)) is not None
        }
        if dataset.is_dir()
        else set()
    )
    next_number = max(used_numbers, default=0) + 1
    output = dataset / f"{prefix}_{next_number:03d}"
    while output.exists():
        next_number += 1
        output = dataset / f"{prefix}_{next_number:03d}"
    return output


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionPlannerError(
            f"Could not read valid JSON from {path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise CollectionPlannerError(f"{path} must contain a JSON object.")
    return metadata
