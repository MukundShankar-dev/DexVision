"""Planning helpers for scaled reach-touch data collection."""

from __future__ import annotations

import json
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
from dexvision.sim.tasks import ReachTouchTargetConfig


DEFAULT_REACH_TOUCH_DATASET = Path("data/demos/raw/reach_touch_target")
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
        target_counts=tuple((target, counts[target]) for target in REACH_TOUCH_TARGET_SITES),
        clean_target_counts=tuple(
            (target, clean_counts[target]) for target in REACH_TOUCH_TARGET_SITES
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
    used_numbers = {
        int(match.group(1))
        for path in dataset.iterdir()
        if path.is_dir() and (match := pattern.fullmatch(path.name)) is not None
    } if dataset.is_dir() else set()
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
        raise CollectionPlannerError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise CollectionPlannerError(f"{path} must contain a JSON object.")
    return metadata
