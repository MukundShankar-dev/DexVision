"""Task specs: the eval-first contract for one skill, loaded from ``configs/tasks/<skill>.yaml``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dexvision import REPO_ROOT

TASK_CONFIG_DIR = REPO_ROOT / "configs" / "tasks"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class StepCheck:
    """A task's verdict on one step."""

    qualifies: bool  # success conditions hold at this step
    success: bool  # they have held for the required dwell
    failure: str | None  # a terminal failure reason, if any


class TaskSpecError(ValueError):
    """Malformed task spec."""


@dataclass(frozen=True)
class TaskSpec:
    skill: str
    observation: tuple[str, ...]
    horizon_steps: int
    splits: Mapping[str, tuple[int, int]]
    test_budget: int
    raw: Mapping[str, Any]  # the full YAML, for skill-specific sections
    path: Path

    def seeds(self, split: str, n: int) -> range:
        """The first ``n`` episode seeds of ``split``. Seeds are never shared across splits."""
        if split not in self.splits:
            raise TaskSpecError(f"Unknown split '{split}'. Use one of {SPLITS}.")
        start, stop = self.splits[split]
        if not 0 < n <= stop - start:
            raise TaskSpecError(f"Split '{split}' has {stop - start} seeds; asked for {n}.")
        return range(start, start + n)


def load_task_spec(skill_or_path: str | Path) -> TaskSpec:
    path = Path(skill_or_path)
    if not path.suffix:
        path = TASK_CONFIG_DIR / f"{skill_or_path}.yaml"
    if not path.is_file():
        raise TaskSpecError(f"Task spec not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    splits = {name: tuple(int(v) for v in raw["splits"][name]) for name in SPLITS}
    ordered = sorted(splits.values())
    for (_, stop), (start, _) in zip(ordered, ordered[1:], strict=False):
        if start < stop:
            raise TaskSpecError(f"Seed splits overlap in {path}.")
    if any(start >= stop for start, stop in ordered):
        raise TaskSpecError(f"Empty seed split in {path}.")
    return TaskSpec(
        skill=str(raw["skill"]),
        observation=tuple(raw["observation"]),
        horizon_steps=int(raw["horizon_steps"]),
        splits=splits,
        test_budget=int(raw["test_budget"]),
        raw=raw,
        path=path,
    )
