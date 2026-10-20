"""The interface every controller implements: baselines, scripted experts and learned policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from dexvision.tasks.features import TaskContext


class Policy(Protocol):
    name: str

    def reset(self, context: TaskContext, seed: int) -> None:
        """Start an episode. ``seed`` is the episode seed, for policies that sample."""

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Map the declared observation fields to one 26-D action."""
