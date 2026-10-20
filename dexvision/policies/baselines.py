"""Sanity baselines. A task whose zero or random baseline succeeds has a broken predicate."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dexvision.sim.workcell import ACTUATOR_TARGETS, ROTATION, TRANSLATION, ActionSpace
from dexvision.tasks.features import TaskContext


class ZeroPolicy:
    """Hold the hand still with the fingers at their neutral targets."""

    name = "zero"

    def __init__(self, action_space: ActionSpace) -> None:
        self._action = action_space.hold()

    def reset(self, context: TaskContext, seed: int) -> None:
        pass

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self._action.copy()


class RandomPolicy:
    """Uniform random actions within the action bounds, freshly sampled every step."""

    name = "random"

    def __init__(self, action_space: ActionSpace) -> None:
        self._space = action_space
        self._rng = np.random.default_rng(0)

    def reset(self, context: TaskContext, seed: int) -> None:
        self._rng = np.random.default_rng([seed, 2])

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        space, rng = self._space, self._rng
        action = space.hold()
        action[TRANSLATION] = rng.uniform(-1, 1, 3) * space.max_translation_m
        action[ROTATION] = rng.uniform(-1, 1, 3) * space.max_rotation_rad
        action[ACTUATOR_TARGETS] = rng.uniform(space.target_low, space.target_high)
        return action
