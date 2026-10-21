"""Reach: bring the measured grasp point to a goal in free space and hold it there."""

from __future__ import annotations

import numpy as np

from dexvision.sim.workcell import Contact, Observation, Workcell
from dexvision.tasks.contacts import is_hand
from dexvision.tasks.features import TaskContext
from dexvision.tasks.spec import StepCheck, TaskSpec

MAX_GOAL_SAMPLES = 1000


def reach_qualifies(grasp_position: np.ndarray, goal: np.ndarray, tolerance_m: float) -> bool:
    """The reach success predicate on the measured grasp point. See configs/tasks/reach.yaml."""
    return float(np.linalg.norm(np.asarray(grasp_position) - goal)) <= tolerance_m


def hand_collisions(contacts: tuple[Contact, ...]) -> set[str]:
    """Non-hand bodies the hand is touching."""
    hits = set()
    for c in contacts:
        if is_hand(c.body1) != is_hand(c.body2):
            hits.add(c.body2 if is_hand(c.body1) else c.body1)
    return hits


class ReachTask:
    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        reset, success = spec.raw["reset"], spec.raw["success"]
        self._center = np.asarray(reset["goal_center_m"], dtype=float)
        self._half_extent = np.asarray(reset["goal_half_extent_m"], dtype=float)
        self._min_distance = float(reset["min_goal_distance_m"])
        self._hand_jitter = np.asarray(reset["hand_jitter_m"], dtype=float)
        self.tolerance_m = float(success["tolerance_m"])
        self.dwell_steps = int(success["dwell_steps"])
        self.fail_on_collision = bool(spec.raw["failure"]["hand_collision"])
        self._context: TaskContext | None = None

    def reset(self, workcell: Workcell, seed: int) -> tuple[TaskContext, Observation]:
        rng = np.random.default_rng([seed, 1])
        hand_offset = rng.uniform(-self._hand_jitter, self._hand_jitter)
        observation = workcell.reset(seed, {}, hand_offset=hand_offset)
        start = observation.grasp_position
        for _ in range(MAX_GOAL_SAMPLES):
            goal = self._center + rng.uniform(-self._half_extent, self._half_extent)
            if np.linalg.norm(goal - start) >= self._min_distance:
                break
        else:
            raise RuntimeError(f"No goal at least {self._min_distance} m from the start")
        workcell.set_goal_marker(goal)  # visual only: no collision, no mass
        self._context = TaskContext("reach", goal_position=goal)
        self._goal = goal
        self._dwell = 0
        self.min_distance_m = float(np.linalg.norm(start - goal))
        return self._context, observation

    def check(self, observation: Observation, contacts: tuple[Contact, ...]) -> StepCheck:
        distance = float(np.linalg.norm(observation.grasp_position - self._goal))
        self.min_distance_m = min(self.min_distance_m, distance)
        qualifies = reach_qualifies(observation.grasp_position, self._goal, self.tolerance_m)
        self._dwell = self._dwell + 1 if qualifies else 0
        failure = None
        if self.fail_on_collision and hand_collisions(contacts):
            failure = "hand_collision"
        return StepCheck(qualifies, self._dwell >= self.dwell_steps and not failure, failure)

    def episode_metrics(self) -> dict[str, float]:
        return {"min_distance_m": round(self.min_distance_m, 4)}
