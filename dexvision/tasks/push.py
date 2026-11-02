"""Push: slide one object across the table to a goal spot, keeping it upright and on the table."""

from __future__ import annotations

import math

import numpy as np

from dexvision.sim.workcell import Contact, Observation, Placement, Workcell
from dexvision.tasks.contacts import bodies_touching, is_hand
from dexvision.tasks.features import TaskContext
from dexvision.tasks.spec import StepCheck, TaskSpec


def push_qualifies(position: np.ndarray, goal: np.ndarray, start_z: float, tolerance_xy_m: float,
                   max_height_change_m: float) -> bool:
    """The push success predicate on the measured object position. See configs/tasks/push.yaml."""
    near = float(np.linalg.norm(np.asarray(position[:2]) - goal[:2])) <= tolerance_xy_m
    resting = abs(float(position[2]) - start_z) <= max_height_change_m
    return near and resting


def push_phase(success: bool, failure: str | None, metrics: dict[str, float]) -> str:
    """How far an episode got, from the success predicate's own pieces (no new thresholds):
    hand on the object, then the object at the goal and resting, then held there for the dwell.
    An episode that never reached the goal is split by whether the object stayed resting."""
    if success:
        return "success"
    if failure == "object_fell_off_table":
        return "fell_off_table"
    if failure == "hand_collision":
        return "hand_collision"
    if metrics["first_contact_step"] < 0:
        return "no_contact"
    if metrics["max_dwell_steps"] > 0:
        return "at_goal_not_held"
    if metrics["tipped_or_lifted"]:
        return "tipped_or_lifted"
    return "not_at_goal"


class PushTask:
    phase = staticmethod(push_phase)

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        reset, success, failure = spec.raw["reset"], spec.raw["success"], spec.raw["failure"]
        self.objects = tuple(spec.raw["objects"])
        self._center = tuple(reset["object_center_xy_m"])
        self._half_extent = tuple(reset["object_half_extent_xy_m"])
        self._yaw_range = tuple(reset["object_yaw_range_rad"])
        self._distance = tuple(reset["push_distance_m"])
        self._angle = tuple(reset["push_angle_rad"])
        self._hand_jitter = np.asarray(reset["hand_jitter_m"], dtype=float)
        self.tolerance_xy_m = float(success["tolerance_xy_m"])
        self.max_height_change_m = float(success["max_height_change_m"])
        self.dwell_steps = int(success["dwell_steps"])
        self.min_object_z_m = float(failure["min_object_z_m"])
        self.collision_bodies = frozenset(failure["hand_collision_bodies"])
        self._context: TaskContext | None = None

    def reset(self, workcell: Workcell, seed: int) -> tuple[TaskContext, Observation]:
        rng = np.random.default_rng([seed, 1])  # independent of the workcell's placement stream
        target = self.objects[int(rng.integers(len(self.objects)))]
        hand_offset = rng.uniform(-self._hand_jitter, self._hand_jitter)
        distance = rng.uniform(*self._distance)
        angle = rng.uniform(*self._angle)
        layout = {target: Placement(self._center, self._half_extent, self._yaw_range)}
        observation = workcell.reset(seed, layout, hand_offset=hand_offset)
        start = observation.objects[target].position.copy()
        goal = start + distance * np.array([math.cos(angle), math.sin(angle), 0.0])
        workcell.set_goal_marker(goal)  # visual only: no collision, no mass
        shape, size = workcell.object_geometry(target)
        self._context = TaskContext("push", target, shape, size, goal_position=goal)
        self._start_z = float(start[2])
        self._dwell = 0
        self.min_goal_distance_m = distance
        self.push_angle_rad = angle
        self.max_height_change = 0.0
        self.push_distance_m = distance
        self._start_xy = start[:2].copy()
        self._step = 0
        self.first_contact_step = -1
        self.max_dwell_steps = 0
        self.final_goal_distance_m = distance
        return self._context, observation

    def check(self, observation: Observation, contacts: tuple[Contact, ...]) -> StepCheck:
        context = self._context
        position = observation.objects[context.target_object].position
        self.min_goal_distance_m = min(
            self.min_goal_distance_m,
            float(np.linalg.norm(position[:2] - context.goal_position[:2])))
        height_change = abs(float(position[2]) - self._start_z)
        self.max_height_change = max(self.max_height_change, height_change)
        qualifies = push_qualifies(position, context.goal_position, self._start_z,
                                   self.tolerance_xy_m, self.max_height_change_m)
        self._dwell = self._dwell + 1 if qualifies else 0
        self._track(position, contacts)
        failure = None
        if position[2] < self.min_object_z_m:
            failure = "object_fell_off_table"
        elif any(
            (is_hand(c.body1) and c.body2 in self.collision_bodies)
            or (is_hand(c.body2) and c.body1 in self.collision_bodies)
            for c in contacts
        ):
            failure = "hand_collision"
        return StepCheck(qualifies, self._dwell >= self.dwell_steps and not failure, failure)

    def _track(self, position: np.ndarray, contacts: tuple[Contact, ...]) -> None:
        """Diagnostic measurements for failure analysis; they never affect the outcome."""
        self._step += 1
        self.max_dwell_steps = max(self.max_dwell_steps, self._dwell)
        touching = bodies_touching(contacts, self._context.target_object)
        if self.first_contact_step < 0 and any(is_hand(b) for b in touching):
            self.first_contact_step = self._step
        self.final_goal_distance_m = float(
            np.linalg.norm(position[:2] - self._context.goal_position[:2]))

    def grid_axes(self) -> tuple[tuple[str, float, float], ...]:
        """(metric, low, high) of the reset range, for success by start position."""
        return tuple((f"start_{axis}_m", c - h, c + h)
                     for axis, c, h in zip("xy", self._center, self._half_extent, strict=True))

    def episode_metrics(self) -> dict[str, float]:
        """Outcome-related measurements, and the reset's start and goal for breakdowns."""
        return {"min_goal_distance_m": round(self.min_goal_distance_m, 4),
                "final_goal_distance_m": round(self.final_goal_distance_m, 4),
                "max_height_change_m": round(self.max_height_change, 4),
                # the success predicate's resting condition was broken at some step
                "tipped_or_lifted": float(self.max_height_change > self.max_height_change_m),
                "first_contact_step": self.first_contact_step,
                "max_dwell_steps": self.max_dwell_steps,
                "push_angle_deg": round(math.degrees(self.push_angle_rad), 1),  # from the reset
                "push_distance_m": round(self.push_distance_m, 4),
                "start_x_m": round(float(self._start_xy[0]), 4),
                "start_y_m": round(float(self._start_xy[1]), 4)}
