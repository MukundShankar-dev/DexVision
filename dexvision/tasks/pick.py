"""Pick: lift one object off the table with a thumb-opposition grasp and hold it."""

from __future__ import annotations

import numpy as np

from dexvision.sim.workcell import Contact, Observation, Placement, Workcell
from dexvision.tasks.contacts import bodies_touching, held_by_opposition, is_hand
from dexvision.tasks.features import TaskContext
from dexvision.tasks.spec import StepCheck, TaskSpec


def pick_qualifies(lift_m: float, touching: set[str], min_lift_m: float) -> bool:
    """The pick success predicate on measured quantities. See configs/tasks/pick.yaml."""
    return lift_m >= min_lift_m and held_by_opposition(touching)


def pick_phase(success: bool, failure: str | None, metrics: dict[str, float]) -> str:
    """How far an episode got, from the success predicate's own pieces (no new thresholds):
    contact, then an opposition grasp, then grasp and lift together, then held for the dwell."""
    if success:
        return "success"
    if failure == "object_fell_off_table":
        return "fell_off_table"
    if metrics["first_contact_step"] < 0:
        return "no_contact"
    if metrics["first_grasp_step"] < 0:
        return "contact_no_grasp"
    if metrics["max_dwell_steps"] == 0:
        return "grasped_not_lifted"
    return "lifted_not_held"


class PickTask:
    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        reset, success = spec.raw["reset"], spec.raw["success"]
        self.objects = tuple(spec.raw["objects"])
        self._center = tuple(reset["object_center_xy_m"])
        self._half_extent = tuple(reset["object_half_extent_xy_m"])
        self._yaw_range = tuple(reset["object_yaw_range_rad"])
        self._hand_jitter = np.asarray(reset["hand_jitter_m"], dtype=float)
        self.min_lift_m = float(success["min_lift_m"])
        self.dwell_steps = int(success["dwell_steps"])
        self.min_object_z_m = float(spec.raw["failure"]["min_object_z_m"])
        self._context: TaskContext | None = None

    def reset(self, workcell: Workcell, seed: int) -> tuple[TaskContext, Observation]:
        rng = np.random.default_rng([seed, 1])  # independent of the workcell's placement stream
        target = self.objects[int(rng.integers(len(self.objects)))]
        hand_offset = rng.uniform(-self._hand_jitter, self._hand_jitter)
        layout = {target: Placement(self._center, self._half_extent, self._yaw_range)}
        observation = workcell.reset(seed, layout, hand_offset=hand_offset)
        shape, size = workcell.object_geometry(target)
        self._context = TaskContext("pick", target, shape, size)
        start = observation.objects[target]
        self._start_z = float(start.position[2])
        self._start_xy = start.position[:2].copy()
        w, x, y, z = start.orientation
        self._start_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        self._dwell = 0
        self._step = 0
        self.max_lift_m = 0.0
        self.max_hand_table_force_n = 0.0
        self.max_dwell_steps = 0
        self.first_contact_step = -1
        self.first_grasp_step = -1
        self.min_grasp_xy_m = float("inf")
        self.shift_before_grasp_m = 0.0
        return self._context, observation

    def check(self, observation: Observation, contacts: tuple[Contact, ...]) -> StepCheck:
        target = self._context.target_object
        z = float(observation.objects[target].position[2])
        lift = z - self._start_z
        self.max_lift_m = max(self.max_lift_m, lift)
        table_force = sum(
            c.normal_force for c in contacts
            if "table" in (c.body1, c.body2) and (is_hand(c.body1) or is_hand(c.body2))
        )
        self.max_hand_table_force_n = max(self.max_hand_table_force_n, table_force)
        touching = bodies_touching(contacts, target)
        qualifies = pick_qualifies(lift, touching, self.min_lift_m)
        self._dwell = self._dwell + 1 if qualifies else 0
        self._track(observation, touching)
        failure = "object_fell_off_table" if z < self.min_object_z_m else None
        return StepCheck(qualifies, self._dwell >= self.dwell_steps, failure)

    def _track(self, observation: Observation, touching: set[str]) -> None:
        """Diagnostic measurements for failure analysis; they never affect the outcome."""
        self._step += 1
        position = observation.objects[self._context.target_object].position
        self.max_dwell_steps = max(self.max_dwell_steps, self._dwell)
        if self.first_contact_step < 0 and any(is_hand(b) for b in touching):
            self.first_contact_step = self._step
        if self.first_grasp_step < 0:
            if held_by_opposition(touching):
                self.first_grasp_step = self._step
            else:  # how far the hand pushed the object around before holding it
                shift = float(np.linalg.norm(position[:2] - self._start_xy))
                self.shift_before_grasp_m = max(self.shift_before_grasp_m, shift)
        reach = float(np.linalg.norm(observation.grasp_position[:2] - position[:2]))
        self.min_grasp_xy_m = min(self.min_grasp_xy_m, reach)

    phase = staticmethod(pick_phase)

    def episode_metrics(self) -> dict[str, float]:
        """Outcome-related measurements, and the start pose for breakdowns by position."""
        return {
            "max_lift_m": round(self.max_lift_m, 4),
            "max_hand_table_force_n": round(self.max_hand_table_force_n, 1),
            "max_dwell_steps": self.max_dwell_steps,
            "first_contact_step": self.first_contact_step,
            "first_grasp_step": self.first_grasp_step,
            # closest horizontal approach of the grasp point to the object centre
            "min_grasp_xy_m": round(self.min_grasp_xy_m, 4),
            # farthest the object was moved (horizontally) before the first grasp
            "shift_before_grasp_m": round(self.shift_before_grasp_m, 4),
            "start_x_m": round(float(self._start_xy[0]), 4),
            "start_y_m": round(float(self._start_xy[1]), 4),
            "start_yaw_rad": round(self._start_yaw, 4),
        }
