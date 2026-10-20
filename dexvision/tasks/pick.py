"""Pick: lift one object off the table with a thumb-opposition grasp and hold it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexvision.sim.workcell import Contact, Observation, Placement, Workcell
from dexvision.tasks.contacts import bodies_touching, held_by_opposition, is_hand
from dexvision.tasks.features import TaskContext
from dexvision.tasks.spec import TaskSpec


@dataclass(frozen=True)
class StepCheck:
    qualifies: bool  # success conditions hold at this step
    success: bool  # they have held for the required dwell
    failure: str | None  # a terminal failure reason, if any


def pick_qualifies(lift_m: float, touching: set[str], min_lift_m: float) -> bool:
    """The pick success predicate on measured quantities. See configs/tasks/pick.yaml."""
    return lift_m >= min_lift_m and held_by_opposition(touching)


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
        self._start_z = float(observation.objects[target].position[2])
        self._dwell = 0
        self.max_lift_m = 0.0
        self.max_hand_table_force_n = 0.0
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
        qualifies = pick_qualifies(lift, bodies_touching(contacts, target), self.min_lift_m)
        self._dwell = self._dwell + 1 if qualifies else 0
        failure = "object_fell_off_table" if z < self.min_object_z_m else None
        return StepCheck(qualifies, self._dwell >= self.dwell_steps, failure)

    def episode_metrics(self) -> dict[str, float]:
        return {
            "max_lift_m": round(self.max_lift_m, 4),
            "max_hand_table_force_n": round(self.max_hand_table_force_n, 1),
        }
