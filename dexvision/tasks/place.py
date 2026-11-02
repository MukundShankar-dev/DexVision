"""Place: pick one object up, carry it, and set it down upright on a goal spot, released."""

from __future__ import annotations

import numpy as np

from dexvision.sim.workcell import Contact, Observation, Placement, Workcell
from dexvision.tasks.contacts import bodies_touching, is_hand
from dexvision.tasks.features import TaskContext
from dexvision.tasks.reach import hand_collisions
from dexvision.tasks.spec import StepCheck, TaskSpec


def tilt(orientation_wxyz: np.ndarray) -> float:
    """Angle (rad) between a body's z-axis and the world vertical."""
    w, x, y, _ = orientation_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)  # world z of the body z-axis
    return float(np.arccos(np.clip(up_z, -1.0, 1.0)))


def place_phase(success: bool, failure: str | None, metrics: dict[str, float]) -> str:
    """How far an episode got, from the success predicate's own pieces (no new thresholds):
    hand on the object, carried (lifted), over the goal, set down resting there, released and
    left for the dwell."""
    if success:
        return "success"
    if failure == "object_fell_off_table":
        return "fell_off_table"
    if failure == "hand_collision":
        return "hand_collision"
    if metrics["first_contact_step"] < 0:
        return "no_contact"
    if not metrics["carried"]:
        return "contact_not_lifted"
    if not metrics["reached_goal"]:
        return "carried_not_at_goal"
    if not metrics["set_down_at_goal"]:
        return "at_goal_not_set_down"
    if metrics["max_dwell_steps"] == 0:
        return "set_down_not_released"
    return "released_not_held"


class PlaceTask:
    phase = staticmethod(place_phase)

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        reset, success, failure = spec.raw["reset"], spec.raw["success"], spec.raw["failure"]
        self.objects = tuple(spec.raw["objects"])
        self._center = tuple(reset["object_center_xy_m"])
        self._half_extent = tuple(reset["object_half_extent_xy_m"])
        self._yaw_range = tuple(reset["object_yaw_range_rad"])
        self._hand_jitter = np.asarray(reset["hand_jitter_m"], dtype=float)
        self._goal_low = np.array([reset["goal_x_m"][0], reset["goal_y_m"][0]])
        self._goal_high = np.array([reset["goal_x_m"][1], reset["goal_y_m"][1]])
        self.min_goal_distance_m = float(reset["min_goal_distance_m"])
        self.min_lift_m = float(success["min_lift_m"])
        self.tolerance_xy_m = float(success["tolerance_xy_m"])
        self.max_height_change_m = float(success["max_height_change_m"])
        self.max_tilt_rad = float(success["max_tilt_rad"])
        self.dwell_steps = int(success["dwell_steps"])
        self.min_object_z_m = float(failure["min_object_z_m"])
        self.collision_bodies = frozenset(failure["hand_collision_bodies"])
        self._context: TaskContext | None = None

    def reset(self, workcell: Workcell, seed: int) -> tuple[TaskContext, Observation]:
        rng = np.random.default_rng([seed, 1])  # independent of the workcell's placement stream
        target = self.objects[int(rng.integers(len(self.objects)))]
        hand_offset = rng.uniform(-self._hand_jitter, self._hand_jitter)
        layout = {target: Placement(self._center, self._half_extent, self._yaw_range)}
        observation = workcell.reset(seed, layout, hand_offset=hand_offset)
        start = observation.objects[target].position.copy()
        while True:  # the goal region is far larger than the excluded disc: this ends quickly
            xy = rng.uniform(self._goal_low, self._goal_high)
            if np.linalg.norm(xy - start[:2]) >= self.min_goal_distance_m:
                break
        goal = np.array([xy[0], xy[1], start[2]])  # the object's centre when resting there
        workcell.set_goal_marker(goal)  # visual only: no collision, no mass
        shape, size = workcell.object_geometry(target)
        self._context = TaskContext("place", target, shape, size, goal_position=goal)
        self._start_z = float(start[2])
        self._dwell = 0
        self._carried = False
        self.max_lift_m = 0.0
        self.landing_speed_mps = -1.0  # set when it first touches the table after a carry
        self.goal_distance_m = float(np.linalg.norm(xy - start[:2]))
        self._start_xy = start[:2].copy()
        self._step = 0
        self.first_contact_step = -1
        self.reached_goal = False
        self.set_down_at_goal = False
        self.max_dwell_steps = 0
        self.final_goal_distance_m = self.goal_distance_m
        return self._context, observation

    def check(self, observation: Observation, contacts: tuple[Contact, ...]) -> StepCheck:
        context = self._context
        state = observation.objects[context.target_object]
        lift = float(state.position[2]) - self._start_z
        self.max_lift_m = max(self.max_lift_m, lift)
        touching = bodies_touching(contacts, context.target_object)
        if self._carried and self.landing_speed_mps < 0 and "table" in touching:
            self.landing_speed_mps = max(0.0, -float(state.linear_velocity[2]))
        self._carried = self._carried or lift >= self.min_lift_m
        qualifies = self._carried and self.at_rest_on_goal(state.position, state.orientation,
                                                           touching)
        self._dwell = self._dwell + 1 if qualifies else 0
        self._track(state, touching)
        failure = None
        if state.position[2] < self.min_object_z_m:
            failure = "object_fell_off_table"
        elif hand_collisions(contacts) & self.collision_bodies:
            failure = "hand_collision"
        return StepCheck(qualifies, self._dwell >= self.dwell_steps and not failure, failure)

    def at_rest_on_goal(self, position: np.ndarray, orientation: np.ndarray,
                        touching: set[str]) -> bool:
        """The place predicate on measured quantities (the carry is checked separately). See
        configs/tasks/place.yaml."""
        goal = self._context.goal_position
        near = float(np.linalg.norm(position[:2] - goal[:2])) <= self.tolerance_xy_m
        resting = (abs(float(position[2]) - self._start_z) <= self.max_height_change_m
                   and tilt(orientation) <= self.max_tilt_rad)
        released = not any(is_hand(b) for b in touching)
        return near and resting and released

    def _track(self, state, touching: set[str]) -> None:
        """Diagnostic measurements for failure analysis; they never affect the outcome. The
        predicate's pieces are recorded once the object has been carried."""
        self._step += 1
        self.max_dwell_steps = max(self.max_dwell_steps, self._dwell)
        if self.first_contact_step < 0 and any(is_hand(b) for b in touching):
            self.first_contact_step = self._step
        goal = self._context.goal_position
        self.final_goal_distance_m = float(np.linalg.norm(state.position[:2] - goal[:2]))
        if self._carried:
            near = self.final_goal_distance_m <= self.tolerance_xy_m
            self.reached_goal = self.reached_goal or near
            # resting on the goal, held or not
            self.set_down_at_goal = self.set_down_at_goal or (
                near and self.at_rest_on_goal(state.position, state.orientation, set()))

    def grid_axes(self) -> tuple[tuple[str, float, float], ...]:
        """(metric, low, high) of the reset range, for success by start position."""
        return tuple((f"start_{axis}_m", c - h, c + h)
                     for axis, c, h in zip("xy", self._center, self._half_extent, strict=True))

    def episode_metrics(self) -> dict[str, float]:
        """Outcome-related measurements, and the reset's start and goal for breakdowns."""
        goal = self._context.goal_position
        return {"max_lift_m": round(self.max_lift_m, 4),
                "landing_speed_mps": round(self.landing_speed_mps, 3),  # -1: never set down
                "goal_distance_m": round(self.goal_distance_m, 4),  # from the reset
                "first_contact_step": self.first_contact_step,
                "carried": float(self._carried),
                "reached_goal": float(self.reached_goal),
                "set_down_at_goal": float(self.set_down_at_goal),
                "max_dwell_steps": self.max_dwell_steps,
                "final_goal_distance_m": round(self.final_goal_distance_m, 4),
                "start_x_m": round(float(self._start_xy[0]), 4),
                "start_y_m": round(float(self._start_xy[1]), 4),
                "goal_x_m": round(float(goal[0]), 4),
                "goal_y_m": round(float(goal[1]), 4)}
