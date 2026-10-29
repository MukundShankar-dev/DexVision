"""Press: push the wall button in, hold it briefly, then let it spring back out."""

from __future__ import annotations

import numpy as np

from dexvision.sim.workcell import Contact, Observation, Workcell
from dexvision.tasks.contacts import bodies_touching, is_hand
from dexvision.tasks.features import TaskContext
from dexvision.tasks.reach import hand_collisions
from dexvision.tasks.spec import StepCheck, TaskSpec


def press_phase(success: bool, failure: str | None, metrics: dict[str, float]) -> str:
    """How far an episode got, from the success predicate's own pieces (no new thresholds):
    hand on the button, then in to press depth, then held there, then released."""
    if success:
        return "success"
    if failure == "hand_collision":
        return "hand_collision"
    if metrics["first_contact_step"] < 0:
        return "no_contact"
    if metrics["max_held_steps"] == 0:
        return "touched_not_pressed"
    if not metrics["pressed"]:
        return "pressed_not_held"
    return "held_not_released"


class PressTask:
    phase = staticmethod(press_phase)

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        success = spec.raw["success"]
        self._hand_jitter = np.asarray(spec.raw["reset"]["hand_jitter_m"], dtype=float)
        reset = spec.raw["reset"]
        self._button_low = np.array([0.0, reset["button_y_m"][0], reset["button_z_m"][0]])
        self._button_high = np.array([0.0, reset["button_y_m"][1], reset["button_z_m"][1]])
        self.press_depth_m = float(success["press_depth_m"])
        self.press_steps = int(success["press_steps"])
        self.release_depth_m = float(success["release_depth_m"])
        self.release_steps = int(success["release_steps"])
        self.collision_bodies = frozenset(spec.raw["failure"]["hand_collision_bodies"])

    def reset(self, workcell: Workcell, seed: int) -> tuple[TaskContext, Observation]:
        rng = np.random.default_rng([seed, 1])  # independent of the workcell's placement stream
        hand_offset = rng.uniform(-self._hand_jitter, self._hand_jitter)
        button = rng.uniform(self._button_low, self._button_high)  # centre; x stays on the wall
        button[0] = workcell.button_mount_default[0]
        observation = workcell.reset(seed, {}, hand_offset=hand_offset,
                                     button_offset=button - workcell.button_mount_default)
        self._held = 0  # consecutive steps pressed in
        self._pressed = False  # a full press has happened
        self._released = 0  # consecutive steps released after the press
        self.deepest_m = 0.0
        self.peak_force_n = 0.0  # hand on button, measured contact normal force
        self._start = observation.button_position.copy()
        self._step = 0
        self.first_contact_step = -1
        self.max_held_steps = 0
        self.min_fingertip_m = np.inf  # closest fingertip to the button face centre
        return TaskContext("press"), observation

    def check(self, observation: Observation, contacts: tuple[Contact, ...]) -> StepCheck:
        depth = observation.button_depth
        self.deepest_m = max(self.deepest_m, depth)
        self._held = self._held + 1 if depth >= self.press_depth_m else 0
        self._pressed = self._pressed or self._held >= self.press_steps
        touching = any(is_hand(b) for b in bodies_touching(contacts, "button"))
        force = sum(c.normal_force for c in contacts
                    if "button" in (c.body1, c.body2) and (is_hand(c.body1) or is_hand(c.body2)))
        self.peak_force_n = max(self.peak_force_n, force)
        self._track(observation, touching)
        out = self._pressed and depth <= self.release_depth_m and not touching
        self._released = self._released + 1 if out else 0
        failure = "hand_collision" if hand_collisions(contacts) & self.collision_bodies else None
        success = self._released >= self.release_steps and not failure
        return StepCheck(out, success, failure)

    def _track(self, observation: Observation, touching: bool) -> None:
        """Diagnostic measurements for failure analysis; they never affect the outcome."""
        self._step += 1
        if touching and self.first_contact_step < 0:
            self.first_contact_step = self._step
        self.max_held_steps = max(self.max_held_steps, self._held)
        tips = np.linalg.norm(observation.fingertip_positions - observation.button_position, axis=1)
        self.min_fingertip_m = min(self.min_fingertip_m, float(tips.min()))

    def grid_axes(self) -> tuple[tuple[str, float, float], ...]:
        """(metric, low, high) of the reset range, for success by start position."""
        return (("start_y_m", self._button_low[1], self._button_high[1]),
                ("start_z_m", self._button_low[2], self._button_high[2]))

    def episode_metrics(self) -> dict[str, float]:
        """Outcome-related measurements, and the button's start position for breakdowns."""
        return {"max_button_depth_m": round(self.deepest_m, 4), "pressed": float(self._pressed),
                "peak_press_force_n": round(self.peak_force_n, 2),
                "first_contact_step": self.first_contact_step,
                "max_held_steps": self.max_held_steps,
                "min_fingertip_button_m": round(self.min_fingertip_m, 4),
                "start_y_m": round(float(self._start[1]), 4),
                "start_z_m": round(float(self._start[2]), 4)}
