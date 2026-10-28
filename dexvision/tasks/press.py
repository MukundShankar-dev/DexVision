"""Press: push the wall button in, hold it briefly, then let it spring back out."""

from __future__ import annotations

import numpy as np

from dexvision.sim.workcell import Contact, Observation, Workcell
from dexvision.tasks.contacts import bodies_touching, is_hand
from dexvision.tasks.features import TaskContext
from dexvision.tasks.reach import hand_collisions
from dexvision.tasks.spec import StepCheck, TaskSpec


class PressTask:
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
        out = self._pressed and depth <= self.release_depth_m and not touching
        self._released = self._released + 1 if out else 0
        failure = "hand_collision" if hand_collisions(contacts) & self.collision_bodies else None
        success = self._released >= self.release_steps and not failure
        return StepCheck(out, success, failure)

    def episode_metrics(self) -> dict[str, float]:
        return {"max_button_depth_m": round(self.deepest_m, 4), "pressed": float(self._pressed),
                "peak_press_force_n": round(self.peak_force_n, 2)}
