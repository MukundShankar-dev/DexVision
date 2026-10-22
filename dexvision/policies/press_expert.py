"""Closed-loop scripted press expert.

The hand makes a pointing fist (index straight, the other fingers curled, thumb tucked), keeps
the palm at its starting orientation, and steers the measured index fingertip: first to a point
in front of the button, then into it, servoing on the measured button depth so it never drives
into the end stop. It holds briefly and pulls back out so the button springs back.

It reads only the declared observation fields; the button's pose is a scene constant in the
expert config. No simulator access, no lookahead.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from dexvision import REPO_ROOT
from dexvision.policies.reach_expert import rotation_vector
from dexvision.sim.workcell import ACTUATOR_TARGETS, ROTATION, TRANSLATION, ActionSpace
from dexvision.tasks.features import TaskContext, rotation_from_6d

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experts" / "press.yaml"
PRESS_AXIS = np.array([1.0, 0.0, 0.0])  # the button slides along +x
INDEX = 0  # fingertip_positions order: ff, mf, rf, lf, th


class PressExpert:
    name = "expert"

    def __init__(self, action_space: ActionSpace, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._space = action_space
        names = action_space.actuator_names
        fingers = action_space.neutral_targets.copy()
        for joint, value in self.config["point_posture"].items():
            fingers[names.index(joint)] = value
        self._fingers = np.clip(fingers, action_space.target_low, action_space.target_high)
        face = np.asarray(self.config["button_face_m"], dtype=float)
        self._face = face
        self._ready_at = face - PRESS_AXIS * self.config["standoff_m"]

    def reset(self, context: TaskContext, seed: int) -> None:
        self._phase, self._phase_steps = "settle", 0
        self._initial_palm = None

    def _enter(self, phase: str) -> None:
        self._phase, self._phase_steps = phase, 0

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        c = self.config
        palm = rotation_from_6d(observation["palm_rotation6d"])
        if self._initial_palm is None:
            self._initial_palm = palm
        tip = observation["fingertip_positions"].reshape(5, 3)[INDEX] + c["tip_offset_m"]
        depth = float(observation["button_depth"][0])
        self._phase_steps += 1
        target = getattr(self, f"_{self._phase}")(tip, depth)

        translation = c["gains"]["position"] * (target - tip)
        if self._phase in ("press", "release"):  # slow and steady in contact
            norm = float(np.linalg.norm(translation))
            if norm > c["max_press_step_m"]:
                translation *= c["max_press_step_m"] / norm
        action = self._space.hold()
        action[TRANSLATION] = translation
        action[ROTATION] = c["gains"]["orientation"] * rotation_vector(self._initial_palm @ palm.T)
        action[ACTUATOR_TARGETS] = self._fingers
        return action

    # Each phase returns the fingertip target for this step and may change the phase.

    def _settle(self, tip: np.ndarray, depth: float) -> np.ndarray:
        if self._phase_steps >= self.config["settle_steps"]:
            self._enter("approach")
        return tip

    def _approach(self, tip: np.ndarray, depth: float) -> np.ndarray:
        if np.linalg.norm(self._ready_at - tip) < self.config["position_tolerance_m"]:
            self._enter("press")
            self._pressed_steps = 0
        return self._ready_at

    def _press(self, tip: np.ndarray, depth: float) -> np.ndarray:
        # count consecutive steps measured pressed; a dip below starts the count again
        pressed = depth >= self.config["pressed_depth_m"]
        self._pressed_steps = self._pressed_steps + 1 if pressed else 0
        if self._pressed_steps >= self.config["hold_steps"]:
            self._enter("release")
        return self._depth_target(tip, depth)

    def _depth_target(self, tip: np.ndarray, depth: float) -> np.ndarray:
        """On the button axis, moved along it by the measured depth error, scaled so that
        ``depth_gain`` of it is commanded this step (before contact the depth is 0, so the tip
        advances at the press speed limit)."""
        c = self.config
        error = (c["depth_target_m"] - depth) * c["depth_gain"] / c["gains"]["position"]
        along = float(tip @ PRESS_AXIS) + error
        return self._face + PRESS_AXIS * (along - float(self._face @ PRESS_AXIS))

    def _release(self, tip: np.ndarray, depth: float) -> np.ndarray:
        if np.linalg.norm(self._ready_at - tip) < self.config["position_tolerance_m"]:
            self._enter("done")
        return self._ready_at

    def _done(self, tip: np.ndarray, depth: float) -> np.ndarray:
        return self._ready_at

    @property
    def phase(self) -> str:
        return self._phase
