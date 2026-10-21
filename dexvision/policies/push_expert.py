"""Closed-loop scripted push expert.

The four fingers bend 90 degrees at the knuckle into a wall facing along the fingers. The hand
turns so the wall faces the goal, comes down behind the object and pushes it, recomputing the
push direction from the measured object position every step. If the object slides off the wall
or the hand ends up beside or ahead of it, the expert lifts and approaches again.

It reads only the declared observation fields. No simulator access, no lookahead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dexvision import REPO_ROOT
from dexvision.policies.pick_expert import wrap
from dexvision.policies.reach_expert import rotation_vector
from dexvision.sim.workcell import ACTUATOR_TARGETS, ROTATION, TRANSLATION, ActionSpace
from dexvision.tasks.features import TaskContext, rotation_from_6d

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experts" / "push.yaml"


def yaw_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def support_distance(shape: str, size: np.ndarray, object_rotation: np.ndarray,
                     direction: np.ndarray) -> float:
    """How far the object extends from its centre along the horizontal unit ``direction``."""
    if shape == "cylinder":
        return float(size[0])
    local = object_rotation.T @ direction
    return float(abs(local[0]) * size[0] + abs(local[1]) * size[1])


@dataclass(frozen=True)
class _Measured:
    """What one observation implies for the push (world frame, m)."""

    grasp: np.ndarray  # [3]
    palm: np.ndarray  # [3, 3]
    direction: np.ndarray  # [3] horizontal unit vector, object -> goal
    distance: float  # object to goal, horizontal
    push_z: float  # grasp height while pushing
    high_z: float  # grasp height that clears the object
    behind: np.ndarray  # [3] start of the push, behind the object
    contact: np.ndarray  # [3] push target for this step
    along: float  # object ahead of the grasp point along the push direction
    lateral: float  # object off the push line
    unreachable: bool


class PushExpert:
    name = "expert"

    def __init__(self, action_space: ActionSpace, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._space = action_space
        names = action_space.actuator_names
        self._fingers = action_space.neutral_targets.copy()
        for name, value in self.config["wall_posture"].items():
            self._fingers[names.index(name)] = value
        self._fingers = np.clip(self._fingers, action_space.target_low, action_space.target_high)

    def reset(self, context: TaskContext, seed: int) -> None:
        self._context = context
        size = context.object_size
        self._height = 2 * (size[2] if context.object_shape == "box" else size[1])
        self._goal = np.asarray(context.goal_position, dtype=float)
        self._phase, self._phase_steps = "approach", 0
        self._initial_palm = None
        self._hold_at = None

    def _enter(self, phase: str) -> None:
        self._phase, self._phase_steps = phase, 0

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        palm = rotation_from_6d(observation["palm_rotation6d"])
        if self._initial_palm is None:
            self._initial_palm = palm
        m = self._measure(observation, palm)
        self._phase_steps += 1
        if m.unreachable and self._phase not in ("hold", "wait"):
            self._enter("wait")
        phase = self._phase
        target = getattr(self, f"_{phase}")(m)

        translation = self.config["gains"]["position"] * (target - m.grasp)
        if phase == "push":  # slow, so the object follows the wall
            norm = float(np.linalg.norm(translation))
            if norm > self.config["max_push_step_m"]:
                translation *= self.config["max_push_step_m"] / norm
        action = self._space.hold()
        action[TRANSLATION] = translation
        # while waiting, hold the current heading; otherwise face the push direction
        action[ROTATION] = self._rotation(palm, None if phase == "wait" else m.direction)
        action[ACTUATOR_TARGETS] = self._fingers
        return action

    def _measure(self, observation: Mapping[str, np.ndarray], palm: np.ndarray) -> _Measured:
        c = self.config
        grasp, obj = observation["grasp_position"], observation["object_position"]
        to_goal = self._goal[:2] - obj[:2]
        distance = float(np.linalg.norm(to_goal))
        direction = np.append(to_goal / max(distance, 1e-9), 0.0)
        reach = support_distance(self._context.object_shape, self._context.object_size,
                                 rotation_from_6d(observation["object_rotation6d"]), direction)
        bottom = obj[2] - self._height / 2
        contact = obj - direction * (reach + c["wall_offset_m"] - c["push_depth_m"])
        # Where the wall must be for the object to sit on the goal. The hand never goes past it,
        # so it decelerates into the goal instead of shoving the object through it.
        finish = self._goal - direction * (reach + c["wall_offset_m"])
        if float((contact - finish)[:2] @ direction[:2]) > 0:
            contact = finish
        relative = obj - grasp
        return _Measured(
            grasp=grasp, palm=palm, direction=direction, distance=distance,
            push_z=bottom + c["fingertip_drop_m"] + c["push_height_m"],
            high_z=bottom + self._height + c["fingertip_drop_m"] + c["clearance_m"],
            behind=obj - direction * (reach + c["wall_offset_m"] + c["standoff_m"]),
            contact=contact,
            along=float(relative[:2] @ direction[:2]),
            lateral=abs(float(relative[0] * direction[1] - relative[1] * direction[0])),
            # the goal needs a push toward the arm (near the goal the direction is noise)
            unreachable=(distance > c["resume_tolerance_m"]
                         and abs(math.atan2(direction[1], direction[0])) > c["max_yaw_rad"]),
        )

    # Each phase returns the grasp-point target for this step and may change the phase.

    def _wait(self, m: _Measured) -> np.ndarray:
        if not m.unreachable:
            self._enter("approach")
        return np.array([m.grasp[0], m.grasp[1], m.high_z])

    def _approach(self, m: _Measured) -> np.ndarray:
        target = np.array([m.behind[0], m.behind[1], m.high_z])
        if self._close(target, m.grasp, m.palm, m.direction):
            self._enter("descend")
        return target

    def _descend(self, m: _Measured) -> np.ndarray:
        target = np.array([m.behind[0], m.behind[1], m.push_z])
        if (np.linalg.norm(target - m.grasp) < self.config["position_tolerance_m"]
                or self._phase_steps > self.config["descend_timeout_steps"]):
            self._enter("push")
        return target

    def _push(self, m: _Measured) -> np.ndarray:
        c = self.config
        if m.distance < c["stop_tolerance_m"]:
            self._enter("hold")
            self._hold_at = m.grasp.copy()
        elif m.lateral > c["max_lateral_error_m"] or m.along < 0:
            self._enter("back_off")
        return np.array([m.contact[0], m.contact[1], m.push_z])

    def _hold(self, m: _Measured) -> np.ndarray:
        if m.distance > self.config["resume_tolerance_m"]:
            self._enter("back_off")
        return self._hold_at

    def _back_off(self, m: _Measured) -> np.ndarray:
        # Slide the wall straight back, away from the object, before rising: rising while still
        # pressed against it drags it up by friction and can topple tall objects.
        if self._phase_steps == 1:
            self._back_to = m.grasp - _finger_direction(m.palm) * self.config["back_off_m"]
        if (np.linalg.norm(self._back_to - m.grasp) < self.config["position_tolerance_m"]
                or self._phase_steps > 10):
            self._enter("retreat")
        return self._back_to

    def _retreat(self, m: _Measured) -> np.ndarray:  # rise straight up, then approach again
        risen = abs(m.high_z - m.grasp[2]) < self.config["position_tolerance_m"]
        if risen or self._phase_steps > 20:
            self._enter("approach")
        return np.array([m.grasp[0], m.grasp[1], m.high_z])

    def _close(self, target, grasp, palm, direction) -> bool:
        c = self.config
        yaw_error = wrap(math.atan2(direction[1], direction[0]) - _hand_yaw(palm))
        return (np.linalg.norm(target - grasp) < c["position_tolerance_m"]
                and abs(yaw_error) < c["yaw_tolerance_rad"])

    def _rotation(self, palm: np.ndarray, direction: np.ndarray | None) -> np.ndarray:
        """Turn about vertical so the fingers point along ``direction`` (None: keep the current
        heading), within the yaw limit; keep the palm level."""
        limit = self.config["max_yaw_rad"]
        yaw = _hand_yaw(palm) if direction is None else math.atan2(direction[1], direction[0])
        desired_yaw = min(max(yaw, -limit), limit)
        initial_yaw = _hand_yaw(self._initial_palm)
        target = yaw_rotation(desired_yaw - initial_yaw) @ self._initial_palm
        return self.config["gains"]["orientation"] * rotation_vector(target @ palm.T)

    @property
    def phase(self) -> str:
        return self._phase


def _finger_direction(palm: np.ndarray) -> np.ndarray:
    """Horizontal unit vector along the fingers."""
    yaw = _hand_yaw(palm)
    return np.array([math.cos(yaw), math.sin(yaw), 0.0])


def _hand_yaw(palm: np.ndarray) -> float:
    return math.atan2(palm[1, 2], palm[0, 2])  # fingers lie along the palm z-axis
