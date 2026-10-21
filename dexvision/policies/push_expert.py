"""Closed-loop scripted push expert.

The hand pushes with one of three flat faces (configs/experts/push.yaml): a wall of fingers bent
90 degrees at the knuckle that pushes along the fingers, or, with the thumb up, the palm (pushes
left of the fingers) or the back of the hand (pushes right). Each approach picks the face that
needs the least hand turn to face the object -> goal direction. The hand comes down behind the
object and pushes it, re-aiming every step from the measured object position. If the object
slides off the face or the hand ends up beside or ahead of it, the expert backs off, lifts and
approaches again.

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


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """[3] axis, angle (rad) -> [3, 3] rotation about the axis (Rodrigues)."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def horizontal(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw), 0.0])


def support_distance(shape: str, size: np.ndarray, object_rotation: np.ndarray,
                     direction: np.ndarray) -> float:
    """How far the object extends from its centre along the horizontal unit ``direction``."""
    if shape == "cylinder":
        return float(size[0])
    local = object_rotation.T @ direction
    return float(abs(local[0]) * size[0] + abs(local[1]) * size[1])


@dataclass(frozen=True)
class Pusher:
    """A hand posture with a flat pushing face. See ``pushers`` in the expert config."""

    name: str
    fingers: np.ndarray  # [20] actuator targets
    roll_rad: float
    pitch_rad: float
    face_yaw_rad: float
    face_offset_m: float
    along_fingers_m: float
    drop_m: float

    def finger_yaw(self, push_yaw: float) -> float:
        """Hand yaw (finger direction) that makes the face push along ``push_yaw``."""
        return wrap(push_yaw - self.face_yaw_rad)


@dataclass(frozen=True)
class _Measured:
    """What one observation implies for the push with the current pusher (world frame, m)."""

    grasp: np.ndarray  # [3]
    palm: np.ndarray  # [3, 3]
    direction: np.ndarray  # [3] horizontal unit vector, object -> goal
    finger_yaw: float  # hand yaw that faces ``direction``, within the limit
    distance: float  # object to goal, horizontal
    push_z: float  # grasp height while pushing
    high_z: float  # grasp height that clears the object
    behind: np.ndarray  # [3] grasp position at the start of the push
    contact: np.ndarray  # [3] grasp target for this push step
    along: float  # object ahead of the face centre along the push direction
    lateral: float  # object off the push line through the face centre
    unreachable: bool  # this pusher cannot face the push direction within the yaw limit


class PushExpert:
    name = "expert"

    def __init__(self, action_space: ActionSpace, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._space = action_space
        names = action_space.actuator_names
        self.pushers = {}
        for name, spec in self.config["pushers"].items():
            fingers = action_space.neutral_targets.copy()
            for joint, value in spec["posture"].items():
                fingers[names.index(joint)] = value
            fingers = np.clip(fingers, action_space.target_low, action_space.target_high)
            self.pushers[name] = Pusher(name, fingers, **{k: float(v) for k, v in spec.items()
                                                          if k != "posture"})

    def reset(self, context: TaskContext, seed: int) -> None:
        self._context = context
        size = context.object_size
        self._height = 2 * (size[2] if context.object_shape == "box" else size[1])
        self._goal = np.asarray(context.goal_position, dtype=float)
        self._phase, self._phase_steps = "approach", 0
        self._pusher = None
        self._bases = None
        self._hold_at = None

    def _enter(self, phase: str) -> None:
        self._phase, self._phase_steps = phase, 0

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        palm = rotation_from_6d(observation["palm_rotation6d"])
        if self._bases is None:
            self._bases = self._pusher_bases(palm)
        if self._phase == "approach" and self._phase_steps == 0:  # each approach chooses again
            self._pusher = self._choose(observation)
        m = self._measure(observation, palm)
        if m.unreachable and self._phase == "approach":  # e.g. the object overshot: switch faces
            self._pusher = self._choose(observation)
            m = self._measure(observation, palm)
        elif m.unreachable and self._phase in ("descend", "push"):
            self._enter("back_off")  # and approach again, which chooses again
        self._phase_steps += 1
        phase = self._phase
        target = getattr(self, f"_{phase}")(m)

        translation = self.config["gains"]["position"] * (target - m.grasp)
        if phase == "push":  # slow, so the object follows the face
            norm = float(np.linalg.norm(translation))
            if norm > self.config["max_push_step_m"]:
                translation *= self.config["max_push_step_m"] / norm
        action = self._space.hold()
        action[TRANSLATION] = translation
        # keep the heading while backing away; otherwise face the push direction
        yaw = _hand_yaw(palm) if phase in ("back_off", "retreat") else m.finger_yaw
        error = self._orientation(yaw) @ palm.T
        action[ROTATION] = self.config["gains"]["orientation"] * rotation_vector(error)
        action[ACTUATOR_TARGETS] = self._pusher.fingers
        return action

    def _pusher_bases(self, palm: np.ndarray) -> dict[str, np.ndarray]:
        """Each pusher's palm orientation at the initial hand yaw: the initial palm rolled about
        the fingers, then pitched about the horizontal axis across them."""
        fingers = palm[:, 2]
        across = np.cross([0.0, 0.0, 1.0], fingers)
        return {name: axis_rotation(across, p.pitch_rad) @ axis_rotation(fingers, p.roll_rad) @ palm
                for name, p in self.pushers.items()}

    def _orientation(self, finger_yaw: float) -> np.ndarray:
        """[3, 3] palm orientation of the current pusher with the fingers at ``finger_yaw``."""
        base = self._bases[self._pusher.name]
        limit = self.config["max_yaw_rad"]
        yaw = min(max(finger_yaw, -limit), limit)
        return axis_rotation([0.0, 0.0, 1.0], yaw - _hand_yaw(base)) @ base

    def _choose(self, observation: Mapping[str, np.ndarray]) -> Pusher:
        """The pusher that needs the least hand turn for the current object -> goal direction."""
        to_goal = self._goal[:2] - observation["object_position"][:2]
        push_yaw = math.atan2(to_goal[1], to_goal[0])
        return min(self.pushers.values(), key=lambda p: abs(p.finger_yaw(push_yaw)))

    def _measure(self, observation: Mapping[str, np.ndarray], palm: np.ndarray) -> _Measured:
        c, p = self.config, self._pusher
        grasp, obj = observation["grasp_position"], observation["object_position"]
        to_goal = self._goal[:2] - obj[:2]
        distance = float(np.linalg.norm(to_goal))
        push_yaw = math.atan2(to_goal[1], to_goal[0])
        direction = horizontal(push_yaw)
        finger_yaw = p.finger_yaw(push_yaw)
        along_fingers = p.along_fingers_m * horizontal(finger_yaw)
        reach = support_distance(self._context.object_shape, self._context.object_size,
                                 rotation_from_6d(observation["object_rotation6d"]), direction)

        def grasp_for(centre: np.ndarray, gap: float) -> np.ndarray:
            """Grasp position that puts the face ``gap`` behind the edge of an object at centre."""
            return centre - direction * (reach + p.face_offset_m + gap) - along_fingers

        bottom = obj[2] - self._height / 2
        contact = grasp_for(obj, -c["push_depth_m"])
        # Where the face must be for the object to sit on the goal. The hand never goes past it,
        # so it decelerates into the goal instead of shoving the object through it.
        finish = grasp_for(self._goal, 0.0)
        if float((contact - finish)[:2] @ direction[:2]) > 0:
            contact = finish
        # the object relative to the centre of the face, along and across the push direction
        relative = obj - (grasp + along_fingers + direction * p.face_offset_m)
        return _Measured(
            grasp=grasp, palm=palm, direction=direction, finger_yaw=finger_yaw, distance=distance,
            push_z=bottom + p.drop_m + c["push_height_m"],
            high_z=bottom + self._height + p.drop_m + c["clearance_m"],
            behind=grasp_for(obj, c["standoff_m"]),
            contact=contact,
            along=float(relative[:2] @ direction[:2]),
            lateral=abs(float(relative[0] * direction[1] - relative[1] * direction[0])),
            # near the goal the direction is noise
            unreachable=(distance > c["resume_tolerance_m"]
                         and abs(finger_yaw) > c["max_yaw_rad"]),
        )

    # Each phase returns the grasp-point target for this step and may change the phase.

    def _approach(self, m: _Measured) -> np.ndarray:
        target = np.array([m.behind[0], m.behind[1], m.high_z])
        turned = rotation_angle(self._orientation(m.finger_yaw) @ m.palm.T)
        if (np.linalg.norm(target - m.grasp) < self.config["position_tolerance_m"]
                and turned < self.config["rotation_tolerance_rad"]):
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
        # Slide the face straight back, away from the object, before rising: rising while still
        # pressed against it drags it up by friction and can topple tall objects.
        if self._phase_steps == 1:
            face = horizontal(_hand_yaw(m.palm) + self._pusher.face_yaw_rad)
            self._back_to = m.grasp - face * self.config["back_off_m"]
        if (np.linalg.norm(self._back_to - m.grasp) < self.config["position_tolerance_m"]
                or self._phase_steps > 10):
            self._enter("retreat")
        return self._back_to

    def _retreat(self, m: _Measured) -> np.ndarray:  # rise straight up, then approach again
        risen = abs(m.high_z - m.grasp[2]) < self.config["position_tolerance_m"]
        if risen or self._phase_steps > 20:
            self._enter("approach")
        return np.array([m.grasp[0], m.grasp[1], m.high_z])

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def pusher(self) -> str | None:
        return self._pusher.name if self._pusher else None


def rotation_angle(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(rotation_vector(matrix)))


def _hand_yaw(palm: np.ndarray) -> float:
    return math.atan2(palm[1, 2], palm[0, 2])  # fingers lie along the palm z-axis
