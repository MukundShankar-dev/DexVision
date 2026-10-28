"""Closed-loop scripted place expert.

It is the pick expert (configs/experts/pick.yaml, unchanged) up to a lift, followed by its own
phases (configs/experts/place.yaml):

- carry: at lift height until the object is over the goal;
- lower: until the object is just above its resting height;
- release: the thumb backs off first, then the fingers open (opening both at once lets the thumb
  sweep a light object away);
- slide_back: along the fingers, so the thumb leaves the object's near side;
- retreat: rise straight up.

Carry and lower steer the hand by the measured object position, so the object, not the hand, is
brought to the goal. If the object falls out of the hand while carried, or is not on the goal
after the retreat, the expert picks it up and places it again.

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
from dexvision.policies.pick_expert import GRASP_POINT_IN_PALM, PickExpert
from dexvision.sim.workcell import ActionSpace
from dexvision.tasks.features import TaskContext, rotation_from_6d

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experts" / "place.yaml"


@dataclass(frozen=True)
class _Measured:
    """What one observation implies for the place phases (world frame, m)."""

    grasp: np.ndarray  # [3] grasp point
    hand_yaw: float  # rad, finger direction
    object_z: float  # object centre height
    to_goal: np.ndarray  # [3] horizontal object -> goal, z = 0


@dataclass(frozen=True)
class _Command:
    target: np.ndarray  # [3] grasp-point target
    step_limit: float  # m per step
    fingers: np.ndarray  # [20] actuator targets


class PlaceExpert(PickExpert):
    name = "expert"

    def __init__(self, action_space: ActionSpace, config_path: Path = DEFAULT_CONFIG) -> None:
        super().__init__(action_space)
        self.place = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._thumb = np.array([n.startswith("TH") for n in action_space.actuator_names])

    def reset(self, context: TaskContext, seed: int) -> None:
        super().reset(context, seed)
        self._goal = np.asarray(context.goal_position, dtype=float)
        self._rest_z = None  # the object's centre height on the table, measured at the start
        self._placing = False  # past the pick: the phases are this expert's own

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        obj = observation["object_position"]
        if self._rest_z is None:
            self._rest_z = float(obj[2])
        if not self._placing:
            if self._phase == "lift" and obj[2] - self._rest_z >= self.place["carry_rise_m"]:
                self._enter("carry")
                self._placing = True
            else:
                return super().act(observation)

        palm = rotation_from_6d(observation["palm_rotation6d"])
        m = _Measured(
            grasp=observation["palm_position"] + palm @ GRASP_POINT_IN_PALM,
            hand_yaw=math.atan2(palm[1, 2], palm[0, 2]),
            object_z=float(obj[2]),
            to_goal=np.array([self._goal[0] - obj[0], self._goal[1] - obj[1], 0.0]),
        )
        self._phase_steps += 1
        command = getattr(self, f"_{self._phase}")(m)
        if not self._placing:  # the phase restarted the pick
            return super().act(observation)
        error = command.target - m.grasp
        norm = float(np.linalg.norm(error)) * self.config["gains"]["position"]
        if norm > command.step_limit:
            error *= command.step_limit / norm
        return self._action(error, 0.0, palm, m.hand_yaw, command.fingers)  # hold the yaw

    # Each phase returns this step's command and may change the phase.

    def _carry(self, m: _Measured) -> _Command:
        p = self.place
        if m.object_z - self._rest_z < p["dropped_below_m"]:
            self._restart()  # it fell out of the hand
        elif np.linalg.norm(m.to_goal) < p["over_goal_tolerance_m"]:
            self._enter("lower")
        target = m.grasp + m.to_goal
        target[2] = self._hold_at[2] + self.config["lift_m"]
        return _Command(target, p["max_carry_step_m"], self._closed)

    def _lower(self, m: _Measured) -> _Command:
        p = self.place
        gap = (self._rest_z + p["set_down_gap_m"]) - m.object_z
        if gap >= 0 or self._phase_steps > p["lower_timeout_steps"]:
            self._enter("release")
            self._release_at = m.grasp.copy()
        return _Command(m.grasp + m.to_goal + [0, 0, gap], p["max_lower_step_m"], self._closed)

    def _release(self, m: _Measured) -> _Command:
        p = self.place
        fingers = self._plan["open"].copy()
        if self._phase_steps <= p["thumb_release_steps"]:
            fingers[~self._thumb] = self._closed[~self._thumb]
        if self._phase_steps > p["thumb_release_steps"] + p["finger_release_steps"]:
            self._enter("slide_back")
            along = np.array([math.cos(m.hand_yaw), math.sin(m.hand_yaw), 0.0])
            self._back_at = self._release_at - along * p["slide_back_m"]
        return _Command(self._release_at, p["max_lower_step_m"], fingers)

    def _slide_back(self, m: _Measured) -> _Command:
        p = self.place
        arrived = np.linalg.norm(self._back_at - m.grasp) < p["over_goal_tolerance_m"]
        if arrived or self._phase_steps > p["move_timeout_steps"]:
            self._enter("retreat")
        return _Command(self._back_at, p["max_lower_step_m"], self._plan["open"])

    def _retreat(self, m: _Measured) -> _Command:
        p = self.place
        target = self._back_at + [0, 0, p["retreat_m"]]
        risen = (np.linalg.norm(target - m.grasp) < p["over_goal_tolerance_m"]
                 or self._phase_steps > 2 * p["move_timeout_steps"])
        if risen and np.linalg.norm(m.to_goal) > p["replace_tolerance_m"]:
            self._restart()  # it was knocked off the goal while letting go
        return _Command(target, p["max_lower_step_m"], self._plan["open"])

    @property
    def _closed(self) -> np.ndarray:
        return self._plan["close"][-1][1]

    def _restart(self) -> None:
        self._enter("approach")
        self._placing = False
        self._attempt = 1
