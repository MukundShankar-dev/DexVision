"""Closed-loop scripted pick expert.

It reads only the observation fields declared in ``configs/tasks/pick.yaml``, the same ones a
learned policy gets. There is no simulator access, no lookahead and no contact sensing. Its
only private state is the phase and the attempt count, and phase changes depend on measured
quantities. Parameters live in ``configs/experts/pick.yaml``.

Phases: approach (above the object, yaw aligned) -> descend -> close -> lift.
If the object does not rise with the hand, the expert opens, backs off and tries again.
The grasp (offset, open posture and staged close) is chosen from the object's height alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from dexvision import REPO_ROOT
from dexvision.sim.workcell import ACTUATOR_TARGETS, ROTATION, TRANSLATION, ActionSpace
from dexvision.tasks.features import TaskContext

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experts" / "pick.yaml"
GRASP_POINT_IN_PALM = np.array([0.0, -0.035, 0.09])  # the menagerie grasp_site, palm frame


def rotation_from_6d(r6: np.ndarray) -> np.ndarray:
    """[6] -> [3, 3] rotation matrix (inverse of features.rotation6d)."""
    c1, c2 = np.asarray(r6[:3]), np.asarray(r6[3:6])
    return np.column_stack([c1, c2, np.cross(c1, c2)])


def wrap(angle: float, period: float = 2 * math.pi) -> float:
    """Wrap into (-period/2, period/2]."""
    return -((-angle + period / 2) % period - period / 2)


def grasp_yaw(object_rotation: np.ndarray, shape: str, size: np.ndarray) -> float:
    """Hand yaw that puts the thumb-finger axis across the object's narrowest horizontal side.

    With the palm down, the thumb tucks under the palm behind the object and the curling fingers
    close in front of it, so the opposition axis lies along the fingers (the hand's yaw).
    Symmetry decides the period: cylinders need no yaw, cubes repeat every 90°, and rectangular
    blocks every 180°. The smallest equivalent rotation is chosen.
    """
    if shape == "cylinder":
        return 0.0
    object_yaw = math.atan2(object_rotation[1, 0], object_rotation[0, 0])
    if abs(size[0] - size[1]) < 1e-3:
        return wrap(object_yaw, math.pi / 2)
    short_axis_yaw = object_yaw if size[0] < size[1] else object_yaw + math.pi / 2
    return wrap(short_axis_yaw, math.pi)


class PickExpert:
    name = "expert"

    def __init__(self, action_space: ActionSpace, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._space = action_space
        self._grasps = {name: self._grasp(g) for name, g in self.config["grasps"].items()}

    def _grasp(self, config: Mapping) -> dict:
        """Resolve one grasp: offset [3], open targets [20], close stages [(end_step, [20])]."""
        targets = self._space.neutral_targets.copy()
        open_targets = self._update(targets, config["open"])
        stages, end, current = [], 0, open_targets
        for stage in config["close"]:
            current = self._update(current, stage["targets"])
            end += int(stage["steps"])
            stages.append((end, current))
        anchor = config.get("anchor", "center")
        if anchor not in ("center", "near_edge_bottom"):
            raise ValueError(f"Unknown grasp anchor '{anchor}'")
        return {"offset": np.asarray(config["offset_m"], dtype=float), "anchor": anchor,
                "open": open_targets, "close": stages}

    def _update(self, targets: np.ndarray, values: Mapping[str, float]) -> np.ndarray:
        names = self._space.actuator_names
        unknown = set(values) - set(names)
        if unknown:
            raise KeyError(f"Unknown actuators in expert posture: {sorted(unknown)}")
        targets = targets.copy()
        for name, value in values.items():
            targets[names.index(name)] = value
        return np.clip(targets, self._space.target_low, self._space.target_high)

    def reset(self, context: TaskContext, seed: int) -> None:
        self._context = context
        size = context.object_size
        if context.object_shape == "box":
            # grasp_yaw puts the fingers along the shorter horizontal side
            self._half_height, self._half_depth = size[2], min(size[0], size[1])
        else:
            self._half_height, self._half_depth = size[1], size[0]
        flat = 2 * self._half_height <= self.config["flat_max_height_m"]
        self._plan = self._grasps["flat" if flat else "default"]
        self._phase = "approach"
        self._phase_steps = 0
        self._attempt = 1
        self._initial_palm_up = None
        self._lift_start = None

    # ------------------------------------------------------------------ control

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        c = self.config
        palm = rotation_from_6d(observation["palm_rotation6d"])
        grasp = observation["palm_position"] + palm @ GRASP_POINT_IN_PALM
        hand_yaw = math.atan2(palm[1, 2], palm[0, 2])  # fingers lie along the palm z-axis
        if self._initial_palm_up is None:
            self._initial_palm_up = _yaw_matrix(-hand_yaw) @ palm[:, 1]
        obj = observation["object_position"]
        yaw = grasp_yaw(rotation_from_6d(observation["object_rotation6d"]),
                        self._context.object_shape, self._context.object_size)
        plan = self._plan
        anchor = np.zeros(3)  # in the hand's yaw frame, relative to the object centre
        if plan["anchor"] == "near_edge_bottom":
            anchor = np.array([-self._half_depth, 0.0, -self._half_height])
        at_grasp = obj + _yaw_matrix(yaw) @ (anchor + plan["offset"])
        above = at_grasp + [0, 0, self._half_height + c["approach_clearance_m"]]

        self._phase_steps += 1
        fingers = plan["open"]
        close_end, closed = plan["close"][-1]
        if self._phase == "approach":
            target = above
            close_enough = (np.linalg.norm(target - grasp) < c["position_tolerance_m"]
                            and abs(wrap(yaw - hand_yaw)) < c["yaw_tolerance_rad"])
            if close_enough:
                self._enter("descend")
        elif self._phase == "descend":
            target = at_grasp
            if (np.linalg.norm(target - grasp) < c["position_tolerance_m"]
                    or self._phase_steps > c["descend_timeout_steps"]):
                self._enter("close")
                self._hold_at = grasp.copy()
        elif self._phase == "close":
            target = self._hold_at
            fingers = next((t for end, t in plan["close"] if self._phase_steps <= end), closed)
            if self._phase_steps > close_end:
                self._enter("lift")
                self._lift_start = (grasp[2], obj[2])
        elif self._phase == "lift":
            target = self._hold_at + [0, 0, c["lift_m"]]
            fingers = closed
            hand_rise, object_rise = grasp[2] - self._lift_start[0], obj[2] - self._lift_start[1]
            slipped = hand_rise > c["lift_check_m"] and object_rise < hand_rise / 2
            if slipped and self._attempt < c["max_attempts"]:
                self._attempt += 1
                self._enter("back_off")
        else:  # back_off: open and rise, then approach again
            target = above
            if self._phase_steps > 10:
                self._enter("approach")

        return self._action(target - grasp, yaw - hand_yaw, palm, hand_yaw, fingers)

    def _enter(self, phase: str) -> None:
        self._phase, self._phase_steps = phase, 0

    def _action(self, position_error, yaw_error, palm, hand_yaw, fingers) -> np.ndarray:
        gains = self.config["gains"]
        action = np.zeros(26)
        action[TRANSLATION] = gains["position"] * position_error
        desired_up = _yaw_matrix(hand_yaw) @ self._initial_palm_up
        level = np.cross(palm[:, 1], desired_up)
        level[2] = 0.0  # leveling only tilts; yaw is controlled separately
        action[ROTATION] = gains["level"] * level + [0, 0, gains["yaw"] * wrap(yaw_error)]
        action[ACTUATOR_TARGETS] = fingers
        return action

    @property
    def phase(self) -> str:
        return self._phase


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
