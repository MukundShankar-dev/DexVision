"""Closed-loop scripted reach expert.

It reads only the declared observation fields: it drives the measured grasp point toward the
goal and holds the palm at the orientation it started with. Fingers stay at their neutral
targets. No simulator access, no lookahead.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np
import yaml

from dexvision import REPO_ROOT
from dexvision.sim.workcell import ROTATION, TRANSLATION, ActionSpace
from dexvision.tasks.features import TaskContext, rotation_from_6d

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experts" / "reach.yaml"


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """[3, 3] rotation matrix -> [3] rotation vector (shortest rotation)."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix).reshape(9))
    if quat[0] < 0:
        quat = -quat
    rotvec = np.empty(3)
    mujoco.mju_quat2Vel(rotvec, quat, 1.0)
    return rotvec


class ReachExpert:
    name = "expert"

    def __init__(self, action_space: ActionSpace, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._space = action_space

    def reset(self, context: TaskContext, seed: int) -> None:
        self._goal = np.asarray(context.goal_position, dtype=float)
        self._initial_palm = None

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        gains = self.config["gains"]
        palm = rotation_from_6d(observation["palm_rotation6d"])
        if self._initial_palm is None:
            self._initial_palm = palm
        action = self._space.hold()
        action[TRANSLATION] = gains["position"] * (self._goal - observation["grasp_position"])
        # world-frame rotation that takes the current palm back to its starting orientation
        action[ROTATION] = gains["orientation"] * rotation_vector(self._initial_palm @ palm.T)
        return action
