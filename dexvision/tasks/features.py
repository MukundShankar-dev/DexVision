"""Named observation features. A task spec lists the names a policy may see.

Every feature is a flat float64 vector. ``object_*`` features describe the episode's target
object and are simulator ground truth (privileged in the state track).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from dexvision.sim.workcell import Observation

SHAPES = ("box", "cylinder")


@dataclass(frozen=True)
class TaskContext:
    """Per-episode facts a policy is allowed to know: the target and its observable geometry."""

    skill: str
    target_object: str
    object_shape: str
    object_size: np.ndarray  # [3], see Workcell.object_geometry


def rotation6d(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix, flattened [6]. Continuous, unlike quaternions."""
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=float))
    return matrix.reshape(3, 3)[:, :2].T.reshape(6)


def _target(observation: Observation, context: TaskContext):
    return observation.objects[context.target_object]


FEATURES: Mapping[str, Callable[[Observation, TaskContext], np.ndarray]] = {
    "palm_position": lambda o, c: o.palm_position,
    "palm_rotation6d": lambda o, c: rotation6d(o.palm_orientation),
    "hand_qpos": lambda o, c: o.hand_qpos,
    "hand_qvel": lambda o, c: o.hand_qvel,
    "fingertip_positions": lambda o, c: o.fingertip_positions.reshape(-1),
    "object_position": lambda o, c: _target(o, c).position,
    "object_rotation6d": lambda o, c: rotation6d(_target(o, c).orientation),
    "object_linear_velocity": lambda o, c: _target(o, c).linear_velocity,
    "object_angular_velocity": lambda o, c: _target(o, c).angular_velocity,
    "object_shape": lambda o, c: np.eye(len(SHAPES))[SHAPES.index(c.object_shape)],
    "object_size": lambda o, c: c.object_size,
}


def observation_dict(
    fields: Sequence[str], observation: Observation, context: TaskContext
) -> dict[str, np.ndarray]:
    """Build exactly the declared fields. Unknown names are an error, never silently skipped."""
    unknown = [name for name in fields if name not in FEATURES]
    if unknown:
        raise KeyError(f"Unknown observation features: {unknown}")
    return {
        name: np.asarray(FEATURES[name](observation, context), dtype=np.float64).copy()
        for name in fields
    }
