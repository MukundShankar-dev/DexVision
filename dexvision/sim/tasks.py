"""Resettable MuJoCo task specifications for DexVision Level 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.dataset_schema import ActionSchema, ObservationSchema
from dexvision.logging.demo_logger import (
    build_level1_action_schema,
    build_level2_observation_schema,
)
from dexvision.sim.mujoco_env import MujocoEnv, MujocoState


DEFAULT_TASK_BOARD_MODEL = Path("assets/mujoco/task_board_scene.xml")
REACH_TOUCH_TARGET_TASK_ID = "reach_touch_target"
BUTTON_PRESS_TASK_ID = "button_press"
PUSH_CUBE_TASK_ID = "push_cube_to_target"
ACTIVE_REACH_TARGET_GEOM = "active_reach_target"
BUTTON_TARGET_RGBA = (0.1, 1.0, 0.2, 1.0)
BUTTON_NON_TARGET_RGBA = (0.22, 0.22, 0.22, 1.0)
DEFAULT_TRACKING_QUALITY_NAMES = (
    "detected",
    "handedness",
    "tracking_confidence",
    "feature_confidence",
    "dropped_frame",
    "reacquired",
)


class TaskError(RuntimeError):
    """Raised when a task specification, reset, or state is invalid."""


@dataclass(frozen=True)
class ReachTouchTargetParameters:
    """Typed goal for a reach-touch episode.

    Provide either a world-frame ``target_pose`` position or a named MuJoCo
    target site. Leaving both unset asks reset() to sample a configured site.
    """

    target_pose: tuple[float, float, float] | None = None
    target_site: str | None = None

    def __post_init__(self) -> None:
        if self.target_pose is not None and self.target_site is not None:
            raise ValueError("target_pose and target_site are mutually exclusive.")
        if self.target_site is not None and not self.target_site:
            raise ValueError("target_site must be a non-empty string.")
        if self.target_pose is not None:
            pose = np.asarray(self.target_pose, dtype=np.float64)
            if pose.shape != (3,) or not np.all(np.isfinite(pose)):
                raise ValueError("target_pose must contain three finite world-frame values.")
            object.__setattr__(self, "target_pose", tuple(float(value) for value in pose))


@dataclass(frozen=True)
class ReachTouchTargetConfig:
    """Reset and metric configuration for the first Level 2 task."""

    target_sites: tuple[str, ...] = (
        "reach_target_left",
        "reach_target_center",
        "reach_target_right",
    )
    touch_site: str = "grasp_site"
    target_marker_body: str = "reach_target_marker"
    target_marker_geom: str = "active_reach_target"
    palm_body: str = "rh_palm"
    base_target_body: str = "dexvision_hand_base_target"
    success_distance_m: float = 0.03
    success_dwell_steps: int = 5
    max_episode_steps: int = 240
    terminate_on_workspace_bounds: bool = False
    # Keep a practical approach margin outside every configured target's
    # 3 cm success region. Exact success-boundary limits caused legitimate
    # left-target approaches to terminate on sub-millimetre tracking drift.
    workspace_min: tuple[float, float, float] = (-0.18, -0.20, 0.37)
    workspace_max: tuple[float, float, float] = (0.26, 0.18, 0.61)

    def __post_init__(self) -> None:
        if not self.target_sites or any(not name for name in self.target_sites):
            raise ValueError("target_sites must contain non-empty MuJoCo site names.")
        if len(set(self.target_sites)) != len(self.target_sites):
            raise ValueError("target_sites must not contain duplicates.")
        if (
            not self.touch_site
            or not self.target_marker_body
            or not self.target_marker_geom
            or not self.palm_body
            or not self.base_target_body
        ):
            raise ValueError("task body and site names must be non-empty.")
        if self.success_distance_m <= 0.0:
            raise ValueError("success_distance_m must be positive.")
        if self.success_dwell_steps <= 0:
            raise ValueError("success_dwell_steps must be positive.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        workspace_min = _finite_vector(self.workspace_min, name="workspace_min")
        workspace_max = _finite_vector(self.workspace_max, name="workspace_max")
        if np.any(workspace_min >= workspace_max):
            raise ValueError("workspace_min must be strictly below workspace_max.")


@dataclass(frozen=True)
class ButtonPressParameters:
    """Typed goal for one button-press episode.

    ``button_id`` may select a configured button explicitly. When it is omitted,
    reset samples a button deterministically from ``seed``. A goal may provide
    either an exact press-depth threshold or a boolean pressed-state target.
    """

    button_id: str | None = None
    target_press_depth: float | None = None
    pressed_state_target: bool | None = None
    approach_pose: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.button_id is not None and not self.button_id:
            raise ValueError("button_id must be a non-empty string.")
        if (
            self.target_press_depth is not None
            and self.pressed_state_target is not None
        ):
            raise ValueError(
                "target_press_depth and pressed_state_target are mutually exclusive."
            )
        if self.target_press_depth is not None:
            depth = float(self.target_press_depth)
            if not np.isfinite(depth) or depth <= 0.0:
                raise ValueError("target_press_depth must be finite and positive.")
            object.__setattr__(self, "target_press_depth", depth)
        if self.pressed_state_target is not None and not isinstance(
            self.pressed_state_target, bool
        ):
            raise ValueError("pressed_state_target must be a boolean.")
        if self.approach_pose is not None:
            pose = _finite_vector(self.approach_pose, name="approach_pose")
            object.__setattr__(
                self,
                "approach_pose",
                tuple(float(value) for value in pose),
            )


@dataclass(frozen=True)
class ButtonPressConfig:
    """Reset, geometry, and metric configuration for ``button_press``."""

    button_ids: tuple[str, ...] = (
        "button_left",
        "button_center",
        "button_right",
    )
    base_target_body: str = "dexvision_hand_base_target"
    default_target_press_depth_m: float = 0.012
    success_dwell_steps: int = 3
    max_episode_steps: int = 240

    def __post_init__(self) -> None:
        if not self.button_ids or any(not name for name in self.button_ids):
            raise ValueError("button_ids must contain non-empty MuJoCo body names.")
        if len(set(self.button_ids)) != len(self.button_ids):
            raise ValueError("button_ids must not contain duplicates.")
        if not self.base_target_body:
            raise ValueError("base_target_body must be non-empty.")
        if (
            not np.isfinite(self.default_target_press_depth_m)
            or self.default_target_press_depth_m <= 0.0
        ):
            raise ValueError("default_target_press_depth_m must be finite and positive.")
        if self.success_dwell_steps <= 0:
            raise ValueError("success_dwell_steps must be positive.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")


@dataclass(frozen=True)
class PushCubeParameters:
    """Typed goal for one cube-push episode.

    ``object_id`` may select a configured movable cube. The target is either a
    world-frame cube-centre position or a named target zone. Leaving both unset
    asks reset() to sample a configured zone deterministically from ``seed``.
    """

    object_id: str | None = None
    target_pose: tuple[float, float, float] | None = None
    target_zone_id: str | None = None
    approach_side: str | None = None

    def __post_init__(self) -> None:
        if self.object_id is not None and not self.object_id:
            raise ValueError("object_id must be a non-empty string.")
        if self.target_pose is not None and self.target_zone_id is not None:
            raise ValueError("target_pose and target_zone_id are mutually exclusive.")
        if self.target_zone_id is not None and not self.target_zone_id:
            raise ValueError("target_zone_id must be a non-empty string.")
        if self.target_pose is not None:
            pose = _finite_vector(self.target_pose, name="target_pose")
            object.__setattr__(
                self,
                "target_pose",
                tuple(float(value) for value in pose),
            )
        if self.approach_side is not None and not self.approach_side:
            raise ValueError("approach_side must be a non-empty string.")


@dataclass(frozen=True)
class PushCubeConfig:
    """Reset, geometry, and metric configuration for ``push_cube_to_target``."""

    object_ids: tuple[str, ...] = ("push_cube",)
    object_start_sites: tuple[str, ...] = (
        "push_cube_start_left",
        "push_cube_start_center",
        "push_cube_start_right",
    )
    target_zone_sites: tuple[str, ...] = (
        "push_target_left",
        "push_target_center",
        "push_target_right",
    )
    approach_sides: tuple[str, ...] = ("left", "right", "front", "back")
    target_marker_body: str = "push_cube_target_marker"
    target_marker_geom: str = "push_cube_target_geom"
    base_target_body: str = "dexvision_hand_base_target"
    base_free_joint: str = "rh_base_freejoint"
    initial_base_x: float = -0.18
    initial_base_z: float = -0.24
    initial_base_orientation: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    target_radius_m: float = 0.035
    success_dwell_steps: int = 5
    max_episode_steps: int = 300
    workspace_min: tuple[float, float, float] = (-0.18, -0.15, -0.05)
    workspace_max: tuple[float, float, float] = (0.18, 0.15, 0.08)

    def __post_init__(self) -> None:
        for field_name, values in (
            ("object_ids", self.object_ids),
            ("object_start_sites", self.object_start_sites),
            ("target_zone_sites", self.target_zone_sites),
            ("approach_sides", self.approach_sides),
        ):
            if not values or any(not value for value in values):
                raise ValueError(f"{field_name} must contain non-empty names.")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must not contain duplicates.")
        if (
            not self.target_marker_body
            or not self.target_marker_geom
            or not self.base_target_body
            or not self.base_free_joint
        ):
            raise ValueError("push-cube body and geom names must be non-empty.")
        if not np.isfinite(self.initial_base_x) or not np.isfinite(
            self.initial_base_z
        ):
            raise ValueError("push-cube initial base position must be finite.")
        initial_orientation = np.asarray(
            self.initial_base_orientation,
            dtype=np.float64,
        )
        if (
            initial_orientation.shape != (4,)
            or not np.all(np.isfinite(initial_orientation))
            or np.linalg.norm(initial_orientation) <= 1e-12
        ):
            raise ValueError(
                "push-cube initial base orientation must be a finite non-zero "
                "wxyz quaternion."
            )
        if not np.isfinite(self.target_radius_m) or self.target_radius_m <= 0.0:
            raise ValueError("target_radius_m must be finite and positive.")
        if self.success_dwell_steps <= 0:
            raise ValueError("success_dwell_steps must be positive.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        workspace_min = _finite_vector(self.workspace_min, name="workspace_min")
        workspace_max = _finite_vector(self.workspace_max, name="workspace_max")
        if np.any(workspace_min >= workspace_max):
            raise ValueError("workspace_min must be strictly below workspace_max.")


@dataclass(frozen=True)
class TaskSpec:
    """Static contract for one resettable Level 2 MuJoCo task."""

    task_id: str
    skill_name: str
    required_objects: tuple[str, ...]
    observation_schema: ObservationSchema
    action_schema: ActionSchema
    success_condition: str
    failure_conditions: tuple[str, ...]
    max_episode_steps: int
    reset_config: dict[str, Any]
    parameter_type: type[Any]
    parameter_schema: Mapping[str, Mapping[str, Any]]
    state_fields: tuple[str, ...]
    success_metric_inputs: tuple[str, ...]
    terminal_state_schema: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ReachTouchTargetState:
    """Reconstructable state and metric inputs for one reach-touch timestep."""

    target_source: str
    target_index: int
    target_position: np.ndarray
    touch_position: np.ndarray
    distance_to_target: float
    palm_contact: bool
    within_success_distance: bool
    dwell_steps: int
    success: bool
    failure_reason: str | None
    step_count: int
    initial_base_position: np.ndarray
    initial_base_orientation: np.ndarray
    initial_robot_qpos: np.ndarray
    initial_robot_qvel: np.ndarray

    def as_task_state(self) -> np.ndarray:
        """Pack state, success inputs, sampled target, and initial state."""

        return np.concatenate(
            (
                self.target_position,
                self.touch_position,
                np.asarray(
                    [
                        self.distance_to_target,
                        float(self.palm_contact),
                        float(self.within_success_distance),
                        float(self.dwell_steps),
                        float(self.success),
                        float(self.failure_reason is not None),
                        float(self.step_count),
                        float(self.target_index),
                    ],
                    dtype=np.float64,
                ),
                self.initial_base_position,
                self.initial_base_orientation,
                self.initial_robot_qpos,
                self.initial_robot_qvel,
            )
        )


@dataclass(frozen=True)
class ButtonPressState:
    """Reconstructable button state and terminal metric inputs."""

    button_id: str
    button_index: int
    button_position: np.ndarray
    approach_pose: np.ndarray
    approach_pose_present: bool
    press_depth: float
    target_press_depth: float
    button_pressed: bool
    target_pressed_state: bool
    within_target: bool
    dwell_steps: int
    success: bool
    failure_reason: str | None
    step_count: int
    initial_button_depth: float
    initial_base_position: np.ndarray
    initial_base_orientation: np.ndarray
    initial_robot_qpos: np.ndarray
    initial_robot_qvel: np.ndarray

    def as_object_state(self) -> np.ndarray:
        """Pack current button depth and world position for episode logging."""

        return np.concatenate(
            (
                np.asarray([self.press_depth], dtype=np.float64),
                self.button_position,
            )
        )

    def as_task_state(self) -> np.ndarray:
        """Pack button metrics, goal parameters, and deterministic initial state."""

        return np.concatenate(
            (
                np.asarray(
                    [
                        self.press_depth,
                        self.target_press_depth,
                        float(self.button_pressed),
                        float(self.target_pressed_state),
                        float(self.dwell_steps),
                    ],
                    dtype=np.float64,
                ),
                self.button_position,
                np.asarray(
                    [
                        float(self.button_index),
                        float(self.approach_pose_present),
                    ],
                    dtype=np.float64,
                ),
                self.approach_pose,
                np.asarray(
                    [
                        float(self.within_target),
                        float(self.success),
                        float(self.failure_reason is not None),
                        float(self.step_count),
                        self.initial_button_depth,
                    ],
                    dtype=np.float64,
                ),
                self.initial_base_position,
                self.initial_base_orientation,
                self.initial_robot_qpos,
                self.initial_robot_qvel,
            )
        )


@dataclass(frozen=True)
class PushCubeState:
    """Reconstructable cube, target, and terminal state for one timestep."""

    object_id: str
    object_index: int
    target_source: str
    target_index: int
    approach_side: str | None
    approach_side_index: int
    object_position: np.ndarray
    object_orientation: np.ndarray
    object_linear_velocity: np.ndarray
    object_angular_velocity: np.ndarray
    target_position: np.ndarray
    distance_to_target: float
    target_radius: float
    within_target: bool
    dwell_steps: int
    success: bool
    failure_reason: str | None
    step_count: int
    initial_object_position: np.ndarray
    initial_object_orientation: np.ndarray
    initial_object_linear_velocity: np.ndarray
    initial_object_angular_velocity: np.ndarray
    initial_base_position: np.ndarray
    initial_base_orientation: np.ndarray
    initial_robot_qpos: np.ndarray
    initial_robot_qvel: np.ndarray

    def as_object_state(self) -> np.ndarray:
        """Pack the current cube pose and free-joint velocity."""

        return np.concatenate(
            (
                self.object_position,
                self.object_orientation,
                self.object_linear_velocity,
                self.object_angular_velocity,
            )
        )

    def as_task_state(self) -> np.ndarray:
        """Pack success inputs, goal parameters, and deterministic initial state."""

        return np.concatenate(
            (
                self.object_position,
                self.target_position,
                np.asarray(
                    [self.distance_to_target, float(self.dwell_steps)],
                    dtype=np.float64,
                ),
                self.object_orientation,
                self.object_linear_velocity,
                self.object_angular_velocity,
                np.asarray(
                    [
                        self.target_radius,
                        float(self.within_target),
                        float(self.success),
                        float(self.failure_reason is not None),
                        float(self.step_count),
                        float(self.object_index),
                        float(self.target_index),
                        float(self.approach_side is not None),
                        float(self.approach_side_index),
                    ],
                    dtype=np.float64,
                ),
                self.initial_object_position,
                self.initial_object_orientation,
                self.initial_object_linear_velocity,
                self.initial_object_angular_velocity,
                self.initial_base_position,
                self.initial_base_orientation,
                self.initial_robot_qpos,
                self.initial_robot_qvel,
            )
        )


def reach_distance(
    touch_position: Sequence[float] | np.ndarray,
    target_position: Sequence[float] | np.ndarray,
) -> float:
    """Return Euclidean touch-to-target distance in metres."""

    touch = _finite_vector(touch_position, name="touch_position")
    target = _finite_vector(target_position, name="target_position")
    return float(np.linalg.norm(touch - target))


def is_reach_touch_success(
    *,
    distance_m: float,
    dwell_steps: int,
    distance_threshold_m: float,
    required_dwell_steps: int,
    palm_contact: bool = True,
) -> bool:
    """Evaluate the fixed distance-and-dwell success condition."""

    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("distance_m must be finite and non-negative.")
    if distance_threshold_m <= 0.0:
        raise ValueError("distance_threshold_m must be positive.")
    if dwell_steps < 0:
        raise ValueError("dwell_steps must be non-negative.")
    if required_dwell_steps <= 0:
        raise ValueError("required_dwell_steps must be positive.")
    return (
        bool(palm_contact)
        and distance_m <= distance_threshold_m
        and dwell_steps >= required_dwell_steps
    )


def reach_touch_failure_reason(
    *,
    touch_position: Sequence[float] | np.ndarray,
    step_count: int,
    max_episode_steps: int,
    workspace_min: Sequence[float] | np.ndarray,
    workspace_max: Sequence[float] | np.ndarray,
    success: bool = False,
    enforce_workspace_bounds: bool = True,
) -> str | None:
    """Return a deterministic failure label from synthetic or live state."""

    if success:
        return None
    touch = _finite_vector(touch_position, name="touch_position")
    minimum = _finite_vector(workspace_min, name="workspace_min")
    maximum = _finite_vector(workspace_max, name="workspace_max")
    if np.any(minimum >= maximum):
        raise ValueError("workspace_min must be strictly below workspace_max.")
    if step_count < 0 or max_episode_steps <= 0:
        raise ValueError("step counts must be non-negative with a positive maximum.")
    if enforce_workspace_bounds and (
        np.any(touch < minimum) or np.any(touch > maximum)
    ):
        return "workspace_bounds"
    if step_count >= max_episode_steps:
        return "timeout"
    return None


def is_button_press_success(
    *,
    press_depth_m: float,
    target_press_depth_m: float,
    button_pressed: bool,
    target_pressed_state: bool,
    dwell_steps: int,
    required_dwell_steps: int,
) -> bool:
    """Recompute button success from saved press-depth and pressed-state inputs."""

    if not np.isfinite(press_depth_m) or press_depth_m < 0.0:
        raise ValueError("press_depth_m must be finite and non-negative.")
    if not np.isfinite(target_press_depth_m) or target_press_depth_m <= 0.0:
        raise ValueError("target_press_depth_m must be finite and positive.")
    if dwell_steps < 0:
        raise ValueError("dwell_steps must be non-negative.")
    if required_dwell_steps <= 0:
        raise ValueError("required_dwell_steps must be positive.")
    depth_matches = (
        press_depth_m >= target_press_depth_m
        if target_pressed_state
        else press_depth_m < target_press_depth_m
    )
    return (
        depth_matches
        and bool(button_pressed) is bool(target_pressed_state)
        and dwell_steps >= required_dwell_steps
    )


def push_cube_distance(
    object_position: Sequence[float] | np.ndarray,
    target_position: Sequence[float] | np.ndarray,
) -> float:
    """Return planar cube-centre distance to a target in metres."""

    object_vector = _finite_vector(object_position, name="object_position")
    target_vector = _finite_vector(target_position, name="target_position")
    return float(np.linalg.norm(object_vector[:2] - target_vector[:2]))


def is_push_cube_success(
    *,
    distance_m: float,
    dwell_steps: int,
    distance_threshold_m: float,
    required_dwell_steps: int,
) -> bool:
    """Recompute push-cube success from saved distance and dwell inputs."""

    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("distance_m must be finite and non-negative.")
    if not np.isfinite(distance_threshold_m) or distance_threshold_m <= 0.0:
        raise ValueError("distance_threshold_m must be finite and positive.")
    if dwell_steps < 0:
        raise ValueError("dwell_steps must be non-negative.")
    if required_dwell_steps <= 0:
        raise ValueError("required_dwell_steps must be positive.")
    return (
        distance_m <= distance_threshold_m
        and dwell_steps >= required_dwell_steps
    )


def push_cube_failure_reason(
    *,
    object_position: Sequence[float] | np.ndarray,
    step_count: int,
    max_episode_steps: int,
    workspace_min: Sequence[float] | np.ndarray,
    workspace_max: Sequence[float] | np.ndarray,
    success: bool = False,
) -> str | None:
    """Return a deterministic cube workspace or timeout failure."""

    if success:
        return None
    position = _finite_vector(object_position, name="object_position")
    minimum = _finite_vector(workspace_min, name="workspace_min")
    maximum = _finite_vector(workspace_max, name="workspace_max")
    if np.any(minimum >= maximum):
        raise ValueError("workspace_min must be strictly below workspace_max.")
    if step_count < 0 or max_episode_steps <= 0:
        raise ValueError("step counts must be non-negative with a positive maximum.")
    if np.any(position < minimum) or np.any(position > maximum):
        return "object_workspace_bounds"
    if step_count >= max_episode_steps:
        return "timeout"
    return None


def configure_push_cube_visibility(env: MujocoEnv, *, visible: bool) -> None:
    """Enable or isolate the push-cube fixture in the shared task scene."""

    mujoco = env._mujoco
    config = PushCubeConfig()
    cube_geom_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"{config.object_ids[0]}_geom",
    )
    target_geom_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        config.target_marker_geom,
    )
    cube_body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        config.object_ids[0],
    )
    if cube_geom_id < 0 or target_geom_id < 0 or cube_body_id < 0:
        raise TaskError("MuJoCo task scene is missing push-cube visual assets.")
    env.model.geom_rgba[cube_geom_id, 3] = 1.0 if visible else 0.0
    env.model.geom_contype[cube_geom_id] = 1 if visible else 0
    env.model.geom_conaffinity[cube_geom_id] = 1 if visible else 0
    env.model.geom_rgba[target_geom_id, 3] = 0.7 if visible else 0.0
    env.model.body_gravcomp[cube_body_id] = 0.0 if visible else 1.0
    mujoco.mj_forward(env.model, env.data)


def configure_push_cube_scene(env: MujocoEnv) -> None:
    """Show only the horizontal cube-push workspace and its target cue."""

    configure_button_press_scene(env)
    mujoco = env._mujoco
    hidden_geom_names = ["task_board"]
    hidden_site_names: list[str] = []
    for button_id in ButtonPressConfig().button_ids:
        hidden_geom_names.append(f"{button_id}_geom")
        hidden_site_names.append(f"{button_id}_site")

    for geom_name in hidden_geom_names:
        geom_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )
        if geom_id < 0:
            raise TaskError(
                f"MuJoCo task scene is missing cube-mode fixture '{geom_name}'."
            )
        env.model.geom_rgba[geom_id, 3] = 0.0
        env.model.geom_contype[geom_id] = 0
        env.model.geom_conaffinity[geom_id] = 0

    for site_name in hidden_site_names:
        site_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise TaskError(
                f"MuJoCo task scene is missing cube-mode fixture '{site_name}'."
            )
        env.model.site_rgba[site_id, 3] = 0.0

    # The fingers-up, camera-facing push pose places the forearm and wrist below
    # the tabletop. Hide and disable those support bodies so they do not appear
    # to clip through the table or prevent the palm from following its mocap
    # target. Palm and finger geometry remains visible and collision-enabled.
    for body_name in ("rh_forearm", "rh_wrist"):
        body_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            raise TaskError(
                f"MuJoCo hand model is missing cube-mode body '{body_name}'."
            )
        geom_ids = np.flatnonzero(env.model.geom_bodyid == body_id)
        env.model.geom_contype[geom_ids] = 0
        env.model.geom_conaffinity[geom_ids] = 0
        env.model.geom_rgba[geom_ids, 3] = 0.0

    configure_push_cube_visibility(env, visible=True)
    mujoco.mj_forward(env.model, env.data)


def configure_button_press_scene(env: MujocoEnv) -> None:
    """Hide and disable reach-touch fixtures in the shared button scene."""

    mujoco = env._mujoco
    for site_name in ReachTouchTargetConfig().target_sites:
        site_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise TaskError(
                f"MuJoCo task scene is missing reach fixture site '{site_name}'."
            )
        env.model.site_rgba[site_id, 3] = 0.0

    geom_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        ACTIVE_REACH_TARGET_GEOM,
    )
    if geom_id < 0:
        raise TaskError(
            f"MuJoCo task scene is missing reach fixture geom "
            f"'{ACTIVE_REACH_TARGET_GEOM}'."
        )
    env.model.geom_rgba[geom_id, 3] = 0.0
    env.model.geom_contype[geom_id] = 0
    env.model.geom_conaffinity[geom_id] = 0
    configure_push_cube_visibility(env, visible=False)
    mujoco.mj_forward(env.model, env.data)


def color_button_press_target(
    env: MujocoEnv,
    button_id: str,
    *,
    button_ids: Sequence[str] = ButtonPressConfig().button_ids,
) -> None:
    """Color the selected button bright green and all other buttons gray."""

    if button_id not in button_ids:
        raise TaskError(
            f"Unknown button target '{button_id}'; expected one of: "
            f"{', '.join(button_ids)}."
        )
    mujoco = env._mujoco
    for candidate_id in button_ids:
        rgba = (
            BUTTON_TARGET_RGBA
            if candidate_id == button_id
            else BUTTON_NON_TARGET_RGBA
        )
        geom_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{candidate_id}_geom",
        )
        site_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"{candidate_id}_site",
        )
        material_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            f"{candidate_id}_material",
        )
        if geom_id < 0 or site_id < 0 or material_id < 0:
            raise TaskError(
                f"MuJoCo task scene is missing visual assets for '{candidate_id}'."
            )
        env.model.geom_rgba[geom_id] = rgba
        env.model.site_rgba[site_id] = rgba
        env.model.mat_rgba[material_id] = rgba
    mujoco.mj_forward(env.model, env.data)


class ReachTouchTargetTask:
    """Reset, step, and extract state for ``reach_touch_target``."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TASK_BOARD_MODEL,
        *,
        config: ReachTouchTargetConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = config or ReachTouchTargetConfig()
        self.env = MujocoEnv(self.model_path)
        configure_push_cube_visibility(self.env, visible=False)
        self._initial_robot_state: MujocoState | None = None
        self._initial_base_position: np.ndarray | None = None
        self._initial_base_orientation: np.ndarray | None = None
        self._target_source: str | None = None
        self._target_index = -1
        self._target_position: np.ndarray | None = None
        self._step_count = 0
        self._dwell_steps = 0
        self.spec = _build_reach_touch_target_spec(self.env, self.config)

    def reset(
        self,
        *,
        seed: int = 0,
        parameters: ReachTouchTargetParameters | None = None,
    ) -> ReachTouchTargetState:
        """Reset deterministically and select a configured or explicit target."""

        parameters = parameters or ReachTouchTargetParameters()
        initial = self.env.reset()
        self._initial_robot_state = initial
        (
            self._initial_base_position,
            self._initial_base_orientation,
        ) = self.env.get_mocap_pose(self.config.base_target_body)
        self._target_source, self._target_index, self._target_position = (
            self._resolve_target(parameters=parameters, seed=seed)
        )
        self._require_inside_workspace(self._target_position, name="target position")
        self._move_target_marker(self._target_position)
        self._step_count = 0
        self._dwell_steps = 0
        return self.get_state()

    def step(
        self,
        action: Sequence[float] | np.ndarray | Mapping[str, float] | None = None,
        *,
        n_steps: int = 1,
    ) -> ReachTouchTargetState:
        """Advance MuJoCo once and update reach success/failure state."""

        self._require_reset()
        self.env.step(action, n_steps=n_steps)
        self._step_count += 1
        palm_contact, touch_position = self._palm_target_contact()
        distance = reach_distance(touch_position, self._target_position)
        if palm_contact and distance <= self.config.success_distance_m:
            self._dwell_steps += 1
        else:
            self._dwell_steps = 0
        return self._make_state(
            touch_position=touch_position,
            palm_contact=palm_contact,
        )

    def get_state(self) -> ReachTouchTargetState:
        """Extract the current task state without advancing simulation."""

        self._require_reset()
        palm_contact, touch_position = self._palm_target_contact()
        return self._make_state(
            touch_position=touch_position,
            palm_contact=palm_contact,
        )

    def task_state_vector(self) -> np.ndarray:
        """Return the current dense task state for future demo logging."""

        return self.get_state().as_task_state()

    def robot_state_vector(self) -> np.ndarray:
        """Pack robot state in the executable Level 2 observation order."""

        self._require_reset()
        state = self.env.get_state()
        base_position, base_orientation = self.env.get_mocap_pose(
            self.config.base_target_body
        )
        return np.concatenate(
            (state.qpos, state.qvel, state.ctrl, base_position, base_orientation)
        )

    def close(self) -> None:
        """Release the underlying MuJoCo environment."""

        self.env.close()

    def __enter__(self) -> "ReachTouchTargetTask":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _resolve_target(
        self,
        *,
        parameters: ReachTouchTargetParameters,
        seed: int,
    ) -> tuple[str, int, np.ndarray]:
        if parameters.target_pose is not None:
            return "target_pose", -1, np.asarray(parameters.target_pose, dtype=np.float64)
        if parameters.target_site is not None:
            if parameters.target_site not in self.config.target_sites:
                allowed = ", ".join(self.config.target_sites)
                raise TaskError(
                    f"Unknown reach target site '{parameters.target_site}'; "
                    f"expected one of: {allowed}."
                )
            index = self.config.target_sites.index(parameters.target_site)
            return parameters.target_site, index, self._site_position(parameters.target_site)

        rng = np.random.default_rng(seed)
        index = int(rng.integers(0, len(self.config.target_sites)))
        site_name = self.config.target_sites[index]
        return site_name, index, self._site_position(site_name)

    def _make_state(
        self,
        *,
        touch_position: np.ndarray,
        palm_contact: bool,
    ) -> ReachTouchTargetState:
        self._require_reset()
        distance = reach_distance(touch_position, self._target_position)
        within = palm_contact and distance <= self.config.success_distance_m
        success = is_reach_touch_success(
            distance_m=distance,
            dwell_steps=self._dwell_steps,
            distance_threshold_m=self.config.success_distance_m,
            required_dwell_steps=self.config.success_dwell_steps,
            palm_contact=palm_contact,
        )
        failure = reach_touch_failure_reason(
            touch_position=touch_position,
            step_count=self._step_count,
            max_episode_steps=self.config.max_episode_steps,
            workspace_min=self.config.workspace_min,
            workspace_max=self.config.workspace_max,
            success=success,
            enforce_workspace_bounds=self.config.terminate_on_workspace_bounds,
        )
        initial = self._initial_robot_state
        if initial is None:  # pragma: no cover - guarded by _require_reset.
            raise TaskError("reach_touch_target must be reset before state extraction.")
        return ReachTouchTargetState(
            target_source=str(self._target_source),
            target_index=self._target_index,
            target_position=np.asarray(self._target_position, dtype=np.float64).copy(),
            touch_position=touch_position.copy(),
            distance_to_target=distance,
            palm_contact=palm_contact,
            within_success_distance=within,
            dwell_steps=self._dwell_steps,
            success=success,
            failure_reason=failure,
            step_count=self._step_count,
            initial_base_position=np.asarray(
                self._initial_base_position, dtype=np.float64
            ).copy(),
            initial_base_orientation=np.asarray(
                self._initial_base_orientation, dtype=np.float64
            ).copy(),
            initial_robot_qpos=initial.qpos.copy(),
            initial_robot_qvel=initial.qvel.copy(),
        )

    def _palm_target_contact(self) -> tuple[bool, np.ndarray]:
        """Return the closest physical palm-to-active-target contact point."""

        target_geom_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_GEOM,
            self.config.target_marker_geom,
        )
        palm_body_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_BODY,
            self.config.palm_body,
        )
        if target_geom_id < 0:
            raise TaskError(
                "MuJoCo task scene is missing active target geom "
                f"'{self.config.target_marker_geom}'."
            )
        if palm_body_id < 0:
            raise TaskError(
                f"MuJoCo hand model is missing palm body '{self.config.palm_body}'."
            )

        contact_positions: list[np.ndarray] = []
        for contact_index in range(int(self.env.data.ncon)):
            contact = self.env.data.contact[contact_index]
            if contact.geom1 == target_geom_id:
                other_geom_id = int(contact.geom2)
            elif contact.geom2 == target_geom_id:
                other_geom_id = int(contact.geom1)
            else:
                continue
            if int(self.env.model.geom_bodyid[other_geom_id]) != palm_body_id:
                continue
            contact_positions.append(
                np.asarray(contact.pos, dtype=np.float64).copy()
            )

        if not contact_positions:
            return False, self._site_position(self.config.touch_site)
        closest = min(
            contact_positions,
            key=lambda position: reach_distance(position, self._target_position),
        )
        return True, closest

    def _site_position(self, site_name: str) -> np.ndarray:
        site_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise TaskError(f"MuJoCo task scene is missing required site '{site_name}'.")
        return np.asarray(self.env.data.site_xpos[site_id], dtype=np.float64).copy()

    def _move_target_marker(self, target_position: np.ndarray) -> None:
        self.env.set_mocap_pose(
            self.config.target_marker_body,
            position=target_position,
            orientation_quat=(1.0, 0.0, 0.0, 0.0),
        )
        self.env._mujoco.mj_forward(self.env.model, self.env.data)

    def _require_inside_workspace(self, position: np.ndarray, *, name: str) -> None:
        minimum = np.asarray(self.config.workspace_min, dtype=np.float64)
        maximum = np.asarray(self.config.workspace_max, dtype=np.float64)
        if np.any(position < minimum) or np.any(position > maximum):
            raise TaskError(
                f"{name} {position.tolist()} is outside configured workspace "
                f"{minimum.tolist()} to {maximum.tolist()}."
            )

    def _require_reset(self) -> None:
        if (
            self._initial_robot_state is None
            or self._target_position is None
            or self._initial_base_position is None
            or self._initial_base_orientation is None
        ):
            raise TaskError("reach_touch_target must be reset before use.")


class ButtonPressTask:
    """Reset, step, and extract state for ``button_press``."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TASK_BOARD_MODEL,
        *,
        config: ButtonPressConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = config or ButtonPressConfig()
        self.env = MujocoEnv(self.model_path)
        configure_button_press_scene(self.env)
        self._initial_robot_state: MujocoState | None = None
        self._initial_base_position: np.ndarray | None = None
        self._initial_base_orientation: np.ndarray | None = None
        self._button_id: str | None = None
        self._button_index = -1
        self._button_position: np.ndarray | None = None
        self._approach_pose = np.zeros(3, dtype=np.float64)
        self._approach_pose_present = False
        self._target_press_depth = self.config.default_target_press_depth_m
        self._target_pressed_state = True
        self._initial_button_depth = 0.0
        self._step_count = 0
        self._dwell_steps = 0
        self.spec = _build_button_press_spec(self.env, self.config)

    def reset(
        self,
        *,
        seed: int = 0,
        parameters: ButtonPressParameters | None = None,
    ) -> ButtonPressState:
        """Reset deterministically and resolve the selected button goal."""

        parameters = parameters or ButtonPressParameters()
        initial = self.env.reset()
        self._initial_robot_state = initial
        (
            self._initial_base_position,
            self._initial_base_orientation,
        ) = self.env.get_mocap_pose(self.config.base_target_body)
        self._button_id, self._button_index = self._resolve_button(
            parameters=parameters,
            seed=seed,
        )
        color_button_press_target(
            self.env,
            self._button_id,
            button_ids=self.config.button_ids,
        )
        self._button_position = self._site_position(
            self._button_site_name(self._button_id)
        )
        self._approach_pose_present = parameters.approach_pose is not None
        self._approach_pose = (
            np.asarray(parameters.approach_pose, dtype=np.float64)
            if parameters.approach_pose is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._target_press_depth = (
            parameters.target_press_depth
            if parameters.target_press_depth is not None
            else self.config.default_target_press_depth_m
        )
        joint_min, joint_max = self._button_joint_range(self._button_id)
        if self._target_press_depth < joint_min or self._target_press_depth > joint_max:
            raise TaskError(
                f"target_press_depth {self._target_press_depth:.6f} m is outside "
                f"the selected button joint range [{joint_min:.6f}, "
                f"{joint_max:.6f}] m."
            )
        self._target_pressed_state = (
            parameters.pressed_state_target
            if parameters.pressed_state_target is not None
            else True
        )
        self._initial_button_depth = self._button_depth(self._button_id)
        self._step_count = 0
        self._dwell_steps = 0
        return self.get_state()

    def step(
        self,
        action: Sequence[float] | np.ndarray | Mapping[str, float] | None = None,
        *,
        n_steps: int = 1,
    ) -> ButtonPressState:
        """Advance MuJoCo and update the consecutive goal-state dwell count."""

        self._require_reset()
        self.env.step(action, n_steps=n_steps)
        self._step_count += 1
        press_depth = self._button_depth(str(self._button_id))
        if self._goal_matches(press_depth):
            self._dwell_steps += 1
        else:
            self._dwell_steps = 0
        return self._make_state(press_depth=press_depth)

    def get_state(self) -> ButtonPressState:
        """Extract the current button/task state without advancing simulation."""

        self._require_reset()
        return self._make_state(
            press_depth=self._button_depth(str(self._button_id))
        )

    def task_state_vector(self) -> np.ndarray:
        """Return the current dense button task state for future logging."""

        return self.get_state().as_task_state()

    def robot_state_vector(self) -> np.ndarray:
        """Pack robot state in the executable Level 2 observation order."""

        self._require_reset()
        state = self.env.get_state()
        base_position, base_orientation = self.env.get_mocap_pose(
            self.config.base_target_body
        )
        return np.concatenate(
            (state.qpos, state.qvel, state.ctrl, base_position, base_orientation)
        )

    def close(self) -> None:
        """Release the underlying MuJoCo environment."""

        self.env.close()

    def __enter__(self) -> "ButtonPressTask":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _resolve_button(
        self,
        *,
        parameters: ButtonPressParameters,
        seed: int,
    ) -> tuple[str, int]:
        if parameters.button_id is not None:
            if parameters.button_id not in self.config.button_ids:
                allowed = ", ".join(self.config.button_ids)
                raise TaskError(
                    f"Unknown button_id '{parameters.button_id}'; "
                    f"expected one of: {allowed}."
                )
            return parameters.button_id, self.config.button_ids.index(
                parameters.button_id
            )
        rng = np.random.default_rng(seed)
        index = int(rng.integers(0, len(self.config.button_ids)))
        return self.config.button_ids[index], index

    def _make_state(self, *, press_depth: float) -> ButtonPressState:
        self._require_reset()
        button_pressed = press_depth >= self._target_press_depth
        within_target = (
            button_pressed
            if self._target_pressed_state
            else not button_pressed
        )
        success = is_button_press_success(
            press_depth_m=press_depth,
            target_press_depth_m=self._target_press_depth,
            button_pressed=button_pressed,
            target_pressed_state=self._target_pressed_state,
            dwell_steps=self._dwell_steps,
            required_dwell_steps=self.config.success_dwell_steps,
        )
        failure = (
            None
            if success or self._step_count < self.config.max_episode_steps
            else "timeout"
        )
        initial = self._initial_robot_state
        if initial is None:  # pragma: no cover - guarded by _require_reset.
            raise TaskError("button_press must be reset before state extraction.")
        return ButtonPressState(
            button_id=str(self._button_id),
            button_index=self._button_index,
            button_position=np.asarray(
                self._button_position, dtype=np.float64
            ).copy(),
            approach_pose=self._approach_pose.copy(),
            approach_pose_present=self._approach_pose_present,
            press_depth=press_depth,
            target_press_depth=self._target_press_depth,
            button_pressed=button_pressed,
            target_pressed_state=self._target_pressed_state,
            within_target=within_target,
            dwell_steps=self._dwell_steps,
            success=success,
            failure_reason=failure,
            step_count=self._step_count,
            initial_button_depth=self._initial_button_depth,
            initial_base_position=np.asarray(
                self._initial_base_position, dtype=np.float64
            ).copy(),
            initial_base_orientation=np.asarray(
                self._initial_base_orientation, dtype=np.float64
            ).copy(),
            initial_robot_qpos=initial.qpos.copy(),
            initial_robot_qvel=initial.qvel.copy(),
        )

    def _goal_matches(self, press_depth: float) -> bool:
        button_pressed = press_depth >= self._target_press_depth
        depth_matches = (
            press_depth >= self._target_press_depth
            if self._target_pressed_state
            else press_depth < self._target_press_depth
        )
        return depth_matches and button_pressed == self._target_pressed_state

    def _button_depth(self, button_id: str) -> float:
        joint_name = self._button_joint_name(button_id)
        joint_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise TaskError(
                f"MuJoCo task scene is missing button joint '{joint_name}'."
            )
        qpos_address = int(self.env.model.jnt_qposadr[joint_id])
        depth = float(self.env.data.qpos[qpos_address])
        if not np.isfinite(depth):
            raise TaskError(f"Button joint '{joint_name}' has non-finite state.")
        return max(0.0, depth)

    def _button_joint_range(self, button_id: str) -> tuple[float, float]:
        joint_name = self._button_joint_name(button_id)
        joint_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise TaskError(
                f"MuJoCo task scene is missing button joint '{joint_name}'."
            )
        joint_range = np.asarray(
            self.env.model.jnt_range[joint_id],
            dtype=np.float64,
        )
        return float(joint_range[0]), float(joint_range[1])

    def _site_position(self, site_name: str) -> np.ndarray:
        site_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise TaskError(f"MuJoCo task scene is missing required site '{site_name}'.")
        return np.asarray(self.env.data.site_xpos[site_id], dtype=np.float64).copy()

    @staticmethod
    def _button_joint_name(button_id: str) -> str:
        return f"{button_id}_joint"

    @staticmethod
    def _button_site_name(button_id: str) -> str:
        return f"{button_id}_site"

    def _require_reset(self) -> None:
        if (
            self._initial_robot_state is None
            or self._button_id is None
            or self._button_position is None
            or self._initial_base_position is None
            or self._initial_base_orientation is None
        ):
            raise TaskError("button_press must be reset before use.")


class PushCubeTask:
    """Reset, step, and extract state for ``push_cube_to_target``."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TASK_BOARD_MODEL,
        *,
        config: PushCubeConfig | None = None,
        enforce_timeout: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = config or PushCubeConfig()
        self.enforce_timeout = bool(enforce_timeout)
        self.env = MujocoEnv(self.model_path)
        configure_push_cube_scene(self.env)
        self._initial_robot_state: MujocoState | None = None
        self._initial_base_position: np.ndarray | None = None
        self._initial_base_orientation: np.ndarray | None = None
        self._object_id: str | None = None
        self._object_index = -1
        self._target_source: str | None = None
        self._target_index = -1
        self._target_position: np.ndarray | None = None
        self._approach_side: str | None = None
        self._initial_object_position: np.ndarray | None = None
        self._initial_object_orientation: np.ndarray | None = None
        self._initial_object_linear_velocity: np.ndarray | None = None
        self._initial_object_angular_velocity: np.ndarray | None = None
        self._step_count = 0
        self._dwell_steps = 0
        self.spec = _build_push_cube_spec(self.env, self.config)

    def reset(
        self,
        *,
        seed: int = 0,
        parameters: PushCubeParameters | None = None,
    ) -> PushCubeState:
        """Reset cube and target deterministically from the supplied seed."""

        parameters = parameters or PushCubeParameters()
        self.env.reset()
        rng = np.random.default_rng(seed)
        self._object_id, self._object_index = self._resolve_object(parameters)
        if parameters.target_zone_id is not None:
            # Named pilot targets use the matching lateral start lane so the
            # first task is a clear planar push rather than a diagonal recovery.
            start_index = self.config.target_zone_sites.index(
                parameters.target_zone_id
            )
        else:
            start_index = int(rng.integers(0, len(self.config.object_start_sites)))
        start_position = self._site_position(
            self.config.object_start_sites[start_index]
        )
        self._require_inside_workspace(start_position, name="object start position")
        self._set_object_state(
            position=start_position,
            orientation=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )
        initial_base_position = np.asarray(
            [
                self.config.initial_base_x,
                start_position[1],
                self.config.initial_base_z,
            ],
            dtype=np.float64,
        )
        initial_base_orientation = np.asarray(
            self.config.initial_base_orientation,
            dtype=np.float64,
        )
        initial_base_orientation /= np.linalg.norm(initial_base_orientation)
        self.env.set_mocap_pose(
            self.config.base_target_body,
            position=initial_base_position,
            orientation_quat=initial_base_orientation,
        )
        base_joint_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_JOINT,
            self.config.base_free_joint,
        )
        if base_joint_id < 0 or int(
            self.env.model.jnt_type[base_joint_id]
        ) != int(self.env._mujoco.mjtJoint.mjJNT_FREE):
            raise TaskError(
                "push-cube scene requires free joint "
                f"'{self.config.base_free_joint}' for aligned hand reset."
            )
        base_qpos_address = int(self.env.model.jnt_qposadr[base_joint_id])
        base_qvel_address = int(self.env.model.jnt_dofadr[base_joint_id])
        self.env.data.qpos[base_qpos_address : base_qpos_address + 3] = (
            initial_base_position
        )
        self.env.data.qpos[base_qpos_address + 3 : base_qpos_address + 7] = (
            initial_base_orientation
        )
        self.env.data.qvel[base_qvel_address : base_qvel_address + 6] = 0.0
        self.env._mujoco.mj_forward(self.env.model, self.env.data)
        (
            self._initial_base_position,
            self._initial_base_orientation,
        ) = self.env.get_mocap_pose(self.config.base_target_body)
        (
            self._target_source,
            self._target_index,
            self._target_position,
        ) = self._resolve_target(parameters=parameters, rng=rng)
        self._require_inside_workspace(self._target_position, name="target position")
        self._move_target_marker(self._target_position)
        self._approach_side = self._resolve_approach_side(parameters)
        (
            self._initial_object_position,
            self._initial_object_orientation,
            self._initial_object_linear_velocity,
            self._initial_object_angular_velocity,
        ) = self._object_state()
        self._initial_robot_state = self.env.get_state()
        self._step_count = 0
        self._dwell_steps = 0
        return self.get_state()

    def step(
        self,
        action: Sequence[float] | np.ndarray | Mapping[str, float] | None = None,
        *,
        n_steps: int = 1,
    ) -> PushCubeState:
        """Advance MuJoCo and update the consecutive in-target dwell count."""

        self._require_reset()
        self.env.step(action, n_steps=n_steps)
        self._step_count += 1
        position, _, _, _ = self._object_state()
        if (
            push_cube_distance(position, self._target_position)
            <= self.config.target_radius_m
        ):
            self._dwell_steps += 1
        else:
            self._dwell_steps = 0
        return self._make_state()

    def get_state(self) -> PushCubeState:
        """Extract current cube, target, and terminal state without stepping."""

        self._require_reset()
        return self._make_state()

    def object_state_vector(self) -> np.ndarray:
        """Return current cube pose and velocity for episode logging."""

        return self.get_state().as_object_state()

    def task_state_vector(self) -> np.ndarray:
        """Return current dense task state for future episode logging."""

        return self.get_state().as_task_state()

    def robot_state_vector(self) -> np.ndarray:
        """Pack robot state in the executable Level 2 observation order."""

        self._require_reset()
        state = self.env.get_state()
        base_position, base_orientation = self.env.get_mocap_pose(
            self.config.base_target_body
        )
        return np.concatenate(
            (state.qpos, state.qvel, state.ctrl, base_position, base_orientation)
        )

    def close(self) -> None:
        """Release the underlying MuJoCo environment."""

        self.env.close()

    def __enter__(self) -> "PushCubeTask":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _resolve_object(
        self,
        parameters: PushCubeParameters,
    ) -> tuple[str, int]:
        object_id = parameters.object_id or self.config.object_ids[0]
        if object_id not in self.config.object_ids:
            allowed = ", ".join(self.config.object_ids)
            raise TaskError(
                f"Unknown object_id '{object_id}'; expected one of: {allowed}."
            )
        return object_id, self.config.object_ids.index(object_id)

    def _resolve_target(
        self,
        *,
        parameters: PushCubeParameters,
        rng: np.random.Generator,
    ) -> tuple[str, int, np.ndarray]:
        if parameters.target_pose is not None:
            return (
                "target_pose",
                -1,
                np.asarray(parameters.target_pose, dtype=np.float64),
            )
        if parameters.target_zone_id is not None:
            if parameters.target_zone_id not in self.config.target_zone_sites:
                allowed = ", ".join(self.config.target_zone_sites)
                raise TaskError(
                    f"Unknown target_zone_id '{parameters.target_zone_id}'; "
                    f"expected one of: {allowed}."
                )
            index = self.config.target_zone_sites.index(parameters.target_zone_id)
            return (
                parameters.target_zone_id,
                index,
                self._site_position(parameters.target_zone_id),
            )
        index = int(rng.integers(0, len(self.config.target_zone_sites)))
        site_name = self.config.target_zone_sites[index]
        return site_name, index, self._site_position(site_name)

    def _resolve_approach_side(
        self,
        parameters: PushCubeParameters,
    ) -> str | None:
        if parameters.approach_side is None:
            return None
        if parameters.approach_side not in self.config.approach_sides:
            allowed = ", ".join(self.config.approach_sides)
            raise TaskError(
                f"Unknown approach_side '{parameters.approach_side}'; "
                f"expected one of: {allowed}."
            )
        return parameters.approach_side

    def _make_state(self) -> PushCubeState:
        self._require_reset()
        position, orientation, linear_velocity, angular_velocity = (
            self._object_state()
        )
        distance = push_cube_distance(position, self._target_position)
        within_target = distance <= self.config.target_radius_m
        success = is_push_cube_success(
            distance_m=distance,
            dwell_steps=self._dwell_steps,
            distance_threshold_m=self.config.target_radius_m,
            required_dwell_steps=self.config.success_dwell_steps,
        )
        failure = push_cube_failure_reason(
            object_position=position,
            step_count=self._step_count,
            max_episode_steps=self.config.max_episode_steps,
            workspace_min=self.config.workspace_min,
            workspace_max=self.config.workspace_max,
            success=success,
        )
        if failure == "timeout" and not self.enforce_timeout:
            failure = None
        initial = self._initial_robot_state
        if initial is None:  # pragma: no cover - guarded by _require_reset.
            raise TaskError("push_cube_to_target must be reset before extraction.")
        return PushCubeState(
            object_id=str(self._object_id),
            object_index=self._object_index,
            target_source=str(self._target_source),
            target_index=self._target_index,
            approach_side=self._approach_side,
            approach_side_index=(
                -1
                if self._approach_side is None
                else self.config.approach_sides.index(self._approach_side)
            ),
            object_position=position,
            object_orientation=orientation,
            object_linear_velocity=linear_velocity,
            object_angular_velocity=angular_velocity,
            target_position=np.asarray(
                self._target_position, dtype=np.float64
            ).copy(),
            distance_to_target=distance,
            target_radius=self.config.target_radius_m,
            within_target=within_target,
            dwell_steps=self._dwell_steps,
            success=success,
            failure_reason=failure,
            step_count=self._step_count,
            initial_object_position=np.asarray(
                self._initial_object_position, dtype=np.float64
            ).copy(),
            initial_object_orientation=np.asarray(
                self._initial_object_orientation, dtype=np.float64
            ).copy(),
            initial_object_linear_velocity=np.asarray(
                self._initial_object_linear_velocity, dtype=np.float64
            ).copy(),
            initial_object_angular_velocity=np.asarray(
                self._initial_object_angular_velocity, dtype=np.float64
            ).copy(),
            initial_base_position=np.asarray(
                self._initial_base_position, dtype=np.float64
            ).copy(),
            initial_base_orientation=np.asarray(
                self._initial_base_orientation, dtype=np.float64
            ).copy(),
            initial_robot_qpos=initial.qpos.copy(),
            initial_robot_qvel=initial.qvel.copy(),
        )

    def _object_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qpos_address, qvel_address = self._object_joint_addresses()
        qpos = np.asarray(
            self.env.data.qpos[qpos_address : qpos_address + 7],
            dtype=np.float64,
        )
        qvel = np.asarray(
            self.env.data.qvel[qvel_address : qvel_address + 6],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
            raise TaskError(f"Object '{self._object_id}' has non-finite state.")
        return (
            qpos[:3].copy(),
            qpos[3:7].copy(),
            qvel[:3].copy(),
            qvel[3:6].copy(),
        )

    def _set_object_state(
        self,
        *,
        position: Sequence[float] | np.ndarray,
        orientation: Sequence[float] | np.ndarray,
        linear_velocity: Sequence[float] | np.ndarray,
        angular_velocity: Sequence[float] | np.ndarray,
    ) -> None:
        qpos_address, qvel_address = self._object_joint_addresses()
        position_vector = _finite_vector(position, name="object_position")
        orientation_vector = np.asarray(orientation, dtype=np.float64)
        linear_vector = _finite_vector(linear_velocity, name="linear_velocity")
        angular_vector = _finite_vector(angular_velocity, name="angular_velocity")
        if (
            orientation_vector.shape != (4,)
            or not np.all(np.isfinite(orientation_vector))
            or np.linalg.norm(orientation_vector) <= 1e-12
        ):
            raise ValueError(
                "object orientation must be a finite non-zero wxyz quaternion."
            )
        orientation_vector = orientation_vector / np.linalg.norm(orientation_vector)
        self.env.data.qpos[qpos_address : qpos_address + 3] = position_vector
        self.env.data.qpos[qpos_address + 3 : qpos_address + 7] = orientation_vector
        self.env.data.qvel[qvel_address : qvel_address + 3] = linear_vector
        self.env.data.qvel[qvel_address + 3 : qvel_address + 6] = angular_vector
        self.env._mujoco.mj_forward(self.env.model, self.env.data)

    def _object_joint_addresses(self) -> tuple[int, int]:
        joint_name = f"{self._object_id}_joint"
        joint_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise TaskError(
                f"MuJoCo task scene is missing object free joint '{joint_name}'."
            )
        if int(self.env.model.jnt_type[joint_id]) != int(
            self.env._mujoco.mjtJoint.mjJNT_FREE
        ):
            raise TaskError(f"Object joint '{joint_name}' must be a free joint.")
        return (
            int(self.env.model.jnt_qposadr[joint_id]),
            int(self.env.model.jnt_dofadr[joint_id]),
        )

    def _site_position(self, site_name: str) -> np.ndarray:
        site_id = self.env._mujoco.mj_name2id(
            self.env.model,
            self.env._mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise TaskError(f"MuJoCo task scene is missing required site '{site_name}'.")
        return np.asarray(self.env.data.site_xpos[site_id], dtype=np.float64).copy()

    def _move_target_marker(self, target_position: np.ndarray) -> None:
        self.env.set_mocap_pose(
            self.config.target_marker_body,
            position=target_position,
            orientation_quat=(1.0, 0.0, 0.0, 0.0),
        )
        self.env._mujoco.mj_forward(self.env.model, self.env.data)

    def _require_inside_workspace(self, position: np.ndarray, *, name: str) -> None:
        minimum = np.asarray(self.config.workspace_min, dtype=np.float64)
        maximum = np.asarray(self.config.workspace_max, dtype=np.float64)
        if np.any(position < minimum) or np.any(position > maximum):
            raise TaskError(
                f"{name} {position.tolist()} is outside configured workspace "
                f"{minimum.tolist()} to {maximum.tolist()}."
            )

    def _require_reset(self) -> None:
        if (
            self._initial_robot_state is None
            or self._initial_base_position is None
            or self._initial_base_orientation is None
            or self._object_id is None
            or self._target_position is None
            or self._initial_object_position is None
            or self._initial_object_orientation is None
            or self._initial_object_linear_velocity is None
            or self._initial_object_angular_velocity is None
        ):
            raise TaskError("push_cube_to_target must be reset before use.")


def _build_reach_touch_target_spec(
    env: MujocoEnv,
    config: ReachTouchTargetConfig,
) -> TaskSpec:
    (
        qpos_names,
        qvel_names,
        actuator_names,
        finger_qpos_indices,
        finger_qvel_indices,
        finger_joint_names,
    ) = _mujoco_observation_order(env)
    action_schema = build_level1_action_schema(actuator_names)
    task_state_dim = 21 + int(env.model.nq) + int(env.model.nv)
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=int(env.model.nq),
        robot_qvel_dim=int(env.model.nv),
        finger_target_dim=int(env.model.nu),
        tracking_quality_dim=len(DEFAULT_TRACKING_QUALITY_NAMES),
        robot_qpos_names=qpos_names,
        robot_qvel_names=qvel_names,
        actuator_names=actuator_names,
        finger_joint_qpos_indices=finger_qpos_indices,
        finger_joint_qvel_indices=finger_qvel_indices,
        finger_joint_names=finger_joint_names,
        tracking_quality_names=DEFAULT_TRACKING_QUALITY_NAMES,
        task_state_dim=task_state_dim,
        target_state_dim=3,
        success_metric_dim=8,
    )
    observation_schema.validate()
    action_schema.validate()
    state_fields = (
        "target_position",
        "touch_position",
        "distance_to_target",
        "palm_contact",
        "within_success_distance",
        "dwell_steps",
        "success",
        "failure",
        "step_count",
        "target_index",
        "initial_base_position",
        "initial_base_orientation",
        "initial_robot_qpos",
        "initial_robot_qvel",
    )
    return TaskSpec(
        task_id=REACH_TOUCH_TARGET_TASK_ID,
        skill_name=REACH_TOUCH_TARGET_TASK_ID,
        required_objects=("reach_target_marker",),
        observation_schema=observation_schema,
        action_schema=action_schema,
        success_condition=(
            f"physical {config.palm_body} contact with {config.target_marker_geom} "
            f"within {config.success_distance_m:.3f} m of target center for "
            f"{config.success_dwell_steps} consecutive control steps"
        ),
        failure_conditions=(
            ("timeout", "workspace_bounds", "tracking_quality")
            if config.terminate_on_workspace_bounds
            else ("timeout", "tracking_quality")
        ),
        max_episode_steps=config.max_episode_steps,
        reset_config={
            "target_sites": config.target_sites,
            "target_pose_units": "metres",
            "target_pose_frame": "MuJoCo world",
            "deterministic_seed": True,
            "initial_robot_state_saved": True,
            "terminate_on_workspace_bounds": config.terminate_on_workspace_bounds,
            "contact_body": config.palm_body,
            "contact_target_geom": config.target_marker_geom,
        },
        parameter_type=ReachTouchTargetParameters,
        parameter_schema={
            "target_pose": {
                "type": "float64",
                "shape": (3,),
                "units": "metres",
                "coordinate_frame": "MuJoCo world",
                "required": False,
            },
            "target_site": {
                "type": "string",
                "shape": (),
                "named_id_source": config.target_sites,
                "required": False,
            },
        },
        state_fields=state_fields,
        success_metric_inputs=(
            "target_position",
            "touch_position",
            "distance_to_target",
            "palm_contact",
        ),
        terminal_state_schema={
            "success": {"type": "boolean", "terminal": True},
            "failure_reason": {
                "type": "string_or_null",
                "values": ("timeout", "workspace_bounds", "tracking_quality"),
                "terminal": True,
            },
            "distance_to_target": {"type": "float64", "units": "metres"},
            "palm_contact": {"type": "boolean"},
            "dwell_steps": {"type": "integer", "units": "control steps"},
        },
    )


def _build_button_press_spec(
    env: MujocoEnv,
    config: ButtonPressConfig,
) -> TaskSpec:
    (
        qpos_names,
        qvel_names,
        actuator_names,
        finger_qpos_indices,
        finger_qvel_indices,
        finger_joint_names,
    ) = _mujoco_observation_order(env)
    action_schema = build_level1_action_schema(actuator_names)
    task_state_dim = 25 + int(env.model.nq) + int(env.model.nv)
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=int(env.model.nq),
        robot_qvel_dim=int(env.model.nv),
        finger_target_dim=int(env.model.nu),
        tracking_quality_dim=len(DEFAULT_TRACKING_QUALITY_NAMES),
        robot_qpos_names=qpos_names,
        robot_qvel_names=qvel_names,
        actuator_names=actuator_names,
        finger_joint_qpos_indices=finger_qpos_indices,
        finger_joint_qvel_indices=finger_qvel_indices,
        finger_joint_names=finger_joint_names,
        tracking_quality_names=DEFAULT_TRACKING_QUALITY_NAMES,
        object_state_dim=4,
        task_state_dim=task_state_dim,
        target_state_dim=13,
        success_metric_dim=5,
    )
    observation_schema.validate()
    action_schema.validate()
    return TaskSpec(
        task_id=BUTTON_PRESS_TASK_ID,
        skill_name=BUTTON_PRESS_TASK_ID,
        required_objects=config.button_ids,
        observation_schema=observation_schema,
        action_schema=action_schema,
        success_condition=(
            "selected button press depth reaches the resolved target and its "
            f"pressed state matches for {config.success_dwell_steps} consecutive "
            "control steps"
        ),
        failure_conditions=("timeout", "tracking_quality"),
        max_episode_steps=config.max_episode_steps,
        reset_config={
            "button_ids": config.button_ids,
            "default_target_press_depth_m": config.default_target_press_depth_m,
            "deterministic_seed": True,
            "initial_button_state_saved": True,
            "initial_robot_state_saved": True,
        },
        parameter_type=ButtonPressParameters,
        parameter_schema={
            "button_id": {
                "type": "string",
                "shape": (),
                "named_id_source": config.button_ids,
                "required": False,
            },
            "target_press_depth": {
                "type": "float64",
                "shape": (),
                "units": "metres",
                "coordinate_frame": "button joint displacement",
                "required": False,
            },
            "pressed_state_target": {
                "type": "boolean",
                "shape": (),
                "required": False,
            },
            "approach_pose": {
                "type": "float64",
                "shape": (3,),
                "units": "metres",
                "coordinate_frame": "MuJoCo world",
                "required": False,
            },
        },
        state_fields=(
            "button_id",
            "button_index",
            "button_position",
            "approach_pose",
            "approach_pose_present",
            "press_depth",
            "target_press_depth",
            "button_pressed",
            "target_pressed_state",
            "within_target",
            "dwell_steps",
            "success",
            "failure_reason",
            "step_count",
            "initial_button_depth",
            "initial_base_position",
            "initial_base_orientation",
            "initial_robot_qpos",
            "initial_robot_qvel",
        ),
        success_metric_inputs=(
            "press_depth",
            "target_press_depth",
            "button_pressed",
            "target_pressed_state",
            "dwell_steps",
        ),
        terminal_state_schema={
            "success": {"type": "boolean", "terminal": True},
            "failure_reason": {
                "type": "string_or_null",
                "values": ("timeout", "tracking_quality"),
                "terminal": True,
            },
            "press_depth": {
                "type": "float64",
                "units": "metres",
                "coordinate_frame": "button joint displacement",
            },
            "button_pressed": {"type": "boolean"},
            "dwell_steps": {"type": "integer", "units": "control steps"},
        },
    )


def _build_push_cube_spec(
    env: MujocoEnv,
    config: PushCubeConfig,
) -> TaskSpec:
    (
        qpos_names,
        qvel_names,
        actuator_names,
        finger_qpos_indices,
        finger_qvel_indices,
        finger_joint_names,
    ) = _mujoco_observation_order(env)
    action_schema = build_level1_action_schema(actuator_names)
    task_state_dim = 47 + int(env.model.nq) + int(env.model.nv)
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=int(env.model.nq),
        robot_qvel_dim=int(env.model.nv),
        finger_target_dim=int(env.model.nu),
        tracking_quality_dim=len(DEFAULT_TRACKING_QUALITY_NAMES),
        robot_qpos_names=qpos_names,
        robot_qvel_names=qvel_names,
        actuator_names=actuator_names,
        finger_joint_qpos_indices=finger_qpos_indices,
        finger_joint_qvel_indices=finger_qvel_indices,
        finger_joint_names=finger_joint_names,
        tracking_quality_names=DEFAULT_TRACKING_QUALITY_NAMES,
        object_state_dim=13,
        task_state_dim=task_state_dim,
        target_state_dim=7,
        success_metric_dim=8,
    )
    observation_schema.validate()
    action_schema.validate()
    return TaskSpec(
        task_id=PUSH_CUBE_TASK_ID,
        skill_name=PUSH_CUBE_TASK_ID,
        required_objects=config.object_ids + (config.target_marker_body,),
        observation_schema=observation_schema,
        action_schema=action_schema,
        success_condition=(
            "selected cube centre remains within "
            f"{config.target_radius_m:.3f} m planar distance of the target for "
            f"{config.success_dwell_steps} consecutive control steps"
        ),
        failure_conditions=(
            "timeout",
            "object_workspace_bounds",
            "tracking_quality",
        ),
        max_episode_steps=config.max_episode_steps,
        reset_config={
            "object_ids": config.object_ids,
            "object_start_sites": config.object_start_sites,
            "target_zone_sites": config.target_zone_sites,
            "target_radius_m": config.target_radius_m,
            "success_dwell_steps": config.success_dwell_steps,
            "deterministic_seed": True,
            "initial_object_state_saved": True,
            "initial_robot_state_saved": True,
        },
        parameter_type=PushCubeParameters,
        parameter_schema={
            "object_id": {
                "type": "string",
                "shape": (),
                "named_id_source": config.object_ids,
                "required": False,
            },
            "target_pose": {
                "type": "float64",
                "shape": (3,),
                "units": "metres",
                "coordinate_frame": "MuJoCo world cube centre",
                "required": False,
            },
            "target_zone_id": {
                "type": "string",
                "shape": (),
                "named_id_source": config.target_zone_sites,
                "required": False,
            },
            "approach_side": {
                "type": "string",
                "shape": (),
                "values": config.approach_sides,
                "required": False,
            },
        },
        state_fields=(
            "object_id",
            "object_index",
            "object_position",
            "object_orientation",
            "object_linear_velocity",
            "object_angular_velocity",
            "target_source",
            "target_index",
            "target_position",
            "target_radius",
            "approach_side",
            "distance_to_target",
            "within_target",
            "dwell_steps",
            "success",
            "failure_reason",
            "step_count",
            "initial_object_position",
            "initial_object_orientation",
            "initial_object_linear_velocity",
            "initial_object_angular_velocity",
            "initial_base_position",
            "initial_base_orientation",
            "initial_robot_qpos",
            "initial_robot_qvel",
        ),
        success_metric_inputs=(
            "object_position",
            "target_position",
            "distance_to_target",
            "dwell_steps",
        ),
        terminal_state_schema={
            "success": {"type": "boolean", "terminal": True},
            "failure_reason": {
                "type": "string_or_null",
                "values": (
                    "timeout",
                    "object_workspace_bounds",
                    "tracking_quality",
                ),
                "terminal": True,
            },
            "object_position": {
                "type": "float64",
                "shape": (3,),
                "units": "metres",
                "coordinate_frame": "MuJoCo world",
            },
            "object_linear_velocity": {
                "type": "float64",
                "shape": (3,),
                "units": "metres/second",
                "coordinate_frame": "MuJoCo free-joint velocity",
            },
            "distance_to_target": {"type": "float64", "units": "metres"},
            "dwell_steps": {"type": "integer", "units": "control steps"},
        },
    )


def _mujoco_observation_order(
    env: MujocoEnv,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    model = env.model
    mujoco_module = env._mujoco
    qpos_names: list[str] = []
    qvel_names: list[str] = []
    actuator_names = tuple(
        mujoco_module.mj_id2name(
            model,
            mujoco_module.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        or f"actuator_{actuator_id}"
        for actuator_id in range(model.nu)
    )
    finger_qpos_indices: list[int] = []
    finger_qvel_indices: list[int] = []
    finger_joint_names: list[str] = []
    free_type = int(mujoco_module.mjtJoint.mjJNT_FREE)
    ball_type = int(mujoco_module.mjtJoint.mjJNT_BALL)

    for joint_id in range(model.njnt):
        joint_name = (
            mujoco_module.mj_id2name(
                model,
                mujoco_module.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            or f"joint_{joint_id}"
        )
        qpos_start = int(model.jnt_qposadr[joint_id])
        qpos_stop = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < model.njnt
            else int(model.nq)
        )
        qvel_start = int(model.jnt_dofadr[joint_id])
        qvel_stop = (
            int(model.jnt_dofadr[joint_id + 1])
            if joint_id + 1 < model.njnt
            else int(model.nv)
        )
        joint_type = int(model.jnt_type[joint_id])
        qpos_suffixes: tuple[str, ...] = ()
        qvel_suffixes: tuple[str, ...] = ()
        if joint_type == free_type:
            qpos_suffixes = ("x", "y", "z", "qw", "qx", "qy", "qz")
            qvel_suffixes = ("vx", "vy", "vz", "wx", "wy", "wz")
        elif joint_type == ball_type:
            qpos_suffixes = ("qw", "qx", "qy", "qz")
            qvel_suffixes = ("wx", "wy", "wz")
        qpos_names.extend(
            (f"{joint_name}/{suffix}" for suffix in qpos_suffixes)
            if qpos_suffixes
            else (joint_name,)
        )
        qvel_names.extend(
            (f"{joint_name}/{suffix}" for suffix in qvel_suffixes)
            if qvel_suffixes
            else (joint_name,)
        )
        if qpos_stop - qpos_start == 1 and qvel_stop - qvel_start == 1:
            finger_qpos_indices.append(qpos_start)
            finger_qvel_indices.append(qvel_start)
            finger_joint_names.append(joint_name)

    if len(qpos_names) != model.nq or len(qvel_names) != model.nv:
        raise TaskError("failed to reconstruct named MuJoCo qpos/qvel order.")
    return (
        tuple(qpos_names),
        tuple(qvel_names),
        actuator_names,
        tuple(finger_qpos_indices),
        tuple(finger_qvel_indices),
        tuple(finger_joint_names),
    )


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values.")
    return vector
