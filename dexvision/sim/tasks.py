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
