"""MuJoCo hand-base mocap control utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from dexvision.features.hand_base import (
    HandScaleSource,
    HandBaseTarget,
    ImagePalmCenterTarget,
    PositionSource,
    no_hand_base_target,
    no_image_palm_center_target,
    normalize_quaternion,
    quaternion_angle,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_nlerp,
)
from dexvision.sim.mujoco_env import MujocoEnv


_IDENTITY_MATRIX = np.eye(3, dtype=np.float64)
BaseControlMode = Literal["image_2d", "pose_3d"]
ImageYAxisMode = Literal["height", "approach"]
PositionMode = Literal["absolute", "relative"]
RotationMode = Literal["palm_delta", "palm_absolute", "fixed"]
DepthAxis = Literal["x", "y", "z"]
OrientationMode = Literal["relative_palm"]
OrientationDof = Literal["roll", "pitch", "yaw"]
DEFAULT_MOCAP_BODY_NAME = "dexvision_hand_base_target"
DEFAULT_BASE_CONTROL_MODE: BaseControlMode = "image_2d"
DEFAULT_BASE_FIXED_Z = 0.14
DEFAULT_DEPTH_AXIS: DepthAxis = "x"
DEFAULT_DEPTH_SOURCE: HandScaleSource = "robust_palm_scale"
DEFAULT_DEPTH_MIN = -0.12
DEFAULT_DEPTH_MAX = 0.16
DEFAULT_ORIENTATION_DOFS: tuple[OrientationDof, ...] = ("roll", "pitch", "yaw")
DEFAULT_ROTATION_OFFSET_QUAT = np.asarray(
    [0.49939, 0.500609, 0.46235, -0.535007],
    dtype=np.float64,
)


@dataclass(frozen=True, eq=False)
class WorkspaceLimits:
    """Axis-aligned workspace clamp for the MuJoCo hand base."""

    minimum: np.ndarray = field(
        default_factory=lambda: np.asarray([-0.18, -0.18, 0.08], dtype=np.float64)
    )
    maximum: np.ndarray = field(
        default_factory=lambda: np.asarray([0.22, 0.18, 0.24], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        minimum = _coerce_vector(self.minimum, "workspace_limits.min")
        maximum = _coerce_vector(self.maximum, "workspace_limits.max")
        if not np.all(minimum <= maximum):
            raise ValueError("workspace_limits.min must be <= workspace_limits.max.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def clamp(self, position: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clamp ``position`` and report whether any axis changed."""

        raw_position = _coerce_vector(position, "position")
        clamped = np.clip(raw_position, self.minimum, self.maximum)
        return clamped, bool(not np.allclose(raw_position, clamped))


@dataclass(frozen=True, eq=False)
class HandBaseControlConfig:
    """Configuration for optional hand-base control."""

    enabled: bool = False
    mocap_body_name: str = DEFAULT_MOCAP_BODY_NAME
    base_control_mode: BaseControlMode = DEFAULT_BASE_CONTROL_MODE
    base_fixed_z: float = DEFAULT_BASE_FIXED_Z
    base_position_scale_x: float = 0.45
    base_position_scale_y: float = 0.35
    base_image_y_axis: ImageYAxisMode = "height"
    enable_depth_control: bool = False
    depth_source: HandScaleSource = DEFAULT_DEPTH_SOURCE
    depth_axis: DepthAxis = DEFAULT_DEPTH_AXIS
    depth_scale: float = 0.35
    depth_sign: float = 1.0
    depth_min: float = DEFAULT_DEPTH_MIN
    depth_max: float = DEFAULT_DEPTH_MAX
    depth_smoothing_alpha: float = 0.25
    depth_deadband: float = 0.03
    depth_hold_on_tracking_loss: bool = True
    position_source: PositionSource = "wrist"
    position_mode: PositionMode = "absolute"
    position_scale: np.ndarray = field(
        default_factory=lambda: np.asarray([0.35, 0.35, 0.35], dtype=np.float64)
    )
    position_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation_mode: RotationMode = "fixed"
    rotation_offset_quat: np.ndarray = field(
        default_factory=lambda: DEFAULT_ROTATION_OFFSET_QUAT.copy()
    )
    workspace_limits: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    base_smoothing_alpha: float = 0.25
    min_confidence: float = 0.2
    max_position_step: float = 0.025
    max_rotation_step_degrees: float = 3.0
    enable_base_orientation: bool = False
    orientation_mode: OrientationMode = "relative_palm"
    orientation_dofs: tuple[OrientationDof, ...] = DEFAULT_ORIENTATION_DOFS
    max_roll_deg: float = 45.0
    max_pitch_deg: float = 45.0
    max_yaw_deg: float = 45.0
    orientation_smoothing_alpha: float = 1.0
    orientation_deadband_deg: float = 0.0
    base_orientation_axis_signs: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )
    base_orientation_remap_matrix: np.ndarray = field(
        default_factory=lambda: _IDENTITY_MATRIX.copy()
    )

    def __post_init__(self) -> None:
        if not self.mocap_body_name:
            raise ValueError("mocap_body_name must be a non-empty string.")
        if self.base_control_mode not in ("image_2d", "pose_3d"):
            raise ValueError("base_control_mode must be 'image_2d' or 'pose_3d'.")
        if not np.isfinite(self.base_fixed_z):
            raise ValueError("base_fixed_z must be finite.")
        if not np.isfinite(self.base_position_scale_x):
            raise ValueError("base_position_scale_x must be finite.")
        if not np.isfinite(self.base_position_scale_y):
            raise ValueError("base_position_scale_y must be finite.")
        if self.base_image_y_axis not in ("height", "approach"):
            raise ValueError("base_image_y_axis must be 'height' or 'approach'.")
        if self.depth_source not in ("palm_width", "robust_palm_scale"):
            raise ValueError("depth_source must be 'palm_width' or 'robust_palm_scale'.")
        if self.depth_axis not in ("x", "y", "z", "approach", "lateral", "height"):
            raise ValueError("depth_axis must be 'x', 'y', 'z', 'approach', 'lateral', or 'height'.")
        if self.depth_scale < 0.0 or not np.isfinite(self.depth_scale):
            raise ValueError("depth_scale must be a finite non-negative value.")
        if not np.isfinite(self.depth_sign) or self.depth_sign == 0.0:
            raise ValueError("depth_sign must be finite and non-zero.")
        if not np.isfinite(self.depth_min) or not np.isfinite(self.depth_max):
            raise ValueError("depth_min and depth_max must be finite.")
        if self.depth_min > self.depth_max:
            raise ValueError("depth_min must be <= depth_max.")
        if not 0.0 < self.depth_smoothing_alpha <= 1.0:
            raise ValueError("depth_smoothing_alpha must be in the range (0.0, 1.0].")
        if self.depth_deadband < 0.0 or not np.isfinite(self.depth_deadband):
            raise ValueError("depth_deadband must be a finite non-negative value.")
        if self.position_source not in ("wrist", "palm_center"):
            raise ValueError("position_source must be 'wrist' or 'palm_center'.")
        if self.position_mode not in ("absolute", "relative"):
            raise ValueError("position_mode must be 'absolute' or 'relative'.")
        if self.rotation_mode not in ("palm_delta", "palm_absolute", "fixed"):
            raise ValueError("rotation_mode must be 'palm_delta', 'palm_absolute', or 'fixed'.")
        if not 0.0 < self.base_smoothing_alpha <= 1.0:
            raise ValueError("base_smoothing_alpha must be in the range (0.0, 1.0].")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in the range [0.0, 1.0].")
        if self.max_position_step <= 0.0:
            raise ValueError("max_position_step must be positive.")
        if self.max_rotation_step_degrees <= 0.0:
            raise ValueError("max_rotation_step_degrees must be positive.")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "base_control_mode",
            _coerce_base_control_mode(str(self.base_control_mode)),
        )
        object.__setattr__(self, "base_fixed_z", float(self.base_fixed_z))
        object.__setattr__(self, "base_position_scale_x", float(self.base_position_scale_x))
        object.__setattr__(self, "base_position_scale_y", float(self.base_position_scale_y))
        object.__setattr__(
            self,
            "base_image_y_axis",
            _coerce_image_y_axis(str(self.base_image_y_axis)),
        )
        object.__setattr__(self, "enable_depth_control", bool(self.enable_depth_control))
        object.__setattr__(
            self,
            "depth_source",
            _coerce_depth_source(str(self.depth_source)),
        )
        object.__setattr__(self, "depth_axis", _coerce_depth_axis(str(self.depth_axis)))
        object.__setattr__(self, "depth_scale", float(self.depth_scale))
        object.__setattr__(self, "depth_sign", 1.0 if self.depth_sign > 0.0 else -1.0)
        object.__setattr__(self, "depth_min", float(self.depth_min))
        object.__setattr__(self, "depth_max", float(self.depth_max))
        object.__setattr__(self, "depth_smoothing_alpha", float(self.depth_smoothing_alpha))
        object.__setattr__(self, "depth_deadband", float(self.depth_deadband))
        object.__setattr__(
            self,
            "depth_hold_on_tracking_loss",
            bool(self.depth_hold_on_tracking_loss),
        )
        object.__setattr__(
            self,
            "position_source",
            _coerce_position_source(str(self.position_source)),
        )
        object.__setattr__(
            self,
            "position_mode",
            _coerce_position_mode(str(self.position_mode)),
        )
        object.__setattr__(
            self,
            "position_scale",
            _coerce_vector(self.position_scale, "position_scale"),
        )
        object.__setattr__(
            self,
            "position_offset",
            _coerce_vector(self.position_offset, "position_offset"),
        )
        object.__setattr__(
            self,
            "rotation_offset_quat",
            _coerce_quaternion(self.rotation_offset_quat, "rotation_offset_quat"),
        )
        object.__setattr__(self, "enable_base_orientation", bool(self.enable_base_orientation))
        object.__setattr__(
            self,
            "orientation_mode",
            _coerce_orientation_mode(str(self.orientation_mode)),
        )
        object.__setattr__(
            self,
            "orientation_dofs",
            _coerce_orientation_dofs(self.orientation_dofs),
        )
        object.__setattr__(
            self,
            "max_roll_deg",
            _coerce_nonnegative_float(self.max_roll_deg, "max_roll_deg"),
        )
        object.__setattr__(
            self,
            "max_pitch_deg",
            _coerce_nonnegative_float(self.max_pitch_deg, "max_pitch_deg"),
        )
        object.__setattr__(
            self,
            "max_yaw_deg",
            _coerce_nonnegative_float(self.max_yaw_deg, "max_yaw_deg"),
        )
        object.__setattr__(
            self,
            "orientation_smoothing_alpha",
            _coerce_unit_interval_alpha(
                self.orientation_smoothing_alpha,
                "orientation_smoothing_alpha",
            ),
        )
        object.__setattr__(
            self,
            "orientation_deadband_deg",
            _coerce_nonnegative_float(
                self.orientation_deadband_deg,
                "orientation_deadband_deg",
            ),
        )
        object.__setattr__(
            self,
            "base_orientation_axis_signs",
            _coerce_orientation_axis_signs(self.base_orientation_axis_signs),
        )
        object.__setattr__(
            self,
            "base_orientation_remap_matrix",
            _coerce_orientation_remap_matrix(self.base_orientation_remap_matrix),
        )

    @property
    def smoothing_alpha(self) -> float:
        """Backward-compatible alias for the Level 1.13 base smoother alpha."""

        return self.base_smoothing_alpha

    @property
    def neutral_base_position(self) -> np.ndarray:
        """Return the configured neutral robot base pose position."""

        return np.asarray(
            [self.position_offset[0], self.position_offset[1], self.base_fixed_z],
            dtype=np.float64,
        )

    @property
    def depth_axis_index(self) -> int:
        """Return the MuJoCo position index controlled by depth."""

        return {"x": 0, "y": 1, "z": 2}[self.depth_axis]

    @property
    def orientation_axis_signs(self) -> np.ndarray:
        """Return the configured relative-orientation axis signs."""

        return self.base_orientation_axis_signs

    @property
    def orientation_remap_matrix(self) -> np.ndarray:
        """Return the configured relative-orientation axis remap matrix."""

        return self.base_orientation_remap_matrix

    @property
    def max_orientation_rpy_degrees(self) -> np.ndarray:
        """Return per-axis roll/pitch/yaw clamps in degrees."""

        return np.asarray(
            [self.max_roll_deg, self.max_pitch_deg, self.max_yaw_deg],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class HandBaseControlStatus:
    """Status returned after applying one hand-base target."""

    enabled: bool
    applied_target: HandBaseTarget
    tracking_valid: bool
    neutral_captured: bool
    clamped: bool
    rate_limited: bool
    control_mode: str = "off"
    orientation_enabled: bool = False
    palm_center: np.ndarray | None = None
    palm_delta: np.ndarray | None = None
    depth_enabled: bool = False
    hand_scale: float | None = None
    neutral_hand_scale: float | None = None
    depth_delta: float | None = None
    depth_target: float | None = None
    depth_axis: str = "x"
    depth_clamped: bool = False
    orientation_calibrated: bool = False
    orientation_delta_rpy_degrees: np.ndarray | None = None
    orientation_clamped: bool = False


@dataclass(frozen=True, eq=False)
class OrientationMappingResult:
    """Mapped relative palm rotation and debug values."""

    quaternion: np.ndarray
    rpy_degrees: np.ndarray
    clamped: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quaternion",
            normalize_quaternion(np.asarray(self.quaternion, dtype=np.float64)),
        )
        object.__setattr__(
            self,
            "rpy_degrees",
            _coerce_vector(self.rpy_degrees, "orientation_delta_rpy_degrees"),
        )
        object.__setattr__(self, "clamped", bool(self.clamped))


class HandBaseMocapController:
    """Map smoothed palm targets onto a MuJoCo mocap body."""

    def __init__(self, env: MujocoEnv, config: HandBaseControlConfig) -> None:
        self.env = env
        self.config = config
        _, neutral_quat = env.get_mocap_pose(config.mocap_body_name)
        self._neutral_mocap_quat = normalize_quaternion(neutral_quat)
        self._source_neutral: HandBaseTarget | None = None
        self._neutral_palm_center: np.ndarray | None = None
        self._neutral_hand_scale: float | None = None
        self._neutral_robot_position = self.config.workspace_limits.clamp(
            self.config.neutral_base_position
        )[0]
        self._neutral_palm_orientation_quat: np.ndarray | None = None
        self._neutral_robot_orientation_quat = self._neutral_mocap_quat.copy()
        self._last_applied = HandBaseTarget(
            position=self.config.workspace_limits.clamp(self.config.neutral_base_position)[0],
            orientation_quat=self._neutral_mocap_quat,
            confidence=0.0,
            valid=True,
        )
        self.reset_to_neutral()

    def reset_source_neutral(self) -> None:
        """Use the next valid tracked target as the user's neutral base pose."""

        self._source_neutral = None

    def reset_image_calibration(self) -> None:
        """Clear the calibrated image-space palm pose neutral."""

        self._neutral_palm_center = None
        self._neutral_hand_scale = None
        self._neutral_robot_position = self.config.workspace_limits.clamp(
            self.config.neutral_base_position
        )[0]
        self._neutral_palm_orientation_quat = None
        self._neutral_robot_orientation_quat = self._neutral_mocap_quat.copy()

    @property
    def neutral_palm_orientation_quat(self) -> np.ndarray | None:
        """Return the calibrated human palm orientation, if captured."""

        if self._neutral_palm_orientation_quat is None:
            return None
        return self._neutral_palm_orientation_quat.copy()

    @property
    def neutral_robot_orientation_quat(self) -> np.ndarray:
        """Return the robot base orientation captured at calibration."""

        return self._neutral_robot_orientation_quat.copy()

    def calibrate_image_2d(
        self,
        target: ImagePalmCenterTarget | None,
        *,
        orientation_target: HandBaseTarget | None = None,
    ) -> bool:
        """Capture the current image palm pose as the human neutral pose."""

        current = target or no_image_palm_center_target()
        tracking_valid = bool(current.valid and current.confidence >= self.config.min_confidence)
        if not tracking_valid:
            return False
        if self.config.enable_depth_control and not _is_valid_hand_scale(current.hand_scale):
            return False
        if self.config.enable_base_orientation:
            if (
                orientation_target is None
                or not orientation_target.valid
                or orientation_target.confidence < self.config.min_confidence
            ):
                return False
            self._neutral_palm_orientation_quat = normalize_quaternion(
                orientation_target.orientation_quat,
            )
        elif orientation_target is not None and orientation_target.valid:
            self._neutral_palm_orientation_quat = normalize_quaternion(
                orientation_target.orientation_quat,
            )
        self._neutral_palm_center = current.palm_center.copy()
        self._neutral_hand_scale = (
            current.hand_scale if _is_valid_hand_scale(current.hand_scale) else None
        )
        self.reset_to_neutral(clear_image_calibration=False)
        self._neutral_robot_position = self._last_applied.position.copy()
        self._neutral_robot_orientation_quat = self._last_applied.orientation_quat.copy()
        return True

    def reset_to_neutral(self, *, clear_image_calibration: bool = False) -> HandBaseControlStatus:
        """Move the mocap body back to its configured neutral pose."""

        if clear_image_calibration:
            self.reset_image_calibration()
            self.reset_source_neutral()
        position, clamped = self.config.workspace_limits.clamp(self.config.neutral_base_position)
        self._last_applied = HandBaseTarget(
            position=position,
            orientation_quat=self._neutral_mocap_quat,
            confidence=0.0,
            valid=True,
        )
        if self.config.enabled:
            self.env.set_mocap_pose(
                self.config.mocap_body_name,
                position=self._last_applied.position,
                orientation_quat=self._last_applied.orientation_quat,
            )
        return HandBaseControlStatus(
            enabled=self.config.enabled,
            applied_target=self._last_applied,
            tracking_valid=False,
            neutral_captured=self._neutral_is_captured(),
            clamped=clamped,
            rate_limited=False,
            control_mode=self.config.base_control_mode,
            orientation_enabled=self.config.enable_base_orientation,
            palm_center=None,
            palm_delta=None,
            depth_enabled=self.config.enable_depth_control,
            hand_scale=None,
            neutral_hand_scale=self._neutral_hand_scale,
            depth_delta=None,
            depth_target=self._last_applied.position[self.config.depth_axis_index],
            depth_axis=self.config.depth_axis,
            depth_clamped=False,
            orientation_calibrated=self._orientation_is_calibrated(),
            orientation_delta_rpy_degrees=None,
            orientation_clamped=False,
        )

    def apply(self, target: HandBaseTarget | None) -> HandBaseControlStatus:
        """Apply a smoothed source target to the MuJoCo mocap body."""

        if not self.config.enabled:
            return HandBaseControlStatus(
                enabled=False,
                applied_target=self._last_applied,
                tracking_valid=False,
                neutral_captured=self._neutral_is_captured(),
                clamped=False,
                rate_limited=False,
                control_mode="off",
                orientation_enabled=False,
                orientation_calibrated=False,
                orientation_clamped=False,
            )

        current = target or no_hand_base_target()
        tracking_valid = bool(current.valid and current.confidence >= self.config.min_confidence)
        if not tracking_valid:
            self.env.set_mocap_pose(
                self.config.mocap_body_name,
                position=self._last_applied.position,
                orientation_quat=self._last_applied.orientation_quat,
            )
            return HandBaseControlStatus(
                enabled=True,
                applied_target=self._last_applied,
                tracking_valid=False,
                neutral_captured=self._neutral_is_captured(),
                clamped=False,
                rate_limited=False,
                control_mode=self.config.base_control_mode,
                orientation_enabled=self.config.enable_base_orientation,
                orientation_calibrated=self._orientation_is_calibrated(),
                orientation_clamped=False,
            )

        if self._source_neutral is None:
            self._source_neutral = current

        mapped_target, clamped = self._map_to_mujoco_target(current)
        mapped_target, rate_limited = self._limit_step(mapped_target)
        self._last_applied = mapped_target
        self.env.set_mocap_pose(
            self.config.mocap_body_name,
            position=mapped_target.position,
            orientation_quat=mapped_target.orientation_quat,
        )
        return HandBaseControlStatus(
            enabled=True,
            applied_target=mapped_target,
            tracking_valid=True,
            neutral_captured=True,
            clamped=clamped,
            rate_limited=rate_limited,
            control_mode=self.config.base_control_mode,
            orientation_enabled=self.config.enable_base_orientation,
            orientation_calibrated=self._orientation_is_calibrated(),
            orientation_clamped=False,
        )

    def apply_image_2d(
        self,
        target: ImagePalmCenterTarget | None,
        *,
        orientation_target: HandBaseTarget | None = None,
    ) -> HandBaseControlStatus:
        """Apply calibrated image-space palm-center translation to the mocap body."""

        if not self.config.enabled:
            return HandBaseControlStatus(
                enabled=False,
                applied_target=self._last_applied,
                tracking_valid=False,
                neutral_captured=self._neutral_is_captured(),
                clamped=False,
                rate_limited=False,
                control_mode="off",
                orientation_enabled=False,
                orientation_calibrated=False,
                orientation_clamped=False,
            )

        current = target or no_image_palm_center_target()
        tracking_valid = bool(current.valid and current.confidence >= self.config.min_confidence)
        if not tracking_valid:
            rate_limited = False
            if self.config.enable_depth_control and not self.config.depth_hold_on_tracking_loss:
                depth_axis_index = self.config.depth_axis_index
                target_position = self._last_applied.position.copy()
                target_position[depth_axis_index] = self._neutral_robot_position[depth_axis_index]
                smoothed_position = target_position.copy()
                smoothed_position[depth_axis_index] = (
                    self.config.depth_smoothing_alpha * target_position[depth_axis_index]
                ) + (
                    (1.0 - self.config.depth_smoothing_alpha)
                    * self._last_applied.position[depth_axis_index]
                )
                decay_target = HandBaseTarget(
                    position=smoothed_position,
                    orientation_quat=self._last_applied.orientation_quat,
                    confidence=current.confidence,
                    valid=True,
                )
                self._last_applied, rate_limited = self._limit_step(decay_target)
            self.env.set_mocap_pose(
                self.config.mocap_body_name,
                position=self._last_applied.position,
                orientation_quat=self._last_applied.orientation_quat,
            )
            return HandBaseControlStatus(
                enabled=True,
                applied_target=self._last_applied,
                tracking_valid=False,
                neutral_captured=self._neutral_is_captured(),
                clamped=False,
                rate_limited=False,
                control_mode=self.config.base_control_mode,
                orientation_enabled=self.config.enable_base_orientation,
                palm_center=None,
                palm_delta=None,
                depth_enabled=self.config.enable_depth_control,
                hand_scale=None,
                neutral_hand_scale=self._neutral_hand_scale,
                depth_delta=None,
                depth_target=self._last_applied.position[self.config.depth_axis_index],
                depth_axis=self.config.depth_axis,
                depth_clamped=False,
                orientation_calibrated=self._orientation_is_calibrated(),
                orientation_delta_rpy_degrees=None,
                orientation_clamped=False,
            )

        if self._neutral_palm_center is None or (
            self.config.enable_depth_control and self._neutral_hand_scale is None
        ):
            neutral_target = HandBaseTarget(
                position=self.config.workspace_limits.clamp(self.config.neutral_base_position)[0],
                orientation_quat=self._neutral_mocap_quat,
                confidence=current.confidence,
                valid=True,
            )
            neutral_target, rate_limited = self._limit_step(neutral_target)
            self._last_applied = neutral_target
            self.env.set_mocap_pose(
                self.config.mocap_body_name,
                position=neutral_target.position,
                orientation_quat=neutral_target.orientation_quat,
            )
            return HandBaseControlStatus(
                enabled=True,
                applied_target=neutral_target,
                tracking_valid=True,
                neutral_captured=False,
                clamped=False,
                rate_limited=rate_limited,
                control_mode=self.config.base_control_mode,
                orientation_enabled=self.config.enable_base_orientation,
                palm_center=current.palm_center.copy(),
                palm_delta=np.zeros(2, dtype=np.float64),
                depth_enabled=self.config.enable_depth_control,
                hand_scale=current.hand_scale,
                neutral_hand_scale=self._neutral_hand_scale,
                depth_delta=0.0,
                depth_target=neutral_target.position[self.config.depth_axis_index],
                depth_axis=self.config.depth_axis,
                depth_clamped=False,
                orientation_calibrated=self._orientation_is_calibrated(),
                orientation_delta_rpy_degrees=None,
                orientation_clamped=False,
            )

        delta = palm_center_delta(current.palm_center, self._neutral_palm_center)
        mapped_position, clamped = map_image_delta_to_base_position(
            delta,
            self.config,
            neutral_position=self._neutral_robot_position,
        )
        depth_delta = None
        depth_clamped = False
        if self.config.enable_depth_control:
            depth_target, depth_delta, depth_clamped = map_hand_scale_to_depth_target(
                current.hand_scale,
                self._neutral_hand_scale,
                self._neutral_robot_position,
                self.config,
            )
            mapped_position[self.config.depth_axis_index] = depth_target
            unclamped_depth_target = depth_target
            mapped_position, workspace_clamped = self.config.workspace_limits.clamp(mapped_position)
            depth_clamped = depth_clamped or (
                not np.isclose(mapped_position[self.config.depth_axis_index], unclamped_depth_target)
            )
            clamped = clamped or depth_clamped or workspace_clamped
        smoothed_position = _smooth_image_2d_position(
            previous=self._last_applied.position,
            current=mapped_position,
            config=self.config,
        )
        (
            orientation_quat,
            orientation_delta_rpy_degrees,
            orientation_clamped,
        ) = self._image_2d_orientation(
            orientation_target,
        )
        mapped_target = HandBaseTarget(
            position=smoothed_position,
            orientation_quat=orientation_quat,
            confidence=current.confidence,
            valid=True,
        )
        mapped_target, rate_limited = self._limit_step(mapped_target)
        self._last_applied = mapped_target
        self.env.set_mocap_pose(
            self.config.mocap_body_name,
            position=mapped_target.position,
            orientation_quat=mapped_target.orientation_quat,
        )
        return HandBaseControlStatus(
            enabled=True,
            applied_target=mapped_target,
            tracking_valid=True,
            neutral_captured=True,
            clamped=clamped,
            rate_limited=rate_limited,
            control_mode=self.config.base_control_mode,
            orientation_enabled=self.config.enable_base_orientation,
            palm_center=current.palm_center.copy(),
            palm_delta=delta,
            depth_enabled=self.config.enable_depth_control,
            hand_scale=current.hand_scale,
            neutral_hand_scale=self._neutral_hand_scale,
            depth_delta=depth_delta,
            depth_target=mapped_target.position[self.config.depth_axis_index],
            depth_axis=self.config.depth_axis,
            depth_clamped=depth_clamped,
            orientation_calibrated=self._orientation_is_calibrated(),
            orientation_delta_rpy_degrees=orientation_delta_rpy_degrees,
            orientation_clamped=orientation_clamped,
        )

    def _map_to_mujoco_target(self, target: HandBaseTarget) -> tuple[HandBaseTarget, bool]:
        source_neutral = self._source_neutral or target
        if self.config.position_mode == "absolute":
            position_signal = target.position
        else:
            position_signal = target.position - source_neutral.position
        raw_position = self.config.position_offset + (self.config.position_scale * position_signal)
        position, clamped = self.config.workspace_limits.clamp(raw_position)

        if not self.config.enable_base_orientation or self.config.rotation_mode == "fixed":
            orientation_quat = self._neutral_mocap_quat
        elif self.config.rotation_mode == "palm_absolute":
            orientation_quat = quaternion_multiply(
                self.config.rotation_offset_quat,
                target.orientation_quat,
            )
        else:
            source_delta_quat = quaternion_multiply(
                target.orientation_quat,
                quaternion_inverse(source_neutral.orientation_quat),
            )
            orientation_quat = quaternion_multiply(source_delta_quat, self._neutral_mocap_quat)

        return (
            HandBaseTarget(
                position=position,
                orientation_quat=orientation_quat,
                confidence=target.confidence,
                valid=True,
            ),
            clamped,
        )

    def _image_2d_orientation(
        self,
        orientation_target: HandBaseTarget | None,
    ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        if not self.config.enable_base_orientation:
            return self._neutral_mocap_quat, None, False
        if self._neutral_palm_orientation_quat is None:
            return self._last_applied.orientation_quat, None, False
        if orientation_target is None:
            return self._last_applied.orientation_quat, None, False
        if (
            not orientation_target.valid
            or orientation_target.confidence < self.config.min_confidence
        ):
            return self._last_applied.orientation_quat, None, False
        source_delta_quat = quaternion_multiply(
            orientation_target.orientation_quat,
            quaternion_inverse(self._neutral_palm_orientation_quat),
        )
        mapped_delta = map_orientation_delta_to_robot_with_status(source_delta_quat, self.config)
        target_orientation_quat = quaternion_multiply(
            mapped_delta.quaternion,
            self._neutral_robot_orientation_quat,
        )
        orientation_quat = quaternion_nlerp(
            self._last_applied.orientation_quat,
            target_orientation_quat,
            self.config.orientation_smoothing_alpha,
        )
        return orientation_quat, mapped_delta.rpy_degrees, mapped_delta.clamped

    def _limit_step(self, target: HandBaseTarget) -> tuple[HandBaseTarget, bool]:
        position, position_limited = _limit_position_step(
            self._last_applied.position,
            target.position,
            self.config.max_position_step,
        )
        orientation_quat, rotation_limited = _limit_rotation_step(
            self._last_applied.orientation_quat,
            target.orientation_quat,
            np.deg2rad(self.config.max_rotation_step_degrees),
        )
        if not position_limited and not rotation_limited:
            return target, False
        return (
            HandBaseTarget(
                position=position,
                orientation_quat=orientation_quat,
                confidence=target.confidence,
                valid=target.valid,
            ),
            True,
        )

    def _neutral_is_captured(self) -> bool:
        if self.config.base_control_mode == "image_2d":
            image_calibrated = self._neutral_palm_center is not None
            if self.config.enable_depth_control:
                image_calibrated = image_calibrated and self._neutral_hand_scale is not None
            if self.config.enable_base_orientation:
                return image_calibrated and self._orientation_is_calibrated()
            return image_calibrated
        return self._source_neutral is not None

    def _orientation_is_calibrated(self) -> bool:
        return self._neutral_palm_orientation_quat is not None


def hand_base_config_from_teleop_config(
    raw_config: object,
    *,
    enable_override: bool = False,
) -> HandBaseControlConfig:
    """Load hand-base config from the Level 1 teleop YAML mapping."""

    if not isinstance(raw_config, Mapping):
        config = HandBaseControlConfig()
    else:
        base_control = raw_config.get("base_control", {})
        if base_control is None:
            base_control = {}
        if not isinstance(base_control, Mapping):
            raise ValueError("base_control must be a mapping when provided.")
        config = hand_base_config_from_mapping(base_control)
    if enable_override:
        return replace(config, enabled=True)
    return config


def hand_base_config_from_mapping(raw_config: Mapping[str, Any] | None) -> HandBaseControlConfig:
    """Coerce a raw config mapping into ``HandBaseControlConfig``."""

    if raw_config is None:
        return HandBaseControlConfig()

    workspace_limits = _read_workspace_limits(raw_config)

    rotation_mode = str(raw_config.get("rotation_mode", "fixed"))
    orientation_axis_signs = raw_config.get(
        "orientation_axis_signs",
        raw_config.get("base_orientation_axis_signs", [1.0, 1.0, 1.0]),
    )
    orientation_remap_matrix = raw_config.get(
        "orientation_remap_matrix",
        raw_config.get("base_orientation_remap_matrix", _IDENTITY_MATRIX),
    )

    return HandBaseControlConfig(
        enabled=bool(raw_config.get("enable_base_control", False)),
        mocap_body_name=str(raw_config.get("mocap_body", DEFAULT_MOCAP_BODY_NAME)),
        base_control_mode=_coerce_base_control_mode(
            str(raw_config.get("base_control_mode", DEFAULT_BASE_CONTROL_MODE))
        ),
        base_fixed_z=_coerce_float(
            raw_config.get("base_fixed_z", DEFAULT_BASE_FIXED_Z),
            "base_fixed_z",
        ),
        base_position_scale_x=_coerce_float(
            raw_config.get("base_position_scale_x", 0.45),
            "base_position_scale_x",
        ),
        base_position_scale_y=_coerce_float(
            raw_config.get("base_position_scale_y", 0.35),
            "base_position_scale_y",
        ),
        base_image_y_axis=_coerce_image_y_axis(
            str(raw_config.get("base_image_y_axis", "height"))
        ),
        enable_depth_control=bool(raw_config.get("enable_depth_control", False)),
        depth_source=_coerce_depth_source(
            str(raw_config.get("depth_source", DEFAULT_DEPTH_SOURCE))
        ),
        depth_axis=_coerce_depth_axis(str(raw_config.get("depth_axis", DEFAULT_DEPTH_AXIS))),
        depth_scale=_coerce_nonnegative_float(raw_config.get("depth_scale", 0.35), "depth_scale"),
        depth_sign=_coerce_depth_sign(raw_config.get("depth_sign", 1.0)),
        depth_min=_coerce_float(raw_config.get("depth_min", DEFAULT_DEPTH_MIN), "depth_min"),
        depth_max=_coerce_float(raw_config.get("depth_max", DEFAULT_DEPTH_MAX), "depth_max"),
        depth_smoothing_alpha=_coerce_float(
            raw_config.get("depth_smoothing_alpha", 0.25),
            "depth_smoothing_alpha",
        ),
        depth_deadband=_coerce_nonnegative_float(
            raw_config.get("depth_deadband", 0.03),
            "depth_deadband",
        ),
        depth_hold_on_tracking_loss=bool(raw_config.get("depth_hold_on_tracking_loss", True)),
        position_source=_coerce_position_source(str(raw_config.get("position_source", "wrist"))),
        position_mode=_coerce_position_mode(str(raw_config.get("position_mode", "absolute"))),
        position_scale=_coerce_scale(raw_config.get("position_scale", [0.35, 0.35, 0.35])),
        position_offset=_coerce_vector(
            raw_config.get("position_offset", [0.0, 0.0, 0.0]),
            "position_offset",
        ),
        rotation_mode=_coerce_rotation_mode(rotation_mode),
        rotation_offset_quat=_coerce_quaternion(
            raw_config.get("rotation_offset_quat", DEFAULT_ROTATION_OFFSET_QUAT),
            "rotation_offset_quat",
        ),
        workspace_limits=workspace_limits,
        base_smoothing_alpha=_coerce_float(
            raw_config.get("base_smoothing_alpha", raw_config.get("smoothing_alpha", 0.25)),
            "base_smoothing_alpha",
        ),
        min_confidence=_coerce_float(
            raw_config.get("min_confidence", 0.2),
            "min_confidence",
        ),
        max_position_step=_coerce_positive_float(
            raw_config.get("max_position_step", 0.025),
            "max_position_step",
        ),
        max_rotation_step_degrees=_coerce_positive_float(
            raw_config.get("max_rotation_step_degrees", 3.0),
            "max_rotation_step_degrees",
        ),
        enable_base_orientation=bool(raw_config.get("enable_base_orientation", False)),
        orientation_mode=_coerce_orientation_mode(
            str(raw_config.get("orientation_mode", "relative_palm"))
        ),
        orientation_dofs=_coerce_orientation_dofs(
            raw_config.get("orientation_dofs", DEFAULT_ORIENTATION_DOFS)
        ),
        max_roll_deg=_coerce_nonnegative_float(
            raw_config.get("max_roll_deg", 45.0),
            "max_roll_deg",
        ),
        max_pitch_deg=_coerce_nonnegative_float(
            raw_config.get("max_pitch_deg", 45.0),
            "max_pitch_deg",
        ),
        max_yaw_deg=_coerce_nonnegative_float(raw_config.get("max_yaw_deg", 45.0), "max_yaw_deg"),
        orientation_smoothing_alpha=_coerce_unit_interval_alpha(
            raw_config.get("orientation_smoothing_alpha", 1.0),
            "orientation_smoothing_alpha",
        ),
        orientation_deadband_deg=_coerce_nonnegative_float(
            raw_config.get("orientation_deadband_deg", 0.0),
            "orientation_deadband_deg",
        ),
        base_orientation_axis_signs=_coerce_orientation_axis_signs(
            orientation_axis_signs
        ),
        base_orientation_remap_matrix=_coerce_orientation_remap_matrix(
            orientation_remap_matrix
        ),
    )


def format_hand_base_status(status: HandBaseControlStatus | None) -> str:
    """Format compact hand-base status for console and camera overlays."""

    if status is None or not status.enabled:
        return "base=off"
    position = status.applied_target.position
    mode = "tracking" if status.tracking_valid else "holding"
    clamp_suffix = ",clamped" if status.clamped else ""
    limit_suffix = ",limited" if status.rate_limited else ""
    if status.control_mode == "image_2d":
        calibration = "cal=yes" if status.neutral_captured else "cal=no"
        palm_delta = (
            status.palm_delta
            if status.palm_delta is not None
            else np.zeros(2, dtype=np.float64)
        )
        orientation = "ori=on" if status.orientation_enabled else "ori=off"
        depth = "depth=on" if status.depth_enabled else "depth=off"
        scale = _format_optional_status_float(status.hand_scale)
        neutral_scale = _format_optional_status_float(status.neutral_hand_scale)
        depth_delta = _format_optional_status_float(status.depth_delta, signed=True)
        depth_target = _format_optional_status_float(status.depth_target, signed=True)
        depth_clamp = "yes" if status.depth_clamped else "no"
        if not status.orientation_enabled:
            orientation_detail = "ori=off"
        elif status.orientation_calibrated and status.orientation_delta_rpy_degrees is not None:
            rpy = status.orientation_delta_rpy_degrees
            orientation_clamp = "yes" if status.orientation_clamped else "no"
            orientation_detail = (
                f"ori r={rpy[0]:+.1f} p={rpy[1]:+.1f} y={rpy[2]:+.1f} "
                f"clamp={orientation_clamp}"
            )
        elif status.orientation_calibrated:
            orientation_detail = "ori=on r=+0.0 p=+0.0 y=+0.0 clamp=no"
        else:
            orientation_detail = "ori=on cal=no"
        return (
            f"base=image_2d {calibration} {mode}{clamp_suffix}{limit_suffix} "
            f"{orientation} {depth} | "
            f"palm dx={palm_delta[0]:+.3f} dy={palm_delta[1]:+.3f} | "
            f"scale={scale} neutral={neutral_scale} d={depth_delta} | "
            f"depth {status.depth_axis}={depth_target} "
            f"clamp={depth_clamp} | "
            f"target x={position[0]:+.3f} y={position[1]:+.3f} z={position[2]:+.3f} | "
            f"{orientation_detail}"
        )
    return (
        f"base={mode}{clamp_suffix}{limit_suffix}:"
        f"x={position[0]:+.3f},y={position[1]:+.3f},z={position[2]:+.3f}"
    )


def palm_center_delta(current: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """Return normalized image palm-center delta from calibration neutral."""

    return _coerce_image_vector(current, "current_palm_center") - _coerce_image_vector(
        neutral,
        "neutral_palm_center",
    )


def map_image_delta_to_base_position(
    palm_delta: np.ndarray,
    config: HandBaseControlConfig,
    *,
    neutral_position: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Map image-space palm delta to a fixed-height MuJoCo base position."""

    delta = _coerce_image_vector(palm_delta, "palm_delta")
    raw_position = (
        config.neutral_base_position.copy()
        if neutral_position is None
        else _coerce_vector(neutral_position, "neutral_position").copy()
    )
    raw_position[1] += config.base_position_scale_x * delta[0]
    if config.base_image_y_axis == "approach":
        raw_position[0] -= config.base_position_scale_y * delta[1]
        raw_position[2] = config.base_fixed_z
    else:
        raw_position[2] -= config.base_position_scale_y * delta[1]
    return config.workspace_limits.clamp(raw_position)


def map_hand_scale_to_depth_target(
    hand_scale: float,
    neutral_hand_scale: float | None,
    neutral_position: np.ndarray,
    config: HandBaseControlConfig,
) -> tuple[float, float, bool]:
    """Map image-space hand-scale change to a MuJoCo base-axis target.

    Returns ``(depth_target, depth_delta, clamped)``. ``depth_delta`` is the
    deadbanded relative scale change, where positive means the hand appears
    larger than the calibration pose.
    """

    neutral_scale = 0.0 if neutral_hand_scale is None else float(neutral_hand_scale)
    if not _is_valid_hand_scale(hand_scale) or not _is_valid_hand_scale(neutral_scale):
        return (
            float(_coerce_vector(neutral_position, "neutral_position")[config.depth_axis_index]),
            0.0,
            False,
        )
    neutral = _coerce_vector(neutral_position, "neutral_position")
    raw_delta = (float(hand_scale) / neutral_scale) - 1.0
    depth_delta = _apply_deadband(raw_delta, config.depth_deadband)
    raw_target = (
        neutral[config.depth_axis_index]
        + (config.depth_sign * config.depth_scale * depth_delta)
    )
    clamped_target = float(np.clip(raw_target, config.depth_min, config.depth_max))
    return clamped_target, depth_delta, bool(not np.isclose(raw_target, clamped_target))


def map_orientation_delta_to_robot(
    human_delta_quat: np.ndarray,
    config: HandBaseControlConfig,
) -> np.ndarray:
    """Map a calibrated human palm rotation delta into the robot frame."""

    return map_orientation_delta_to_robot_with_status(human_delta_quat, config).quaternion


def map_orientation_delta_to_robot_with_status(
    human_delta_quat: np.ndarray,
    config: HandBaseControlConfig,
) -> OrientationMappingResult:
    """Map a human palm delta into the robot frame with RPY debug metadata."""

    rotation_vector = quaternion_to_rotation_vector(human_delta_quat)
    signed_vector = config.base_orientation_axis_signs * rotation_vector
    mapped_vector = np.asarray(
        config.base_orientation_remap_matrix @ signed_vector,
        dtype=np.float64,
    )
    mapped_quat = rotation_vector_to_quaternion(mapped_vector)
    mapped_rpy_degrees = quaternion_to_euler_degrees(mapped_quat)
    staged_rpy_degrees = _stage_orientation_rpy_degrees(mapped_rpy_degrees, config)
    clamped_rpy_degrees, clamped = _clamp_orientation_rpy_degrees(
        staged_rpy_degrees,
        config,
    )
    return OrientationMappingResult(
        quaternion=euler_degrees_to_quaternion(clamped_rpy_degrees),
        rpy_degrees=clamped_rpy_degrees,
        clamped=clamped,
    )


def euler_degrees_to_quaternion(rpy_degrees: np.ndarray) -> np.ndarray:
    """Convert XYZ roll/pitch/yaw degrees to a ``[w, x, y, z]`` quaternion."""

    roll, pitch, yaw = np.deg2rad(_coerce_vector(rpy_degrees, "rpy_degrees"))
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    return normalize_quaternion(
        np.asarray(
            [
                (cr * cp * cy) + (sr * sp * sy),
                (sr * cp * cy) - (cr * sp * sy),
                (cr * sp * cy) + (sr * cp * sy),
                (cr * cp * sy) - (sr * sp * cy),
            ],
            dtype=np.float64,
        )
    )


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a shortest-path rotation vector."""

    quat = normalize_quaternion(quaternion)
    vector = quat[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = float(2.0 * np.arctan2(vector_norm, quat[0]))
    return np.asarray((vector / vector_norm) * angle, dtype=np.float64)


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a MuJoCo-compatible ``[w, x, y, z]`` quaternion."""

    vector = _coerce_vector(rotation_vector, "rotation_vector")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = vector / angle
    half_angle = 0.5 * angle
    return normalize_quaternion(
        np.asarray(
            [
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ],
            dtype=np.float64,
        )
    )


def quaternion_to_euler_degrees(quaternion: np.ndarray) -> np.ndarray:
    """Return approximate XYZ roll/pitch/yaw angles in degrees."""

    w, x, y, z = normalize_quaternion(quaternion)
    roll = np.arctan2(2.0 * ((w * x) + (y * z)), 1.0 - (2.0 * ((x * x) + (y * y))))
    pitch = np.arcsin(np.clip(2.0 * ((w * y) - (z * x)), -1.0, 1.0))
    yaw = np.arctan2(2.0 * ((w * z) + (x * y)), 1.0 - (2.0 * ((y * y) + (z * z))))
    return np.rad2deg(np.asarray([roll, pitch, yaw], dtype=np.float64))


def _stage_orientation_rpy_degrees(
    rpy_degrees: np.ndarray,
    config: HandBaseControlConfig,
) -> np.ndarray:
    staged = np.zeros(3, dtype=np.float64)
    raw = _coerce_vector(rpy_degrees, "orientation_rpy_degrees")
    dof_to_index = {"roll": 0, "pitch": 1, "yaw": 2}
    for dof in config.orientation_dofs:
        staged[dof_to_index[dof]] = raw[dof_to_index[dof]]
    if config.orientation_deadband_deg <= 0.0:
        return staged
    return np.asarray(
        [
            _apply_deadband(value, config.orientation_deadband_deg)
            for value in staged
        ],
        dtype=np.float64,
    )


def _clamp_orientation_rpy_degrees(
    rpy_degrees: np.ndarray,
    config: HandBaseControlConfig,
) -> tuple[np.ndarray, bool]:
    raw = _coerce_vector(rpy_degrees, "orientation_rpy_degrees")
    maximum = config.max_orientation_rpy_degrees
    clamped = np.clip(raw, -maximum, maximum)
    return clamped, bool(not np.allclose(raw, clamped))


def _limit_position_step(
    previous: np.ndarray,
    target: np.ndarray,
    max_step: float,
) -> tuple[np.ndarray, bool]:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(previous, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance <= max_step:
        return target, False
    return np.asarray(previous + ((delta / distance) * max_step), dtype=np.float64), True


def _limit_rotation_step(
    previous: np.ndarray,
    target: np.ndarray,
    max_angle: float,
) -> tuple[np.ndarray, bool]:
    angle = quaternion_angle(previous, target)
    if angle <= max_angle:
        return normalize_quaternion(target), False
    fraction = max_angle / angle
    return quaternion_nlerp(previous, target, fraction), True


def _ema_position(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray((alpha * current) + ((1.0 - alpha) * previous), dtype=np.float64)


def _smooth_image_2d_position(
    *,
    previous: np.ndarray,
    current: np.ndarray,
    config: HandBaseControlConfig,
) -> np.ndarray:
    previous_position = _coerce_vector(previous, "previous_position")
    current_position = _coerce_vector(current, "current_position")
    smoothed = _ema_position(
        previous_position,
        current_position,
        config.base_smoothing_alpha,
    )
    if config.enable_depth_control:
        axis_index = config.depth_axis_index
        smoothed[axis_index] = (
            config.depth_smoothing_alpha * current_position[axis_index]
        ) + ((1.0 - config.depth_smoothing_alpha) * previous_position[axis_index])
    return smoothed


def _apply_deadband(value: float, deadband: float) -> float:
    safe_value = float(value)
    if abs(safe_value) <= deadband:
        return 0.0
    return safe_value - (float(np.sign(safe_value)) * deadband)


def _is_valid_hand_scale(value: float) -> bool:
    return bool(np.isfinite(value) and value > 1e-9)


def _format_optional_status_float(value: float | None, *, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _read_workspace_limits(raw_config: Mapping[str, Any]) -> WorkspaceLimits:
    legacy_limits = raw_config.get("workspace_limits", {})
    if legacy_limits is None:
        legacy_limits = {}
    if not isinstance(legacy_limits, Mapping):
        raise ValueError("base_control.workspace_limits must be a mapping.")
    minimum = raw_config.get("base_workspace_min", legacy_limits.get("min", [-0.18, -0.18, 0.08]))
    maximum = raw_config.get("base_workspace_max", legacy_limits.get("max", [0.22, 0.18, 0.24]))
    return WorkspaceLimits(
        minimum=_coerce_vector(minimum, "base_workspace_min"),
        maximum=_coerce_vector(maximum, "base_workspace_max"),
    )


def _coerce_scale(value: object) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.asarray([float(value), float(value), float(value)], dtype=np.float64)
    return _coerce_vector(value, "position_scale")


def _coerce_vector(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be a finite numeric sequence with shape [3].")
    return array


def _coerce_image_vector(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be a finite numeric sequence with shape [2].")
    return array


def _coerce_orientation_axis_signs(value: object) -> np.ndarray:
    signs = _coerce_vector(value, "base_orientation_axis_signs")
    if not np.all(np.isin(signs, [-1.0, 1.0])):
        raise ValueError("base_orientation_axis_signs values must each be -1.0 or 1.0.")
    return signs


def _coerce_orientation_remap_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("base_orientation_remap_matrix must be finite with shape [3, 3].")
    return matrix


def _coerce_quaternion(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be a finite numeric sequence with shape [4].")
    return normalize_quaternion(array)


def _coerce_float(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _coerce_positive_float(value: object, field_name: str) -> float:
    result = _coerce_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _coerce_nonnegative_float(value: object, field_name: str) -> float:
    result = _coerce_float(value, field_name)
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return result


def _coerce_unit_interval_alpha(value: object, field_name: str) -> float:
    result = _coerce_float(value, field_name)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{field_name} must be in the range (0.0, 1.0].")
    return result


def _coerce_depth_sign(value: object) -> float:
    result = _coerce_float(value, "depth_sign")
    if result == 0.0:
        raise ValueError("depth_sign must be non-zero.")
    return 1.0 if result > 0.0 else -1.0


def _coerce_orientation_mode(value: str) -> OrientationMode:
    if value not in ("relative_palm",):
        raise ValueError("orientation_mode must be 'relative_palm'.")
    return value  # type: ignore[return-value]


def _coerce_orientation_dofs(value: object) -> tuple[OrientationDof, ...]:
    if isinstance(value, str):
        raw_dofs = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        try:
            raw_dofs = tuple(str(part).strip() for part in value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                "orientation_dofs must be a comma-separated string or sequence."
            ) from exc
    if not raw_dofs:
        raise ValueError("orientation_dofs must include at least one of roll, pitch, yaw.")

    allowed = ("roll", "pitch", "yaw")
    normalized: list[OrientationDof] = []
    for dof in raw_dofs:
        if dof not in allowed:
            raise ValueError("orientation_dofs values must be roll, pitch, or yaw.")
        if dof not in normalized:
            normalized.append(dof)  # type: ignore[arg-type]
    return tuple(normalized)


def _coerce_rotation_mode(value: str) -> RotationMode:
    if value == "palm":
        value = "palm_absolute"
    if value == "absolute":
        value = "palm_absolute"
    if value not in ("palm_delta", "palm_absolute", "fixed"):
        raise ValueError(
            "rotation_mode must be 'palm_delta', 'palm_absolute', 'palm', "
            "'absolute', or 'fixed'."
        )
    return value  # type: ignore[return-value]


def _coerce_base_control_mode(value: str) -> BaseControlMode:
    if value not in ("image_2d", "pose_3d"):
        raise ValueError("base_control_mode must be 'image_2d' or 'pose_3d'.")
    return value  # type: ignore[return-value]


def _coerce_image_y_axis(value: str) -> ImageYAxisMode:
    if value not in ("height", "approach"):
        raise ValueError("base_image_y_axis must be 'height' or 'approach'.")
    return value  # type: ignore[return-value]


def _coerce_depth_source(value: str) -> HandScaleSource:
    if value not in ("palm_width", "robust_palm_scale"):
        raise ValueError("depth_source must be 'palm_width' or 'robust_palm_scale'.")
    return value  # type: ignore[return-value]


def _coerce_depth_axis(value: str) -> DepthAxis:
    aliases = {"approach": "x", "lateral": "y", "height": "z"}
    axis = aliases.get(value, value)
    if axis not in ("x", "y", "z"):
        raise ValueError("depth_axis must be 'x', 'y', 'z', 'approach', 'lateral', or 'height'.")
    return axis  # type: ignore[return-value]


def _coerce_position_source(value: str) -> PositionSource:
    if value not in ("wrist", "palm_center"):
        raise ValueError("position_source must be 'wrist' or 'palm_center'.")
    return value  # type: ignore[return-value]


def _coerce_position_mode(value: str) -> PositionMode:
    if value not in ("absolute", "relative"):
        raise ValueError("position_mode must be 'absolute' or 'relative'.")
    return value  # type: ignore[return-value]
