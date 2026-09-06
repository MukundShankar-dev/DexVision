"""Constrained image-space rate control for the Level 4 workcell pilot."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexvision.features.hand_base import (
    HandBaseTarget,
    ImagePalmCenterTarget,
    no_image_palm_center_target,
    normalize_quaternion,
)
from dexvision.sim.hand_base_control import (
    HandBaseControlConfig,
    HandBaseControlStatus,
)
from dexvision.sim.mujoco_env import MujocoEnv


@dataclass(frozen=True, eq=False)
class WorkcellRateControlConfig:
    """Tuning and virtual-fixture parameters for one reach trial."""

    goal_position: np.ndarray
    control_rate_hz: float = 30.0
    image_deadband: float = 0.035
    image_full_scale: float = 0.22
    depth_deadband: float = 0.05
    depth_full_scale: float = 0.35
    response_exponent: float = 2.0
    max_velocity_m_s: np.ndarray = field(
        default_factory=lambda: np.asarray([0.12, 0.15, 0.10], dtype=np.float64)
    )
    transit_height_m: float = 0.19
    descent_radius_m: float = 0.035

    def __post_init__(self) -> None:
        goal = np.asarray(self.goal_position, dtype=np.float64)
        velocity = np.asarray(self.max_velocity_m_s, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal_position must be a finite 3-vector.")
        if velocity.shape != (3,) or np.any(velocity <= 0.0):
            raise ValueError("max_velocity_m_s must be a positive 3-vector.")
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive.")
        if not 0.0 <= self.image_deadband < self.image_full_scale:
            raise ValueError("image_deadband must be below image_full_scale.")
        if not 0.0 <= self.depth_deadband < self.depth_full_scale:
            raise ValueError("depth_deadband must be below depth_full_scale.")
        if self.response_exponent < 1.0:
            raise ValueError("response_exponent must be at least 1.0.")
        if self.descent_radius_m <= 0.0:
            raise ValueError("descent_radius_m must be positive.")
        if self.transit_height_m < goal[2]:
            raise ValueError("transit_height_m must not be below the goal.")
        object.__setattr__(self, "goal_position", goal)
        object.__setattr__(self, "max_velocity_m_s", velocity)


class WorkcellRateController:
    """Treat hand displacement as a velocity joystick with a safe descent corridor."""

    def __init__(
        self,
        env: MujocoEnv,
        base_config: HandBaseControlConfig,
        rate_config: WorkcellRateControlConfig,
    ) -> None:
        self.env = env
        self.config = base_config
        self.rate_config = rate_config
        _, orientation = env.get_mocap_pose(base_config.mocap_body_name)
        self._orientation = normalize_quaternion(orientation)
        self._neutral_palm_center: np.ndarray | None = None
        self._neutral_hand_scale: float | None = None
        neutral, _ = base_config.workspace_limits.clamp(
            base_config.neutral_base_position
        )
        self._last_applied = HandBaseTarget(
            position=neutral,
            orientation_quat=self._orientation,
            confidence=0.0,
            valid=True,
        )
        self._requested_height_m = float(neutral[2])
        self.reset_to_neutral()

    def calibrate_image_2d(
        self,
        target: ImagePalmCenterTarget | None,
        *,
        orientation_target: HandBaseTarget | None = None,
    ) -> bool:
        """Capture the comfortable centered hand pose and reset robot position."""

        del orientation_target
        current = target or no_image_palm_center_target()
        if not current.valid or current.confidence < self.config.min_confidence:
            return False
        if not np.isfinite(current.hand_scale) or current.hand_scale <= 1e-9:
            return False
        self._neutral_palm_center = current.palm_center.copy()
        self._neutral_hand_scale = float(current.hand_scale)
        self.reset_to_neutral(clear_image_calibration=False)
        return True

    def reset_to_neutral(
        self, *, clear_image_calibration: bool = False
    ) -> HandBaseControlStatus:
        """Return to the workcell neutral and optionally require recalibration."""

        if clear_image_calibration:
            self._neutral_palm_center = None
            self._neutral_hand_scale = None
        position, clamped = self.config.workspace_limits.clamp(
            self.config.neutral_base_position
        )
        self._last_applied = HandBaseTarget(
            position=position,
            orientation_quat=self._orientation,
            confidence=0.0,
            valid=True,
        )
        self._requested_height_m = float(position[2])
        if self.config.enabled:
            self.env.set_mocap_pose(
                self.config.mocap_body_name,
                position=position,
                orientation_quat=self._orientation,
            )
        return self._status(
            tracking_valid=False,
            clamped=clamped,
            palm_center=None,
            palm_delta=None,
            depth_delta=None,
        )

    def apply_image_2d(
        self,
        target: ImagePalmCenterTarget | None,
        *,
        orientation_target: HandBaseTarget | None = None,
    ) -> HandBaseControlStatus:
        """Integrate one bounded velocity command and apply virtual fixtures."""

        del orientation_target
        current = target or no_image_palm_center_target()
        tracking_valid = bool(
            current.valid and current.confidence >= self.config.min_confidence
        )
        calibrated = (
            self._neutral_palm_center is not None
            and self._neutral_hand_scale is not None
        )
        if not tracking_valid or not calibrated:
            self.env.set_mocap_pose(
                self.config.mocap_body_name,
                position=self._last_applied.position,
                orientation_quat=self._orientation,
            )
            return self._status(
                tracking_valid=tracking_valid,
                clamped=False,
                palm_center=current.palm_center.copy() if tracking_valid else None,
                palm_delta=np.zeros(2, dtype=np.float64) if tracking_valid else None,
                depth_delta=None,
            )

        assert self._neutral_palm_center is not None
        assert self._neutral_hand_scale is not None
        palm_delta = current.palm_center - self._neutral_palm_center
        raw_depth_delta = (float(current.hand_scale) / self._neutral_hand_scale) - 1.0
        command = np.asarray(
            [
                _nonlinear_rate(
                    raw_depth_delta,
                    deadband=self.rate_config.depth_deadband,
                    full_scale=self.rate_config.depth_full_scale,
                    exponent=self.rate_config.response_exponent,
                ),
                _nonlinear_rate(
                    float(palm_delta[0]),
                    deadband=self.rate_config.image_deadband,
                    full_scale=self.rate_config.image_full_scale,
                    exponent=self.rate_config.response_exponent,
                ),
                -_nonlinear_rate(
                    float(palm_delta[1]),
                    deadband=self.rate_config.image_deadband,
                    full_scale=self.rate_config.image_full_scale,
                    exponent=self.rate_config.response_exponent,
                ),
            ],
            dtype=np.float64,
        )
        velocity = command * self.rate_config.max_velocity_m_s
        candidate = self._last_applied.position + (
            velocity / self.rate_config.control_rate_hz
        )
        candidate, fixture_limited = _apply_reach_virtual_fixture(
            previous=self._last_applied.position,
            candidate=candidate,
            config=self.rate_config,
        )
        candidate, workspace_limited = self.config.workspace_limits.clamp(candidate)
        self._last_applied = HandBaseTarget(
            position=candidate,
            orientation_quat=self._orientation,
            confidence=current.confidence,
            valid=True,
        )
        self.env.set_mocap_pose(
            self.config.mocap_body_name,
            position=candidate,
            orientation_quat=self._orientation,
        )
        return self._status(
            tracking_valid=True,
            clamped=fixture_limited or workspace_limited,
            palm_center=current.palm_center.copy(),
            palm_delta=palm_delta,
            depth_delta=raw_depth_delta,
        )

    def _status(
        self,
        *,
        tracking_valid: bool,
        clamped: bool,
        palm_center: np.ndarray | None,
        palm_delta: np.ndarray | None,
        depth_delta: float | None,
    ) -> HandBaseControlStatus:
        calibrated = (
            self._neutral_palm_center is not None
            and self._neutral_hand_scale is not None
        )
        return HandBaseControlStatus(
            enabled=self.config.enabled,
            applied_target=self._last_applied,
            tracking_valid=tracking_valid,
            neutral_captured=calibrated,
            clamped=clamped,
            rate_limited=False,
            control_mode="image_2d_rate",
            orientation_enabled=False,
            palm_center=palm_center,
            palm_delta=palm_delta,
            depth_enabled=True,
            hand_scale=None,
            neutral_hand_scale=self._neutral_hand_scale,
            depth_delta=depth_delta,
            depth_target=float(self._last_applied.position[0]),
            depth_axis="x",
            depth_clamped=clamped,
            orientation_calibrated=False,
            orientation_delta_rpy_degrees=None,
            orientation_clamped=False,
        )


def _nonlinear_rate(
    value: float, *, deadband: float, full_scale: float, exponent: float
) -> float:
    """Map centered input to [-1, 1] with zero drift and fine control near center."""

    magnitude = abs(float(value))
    if magnitude <= deadband:
        return 0.0
    normalized = min(1.0, (magnitude - deadband) / (full_scale - deadband))
    return float(np.sign(value) * (normalized**exponent))


def _apply_reach_virtual_fixture(
    *,
    previous: np.ndarray,
    candidate: np.ndarray,
    config: WorkcellRateControlConfig,
) -> tuple[np.ndarray, bool]:
    """Require high travel and keep low motion inside the target descent column."""

    constrained = np.asarray(candidate, dtype=np.float64).copy()
    goal = config.goal_position
    horizontal_distance = float(np.linalg.norm(constrained[:2] - goal[:2]))
    changed = False

    if horizontal_distance > config.descent_radius_m:
        if constrained[2] < config.transit_height_m:
            constrained[2] = config.transit_height_m
            changed = True
    elif constrained[2] < goal[2]:
        constrained[2] = goal[2]
        changed = True

    if previous[2] < config.transit_height_m and horizontal_distance > 0.0:
        offset = constrained[:2] - goal[:2]
        if horizontal_distance > config.descent_radius_m:
            constrained[:2] = goal[:2] + (
                offset * (config.descent_radius_m / horizontal_distance)
            )
            changed = True

    return constrained, changed
