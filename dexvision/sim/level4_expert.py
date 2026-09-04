"""Deterministic scripted-expert boundary for Level 4 workcell tasks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dexvision.sim.world_state import WorldState


BASE_ACTION_NAMES = (
    "base_position_target/x",
    "base_position_target/y",
    "base_position_target/z",
    "base_orientation_target/qw",
    "base_orientation_target/qx",
    "base_orientation_target/qy",
    "base_orientation_target/qz",
)


class Level4ExpertError(ValueError):
    """Raised when a scripted expert cannot safely serve a requested task."""


@dataclass(frozen=True)
class RequestedAction:
    """One complete, named Level 4 requested-action row."""

    names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.values):
            raise Level4ExpertError(
                "requested action names and values must have the same non-zero length."
            )
        if len(set(self.names)) != len(self.names):
            raise Level4ExpertError("requested action names must be unique.")
        if not np.all(np.isfinite(np.asarray(self.values, dtype=np.float64))):
            raise Level4ExpertError("requested action values must be finite.")

    def as_array(self) -> np.ndarray:
        """Return a fresh dense action row in named-layout order."""

        return np.asarray(self.values, dtype=np.float64)

    @property
    def base_position(self) -> np.ndarray:
        return np.asarray(self.values[:3], dtype=np.float64)

    @property
    def base_orientation_wxyz(self) -> np.ndarray:
        return np.asarray(self.values[3:7], dtype=np.float64)

    @property
    def finger_targets(self) -> dict[str, float]:
        return {
            name.removeprefix("finger_actuator_targets/"): float(value)
            for name, value in zip(self.names[7:], self.values[7:], strict=True)
        }


class ExpertController(Protocol):
    """Common deterministic Level 4 scripted-expert interface."""

    def reset(self, task: object, world_state: WorldState) -> None:
        """Bind a typed task and initial simulator state."""

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]:
        """Return nominal request, causal phase, terminal flag, and reason."""


@dataclass(frozen=True)
class SafeWaypointReachConfig:
    """Frozen bounded planner and copied-state validation parameters."""

    transit_height_m: float = 0.22
    corridor_entry_height_m: float = 0.18
    max_position_step_m: float = 0.01
    waypoint_tolerance_m: float = 1e-6
    required_goal_dwell_steps: int = 5
    sim_steps_per_action: int = 17
    maximum_non_target_disturbance_m: float = 0.005
    joint_limit_tolerance_rad: float = 1e-6

    def __post_init__(self) -> None:
        positive = {
            "transit_height_m": self.transit_height_m,
            "corridor_entry_height_m": self.corridor_entry_height_m,
            "max_position_step_m": self.max_position_step_m,
            "waypoint_tolerance_m": self.waypoint_tolerance_m,
            "required_goal_dwell_steps": self.required_goal_dwell_steps,
            "sim_steps_per_action": self.sim_steps_per_action,
            "maximum_non_target_disturbance_m": (
                self.maximum_non_target_disturbance_m
            ),
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise Level4ExpertError("reach expert configuration values must be positive.")
        if self.corridor_entry_height_m > self.transit_height_m:
            raise Level4ExpertError(
                "corridor entry height must not exceed transit height."
            )
        if self.joint_limit_tolerance_rad < 0.0:
            raise Level4ExpertError("joint limit tolerance must be non-negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SafeWaypointReachConfig":
        """Build the validated expert config from the frozen dataset config."""

        fields = {
            "transit_height_m": float,
            "corridor_entry_height_m": float,
            "max_position_step_m": float,
            "waypoint_tolerance_m": float,
            "required_goal_dwell_steps": int,
            "sim_steps_per_action": int,
            "maximum_non_target_disturbance_m": float,
            "joint_limit_tolerance_rad": float,
        }
        missing = [name for name in fields if name not in values]
        if missing:
            raise Level4ExpertError(
                "scripted reach config is missing: " + ", ".join(missing)
            )
        return cls(**{name: cast(values[name]) for name, cast in fields.items()})


@dataclass(frozen=True)
class DeterministicButtonPressConfig:
    """Fixed-posture button approach, press, and release parameters."""

    transit_height_m: float = 0.22
    precontact_offset_m: tuple[float, float, float] = (-0.15, -0.011, -0.035)
    press_offset_m: tuple[float, float, float] = (-0.095, -0.011, -0.035)
    transit_step_m: float = 0.01
    press_step_m: float = 0.002
    retract_step_m: float = 0.005
    waypoint_tolerance_m: float = 1e-6
    required_press_dwell_steps: int = 3
    release_depth_m: float = 0.002
    release_hold_steps: int = 5
    sim_steps_per_action: int = 17
    maximum_non_target_disturbance_m: float = 0.005
    joint_limit_tolerance_rad: float = 1e-6

    def __post_init__(self) -> None:
        scalar_positive = (
            self.transit_height_m,
            self.transit_step_m,
            self.press_step_m,
            self.retract_step_m,
            self.waypoint_tolerance_m,
            self.required_press_dwell_steps,
            self.release_hold_steps,
            self.sim_steps_per_action,
            self.maximum_non_target_disturbance_m,
        )
        if any(float(value) <= 0.0 for value in scalar_positive):
            raise Level4ExpertError("button expert configuration values must be positive.")
        if self.release_depth_m < 0.0 or self.joint_limit_tolerance_rad < 0.0:
            raise Level4ExpertError(
                "button release depth and joint-limit tolerance must be non-negative."
            )
        for name, value in (
            ("precontact_offset_m", self.precontact_offset_m),
            ("press_offset_m", self.press_offset_m),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise Level4ExpertError(f"{name} must be a finite 3-vector.")
        if self.press_offset_m[0] <= self.precontact_offset_m[0]:
            raise Level4ExpertError("button press must advance along the positive x normal.")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object]
    ) -> "DeterministicButtonPressConfig":
        """Build the controller config from the frozen dataset plan."""

        fields = (
            "transit_height_m",
            "precontact_offset_m",
            "press_offset_m",
            "transit_step_m",
            "press_step_m",
            "retract_step_m",
            "waypoint_tolerance_m",
            "required_press_dwell_steps",
            "release_depth_m",
            "release_hold_steps",
            "sim_steps_per_action",
            "maximum_non_target_disturbance_m",
            "joint_limit_tolerance_rad",
        )
        missing = [name for name in fields if name not in values]
        if missing:
            raise Level4ExpertError(
                "scripted button config is missing: " + ", ".join(missing)
            )
        return cls(
            transit_height_m=float(values["transit_height_m"]),
            precontact_offset_m=tuple(
                float(value) for value in values["precontact_offset_m"]  # type: ignore[union-attr]
            ),
            press_offset_m=tuple(
                float(value) for value in values["press_offset_m"]  # type: ignore[union-attr]
            ),
            transit_step_m=float(values["transit_step_m"]),
            press_step_m=float(values["press_step_m"]),
            retract_step_m=float(values["retract_step_m"]),
            waypoint_tolerance_m=float(values["waypoint_tolerance_m"]),
            required_press_dwell_steps=int(values["required_press_dwell_steps"]),
            release_depth_m=float(values["release_depth_m"]),
            release_hold_steps=int(values["release_hold_steps"]),
            sim_steps_per_action=int(values["sim_steps_per_action"]),
            maximum_non_target_disturbance_m=float(
                values["maximum_non_target_disturbance_m"]
            ),
            joint_limit_tolerance_rad=float(values["joint_limit_tolerance_rad"]),
        )


@dataclass(frozen=True)
class DeterministicPushConfig:
    """Configuration for a fixed-posture task-axis fingertip push."""

    transit_height_m: float
    approach_gap_m: float
    fingertip_lateral_offset_m: float
    family_parameters: Mapping[str, Mapping[str, object]]
    transit_step_m: float
    descent_step_m: float
    push_step_m: float
    target_stop_distance_m: float
    retract_distance_m: float
    retract_step_m: float
    orientation_step_rad: float
    waypoint_tolerance_m: float
    required_goal_dwell_steps: int
    release_hold_steps: int
    required_terminal_dwell_steps: int
    maximum_object_tilt_rad: float
    maximum_push_actions: int
    sim_steps_per_action: int
    maximum_non_target_disturbance_m: float
    joint_limit_tolerance_rad: float

    def __post_init__(self) -> None:
        positive = (
            self.transit_height_m,
            self.approach_gap_m,
            self.fingertip_lateral_offset_m,
            self.transit_step_m,
            self.descent_step_m,
            self.push_step_m,
            self.target_stop_distance_m,
            self.retract_distance_m,
            self.retract_step_m,
            self.orientation_step_rad,
            self.waypoint_tolerance_m,
            self.required_goal_dwell_steps,
            self.release_hold_steps,
            self.required_terminal_dwell_steps,
            self.maximum_object_tilt_rad,
            self.maximum_push_actions,
            self.sim_steps_per_action,
            self.maximum_non_target_disturbance_m,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise Level4ExpertError("push expert configuration values must be positive.")
        if self.joint_limit_tolerance_rad < 0.0:
            raise Level4ExpertError("joint-limit tolerance must be non-negative.")
        if set(self.family_parameters) != {"cuboid", "flat_puck"}:
            raise Level4ExpertError(
                "push family parameters must define cuboid and flat_puck."
            )
        for family, raw in self.family_parameters.items():
            required = {
                "wrist_pitch_deg",
                "control_height_m",
                "fingertip_forward_offset_m",
                "control_side",
                "index_curl",
            }
            if set(raw) != required or raw["control_side"] not in {"ahead", "behind"}:
                raise Level4ExpertError(f"invalid push parameters for {family}.")
            if any(
                not math.isfinite(float(raw[name])) or float(raw[name]) <= 0.0
                for name in required - {"control_side", "index_curl"}
            ):
                raise Level4ExpertError(f"invalid numeric push parameters for {family}.")
            if not 0.0 <= float(raw["index_curl"]) <= 1.0:
                raise Level4ExpertError(
                    f"push index curl for {family} must be in [0, 1]."
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "DeterministicPushConfig":
        """Build the push controller from the frozen Level 4 plan."""

        family_parameters = values.get("family_parameters")
        if not isinstance(family_parameters, Mapping):
            raise Level4ExpertError("scripted push config requires family_parameters.")
        required = (
            "transit_height_m",
            "approach_gap_m",
            "fingertip_lateral_offset_m",
            "transit_step_m",
            "descent_step_m",
            "push_step_m",
            "target_stop_distance_m",
            "retract_distance_m",
            "retract_step_m",
            "orientation_step_rad",
            "waypoint_tolerance_m",
            "required_goal_dwell_steps",
            "release_hold_steps",
            "required_terminal_dwell_steps",
            "maximum_object_tilt_rad",
            "maximum_push_actions",
            "sim_steps_per_action",
            "maximum_non_target_disturbance_m",
            "joint_limit_tolerance_rad",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise Level4ExpertError(
                "scripted push config is missing: " + ", ".join(missing)
            )
        families = {
            str(name): dict(raw)
            for name, raw in family_parameters.items()
            if isinstance(raw, Mapping)
        }
        return cls(
            transit_height_m=float(values["transit_height_m"]),
            approach_gap_m=float(values["approach_gap_m"]),
            fingertip_lateral_offset_m=float(
                values["fingertip_lateral_offset_m"]
            ),
            family_parameters=families,
            transit_step_m=float(values["transit_step_m"]),
            descent_step_m=float(values["descent_step_m"]),
            push_step_m=float(values["push_step_m"]),
            target_stop_distance_m=float(values["target_stop_distance_m"]),
            retract_distance_m=float(values["retract_distance_m"]),
            retract_step_m=float(values["retract_step_m"]),
            orientation_step_rad=float(values["orientation_step_rad"]),
            waypoint_tolerance_m=float(values["waypoint_tolerance_m"]),
            required_goal_dwell_steps=int(values["required_goal_dwell_steps"]),
            release_hold_steps=int(values["release_hold_steps"]),
            required_terminal_dwell_steps=int(values["required_terminal_dwell_steps"]),
            maximum_object_tilt_rad=float(values["maximum_object_tilt_rad"]),
            maximum_push_actions=int(values["maximum_push_actions"]),
            sim_steps_per_action=int(values["sim_steps_per_action"]),
            maximum_non_target_disturbance_m=float(
                values["maximum_non_target_disturbance_m"]
            ),
            joint_limit_tolerance_rad=float(values["joint_limit_tolerance_rad"]),
        )


@dataclass(frozen=True)
class GraspFamilyTemplate:
    """One object-relative grasp pose and scalar closure for an object family."""

    object_relative_position_m: tuple[float, float, float]
    wrist_orientation_wxyz: tuple[float, float, float, float]
    negative_object_yaw_to_wrist_yaw_gain: float
    orientation_symmetry: str
    orientation_feedback_enabled: bool
    transport_orientation_feedback_enabled: bool
    grasp_synergy: float
    lift_distance_m: float

    def __post_init__(self) -> None:
        position = np.asarray(self.object_relative_position_m, dtype=np.float64)
        orientation = np.asarray(self.wrist_orientation_wxyz, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise Level4ExpertError(
                "grasp object_relative_position_m must be a finite 3-vector."
            )
        if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
            raise Level4ExpertError(
                "grasp wrist_orientation_wxyz must be a finite quaternion."
            )
        if not math.isclose(float(np.linalg.norm(orientation)), 1.0, abs_tol=1e-6):
            raise Level4ExpertError("grasp wrist orientation must be normalized.")
        if not math.isfinite(self.negative_object_yaw_to_wrist_yaw_gain):
            raise Level4ExpertError("grasp object-yaw wrist-yaw gain must be finite.")
        if self.orientation_symmetry not in {"none", "axial_z"}:
            raise Level4ExpertError(
                "grasp orientation_symmetry must be none or axial_z."
            )
        if not 0.0 < self.grasp_synergy <= 1.0:
            raise Level4ExpertError("grasp synergy must be in (0, 1].")
        if not math.isfinite(self.lift_distance_m) or self.lift_distance_m <= 0.0:
            raise Level4ExpertError("grasp lift distance must be positive.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "GraspFamilyTemplate":
        required = {
            "object_relative_position_m",
            "negative_object_yaw_to_wrist_yaw_gain",
            "wrist_orientation_wxyz",
            "orientation_symmetry",
            "orientation_feedback_enabled",
            "transport_orientation_feedback_enabled",
            "grasp_synergy",
            "lift_distance_m",
        }
        if set(values) != required:
            raise Level4ExpertError(
                "grasp family template must define exactly: "
                + ", ".join(sorted(required))
            )
        return cls(
            object_relative_position_m=tuple(
                float(value)
                for value in values["object_relative_position_m"]  # type: ignore[union-attr]
            ),
            wrist_orientation_wxyz=tuple(
                float(value)
                for value in values["wrist_orientation_wxyz"]  # type: ignore[union-attr]
            ),
            negative_object_yaw_to_wrist_yaw_gain=float(
                values["negative_object_yaw_to_wrist_yaw_gain"]
            ),
            orientation_symmetry=str(values["orientation_symmetry"]),
            orientation_feedback_enabled=bool(values["orientation_feedback_enabled"]),
            transport_orientation_feedback_enabled=bool(
                values["transport_orientation_feedback_enabled"]
            ),
            grasp_synergy=float(values["grasp_synergy"]),
            lift_distance_m=float(values["lift_distance_m"]),
        )


@dataclass(frozen=True)
class DeterministicGraspLiftConfig:
    """Configuration-owned family templates and lift/hold qualification."""

    hand_poses: Mapping[str, str]
    family_templates: Mapping[str, GraspFamilyTemplate]
    transit_height_m: float
    transit_step_m: float
    descent_step_m: float
    lift_step_m: float
    synergy_step: float
    orientation_step_rad: float
    orientation_preservation_policy: str
    orientation_correction_step_rad: float
    maximum_object_orientation_deviation_rad: float
    maximum_terminal_orientation_error_rad: float
    waypoint_tolerance_m: float
    required_hold_steps: int
    maximum_closed_acquisition_actions: int
    maximum_retention_gap_steps: int
    maximum_hold_speed_m_s: float
    sim_steps_per_action: int
    maximum_non_target_disturbance_m: float
    joint_limit_tolerance_rad: float

    def __post_init__(self) -> None:
        if dict(self.hand_poses) != {
            "open": "configured_retargeter_open",
            "closed": "configured_retargeter_full_flexion",
        }:
            raise Level4ExpertError(
                "grasp hand_poses must name the configured open and full-flexion poses."
            )
        if set(self.family_templates) != {"cuboid", "cylinder", "flat_puck"}:
            raise Level4ExpertError(
                "grasp family templates must define cuboid, cylinder, and flat_puck."
            )
        positive = (
            self.transit_height_m,
            self.transit_step_m,
            self.descent_step_m,
            self.lift_step_m,
            self.synergy_step,
            self.orientation_step_rad,
            self.orientation_correction_step_rad,
            self.maximum_object_orientation_deviation_rad,
            self.maximum_terminal_orientation_error_rad,
            self.waypoint_tolerance_m,
            self.required_hold_steps,
            self.maximum_closed_acquisition_actions,
            self.maximum_retention_gap_steps,
            self.maximum_hold_speed_m_s,
            self.sim_steps_per_action,
            self.maximum_non_target_disturbance_m,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise Level4ExpertError("grasp expert configuration values must be positive.")
        if self.synergy_step > 1.0:
            raise Level4ExpertError("grasp synergy_step must not exceed one.")
        if (
            self.orientation_preservation_policy
            != "shape_aware_hammer_grip_with_world_orientation_hold"
        ):
            raise Level4ExpertError(
                "grasp orientation preservation must use the shape-aware hammer grip "
                "with a world-orientation hold."
            )
        if (
            self.maximum_terminal_orientation_error_rad
            > self.maximum_object_orientation_deviation_rad
        ):
            raise Level4ExpertError(
                "terminal orientation error must not exceed maximum deviation."
            )
        if self.joint_limit_tolerance_rad < 0.0:
            raise Level4ExpertError("joint-limit tolerance must be non-negative.")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object]
    ) -> "DeterministicGraspLiftConfig":
        templates = values.get("family_templates")
        hand_poses = values.get("hand_poses")
        if not isinstance(templates, Mapping) or not isinstance(hand_poses, Mapping):
            raise Level4ExpertError(
                "scripted grasp config requires hand_poses and family_templates."
            )
        required = (
            "transit_height_m",
            "transit_step_m",
            "descent_step_m",
            "lift_step_m",
            "synergy_step",
            "orientation_step_rad",
            "orientation_preservation_policy",
            "orientation_correction_step_rad",
            "maximum_object_orientation_deviation_rad",
            "maximum_terminal_orientation_error_rad",
            "waypoint_tolerance_m",
            "required_hold_steps",
            "maximum_closed_acquisition_actions",
            "maximum_retention_gap_steps",
            "maximum_hold_speed_m_s",
            "sim_steps_per_action",
            "maximum_non_target_disturbance_m",
            "joint_limit_tolerance_rad",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise Level4ExpertError(
                "scripted grasp config is missing: " + ", ".join(missing)
            )
        family_templates = {
            str(name): GraspFamilyTemplate.from_mapping(raw)
            for name, raw in templates.items()
            if isinstance(raw, Mapping)
        }
        return cls(
            hand_poses={str(name): str(value) for name, value in hand_poses.items()},
            family_templates=family_templates,
            transit_height_m=float(values["transit_height_m"]),
            transit_step_m=float(values["transit_step_m"]),
            descent_step_m=float(values["descent_step_m"]),
            lift_step_m=float(values["lift_step_m"]),
            synergy_step=float(values["synergy_step"]),
            orientation_step_rad=float(values["orientation_step_rad"]),
            orientation_preservation_policy=str(
                values["orientation_preservation_policy"]
            ),
            orientation_correction_step_rad=float(
                values["orientation_correction_step_rad"]
            ),
            maximum_object_orientation_deviation_rad=float(
                values["maximum_object_orientation_deviation_rad"]
            ),
            maximum_terminal_orientation_error_rad=float(
                values["maximum_terminal_orientation_error_rad"]
            ),
            waypoint_tolerance_m=float(values["waypoint_tolerance_m"]),
            required_hold_steps=int(values["required_hold_steps"]),
            maximum_closed_acquisition_actions=int(
                values["maximum_closed_acquisition_actions"]
            ),
            maximum_retention_gap_steps=int(values["maximum_retention_gap_steps"]),
            maximum_hold_speed_m_s=float(values["maximum_hold_speed_m_s"]),
            sim_steps_per_action=int(values["sim_steps_per_action"]),
            maximum_non_target_disturbance_m=float(
                values["maximum_non_target_disturbance_m"]
            ),
            joint_limit_tolerance_rad=float(values["joint_limit_tolerance_rad"]),
        )


@dataclass(frozen=True)
class DeterministicPlaceConfig:
    """Configuration for held transport, release, settling, and retraction."""

    orientation_policy: str
    transport_object_height_m: float
    placement_clearance_m: float
    maximum_placement_center_x_m: float
    transport_step_m: float
    descent_step_m: float
    release_synergy_step: float
    release_clearance_height_m: float
    family_release_backoff_x_m: Mapping[str, float]
    family_target_offset_xy_m: Mapping[str, tuple[float, float]]
    retract_height_m: float
    retract_step_m: float
    waypoint_tolerance_m: float
    release_position_tolerance_m: float
    required_terminal_dwell_steps: int
    maximum_settle_actions: int
    maximum_total_actions: int
    sim_steps_per_action: int
    maximum_non_target_disturbance_m: float
    joint_limit_tolerance_rad: float

    def __post_init__(self) -> None:
        if self.orientation_policy != "keep_qualified_grasp_orientation":
            raise Level4ExpertError(
                "place orientation_policy must keep the qualified grasp orientation."
            )
        positive = (
            self.transport_object_height_m,
            self.placement_clearance_m,
            self.transport_step_m,
            self.descent_step_m,
            self.release_synergy_step,
            self.release_clearance_height_m,
            self.retract_height_m,
            self.retract_step_m,
            self.waypoint_tolerance_m,
            self.release_position_tolerance_m,
            self.required_terminal_dwell_steps,
            self.maximum_settle_actions,
            self.maximum_total_actions,
            self.sim_steps_per_action,
            self.maximum_non_target_disturbance_m,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in positive
        ):
            raise Level4ExpertError("place expert configuration values must be positive.")
        if self.release_synergy_step > 1.0:
            raise Level4ExpertError("place release_synergy_step must not exceed one.")
        if set(self.family_target_offset_xy_m) != {"cuboid", "cylinder", "flat_puck"}:
            raise Level4ExpertError(
                "place family target offsets must cover cuboid, cylinder, and flat_puck."
            )
        if any(
            len(offset) != 2
            or not all(math.isfinite(float(value)) for value in offset)
            for offset in self.family_target_offset_xy_m.values()
        ):
            raise Level4ExpertError("place family target offsets must be finite xy pairs.")
        if set(self.family_release_backoff_x_m) != {
            "cuboid",
            "cylinder",
            "flat_puck",
        } or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.family_release_backoff_x_m.values()
        ):
            raise Level4ExpertError(
                "place family release backoffs must be finite non-negative values."
            )
        if self.joint_limit_tolerance_rad < 0.0:
            raise Level4ExpertError("joint-limit tolerance must be non-negative.")
        if not math.isfinite(self.maximum_placement_center_x_m):
            raise Level4ExpertError("maximum placement center x must be finite.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "DeterministicPlaceConfig":
        required = (
            "orientation_policy",
            "transport_object_height_m",
            "placement_clearance_m",
            "maximum_placement_center_x_m",
            "transport_step_m",
            "descent_step_m",
            "release_synergy_step",
            "release_clearance_height_m",
            "family_release_backoff_x_m",
            "family_target_offset_xy_m",
            "retract_height_m",
            "retract_step_m",
            "waypoint_tolerance_m",
            "release_position_tolerance_m",
            "required_terminal_dwell_steps",
            "maximum_settle_actions",
            "maximum_total_actions",
            "sim_steps_per_action",
            "maximum_non_target_disturbance_m",
            "joint_limit_tolerance_rad",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise Level4ExpertError(
                "scripted place config is missing: " + ", ".join(missing)
            )
        return cls(
            orientation_policy=str(values["orientation_policy"]),
            transport_object_height_m=float(values["transport_object_height_m"]),
            placement_clearance_m=float(values["placement_clearance_m"]),
            maximum_placement_center_x_m=float(
                values["maximum_placement_center_x_m"]
            ),
            transport_step_m=float(values["transport_step_m"]),
            descent_step_m=float(values["descent_step_m"]),
            release_synergy_step=float(values["release_synergy_step"]),
            release_clearance_height_m=float(
                values["release_clearance_height_m"]
            ),
            family_release_backoff_x_m={
                str(family): float(backoff)
                for family, backoff in dict(
                    values["family_release_backoff_x_m"]
                ).items()
            },
            family_target_offset_xy_m={
                str(family): tuple(float(component) for component in offset)
                for family, offset in dict(values["family_target_offset_xy_m"]).items()
            },
            retract_height_m=float(values["retract_height_m"]),
            retract_step_m=float(values["retract_step_m"]),
            waypoint_tolerance_m=float(values["waypoint_tolerance_m"]),
            release_position_tolerance_m=float(
                values["release_position_tolerance_m"]
            ),
            required_terminal_dwell_steps=int(
                values["required_terminal_dwell_steps"]
            ),
            maximum_settle_actions=int(values["maximum_settle_actions"]),
            maximum_total_actions=int(values["maximum_total_actions"]),
            sim_steps_per_action=int(values["sim_steps_per_action"]),
            maximum_non_target_disturbance_m=float(
                values["maximum_non_target_disturbance_m"]
            ),
            joint_limit_tolerance_rad=float(values["joint_limit_tolerance_rad"]),
        )


@dataclass(frozen=True)
class WaypointValidationResult:
    """Result of pre-execution validation in a copied MuJoCo state."""

    valid: bool
    reason: str | None
    checked_actions: int
    maximum_non_target_disturbance_m: float


def level4_action_names(finger_target_names: Sequence[str]) -> tuple[str, ...]:
    """Return the frozen complete action names for an actuator ordering."""

    fingers = tuple(str(name) for name in finger_target_names)
    if not fingers or any(not name for name in fingers):
        raise Level4ExpertError("finger target names must be non-empty strings.")
    return BASE_ACTION_NAMES + tuple(
        f"finger_actuator_targets/{name}" for name in fingers
    )


class SafeWaypointReachExpert:
    """Scripted pre-contact reach through rise/transit/corridor/descent."""

    def __init__(
        self,
        *,
        finger_targets: Mapping[str, float],
        config: SafeWaypointReachConfig | None = None,
    ) -> None:
        if not finger_targets:
            raise Level4ExpertError("reach expert requires named finger targets.")
        self._finger_targets = {
            str(name): float(value) for name, value in finger_targets.items()
        }
        if any(
            not name or not math.isfinite(value)
            for name, value in self._finger_targets.items()
        ):
            raise Level4ExpertError("finger targets must be finite and named.")
        self.config = config or SafeWaypointReachConfig()
        self._names = level4_action_names(tuple(self._finger_targets))
        self._task: object | None = None
        self._waypoints: tuple[np.ndarray, ...] = ()
        self._waypoint_index = 0
        self._orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._initial_entities: dict[str, np.ndarray] = {}
        self._goal_position = np.zeros(3, dtype=np.float64)
        self._last_requested_position = np.zeros(3, dtype=np.float64)
        self._goal_dwell = 0
        self._terminal_reason: str | None = None
        self._validation: WaypointValidationResult | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def waypoints(self) -> tuple[np.ndarray, ...]:
        return tuple(waypoint.copy() for waypoint in self._waypoints)

    @property
    def validation(self) -> WaypointValidationResult | None:
        return self._validation

    def reset(self, task: object, world_state: WorldState) -> None:
        """Build and validate the deterministic path without mutating live state."""

        skill_name = getattr(task, "skill_name", None)
        goal = getattr(task, "goal", None)
        workcell = getattr(task, "workcell", None)
        if skill_name != "reach_object" or not isinstance(goal, Mapping):
            raise Level4ExpertError(
                "SafeWaypointReachExpert only supports reach_object in Level 4.3A."
            )
        if workcell is None:
            raise Level4ExpertError("reach task must expose its workcell.")
        pose = np.asarray(goal.get("approach_pose"), dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise Level4ExpertError("reach approach_pose must be a finite 7-vector.")
        expected_names = tuple(
            str(field["name"])
            for field in sorted(
                task.collection_config["action_contract"]["named_layout"],
                key=lambda item: int(item["index"]),
            )
        )
        if self._names != expected_names:
            raise Level4ExpertError(
                "scripted expert action names do not match the frozen Level 4 layout."
            )

        start = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
        self._goal_position = pose[:3].copy()
        self._last_requested_position = start.copy()
        transit_z = max(
            self.config.transit_height_m,
            float(start[2]),
            float(self._goal_position[2]),
        )
        corridor_z = min(
            transit_z,
            max(self.config.corridor_entry_height_m, float(self._goal_position[2])),
        )
        self._waypoints = _deduplicate_waypoints(
            (
                np.asarray([start[0], start[1], transit_z], dtype=np.float64),
                np.asarray(
                    [self._goal_position[0], self._goal_position[1], transit_z],
                    dtype=np.float64,
                ),
                np.asarray(
                    [self._goal_position[0], self._goal_position[1], corridor_z],
                    dtype=np.float64,
                ),
                self._goal_position.copy(),
            )
        )
        self._orientation = _normalize_quaternion(pose[3:])
        self._initial_entities = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in world_state.entities
        }
        self._task = task
        self._waypoint_index = 0
        self._goal_dwell = 0
        self._terminal_reason = None
        self._validation = validate_reach_waypoints_on_copy(
            task=task,
            initial_world_state=world_state,
            waypoints=self._waypoints,
            orientation_wxyz=self._orientation,
            finger_targets=self._finger_targets,
            config=self.config,
        )
        if not self._validation.valid:
            self._terminal_reason = self._validation.reason

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]:
        """Advance the causal waypoint state machine and emit one full action."""

        if self._task is None or self._validation is None:
            raise Level4ExpertError("reset() must be called before step().")
        current = np.asarray(
            world_state.robot.end_effector_position, dtype=np.float64
        )
        if self._terminal_reason is not None:
            return self._action(current), "approach", True, self._terminal_reason

        disturbance = self._maximum_non_target_disturbance(world_state)
        if disturbance > self.config.maximum_non_target_disturbance_m:
            self._terminal_reason = "non_target_disturbance"
            return self._action(current), "approach", True, self._terminal_reason

        while self._waypoint_index < len(self._waypoints):
            waypoint = self._waypoints[self._waypoint_index]
            if (
                float(np.linalg.norm(waypoint - self._last_requested_position))
                > self.config.waypoint_tolerance_m
            ):
                break
            self._waypoint_index += 1

        at_goal = float(np.linalg.norm(self._goal_position - current)) <= 0.025
        self._goal_dwell = self._goal_dwell + 1 if at_goal else 0
        done = self._goal_dwell >= self.config.required_goal_dwell_steps
        if done:
            return self._action(self._goal_position), "approach", True, None
        if self._waypoint_index >= len(self._waypoints):
            target = self._goal_position
        else:
            target = _bounded_step(
                self._last_requested_position,
                self._waypoints[self._waypoint_index],
                self.config.max_position_step_m,
            )
        self._last_requested_position = target.copy()
        return self._action(target), "approach", False, None

    def _action(self, position: np.ndarray) -> RequestedAction:
        values = (
            *np.asarray(position, dtype=np.float64).tolist(),
            *self._orientation.tolist(),
            *(self._finger_targets[name] for name in self._finger_targets),
        )
        return RequestedAction(self._names, tuple(float(value) for value in values))

    def _maximum_non_target_disturbance(self, state: WorldState) -> float:
        goal_entity = str(getattr(self._task, "goal")["entity_id"])
        distances = [
            float(np.linalg.norm(np.asarray(entity.position) - initial))
            for entity in state.entities
            if entity.object_id != goal_entity
            and (initial := self._initial_entities.get(entity.object_id)) is not None
        ]
        return max(distances, default=0.0)


class DeterministicButtonPressExpert:
    """Press one workcell button along its normal, then fully retract."""

    def __init__(
        self,
        *,
        finger_targets: Mapping[str, float],
        config: DeterministicButtonPressConfig | None = None,
    ) -> None:
        if not finger_targets:
            raise Level4ExpertError("button expert requires named finger targets.")
        self._finger_targets = {
            str(name): float(value) for name, value in finger_targets.items()
        }
        if any(
            not name or not math.isfinite(value)
            for name, value in self._finger_targets.items()
        ):
            raise Level4ExpertError("finger targets must be finite and named.")
        self.config = config or DeterministicButtonPressConfig()
        self._names = level4_action_names(tuple(self._finger_targets))
        self._task: object | None = None
        self._waypoints: tuple[np.ndarray, ...] = ()
        self._waypoint_index = 0
        self._orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._precontact = np.zeros(3, dtype=np.float64)
        self._press = np.zeros(3, dtype=np.float64)
        self._last_requested_position = np.zeros(3, dtype=np.float64)
        self._initial_entities: dict[str, np.ndarray] = {}
        self._phase = "approach"
        self._press_dwell = 0
        self._release_dwell = 0
        self._terminal_reason: str | None = None
        self._validation: WaypointValidationResult | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def waypoints(self) -> tuple[np.ndarray, ...]:
        return tuple(waypoint.copy() for waypoint in self._waypoints)

    @property
    def validation(self) -> WaypointValidationResult | None:
        return self._validation

    def reset(self, task: object, world_state: WorldState) -> None:
        """Resolve and validate one button trajectory without changing live mjData."""

        skill_name = getattr(task, "skill_name", None)
        goal = getattr(task, "goal", None)
        workcell = getattr(task, "workcell", None)
        if skill_name != "press_button" or not isinstance(goal, Mapping):
            raise Level4ExpertError(
                "DeterministicButtonPressExpert only supports press_button in Level 4.3B."
            )
        if workcell is None:
            raise Level4ExpertError("button task must expose its workcell.")
        expected_names = tuple(
            str(field["name"])
            for field in sorted(
                task.collection_config["action_contract"]["named_layout"],
                key=lambda item: int(item["index"]),
            )
        )
        if self._names != expected_names:
            raise Level4ExpertError(
                "scripted expert action names do not match the frozen Level 4 layout."
            )

        button_id = str(goal["button_id"])
        button_position = np.asarray(
            world_state.require_entity(button_id).position, dtype=np.float64
        )
        start = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
        self._precontact = button_position + np.asarray(
            self.config.precontact_offset_m, dtype=np.float64
        )
        self._press = button_position + np.asarray(
            self.config.press_offset_m, dtype=np.float64
        )
        transit_z = max(
            self.config.transit_height_m,
            float(start[2]),
            float(self._precontact[2]),
        )
        self._waypoints = _deduplicate_waypoints(
            (
                np.asarray([start[0], start[1], transit_z], dtype=np.float64),
                np.asarray(
                    [self._precontact[0], self._precontact[1], transit_z],
                    dtype=np.float64,
                ),
                self._precontact.copy(),
            )
        )
        self._orientation = np.asarray(
            world_state.robot.base_orientation_wxyz, dtype=np.float64
        )
        self._last_requested_position = start.copy()
        self._initial_entities = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in world_state.entities
        }
        self._task = task
        self._waypoint_index = 0
        self._phase = "approach"
        self._press_dwell = 0
        self._release_dwell = 0
        self._terminal_reason = None
        self._validation = validate_button_trajectory_on_copy(
            task=task,
            initial_world_state=world_state,
            waypoints=self._waypoints,
            press_position=self._press,
            precontact_position=self._precontact,
            orientation_wxyz=self._orientation,
            finger_targets=self._finger_targets,
            config=self.config,
        )
        if not self._validation.valid:
            self._terminal_reason = self._validation.reason

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]:
        """Emit approach, fixture-contact, and retract actions causally."""

        if self._task is None or self._validation is None:
            raise Level4ExpertError("reset() must be called before step().")
        current = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
        if self._terminal_reason is not None:
            return self._action(current), self._phase, True, self._terminal_reason
        if self._maximum_non_target_disturbance(world_state) > (
            self.config.maximum_non_target_disturbance_m
        ):
            self._terminal_reason = "non_target_disturbance"
            return self._action(current), self._phase, True, self._terminal_reason

        button_id = str(getattr(self._task, "goal")["button_id"])
        contact_reason = _unsafe_button_contact_reason(
            world_state, button_id=button_id
        )
        if contact_reason is not None:
            self._terminal_reason = contact_reason
            return self._action(current), self._phase, True, self._terminal_reason
        fixture = world_state.require_fixture(button_id)
        button_contact = any(
            button_id in pair and any(name.startswith("rh_") for name in pair)
            for pair in world_state.contacts
        )
        if self._phase == "approach":
            while self._waypoint_index < len(self._waypoints):
                waypoint = self._waypoints[self._waypoint_index]
                if np.linalg.norm(waypoint - self._last_requested_position) > (
                    self.config.waypoint_tolerance_m
                ):
                    target = _bounded_step(
                        self._last_requested_position,
                        waypoint,
                        self.config.transit_step_m,
                    )
                    self._last_requested_position = target.copy()
                    return self._action(target), "approach", False, None
                self._waypoint_index += 1
            if button_contact:
                self._phase = "fixture_contact"
            else:
                target = _bounded_step(
                    self._last_requested_position,
                    self._press,
                    self.config.press_step_m,
                )
                self._last_requested_position = target.copy()
                return self._action(target), "approach", False, None

        if self._phase == "fixture_contact":
            qualifies = (
                fixture.press_depth_m
                >= float(getattr(self._task, "goal")["target_press_depth_m"])
                and fixture.pressed
            )
            self._press_dwell = self._press_dwell + 1 if qualifies else 0
            if self._press_dwell >= self.config.required_press_dwell_steps:
                self._phase = "retract"
            else:
                self._last_requested_position = self._press.copy()
                return self._action(self._press), "fixture_contact", False, None

        target = _bounded_step(
            self._last_requested_position,
            self._precontact,
            self.config.retract_step_m,
        )
        self._last_requested_position = target.copy()
        released = (
            fixture.press_depth_m <= self.config.release_depth_m
            and not button_contact
            and np.linalg.norm(target - self._precontact)
            <= self.config.waypoint_tolerance_m
        )
        self._release_dwell = self._release_dwell + 1 if released else 0
        done = self._release_dwell >= self.config.release_hold_steps
        return self._action(target), "retract", done, None

    def _action(self, position: np.ndarray) -> RequestedAction:
        values = (
            *np.asarray(position, dtype=np.float64).tolist(),
            *self._orientation.tolist(),
            *(self._finger_targets[name] for name in self._finger_targets),
        )
        return RequestedAction(self._names, tuple(float(value) for value in values))

    def _maximum_non_target_disturbance(self, state: WorldState) -> float:
        button_id = str(getattr(self._task, "goal")["button_id"])
        return max(
            (
                float(np.linalg.norm(np.asarray(entity.position) - initial))
                for entity in state.entities
                if entity.object_id != button_id
                and (initial := self._initial_entities.get(entity.object_id)) is not None
            ),
            default=0.0,
        )


class DeterministicPushExpert:
    """Push one object along its start-to-target axis with one fixed finger."""

    def __init__(
        self,
        *,
        finger_targets: Mapping[str, float],
        config: DeterministicPushConfig,
    ) -> None:
        if not finger_targets or any(
            not name or not math.isfinite(float(value))
            for name, value in finger_targets.items()
        ):
            raise Level4ExpertError("push expert requires finite named finger targets.")
        self._finger_targets = {
            str(name): float(value) for name, value in finger_targets.items()
        }
        self.config = config
        self._names = level4_action_names(tuple(self._finger_targets))
        self._task: object | None = None
        self._metric_task: object | None = None
        self._waypoints: tuple[np.ndarray, ...] = ()
        self._waypoint_index = 0
        self._orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._target_orientation = self._orientation.copy()
        self._direction = np.zeros(2, dtype=np.float64)
        self._precontact = np.zeros(3, dtype=np.float64)
        self._last_requested_position = np.zeros(3, dtype=np.float64)
        self._initial_entities: dict[str, np.ndarray] = {}
        self._phase = "approach"
        self._goal_dwell = 0
        self._release_dwell = 0
        self._push_actions = 0
        self._retract_target: np.ndarray | None = None
        self._terminal_reason: str | None = None
        self._validation: WaypointValidationResult | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def waypoints(self) -> tuple[np.ndarray, ...]:
        return tuple(waypoint.copy() for waypoint in self._waypoints)

    @property
    def direction_xy(self) -> np.ndarray:
        return self._direction.copy()

    @property
    def validation(self) -> WaypointValidationResult | None:
        return self._validation

    def reset(self, task: object, world_state: WorldState) -> None:
        """Resolve a task-local plan and qualify it in copied MuJoCo state."""

        goal = getattr(task, "goal", None)
        if getattr(task, "skill_name", None) != "push_object_to_target" or not isinstance(
            goal, Mapping
        ):
            raise Level4ExpertError(
                "DeterministicPushExpert only supports push_object_to_target in Level 4.3C."
            )
        expected_names = tuple(
            str(field["name"])
            for field in sorted(
                task.collection_config["action_contract"]["named_layout"],
                key=lambda item: int(item["index"]),
            )
        )
        if self._names != expected_names:
            raise Level4ExpertError(
                "scripted expert action names do not match the frozen Level 4 layout."
            )
        object_id = str(goal["object_id"])
        target_id = str(goal["target_zone"])
        object_state = world_state.require_entity(object_id)
        source = np.asarray(object_state.position)
        target = np.asarray(world_state.require_entity(target_id).position)
        delta = target[:2] - source[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 0.0:
            raise Level4ExpertError("push start and target must be distinct.")
        self._direction = delta / distance
        perpendicular = np.asarray([-self._direction[1], self._direction[0]])
        spec = next(
            item for item in task.workcell.config.objects if item.object_id == object_id
        )
        family = spec.family
        if family not in self.config.family_parameters:
            raise Level4ExpertError(f"push expert does not support family {family!r}.")
        parameters = self.config.family_parameters[family]
        pitch = math.radians(float(parameters["wrist_pitch_deg"]))
        yaw = math.atan2(self._direction[1], self._direction[0])
        control_side = str(parameters["control_side"])
        forward_offset = float(parameters["fingertip_forward_offset_m"])
        if control_side == "ahead":
            yaw += math.pi
            control_xy = (
                source[:2]
                + self._direction
                * (forward_offset - spec.footprint_radius_m - self.config.approach_gap_m)
                + perpendicular * self.config.fingertip_lateral_offset_m
            )
        else:
            control_xy = (
                source[:2]
                - self._direction
                * (forward_offset + spec.footprint_radius_m + self.config.approach_gap_m)
                - perpendicular * self.config.fingertip_lateral_offset_m
            )
        self._orientation = np.asarray(
            world_state.robot.base_orientation_wxyz, dtype=np.float64
        )
        self._target_orientation = _yaw_pitch_quaternion(yaw=yaw, pitch=pitch)
        self._precontact = np.asarray(
            [*control_xy, float(parameters["control_height_m"])], dtype=np.float64
        )
        start = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
        transit_z = max(self.config.transit_height_m, float(start[2]))
        self._waypoints = _deduplicate_waypoints(
            (
                np.asarray([start[0], start[1], transit_z]),
                np.asarray([control_xy[0], control_xy[1], transit_z]),
                self._precontact.copy(),
            )
        )
        self._last_requested_position = start.copy()
        self._initial_entities = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in world_state.entities
        }
        self._task = task
        self._metric_task = task.workcell.create_task(
            "push_object_to_target", **dict(goal)
        )
        self._waypoint_index = 0
        self._phase = "approach"
        self._goal_dwell = 0
        self._release_dwell = 0
        self._push_actions = 0
        self._retract_target = None
        self._terminal_reason = None
        self._validation = validate_push_trajectory_on_copy(
            task=task,
            initial_world_state=world_state,
            waypoints=self._waypoints,
            direction_xy=self._direction,
            initial_orientation_wxyz=self._orientation,
            target_orientation_wxyz=self._target_orientation,
            finger_targets=self._finger_targets,
            config=self.config,
        )
        if not self._validation.valid:
            self._terminal_reason = self._validation.reason

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]:
        """Advance the causal approach, contact, settle, and retract phases."""

        if self._task is None or self._metric_task is None or self._validation is None:
            raise Level4ExpertError("reset() must be called before step().")
        current = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
        if self._terminal_reason is not None:
            return self._action(current), self._phase, True, self._terminal_reason
        object_id = str(getattr(self._task, "goal")["object_id"])
        reason = _unsafe_push_contact_reason(world_state, object_id=object_id)
        if reason is not None:
            self._terminal_reason = reason
            return self._action(current), self._phase, True, reason
        if self._maximum_non_target_disturbance(world_state) > (
            self.config.maximum_non_target_disturbance_m
        ):
            self._terminal_reason = "non_target_disturbance"
            return self._action(current), self._phase, True, self._terminal_reason

        object_contact = any(
            object_id in pair and any(name.startswith("rh_") for name in pair)
            for pair in world_state.contacts
        )
        if self._phase == "approach":
            if self._waypoint_index == 1 and not np.allclose(
                self._orientation,
                self._target_orientation,
                atol=self.config.waypoint_tolerance_m,
                rtol=0.0,
            ):
                self._orientation = _bounded_quaternion_step(
                    self._orientation,
                    self._target_orientation,
                    self.config.orientation_step_rad,
                )
                return (
                    self._action(self._last_requested_position),
                    "approach",
                    False,
                    None,
                )
            while self._waypoint_index < len(self._waypoints):
                waypoint = self._waypoints[self._waypoint_index]
                if np.linalg.norm(waypoint - self._last_requested_position) > (
                    self.config.waypoint_tolerance_m
                ):
                    step_size = (
                        self.config.descent_step_m
                        if self._waypoint_index == len(self._waypoints) - 1
                        else self.config.transit_step_m
                    )
                    target = _bounded_step(
                        self._last_requested_position, waypoint, step_size
                    )
                    self._last_requested_position = target.copy()
                    return self._action(target), "approach", False, None
                self._waypoint_index += 1
                if self._waypoint_index == 1:
                    return (
                        self._action(self._last_requested_position),
                        "approach",
                        False,
                        None,
                    )
            if object_contact:
                self._phase = "push_contact"

        object_state = world_state.require_entity(object_id)
        metric = self._metric_task.evaluate(world_state)
        target_state = world_state.require_entity(
            str(getattr(self._task, "goal")["target_zone"])
        )
        distance = math.dist(object_state.position[:2], target_state.position[:2])
        tilt = _object_upright_tilt_rad(object_state.orientation_wxyz)
        if self._phase != "approach" and tilt > self.config.maximum_object_tilt_rad:
            self._terminal_reason = "object_tipped"
            return self._action(current), self._phase, True, self._terminal_reason
        if self._phase in {"approach", "push_contact"}:
            if distance <= self.config.target_stop_distance_m:
                self._phase = "settle"
            else:
                self._phase = "push_contact" if object_contact else self._phase
                self._push_actions += 1
                if self._push_actions > self.config.maximum_push_actions:
                    self._terminal_reason = "push_timeout"
                    return self._action(current), self._phase, True, self._terminal_reason
                target = self._last_requested_position.copy()
                target[:2] += self._direction * self.config.push_step_m
                self._last_requested_position = target.copy()
                return self._action(target), self._phase, False, None

        if self._phase == "settle":
            qualifies = metric.qualifies
            self._goal_dwell = self._goal_dwell + 1 if qualifies else 0
            if self._goal_dwell < self.config.required_goal_dwell_steps:
                return self._action(self._last_requested_position), "settle", False, None
            self._phase = "retract"
            self._retract_target = self._last_requested_position.copy()
            self._retract_target[:2] -= (
                self._direction * self.config.retract_distance_m
            )

        assert self._retract_target is not None
        target = _bounded_step(
            self._last_requested_position,
            self._retract_target,
            self.config.retract_step_m,
        )
        self._last_requested_position = target.copy()
        released = (
            not object_contact
            and np.linalg.norm(target - self._retract_target)
            <= self.config.waypoint_tolerance_m
        )
        terminal_qualifies = released and metric.qualifies
        self._release_dwell = self._release_dwell + 1 if terminal_qualifies else 0
        return (
            self._action(target),
            "retract",
            self._release_dwell
            >= max(
                self.config.release_hold_steps,
                self.config.required_terminal_dwell_steps,
            ),
            None,
        )

    def _action(self, position: np.ndarray) -> RequestedAction:
        values = (
            *position.tolist(),
            *self._orientation.tolist(),
            *(self._finger_targets[name] for name in self._finger_targets),
        )
        return RequestedAction(self._names, tuple(float(value) for value in values))

    def _maximum_non_target_disturbance(self, state: WorldState) -> float:
        object_id = str(getattr(self._task, "goal")["object_id"])
        return max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(entity.position[:2]) - np.asarray(initial[:2])
                    )
                )
                for entity in state.entities
                if entity.object_id != object_id
                and (initial := self._initial_entities.get(entity.object_id)) is not None
            ),
            default=0.0,
        )


class DeterministicGraspLiftExpert:
    """Approach, close, lift, and physically qualify one family grasp."""

    def __init__(
        self,
        *,
        open_finger_targets: Mapping[str, float],
        closed_finger_targets: Mapping[str, float],
        config: DeterministicGraspLiftConfig,
    ) -> None:
        if tuple(open_finger_targets) != tuple(closed_finger_targets):
            raise Level4ExpertError(
                "grasp open and closed poses must use the same actuator order."
            )
        if not open_finger_targets or any(
            not name
            or not math.isfinite(float(open_finger_targets[name]))
            or not math.isfinite(float(closed_finger_targets[name]))
            for name in open_finger_targets
        ):
            raise Level4ExpertError("grasp hand poses require finite named targets.")
        self._open_targets = {
            str(name): float(value) for name, value in open_finger_targets.items()
        }
        self._closed_targets = {
            str(name): float(value) for name, value in closed_finger_targets.items()
        }
        self.config = config
        self._names = level4_action_names(tuple(self._open_targets))
        self._task: object | None = None
        self._template: GraspFamilyTemplate | None = None
        self._waypoints: tuple[np.ndarray, ...] = ()
        self._waypoint_index = 0
        self._orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._target_orientation = self._orientation.copy()
        self._last_requested_position = np.zeros(3, dtype=np.float64)
        self._lift_target = np.zeros(3, dtype=np.float64)
        self._initial_entities: dict[str, np.ndarray] = {}
        self._initial_object_z = 0.0
        self._initial_object_orientation = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._orientation_symmetry = "none"
        self._orientation_feedback_enabled = False
        self._synergy = 0.0
        self._phase = "approach"
        self._ever_held = False
        self._hold_dwell = 0
        self._closed_acquisition_actions = 0
        self._retention_gap = 0
        self._terminal_reason: str | None = None
        self._validation: WaypointValidationResult | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def waypoints(self) -> tuple[np.ndarray, ...]:
        return tuple(waypoint.copy() for waypoint in self._waypoints)

    @property
    def grasp_synergy(self) -> float:
        return 0.0 if self._template is None else self._template.grasp_synergy

    @property
    def validation(self) -> WaypointValidationResult | None:
        return self._validation

    def reset(self, task: object, world_state: WorldState) -> None:
        """Resolve the object-relative family template and validate a copied run."""

        goal = getattr(task, "goal", None)
        if getattr(task, "skill_name", None) != "pick_object" or not isinstance(
            goal, Mapping
        ):
            raise Level4ExpertError(
                "DeterministicGraspLiftExpert only supports pick_object in Level 4.3D."
            )
        expected_names = tuple(
            str(field["name"])
            for field in sorted(
                task.collection_config["action_contract"]["named_layout"],
                key=lambda item: int(item["index"]),
            )
        )
        if self._names != expected_names:
            raise Level4ExpertError(
                "scripted expert action names do not match the frozen Level 4 layout."
            )
        object_id = str(goal["object_id"])
        spec = next(
            item for item in task.workcell.config.objects if item.object_id == object_id
        )
        template = self.config.family_templates[spec.family]
        object_state = world_state.require_entity(object_id)
        source = np.asarray(object_state.position)
        grasp_position = source + np.asarray(template.object_relative_position_m)
        start = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
        transit_z = max(self.config.transit_height_m, float(start[2]))
        self._waypoints = _deduplicate_waypoints(
            (
                np.asarray([start[0], start[1], transit_z]),
                np.asarray([grasp_position[0], grasp_position[1], transit_z]),
                grasp_position,
            )
        )
        self._template = template
        self._orientation = np.asarray(
            world_state.robot.base_orientation_wxyz, dtype=np.float64
        )
        self._target_orientation = _conditioned_grasp_orientation(
            template,
            np.asarray(object_state.orientation_wxyz, dtype=np.float64),
        )
        self._last_requested_position = start.copy()
        self._lift_target = grasp_position + np.asarray(
            [0.0, 0.0, template.lift_distance_m]
        )
        self._initial_entities = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in world_state.entities
        }
        self._initial_object_z = float(source[2])
        self._initial_object_orientation = _normalize_quaternion(
            np.asarray(
                world_state.require_entity(object_id).orientation_wxyz,
                dtype=np.float64,
            )
        )
        self._orientation_symmetry = template.orientation_symmetry
        self._orientation_feedback_enabled = template.orientation_feedback_enabled
        self._task = task
        self._waypoint_index = 0
        self._synergy = 0.0
        self._phase = "approach"
        self._ever_held = False
        self._hold_dwell = 0
        self._closed_acquisition_actions = 0
        self._retention_gap = 0
        self._terminal_reason = None
        self._validation = validate_grasp_lift_trajectory_on_copy(
            task=task,
            initial_world_state=world_state,
            waypoints=self._waypoints,
            lift_target=self._lift_target,
            initial_orientation_wxyz=self._orientation,
            target_orientation_wxyz=self._target_orientation,
            open_finger_targets=self._open_targets,
            closed_finger_targets=self._closed_targets,
            template=template,
            config=self.config,
        )
        if not self._validation.valid:
            self._terminal_reason = self._validation.reason

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]:
        """Advance causal approach/acquire/lift/stabilize phases."""

        if self._task is None or self._template is None or self._validation is None:
            raise Level4ExpertError("reset() must be called before step().")
        current = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
        if self._terminal_reason is not None:
            return self._action(current), self._phase, True, self._terminal_reason
        object_id = str(getattr(self._task, "goal")["object_id"])
        object_state = world_state.require_entity(object_id)
        relation = world_state.relation_for(object_id)
        held = relation.held_by == "rh_palm"
        acquired = held or _target_hand_contact_body_count(
            world_state, object_id=object_id
        ) >= 1
        self._ever_held = self._ever_held or held
        self._retention_gap = (
            0 if acquired else self._retention_gap + int(self._ever_held)
        )
        lift_height = float(object_state.position[2]) - self._initial_object_z
        orientation_error = _object_orientation_error(
            self._initial_object_orientation,
            np.asarray(object_state.orientation_wxyz, dtype=np.float64),
            symmetry=self._orientation_symmetry,
        )
        if held and self._orientation_feedback_enabled:
            self._orientation = _orientation_preserving_hand_target(
                reference_object_orientation=self._initial_object_orientation,
                observed_object_orientation=np.asarray(
                    object_state.orientation_wxyz, dtype=np.float64
                ),
                observed_hand_orientation=np.asarray(
                    world_state.robot.base_orientation_wxyz, dtype=np.float64
                ),
                prior_requested_hand_orientation=self._orientation,
                maximum_step_rad=self.config.orientation_correction_step_rad,
                symmetry=self._orientation_symmetry,
            )
        allow_table = self._phase in {"approach", "acquire"} or lift_height < 0.040
        reason = _unsafe_grasp_contact_reason(
            world_state, object_id=object_id, allow_table_contact=allow_table
        )
        if reason is None and self._maximum_non_target_disturbance(world_state) > (
            self.config.maximum_non_target_disturbance_m
        ):
            reason = "non_target_disturbance"
        if reason is None and _has_joint_limit_violation(
            self._task.workcell, tolerance=self.config.joint_limit_tolerance_rad
        ):
            reason = "joint_limit_violation"
        if reason is None and self._retention_gap > (
            self.config.maximum_retention_gap_steps
        ):
            reason = "slip_drop"
        if (
            reason is None
            and self._ever_held
            and orientation_error
            > self.config.maximum_object_orientation_deviation_rad
        ):
            reason = "object_orientation_deviation"
        if reason is not None:
            self._terminal_reason = reason
            return self._action(current), self._phase, True, reason

        if self._phase == "approach":
            if self._waypoint_index == 1 and not np.allclose(
                self._orientation,
                self._target_orientation,
                atol=self.config.waypoint_tolerance_m,
                rtol=0.0,
            ):
                self._orientation = _bounded_quaternion_step(
                    self._orientation,
                    self._target_orientation,
                    self.config.orientation_step_rad,
                )
                return self._action(self._last_requested_position), "approach", False, None
            while self._waypoint_index < len(self._waypoints):
                waypoint = self._waypoints[self._waypoint_index]
                if np.linalg.norm(waypoint - self._last_requested_position) > (
                    self.config.waypoint_tolerance_m
                ):
                    step_size = (
                        self.config.descent_step_m
                        if self._waypoint_index == len(self._waypoints) - 1
                        else self.config.transit_step_m
                    )
                    target = _bounded_step(
                        self._last_requested_position, waypoint, step_size
                    )
                    self._last_requested_position = target.copy()
                    return self._action(target), "approach", False, None
                self._waypoint_index += 1
                if self._waypoint_index == 1:
                    return self._action(self._last_requested_position), "approach", False, None
            self._phase = "acquire"

        if self._phase == "acquire":
            self._synergy = min(
                self._template.grasp_synergy,
                self._synergy + self.config.synergy_step,
            )
            if self._synergy >= self._template.grasp_synergy:
                if not acquired:
                    self._closed_acquisition_actions += 1
                    if self._closed_acquisition_actions >= (
                        self.config.maximum_closed_acquisition_actions
                    ):
                        self._terminal_reason = "failed_acquisition"
                        return (
                            self._action(self._last_requested_position),
                            "acquire",
                            True,
                            self._terminal_reason,
                        )
                    return self._action(self._last_requested_position), "acquire", False, None
                self._phase = "lift"
            return self._action(self._last_requested_position), "acquire", False, None

        if self._phase == "lift":
            if np.linalg.norm(self._lift_target - self._last_requested_position) > (
                self.config.waypoint_tolerance_m
            ):
                target = _bounded_step(
                    self._last_requested_position,
                    self._lift_target,
                    self.config.lift_step_m,
                )
                self._last_requested_position = target.copy()
                return self._action(target), "lift", False, None
            self._phase = "stabilize"

        speed = float(np.linalg.norm(np.asarray(object_state.linear_velocity)))
        stable = (
            held
            and relation.supported_by is None
            and lift_height >= 0.040
            and speed <= self.config.maximum_hold_speed_m_s
            and orientation_error
            <= self.config.maximum_terminal_orientation_error_rad
        )
        self._hold_dwell = self._hold_dwell + 1 if stable else 0
        done = self._hold_dwell >= self.config.required_hold_steps
        return self._action(self._last_requested_position), "stabilize", done, None

    def _action(self, position: np.ndarray) -> RequestedAction:
        finger_targets = _interpolate_finger_targets(
            self._open_targets, self._closed_targets, self._synergy
        )
        values = (
            *np.asarray(position, dtype=np.float64).tolist(),
            *self._orientation.tolist(),
            *(finger_targets[name] for name in self._open_targets),
        )
        return RequestedAction(self._names, tuple(float(value) for value in values))

    def _maximum_non_target_disturbance(self, state: WorldState) -> float:
        object_id = str(getattr(self._task, "goal")["object_id"])
        return max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(entity.position[:2])
                        - self._initial_entities[entity.object_id][:2]
                    )
                )
                for entity in state.entities
                if entity.object_id != object_id
                and entity.object_id in self._initial_entities
            ),
            default=0.0,
        )


@dataclass(frozen=True)
class _ExpertTaskView:
    """Minimal task adapter used to compose existing scripted experts."""

    skill_name: str
    goal: Mapping[str, object]
    workcell: object
    collection_config: Mapping[str, object]


class DeterministicPlaceExpert:
    """Transport one genuinely held object, release it, settle, and retract."""

    def __init__(
        self,
        *,
        open_finger_targets: Mapping[str, float],
        closed_finger_targets: Mapping[str, float],
        grasp_config: DeterministicGraspLiftConfig,
        config: DeterministicPlaceConfig,
        validate_on_reset: bool = True,
    ) -> None:
        if tuple(open_finger_targets) != tuple(closed_finger_targets):
            raise Level4ExpertError(
                "place open and closed poses must use the same actuator order."
            )
        self._open_targets = {
            str(name): float(value) for name, value in open_finger_targets.items()
        }
        self._closed_targets = {
            str(name): float(value) for name, value in closed_finger_targets.items()
        }
        if not self._open_targets or any(
            not name
            or not math.isfinite(self._open_targets[name])
            or not math.isfinite(self._closed_targets[name])
            for name in self._open_targets
        ):
            raise Level4ExpertError("place hand poses require finite named targets.")
        self.grasp_config = grasp_config
        self.config = config
        self._validate_on_reset = validate_on_reset
        self._names = level4_action_names(tuple(self._open_targets))
        self._task: object | None = None
        self._metric_task: object | None = None
        self._waypoints: tuple[np.ndarray, ...] = ()
        self._waypoint_index = 0
        self._orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._target_orientation = self._orientation.copy()
        self._preserved_object_orientation = self._orientation.copy()
        self._orientation_symmetry = "none"
        self._orientation_feedback_enabled = False
        self._last_requested_position = np.zeros(3, dtype=np.float64)
        self._release_clearance_target = np.zeros(3, dtype=np.float64)
        self._retract_target = np.zeros(3, dtype=np.float64)
        self._release_backoff_x = 0.0
        self._placement_plan_resolved = False
        self._initial_entities: dict[str, np.ndarray] = {}
        self._synergy = 0.0
        self._phase = "transport"
        self._settle_actions = 0
        self._release_detected = False
        self._retention_gap = 0
        self._terminal_dwell = 0
        self._terminal_reason: str | None = None
        self._validation: WaypointValidationResult | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def waypoints(self) -> tuple[np.ndarray, ...]:
        return tuple(point.copy() for point in self._waypoints)

    @property
    def validation(self) -> WaypointValidationResult | None:
        return self._validation

    def reset(self, task: object, world_state: WorldState) -> None:
        """Resolve held-object waypoints and validate them in copied state."""

        goal = getattr(task, "goal", None)
        if getattr(task, "skill_name", None) not in {
            "place_held_object",
            "pick_place_sequence",
        } or not isinstance(goal, Mapping):
            raise Level4ExpertError(
                "DeterministicPlaceExpert requires place_held_object or "
                "pick_place_sequence."
            )
        expected_names = tuple(
            str(field["name"])
            for field in sorted(
                task.collection_config["action_contract"]["named_layout"],
                key=lambda item: int(item["index"]),
            )
        )
        if self._names != expected_names:
            raise Level4ExpertError(
                "scripted expert action names do not match the frozen Level 4 layout."
            )
        object_id = str(goal["object_id"])
        target_id = str(goal["target_id"])
        relation = world_state.relation_for(object_id)
        if relation.held_by != "rh_palm" or relation.supported_by is not None:
            self._validation = WaypointValidationResult(
                False, "object_not_genuinely_held", 0, 0.0
            )
            self._terminal_reason = "object_not_genuinely_held"
            self._task = task
            return

        spec = next(
            item for item in task.workcell.config.objects if item.object_id == object_id
        )
        template = self.grasp_config.family_templates[spec.family]
        current = np.asarray(
            world_state.robot.end_effector_position, dtype=np.float64
        )
        object_position = np.asarray(
            world_state.require_entity(object_id).position, dtype=np.float64
        )
        raised = current.copy()
        raised[2] += max(
            0.0, self.config.transport_object_height_m - object_position[2]
        )
        self._waypoints = (raised,)
        self._waypoint_index = 0
        grasp_orientation = _normalize_quaternion(
            np.asarray(world_state.robot.base_orientation_wxyz, dtype=np.float64)
        )
        self._orientation = grasp_orientation
        self._target_orientation = grasp_orientation.copy()
        self._preserved_object_orientation = _normalize_quaternion(
            np.asarray(
                world_state.require_entity(object_id).orientation_wxyz,
                dtype=np.float64,
            )
        )
        self._orientation_symmetry = template.orientation_symmetry
        self._orientation_feedback_enabled = (
            template.transport_orientation_feedback_enabled
        )
        self._release_backoff_x = self.config.family_release_backoff_x_m[spec.family]
        self._last_requested_position = current.copy()
        self._release_clearance_target = raised.copy()
        self._retract_target = raised.copy()
        self._placement_plan_resolved = False
        self._initial_entities = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in world_state.entities
        }
        self._synergy = template.grasp_synergy
        self._phase = "transport"
        self._settle_actions = 0
        self._release_detected = False
        self._retention_gap = 0
        self._terminal_dwell = 0
        self._terminal_reason = None
        self._task = task
        self._metric_task = task.workcell.create_task(
            "place_held_object", object_id=object_id, target_id=target_id
        )
        if self._validate_on_reset:
            self._validation = validate_place_trajectory_on_copy(
                task=task,
                initial_world_state=world_state,
                open_finger_targets=self._open_targets,
                closed_finger_targets=self._closed_targets,
                grasp_config=self.grasp_config,
                config=self.config,
            )
        else:
            self._validation = WaypointValidationResult(True, None, 0, 0.0)
        if not self._validation.valid:
            self._terminal_reason = self._validation.reason

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]:
        """Advance transport, placement, release, settle, and retract."""

        if self._task is None or self._validation is None:
            raise Level4ExpertError("reset() must be called before step().")
        current = np.asarray(
            world_state.robot.end_effector_position, dtype=np.float64
        )
        if self._terminal_reason is not None:
            return self._action(current), self._phase, True, self._terminal_reason
        object_id = str(getattr(self._task, "goal")["object_id"])
        target_id = str(getattr(self._task, "goal")["target_id"])
        object_state = world_state.require_entity(object_id)
        relation = world_state.relation_for(object_id)
        target = world_state.require_entity(target_id)
        distance = math.dist(object_state.position, target.position)
        orientation_error = _object_orientation_error(
            self._preserved_object_orientation,
            np.asarray(object_state.orientation_wxyz, dtype=np.float64),
            symmetry=self._orientation_symmetry,
        )
        if (
            self._orientation_feedback_enabled
            and relation.held_by == "rh_palm"
            and self._phase in {
            "transport",
            "place",
            "release",
            }
        ):
            self._orientation = _orientation_preserving_hand_target(
                reference_object_orientation=self._preserved_object_orientation,
                observed_object_orientation=np.asarray(
                    object_state.orientation_wxyz, dtype=np.float64
                ),
                observed_hand_orientation=np.asarray(
                    world_state.robot.base_orientation_wxyz, dtype=np.float64
                ),
                prior_requested_hand_orientation=self._orientation,
                maximum_step_rad=self.grasp_config.orientation_correction_step_rad,
                symmetry=self._orientation_symmetry,
            )

        reason = _unsafe_place_contact_reason(world_state, object_id=object_id)
        if reason is None and self._maximum_non_target_disturbance(world_state) > (
            self.config.maximum_non_target_disturbance_m
        ):
            reason = "non_target_disturbance"
        if reason is None and _has_joint_limit_violation(
            self._task.workcell, tolerance=self.config.joint_limit_tolerance_rad
        ):
            reason = "joint_limit_violation"
        if (
            reason is None
            and relation.held_by == "rh_palm"
            and orientation_error
            > self.grasp_config.maximum_object_orientation_deviation_rad
        ):
            reason = "object_orientation_deviation"
        if reason is None and self._phase in {"transport", "place"}:
            retained = (
                relation.held_by == "rh_palm"
                or _target_hand_contact_body_count(
                    world_state, object_id=object_id
                )
                >= 1
            )
            self._retention_gap = (
                0 if retained else self._retention_gap + 1
            )
            if not retained and distance <= (
                self.config.release_position_tolerance_m
            ):
                if self._phase == "transport":
                    self._phase = "place"
                    return self._action(current), "place", False, None
                self._phase = "release"
            elif self._retention_gap > self.grasp_config.maximum_retention_gap_steps:
                reason = "premature_release"
        if reason is not None:
            self._terminal_reason = reason
            return self._action(current), self._phase, True, reason

        if self._phase in {"transport", "place"}:
            if self._waypoint_index >= 1 and not self._placement_plan_resolved:
                self._resolve_placement_plan(world_state)
            while self._waypoint_index < len(self._waypoints):
                waypoint = self._waypoints[self._waypoint_index]
                phase = (
                    "place"
                    if self._placement_plan_resolved
                    and self._waypoint_index == len(self._waypoints) - 1
                    else "transport"
                )
                self._phase = phase
                if np.linalg.norm(waypoint - self._last_requested_position) > (
                    self.config.waypoint_tolerance_m
                ):
                    step_size = (
                        self.config.descent_step_m
                        if phase == "place"
                        else self.config.transport_step_m
                    )
                    requested = _bounded_step(
                        self._last_requested_position, waypoint, step_size
                    )
                    self._last_requested_position = requested.copy()
                    return self._action(requested), phase, False, None
                self._waypoint_index += 1
            if not self._placement_plan_resolved:
                self._phase = "transport"
                return (
                    self._action(self._last_requested_position),
                    "transport",
                    False,
                    None,
                )
            self._phase = "release"

        if self._phase == "release":
            if not self._release_detected and relation.held_by is not None:
                self._synergy = max(
                    0.0, self._synergy - self.config.release_synergy_step
                )
                return (
                    self._action(self._last_requested_position),
                    "release",
                    False,
                    None,
                )
            if not self._release_detected:
                self._release_detected = True
                self._release_clearance_target = self._last_requested_position.copy()
                self._release_clearance_target[2] += (
                    self.config.release_clearance_height_m
                )
                if float(target.position[0]) > 0.0:
                    self._release_clearance_target[0] -= (
                        self._release_backoff_x
                    )
            self._synergy = max(
                0.0, self._synergy - self.config.release_synergy_step
            )
            requested = _bounded_step(
                self._last_requested_position,
                self._release_clearance_target,
                self.config.retract_step_m,
            )
            self._last_requested_position = requested.copy()
            if np.linalg.norm(requested - self._release_clearance_target) > (
                self.config.waypoint_tolerance_m
            ) or self._synergy > 0.0:
                return self._action(requested), "release", False, None
            self._phase = "settle"

        assert self._metric_task is not None
        metric = self._metric_task.evaluate(world_state)
        if self._phase == "settle":
            self._settle_actions += 1
            if self._settle_actions > self.config.maximum_settle_actions:
                self._terminal_reason = "settle_timeout"
                return self._action(current), "settle", True, self._terminal_reason
            if metric.success:
                self._phase = "retract"
            else:
                return self._action(self._last_requested_position), "settle", False, None

        requested = _bounded_step(
            self._last_requested_position,
            self._retract_target,
            self.config.retract_step_m,
        )
        self._last_requested_position = requested.copy()
        at_retract = (
            np.linalg.norm(requested - self._retract_target)
            <= self.config.waypoint_tolerance_m
        )
        target_contact = any(
            object_id in pair and any(name.startswith("rh_") for name in pair)
            for pair in world_state.contacts
        )
        terminal = at_retract and not target_contact and metric.qualifies
        self._terminal_dwell = self._terminal_dwell + 1 if terminal else 0
        done = self._terminal_dwell >= self.config.required_terminal_dwell_steps
        return self._action(requested), "retract", done, None

    def _action(self, position: np.ndarray) -> RequestedAction:
        finger_targets = _interpolate_finger_targets(
            self._open_targets, self._closed_targets, self._synergy
        )
        values = (
            *np.asarray(position, dtype=np.float64).tolist(),
            *self._orientation.tolist(),
            *(finger_targets[name] for name in self._open_targets),
        )
        return RequestedAction(self._names, tuple(float(value) for value in values))

    def _resolve_placement_plan(self, world_state: WorldState) -> None:
        object_id = str(getattr(self._task, "goal")["object_id"])
        target_id = str(getattr(self._task, "goal")["target_id"])
        spec = next(
            item
            for item in self._task.workcell.config.objects
            if item.object_id == object_id
        )
        current = np.asarray(
            world_state.robot.end_effector_position, dtype=np.float64
        )
        object_position = np.asarray(
            world_state.require_entity(object_id).position, dtype=np.float64
        )
        target_position = np.asarray(
            world_state.require_entity(target_id).position, dtype=np.float64
        )
        held_offset = object_position - current
        over_target = self._last_requested_position.copy()
        over_target[:2] += target_position[:2] - object_position[:2]
        desired_object = target_position.copy()
        desired_object[:2] += np.asarray(
            self.config.family_target_offset_xy_m[spec.family], dtype=np.float64
        )
        desired_object[0] = min(
            desired_object[0], self.config.maximum_placement_center_x_m
        )
        desired_object[2] = (
            max(float(target_position[2]), spec.resting_height_m)
            + self.config.placement_clearance_m
        )
        placement = desired_object - held_offset
        self._waypoints = self._waypoints + _deduplicate_waypoints(
            (over_target, placement)
        )
        self._retract_target = placement + np.asarray(
            [0.0, 0.0, self.config.retract_height_m], dtype=np.float64
        )
        if float(target_position[0]) > 0.0:
            self._retract_target[0] -= self._release_backoff_x
        self._placement_plan_resolved = True

    def _maximum_non_target_disturbance(self, state: WorldState) -> float:
        object_id = str(getattr(self._task, "goal")["object_id"])
        return max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(entity.position[:2])
                        - self._initial_entities[entity.object_id][:2]
                    )
                )
                for entity in state.entities
                if entity.object_id != object_id
                and entity.object_id in self._initial_entities
            ),
            default=0.0,
        )


class DeterministicPickPlaceExpert:
    """Compose the qualified grasp-and-lift expert with deterministic place."""

    def __init__(
        self,
        *,
        open_finger_targets: Mapping[str, float],
        closed_finger_targets: Mapping[str, float],
        grasp_config: DeterministicGraspLiftConfig,
        place_config: DeterministicPlaceConfig,
        validate_on_reset: bool = True,
    ) -> None:
        self._open_targets = dict(open_finger_targets)
        self._closed_targets = dict(closed_finger_targets)
        self.grasp_config = grasp_config
        self.place_config = place_config
        self._validate_on_reset = validate_on_reset
        self._grasp = DeterministicGraspLiftExpert(
            open_finger_targets=open_finger_targets,
            closed_finger_targets=closed_finger_targets,
            config=grasp_config,
        )
        self._place = DeterministicPlaceExpert(
            open_finger_targets=open_finger_targets,
            closed_finger_targets=closed_finger_targets,
            grasp_config=grasp_config,
            config=place_config,
            # The complete composed trajectory is qualified once by
            # validate_pick_place_trajectory_on_copy below. Re-validating only
            # the place suffix from a numerically evolved live hand state can
            # disagree with that already-qualified full trajectory.
            validate_on_reset=False,
        )
        self._task: object | None = None
        self._stage = "grasp"
        self._place_reset = False
        self._validation: WaypointValidationResult | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._grasp.action_names

    @property
    def waypoints(self) -> tuple[np.ndarray, ...]:
        return self._grasp.waypoints + self._place.waypoints

    @property
    def validation(self) -> WaypointValidationResult | None:
        return self._validation

    def reset(self, task: object, world_state: WorldState) -> None:
        goal = getattr(task, "goal", None)
        if getattr(task, "skill_name", None) != "pick_place_sequence" or not isinstance(
            goal, Mapping
        ):
            raise Level4ExpertError(
                "DeterministicPickPlaceExpert only supports pick_place_sequence."
            )
        pick_task = _ExpertTaskView(
            skill_name="pick_object",
            goal={"object_id": str(goal["object_id"])},
            workcell=task.workcell,
            collection_config=task.collection_config,
        )
        self._grasp.reset(pick_task, world_state)
        if self._grasp.validation is None or not self._grasp.validation.valid:
            self._validation = self._grasp.validation
        elif self._validate_on_reset:
            self._validation = validate_pick_place_trajectory_on_copy(
                task=task,
                initial_world_state=world_state,
                open_finger_targets=self._open_targets,
                closed_finger_targets=self._closed_targets,
                grasp_config=self.grasp_config,
                place_config=self.place_config,
            )
        else:
            self._validation = WaypointValidationResult(True, None, 0, 0.0)
        self._task = task
        self._stage = "grasp"
        self._place_reset = False

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]:
        if self._task is None or self._validation is None:
            raise Level4ExpertError("reset() must be called before step().")
        if not self._validation.valid:
            current = np.asarray(
                world_state.robot.end_effector_position, dtype=np.float64
            )
            action = self._grasp._action(current)
            return action, self._stage, True, self._validation.reason
        if self._stage == "grasp":
            action, phase, done, reason = self._grasp.step(world_state)
            if reason is not None:
                return action, phase, True, reason
            if done:
                self._stage = "place"
            return action, phase, False, None
        if not self._place_reset:
            self._place.reset(self._task, world_state)
            self._place_reset = True
            if self._place.validation is None or not self._place.validation.valid:
                action = self._grasp._action(
                    np.asarray(
                        world_state.robot.end_effector_position, dtype=np.float64
                    )
                )
                reason = (
                    self._place.validation.reason
                    if self._place.validation is not None
                    else "place_validation_failed"
                )
                return action, "transport", True, reason
        return self._place.step(world_state)


def validate_place_trajectory_on_copy(
    *,
    task: object,
    initial_world_state: WorldState,
    open_finger_targets: Mapping[str, float],
    closed_finger_targets: Mapping[str, float],
    grasp_config: DeterministicGraspLiftConfig,
    config: DeterministicPlaceConfig,
) -> WaypointValidationResult:
    """Qualify a complete held-object placement in copied MuJoCo state."""

    from dexvision.sim.workcell import Workcell

    live = task.workcell
    scratch = Workcell(live.config.config_path)
    scratch.env.model.geom_condim[:] = live.env.model.geom_condim
    scratch.env.model.geom_friction[:] = live.env.model.geom_friction
    checked = 0
    maximum_disturbance = 0.0
    object_id = str(task.goal["object_id"])
    target_id = str(task.goal["target_id"])
    try:
        scratch.reset(seed=int(live._seed))
        scratch.env._mujoco.mj_copyData(
            scratch.env.data, scratch.env.model, live.env.data
        )
        scratch.env._mujoco.mj_forward(scratch.env.model, scratch.env.data)
        state = scratch.get_world_state()
        initial_positions = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in state.entities
        }
        preserved_orientation = state.require_entity(object_id).orientation_wxyz
        view = _ExpertTaskView(
            skill_name="place_held_object",
            goal={"object_id": object_id, "target_id": target_id},
            workcell=scratch,
            collection_config=task.collection_config,
        )
        candidate = DeterministicPlaceExpert(
            open_finger_targets=open_finger_targets,
            closed_finger_targets=closed_finger_targets,
            grasp_config=grasp_config,
            config=config,
            validate_on_reset=False,
        )
        candidate.reset(view, state)
        if candidate.validation is None or not candidate.validation.valid:
            reason = (
                candidate.validation.reason
                if candidate.validation is not None
                else "place_validation_failed"
            )
            return WaypointValidationResult(False, reason, checked, maximum_disturbance)

        for _ in range(config.maximum_total_actions):
            action, phase, done, reason = candidate.step(state)
            if reason is not None:
                return WaypointValidationResult(
                    False, reason, checked, maximum_disturbance
                )
            requested = action.base_position
            workspace_reason = _requested_workspace_reason(scratch, requested)
            if workspace_reason is not None:
                return WaypointValidationResult(
                    False, workspace_reason, checked, maximum_disturbance
                )
            scratch.env.set_mocap_pose(
                str(scratch.config.scene["hand_base_target"]),
                position=requested,
                orientation_quat=action.base_orientation_wxyz,
            )
            scratch.env.set_joint_targets(action.finger_targets)
            if _orientation_hold_phase(phase):
                scratch.preserve_object_orientation(object_id, preserved_orientation)
            state = scratch.step(n_steps=config.sim_steps_per_action)
            checked += 1
            maximum_disturbance = max(
                maximum_disturbance,
                _maximum_planar_non_target_disturbance(
                    state, object_id=object_id, initial_positions=initial_positions
                ),
            )
            if maximum_disturbance > config.maximum_non_target_disturbance_m:
                return WaypointValidationResult(
                    False, "non_target_disturbance", checked, maximum_disturbance
                )
            if done:
                final_metric = scratch.create_task(
                    "place_held_object", object_id=object_id, target_id=target_id
                ).evaluate(state)
                supported = state.relation_for(object_id).supported_by == str(
                    scratch.config.scene["table_body"]
                )
                if not final_metric.qualifies or not supported:
                    return WaypointValidationResult(
                        False,
                        "place_terminal_metric_not_satisfied",
                        checked,
                        maximum_disturbance,
                    )
                return WaypointValidationResult(
                    True, None, checked, maximum_disturbance
                )
        return WaypointValidationResult(
            False, "place_timeout", checked, maximum_disturbance
        )
    finally:
        scratch.close()


def _orientation_hold_phase(phase: str) -> bool:
    """Return whether the selected object must retain its reset orientation."""

    return phase in {"lift", "stabilize", "transport", "place"}


def validate_pick_place_trajectory_on_copy(
    *,
    task: object,
    initial_world_state: WorldState,
    open_finger_targets: Mapping[str, float],
    closed_finger_targets: Mapping[str, float],
    grasp_config: DeterministicGraspLiftConfig,
    place_config: DeterministicPlaceConfig,
) -> WaypointValidationResult:
    """Qualify the complete composed pick/place before live recording."""

    from dexvision.sim.workcell import Workcell

    live = task.workcell
    scratch = Workcell(live.config.config_path)
    scratch.env.model.geom_condim[:] = live.env.model.geom_condim
    scratch.env.model.geom_friction[:] = live.env.model.geom_friction
    checked = 0
    maximum_disturbance = 0.0
    object_id = str(task.goal["object_id"])
    target_id = str(task.goal["target_id"])
    try:
        scratch.reset(seed=int(live._seed))
        scratch.env._mujoco.mj_copyData(
            scratch.env.data, scratch.env.model, live.env.data
        )
        scratch.env._mujoco.mj_forward(scratch.env.model, scratch.env.data)
        state = scratch.get_world_state()
        initial_positions = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in state.entities
        }
        preserved_orientation = initial_world_state.require_entity(
            object_id
        ).orientation_wxyz
        view = _ExpertTaskView(
            skill_name="pick_place_sequence",
            goal={"object_id": object_id, "target_id": target_id},
            workcell=scratch,
            collection_config=task.collection_config,
        )
        candidate = DeterministicPickPlaceExpert(
            open_finger_targets=open_finger_targets,
            closed_finger_targets=closed_finger_targets,
            grasp_config=grasp_config,
            place_config=place_config,
            validate_on_reset=False,
        )
        candidate.reset(view, state)
        if candidate.validation is None or not candidate.validation.valid:
            reason = (
                candidate.validation.reason
                if candidate.validation is not None
                else "pick_place_validation_failed"
            )
            return WaypointValidationResult(False, reason, checked, maximum_disturbance)

        for _ in range(place_config.maximum_total_actions):
            action, phase, done, reason = candidate.step(state)
            if reason is not None:
                return WaypointValidationResult(
                    False, reason, checked, maximum_disturbance
                )
            requested = action.base_position
            workspace_reason = _requested_workspace_reason(scratch, requested)
            if workspace_reason is not None:
                return WaypointValidationResult(
                    False, workspace_reason, checked, maximum_disturbance
                )
            scratch.env.set_mocap_pose(
                str(scratch.config.scene["hand_base_target"]),
                position=requested,
                orientation_quat=action.base_orientation_wxyz,
            )
            scratch.env.set_joint_targets(action.finger_targets)
            if _orientation_hold_phase(phase):
                scratch.preserve_object_orientation(object_id, preserved_orientation)
            state = scratch.step(n_steps=place_config.sim_steps_per_action)
            checked += 1
            maximum_disturbance = max(
                maximum_disturbance,
                _maximum_planar_non_target_disturbance(
                    state, object_id=object_id, initial_positions=initial_positions
                ),
            )
            if maximum_disturbance > place_config.maximum_non_target_disturbance_m:
                return WaypointValidationResult(
                    False, "non_target_disturbance", checked, maximum_disturbance
                )
            if done:
                final_metric = scratch.create_task(
                    "place_held_object", object_id=object_id, target_id=target_id
                ).evaluate(state)
                supported = state.relation_for(object_id).supported_by == str(
                    scratch.config.scene["table_body"]
                )
                if not final_metric.qualifies or not supported:
                    return WaypointValidationResult(
                        False,
                        "pick_place_terminal_metric_not_satisfied",
                        checked,
                        maximum_disturbance,
                    )
                return WaypointValidationResult(
                    True, None, checked, maximum_disturbance
                )
        return WaypointValidationResult(
            False, "pick_place_timeout", checked, maximum_disturbance
        )
    finally:
        scratch.close()


def validate_grasp_lift_trajectory_on_copy(
    *,
    task: object,
    initial_world_state: WorldState,
    waypoints: Sequence[np.ndarray],
    lift_target: np.ndarray,
    initial_orientation_wxyz: np.ndarray,
    target_orientation_wxyz: np.ndarray,
    open_finger_targets: Mapping[str, float],
    closed_finger_targets: Mapping[str, float],
    template: GraspFamilyTemplate,
    config: DeterministicGraspLiftConfig,
) -> WaypointValidationResult:
    """Validate approach, acquisition, physical lift, and stable hold on copied state."""

    from dexvision.sim.workcell import Workcell

    live = task.workcell
    scratch = Workcell(live.config.config_path)
    scratch.env.model.geom_condim[:] = live.env.model.geom_condim
    scratch.env.model.geom_friction[:] = live.env.model.geom_friction
    checked = 0
    maximum_disturbance = 0.0
    try:
        scratch.reset(seed=int(live._seed))
        scratch.env._mujoco.mj_copyData(
            scratch.env.data, scratch.env.model, live.env.data
        )
        scratch.env._mujoco.mj_forward(scratch.env.model, scratch.env.data)
        metric_task = scratch.create_task("pick_object", **dict(task.goal))
        object_id = str(task.goal["object_id"])
        initial_relation = initial_world_state.relation_for(object_id)
        if initial_relation.supported_by != str(scratch.config.scene["table_body"]):
            return WaypointValidationResult(
                False, "object_not_supported", checked, maximum_disturbance
            )
        initial_object_z = initial_world_state.require_entity(object_id).position[2]
        initial_object_orientation = _normalize_quaternion(
            np.asarray(
                initial_world_state.require_entity(object_id).orientation_wxyz,
                dtype=np.float64,
            )
        )
        initial_positions = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in initial_world_state.entities
        }
        workspace = live.config.requirements["workcell"]["safe_workspace"]
        margin = float(workspace.get("margin_m", 0.0))
        minimum = np.asarray(workspace["min"], dtype=np.float64) + margin
        maximum = np.asarray(workspace["max"], dtype=np.float64) - margin
        current = np.asarray(
            initial_world_state.robot.end_effector_position, dtype=np.float64
        )

        def apply(
            candidate: np.ndarray,
            orientation: np.ndarray,
            targets: Mapping[str, float],
            *,
            hold_orientation: bool = False,
        ) -> tuple[str | None, WorldState]:
            nonlocal checked, maximum_disturbance
            if np.any(candidate < minimum) or np.any(candidate > maximum):
                return "workspace_violation", scratch.get_world_state()
            scratch.env.set_mocap_pose(
                str(scratch.config.scene["hand_base_target"]),
                position=candidate,
                orientation_quat=orientation,
            )
            scratch.env.set_joint_targets(targets)
            if hold_orientation:
                scratch.preserve_object_orientation(
                    object_id, initial_object_orientation
                )
            state = scratch.step(n_steps=config.sim_steps_per_action)
            checked += 1
            if _has_joint_limit_violation(
                scratch, tolerance=config.joint_limit_tolerance_rad
            ):
                return "joint_limit_violation", state
            object_state = state.require_entity(object_id)
            lift_height = float(object_state.position[2]) - initial_object_z
            reason = _unsafe_grasp_contact_reason(
                state,
                object_id=object_id,
                allow_table_contact=lift_height < 0.040,
            )
            if reason is not None:
                return reason, state
            for entity in state.entities:
                if entity.object_id == object_id:
                    continue
                initial = initial_positions.get(entity.object_id)
                if initial is not None:
                    maximum_disturbance = max(
                        maximum_disturbance,
                        float(
                            np.linalg.norm(
                                np.asarray(entity.position[:2]) - initial[:2]
                            )
                        ),
                    )
            if maximum_disturbance > config.maximum_non_target_disturbance_m:
                return "non_target_disturbance", state
            if state.relation_for(object_id).held_by == "rh_palm":
                orientation_error = _object_orientation_error(
                    initial_object_orientation,
                    np.asarray(
                        state.require_entity(object_id).orientation_wxyz,
                        dtype=np.float64,
                    ),
                    symmetry=template.orientation_symmetry,
                )
                if orientation_error > config.maximum_object_orientation_deviation_rad:
                    return "object_orientation_deviation", state
            return None, state

        def preserve_orientation(
            state: WorldState, prior_requested: np.ndarray
        ) -> np.ndarray:
            if (
                not template.orientation_feedback_enabled
                or state.relation_for(object_id).held_by != "rh_palm"
            ):
                return prior_requested
            return _orientation_preserving_hand_target(
                reference_object_orientation=initial_object_orientation,
                observed_object_orientation=np.asarray(
                    state.require_entity(object_id).orientation_wxyz,
                    dtype=np.float64,
                ),
                observed_hand_orientation=np.asarray(
                    state.robot.base_orientation_wxyz, dtype=np.float64
                ),
                prior_requested_hand_orientation=prior_requested,
                maximum_step_rad=config.orientation_correction_step_rad,
                symmetry=template.orientation_symmetry,
            )

        scratch.env.set_joint_targets(open_finger_targets)
        orientation = initial_orientation_wxyz.copy()
        for index, waypoint in enumerate(waypoints):
            destination = np.asarray(waypoint, dtype=np.float64)
            step_size = (
                config.descent_step_m
                if index == len(waypoints) - 1
                else config.transit_step_m
            )
            while np.linalg.norm(destination - current) > config.waypoint_tolerance_m:
                current = _bounded_step(current, destination, step_size)
                reason, _ = apply(current, orientation, open_finger_targets)
                if reason is not None:
                    return WaypointValidationResult(
                        False, reason, checked, maximum_disturbance
                    )
            if index == 0:
                while not np.allclose(
                    orientation,
                    target_orientation_wxyz,
                    atol=config.waypoint_tolerance_m,
                    rtol=0.0,
                ):
                    orientation = _bounded_quaternion_step(
                        orientation,
                        target_orientation_wxyz,
                        config.orientation_step_rad,
                    )
                    reason, _ = apply(current, orientation, open_finger_targets)
                    if reason is not None:
                        return WaypointValidationResult(
                            False, reason, checked, maximum_disturbance
                        )

        synergy = 0.0
        state = scratch.get_world_state()
        while synergy < template.grasp_synergy:
            orientation = preserve_orientation(state, orientation)
            synergy = min(template.grasp_synergy, synergy + config.synergy_step)
            targets = _interpolate_finger_targets(
                open_finger_targets, closed_finger_targets, synergy
            )
            reason, state = apply(current, orientation, targets)
            if reason is not None:
                return WaypointValidationResult(
                    False, reason, checked, maximum_disturbance
                )
        for _ in range(config.maximum_closed_acquisition_actions):
            if (
                state.relation_for(object_id).held_by == "rh_palm"
                or _target_hand_contact_body_count(state, object_id=object_id) >= 1
            ):
                break
            orientation = preserve_orientation(state, orientation)
            reason, state = apply(current, orientation, targets)
            if reason is not None:
                return WaypointValidationResult(
                    False, reason, checked, maximum_disturbance
                )
        if (
            state.relation_for(object_id).held_by != "rh_palm"
            and _target_hand_contact_body_count(state, object_id=object_id) < 1
        ):
            return WaypointValidationResult(
                False, "failed_acquisition", checked, maximum_disturbance
            )

        retention_gap = 0
        while np.linalg.norm(lift_target - current) > config.waypoint_tolerance_m:
            orientation = preserve_orientation(state, orientation)
            current = _bounded_step(current, lift_target, config.lift_step_m)
            reason, state = apply(
                current, orientation, targets, hold_orientation=True
            )
            if reason is not None:
                return WaypointValidationResult(
                    False, reason, checked, maximum_disturbance
                )
            retained = (
                state.relation_for(object_id).held_by == "rh_palm"
                or _target_hand_contact_body_count(state, object_id=object_id) >= 1
            )
            retention_gap = 0 if retained else retention_gap + 1
            if retention_gap > config.maximum_retention_gap_steps:
                return WaypointValidationResult(
                    False, "slip_drop", checked, maximum_disturbance
                )

        stable_dwell = 0
        metric_success = False
        for _ in range(config.required_hold_steps + 30):
            orientation = preserve_orientation(state, orientation)
            reason, state = apply(
                current, orientation, targets, hold_orientation=True
            )
            if reason is not None:
                return WaypointValidationResult(
                    False, reason, checked, maximum_disturbance
                )
            metric = metric_task.evaluate(state)
            relation = state.relation_for(object_id)
            speed = float(
                np.linalg.norm(
                    np.asarray(state.require_entity(object_id).linear_velocity)
                )
            )
            orientation_error = _object_orientation_error(
                initial_object_orientation,
                np.asarray(
                    state.require_entity(object_id).orientation_wxyz,
                    dtype=np.float64,
                ),
                symmetry=template.orientation_symmetry,
            )
            stable = (
                metric.qualifies
                and relation.supported_by is None
                and speed <= config.maximum_hold_speed_m_s
                and orientation_error
                <= config.maximum_terminal_orientation_error_rad
            )
            stable_dwell = stable_dwell + 1 if stable else 0
            metric_success = metric_success or metric.success
            if stable_dwell >= config.required_hold_steps and metric_success:
                return WaypointValidationResult(
                    True, None, checked, maximum_disturbance
                )
        return WaypointValidationResult(
            False, "lift_hold_metric_not_satisfied", checked, maximum_disturbance
        )
    finally:
        scratch.close()


def validate_reach_waypoints_on_copy(
    *,
    task: object,
    initial_world_state: WorldState,
    waypoints: Sequence[np.ndarray],
    orientation_wxyz: np.ndarray,
    finger_targets: Mapping[str, float],
    config: SafeWaypointReachConfig,
) -> WaypointValidationResult:
    """Validate reach segments in a separate workcell containing copied mjData."""

    from dexvision.sim.workcell import Workcell

    live_workcell = task.workcell
    scratch = Workcell(live_workcell.config.config_path)
    checked = 0
    maximum_disturbance = 0.0
    try:
        scratch.reset(seed=int(live_workcell._seed))
        scratch.env._mujoco.mj_copyData(
            scratch.env.data,
            scratch.env.model,
            live_workcell.env.data,
        )
        scratch.env._mujoco.mj_forward(scratch.env.model, scratch.env.data)
        current = np.asarray(
            initial_world_state.robot.end_effector_position, dtype=np.float64
        )
        workspace = live_workcell.config.requirements["workcell"]["safe_workspace"]
        margin = float(workspace.get("margin_m", 0.0))
        minimum = np.asarray(workspace["min"], dtype=np.float64) + margin
        maximum = np.asarray(workspace["max"], dtype=np.float64) - margin
        initial_positions = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in initial_world_state.entities
        }
        goal_entity = str(task.goal["entity_id"])
        scratch.env.set_joint_targets(finger_targets)

        for waypoint in waypoints:
            destination = np.asarray(waypoint, dtype=np.float64)
            while (
                float(np.linalg.norm(destination - current))
                > config.waypoint_tolerance_m
            ):
                current = _bounded_step(
                    current, destination, config.max_position_step_m
                )
                if np.any(current < minimum) or np.any(current > maximum):
                    return WaypointValidationResult(
                        False, "workspace_violation", checked, maximum_disturbance
                    )
                scratch.env.set_mocap_pose(
                    str(scratch.config.scene["hand_base_target"]),
                    position=current,
                    orientation_quat=orientation_wxyz,
                )
                scratch.step(n_steps=config.sim_steps_per_action)
                checked += 1
                if _has_joint_limit_violation(
                    scratch, tolerance=config.joint_limit_tolerance_rad
                ):
                    return WaypointValidationResult(
                        False, "joint_limit_violation", checked, maximum_disturbance
                    )
                state = scratch.get_world_state()
                contact_reason = _unsafe_hand_contact_reason(
                    state,
                    table_body=str(scratch.config.scene["table_body"]),
                    fixture_ids=tuple(scratch.config.fixture_ids),
                )
                if contact_reason is not None:
                    return WaypointValidationResult(
                        False, contact_reason, checked, maximum_disturbance
                    )
                for entity in state.entities:
                    if entity.object_id == goal_entity:
                        continue
                    initial = initial_positions.get(entity.object_id)
                    if initial is None:
                        continue
                    maximum_disturbance = max(
                        maximum_disturbance,
                        float(
                            np.linalg.norm(
                                np.asarray(entity.position, dtype=np.float64) - initial
                            )
                        ),
                    )
                if maximum_disturbance > config.maximum_non_target_disturbance_m:
                    return WaypointValidationResult(
                        False,
                        "non_target_disturbance",
                        checked,
                        maximum_disturbance,
                    )
        return WaypointValidationResult(True, None, checked, maximum_disturbance)
    finally:
        scratch.close()


def validate_button_trajectory_on_copy(
    *,
    task: object,
    initial_world_state: WorldState,
    waypoints: Sequence[np.ndarray],
    press_position: np.ndarray,
    precontact_position: np.ndarray,
    orientation_wxyz: np.ndarray,
    finger_targets: Mapping[str, float],
    config: DeterministicButtonPressConfig,
) -> WaypointValidationResult:
    """Qualify the full press-and-release sequence in copied MuJoCo state."""

    from dexvision.sim.workcell import Workcell

    live_workcell = task.workcell
    scratch = Workcell(live_workcell.config.config_path)
    checked = 0
    maximum_disturbance = 0.0
    try:
        scratch.reset(seed=int(live_workcell._seed))
        scratch.env._mujoco.mj_copyData(
            scratch.env.data,
            scratch.env.model,
            live_workcell.env.data,
        )
        scratch.env._mujoco.mj_forward(scratch.env.model, scratch.env.data)
        metric_task = scratch.create_task("press_button", **dict(task.goal))
        current = np.asarray(
            initial_world_state.robot.end_effector_position, dtype=np.float64
        )
        workspace = live_workcell.config.requirements["workcell"]["safe_workspace"]
        margin = float(workspace.get("margin_m", 0.0))
        minimum = np.asarray(workspace["min"], dtype=np.float64) + margin
        maximum = np.asarray(workspace["max"], dtype=np.float64) - margin
        initial_positions = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in initial_world_state.entities
        }
        button_id = str(task.goal["button_id"])
        scratch.env.set_joint_targets(finger_targets)

        def apply(candidate: np.ndarray) -> tuple[str | None, object]:
            nonlocal checked, maximum_disturbance
            if np.any(candidate < minimum) or np.any(candidate > maximum):
                return "workspace_violation", scratch.get_world_state()
            scratch.env.set_mocap_pose(
                str(scratch.config.scene["hand_base_target"]),
                position=candidate,
                orientation_quat=orientation_wxyz,
            )
            scratch.step(n_steps=config.sim_steps_per_action)
            checked += 1
            if _has_joint_limit_violation(
                scratch,
                tolerance=config.joint_limit_tolerance_rad,
                ignored_joint_names=("start_button_joint",),
            ):
                return "joint_limit_violation", scratch.get_world_state()
            state = scratch.get_world_state()
            reason = _unsafe_button_contact_reason(state, button_id=button_id)
            if reason is not None:
                return reason, state
            for entity in state.entities:
                if entity.object_id == button_id:
                    continue
                initial = initial_positions.get(entity.object_id)
                if initial is not None:
                    maximum_disturbance = max(
                        maximum_disturbance,
                        float(np.linalg.norm(np.asarray(entity.position) - initial)),
                    )
            if maximum_disturbance > config.maximum_non_target_disturbance_m:
                return "non_target_disturbance", state
            return None, state

        for waypoint in waypoints:
            destination = np.asarray(waypoint, dtype=np.float64)
            while np.linalg.norm(destination - current) > config.waypoint_tolerance_m:
                current = _bounded_step(current, destination, config.transit_step_m)
                reason, _ = apply(current)
                if reason is not None:
                    return WaypointValidationResult(
                        False, reason, checked, maximum_disturbance
                    )

        success = False
        while np.linalg.norm(press_position - current) > config.waypoint_tolerance_m:
            current = _bounded_step(current, press_position, config.press_step_m)
            reason, state = apply(current)
            if reason is not None:
                return WaypointValidationResult(False, reason, checked, maximum_disturbance)
            success = metric_task.evaluate(state).success or success
        for _ in range(config.required_press_dwell_steps + 2):
            reason, state = apply(current)
            if reason is not None:
                return WaypointValidationResult(False, reason, checked, maximum_disturbance)
            success = metric_task.evaluate(state).success or success
        if not success:
            return WaypointValidationResult(
                False, "button_metric_not_satisfied", checked, maximum_disturbance
            )

        while np.linalg.norm(precontact_position - current) > config.waypoint_tolerance_m:
            current = _bounded_step(current, precontact_position, config.retract_step_m)
            reason, _ = apply(current)
            if reason is not None:
                return WaypointValidationResult(False, reason, checked, maximum_disturbance)
        released = False
        for _ in range(config.release_hold_steps + 8):
            reason, state = apply(current)
            if reason is not None:
                return WaypointValidationResult(False, reason, checked, maximum_disturbance)
            fixture = state.require_fixture(button_id)
            in_contact = any(
                button_id in pair and any(name.startswith("rh_") for name in pair)
                for pair in state.contacts
            )
            released = (
                fixture.press_depth_m <= config.release_depth_m and not in_contact
            )
            if released:
                break
        if not released:
            return WaypointValidationResult(
                False, "button_release_failed", checked, maximum_disturbance
            )
        return WaypointValidationResult(True, None, checked, maximum_disturbance)
    finally:
        scratch.close()


def validate_push_trajectory_on_copy(
    *,
    task: object,
    initial_world_state: WorldState,
    waypoints: Sequence[np.ndarray],
    direction_xy: np.ndarray,
    initial_orientation_wxyz: np.ndarray,
    target_orientation_wxyz: np.ndarray,
    finger_targets: Mapping[str, float],
    config: DeterministicPushConfig,
) -> WaypointValidationResult:
    """Validate approach, axial push, metric dwell, and retract on copied state."""

    from dexvision.sim.workcell import Workcell

    live = task.workcell
    scratch = Workcell(live.config.config_path)
    checked = 0
    maximum_disturbance = 0.0
    try:
        scratch.reset(seed=int(live._seed))
        scratch.env._mujoco.mj_copyData(
            scratch.env.data, scratch.env.model, live.env.data
        )
        scratch.env._mujoco.mj_forward(scratch.env.model, scratch.env.data)
        metric_task = scratch.create_task("push_object_to_target", **dict(task.goal))
        current = np.asarray(
            initial_world_state.robot.end_effector_position, dtype=np.float64
        )
        workspace = live.config.requirements["workcell"]["safe_workspace"]
        margin = float(workspace.get("margin_m", 0.0))
        minimum = np.asarray(workspace["min"], dtype=np.float64) + margin
        maximum = np.asarray(workspace["max"], dtype=np.float64) - margin
        initial_positions = {
            entity.object_id: np.asarray(entity.position, dtype=np.float64)
            for entity in initial_world_state.entities
        }
        object_id = str(task.goal["object_id"])
        target_id = str(task.goal["target_zone"])
        scratch.env.set_joint_targets(finger_targets)

        def apply(
            candidate: np.ndarray, orientation: np.ndarray
        ) -> tuple[str | None, WorldState]:
            nonlocal checked, maximum_disturbance
            if np.any(candidate < minimum) or np.any(candidate > maximum):
                return "workspace_violation", scratch.get_world_state()
            scratch.env.set_mocap_pose(
                str(scratch.config.scene["hand_base_target"]),
                position=candidate,
                orientation_quat=orientation,
            )
            state = scratch.step(n_steps=config.sim_steps_per_action)
            checked += 1
            if _has_joint_limit_violation(
                scratch, tolerance=config.joint_limit_tolerance_rad
            ):
                return "joint_limit_violation", state
            reason = _unsafe_push_contact_reason(state, object_id=object_id)
            if reason is not None:
                return reason, state
            for entity in state.entities:
                if entity.object_id == object_id:
                    continue
                initial = initial_positions.get(entity.object_id)
                if initial is not None:
                    maximum_disturbance = max(
                        maximum_disturbance,
                        float(
                            np.linalg.norm(
                                np.asarray(entity.position[:2])
                                - np.asarray(initial[:2])
                            )
                        ),
                    )
            if maximum_disturbance > config.maximum_non_target_disturbance_m:
                return "non_target_disturbance", state
            metric = metric_task.evaluate(state)
            if not metric.values["object_on_board"]:
                return "object_workspace_violation", state
            if (
                _object_upright_tilt_rad(
                    state.require_entity(object_id).orientation_wxyz
                )
                > config.maximum_object_tilt_rad
            ):
                return "object_tipped", state
            return None, state

        orientation = initial_orientation_wxyz.copy()
        for index, waypoint in enumerate(waypoints):
            destination = np.asarray(waypoint, dtype=np.float64)
            step_size = (
                config.descent_step_m
                if index == len(waypoints) - 1
                else config.transit_step_m
            )
            while np.linalg.norm(destination - current) > config.waypoint_tolerance_m:
                current = _bounded_step(current, destination, step_size)
                reason, _ = apply(current, orientation)
                if reason is not None:
                    return WaypointValidationResult(
                        False, reason, checked, maximum_disturbance
                    )
            if index == 0:
                while not np.allclose(
                    orientation,
                    target_orientation_wxyz,
                    atol=config.waypoint_tolerance_m,
                    rtol=0.0,
                ):
                    orientation = _bounded_quaternion_step(
                        orientation,
                        target_orientation_wxyz,
                        config.orientation_step_rad,
                    )
                    reason, _ = apply(current, orientation)
                    if reason is not None:
                        return WaypointValidationResult(
                            False, reason, checked, maximum_disturbance
                        )

        result = metric_task.evaluate(scratch.get_world_state())
        for _ in range(config.maximum_push_actions):
            object_state = scratch.get_world_state().require_entity(object_id)
            target_state = scratch.get_world_state().require_entity(target_id)
            distance = math.dist(object_state.position[:2], target_state.position[:2])
            if distance <= config.target_stop_distance_m:
                break
            current = current.copy()
            current[:2] += direction_xy * config.push_step_m
            reason, state = apply(current, orientation)
            if reason is not None:
                return WaypointValidationResult(False, reason, checked, maximum_disturbance)
            result = metric_task.evaluate(state)
        else:
            return WaypointValidationResult(
                False, "push_timeout", checked, maximum_disturbance
            )

        dwell = 0
        for _ in range(config.required_goal_dwell_steps + 10):
            reason, state = apply(current, orientation)
            if reason is not None:
                return WaypointValidationResult(False, reason, checked, maximum_disturbance)
            result = metric_task.evaluate(state)
            dwell = dwell + 1 if result.qualifies else 0
            if dwell >= config.required_goal_dwell_steps:
                break
        if dwell < config.required_goal_dwell_steps:
            return WaypointValidationResult(
                False, "push_metric_not_satisfied", checked, maximum_disturbance
            )

        retract = current.copy()
        retract[:2] -= direction_xy * config.retract_distance_m
        while np.linalg.norm(retract - current) > config.waypoint_tolerance_m:
            current = _bounded_step(current, retract, config.retract_step_m)
            reason, _ = apply(current, orientation)
            if reason is not None:
                return WaypointValidationResult(False, reason, checked, maximum_disturbance)
        terminal_dwell = 0
        for _ in range(config.required_terminal_dwell_steps + 10):
            reason, state = apply(current, orientation)
            if reason is not None:
                return WaypointValidationResult(
                    False, reason, checked, maximum_disturbance
                )
            object_contact = any(
                object_id in pair and any(name.startswith("rh_") for name in pair)
                for pair in state.contacts
            )
            terminal_dwell = (
                terminal_dwell + 1
                if not object_contact and metric_task.evaluate(state).qualifies
                else 0
            )
            if terminal_dwell >= config.required_terminal_dwell_steps:
                break
        if terminal_dwell < config.required_terminal_dwell_steps:
            return WaypointValidationResult(
                False, "push_terminal_metric_not_satisfied", checked, maximum_disturbance
            )
        return WaypointValidationResult(True, None, checked, maximum_disturbance)
    finally:
        scratch.close()


def _bounded_step(
    current: np.ndarray, destination: np.ndarray, maximum_step: float
) -> np.ndarray:
    delta = np.asarray(destination, dtype=np.float64) - np.asarray(
        current, dtype=np.float64
    )
    distance = float(np.linalg.norm(delta))
    if distance <= maximum_step:
        return np.asarray(destination, dtype=np.float64).copy()
    return np.asarray(current, dtype=np.float64) + delta * (maximum_step / distance)


def _deduplicate_waypoints(
    waypoints: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for waypoint in waypoints:
        candidate = np.asarray(waypoint, dtype=np.float64)
        if not result or not np.array_equal(candidate, result[-1]):
            result.append(candidate.copy())
    return tuple(result)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or not math.isfinite(norm) or norm <= 0.0:
        raise Level4ExpertError("expert orientation must be a finite quaternion.")
    values = values / norm
    return -values if values[0] < 0.0 else values


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    values = _normalize_quaternion(quaternion)
    return np.asarray([values[0], -values[1], -values[2], -values[3]])


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two wxyz quaternions and return a canonical unit quaternion."""

    w1, x1, y1, z1 = _normalize_quaternion(first)
    w2, x2, y2, z2 = _normalize_quaternion(second)
    return _normalize_quaternion(
        np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
    )


def _quaternion_angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(
        float(np.dot(_normalize_quaternion(first), _normalize_quaternion(second)))
    )
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _conditioned_grasp_orientation(
    template: GraspFamilyTemplate, object_orientation: np.ndarray
) -> np.ndarray:
    """Align the hammer-grip frame with the selected object's observed yaw."""

    object_quaternion = _normalize_quaternion(object_orientation)
    w, x, y, z = object_quaternion
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    yaw_adjustment = template.negative_object_yaw_to_wrist_yaw_gain * min(yaw, 0.0)
    adjustment = np.asarray(
        [
            math.cos(yaw_adjustment / 2.0),
            0.0,
            0.0,
            math.sin(yaw_adjustment / 2.0),
        ],
        dtype=np.float64,
    )
    return _quaternion_multiply(
        adjustment,
        np.asarray(template.wrist_orientation_wxyz, dtype=np.float64),
    )


def _quaternion_rotate_vector(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    values = _normalize_quaternion(quaternion)
    xyz = np.asarray(vector, dtype=np.float64)
    pure = np.asarray([0.0, *xyz.tolist()], dtype=np.float64)
    rotated = _quaternion_multiply_raw(
        _quaternion_multiply_raw(values, pure), _quaternion_conjugate(values)
    )
    return rotated[1:]


def _quaternion_multiply_raw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(first, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(second, dtype=np.float64)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _axial_orientation_correction(
    reference: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    local_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    reference_axis = _quaternion_rotate_vector(reference, local_axis)
    observed_axis = _quaternion_rotate_vector(observed, local_axis)
    dot = float(np.clip(np.dot(observed_axis, reference_axis), -1.0, 1.0))
    cross = np.cross(observed_axis, reference_axis)
    if dot <= -1.0 + 1e-9:
        orthogonal = np.cross(observed_axis, np.asarray([1.0, 0.0, 0.0]))
        if np.linalg.norm(orthogonal) <= 1e-9:
            orthogonal = np.cross(observed_axis, np.asarray([0.0, 1.0, 0.0]))
        orthogonal /= np.linalg.norm(orthogonal)
        return np.asarray([0.0, *orthogonal.tolist()], dtype=np.float64)
    return _normalize_quaternion(np.asarray([1.0 + dot, *cross.tolist()]))


def _object_orientation_error(
    reference: np.ndarray, observed: np.ndarray, *, symmetry: str
) -> float:
    if symmetry == "none":
        return _quaternion_angular_distance(reference, observed)
    if symmetry != "axial_z":
        raise Level4ExpertError(f"unsupported orientation symmetry: {symmetry}")
    local_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    reference_axis = _quaternion_rotate_vector(reference, local_axis)
    observed_axis = _quaternion_rotate_vector(observed, local_axis)
    dot = float(np.clip(np.dot(reference_axis, observed_axis), -1.0, 1.0))
    return math.acos(dot)


def _orientation_preserving_hand_target(
    *,
    reference_object_orientation: np.ndarray,
    observed_object_orientation: np.ndarray,
    observed_hand_orientation: np.ndarray,
    prior_requested_hand_orientation: np.ndarray,
    maximum_step_rad: float,
    symmetry: str,
) -> np.ndarray:
    correction = (
        _quaternion_multiply(
            reference_object_orientation,
            _quaternion_conjugate(observed_object_orientation),
        )
        if symmetry == "none"
        else _axial_orientation_correction(
            reference_object_orientation, observed_object_orientation
        )
    )
    target = _quaternion_multiply(correction, observed_hand_orientation)
    return _bounded_quaternion_step(
        prior_requested_hand_orientation,
        target,
        maximum_step_rad,
    )


def _yaw_pitch_quaternion(*, yaw: float, pitch: float) -> np.ndarray:
    """Return canonical ``q_z(yaw) * q_y(pitch)`` in wxyz order."""

    yaw_cos = math.cos(yaw / 2.0)
    yaw_sin = math.sin(yaw / 2.0)
    pitch_cos = math.cos(pitch / 2.0)
    pitch_sin = math.sin(pitch / 2.0)
    return _normalize_quaternion(
        np.asarray(
            [
                yaw_cos * pitch_cos,
                -yaw_sin * pitch_sin,
                yaw_cos * pitch_sin,
                yaw_sin * pitch_cos,
            ]
        )
    )


def _object_upright_tilt_rad(orientation_wxyz: Sequence[float]) -> float:
    """Return the unsigned angle between an object's local and world up axes."""

    quaternion = _normalize_quaternion(np.asarray(orientation_wxyz, dtype=np.float64))
    _, x, y, _ = quaternion
    local_up_dot_world_up = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(local_up_dot_world_up, -1.0, 1.0)))


def _bounded_quaternion_step(
    current: np.ndarray, target: np.ndarray, maximum_angle: float
) -> np.ndarray:
    """Move along the shortest quaternion arc by at most ``maximum_angle``."""

    start = _normalize_quaternion(current)
    destination = _normalize_quaternion(target)
    dot = float(np.dot(start, destination))
    if dot < 0.0:
        destination = -destination
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * math.acos(dot)
    if angle <= maximum_angle:
        return _normalize_quaternion(destination)
    fraction = maximum_angle / angle
    if dot > 0.9995:
        return _normalize_quaternion(start + fraction * (destination - start))
    theta = math.acos(dot)
    return _normalize_quaternion(
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * start
        + math.sin(fraction * theta) / math.sin(theta) * destination
    )


def _has_joint_limit_violation(
    workcell: object,
    *,
    tolerance: float,
    ignored_joint_names: Sequence[str] = (),
) -> bool:
    model = workcell.env.model
    data = workcell.env.data
    mujoco = workcell.env._mujoco
    free_joint = int(mujoco.mjtJoint.mjJNT_FREE)
    ball_joint = int(mujoco.mjtJoint.mjJNT_BALL)
    ignored = set(ignored_joint_names)
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name in ignored:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        if int(model.jnt_type[joint_id]) in {free_joint, ball_joint}:
            continue
        qpos = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        if qpos < lower - tolerance or qpos > upper + tolerance:
            return True
    return False


def _requested_workspace_reason(workcell: object, position: np.ndarray) -> str | None:
    workspace = workcell.config.requirements["workcell"]["safe_workspace"]
    margin = float(workspace.get("margin_m", 0.0))
    minimum = np.asarray(workspace["min"], dtype=np.float64) + margin
    maximum = np.asarray(workspace["max"], dtype=np.float64) - margin
    return (
        "workspace_violation"
        if np.any(position < minimum) or np.any(position > maximum)
        else None
    )


def _maximum_planar_non_target_disturbance(
    state: WorldState,
    *,
    object_id: str,
    initial_positions: Mapping[str, np.ndarray],
) -> float:
    return max(
        (
            float(
                np.linalg.norm(
                    np.asarray(entity.position[:2], dtype=np.float64)
                    - initial_positions[entity.object_id][:2]
                )
            )
            for entity in state.entities
            if entity.object_id != object_id
            and entity.object_id in initial_positions
        ),
        default=0.0,
    )


def _unsafe_hand_contact_reason(
    state: WorldState,
    *,
    table_body: str,
    fixture_ids: Sequence[str],
) -> str | None:
    fixtures = set(fixture_ids)
    for left, right in state.contacts:
        hand_contact = left.startswith("rh_") or right.startswith("rh_")
        if not hand_contact:
            continue
        if table_body in {left, right}:
            return "table_contact"
        if left in fixtures or right in fixtures:
            return "fixture_contact"
    return None


def _unsafe_button_contact_reason(
    state: WorldState, *, button_id: str
) -> str | None:
    """Allow hand self-contact and the requested button, rejecting all else."""

    for left, right in state.contacts:
        left_hand = left.startswith("rh_")
        right_hand = right.startswith("rh_")
        if not left_hand and not right_hand:
            continue
        if left_hand and right_hand:
            continue
        other = right if left_hand else left
        if other == button_id:
            continue
        if other == "workcell_table":
            return "table_contact"
        if "fixture" in other or "button" in other:
            return "wrong_fixture_contact"
        return "wrong_contact"
    return None


def _unsafe_push_contact_reason(
    state: WorldState, *, object_id: str
) -> str | None:
    """Allow hand self-contact and contact with only the requested object."""

    for left, right in state.contacts:
        left_hand = left.startswith("rh_")
        right_hand = right.startswith("rh_")
        if not left_hand and not right_hand:
            continue
        if left_hand and right_hand:
            continue
        other = right if left_hand else left
        if other == object_id:
            continue
        if other == "workcell_table":
            return "table_contact"
        if "fixture" in other or "button" in other:
            return "fixture_contact"
        return "wrong_object_contact"
    return None


def _unsafe_grasp_contact_reason(
    state: WorldState,
    *,
    object_id: str,
    allow_table_contact: bool,
) -> str | None:
    """Reject external grasp contacts except target and transient support contact."""

    for left, right in state.contacts:
        left_hand = left.startswith("rh_")
        right_hand = right.startswith("rh_")
        if not left_hand and not right_hand:
            continue
        if left_hand and right_hand:
            continue
        other = right if left_hand else left
        if other == object_id:
            continue
        if other == "workcell_table" and allow_table_contact:
            continue
        if other == "workcell_table":
            return "table_contact_after_lift"
        if other in {entity.object_id for entity in state.entities}:
            return "wrong_object_contact"
        return "wrong_fixture_contact"
    return None


def _target_hand_contact_body_count(
    state: WorldState, *, object_id: str
) -> int:
    """Count distinct hand bodies physically contacting the selected object."""

    bodies: set[str] = set()
    for left, right in state.contacts:
        if left == object_id and right.startswith("rh_"):
            bodies.add(right)
        elif right == object_id and left.startswith("rh_"):
            bodies.add(left)
    return len(bodies)


def _unsafe_place_contact_reason(
    state: WorldState, *, object_id: str
) -> str | None:
    """Allow hand self-contact and the held/placed object, rejecting all else."""

    for left, right in state.contacts:
        left_hand = left.startswith("rh_")
        right_hand = right.startswith("rh_")
        if not left_hand and not right_hand:
            continue
        if left_hand and right_hand:
            continue
        other = right if left_hand else left
        if other == object_id:
            continue
        if other == "workcell_table":
            return "table_contact"
        if other in {entity.object_id for entity in state.entities}:
            return "wrong_object_contact"
        return "wrong_fixture_contact"
    return None


def _interpolate_finger_targets(
    open_targets: Mapping[str, float],
    closed_targets: Mapping[str, float],
    synergy: float,
) -> dict[str, float]:
    """Expand one scalar grasp synergy into the frozen named actuator layout."""

    value = float(np.clip(synergy, 0.0, 1.0))
    return {
        name: float(open_value + value * (closed_targets[name] - open_value))
        for name, open_value in open_targets.items()
    }
