"""Disk logger for Level 2 demonstration episodes."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.dataset_schema import (
    ActionSchema,
    DemoEpisode,
    ObservationFieldLayout,
    ObservationSchema,
    is_level4_episode,
    validate_demo,
)
from dexvision.logging.phase_labels import phases_to_intervals


DEFAULT_ACTION_SCHEMA_VERSION = "level1.13/full-action-v1"
DEFAULT_OBSERVATION_SCHEMA_VERSION = "level2/observation-layout-v2"


class DemoLoggerError(RuntimeError):
    """Raised when a demo episode cannot be recorded or saved."""


@dataclass(frozen=True)
class DemoStepData:
    """Synchronized values for one recorded timestep.

    All arrays are copied when appended. Shapes are:
    features ``[F]``, action ``[A]``, robot_state ``[R]``,
    tracking_quality ``[Q]``, optional landmarks ``[21, 3]`` or another
    landmark-compatible shape, optional object_state ``[O]``, and optional
    task_state ``[S]``.
    """

    features: np.ndarray
    action: np.ndarray
    robot_state: np.ndarray
    tracking_quality: np.ndarray
    timestamp: float
    landmarks: np.ndarray | None = None
    object_state: np.ndarray | None = None
    task_state: np.ndarray | None = None
    requested_action: np.ndarray | None = None
    commanded_action: np.ndarray | None = None
    applied_action: np.ndarray | None = None
    prior_commanded_action: np.ndarray | None = None
    prior_applied_action: np.ndarray | None = None
    safety_mask: np.ndarray | None = None
    safety_reason: Sequence[str] | None = None
    request_source: str | None = None
    online_phase: str | None = None
    audited_phase: str | None = None
    phase_relevance_mask: np.ndarray | None = None
    intervention: bool | None = None
    failure_reason: str | None = None
    action_timestamp: float | None = None
    task_timestamp: float | None = None
    state_timestamp: float | None = None
    rgb_frame: np.ndarray | None = None
    rgb_timestamp: float | None = None


class DemoLogger:
    """Collect and save one Level 2 demo episode directory."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        action_schema: ActionSchema,
        observation_schema: ObservationSchema,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.overwrite = bool(overwrite)
        self._metadata: dict[str, Any] | None = None
        self._steps: list[DemoStepData] = []
        self._closed = False

    def start_episode(self, metadata: Mapping[str, Any]) -> None:
        """Start a new episode with required metadata.

        The logger stores schema versions in metadata before validation so saved
        episodes are self-describing.
        """

        if self._metadata is not None:
            raise DemoLoggerError("episode has already been started.")
        if self._closed:
            raise DemoLoggerError("cannot start an episode after close().")
        if not isinstance(metadata, Mapping):
            raise DemoLoggerError("metadata must be a mapping.")

        copied = dict(metadata)
        copied.setdefault("action_schema_version", self.action_schema.version)
        copied.setdefault("observation_schema_version", self.observation_schema.version)
        copied["action_schema"] = action_schema_to_metadata(self.action_schema)
        copied["observation_schema"] = observation_schema_to_metadata(self.observation_schema)
        self._metadata = copied

    def append(self, step_data: DemoStepData) -> None:
        """Append one synchronized timestep."""

        if self._metadata is None:
            raise DemoLoggerError("start_episode() must be called before append().")
        if self._closed:
            raise DemoLoggerError("cannot append after close().")
        if is_level4_episode(self._metadata):
            step_data = self._complete_level4_step(step_data)

        self._steps.append(
            DemoStepData(
                features=_copy_step_array(step_data.features, name="features", ndim=1),
                action=_copy_step_array(step_data.action, name="action", ndim=1),
                robot_state=_copy_step_array(step_data.robot_state, name="robot_state", ndim=1),
                tracking_quality=_copy_step_array(
                    step_data.tracking_quality,
                    name="tracking_quality",
                    ndim=1,
                ),
                timestamp=_coerce_timestamp(step_data.timestamp),
                landmarks=_copy_optional_step_array(
                    step_data.landmarks,
                    name="landmarks",
                    min_ndim=1,
                ),
                object_state=_copy_optional_step_array(
                    step_data.object_state,
                    name="object_state",
                    ndim=1,
                ),
                task_state=_copy_optional_step_array(
                    step_data.task_state,
                    name="task_state",
                    ndim=1,
                ),
                requested_action=_copy_optional_step_array(
                    step_data.requested_action,
                    name="requested_action",
                    ndim=1,
                ),
                commanded_action=_copy_optional_step_array(
                    step_data.commanded_action,
                    name="commanded_action",
                    ndim=1,
                ),
                applied_action=_copy_optional_step_array(
                    step_data.applied_action,
                    name="applied_action",
                    ndim=1,
                ),
                prior_commanded_action=_copy_optional_step_array(
                    step_data.prior_commanded_action,
                    name="prior_commanded_action",
                    ndim=1,
                ),
                prior_applied_action=_copy_optional_step_array(
                    step_data.prior_applied_action,
                    name="prior_applied_action",
                    ndim=1,
                ),
                safety_mask=_copy_optional_step_array(
                    step_data.safety_mask,
                    name="safety_mask",
                    ndim=1,
                ),
                safety_reason=_copy_optional_strings(
                    step_data.safety_reason,
                    name="safety_reason",
                ),
                request_source=_copy_optional_string(
                    step_data.request_source,
                    name="request_source",
                ),
                online_phase=_copy_optional_string(
                    step_data.online_phase,
                    name="online_phase",
                ),
                audited_phase=_copy_optional_string(
                    step_data.audited_phase,
                    name="audited_phase",
                    allow_empty=True,
                ),
                phase_relevance_mask=_copy_optional_step_array(
                    step_data.phase_relevance_mask,
                    name="phase_relevance_mask",
                    ndim=1,
                ),
                intervention=(
                    bool(step_data.intervention)
                    if step_data.intervention is not None
                    else None
                ),
                failure_reason=_copy_optional_string(
                    step_data.failure_reason,
                    name="failure_reason",
                    allow_empty=True,
                ),
                action_timestamp=_copy_optional_timestamp(step_data.action_timestamp),
                task_timestamp=_copy_optional_timestamp(step_data.task_timestamp),
                state_timestamp=_copy_optional_timestamp(step_data.state_timestamp),
                rgb_frame=_copy_optional_rgb_frame(step_data.rgb_frame),
                rgb_timestamp=_copy_optional_timestamp(step_data.rgb_timestamp),
            )
        )

    def _complete_level4_step(self, step_data: DemoStepData) -> DemoStepData:
        """Fill lossless Level 4 fields when adapting an existing recorder loop."""

        action = _copy_step_array(step_data.action, name="action", ndim=1)
        requested = step_data.requested_action if step_data.requested_action is not None else action
        commanded = step_data.commanded_action if step_data.commanded_action is not None else action
        applied = step_data.applied_action if step_data.applied_action is not None else action
        if self._steps:
            previous = self._steps[-1]
            prior_commanded = (
                step_data.prior_commanded_action
                if step_data.prior_commanded_action is not None
                else previous.commanded_action
            )
            prior_applied = (
                step_data.prior_applied_action
                if step_data.prior_applied_action is not None
                else previous.applied_action
            )
        else:
            prior_commanded = (
                step_data.prior_commanded_action
                if step_data.prior_commanded_action is not None
                else commanded
            )
            prior_applied = (
                step_data.prior_applied_action
                if step_data.prior_applied_action is not None
                else applied
            )
            assert self._metadata is not None
            self._metadata.setdefault(
                "initial_commanded_action", np.asarray(prior_commanded).tolist()
            )
            self._metadata.setdefault(
                "initial_applied_action", np.asarray(prior_applied).tolist()
            )
        assert self._metadata is not None
        phase = step_data.online_phase or self._metadata.get("initial_online_phase")
        if not isinstance(phase, str) or not phase:
            raise DemoLoggerError(
                "Level 4 steps require online_phase or metadata initial_online_phase."
            )
        phase_contract = self._metadata.get("phase_contract")
        if not isinstance(phase_contract, Mapping):
            raise DemoLoggerError("Level 4 metadata phase_contract must be a mapping.")
        masks = phase_contract.get("action_relevance_masks")
        if not isinstance(masks, Mapping) or phase not in masks:
            raise DemoLoggerError(f"Level 4 phase '{phase}' has no action relevance mask.")
        relevance = (
            step_data.phase_relevance_mask
            if step_data.phase_relevance_mask is not None
            else np.asarray(masks[phase], dtype=np.uint8)
        )
        request_source = step_data.request_source
        if request_source is None:
            request_source = {
                "teleoperation": "operator",
                "scripted": "script",
                "policy_rollout": "policy",
                "corrective_intervention": "operator",
            }.get(str(self._metadata.get("source")))
        return replace(
            step_data,
            requested_action=requested,
            commanded_action=commanded,
            applied_action=applied,
            prior_commanded_action=prior_commanded,
            prior_applied_action=prior_applied,
            safety_mask=(
                step_data.safety_mask
                if step_data.safety_mask is not None
                else np.zeros(action.shape, dtype=np.uint8)
            ),
            safety_reason=(
                step_data.safety_reason
                if step_data.safety_reason is not None
                else ("none",) * action.size
            ),
            request_source=request_source,
            online_phase=phase,
            phase_relevance_mask=relevance,
            intervention=(step_data.intervention if step_data.intervention is not None else False),
            failure_reason=(step_data.failure_reason if step_data.failure_reason is not None else ""),
            action_timestamp=(
                step_data.action_timestamp
                if step_data.action_timestamp is not None
                else step_data.timestamp
            ),
            task_timestamp=(
                step_data.task_timestamp
                if step_data.task_timestamp is not None
                else step_data.timestamp
            ),
            state_timestamp=(
                step_data.state_timestamp
                if step_data.state_timestamp is not None
                else step_data.timestamp
            ),
        )

    def close(self, *, success: bool | None = None) -> DemoEpisode:
        """Validate and write the episode directory, then return the episode."""

        if self._metadata is None:
            raise DemoLoggerError("start_episode() must be called before close().")
        if self._closed:
            raise DemoLoggerError("episode has already been closed.")
        if not self._steps:
            raise DemoLoggerError("cannot close an empty demo episode.")

        episode = self._build_episode(success=success)
        validate_demo(
            episode,
            action_schema=self.action_schema,
            observation_schema=self.observation_schema,
        )
        self._write_episode(episode)
        self._closed = True
        return episode

    @property
    def step_count(self) -> int:
        """Number of timesteps appended so far."""

        return len(self._steps)

    def _build_episode(self, *, success: bool | None) -> DemoEpisode:
        features = np.stack([step.features for step in self._steps], axis=0)
        actions = np.stack([step.action for step in self._steps], axis=0)
        robot_states = np.stack([step.robot_state for step in self._steps], axis=0)
        tracking_quality = np.stack([step.tracking_quality for step in self._steps], axis=0)
        timestamps = np.asarray([step.timestamp for step in self._steps], dtype=np.float64)

        metadata = dict(self._metadata or {})
        online_phases = _stack_optional_strings(
            [step.online_phase for step in self._steps],
            name="online_phases",
        )
        if online_phases is not None:
            metadata["phase_intervals"] = [
                interval.to_dict() for interval in phases_to_intervals(online_phases.tolist())
            ]
        return DemoEpisode(
            metadata=metadata,
            landmarks=_stack_optional_steps(
                [step.landmarks for step in self._steps],
                name="landmarks",
            ),
            features=features,
            actions=actions,
            robot_states=robot_states,
            object_states=_stack_optional_steps(
                [step.object_state for step in self._steps],
                name="object_states",
            ),
            task_states=_stack_optional_steps(
                [step.task_state for step in self._steps],
                name="task_states",
            ),
            tracking_quality=tracking_quality,
            timestamps=timestamps,
            success=success,
            requested_actions=_stack_optional_steps(
                [step.requested_action for step in self._steps],
                name="requested_actions",
            ),
            commanded_actions=_stack_optional_steps(
                [step.commanded_action for step in self._steps],
                name="commanded_actions",
            ),
            applied_actions=_stack_optional_steps(
                [step.applied_action for step in self._steps],
                name="applied_actions",
            ),
            prior_commanded_actions=_stack_optional_steps(
                [step.prior_commanded_action for step in self._steps],
                name="prior_commanded_actions",
            ),
            prior_applied_actions=_stack_optional_steps(
                [step.prior_applied_action for step in self._steps],
                name="prior_applied_actions",
            ),
            safety_masks=_stack_optional_steps(
                [step.safety_mask for step in self._steps],
                name="safety_masks",
            ),
            safety_reasons=_stack_optional_string_rows(
                [step.safety_reason for step in self._steps],
                name="safety_reasons",
            ),
            request_sources=_stack_optional_strings(
                [step.request_source for step in self._steps],
                name="request_sources",
            ),
            online_phases=online_phases,
            audited_phases=_stack_optional_strings(
                [step.audited_phase for step in self._steps],
                name="audited_phases",
            ),
            phase_relevance_masks=_stack_optional_steps(
                [step.phase_relevance_mask for step in self._steps],
                name="phase_relevance_masks",
            ),
            intervention_flags=_stack_optional_scalars(
                [step.intervention for step in self._steps],
                name="intervention_flags",
                dtype=np.uint8,
            ),
            failure_reasons=_stack_optional_strings(
                [step.failure_reason for step in self._steps],
                name="failure_reasons",
            ),
            action_timestamps=_stack_optional_scalars(
                [step.action_timestamp for step in self._steps],
                name="action_timestamps",
                dtype=np.float64,
            ),
            task_timestamps=_stack_optional_scalars(
                [step.task_timestamp for step in self._steps],
                name="task_timestamps",
                dtype=np.float64,
            ),
            state_timestamps=_stack_optional_scalars(
                [step.state_timestamp for step in self._steps],
                name="state_timestamps",
                dtype=np.float64,
            ),
            rgb_frames=_stack_optional_steps(
                [step.rgb_frame for step in self._steps],
                name="rgb_frames",
            ),
            rgb_timestamps=_stack_optional_scalars(
                [step.rgb_timestamp for step in self._steps],
                name="rgb_timestamps",
                dtype=np.float64,
            ),
        )

    def _write_episode(self, episode: DemoEpisode) -> None:
        if is_level4_episode(episode.metadata):
            if self.output_dir.exists():
                raise DemoLoggerError(
                    f"append-only Level 4 episode already exists: {self.output_dir}"
                )
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.output_dir.name}.writing-",
                    dir=self.output_dir.parent,
                )
            )
            try:
                _write_episode_files(temporary_dir, episode)
                temporary_dir.replace(self.output_dir)
            except Exception:
                shutil.rmtree(temporary_dir, ignore_errors=True)
                raise
            return
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise DemoLoggerError(f"output path exists but is not a directory: {self.output_dir}")
            if any(self.output_dir.iterdir()):
                if not self.overwrite:
                    raise DemoLoggerError(
                        f"output directory is not empty: {self.output_dir}. "
                        "Use --overwrite to replace it."
                    )
                shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_episode_files(self.output_dir, episode)


def build_level1_action_schema(finger_target_names: Sequence[str]) -> ActionSchema:
    """Return the default full Level 1.13 action schema for a target list."""

    names = tuple(str(name) for name in finger_target_names)
    if not names or any(not name for name in names):
        raise DemoLoggerError("finger_target_names must contain at least one non-empty name.")
    return ActionSchema(
        version=DEFAULT_ACTION_SCHEMA_VERSION,
        base_position_target=(0, 3),
        base_orientation_target=(3, 7),
        finger_actuator_targets=(7, 7 + len(names)),
        representation_notes={
            "base_position_target": "MuJoCo/world-space hand-base target position, shape [3]",
            "base_orientation_target": "MuJoCo wxyz hand-base target quaternion, shape [4]",
            "finger_actuator_targets": "MuJoCo actuator targets ordered by finger_target_names",
            "finger_target_names": names,
        },
    )


def build_level2_observation_schema(
    *,
    robot_qpos_dim: int,
    robot_qvel_dim: int,
    finger_target_dim: int,
    tracking_quality_dim: int,
    robot_qpos_names: Sequence[str] | None = None,
    robot_qvel_names: Sequence[str] | None = None,
    actuator_names: Sequence[str] | None = None,
    finger_joint_qpos_indices: Sequence[int] | None = None,
    finger_joint_qvel_indices: Sequence[int] | None = None,
    finger_joint_names: Sequence[str] | None = None,
    tracking_quality_names: Sequence[str] | None = None,
    object_state_dim: int | None = None,
    task_state_dim: int | None = None,
    target_state_dim: int | None = None,
    success_metric_dim: int | None = None,
) -> ObservationSchema:
    """Return the executable Level 2 dense observation layout."""

    qpos_dim = _positive_dim(robot_qpos_dim, "robot_qpos_dim")
    qvel_dim = _positive_dim(robot_qvel_dim, "robot_qvel_dim")
    control_dim = _positive_dim(finger_target_dim, "finger_target_dim")
    tracking_dim = _positive_dim(tracking_quality_dim, "tracking_quality_dim")
    qpos_names = _ordered_names(robot_qpos_names, qpos_dim, prefix="qpos")
    qvel_names = _ordered_names(robot_qvel_names, qvel_dim, prefix="qvel")
    control_names = _ordered_names(actuator_names, control_dim, prefix="actuator")
    tracking_names = _ordered_names(
        tracking_quality_names,
        tracking_dim,
        prefix="tracking_quality",
    )
    finger_qpos_indices = _selection_indices(
        finger_joint_qpos_indices,
        source_dim=qpos_dim,
        default_dim=control_dim,
        field_name="finger_joint_qpos_indices",
    )
    finger_qvel_indices = _selection_indices(
        finger_joint_qvel_indices,
        source_dim=qvel_dim,
        default_dim=control_dim,
        field_name="finger_joint_qvel_indices",
    )
    if len(finger_qpos_indices) != len(finger_qvel_indices):
        raise DemoLoggerError(
            "finger joint position and velocity selections must have the same length."
        )
    selected_finger_names = _ordered_names(
        finger_joint_names,
        len(finger_qpos_indices),
        prefix="finger_joint",
    )

    qpos_start = 0
    qvel_start = qpos_start + qpos_dim
    control_start = qvel_start + qvel_dim
    base_position_start = control_start + control_dim
    base_orientation_start = base_position_start + 3
    fields = [
        "robot_qpos",
        "robot_qvel",
        "actuator_controls",
        "base_position",
        "base_orientation",
        "finger_joint_positions",
        "finger_joint_velocities",
        "tracking_quality",
    ]
    shapes: dict[str, tuple[int, ...]] = {
        "robot_qpos": (qpos_dim,),
        "robot_qvel": (qvel_dim,),
        "actuator_controls": (control_dim,),
        "base_position": (3,),
        "base_orientation": (4,),
        "finger_joint_positions": (len(finger_qpos_indices),),
        "finger_joint_velocities": (len(finger_qvel_indices),),
        "tracking_quality": (tracking_dim,),
    }
    layouts: dict[str, ObservationFieldLayout] = {
        "robot_qpos": ObservationFieldLayout(
            source_array="robot_states",
            column_range=(qpos_start, qvel_start),
            shape=shapes["robot_qpos"],
            dtype="float64",
            units="MuJoCo generalized-position units: metres, unit quaternion, or radians",
            coordinate_frame="MuJoCo model coordinates; free-joint pose is in world frame",
            normalization="Normalize each named degree of freedom with training-set statistics.",
            names=qpos_names,
        ),
        "robot_qvel": ObservationFieldLayout(
            source_array="robot_states",
            column_range=(qvel_start, control_start),
            shape=shapes["robot_qvel"],
            dtype="float64",
            units="MuJoCo generalized-velocity units: metres/second or radians/second",
            coordinate_frame="MuJoCo model velocity coordinates",
            normalization="Normalize each named degree of freedom with training-set statistics.",
            names=qvel_names,
        ),
        "actuator_controls": ObservationFieldLayout(
            source_array="robot_states",
            column_range=(control_start, base_position_start),
            shape=shapes["actuator_controls"],
            dtype="float64",
            units="MuJoCo actuator control units",
            coordinate_frame="MuJoCo actuator order recorded by actuator names",
            normalization="Scale per actuator using control limits before learning.",
            names=control_names,
        ),
        "base_position": ObservationFieldLayout(
            source_array="robot_states",
            column_range=(base_position_start, base_orientation_start),
            shape=shapes["base_position"],
            dtype="float64",
            units="metres",
            coordinate_frame="MuJoCo world frame",
            normalization="Center on workspace neutral and scale by workspace range.",
            names=("x", "y", "z"),
        ),
        "base_orientation": ObservationFieldLayout(
            source_array="robot_states",
            column_range=(base_orientation_start, base_orientation_start + 4),
            shape=shapes["base_orientation"],
            dtype="float64",
            units="unitless normalized quaternion",
            coordinate_frame="MuJoCo world frame; wxyz quaternion",
            normalization="Convert to a continuous rotation representation before learning.",
            names=("qw", "qx", "qy", "qz"),
        ),
        "finger_joint_positions": ObservationFieldLayout(
            source_array="robot_states",
            column_indices=tuple(qpos_start + index for index in finger_qpos_indices),
            shape=shapes["finger_joint_positions"],
            dtype="float64",
            units="radians",
            coordinate_frame="Named MuJoCo hand-joint coordinates",
            normalization="Scale each named joint by its configured joint range.",
            names=selected_finger_names,
        ),
        "finger_joint_velocities": ObservationFieldLayout(
            source_array="robot_states",
            column_indices=tuple(qvel_start + index for index in finger_qvel_indices),
            shape=shapes["finger_joint_velocities"],
            dtype="float64",
            units="radians/second",
            coordinate_frame="Named MuJoCo hand-joint velocity coordinates",
            normalization="Normalize each named joint velocity with training-set statistics.",
            names=selected_finger_names,
        ),
        "tracking_quality": ObservationFieldLayout(
            source_array="tracking_quality",
            column_range=(0, tracking_dim),
            shape=shapes["tracking_quality"],
            dtype="float64",
            units="unitless flags, categorical code, and confidence values",
            coordinate_frame="camera/tracker status",
            normalization="Keep flags/codes categorical and confidence values in [0, 1].",
            names=tracking_names,
        ),
    }
    optional_fields: list[str] = []
    if object_state_dim is not None:
        object_dim = _positive_dim(object_state_dim, "object_state_dim")
        fields.append("object_state")
        shapes["object_state"] = (object_dim,)
        optional_fields.append("object_state")
        layouts["object_state"] = _optional_layout(
            source_array="object_states",
            shape=(object_dim,),
            name_prefix="object_state",
            units="task-defined SI units",
            coordinate_frame="task-defined MuJoCo world/object frames",
            normalization="Normalize named object fields with training-set statistics.",
            absence_rule="object_states.npy is omitted when the task has no objects.",
        )
    if task_state_dim is not None:
        task_dim = _positive_dim(task_state_dim, "task_state_dim")
        fields.append("task_state")
        shapes["task_state"] = (task_dim,)
        optional_fields.append("task_state")
        layouts["task_state"] = _optional_layout(
            source_array="task_states",
            shape=(task_dim,),
            name_prefix="task_state",
            units="task-defined SI units and unitless flags",
            coordinate_frame="task-defined world/target frames",
            normalization="Normalize continuous task fields; keep flags categorical.",
            absence_rule="task_states.npy is omitted when the task has no task state.",
        )
    if target_state_dim is not None:
        target_dim = _positive_dim(target_state_dim, "target_state_dim")
        if task_state_dim is None or target_dim > task_state_dim:
            raise DemoLoggerError(
                "target_state_dim must not exceed the containing task_state_dim."
            )
        fields.append("target_state")
        shapes["target_state"] = (target_dim,)
        optional_fields.append("target_state")
        layouts["target_state"] = _optional_layout(
            source_array="task_states",
            shape=(target_dim,),
            name_prefix="target_state",
            units="task-defined SI units",
            coordinate_frame="task-defined MuJoCo world/target frame",
            normalization="Normalize named target fields with training-set statistics.",
            absence_rule="task_states.npy is omitted when the task has no target state.",
        )
    if success_metric_dim is not None:
        metric_dim = _positive_dim(success_metric_dim, "success_metric_dim")
        if task_state_dim is None or metric_dim > task_state_dim:
            raise DemoLoggerError(
                "success_metric_dim must not exceed the containing task_state_dim."
            )
        fields.append("success_metric_inputs")
        shapes["success_metric_inputs"] = (metric_dim,)
        optional_fields.append("success_metric_inputs")
        layouts["success_metric_inputs"] = _optional_layout(
            source_array="task_states",
            shape=(metric_dim,),
            name_prefix="success_metric",
            units="task-defined SI units and unitless flags",
            coordinate_frame="task-defined metric frame",
            normalization="Preserve raw relabeling inputs; normalize only policy copies.",
            absence_rule=(
                "task_states.npy is omitted when the task has no recomputable success metric."
            ),
        )

    return ObservationSchema(
        version=DEFAULT_OBSERVATION_SCHEMA_VERSION,
        fields=tuple(fields),
        shapes=shapes,
        optional_fields=tuple(optional_fields),
        layouts=layouts,
    )


def load_logged_demo(output_dir: str | Path) -> DemoEpisode:
    """Load arrays saved by :class:`DemoLogger` without replaying them."""

    path = Path(output_dir)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise DemoLoggerError(f"metadata.json does not exist in demo directory: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return DemoEpisode(
        metadata=metadata,
        landmarks=_load_optional_npy(path / "landmarks.npy"),
        features=_load_required_npy(path / "features.npy"),
        actions=_load_required_npy(path / "actions.npy"),
        robot_states=_load_required_npy(path / "robot_states.npy"),
        object_states=_load_optional_npy(path / "object_states.npy"),
        task_states=_load_optional_npy(path / "task_states.npy"),
        tracking_quality=_load_required_npy(path / "tracking_quality.npy"),
        timestamps=_load_required_npy(path / "timestamps.npy"),
        success=metadata.get("success"),
        requested_actions=_load_optional_npy(path / "requested_actions.npy"),
        commanded_actions=_load_optional_npy(path / "commanded_actions.npy"),
        applied_actions=_load_optional_npy(path / "applied_actions.npy"),
        prior_commanded_actions=_load_optional_npy(path / "prior_commanded_actions.npy"),
        prior_applied_actions=_load_optional_npy(path / "prior_applied_actions.npy"),
        safety_masks=_load_optional_npy(path / "safety_masks.npy"),
        safety_reasons=_load_optional_npy(path / "safety_reasons.npy"),
        request_sources=_load_optional_npy(path / "request_sources.npy"),
        online_phases=_load_optional_npy(path / "online_phases.npy"),
        audited_phases=_load_optional_npy(path / "audited_phases.npy"),
        phase_relevance_masks=_load_optional_npy(path / "phase_relevance_masks.npy"),
        intervention_flags=_load_optional_npy(path / "intervention_flags.npy"),
        failure_reasons=_load_optional_npy(path / "failure_reasons.npy"),
        action_timestamps=_load_optional_npy(path / "action_timestamps.npy"),
        task_timestamps=_load_optional_npy(path / "task_timestamps.npy"),
        state_timestamps=_load_optional_npy(path / "state_timestamps.npy"),
        rgb_frames=_load_optional_npy(path / "rgb_frames.npy"),
        rgb_timestamps=_load_optional_npy(path / "rgb_timestamps.npy"),
    )


def action_schema_to_metadata(action_schema: ActionSchema) -> dict[str, Any]:
    """Convert an action schema to JSON-friendly metadata."""

    return {
        "version": action_schema.version,
        "base_position_target": _range_to_tuple(action_schema.base_position_target),
        "base_orientation_target": _range_to_tuple(action_schema.base_orientation_target),
        "finger_actuator_targets": _range_to_tuple(action_schema.finger_actuator_targets),
        "representation_notes": dict(action_schema.representation_notes),
    }


def observation_schema_to_metadata(observation_schema: ObservationSchema) -> dict[str, Any]:
    """Convert an observation schema to JSON-friendly metadata."""

    payload = asdict(observation_schema)
    payload["shapes"] = {
        name: tuple(shape) for name, shape in observation_schema.shapes.items()
    }
    return payload


def action_schema_from_metadata(metadata: Mapping[str, Any]) -> ActionSchema:
    """Reconstruct a saved action schema without importing runtime control code."""

    payload = metadata.get("action_schema")
    if not isinstance(payload, Mapping):
        raise DemoLoggerError("metadata action_schema must be a mapping.")
    try:
        return ActionSchema(
            version=str(payload["version"]),
            base_position_target=_metadata_range(payload["base_position_target"]),
            base_orientation_target=_metadata_range(payload["base_orientation_target"]),
            finger_actuator_targets=_metadata_range(payload["finger_actuator_targets"]),
            representation_notes=dict(payload.get("representation_notes", {})),
        )
    except KeyError as exc:
        raise DemoLoggerError(f"action_schema is missing field: {exc.args[0]}") from exc


def observation_schema_from_metadata(metadata: Mapping[str, Any]) -> ObservationSchema:
    """Reconstruct a saved executable observation schema."""

    payload = metadata.get("observation_schema")
    if not isinstance(payload, Mapping):
        raise DemoLoggerError("metadata observation_schema must be a mapping.")
    raw_layouts = payload.get("layouts", {})
    if not isinstance(raw_layouts, Mapping):
        raise DemoLoggerError("observation_schema layouts must be a mapping.")
    layouts: dict[str, ObservationFieldLayout] = {}
    for name, raw in raw_layouts.items():
        if not isinstance(raw, Mapping):
            raise DemoLoggerError(f"observation layout '{name}' must be a mapping.")
        layouts[str(name)] = ObservationFieldLayout(
            source_array=str(raw["source_array"]),
            shape=tuple(int(item) for item in raw["shape"]),
            dtype=str(raw["dtype"]),
            units=str(raw["units"]),
            coordinate_frame=str(raw["coordinate_frame"]),
            normalization=str(raw["normalization"]),
            column_range=(
                _metadata_range(raw["column_range"])
                if raw.get("column_range") is not None
                else None
            ),
            column_indices=tuple(int(item) for item in raw.get("column_indices", ())),
            names=tuple(str(item) for item in raw.get("names", ())),
            optional=bool(raw.get("optional", False)),
            absence_rule=raw.get("absence_rule"),
            mask_field=raw.get("mask_field"),
        )
    try:
        shapes = {
            str(name): tuple(int(item) for item in shape)
            for name, shape in payload["shapes"].items()
        }
        return ObservationSchema(
            version=str(payload["version"]),
            fields=tuple(str(item) for item in payload["fields"]),
            shapes=shapes,
            optional_fields=tuple(str(item) for item in payload.get("optional_fields", ())),
            layouts=layouts,
            compatibility_notes=tuple(
                str(item) for item in payload.get("compatibility_notes", ())
            ),
        )
    except KeyError as exc:
        raise DemoLoggerError(f"observation_schema is missing field: {exc.args[0]}") from exc


def _copy_step_array(
    value: np.ndarray,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise DemoLoggerError(f"{name} must be {ndim}D, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise DemoLoggerError(f"{name} must contain only finite values.")
    return array.copy()


def _copy_optional_step_array(
    value: np.ndarray | None,
    *,
    name: str,
    ndim: int | None = None,
    min_ndim: int | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise DemoLoggerError(f"{name} must be {ndim}D, got shape {array.shape}.")
    if min_ndim is not None and array.ndim < min_ndim:
        raise DemoLoggerError(f"{name} must be at least {min_ndim}D, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise DemoLoggerError(f"{name} must contain only finite values.")
    return array.copy()


def _copy_optional_strings(
    value: Sequence[str] | None,
    *,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise DemoLoggerError(f"{name} must be a sequence of strings, not one string.")
    values = tuple(value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise DemoLoggerError(f"{name} must contain non-empty strings.")
    return np.asarray(values, dtype=np.str_)


def _copy_optional_string(
    value: str | None,
    *,
    name: str,
    allow_empty: bool = False,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or (not allow_empty and not value):
        raise DemoLoggerError(f"{name} must be a string{' or empty' if allow_empty else ''}.")
    return value


def _copy_optional_timestamp(value: float | None) -> float | None:
    return None if value is None else _coerce_timestamp(value)


def _copy_optional_rgb_frame(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise DemoLoggerError(f"rgb_frame must have shape [H, W, 3], got {frame.shape}.")
    if frame.dtype != np.uint8:
        raise DemoLoggerError("rgb_frame must use uint8 RGB pixels.")
    return frame.copy()


def _coerce_timestamp(timestamp: float) -> float:
    value = float(timestamp)
    if not np.isfinite(value):
        raise DemoLoggerError("timestamp must be finite.")
    return value


def _stack_optional_steps(
    values: Sequence[np.ndarray | None],
    *,
    name: str,
) -> np.ndarray | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise DemoLoggerError(f"{name} must be provided for every step or no steps.")
    try:
        return np.stack([value for value in values if value is not None], axis=0)
    except ValueError as exc:
        raise DemoLoggerError(f"{name} shapes must match across timesteps.") from exc


def _stack_optional_strings(
    values: Sequence[str | None],
    *,
    name: str,
) -> np.ndarray | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise DemoLoggerError(f"{name} must be provided for every step or no steps.")
    return np.asarray([value for value in values if value is not None], dtype=np.str_)


def _stack_optional_string_rows(
    values: Sequence[Sequence[str] | np.ndarray | None],
    *,
    name: str,
) -> np.ndarray | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise DemoLoggerError(f"{name} must be provided for every step or no steps.")
    try:
        return np.stack([np.asarray(value, dtype=np.str_) for value in values], axis=0)
    except ValueError as exc:
        raise DemoLoggerError(f"{name} widths must match across timesteps.") from exc


def _stack_optional_scalars(
    values: Sequence[float | bool | None],
    *,
    name: str,
    dtype: np.dtype[Any] | type,
) -> np.ndarray | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise DemoLoggerError(f"{name} must be provided for every step or no steps.")
    return np.asarray([value for value in values if value is not None], dtype=dtype)


def _write_episode_files(output_dir: Path, episode: DemoEpisode) -> None:
    metadata = dict(episode.metadata)
    metadata["success"] = episode.success
    metadata["num_steps"] = int(episode.timestamps.shape[0])
    (output_dir / "metadata.json").write_text(
        json.dumps(_to_jsonable(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arrays = {
        "features": episode.features,
        "actions": episode.actions,
        "robot_states": episode.robot_states,
        "tracking_quality": episode.tracking_quality,
        "timestamps": episode.timestamps,
        "landmarks": episode.landmarks,
        "object_states": episode.object_states,
        "task_states": episode.task_states,
        "requested_actions": episode.requested_actions,
        "commanded_actions": episode.commanded_actions,
        "applied_actions": episode.applied_actions,
        "prior_commanded_actions": episode.prior_commanded_actions,
        "prior_applied_actions": episode.prior_applied_actions,
        "safety_masks": episode.safety_masks,
        "safety_reasons": episode.safety_reasons,
        "request_sources": episode.request_sources,
        "online_phases": episode.online_phases,
        "audited_phases": episode.audited_phases,
        "phase_relevance_masks": episode.phase_relevance_masks,
        "intervention_flags": episode.intervention_flags,
        "failure_reasons": episode.failure_reasons,
        "action_timestamps": episode.action_timestamps,
        "task_timestamps": episode.task_timestamps,
        "state_timestamps": episode.state_timestamps,
        "rgb_frames": episode.rgb_frames,
        "rgb_timestamps": episode.rgb_timestamps,
    }
    for name, value in arrays.items():
        if value is not None:
            np.save(output_dir / f"{name}.npy", value, allow_pickle=False)


def _positive_dim(value: int, field_name: str) -> int:
    dimension = int(value)
    if dimension <= 0:
        raise DemoLoggerError(f"{field_name} must be positive.")
    return dimension


def _ordered_names(
    values: Sequence[str] | None,
    expected_dim: int,
    *,
    prefix: str,
) -> tuple[str, ...]:
    if values is None:
        return tuple(f"{prefix}[{index}]" for index in range(expected_dim))
    names = tuple(str(value) for value in values)
    if len(names) != expected_dim:
        raise DemoLoggerError(
            f"{prefix} names must contain {expected_dim} entries, got {len(names)}."
        )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise DemoLoggerError(f"{prefix} names must be unique non-empty strings.")
    return names


def _selection_indices(
    values: Sequence[int] | None,
    *,
    source_dim: int,
    default_dim: int,
    field_name: str,
) -> tuple[int, ...]:
    if values is None:
        if default_dim > source_dim:
            raise DemoLoggerError(
                f"{field_name} is required because finger_target_dim {default_dim} "
                f"exceeds source dimension {source_dim}."
            )
        return tuple(range(source_dim - default_dim, source_dim))
    indices = tuple(values)
    if not indices:
        raise DemoLoggerError(f"{field_name} must contain at least one index.")
    if any(not isinstance(index, int) or index < 0 or index >= source_dim for index in indices):
        raise DemoLoggerError(f"{field_name} must contain indices in [0, {source_dim}).")
    if len(set(indices)) != len(indices):
        raise DemoLoggerError(f"{field_name} must not contain duplicate indices.")
    return indices


def _optional_layout(
    *,
    source_array: str,
    shape: tuple[int, ...],
    name_prefix: str,
    units: str,
    coordinate_frame: str,
    normalization: str,
    absence_rule: str,
) -> ObservationFieldLayout:
    width = int(np.prod(shape))
    return ObservationFieldLayout(
        source_array=source_array,
        column_range=(0, width),
        shape=shape,
        dtype="float64",
        units=units,
        coordinate_frame=coordinate_frame,
        normalization=normalization,
        names=tuple(f"{name_prefix}[{index}]" for index in range(width)),
        optional=True,
        absence_rule=absence_rule,
    )


def _range_to_tuple(index_range: object) -> tuple[int, int]:
    if isinstance(index_range, slice):
        if index_range.start is None or index_range.stop is None:
            raise DemoLoggerError("schema slices must have start and stop.")
        return (int(index_range.start), int(index_range.stop))
    if isinstance(index_range, tuple) and len(index_range) == 2:
        return (int(index_range[0]), int(index_range[1]))
    raise DemoLoggerError("schema ranges must be slices or (start, stop) tuples.")


def _metadata_range(value: Any) -> tuple[int, int]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise DemoLoggerError("saved schema ranges must contain [start, stop].")
    start, stop = value
    if not isinstance(start, int) or not isinstance(stop, int):
        raise DemoLoggerError("saved schema range bounds must be integers.")
    return start, stop


def _load_required_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise DemoLoggerError(f"required demo array is missing: {path}")
    return np.load(path)


def _load_optional_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    raise DemoLoggerError(f"metadata contains a non-JSON-serializable value: {type(value)!r}")
