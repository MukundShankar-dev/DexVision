"""Schema and validation utilities for recorded demonstration episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class DemoSchemaError(ValueError):
    """Raised when a demo episode does not match the dataset schema."""


IndexRange = slice | tuple[int, int]

REQUIRED_METADATA_FIELDS = (
    "skill_name",
    "task_name",
    "task_id",
    "episode_id",
    "action_schema_version",
    "observation_schema_version",
    "robot_model",
    "retargeter_config",
    "control_rate_hz",
    "teleop_config",
    "task_config",
)

REQUIRED_OBSERVATION_FIELDS = (
    "robot_qpos",
    "robot_qvel",
    "base_position",
    "base_orientation",
    "finger_joint_positions",
    "finger_joint_velocities",
    "tracking_quality",
)
FREE_SPACE_GESTURE_LABELS = (
    "open_palm",
    "fist",
    "point",
    "pinch",
    "peace_sign",
    "wave",
)


@dataclass(frozen=True)
class ActionSchema:
    """Column layout for one full Level 1.13 teleoperation action vector."""

    version: str
    base_position_target: IndexRange
    base_orientation_target: IndexRange
    finger_actuator_targets: IndexRange
    representation_notes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, action_dim: int | None = None) -> None:
        """Validate that this schema can reconstruct the full action command."""

        if not self.version:
            raise DemoSchemaError("action schema version is required.")

        base_position = _coerce_range(
            self.base_position_target,
            field_name="base_position_target",
        )
        base_orientation = _coerce_range(
            self.base_orientation_target,
            field_name="base_orientation_target",
        )
        finger_targets = _coerce_range(
            self.finger_actuator_targets,
            field_name="finger_actuator_targets",
        )

        _require_range_length(base_position, 3, field_name="base_position_target")
        _require_range_length(base_orientation, 4, field_name="base_orientation_target")
        if _range_length(finger_targets) <= 0:
            raise DemoSchemaError("finger_actuator_targets must contain at least one column.")

        ranges = (
            ("base_position_target", base_position),
            ("base_orientation_target", base_orientation),
            ("finger_actuator_targets", finger_targets),
        )
        _validate_non_overlapping_ranges(ranges)

        if action_dim is not None and action_dim < self.action_dim:
            raise DemoSchemaError(
                "actions width does not contain the full Level 1.13 action schema: "
                f"expected at least {self.action_dim} columns, got {action_dim}."
            )

    @property
    def action_dim(self) -> int:
        """Minimum number of columns needed to contain this action schema."""

        ranges = (
            _coerce_range(self.base_position_target, field_name="base_position_target"),
            _coerce_range(self.base_orientation_target, field_name="base_orientation_target"),
            _coerce_range(self.finger_actuator_targets, field_name="finger_actuator_targets"),
        )
        return max(stop for _start, stop in ranges)

    def split(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Return base, orientation, and finger slices from an action row or batch."""

        action_array = np.asarray(action)
        if action_array.ndim not in {1, 2}:
            raise DemoSchemaError("action must be a 1D row or 2D action array.")
        if action_array.shape[-1] < self.action_dim:
            raise DemoSchemaError(
                f"action width must be at least {self.action_dim}, got {action_array.shape[-1]}."
            )

        self.validate(action_dim=action_array.shape[-1])
        return {
            "base_position_target": action_array[
                ..., _as_slice(self.base_position_target, field_name="base_position_target")
            ],
            "base_orientation_target": action_array[
                ..., _as_slice(self.base_orientation_target, field_name="base_orientation_target")
            ],
            "finger_actuator_targets": action_array[
                ..., _as_slice(self.finger_actuator_targets, field_name="finger_actuator_targets")
            ],
        }


@dataclass(frozen=True)
class ObservationSchema:
    """Named observation/state fields saved alongside a demonstration."""

    version: str
    fields: tuple[str, ...]
    shapes: Mapping[str, tuple[int, ...]]
    optional_fields: tuple[str, ...] = ()

    def validate(self, *, task_config: Mapping[str, Any] | None = None) -> None:
        """Validate required observation fields and declared shapes."""

        if not self.version:
            raise DemoSchemaError("observation schema version is required.")
        if not self.fields:
            raise DemoSchemaError("observation schema must declare at least one field.")

        field_set = set(self.fields)
        missing_base_fields = [field for field in REQUIRED_OBSERVATION_FIELDS if field not in field_set]
        if missing_base_fields:
            raise DemoSchemaError(
                "observation schema is missing required fields: "
                + ", ".join(missing_base_fields)
            )

        missing_shapes = [field for field in self.fields if field not in self.shapes]
        if missing_shapes:
            raise DemoSchemaError(
                "observation schema is missing shapes for fields: " + ", ".join(missing_shapes)
            )
        for field_name, shape in self.shapes.items():
            if field_name not in field_set:
                raise DemoSchemaError(
                    f"observation schema shape declared for unknown field '{field_name}'."
                )
            if not shape or any(dimension <= 0 for dimension in shape):
                raise DemoSchemaError(
                    f"observation schema field '{field_name}' must have a positive shape."
                )

        task_config = task_config or {}
        task_fields = _string_tuple(task_config.get("required_observation_fields", ()))
        missing_task_fields = [field for field in task_fields if field not in field_set]
        if missing_task_fields:
            raise DemoSchemaError(
                "observation schema is missing task-required fields: "
                + ", ".join(missing_task_fields)
            )


@dataclass
class DemoEpisode:
    """One recorded demonstration episode stored as synchronized arrays.

    The first dimension of every array is time ``T``. Shape examples:
    landmarks ``[T, 21, 3]`` or ``None``; features ``[T, F]``; actions
    ``[T, A]``; robot_states ``[T, R]``; tracking_quality ``[T, Q]``;
    timestamps ``[T]``.
    """

    metadata: dict
    landmarks: np.ndarray | None
    features: np.ndarray
    actions: np.ndarray
    robot_states: np.ndarray
    object_states: np.ndarray | None
    task_states: np.ndarray | None
    tracking_quality: np.ndarray
    timestamps: np.ndarray
    success: bool | None


def validate_demo(
    episode: DemoEpisode,
    *,
    action_schema: ActionSchema,
    observation_schema: ObservationSchema,
) -> None:
    """Validate a synthetic or recorded demo episode.

    Validation is intentionally independent of camera, GUI, MuJoCo viewer, and
    learning code so it can run in automated tests.
    """

    if not isinstance(episode.metadata, dict):
        raise DemoSchemaError("metadata must be a dict.")
    _validate_metadata(episode.metadata)

    features = _required_array(episode.features, name="features", ndim=2)
    actions = _required_array(episode.actions, name="actions", ndim=2)
    robot_states = _required_array(episode.robot_states, name="robot_states", ndim=2)
    tracking_quality = _required_array(
        episode.tracking_quality,
        name="tracking_quality",
        ndim=2,
    )
    timestamps = _required_array(episode.timestamps, name="timestamps", ndim=1)

    time_steps = int(timestamps.shape[0])
    if time_steps == 0:
        raise DemoSchemaError("demo episode must contain at least one timestep.")
    _require_time_dim(features, time_steps, name="features")
    _require_time_dim(actions, time_steps, name="actions")
    _require_time_dim(robot_states, time_steps, name="robot_states")
    _require_time_dim(tracking_quality, time_steps, name="tracking_quality")

    if episode.landmarks is not None:
        landmarks = _required_array(episode.landmarks, name="landmarks", min_ndim=2)
        _require_time_dim(landmarks, time_steps, name="landmarks")
    if episode.object_states is not None:
        object_states = _required_array(episode.object_states, name="object_states", ndim=2)
        _require_time_dim(object_states, time_steps, name="object_states")
    if episode.task_states is not None:
        task_states = _required_array(episode.task_states, name="task_states", ndim=2)
        _require_time_dim(task_states, time_steps, name="task_states")

    if np.any(np.diff(timestamps) < 0.0):
        raise DemoSchemaError("timestamps must be monotonic nondecreasing.")

    _validate_tracking_quality(tracking_quality)

    action_schema.validate(action_dim=actions.shape[1])
    split_action = action_schema.split(actions)
    for field_name, values in split_action.items():
        if field_name == "base_orientation_target":
            _validate_quaternions(values)

    task_config = _mapping_field(episode.metadata, "task_config")
    observation_schema.validate(task_config=task_config)
    _validate_schema_versions(episode.metadata, action_schema, observation_schema)
    _validate_task_state_requirements(episode, observation_schema)


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise DemoSchemaError("metadata is missing required fields: " + ", ".join(missing))

    for metadata_field in ("skill_name", "task_name", "task_id", "episode_id"):
        value = metadata[metadata_field]
        if not isinstance(value, str) or not value:
            raise DemoSchemaError(
                f"metadata field '{metadata_field}' must be a non-empty string."
            )

    control_rate = metadata["control_rate_hz"]
    if not isinstance(control_rate, (int, float)) or control_rate <= 0.0:
        raise DemoSchemaError("metadata field 'control_rate_hz' must be positive.")

    for metadata_field in ("teleop_config", "task_config"):
        if not isinstance(metadata[metadata_field], Mapping):
            raise DemoSchemaError(
                f"metadata field '{metadata_field}' must be a mapping."
            )

    if "gesture_label" in metadata:
        label = metadata["gesture_label"]
        if label is not None and label not in FREE_SPACE_GESTURE_LABELS:
            allowed = ", ".join(FREE_SPACE_GESTURE_LABELS)
            raise DemoSchemaError(f"metadata field 'gesture_label' must be one of: {allowed}.")


def _validate_schema_versions(
    metadata: Mapping[str, Any],
    action_schema: ActionSchema,
    observation_schema: ObservationSchema,
) -> None:
    if metadata["action_schema_version"] != action_schema.version:
        raise DemoSchemaError(
            "metadata action_schema_version does not match the supplied action schema."
        )
    if metadata["observation_schema_version"] != observation_schema.version:
        raise DemoSchemaError(
            "metadata observation_schema_version does not match the supplied observation schema."
        )


def _validate_task_state_requirements(
    episode: DemoEpisode,
    observation_schema: ObservationSchema,
) -> None:
    task_config = _mapping_field(episode.metadata, "task_config")
    required_objects = _string_tuple(task_config.get("required_objects", ()))
    if required_objects and episode.object_states is None:
        raise DemoSchemaError(
            "object_states are required because task_config declares required_objects."
        )

    if bool(task_config.get("requires_task_state", False)) and episode.task_states is None:
        raise DemoSchemaError("task_states are required by task_config.requires_task_state.")

    success_metric_required = bool(
        task_config.get("requires_success_metric_inputs", False)
        or task_config.get("success_metric_fields")
    )
    if success_metric_required:
        if episode.task_states is None:
            raise DemoSchemaError("task_states are required for success metric relabeling.")
        if "success_metric_inputs" not in observation_schema.fields:
            raise DemoSchemaError(
                "observation schema must include success_metric_inputs when task "
                "success metrics are required."
            )


def _required_array(
    value: np.ndarray,
    *,
    name: str,
    ndim: int | None = None,
    min_ndim: int | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise DemoSchemaError(f"{name} must be {ndim}D, got shape {array.shape}.")
    if min_ndim is not None and array.ndim < min_ndim:
        raise DemoSchemaError(f"{name} must be at least {min_ndim}D, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise DemoSchemaError(f"{name} must be numeric.")
    if np.any(~np.isfinite(array)):
        raise DemoSchemaError(f"{name} must not contain NaN or infinity values.")
    return array


def _require_time_dim(array: np.ndarray, time_steps: int, *, name: str) -> None:
    if array.shape[0] != time_steps:
        raise DemoSchemaError(
            f"{name} time dimension {array.shape[0]} does not match timestamps length "
            f"{time_steps}."
        )


def _validate_tracking_quality(tracking_quality: np.ndarray) -> None:
    if tracking_quality.shape[1] < 4:
        raise DemoSchemaError(
            "tracking_quality must include at least detected flag, handedness code, "
            "hand tracking confidence, and feature/control confidence columns."
        )
    detected = tracking_quality[:, 0]
    if np.any((detected < 0.0) | (detected > 1.0)):
        raise DemoSchemaError("tracking_quality detected flag must be in [0, 1].")
    confidences = tracking_quality[:, 2:4]
    if np.any((confidences < 0.0) | (confidences > 1.0)):
        raise DemoSchemaError("tracking_quality confidence columns must be in [0, 1].")


def _validate_quaternions(quaternions: np.ndarray) -> None:
    norms = np.linalg.norm(quaternions, axis=-1)
    if np.any(norms <= 0.0):
        raise DemoSchemaError("base_orientation_target quaternions must be nonzero.")
    if np.any(~np.isclose(norms, 1.0, atol=1e-3)):
        raise DemoSchemaError("base_orientation_target quaternions must be normalized.")


def _coerce_range(index_range: IndexRange, *, field_name: str) -> tuple[int, int]:
    if isinstance(index_range, slice):
        if index_range.step not in (None, 1):
            raise DemoSchemaError(f"{field_name} slice step must be 1 or None.")
        if index_range.start is None or index_range.stop is None:
            raise DemoSchemaError(f"{field_name} slice must have start and stop.")
        start = index_range.start
        stop = index_range.stop
    else:
        if len(index_range) != 2:
            raise DemoSchemaError(f"{field_name} tuple must be (start, stop).")
        start, stop = index_range
    if not isinstance(start, int) or not isinstance(stop, int):
        raise DemoSchemaError(f"{field_name} start and stop must be integers.")
    if start < 0 or stop <= start:
        raise DemoSchemaError(f"{field_name} must satisfy 0 <= start < stop.")
    return start, stop


def _as_slice(index_range: IndexRange, *, field_name: str) -> slice:
    start, stop = _coerce_range(index_range, field_name=field_name)
    return slice(start, stop)


def _range_length(index_range: tuple[int, int]) -> int:
    start, stop = index_range
    return stop - start


def _require_range_length(index_range: tuple[int, int], length: int, *, field_name: str) -> None:
    actual = _range_length(index_range)
    if actual != length:
        raise DemoSchemaError(f"{field_name} must have length {length}, got {actual}.")


def _validate_non_overlapping_ranges(
    named_ranges: Sequence[tuple[str, tuple[int, int]]],
) -> None:
    seen: dict[int, str] = {}
    for field_name, (start, stop) in named_ranges:
        for column in range(start, stop):
            if column in seen:
                raise DemoSchemaError(
                    f"{field_name} overlaps column {column} with {seen[column]}."
                )
            seen[column] = field_name


def _mapping_field(metadata: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = metadata[field_name]
    if not isinstance(value, Mapping):
        raise DemoSchemaError(f"metadata field '{field_name}' must be a mapping.")
    return value


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise DemoSchemaError("expected a sequence of strings.")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise DemoSchemaError("expected a sequence of non-empty strings.")
    return result
