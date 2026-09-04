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
EXECUTABLE_OBSERVATION_FIELDS = REQUIRED_OBSERVATION_FIELDS + ("actuator_controls",)
OBSERVATION_SOURCE_ARRAYS = (
    "robot_states",
    "tracking_quality",
    "object_states",
    "task_states",
)
FREE_SPACE_GESTURE_LABELS = (
    "open_palm",
    "fist",
    "point",
    "pinch",
    "peace_sign",
    "wave",
)

LEVEL4_EPISODE_SCHEMA_VERSION = "level4/episode-v1"
LEVEL4_ACTION_SCHEMA_VERSION = "level4/request-command-apply-v1"
LEVEL4_REQUIRED_METADATA_FIELDS = (
    "recording_session_id",
    "operator_id",
    "source",
    "typed_goal",
    "object_instance_ids",
    "goal_condition_id",
    "reset_state",
    "random_seed",
    "camera_or_render_config",
    "code_version",
    "config_version",
    "schema_versions",
    "phase_contract",
    "action_contract",
)
LEVEL4_SOURCES = (
    "teleoperation",
    "scripted",
    "policy_rollout",
    "corrective_intervention",
)
LEVEL4_REQUEST_SOURCES = ("operator", "script", "policy")


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
class ObservationFieldLayout:
    """Executable mapping from one dense saved array to one observation field."""

    source_array: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    coordinate_frame: str
    normalization: str
    column_range: tuple[int, int] | None = None
    column_indices: tuple[int, ...] = ()
    names: tuple[str, ...] = ()
    optional: bool = False
    absence_rule: str | None = None
    mask_field: str | None = None

    def validate(self, *, field_name: str) -> None:
        """Validate one dense-array selection and its semantic metadata."""

        if self.source_array not in OBSERVATION_SOURCE_ARRAYS:
            allowed = ", ".join(OBSERVATION_SOURCE_ARRAYS)
            raise DemoSchemaError(
                f"observation field '{field_name}' has unknown source_array "
                f"'{self.source_array}'; expected one of: {allowed}."
            )
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise DemoSchemaError(
                f"observation field '{field_name}' must have a positive shape."
            )
        if not self.dtype:
            raise DemoSchemaError(f"observation field '{field_name}' must declare dtype.")
        for attribute_name, value in (
            ("units", self.units),
            ("coordinate_frame", self.coordinate_frame),
            ("normalization", self.normalization),
        ):
            if not isinstance(value, str) or not value:
                raise DemoSchemaError(
                    f"observation field '{field_name}' must declare {attribute_name}."
                )

        has_range = self.column_range is not None
        has_indices = bool(self.column_indices)
        if has_range == has_indices:
            raise DemoSchemaError(
                f"observation field '{field_name}' must declare exactly one of "
                "column_range or column_indices."
            )
        if self.column_range is not None:
            selection_width = _range_length(
                _coerce_range(self.column_range, field_name=f"{field_name}.column_range")
            )
        else:
            if any(not isinstance(index, int) or index < 0 for index in self.column_indices):
                raise DemoSchemaError(
                    f"observation field '{field_name}' column_indices must be "
                    "non-negative integers."
                )
            if len(set(self.column_indices)) != len(self.column_indices):
                raise DemoSchemaError(
                    f"observation field '{field_name}' column_indices must be unique."
                )
            selection_width = len(self.column_indices)

        expected_width = int(np.prod(self.shape))
        if selection_width != expected_width:
            raise DemoSchemaError(
                f"observation field '{field_name}' selects {selection_width} columns "
                f"but declares shape {self.shape} ({expected_width} values)."
            )
        if self.names and len(self.names) != selection_width:
            raise DemoSchemaError(
                f"observation field '{field_name}' has {len(self.names)} names for "
                f"{selection_width} selected columns."
            )
        if self.names and (
            any(not isinstance(name, str) or not name for name in self.names)
            or len(set(self.names)) != len(self.names)
        ):
            raise DemoSchemaError(
                f"observation field '{field_name}' names must be unique non-empty strings."
            )
        if self.optional and not self.absence_rule and not self.mask_field:
            raise DemoSchemaError(
                f"optional observation field '{field_name}' must declare an "
                "absence_rule or mask_field."
            )
        if not self.optional and (self.absence_rule or self.mask_field):
            raise DemoSchemaError(
                f"required observation field '{field_name}' cannot declare optional "
                "absence or mask behavior."
            )

    @property
    def maximum_column(self) -> int:
        """Exclusive upper column bound used by this field."""

        if self.column_range is not None:
            return _coerce_range(self.column_range, field_name="column_range")[1]
        return max(self.column_indices) + 1

    def extract(self, source: np.ndarray) -> np.ndarray:
        """Extract this field from a validated two-dimensional source array."""

        if self.column_range is not None:
            start, stop = _coerce_range(self.column_range, field_name="column_range")
            values = source[:, start:stop]
        else:
            values = source[:, self.column_indices]
        return values.reshape((source.shape[0], *self.shape))


@dataclass(frozen=True)
class ObservationSchema:
    """Named observation/state fields saved alongside a demonstration."""

    version: str
    fields: tuple[str, ...]
    shapes: Mapping[str, tuple[int, ...]]
    optional_fields: tuple[str, ...] = ()
    layouts: Mapping[str, ObservationFieldLayout] = field(default_factory=dict)
    compatibility_notes: tuple[str, ...] = ()

    def validate(self, *, task_config: Mapping[str, Any] | None = None) -> None:
        """Validate required observation fields and declared shapes."""

        if not self.version:
            raise DemoSchemaError("observation schema version is required.")
        if not self.fields:
            raise DemoSchemaError("observation schema must declare at least one field.")

        field_set = set(self.fields)
        required_fields = (
            EXECUTABLE_OBSERVATION_FIELDS if self.layouts else REQUIRED_OBSERVATION_FIELDS
        )
        missing_base_fields = [field for field in required_fields if field not in field_set]
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

        unknown_optional = set(self.optional_fields) - field_set
        if unknown_optional:
            raise DemoSchemaError(
                "observation schema optional_fields contains unknown fields: "
                + ", ".join(sorted(unknown_optional))
            )

        if self.layouts:
            missing_layouts = field_set - set(self.layouts)
            unknown_layouts = set(self.layouts) - field_set
            if missing_layouts:
                raise DemoSchemaError(
                    "observation schema is missing executable layouts for fields: "
                    + ", ".join(sorted(missing_layouts))
                )
            if unknown_layouts:
                raise DemoSchemaError(
                    "observation schema has layouts for unknown fields: "
                    + ", ".join(sorted(unknown_layouts))
                )
            for field_name in self.fields:
                layout = self.layouts[field_name]
                layout.validate(field_name=field_name)
                if tuple(layout.shape) != tuple(self.shapes[field_name]):
                    raise DemoSchemaError(
                        f"observation field '{field_name}' layout shape {layout.shape} "
                        f"does not match declared shape {self.shapes[field_name]}."
                    )
                declared_optional = field_name in self.optional_fields
                if layout.optional != declared_optional:
                    raise DemoSchemaError(
                        f"observation field '{field_name}' optional flag does not match "
                        "optional_fields."
                    )
                if field_name in {
                    "robot_qpos",
                    "robot_qvel",
                    "actuator_controls",
                    "finger_joint_positions",
                    "finger_joint_velocities",
                    "tracking_quality",
                } and not layout.names:
                    raise DemoSchemaError(
                        f"observation field '{field_name}' must preserve ordered names."
                    )

        task_config = task_config or {}
        task_fields = _string_tuple(task_config.get("required_observation_fields", ()))
        missing_task_fields = [field for field in task_fields if field not in field_set]
        if missing_task_fields:
            raise DemoSchemaError(
                "observation schema is missing task-required fields: "
                + ", ".join(missing_task_fields)
            )

    @property
    def executable(self) -> bool:
        """Whether every declared field has an executable dense-array layout."""

        return bool(self.layouts)

    def extract(
        self,
        source_arrays: Mapping[str, np.ndarray | None],
        *,
        time_steps: int | None = None,
    ) -> dict[str, np.ndarray | None]:
        """Validate dense widths and reconstruct every declared observation field."""

        if not self.layouts:
            raise DemoSchemaError(
                f"observation schema '{self.version}' is a legacy shape-only schema; "
                "it remains replay-compatible but does not support field extraction."
            )
        self.validate()

        fields_by_source: dict[str, list[str]] = {}
        for field_name, layout in self.layouts.items():
            fields_by_source.setdefault(layout.source_array, []).append(field_name)

        extracted: dict[str, np.ndarray | None] = {}
        observed_time_steps = time_steps
        for source_name, field_names in fields_by_source.items():
            source = source_arrays.get(source_name)
            if source is None:
                required = [
                    field_name
                    for field_name in field_names
                    if not self.layouts[field_name].optional
                ]
                if required:
                    raise DemoSchemaError(
                        f"{source_name} is required by observation fields: "
                        + ", ".join(required)
                    )
                for field_name in field_names:
                    extracted[field_name] = None
                continue

            array = _required_array(source, name=source_name, ndim=2)
            if observed_time_steps is None:
                observed_time_steps = int(array.shape[0])
            _require_time_dim(array, observed_time_steps, name=source_name)
            expected_width = max(
                self.layouts[field_name].maximum_column for field_name in field_names
            )
            if array.shape[1] != expected_width:
                raise DemoSchemaError(
                    f"{source_name} width does not match its executable observation "
                    f"layout: expected {expected_width}, got {array.shape[1]}."
                )

            for field_name in field_names:
                layout = self.layouts[field_name]
                if np.dtype(array.dtype).name != np.dtype(layout.dtype).name:
                    raise DemoSchemaError(
                        f"{source_name} dtype does not match observation field "
                        f"'{field_name}': expected {layout.dtype}, got {array.dtype}."
                    )
                extracted[field_name] = layout.extract(array)

        return {field_name: extracted[field_name] for field_name in self.fields}


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
    requested_actions: np.ndarray | None = None
    commanded_actions: np.ndarray | None = None
    applied_actions: np.ndarray | None = None
    prior_commanded_actions: np.ndarray | None = None
    prior_applied_actions: np.ndarray | None = None
    safety_masks: np.ndarray | None = None
    safety_reasons: np.ndarray | None = None
    request_sources: np.ndarray | None = None
    online_phases: np.ndarray | None = None
    audited_phases: np.ndarray | None = None
    phase_relevance_masks: np.ndarray | None = None
    intervention_flags: np.ndarray | None = None
    failure_reasons: np.ndarray | None = None
    action_timestamps: np.ndarray | None = None
    task_timestamps: np.ndarray | None = None
    state_timestamps: np.ndarray | None = None
    rgb_frames: np.ndarray | None = None
    rgb_timestamps: np.ndarray | None = None


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
    if observation_schema.executable:
        extract_observations(episode, observation_schema=observation_schema)
    _validate_schema_versions(episode.metadata, action_schema, observation_schema)
    _validate_task_state_requirements(episode, observation_schema)
    if episode.metadata.get("episode_schema_version") == LEVEL4_EPISODE_SCHEMA_VERSION:
        _validate_level4_episode(episode, action_schema=action_schema)


def is_level4_episode(metadata: Mapping[str, Any]) -> bool:
    """Return whether metadata explicitly declares the Level 4 episode schema."""

    return metadata.get("episode_schema_version") == LEVEL4_EPISODE_SCHEMA_VERSION


def _validate_level4_episode(
    episode: DemoEpisode,
    *,
    action_schema: ActionSchema,
) -> None:
    metadata = episode.metadata
    missing = [field for field in LEVEL4_REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise DemoSchemaError(
            "Level 4 metadata is missing required fields: " + ", ".join(missing)
        )

    for field_name in ("recording_session_id", "operator_id", "goal_condition_id"):
        value = metadata[field_name]
        if not isinstance(value, str) or not value.strip():
            raise DemoSchemaError(
                f"Level 4 metadata field '{field_name}' must be a non-empty string."
            )
    source = metadata["source"]
    if source not in LEVEL4_SOURCES:
        raise DemoSchemaError(
            "Level 4 metadata field 'source' must be one of: "
            + ", ".join(LEVEL4_SOURCES)
            + "."
        )
    for field_name in ("typed_goal", "reset_state", "schema_versions"):
        if not isinstance(metadata[field_name], Mapping):
            raise DemoSchemaError(
                f"Level 4 metadata field '{field_name}' must be a mapping."
            )
    object_ids = metadata["object_instance_ids"]
    if isinstance(object_ids, str) or not isinstance(object_ids, Sequence):
        raise DemoSchemaError(
            "Level 4 metadata field 'object_instance_ids' must be a sequence of ids."
        )
    if any(not isinstance(item, str) or not item for item in object_ids):
        raise DemoSchemaError("object_instance_ids must contain non-empty strings.")
    if not isinstance(metadata["random_seed"], int):
        raise DemoSchemaError("Level 4 metadata field 'random_seed' must be an integer.")
    for field_name in ("code_version", "config_version"):
        if not isinstance(metadata[field_name], str) or not metadata[field_name]:
            raise DemoSchemaError(
                f"Level 4 metadata field '{field_name}' must be a non-empty string."
            )

    schema_versions = metadata["schema_versions"]
    required_versions = ("episode", "observation", "action", "world_state", "phase", "safety")
    missing_versions = [name for name in required_versions if not schema_versions.get(name)]
    if missing_versions:
        raise DemoSchemaError(
            "Level 4 schema_versions is missing: " + ", ".join(missing_versions)
        )
    if schema_versions["episode"] != LEVEL4_EPISODE_SCHEMA_VERSION:
        raise DemoSchemaError("Level 4 episode schema version does not match metadata.")
    if schema_versions["action"] != LEVEL4_ACTION_SCHEMA_VERSION:
        raise DemoSchemaError(
            f"Level 4 action schema must be '{LEVEL4_ACTION_SCHEMA_VERSION}'."
        )
    phase_contract = metadata["phase_contract"]
    action_contract = metadata["action_contract"]
    if not isinstance(phase_contract, Mapping):
        raise DemoSchemaError("Level 4 metadata field 'phase_contract' must be a mapping.")
    if not isinstance(action_contract, Mapping):
        raise DemoSchemaError("Level 4 metadata field 'action_contract' must be a mapping.")
    if source == "corrective_intervention":
        for field_name in ("intervention_interval", "failure_reason", "source_episode_id"):
            if metadata.get(field_name) in (None, ""):
                raise DemoSchemaError(
                    f"corrective_intervention metadata requires '{field_name}'."
                )
        trigger_source = metadata.get("trigger_source")
        if trigger_source not in {"teleoperation", "scripted", "policy_rollout"}:
            raise DemoSchemaError(
                "corrective_intervention trigger_source must be teleoperation, scripted, "
                "or policy_rollout."
            )
        if trigger_source == "policy_rollout" and not metadata.get(
            "source_policy_checkpoint"
        ):
            raise DemoSchemaError(
                "policy-triggered correction requires source_policy_checkpoint."
            )
    if source == "policy_rollout" and not metadata.get("source_policy_checkpoint"):
        raise DemoSchemaError("policy_rollout metadata requires source_policy_checkpoint.")

    time_steps = int(episode.timestamps.shape[0])
    action_dim = action_schema.action_dim
    action_arrays = {
        "requested_actions": episode.requested_actions,
        "commanded_actions": episode.commanded_actions,
        "applied_actions": episode.applied_actions,
        "prior_commanded_actions": episode.prior_commanded_actions,
        "prior_applied_actions": episode.prior_applied_actions,
    }
    checked_actions: dict[str, np.ndarray] = {}
    for name, value in action_arrays.items():
        if value is None:
            raise DemoSchemaError(f"{name} is required for a Level 4 episode.")
        array = _required_array(value, name=name, ndim=2)
        _require_time_dim(array, time_steps, name=name)
        if array.shape[1] != action_dim:
            raise DemoSchemaError(
                f"{name} width must be {action_dim}, got {array.shape[1]}."
            )
        checked_actions[name] = array

    applied = checked_actions["applied_actions"]
    if not np.array_equal(episode.actions, applied):
        raise DemoSchemaError("actions must exactly mirror applied_actions for Level 4.")
    _validate_prior_actions(checked_actions, metadata)
    _validate_level4_action_safety(
        commanded=checked_actions["commanded_actions"],
        applied=applied,
        safety_masks=episode.safety_masks,
        safety_reasons=episode.safety_reasons,
        time_steps=time_steps,
        action_dim=action_dim,
        allowed_reasons=action_contract.get("safety_reason_codes"),
    )
    _validate_level4_strings(
        episode.request_sources,
        name="request_sources",
        time_steps=time_steps,
        allowed=LEVEL4_REQUEST_SOURCES,
        allow_empty=False,
    )
    _validate_level4_strings(
        episode.online_phases,
        name="online_phases",
        time_steps=time_steps,
        allowed=None,
        allow_empty=False,
    )
    if episode.audited_phases is not None:
        _validate_level4_strings(
            episode.audited_phases,
            name="audited_phases",
            time_steps=time_steps,
            allowed=None,
            allow_empty=True,
        )
    relevance = _required_array(
        episode.phase_relevance_masks,
        name="phase_relevance_masks",
        ndim=2,
    )
    _require_time_dim(relevance, time_steps, name="phase_relevance_masks")
    if relevance.shape[1] != action_dim:
        raise DemoSchemaError(
            f"phase_relevance_masks width must be {action_dim}, got {relevance.shape[1]}."
        )
    if np.any((relevance != 0) & (relevance != 1)):
        raise DemoSchemaError("phase_relevance_masks must contain only 0/1 values.")
    _validate_phase_relevance(
        phases=np.asarray(episode.online_phases),
        relevance=relevance,
        phase_contract=phase_contract,
        action_dim=action_dim,
    )

    intervention = _required_array(
        episode.intervention_flags,
        name="intervention_flags",
        ndim=1,
    )
    _require_time_dim(intervention, time_steps, name="intervention_flags")
    if np.any((intervention != 0) & (intervention != 1)):
        raise DemoSchemaError("intervention_flags must contain only 0/1 values.")
    _validate_level4_strings(
        episode.failure_reasons,
        name="failure_reasons",
        time_steps=time_steps,
        allowed=None,
        allow_empty=True,
    )
    _validate_intervention_interval(
        intervention=intervention,
        failure_reasons=np.asarray(episode.failure_reasons),
        metadata=metadata,
    )
    _validate_level4_timestamps(
        episode,
        time_steps=time_steps,
        action_contract=action_contract,
    )
    _validate_level4_rgb(episode, time_steps=time_steps)
    from dexvision.logging.phase_labels import validate_phase_intervals

    phase_intervals = metadata.get("phase_intervals")
    if not isinstance(phase_intervals, Sequence) or isinstance(phase_intervals, str):
        raise DemoSchemaError("Level 4 metadata phase_intervals must be a sequence.")
    try:
        validated_intervals = validate_phase_intervals(
            phase_intervals,
            frame_count=time_steps,
            phases=np.asarray(episode.online_phases).tolist(),
        )
    except ValueError as exc:
        raise DemoSchemaError(f"invalid Level 4 phase intervals: {exc}") from exc
    _validate_phase_transition_order(validated_intervals, phase_contract=phase_contract)
    orientation_start, orientation_stop = _coerce_range(
        action_schema.base_orientation_target,
        field_name="base_orientation_target",
    )
    _validate_applied_quaternion_continuity(applied[:, orientation_start:orientation_stop])


def _validate_prior_actions(
    actions: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    initial_fields = (
        ("initial_commanded_action", "prior_commanded_actions", "commanded_actions"),
        ("initial_applied_action", "prior_applied_actions", "applied_actions"),
    )
    for metadata_name, prior_name, current_name in initial_fields:
        if metadata_name not in metadata:
            raise DemoSchemaError(f"Level 4 metadata is missing '{metadata_name}'.")
        initial = np.asarray(metadata[metadata_name], dtype=np.float64)
        prior = actions[prior_name]
        current = actions[current_name]
        if initial.shape != (current.shape[1],):
            raise DemoSchemaError(
                f"{metadata_name} must have shape [{current.shape[1]}], got {initial.shape}."
            )
        if not np.array_equal(prior[0], initial):
            raise DemoSchemaError(f"{prior_name}[0] must equal metadata {metadata_name}.")
        if current.shape[0] > 1 and not np.array_equal(prior[1:], current[:-1]):
            raise DemoSchemaError(f"{prior_name} does not reproduce the prior sample exactly.")


def _validate_level4_action_safety(
    *,
    commanded: np.ndarray,
    applied: np.ndarray,
    safety_masks: np.ndarray | None,
    safety_reasons: np.ndarray | None,
    time_steps: int,
    action_dim: int,
    allowed_reasons: object,
) -> None:
    masks = _required_array(safety_masks, name="safety_masks", ndim=2)
    _require_time_dim(masks, time_steps, name="safety_masks")
    if masks.shape[1] != action_dim:
        raise DemoSchemaError(
            f"safety_masks width must be {action_dim}, got {masks.shape[1]}."
        )
    if np.any((masks != 0) & (masks != 1)):
        raise DemoSchemaError("safety_masks must contain only 0/1 values.")
    reasons = np.asarray(safety_reasons)
    if reasons.ndim != 2 or reasons.shape != (time_steps, action_dim):
        raise DemoSchemaError(
            f"safety_reasons must have shape [{time_steps}, {action_dim}], got {reasons.shape}."
        )
    if not np.issubdtype(reasons.dtype, np.str_):
        raise DemoSchemaError("safety_reasons must be a string array.")
    changed = ~np.isclose(commanded, applied, rtol=0.0, atol=1e-12)
    if np.any(changed & (masks == 0)):
        raise DemoSchemaError(
            "commanded/applied differences must be identified by safety_masks."
        )
    if np.any((masks == 0) & (reasons != "none")):
        raise DemoSchemaError("unmasked action fields must use safety reason 'none'.")
    if np.any((masks == 1) & (reasons == "none")):
        raise DemoSchemaError("masked action fields must carry a non-'none' safety reason.")
    if isinstance(allowed_reasons, str) or not isinstance(allowed_reasons, Sequence):
        raise DemoSchemaError("action_contract safety_reason_codes must be a sequence.")
    allowed = set(allowed_reasons)
    if not allowed or any(not isinstance(reason, str) or not reason for reason in allowed):
        raise DemoSchemaError("action_contract safety_reason_codes must contain strings.")
    unknown = sorted(set(reasons.reshape(-1).tolist()) - allowed)
    if unknown:
        raise DemoSchemaError("safety_reasons contains unknown codes: " + ", ".join(unknown))


def _validate_intervention_interval(
    *,
    intervention: np.ndarray,
    failure_reasons: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    active = np.flatnonzero(intervention == 1)
    interval = metadata.get("intervention_interval")
    if active.size == 0:
        if interval not in (None, []):
            raise DemoSchemaError(
                "intervention_interval must be null when no intervention frames are saved."
            )
        return
    if isinstance(interval, str) or not isinstance(interval, Sequence) or len(interval) != 2:
        raise DemoSchemaError("intervention_interval must contain [start_frame, end_frame].")
    start, end = interval
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise DemoSchemaError("intervention_interval must use valid integer frame bounds.")
    expected = np.arange(start, end)
    if not np.array_equal(active, expected):
        raise DemoSchemaError(
            "intervention_flags must exactly match the saved intervention_interval."
        )
    if not metadata.get("failure_reason"):
        raise DemoSchemaError("intervention frames require a metadata failure_reason.")
    if np.any(failure_reasons[start:end] == ""):
        raise DemoSchemaError("intervention frames require per-frame failure_reasons.")


def _validate_phase_relevance(
    *,
    phases: np.ndarray,
    relevance: np.ndarray,
    phase_contract: Mapping[str, Any],
    action_dim: int,
) -> None:
    vocabulary = phase_contract.get("vocabulary")
    if isinstance(vocabulary, str) or not isinstance(vocabulary, Sequence):
        raise DemoSchemaError("phase_contract vocabulary must be a sequence.")
    vocabulary_set = set(vocabulary)
    unknown_phases = sorted(set(phases.tolist()) - vocabulary_set)
    if unknown_phases:
        raise DemoSchemaError("online_phases contains unknown phases: " + ", ".join(unknown_phases))
    raw_masks = phase_contract.get("action_relevance_masks")
    if not isinstance(raw_masks, Mapping):
        raise DemoSchemaError("phase_contract action_relevance_masks must be a mapping.")
    for index, phase in enumerate(phases.tolist()):
        if phase not in raw_masks:
            raise DemoSchemaError(f"phase_contract has no relevance mask for '{phase}'.")
        expected = np.asarray(raw_masks[phase], dtype=np.uint8)
        if expected.shape != (action_dim,):
            raise DemoSchemaError(
                f"phase relevance mask for '{phase}' must have shape [{action_dim}]."
            )
        if not np.array_equal(relevance[index], expected):
            raise DemoSchemaError(
                f"phase_relevance_masks[{index}] does not match phase '{phase}'."
            )


def _validate_phase_transition_order(
    intervals: Sequence[Any],
    *,
    phase_contract: Mapping[str, Any],
) -> None:
    raw_transitions = phase_contract.get("transitions")
    if isinstance(raw_transitions, str) or not isinstance(raw_transitions, Sequence):
        raise DemoSchemaError("phase_contract transitions must be a sequence.")
    allowed: set[tuple[str, str]] = set()
    for transition in raw_transitions:
        if not isinstance(transition, Mapping):
            raise DemoSchemaError("phase_contract transitions must contain mappings.")
        source = transition.get("from")
        target = transition.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            raise DemoSchemaError("phase transition from/to values must be strings.")
        allowed.add((source, target))
    observed = [
        (left.phase, right.phase) for left, right in zip(intervals, intervals[1:])
    ]
    invalid = [edge for edge in observed if edge not in allowed]
    if invalid:
        raise DemoSchemaError(f"online phase transition is not permitted: {invalid[0]}.")


def _validate_applied_quaternion_continuity(quaternions: np.ndarray) -> None:
    _validate_quaternions(quaternions)
    first = quaternions[0]
    nonzero = first[np.abs(first) > 1e-12]
    if nonzero.size and nonzero[0] < 0.0:
        raise DemoSchemaError(
            "first applied quaternion violates the canonical positive-sign rule."
        )
    if quaternions.shape[0] > 1:
        dots = np.sum(quaternions[1:] * quaternions[:-1], axis=1)
        if np.any(dots < 0.0):
            raise DemoSchemaError("applied quaternion sequence is not sign-continuous.")


def _validate_level4_strings(
    values: np.ndarray | None,
    *,
    name: str,
    time_steps: int,
    allowed: Sequence[str] | None,
    allow_empty: bool,
) -> None:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != time_steps:
        raise DemoSchemaError(f"{name} must have shape [{time_steps}], got {array.shape}.")
    if not np.issubdtype(array.dtype, np.str_):
        raise DemoSchemaError(f"{name} must be a string array.")
    if not allow_empty and np.any(array == ""):
        raise DemoSchemaError(f"{name} must not contain empty values.")
    if allowed is not None:
        invalid = sorted(set(array.tolist()) - set(allowed))
        if invalid:
            raise DemoSchemaError(f"{name} contains unsupported values: {', '.join(invalid)}.")


def _validate_level4_timestamps(
    episode: DemoEpisode,
    *,
    time_steps: int,
    action_contract: Mapping[str, Any],
) -> None:
    timestamp_arrays = {
        "action_timestamps": episode.action_timestamps,
        "task_timestamps": episode.task_timestamps,
        "state_timestamps": episode.state_timestamps,
    }
    for name, value in timestamp_arrays.items():
        array = _required_array(value, name=name, ndim=1)
        _require_time_dim(array, time_steps, name=name)
        if np.any(np.diff(array) < 0.0):
            raise DemoSchemaError(f"{name} must be monotonic nondecreasing.")
    if not np.array_equal(episode.timestamps, episode.state_timestamps):
        raise DemoSchemaError("timestamps must exactly mirror state_timestamps for Level 4.")
    maximum_skew = action_contract.get("max_state_action_timestamp_skew_s", 0.005)
    if not isinstance(maximum_skew, (int, float)) or maximum_skew < 0.0:
        raise DemoSchemaError(
            "action_contract max_state_action_timestamp_skew_s must be non-negative."
        )
    state = np.asarray(episode.state_timestamps)
    for name, value in (
        ("action_timestamps", episode.action_timestamps),
        ("task_timestamps", episode.task_timestamps),
    ):
        if np.any(np.abs(np.asarray(value) - state) > float(maximum_skew)):
            raise DemoSchemaError(
                f"{name} exceeds max state alignment skew {maximum_skew:.6f}s."
            )


def _validate_level4_rgb(episode: DemoEpisode, *, time_steps: int) -> None:
    if episode.rgb_frames is None:
        if episode.rgb_timestamps is not None:
            raise DemoSchemaError("rgb_timestamps cannot be saved without rgb_frames.")
        return
    frames = np.asarray(episode.rgb_frames)
    if frames.ndim != 4 or frames.shape[0] != time_steps or frames.shape[-1] != 3:
        raise DemoSchemaError(
            "rgb_frames must have shape [T, H, W, 3] aligned with state samples."
        )
    if frames.dtype != np.uint8:
        raise DemoSchemaError("rgb_frames must use uint8 RGB pixels.")
    rgb_timestamps = _required_array(
        episode.rgb_timestamps,
        name="rgb_timestamps",
        ndim=1,
    )
    _require_time_dim(rgb_timestamps, time_steps, name="rgb_timestamps")
    if np.any(np.diff(rgb_timestamps) < 0.0):
        raise DemoSchemaError("rgb_timestamps must be monotonic nondecreasing.")
    camera_config = episode.metadata.get("camera_or_render_config")
    if not isinstance(camera_config, Mapping):
        raise DemoSchemaError(
            "camera_or_render_config must be a mapping when RGB frames are enabled."
        )
    for key in ("version", "calibration_version"):
        if not camera_config.get(key):
            raise DemoSchemaError(
                f"camera_or_render_config must include non-empty '{key}' when RGB is enabled."
            )
    maximum_skew = camera_config.get("max_rgb_state_timestamp_skew_s", 0.017)
    if not isinstance(maximum_skew, (int, float)) or maximum_skew < 0.0:
        raise DemoSchemaError(
            "camera_or_render_config max_rgb_state_timestamp_skew_s must be non-negative."
        )
    if np.any(
        np.abs(rgb_timestamps - np.asarray(episode.state_timestamps)) > float(maximum_skew)
    ):
        raise DemoSchemaError(
            f"rgb_timestamps exceeds max state alignment skew {maximum_skew:.6f}s."
        )


def extract_observations(
    episode: DemoEpisode,
    *,
    observation_schema: ObservationSchema,
) -> dict[str, np.ndarray | None]:
    """Reconstruct all declared observation fields from one saved episode."""

    return observation_schema.extract(
        {
            "robot_states": episode.robot_states,
            "tracking_quality": episode.tracking_quality,
            "object_states": episode.object_states,
            "task_states": episode.task_states,
        },
        time_steps=int(episode.timestamps.shape[0]),
    )


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
