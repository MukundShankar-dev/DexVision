"""Disk logger for Level 2 demonstration episodes."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.dataset_schema import (
    ActionSchema,
    DemoEpisode,
    ObservationFieldLayout,
    ObservationSchema,
    validate_demo,
)


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
            )
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

        return DemoEpisode(
            metadata=dict(self._metadata or {}),
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
        )

    def _write_episode(self, episode: DemoEpisode) -> None:
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
        metadata = dict(episode.metadata)
        metadata["success"] = episode.success
        metadata["num_steps"] = int(episode.timestamps.shape[0])
        (self.output_dir / "metadata.json").write_text(
            json.dumps(_to_jsonable(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.save(self.output_dir / "features.npy", episode.features)
        np.save(self.output_dir / "actions.npy", episode.actions)
        np.save(self.output_dir / "robot_states.npy", episode.robot_states)
        np.save(self.output_dir / "tracking_quality.npy", episode.tracking_quality)
        np.save(self.output_dir / "timestamps.npy", episode.timestamps)
        if episode.landmarks is not None:
            np.save(self.output_dir / "landmarks.npy", episode.landmarks)
        if episode.object_states is not None:
            np.save(self.output_dir / "object_states.npy", episode.object_states)
        if episode.task_states is not None:
            np.save(self.output_dir / "task_states.npy", episode.task_states)


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
