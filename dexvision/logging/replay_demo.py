"""Load and replay saved Level 2 demonstration episodes."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dexvision.logging.dataset_schema import (
    ActionSchema,
    DemoEpisode,
    DemoSchemaError,
    ObservationSchema,
    validate_demo,
)
from dexvision.logging.demo_logger import DemoLoggerError, load_logged_demo


DEFAULT_MOCAP_BODY_NAME = "dexvision_hand_base_target"


class DemoReplayError(RuntimeError):
    """Raised when a saved demo cannot be loaded or replayed."""


class ReplayEnv(Protocol):
    """MuJoCo-like interface needed by the replay loop."""

    def reset(self) -> object:
        """Reset the simulation before replay starts."""

    def set_mocap_pose(
        self,
        body_name: str,
        *,
        position: Sequence[float] | np.ndarray,
        orientation_quat: Sequence[float] | np.ndarray,
    ) -> None:
        """Apply the recorded base target pose."""

    def set_joint_targets(self, joint_targets: Mapping[str, float]) -> None:
        """Apply recorded finger actuator targets by actuator name."""

    def step(self, *, n_steps: int = 1) -> object:
        """Advance the simulation."""


@dataclass(frozen=True)
class LoadedReplayDemo:
    """A validated demo plus replay metadata reconstructed from disk."""

    demo_dir: Path
    episode: DemoEpisode
    action_schema: ActionSchema
    observation_schema: ObservationSchema
    finger_target_names: tuple[str, ...]
    model_path: Path
    mocap_body_name: str


@dataclass(frozen=True)
class ReplayStep:
    """One split Level 1.13 action ready to apply to MuJoCo."""

    index: int
    timestamp: float
    base_position_target: np.ndarray
    base_orientation_target: np.ndarray
    finger_actuator_targets: dict[str, float]


@dataclass(frozen=True)
class ReplayResult:
    """Summary returned after replaying an episode."""

    steps_replayed: int
    first_timestamp: float | None
    last_timestamp: float | None
    final_sim_time: float | None
    stopped_early: bool = False


ProgressCallback = Callable[[ReplayStep, object], None]
SleepFn = Callable[[float], None]
ViewerSync = Callable[[], None]
StopCallback = Callable[[], bool]


def load_replay_demo(
    demo_dir: str | Path,
    *,
    model_override: str | Path | None = None,
    mocap_body_override: str | None = None,
) -> LoadedReplayDemo:
    """Load, reconstruct schemas, and validate a saved demo directory."""

    path = Path(demo_dir)
    if not path.exists():
        raise DemoReplayError(f"demo directory does not exist: {path}")
    if not path.is_dir():
        raise DemoReplayError(f"demo path is not a directory: {path}")

    try:
        episode = load_logged_demo(path)
        action_schema = action_schema_from_metadata(episode.metadata)
        observation_schema = observation_schema_from_metadata(episode.metadata)
        validate_demo(
            episode,
            action_schema=action_schema,
            observation_schema=observation_schema,
        )
    except (DemoLoggerError, DemoSchemaError) as exc:
        raise DemoReplayError(f"invalid saved demo '{path}': {exc}") from exc

    finger_target_names = finger_target_names_from_metadata(
        episode.metadata,
        action_schema=action_schema,
    )
    return LoadedReplayDemo(
        demo_dir=path,
        episode=episode,
        action_schema=action_schema,
        observation_schema=observation_schema,
        finger_target_names=finger_target_names,
        model_path=_model_path_from_metadata(
            episode.metadata,
            model_override=model_override,
        ),
        mocap_body_name=_mocap_body_from_metadata(
            episode.metadata,
            mocap_body_override=mocap_body_override,
        ),
    )


def action_schema_from_metadata(metadata: Mapping[str, Any]) -> ActionSchema:
    """Reconstruct the saved full-action schema from ``metadata.json``."""

    payload = _mapping_value(
        metadata,
        "action_schema",
        message=(
            "metadata.json is missing action_schema; cannot replay the full "
            "Level 1.13 action."
        ),
    )
    return ActionSchema(
        version=str(payload.get("version") or metadata.get("action_schema_version") or ""),
        base_position_target=_range_value(payload, "base_position_target"),
        base_orientation_target=_range_value(payload, "base_orientation_target"),
        finger_actuator_targets=_range_value(payload, "finger_actuator_targets"),
        representation_notes=_optional_mapping(payload.get("representation_notes")),
    )


def observation_schema_from_metadata(metadata: Mapping[str, Any]) -> ObservationSchema:
    """Reconstruct the saved observation schema from ``metadata.json``."""

    payload = _mapping_value(
        metadata,
        "observation_schema",
        message="metadata.json is missing observation_schema; cannot validate replay input.",
    )
    shapes_payload = _mapping_value(
        payload,
        "shapes",
        message="metadata observation_schema is missing shapes.",
    )
    return ObservationSchema(
        version=str(payload.get("version") or metadata.get("observation_schema_version") or ""),
        fields=_string_tuple_value(payload.get("fields"), "observation_schema.fields"),
        shapes={
            str(name): _shape_value(shape, f"observation_schema.shapes.{name}")
            for name, shape in shapes_payload.items()
        },
        optional_fields=_string_tuple_value(
            payload.get("optional_fields", ()),
            "observation_schema.optional_fields",
        ),
    )


def finger_target_names_from_metadata(
    metadata: Mapping[str, Any],
    *,
    action_schema: ActionSchema,
) -> tuple[str, ...]:
    """Return actuator names in the same order as the saved action columns."""

    raw_names = metadata.get("finger_target_names")
    if raw_names is None:
        raw_names = action_schema.representation_notes.get("finger_target_names")
    names = _string_tuple_value(
        raw_names,
        "finger_target_names",
        missing_message=(
            "metadata.json is missing finger_target_names; cannot map recorded "
            "finger_actuator_targets back to MuJoCo actuators."
        ),
    )
    expected = _range_length(action_schema.finger_actuator_targets)
    if len(names) != expected:
        raise DemoReplayError(
            "finger_target_names length does not match finger_actuator_targets: "
            f"expected {expected}, got {len(names)}."
        )
    return names


def iter_replay_steps(loaded_demo: LoadedReplayDemo) -> tuple[ReplayStep, ...]:
    """Split all recorded action rows into replay-ready steps."""

    split_actions = loaded_demo.action_schema.split(loaded_demo.episode.actions)
    base_positions = split_actions["base_position_target"]
    base_orientations = split_actions["base_orientation_target"]
    finger_targets = split_actions["finger_actuator_targets"]
    timestamps = np.asarray(loaded_demo.episode.timestamps, dtype=np.float64)

    steps: list[ReplayStep] = []
    for index, timestamp in enumerate(timestamps):
        steps.append(
            ReplayStep(
                index=index,
                timestamp=float(timestamp),
                base_position_target=np.asarray(base_positions[index], dtype=np.float64).copy(),
                base_orientation_target=_normalized_quaternion(base_orientations[index]),
                finger_actuator_targets={
                    name: float(value)
                    for name, value in zip(
                        loaded_demo.finger_target_names,
                        finger_targets[index],
                        strict=True,
                    )
                },
            )
        )
    return tuple(steps)


def apply_replay_step(
    env: ReplayEnv,
    step: ReplayStep,
    *,
    mocap_body_name: str,
    sim_steps_per_action: int,
) -> object:
    """Apply one recorded action and step the simulation."""

    if sim_steps_per_action <= 0:
        raise DemoReplayError("sim_steps_per_action must be a positive integer.")
    env.set_mocap_pose(
        mocap_body_name,
        position=step.base_position_target,
        orientation_quat=step.base_orientation_target,
    )
    env.set_joint_targets(step.finger_actuator_targets)
    return env.step(n_steps=sim_steps_per_action)


def replay_loaded_demo(
    loaded_demo: LoadedReplayDemo,
    env: ReplayEnv,
    *,
    speed: float = 1.0,
    sim_steps_per_action: int = 1,
    max_steps: int | None = None,
    sleep_fn: SleepFn = time.sleep,
    viewer_sync: ViewerSync | None = None,
    should_stop: StopCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ReplayResult:
    """Replay a validated demo into an environment.

    ``speed=1.0`` follows recorded timestamp spacing. Values below 1.0 replay
    more slowly, and values above 1.0 replay faster.
    """

    if speed <= 0.0:
        raise DemoReplayError("speed must be positive.")
    if sim_steps_per_action <= 0:
        raise DemoReplayError("sim_steps_per_action must be a positive integer.")
    if max_steps is not None and max_steps <= 0:
        raise DemoReplayError("max_steps must be positive when provided.")

    steps = iter_replay_steps(loaded_demo)
    if max_steps is not None:
        steps = steps[:max_steps]
    if not steps:
        raise DemoReplayError("demo contains no replayable actions.")

    env.reset()
    previous_timestamp: float | None = None
    final_sim_time: float | None = None
    steps_replayed = 0
    stopped_early = False

    for step in steps:
        if should_stop is not None and should_stop():
            stopped_early = True
            break
        if previous_timestamp is not None:
            delay = max(0.0, (step.timestamp - previous_timestamp) / speed)
            if delay > 0.0:
                sleep_fn(delay)

        state = apply_replay_step(
            env,
            step,
            mocap_body_name=loaded_demo.mocap_body_name,
            sim_steps_per_action=sim_steps_per_action,
        )
        final_sim_time = _state_time(state)
        steps_replayed += 1
        previous_timestamp = step.timestamp

        if progress_callback is not None:
            progress_callback(step, state)
        if viewer_sync is not None:
            viewer_sync()

    first_timestamp = steps[0].timestamp if steps_replayed else None
    last_timestamp = previous_timestamp if steps_replayed else None
    return ReplayResult(
        steps_replayed=steps_replayed,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        final_sim_time=final_sim_time,
        stopped_early=stopped_early,
    )


def _model_path_from_metadata(
    metadata: Mapping[str, Any],
    *,
    model_override: str | Path | None,
) -> Path:
    if model_override is not None:
        return Path(model_override)
    raw_model = metadata.get("robot_model")
    if not isinstance(raw_model, str) or not raw_model:
        raise DemoReplayError("metadata field 'robot_model' must be a non-empty string.")
    return Path(raw_model)


def _mocap_body_from_metadata(
    metadata: Mapping[str, Any],
    *,
    mocap_body_override: str | None,
) -> str:
    if mocap_body_override is not None:
        if not mocap_body_override:
            raise DemoReplayError("mocap_body_override cannot be empty.")
        return mocap_body_override

    teleop_config = metadata.get("teleop_config", {})
    if isinstance(teleop_config, Mapping):
        base_config = teleop_config.get("base_control", {})
        if isinstance(base_config, Mapping):
            raw_body = base_config.get("mocap_body", base_config.get("mocap_body_name"))
            if isinstance(raw_body, str) and raw_body:
                return raw_body
    return DEFAULT_MOCAP_BODY_NAME


def _mapping_value(
    payload: Mapping[str, Any],
    key: str,
    *,
    message: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DemoReplayError(message)
    return value


def _optional_mapping(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DemoReplayError("schema representation_notes must be a mapping.")
    return dict(value)


def _range_value(payload: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise DemoReplayError(f"metadata action_schema.{key} must be [start, stop].")
    start = int(value[0])
    stop = int(value[1])
    if start < 0 or stop <= start:
        raise DemoReplayError(f"metadata action_schema.{key} must satisfy 0 <= start < stop.")
    return (start, stop)


def _range_length(index_range: object) -> int:
    if isinstance(index_range, slice):
        if index_range.start is None or index_range.stop is None:
            raise DemoReplayError("schema slices must have start and stop.")
        return int(index_range.stop) - int(index_range.start)
    if isinstance(index_range, tuple) and len(index_range) == 2:
        return int(index_range[1]) - int(index_range[0])
    raise DemoReplayError("schema ranges must be slices or (start, stop) tuples.")


def _string_tuple_value(
    value: object,
    field_name: str,
    *,
    missing_message: str | None = None,
) -> tuple[str, ...]:
    if value is None:
        raise DemoReplayError(missing_message or f"metadata field '{field_name}' is required.")
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        raise DemoReplayError(f"metadata field '{field_name}' must be a sequence of strings.")
    if not all(isinstance(item, str) and item for item in values):
        raise DemoReplayError(f"metadata field '{field_name}' must contain non-empty strings.")
    return values


def _shape_value(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise DemoReplayError(f"metadata field '{field_name}' must be a shape sequence.")
    shape = tuple(int(dimension) for dimension in value)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise DemoReplayError(f"metadata field '{field_name}' must contain positive dimensions.")
    return shape


def _normalized_quaternion(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,):
        raise DemoReplayError(f"base orientation must have shape [4], got {quat.shape}.")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        raise DemoReplayError("base orientation quaternion must be finite and nonzero.")
    return quat / norm


def _state_time(state: object) -> float | None:
    value = getattr(state, "time", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
