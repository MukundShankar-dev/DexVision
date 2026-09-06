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
    ObservationFieldLayout,
    ObservationSchema,
    validate_demo,
)
from dexvision.logging.demo_logger import DemoLoggerError, load_logged_demo
from dexvision.sim.tasks import (
    BUTTON_PRESS_TASK_ID,
    PUSH_CUBE_TASK_ID,
    color_button_press_target,
    configure_button_press_scene,
    configure_push_cube_scene,
)


DEFAULT_MOCAP_BODY_NAME = "dexvision_hand_base_target"
REACH_TOUCH_TARGET_TASK_ID = "reach_touch_target"
DEFAULT_REACH_TOUCH_TARGET_MARKER_BODY = "reach_target_marker"
DEFAULT_PUSH_CUBE_TARGET_MARKER_BODY = "push_cube_target_marker"


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

    def preserve_free_joint_orientation(
        self,
        joint_name: str,
        orientation_quat: Sequence[float] | np.ndarray,
    ) -> None:
        """Restore one free joint's orientation without constraining translation."""

    def configure_contact_dynamics(
        self,
        *,
        table_geom_name: str,
        table_condim: int,
        geom_friction: Mapping[str, Sequence[float | None]],
    ) -> None:
        """Restore task-local contact parameters recorded in episode metadata."""

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
    version = str(payload.get("version") or metadata.get("observation_schema_version") or "")
    fields = _string_tuple_value(payload.get("fields"), "observation_schema.fields")
    shapes = {
        str(name): _shape_value(shape, f"observation_schema.shapes.{name}")
        for name, shape in shapes_payload.items()
    }
    optional_fields = _string_tuple_value(
        payload.get("optional_fields", ()),
        "observation_schema.optional_fields",
    )
    raw_layouts = payload.get("layouts")
    if not raw_layouts:
        return adapt_legacy_observation_schema(
            version=version,
            fields=fields,
            shapes=shapes,
            optional_fields=optional_fields,
        )
    if not isinstance(raw_layouts, Mapping):
        raise DemoReplayError("metadata observation_schema.layouts must be a mapping.")

    layouts: dict[str, ObservationFieldLayout] = {}
    for field_name, raw_layout in raw_layouts.items():
        if not isinstance(raw_layout, Mapping):
            raise DemoReplayError(
                f"metadata observation layout '{field_name}' must be a mapping."
            )
        raw_range = raw_layout.get("column_range")
        column_range = (
            None
            if raw_range is None
            else _range_value(raw_layout, "column_range")
        )
        layouts[str(field_name)] = ObservationFieldLayout(
            source_array=str(raw_layout.get("source_array") or ""),
            shape=_shape_value(
                raw_layout.get("shape"),
                f"observation_schema.layouts.{field_name}.shape",
            ),
            dtype=str(raw_layout.get("dtype") or ""),
            units=str(raw_layout.get("units") or ""),
            coordinate_frame=str(raw_layout.get("coordinate_frame") or ""),
            normalization=str(raw_layout.get("normalization") or ""),
            column_range=column_range,
            column_indices=_integer_tuple_value(
                raw_layout.get("column_indices", ()),
                f"observation_schema.layouts.{field_name}.column_indices",
            ),
            names=_string_tuple_value(
                raw_layout.get("names", ()),
                f"observation_schema.layouts.{field_name}.names",
            ),
            optional=bool(raw_layout.get("optional", False)),
            absence_rule=_optional_string(raw_layout.get("absence_rule")),
            mask_field=_optional_string(raw_layout.get("mask_field")),
        )
    return ObservationSchema(
        version=version,
        fields=fields,
        shapes=shapes,
        optional_fields=optional_fields,
        layouts=layouts,
        compatibility_notes=_string_tuple_value(
            payload.get("compatibility_notes", ()),
            "observation_schema.compatibility_notes",
        ),
    )


def adapt_legacy_observation_schema(
    *,
    version: str,
    fields: tuple[str, ...],
    shapes: Mapping[str, tuple[int, ...]],
    optional_fields: tuple[str, ...],
) -> ObservationSchema:
    """Keep v1 shape-only demos replayable without inventing field mappings."""

    if version != "level2/observation-v1":
        raise DemoReplayError(
            f"observation schema '{version}' is missing executable layouts."
        )
    return ObservationSchema(
        version=version,
        fields=fields,
        shapes=shapes,
        optional_fields=optional_fields,
        layouts={},
        compatibility_notes=(
            "Legacy Level 2.4 shape-only observation schema: full actions remain replayable, "
            "but dense observation extraction requires migration to observation-layout-v2.",
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
    reset_env: bool = True,
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

    if reset_env:
        env.reset()
        _restore_task_replay_state(loaded_demo, env)
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

        if _apply_level4_orientation_hold(loaded_demo, env, step.index):
            remaining = sim_steps_per_action
            while remaining > 0:
                chunk_steps = min(8, remaining)
                state = apply_replay_step(
                    env,
                    step,
                    mocap_body_name=loaded_demo.mocap_body_name,
                    sim_steps_per_action=chunk_steps,
                )
                remaining -= chunk_steps
                if remaining > 0:
                    _apply_level4_orientation_hold(loaded_demo, env, step.index)
        else:
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


def _apply_level4_orientation_hold(
    loaded_demo: LoadedReplayDemo,
    env: ReplayEnv,
    step_index: int,
) -> bool:
    """Replay the declared rotation-only hold for scripted grasp phases."""

    metadata = loaded_demo.episode.metadata
    if (
        metadata.get("task_id") != "level4_workcell"
        or metadata.get("source") != "scripted"
        or metadata.get("skill_name") not in {"pick_object", "pick_place_sequence"}
    ):
        return False
    phases = loaded_demo.episode.online_phases
    if phases is None or str(phases[step_index]) not in {
        "lift",
        "stabilize",
        "transport",
        "place",
    }:
        return False
    teleop_config = _mapping_value(
        metadata,
        "teleop_config",
        message="scripted Level 4 replay is missing teleop_config.",
    )
    scripted_expert = _mapping_value(
        teleop_config,
        "scripted_expert",
        message="scripted Level 4 replay is missing scripted_expert metadata.",
    )
    grasp = scripted_expert
    if metadata.get("skill_name") == "pick_place_sequence":
        grasp = _mapping_value(
            scripted_expert,
            "grasp",
            message="scripted pick/place replay is missing grasp metadata.",
        )
    if (
        grasp.get("orientation_preservation_policy")
        != "shape_aware_hammer_grip_with_world_orientation_hold"
    ):
        return False
    typed_goal = _mapping_value(
        metadata,
        "typed_goal",
        message="scripted grasp replay is missing typed_goal metadata.",
    )
    object_id = typed_goal.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        raise DemoReplayError(
            "scripted grasp replay typed_goal.object_id must be a non-empty string."
        )
    task_config = _mapping_value(
        metadata,
        "task_config",
        message="scripted grasp replay is missing task_config metadata.",
    )
    initial_state = _mapping_value(
        task_config,
        "initial_state",
        message="scripted grasp replay is missing initial_state metadata.",
    )
    objects = _mapping_value(
        initial_state,
        "objects",
        message="scripted grasp replay is missing initial object metadata.",
    )
    object_state = _mapping_value(
        objects,
        object_id,
        message=f"scripted grasp replay is missing state for object {object_id!r}.",
    )
    joint_name = object_state.get("joint_name")
    if not isinstance(joint_name, str) or not joint_name:
        raise DemoReplayError(
            f"scripted grasp replay object {object_id!r} is missing joint_name."
        )
    orientation = _normalized_quaternion(
        _finite_vector(
            object_state.get("orientation_wxyz"),
            size=4,
            field_name=f"initial_state.objects.{object_id}.orientation_wxyz",
        )
    )
    env.preserve_free_joint_orientation(joint_name, orientation)
    return True


def _restore_task_replay_state(
    loaded_demo: LoadedReplayDemo,
    env: ReplayEnv,
) -> None:
    """Restore task objects that are not part of the recorded action vector."""

    metadata = loaded_demo.episode.metadata
    if metadata.get("task_id") == "level4_workcell":
        _restore_level4_workcell_replay_state(loaded_demo, env)
        return
    if metadata.get("task_id") == BUTTON_PRESS_TASK_ID:
        configure_button_press_scene(env)  # type: ignore[arg-type]
        task_config = _mapping_value(
            metadata,
            "task_config",
            message="button_press metadata is missing task_config.",
        )
        button_id = task_config.get("resolved_button_id")
        if not isinstance(button_id, str) or not button_id:
            raise DemoReplayError(
                "button_press metadata task_config.resolved_button_id "
                "must be a non-empty string."
            )
        color_button_press_target(  # type: ignore[arg-type]
            env,
            button_id,
        )
        return
    if metadata.get("task_id") == PUSH_CUBE_TASK_ID:
        _restore_push_cube_replay_state(loaded_demo, env)
        return
    if metadata.get("task_id") != REACH_TOUCH_TARGET_TASK_ID:
        return

    task_config = _mapping_value(
        metadata,
        "task_config",
        message="reach_touch_target metadata is missing task_config.",
    )
    target_position = np.asarray(task_config.get("target_position"), dtype=np.float64)
    if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
        raise DemoReplayError(
            "reach_touch_target metadata task_config.target_position "
            "must contain three finite world-frame values."
        )
    marker_body = task_config.get(
        "target_marker_body",
        DEFAULT_REACH_TOUCH_TARGET_MARKER_BODY,
    )
    if not isinstance(marker_body, str) or not marker_body:
        raise DemoReplayError(
            "reach_touch_target metadata task_config.target_marker_body "
            "must be a non-empty string."
        )
    env.set_mocap_pose(
        marker_body,
        position=target_position,
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def _restore_push_cube_replay_state(
    loaded_demo: LoadedReplayDemo,
    env: ReplayEnv,
) -> None:
    """Restore the saved cube start pose/velocity and target marker."""

    task_config = _mapping_value(
        loaded_demo.episode.metadata,
        "task_config",
        message="push_cube_to_target metadata is missing task_config.",
    )
    object_id = task_config.get("resolved_object_id")
    if not isinstance(object_id, str) or not object_id:
        raise DemoReplayError(
            "push_cube_to_target metadata task_config.resolved_object_id "
            "must be a non-empty string."
        )
    initial_position = _finite_vector(
        task_config.get("initial_object_position"),
        size=3,
        field_name="task_config.initial_object_position",
    )
    initial_orientation = _normalized_quaternion(
        _finite_vector(
            task_config.get("initial_object_orientation"),
            size=4,
            field_name="task_config.initial_object_orientation",
        )
    )
    initial_linear_velocity = _finite_vector(
        task_config.get("initial_object_linear_velocity"),
        size=3,
        field_name="task_config.initial_object_linear_velocity",
    )
    initial_angular_velocity = _finite_vector(
        task_config.get("initial_object_angular_velocity"),
        size=3,
        field_name="task_config.initial_object_angular_velocity",
    )
    target_position = _finite_vector(
        task_config.get("target_position"),
        size=3,
        field_name="task_config.target_position",
    )
    initial_base_position = _finite_vector(
        task_config.get("initial_base_position"),
        size=3,
        field_name="task_config.initial_base_position",
    )
    initial_base_orientation = _normalized_quaternion(
        _finite_vector(
            task_config.get("initial_base_orientation"),
            size=4,
            field_name="task_config.initial_base_orientation",
        )
    )
    base_free_joint = task_config.get("base_free_joint", "rh_base_freejoint")
    if not isinstance(base_free_joint, str) or not base_free_joint:
        raise DemoReplayError(
            "push_cube_to_target metadata task_config.base_free_joint "
            "must be a non-empty string."
        )
    marker_body = task_config.get(
        "target_marker_body",
        DEFAULT_PUSH_CUBE_TARGET_MARKER_BODY,
    )
    if not isinstance(marker_body, str) or not marker_body:
        raise DemoReplayError(
            "push_cube_to_target metadata task_config.target_marker_body "
            "must be a non-empty string."
        )

    configure_push_cube_scene(env)  # type: ignore[arg-type]
    env.set_mocap_pose(
        loaded_demo.mocap_body_name,
        position=initial_base_position,
        orientation_quat=initial_base_orientation,
    )
    _set_free_joint_state(
        env,
        joint_name=base_free_joint,
        position=initial_base_position,
        orientation=initial_base_orientation,
        linear_velocity=np.zeros(3, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
    )
    _set_free_joint_state(
        env,
        joint_name=f"{object_id}_joint",
        position=initial_position,
        orientation=initial_orientation,
        linear_velocity=initial_linear_velocity,
        angular_velocity=initial_angular_velocity,
    )
    env.set_mocap_pose(
        marker_body,
        position=target_position,
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    _forward_env(env)


def _restore_level4_workcell_replay_state(
    loaded_demo: LoadedReplayDemo,
    env: ReplayEnv,
) -> None:
    """Restore every randomized workcell object from immutable episode metadata."""

    task_config = _mapping_value(
        loaded_demo.episode.metadata,
        "task_config",
        message="level4_workcell metadata is missing task_config.",
    )
    initial_state = _mapping_value(
        task_config,
        "initial_state",
        message="level4_workcell task_config is missing initial_state.",
    )
    objects = _mapping_value(
        initial_state,
        "objects",
        message="level4_workcell initial_state is missing objects.",
    )
    if not objects:
        raise DemoReplayError("level4_workcell initial_state.objects must not be empty.")
    contact = task_config.get("contact_dynamics")
    if contact is not None:
        if not isinstance(contact, Mapping):
            raise DemoReplayError("level4_workcell contact_dynamics must be a mapping.")
        table_geom_name = contact.get("table_geom_name")
        table_condim = contact.get("table_condim")
        geom_friction = contact.get("geom_friction")
        if not isinstance(table_geom_name, str) or not table_geom_name:
            raise DemoReplayError(
                "level4_workcell contact_dynamics.table_geom_name is invalid."
            )
        if not isinstance(table_condim, int) or isinstance(table_condim, bool):
            raise DemoReplayError(
                "level4_workcell contact_dynamics.table_condim must be an integer."
            )
        if not isinstance(geom_friction, Mapping):
            raise DemoReplayError(
                "level4_workcell contact_dynamics.geom_friction must be a mapping."
            )
        env.configure_contact_dynamics(
            table_geom_name=table_geom_name,
            table_condim=table_condim,
            geom_friction=geom_friction,
        )
    entity_positions = initial_state.get("entity_positions_m")
    if isinstance(entity_positions, Mapping) and "start_button" in entity_positions:
        _set_world_body_position(
            env,
            body_name="start_button",
            position=_finite_vector(
                entity_positions["start_button"],
                size=3,
                field_name="initial_state.entity_positions_m.start_button",
            ),
        )
    for object_id, raw_state in objects.items():
        if not isinstance(object_id, str) or not isinstance(raw_state, Mapping):
            raise DemoReplayError(
                "level4_workcell initial object states must map ids to mappings."
            )
        joint_name = raw_state.get("joint_name")
        if not isinstance(joint_name, str) or not joint_name:
            raise DemoReplayError(
                f"level4_workcell object {object_id!r} is missing joint_name."
            )
        _set_free_joint_state(
            env,
            joint_name=joint_name,
            position=_finite_vector(
                raw_state.get("position_m"),
                size=3,
                field_name=f"initial_state.objects.{object_id}.position_m",
            ),
            orientation=_normalized_quaternion(
                _finite_vector(
                    raw_state.get("orientation_wxyz"),
                    size=4,
                    field_name=(
                        f"initial_state.objects.{object_id}.orientation_wxyz"
                    ),
                )
            ),
            linear_velocity=_finite_vector(
                raw_state.get("linear_velocity_mps"),
                size=3,
                field_name=(
                    f"initial_state.objects.{object_id}.linear_velocity_mps"
                ),
            ),
            angular_velocity=_finite_vector(
                raw_state.get("angular_velocity_radps"),
                size=3,
                field_name=(
                    f"initial_state.objects.{object_id}.angular_velocity_radps"
                ),
            ),
        )
    typed_goal = loaded_demo.episode.metadata.get("typed_goal")
    skill_name = loaded_demo.episode.metadata.get("skill_name")
    if skill_name in {
        "reach_object",
        "pick_object",
        "pick_place_sequence",
        "press_button",
        "push_object_to_target",
    } and isinstance(
        typed_goal, Mapping
    ):
        if skill_name == "reach_object":
            entity_id = typed_goal.get("entity_id")
        elif skill_name in {"pick_object", "pick_place_sequence"}:
            entity_id = typed_goal.get("object_id")
        elif skill_name == "press_button":
            entity_id = typed_goal.get("button_id")
        else:
            entity_id = typed_goal.get("object_id")
        entity_positions = initial_state.get("entity_positions_m")
        raw_entity_position = (
            entity_positions.get(entity_id)
            if isinstance(entity_id, str) and isinstance(entity_positions, Mapping)
            else None
        )
        if raw_entity_position is None and isinstance(entity_id, str):
            entity_state = objects.get(entity_id)
            if isinstance(entity_state, Mapping):
                raw_entity_position = entity_state.get("position_m")
        if raw_entity_position is None:
            raise DemoReplayError(
                "scripted replay goal entity position is missing from initial state."
            )
        entity_position = _finite_vector(
            raw_entity_position,
            size=3,
            field_name=f"initial_state.entity_positions_m.{entity_id}",
        )
        env.set_mocap_pose(
            "pilot_target_outline",
            position=entity_position,
            orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        goal_position = entity_position
        if skill_name == "reach_object":
            goal_position = _finite_vector(
                typed_goal.get("approach_pose"),
                size=7,
                field_name="typed_goal.approach_pose",
            )[:3]
        elif skill_name == "pick_object":
            goal_position = entity_position.copy()
            goal_position[2] += 0.08
        elif skill_name == "pick_place_sequence":
            target_id = typed_goal.get("target_id")
            raw_target_position = (
                entity_positions.get(target_id)
                if isinstance(target_id, str) and isinstance(entity_positions, Mapping)
                else None
            )
            if raw_target_position is None:
                raise DemoReplayError(
                    "scripted pick/place replay target position is missing from "
                    "initial state."
                )
            goal_position = _finite_vector(
                raw_target_position,
                size=3,
                field_name=f"initial_state.entity_positions_m.{target_id}",
            )
            goal_position[2] = max(0.06, goal_position[2] + 0.05)
        elif skill_name == "push_object_to_target":
            target_id = typed_goal.get("target_zone")
            raw_target_position = (
                entity_positions.get(target_id)
                if isinstance(target_id, str) and isinstance(entity_positions, Mapping)
                else None
            )
            if raw_target_position is None:
                raise DemoReplayError(
                    "scripted push replay target position is missing from initial state."
                )
            goal_position = _finite_vector(
                raw_target_position,
                size=3,
                field_name=f"initial_state.entity_positions_m.{target_id}",
            )
        env.set_mocap_pose(
            "pilot_goal_marker",
            position=goal_position,
            orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
    _forward_env(env)


def _set_world_body_position(
    env: ReplayEnv, *, body_name: str, position: np.ndarray
) -> None:
    """Restore a direct world child whose pose changed between scene revisions."""

    model = getattr(env, "model", None)
    mujoco_module = getattr(env, "_mujoco", None)
    if model is None or mujoco_module is None:
        raise DemoReplayError(
            "Level 4 fixture replay requires a MuJoCo model and module handle."
        )
    body_id = int(
        mujoco_module.mj_name2id(
            model, mujoco_module.mjtObj.mjOBJ_BODY, body_name
        )
    )
    if body_id < 0:
        raise DemoReplayError(f"Level 4 replay model is missing body {body_name!r}.")
    if int(model.body_parentid[body_id]) != 0:
        raise DemoReplayError(
            f"Level 4 replay body {body_name!r} must be a direct world child."
        )
    model.body_pos[body_id] = position


def _set_free_joint_state(
    env: ReplayEnv,
    *,
    joint_name: str,
    position: np.ndarray,
    orientation: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> None:
    """Set one MuJoCo free joint through the replay environment."""

    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    mujoco_module = getattr(env, "_mujoco", None)
    if model is None or data is None or mujoco_module is None:
        raise DemoReplayError(
            "push_cube_to_target semantic replay requires a MuJoCo environment "
            "with model, data, and _mujoco attributes."
        )
    joint_id = mujoco_module.mj_name2id(
        model,
        mujoco_module.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    if joint_id < 0:
        raise DemoReplayError(
            f"push_cube_to_target replay model is missing free joint {joint_name!r}."
        )
    if int(model.jnt_type[joint_id]) != int(mujoco_module.mjtJoint.mjJNT_FREE):
        raise DemoReplayError(
            f"push_cube_to_target replay joint {joint_name!r} must be a free joint."
        )
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 3] = position
    data.qpos[qpos_address + 3 : qpos_address + 7] = orientation
    data.qvel[qvel_address : qvel_address + 3] = linear_velocity
    data.qvel[qvel_address + 3 : qvel_address + 6] = angular_velocity


def _forward_env(env: ReplayEnv) -> None:
    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    mujoco_module = getattr(env, "_mujoco", None)
    if model is None or data is None or mujoco_module is None:
        raise DemoReplayError(
            "semantic replay requires a MuJoCo environment with model, data, "
            "and _mujoco attributes."
        )
    mujoco_module.mj_forward(model, data)


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
        raise DemoReplayError(f"metadata schema field '{key}' must be [start, stop].")
    start = int(value[0])
    stop = int(value[1])
    if start < 0 or stop <= start:
        raise DemoReplayError(
            f"metadata schema field '{key}' must satisfy 0 <= start < stop."
        )
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


def _integer_tuple_value(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise DemoReplayError(f"metadata field '{field_name}' must be an integer sequence.")
    values = tuple(value)
    if not all(isinstance(item, int) and item >= 0 for item in values):
        raise DemoReplayError(
            f"metadata field '{field_name}' must contain non-negative integers."
        )
    return values


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DemoReplayError("optional schema text fields must be non-empty strings.")
    return value


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


def _finite_vector(
    value: object,
    *,
    size: int,
    field_name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise DemoReplayError(
            f"push_cube_to_target metadata {field_name} must contain "
            f"{size} finite values."
        )
    return vector


def _state_time(state: object) -> float | None:
    value = getattr(state, "time", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
