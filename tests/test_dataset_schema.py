from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dexvision.logging.dataset_schema import (
    ActionSchema,
    DemoEpisode,
    DemoSchemaError,
    ObservationSchema,
    extract_observations,
    validate_demo,
)
from dexvision.logging.demo_logger import build_level2_observation_schema


def _action_schema() -> ActionSchema:
    return ActionSchema(
        version="level1.13/full-action-v1",
        base_position_target=(0, 3),
        base_orientation_target=(3, 7),
        finger_actuator_targets=(7, 12),
        representation_notes={"base_orientation_target": "MuJoCo wxyz quaternion"},
    )


def _observation_schema() -> ObservationSchema:
    fields = (
        "robot_qpos",
        "robot_qvel",
        "base_position",
        "base_orientation",
        "finger_joint_positions",
        "finger_joint_velocities",
        "tracking_quality",
        "object_state",
        "task_state",
        "target_state",
        "success_metric_inputs",
    )
    return ObservationSchema(
        version="level2/observation-v1",
        fields=fields,
        shapes={
            "robot_qpos": (30,),
            "robot_qvel": (30,),
            "base_position": (3,),
            "base_orientation": (4,),
            "finger_joint_positions": (24,),
            "finger_joint_velocities": (24,),
            "tracking_quality": (6,),
            "object_state": (13,),
            "task_state": (8,),
            "target_state": (7,),
            "success_metric_inputs": (4,),
        },
        optional_fields=(),
    )


def _metadata() -> dict:
    return {
        "skill_name": "push_cube",
        "task_name": "Push cube to target",
        "task_id": "push_cube_to_target",
        "episode_id": "demo_0001",
        "action_schema_version": "level1.13/full-action-v1",
        "observation_schema_version": "level2/observation-v1",
        "robot_model": "assets/mujoco/hand_scene.xml",
        "retargeter_config": "configs/level1_teleop.yaml",
        "control_rate_hz": 30.0,
        "teleop_config": {"base_control": True, "finger_control": True},
        "task_config": {
            "required_objects": ("cube",),
            "requires_task_state": True,
            "requires_success_metric_inputs": True,
            "required_observation_fields": (
                "object_state",
                "task_state",
                "target_state",
                "success_metric_inputs",
            ),
            "success_metric_fields": ("cube_position", "target_position"),
        },
    }


def _valid_episode(time_steps: int = 4) -> DemoEpisode:
    actions = np.zeros((time_steps, _action_schema().action_dim), dtype=np.float64)
    actions[:, 3] = 1.0
    return DemoEpisode(
        metadata=_metadata(),
        landmarks=np.zeros((time_steps, 21, 3), dtype=np.float64),
        features=np.ones((time_steps, 16), dtype=np.float64),
        actions=actions,
        robot_states=np.ones((time_steps, 64), dtype=np.float64),
        object_states=np.ones((time_steps, 13), dtype=np.float64),
        task_states=np.ones((time_steps, 8), dtype=np.float64),
        tracking_quality=np.asarray(
            [[1.0, 1.0, 0.95, 0.90, 0.0, 0.0] for _ in range(time_steps)],
            dtype=np.float64,
        ),
        timestamps=np.arange(time_steps, dtype=np.float64) / 30.0,
        success=True,
    )


def _validate(episode: DemoEpisode) -> None:
    validate_demo(
        episode,
        action_schema=_action_schema(),
        observation_schema=_observation_schema(),
    )


def test_valid_demo_passes_validation_and_action_schema_splits_full_command() -> None:
    episode = _valid_episode()

    _validate(episode)

    action_parts = _action_schema().split(episode.actions)
    assert action_parts["base_position_target"].shape == (4, 3)
    assert action_parts["base_orientation_target"].shape == (4, 4)
    assert action_parts["finger_actuator_targets"].shape == (4, 5)


def test_invalid_time_dimensions_are_caught() -> None:
    episode = _valid_episode()
    episode.features = episode.features[:-1]

    with pytest.raises(DemoSchemaError, match="features time dimension"):
        _validate(episode)


def test_nans_in_required_arrays_are_caught() -> None:
    episode = _valid_episode()
    episode.actions[2, 0] = np.nan

    with pytest.raises(DemoSchemaError, match="actions must not contain"):
        _validate(episode)


def test_missing_metadata_is_caught() -> None:
    episode = _valid_episode()
    del episode.metadata["robot_model"]

    with pytest.raises(DemoSchemaError, match="metadata is missing required fields"):
        _validate(episode)


def test_full_level_1_13_action_schema_is_required() -> None:
    episode = _valid_episode()
    bad_schema = replace(_action_schema(), base_orientation_target=(3, 6))

    with pytest.raises(DemoSchemaError, match="base_orientation_target must have length 4"):
        validate_demo(
            episode,
            action_schema=bad_schema,
            observation_schema=_observation_schema(),
        )


def test_actions_must_have_enough_columns_for_full_action_schema() -> None:
    episode = _valid_episode()
    episode.actions = episode.actions[:, :7]

    with pytest.raises(DemoSchemaError, match="full Level 1.13 action schema"):
        _validate(episode)


def test_tracking_quality_is_validated() -> None:
    episode = _valid_episode()
    episode.tracking_quality[1, 2] = 1.5

    with pytest.raises(DemoSchemaError, match="tracking_quality confidence"):
        _validate(episode)


def test_skill_and_task_identifiers_are_required() -> None:
    episode = _valid_episode()
    episode.metadata["task_id"] = ""

    with pytest.raises(DemoSchemaError, match="task_id"):
        _validate(episode)


def test_optional_gesture_label_is_validated_when_present() -> None:
    episode = _valid_episode()
    episode.metadata["gesture_label"] = "not_a_gesture"

    with pytest.raises(DemoSchemaError, match="gesture_label"):
        _validate(episode)


def test_success_metric_inputs_are_validated_when_required_by_task() -> None:
    episode = _valid_episode()
    episode.task_states = None

    with pytest.raises(DemoSchemaError, match="task_states are required"):
        _validate(episode)


def test_observation_schema_must_include_success_metric_inputs_when_required() -> None:
    episode = _valid_episode()
    observation_schema = _observation_schema()
    fields = tuple(field for field in observation_schema.fields if field != "success_metric_inputs")
    shapes = dict(observation_schema.shapes)
    del shapes["success_metric_inputs"]
    observation_schema = ObservationSchema(
        version=observation_schema.version,
        fields=fields,
        shapes=shapes,
        optional_fields=observation_schema.optional_fields,
    )

    with pytest.raises(DemoSchemaError, match="success_metric_inputs"):
        validate_demo(
            episode,
            action_schema=_action_schema(),
            observation_schema=observation_schema,
        )


def _executable_observation_schema() -> ObservationSchema:
    return build_level2_observation_schema(
        robot_qpos_dim=4,
        robot_qvel_dim=4,
        finger_target_dim=2,
        tracking_quality_dim=6,
        robot_qpos_names=("base/x", "base/qw", "joint_a", "joint_b"),
        robot_qvel_names=("base/vx", "base/wx", "joint_a", "joint_b"),
        actuator_names=("actuator_b", "actuator_a"),
        finger_joint_qpos_indices=(2, 3),
        finger_joint_qvel_indices=(2, 3),
        finger_joint_names=("joint_a", "joint_b"),
        tracking_quality_names=(
            "detected",
            "handedness",
            "tracking_confidence",
            "feature_confidence",
            "dropped_frame",
            "reacquired",
        ),
        object_state_dim=3,
        task_state_dim=5,
        target_state_dim=3,
        success_metric_dim=2,
    )


def _executable_episode() -> DemoEpisode:
    time_steps = 2
    episode = _valid_episode(time_steps)
    episode.metadata["observation_schema_version"] = _executable_observation_schema().version
    episode.robot_states = np.arange(time_steps * 17, dtype=np.float64).reshape(time_steps, 17)
    episode.object_states = np.arange(time_steps * 3, dtype=np.float64).reshape(time_steps, 3)
    episode.task_states = np.arange(time_steps * 5, dtype=np.float64).reshape(time_steps, 5)
    return episode


def test_executable_layout_extracts_every_declared_field() -> None:
    schema = _executable_observation_schema()
    episode = _executable_episode()

    validate_demo(
        episode,
        action_schema=_action_schema(),
        observation_schema=schema,
    )
    observations = extract_observations(episode, observation_schema=schema)

    assert tuple(observations) == schema.fields
    assert observations["robot_qpos"][0] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert observations["robot_qvel"][0] == pytest.approx([4.0, 5.0, 6.0, 7.0])
    assert observations["actuator_controls"][0] == pytest.approx([8.0, 9.0])
    assert observations["base_position"][0] == pytest.approx([10.0, 11.0, 12.0])
    assert observations["base_orientation"][0] == pytest.approx([13.0, 14.0, 15.0, 16.0])
    assert observations["finger_joint_positions"][0] == pytest.approx([2.0, 3.0])
    assert observations["finger_joint_velocities"][0] == pytest.approx([6.0, 7.0])
    assert observations["tracking_quality"].shape == (2, 6)
    assert observations["object_state"].shape == (2, 3)
    assert observations["task_state"].shape == (2, 5)
    assert observations["target_state"].shape == (2, 3)
    assert observations["success_metric_inputs"].shape == (2, 2)
    for layout in schema.layouts.values():
        assert layout.units
        assert layout.coordinate_frame
        assert layout.normalization


def test_executable_layout_validates_dense_array_widths() -> None:
    episode = _executable_episode()
    episode.robot_states = episode.robot_states[:, :-1]

    with pytest.raises(DemoSchemaError, match="robot_states width"):
        validate_demo(
            episode,
            action_schema=_action_schema(),
            observation_schema=_executable_observation_schema(),
        )


def test_executable_layout_preserves_named_mujoco_and_actuator_order() -> None:
    schema = _executable_observation_schema()

    assert schema.layouts["robot_qpos"].names == (
        "base/x",
        "base/qw",
        "joint_a",
        "joint_b",
    )
    assert schema.layouts["robot_qvel"].names[-2:] == ("joint_a", "joint_b")
    assert schema.layouts["actuator_controls"].names == ("actuator_b", "actuator_a")
    assert schema.layouts["finger_joint_positions"].column_indices == (2, 3)
    assert schema.layouts["finger_joint_velocities"].column_indices == (6, 7)


def test_optional_object_and_task_sources_have_explicit_absence_rules() -> None:
    schema = _executable_observation_schema()
    episode = _executable_episode()
    episode.metadata["task_config"] = {
        "required_objects": (),
        "requires_task_state": False,
        "requires_success_metric_inputs": False,
        "required_observation_fields": (),
    }
    episode.object_states = None
    episode.task_states = None

    observations = extract_observations(episode, observation_schema=schema)

    assert observations["object_state"] is None
    assert observations["task_state"] is None
    assert schema.layouts["object_state"].absence_rule
    assert schema.layouts["task_state"].absence_rule
