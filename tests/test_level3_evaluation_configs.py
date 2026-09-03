from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
import yaml

from dexvision.learning.splits import (
    EpisodeDescriptor,
    deterministic_episode_split,
    split_config_from_mapping,
)
from dexvision.sim.tasks import ButtonPressConfig, PushCubeConfig


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATHS = {
    "button": ROOT / "configs" / "level3_button_evaluation.yaml",
    "push": ROOT / "configs" / "level3_push_evaluation.yaml",
}
DATASET_PATHS = {
    "button": ROOT / "configs" / "button_press_dataset.yaml",
    "push": ROOT / "configs" / "push_cube_dataset.yaml",
}
EXPECTED_PROTOCOL_DIGESTS = {
    "button": "c2342bfaf0fb84a7cc0602e04c8f17760d2953619c4275ddd33ae67804a698d2",
    "push": "5cd217efb303d50afbab4d77aaf59efd7f92b736ce7e0c5ae7dfe25e5d0625ee",
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    assert isinstance(document, dict)
    return document


def _resolve_callable(reference: str):
    module_name, attribute = reference.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute)


def _descriptor(episode_id: str, goal_id: str) -> EpisodeDescriptor:
    return EpisodeDescriptor(
        episode_id=episode_id,
        goal_id=goal_id,
        data_digest=f"digest-{episode_id}",
        action_schema_version="level1.13/full-action-v1",
        observation_schema_version="level2/observation-layout-v2",
    )


@pytest.mark.parametrize("name", ("button", "push"))
def test_offline_split_is_deterministic_and_keeps_episodes_whole(name: str) -> None:
    protocol = _load_yaml(PROTOCOL_PATHS[name])
    split = protocol["offline_split"]
    config = split_config_from_mapping(split, version=protocol["version"])
    episodes = tuple(
        _descriptor(f"{goal_id}-{index}", goal_id)
        for goal_id in protocol["training_goals"]
        for index in range(5)
    )

    first = deterministic_episode_split(episodes, config)
    second = deterministic_episode_split(reversed(episodes), config)

    assert first.assignment_by_episode() == second.assignment_by_episode()
    assert len(first.assignments) == len(episodes)
    assert len(first.assignment_by_episode()) == len(episodes)
    assert {item.split for item in first.assignments} == {"train", "validation"}
    assert split["group_by_episode"] is True
    assert split["normalization_source"] == "train_only"
    assert split["test_fraction"] == 0.0
    assert split["claim_cross_session_generalization"] is False


@pytest.mark.parametrize("name", ("button", "push"))
def test_reserved_goals_are_unchanged_and_disjoint(name: str) -> None:
    protocol = _load_yaml(PROTOCOL_PATHS[name])
    dataset = _load_yaml(DATASET_PATHS[name])

    assert protocol["dataset_config_version"] == dataset["version"]
    assert protocol["training_goals"] == dataset["training_goals"]
    assert (
        protocol["held_out_rollout_goals"]
        == dataset["held_out_evaluation_goals"]
    )
    assert set(protocol["training_goals"]).isdisjoint(
        protocol["held_out_rollout_goals"]
    )


def test_button_matrix_metrics_and_gates_are_executable() -> None:
    protocol = _load_yaml(PROTOCOL_PATHS["button"])
    metrics = protocol["terminal_metrics"]
    offsets = protocol["initial_base_position_offsets_m"]
    gates = protocol["acceptance_gates"]
    task_config = ButtonPressConfig()

    assert protocol["version"] == "level3/button-evaluation-v1"
    assert len(protocol["training_goals"]) * len(offsets) == 63
    assert len(protocol["held_out_rollout_goals"]) * len(offsets) == 21
    assert len({tuple(value) for value in offsets.values()}) == 7
    assert max(abs(value) for offset in offsets.values() for value in offset) == 0.005
    assert metrics["required_dwell_steps"] == task_config.success_dwell_steps

    success = _resolve_callable(metrics["success_callable"])
    assert success(
        press_depth_m=0.011,
        target_press_depth_m=0.011,
        button_pressed=True,
        target_pressed_state=True,
        dwell_steps=task_config.success_dwell_steps,
        required_dwell_steps=metrics["required_dwell_steps"],
    )
    assert not success(
        press_depth_m=0.010,
        target_press_depth_m=0.011,
        button_pressed=False,
        target_pressed_state=True,
        dwell_steps=task_config.success_dwell_steps,
        required_dwell_steps=metrics["required_dwell_steps"],
    )
    assert gates["minimum_training_goal_success_rate"] == 0.80
    assert gates["minimum_held_out_goal_success_rate"] == 0.80
    assert gates["maximum_mean_final_press_depth_shortfall_m"] == 0.001
    assert gates["maximum_mean_action_jerk"] == 0.20
    assert all(
        gates[key] == 0
        for key in (
            "maximum_invalid_action_count",
            "maximum_workspace_violation_count",
            "maximum_joint_limit_violation_count",
        )
    )


def test_push_matrix_metrics_and_gates_are_executable() -> None:
    protocol = _load_yaml(PROTOCOL_PATHS["push"])
    metrics = protocol["terminal_metrics"]
    offsets = protocol["initial_state_position_offsets_m"]
    geometry = protocol["task_geometry"]
    gates = protocol["acceptance_gates"]
    task_config = PushCubeConfig()

    assert protocol["version"] == "level3/push-evaluation-v1"
    assert len(protocol["training_goals"]) * len(offsets) == 15
    assert len(protocol["held_out_rollout_goals"]) * len(offsets) == 15
    assert len({tuple(value) for value in offsets.values()}) == 5
    assert all(offset[2] == 0.0 for offset in offsets.values())
    assert max(abs(value) for offset in offsets.values() for value in offset) == 0.01
    assert geometry["target_radius_m"] == task_config.target_radius_m
    assert geometry["success_dwell_steps"] == task_config.success_dwell_steps
    assert geometry["max_episode_steps"] == task_config.max_episode_steps
    assert tuple(geometry["object_workspace_min_m"]) == task_config.workspace_min
    assert tuple(geometry["object_workspace_max_m"]) == task_config.workspace_max

    distance = _resolve_callable(metrics["distance_callable"])
    success = _resolve_callable(metrics["success_callable"])
    failure = _resolve_callable(metrics["failure_callable"])
    assert distance((-0.09, 0.0, -0.015), (0.09, 0.0, -0.015)) == pytest.approx(
        0.18
    )
    assert success(
        distance_m=0.03,
        dwell_steps=geometry["success_dwell_steps"],
        distance_threshold_m=geometry["target_radius_m"],
        required_dwell_steps=geometry["success_dwell_steps"],
    )
    assert (
        failure(
            object_position=(0.19, 0.0, -0.015),
            step_count=1,
            max_episode_steps=geometry["max_episode_steps"],
            workspace_min=geometry["object_workspace_min_m"],
            workspace_max=geometry["object_workspace_max_m"],
        )
        == "object_workspace_bounds"
    )
    assert gates["minimum_training_goal_success_rate"] == 0.80
    assert gates["minimum_held_out_goal_success_rate"] == 0.80
    assert gates["maximum_mean_final_planar_distance_m"] == geometry[
        "target_radius_m"
    ]
    assert gates["maximum_mean_action_jerk"] == 0.20
    assert all(
        gates[key] == 0
        for key in (
            "maximum_invalid_action_count",
            "maximum_object_workspace_violation_count",
            "maximum_joint_limit_violation_count",
        )
    )


@pytest.mark.parametrize("name", ("button", "push"))
def test_v1_protocol_bytes_are_pinned_and_require_a_new_version(name: str) -> None:
    path = PROTOCOL_PATHS[name]
    protocol = _load_yaml(path)
    freeze = protocol["freeze_policy"]

    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_DIGESTS[
        name
    ]
    assert freeze["status"] == "frozen_before_training"
    assert (
        freeze["after_first_evaluation"]
        == "preserve_this_file_and_create_a_new_version"
    )
    assert "acceptance_gates" in freeze["forbidden_without_version_bump"]
