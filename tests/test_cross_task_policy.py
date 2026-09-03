from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.evaluate_policy import (
    ActionBounds,
    ManipulationBackendState,
    MujocoButtonRolloutBackend,
    MujocoPushRolloutBackend,
    evaluate_manipulation_policy,
    load_manipulation_evaluation_protocol,
)
from dexvision.learning.models import (
    BASE_ORIENTATION_ACTION_NAMES,
    BASE_POSITION_ACTION_NAMES,
)
from dexvision.learning.train_bc import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = (
    BASE_POSITION_ACTION_NAMES
    + BASE_ORIENTATION_ACTION_NAMES
    + ("finger_actuator_targets/finger",)
)


class _Policy:
    observation_names = ("state/value",)
    observation_schema_version = "level2/observation-layout-v2"
    action_schema_version = "level1.13/full-action-v1"
    dataset_action_names = ACTION_NAMES
    output_action_names = ACTION_NAMES
    checkpoint_digest = "test-checkpoint"
    dataset_digest = "test-dataset"
    split_manifest_digest = "test-split"
    experiment_config_digest = "test-config"
    schema_digest = "test-schema"
    selected_epoch = 2
    selected_validation_loss = 0.25

    def __init__(self, goal_names: tuple[str, ...]) -> None:
        self.goal_names = goal_names

    def predict(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        del observation, goal
        return np.asarray([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0])


class _Backend:
    max_episode_steps = 3
    observation_schema_version = "level2/observation-layout-v2"
    action_schema_version = "level1.13/full-action-v1"

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, scenario):
        del scenario
        self.steps = 0
        return ManipulationBackendState(False, None, 1.0, 0.0, 0)

    def observation(self, names, *, previous_action):
        del previous_action
        assert tuple(names) == ("state/value",)
        return np.asarray([self.steps], dtype=np.float64)

    def initial_action(self, action_names):
        assert tuple(action_names) == ACTION_NAMES
        return np.asarray([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0])

    def action_bounds(self, action_names):
        assert tuple(action_names) == ACTION_NAMES
        return ActionBounds(
            lower=np.asarray([-1.0] * len(ACTION_NAMES)),
            upper=np.asarray([1.0] * len(ACTION_NAMES)),
            workspace_indices=(0, 1, 2),
            joint_indices=(7,),
        )

    def step(self, action_names, action, *, n_steps):
        del action_names, action, n_steps
        self.steps += 1
        return ManipulationBackendState(
            self.steps >= 2,
            None,
            0.0,
            0.02,
            self.steps,
        )

    def close(self):
        pass


@pytest.mark.parametrize(
    ("config_name", "scenario_count", "goal_names"),
    (
        (
            "level3_button_evaluation.yaml",
            84,
            (
                "button_index",
                "button_position/x",
                "button_position/y",
                "button_position/z",
                "target_press_depth",
                "target_pressed_state",
                "approach_pose_present",
                "approach_pose/x",
                "approach_pose/y",
                "approach_pose/z",
            ),
        ),
        (
            "level3_push_evaluation.yaml",
            30,
            (
                "object_index",
                "target_index",
                "target_position/x",
                "target_position/y",
                "target_position/z",
                "target_radius",
                "initial_object_position/x",
                "initial_object_position/y",
                "initial_object_position/z",
                "approach_side/left",
                "approach_side/front",
                "approach_side/right",
            ),
        ),
    ),
)
def test_frozen_cross_task_matrix_runs_and_saves_failures_and_provenance(
    tmp_path: Path,
    config_name: str,
    scenario_count: int,
    goal_names: tuple[str, ...],
) -> None:
    protocol = load_manipulation_evaluation_protocol(ROOT / "configs" / config_name)
    report = evaluate_manipulation_policy(
        _Policy(goal_names),
        protocol,
        output_dir=tmp_path,
        backend_factory=_Backend,
        sim_steps_per_action=1,
    )

    assert len(report.results) == scenario_count
    assert report.metrics["training_goal_success_rate"] == 1.0
    assert report.metrics["held_out_goal_success_rate"] == 1.0
    assert report.selected_epoch == 2
    assert report.split_manifest_digest == "test-split"
    assert report.training_config_digest == "test-config"
    assert len(tuple((tmp_path / "trajectories").glob("*.npz"))) == scenario_count


def test_all_task_training_configs_share_model_optimizer_and_selection_version() -> None:
    paths = (
        ROOT / "configs/level3_reach_bc_v2.yaml",
        ROOT / "configs/level3_button_bc.yaml",
        ROOT / "configs/level3_push_bc.yaml",
    )
    configs = tuple(load_experiment_config(path) for path in paths)

    assert {config.skill_name for config in configs} == {
        "reach_touch_target",
        "button_press",
        "push_cube_to_target",
    }
    assert {config.version for config in configs} == {"level3/bc-training-v2"}
    assert len({config.model for config in configs}) == 1
    assert len({config.training for config in configs}) == 1
    assert all(config.best_checkpoint_name != config.checkpoint_name for config in configs)
    original_reach = load_experiment_config(ROOT / "configs/level3_bc.yaml")
    corrected_reach = configs[0]
    assert corrected_reach.dataset_root == original_reach.dataset_root
    assert corrected_reach.evaluation_config == original_reach.evaluation_config
    assert corrected_reach.observation_fields == original_reach.observation_fields
    assert corrected_reach.include_previous_action == original_reach.include_previous_action
    assert corrected_reach.output_action_names == original_reach.output_action_names
    assert corrected_reach.model == original_reach.model
    assert corrected_reach.training == original_reach.training


@pytest.mark.parametrize(
    ("protocol_name", "backend_type"),
    (
        ("level3_button_evaluation.yaml", MujocoButtonRolloutBackend),
        ("level3_push_evaluation.yaml", MujocoPushRolloutBackend),
    ),
)
def test_manipulation_mujoco_backend_smoke(protocol_name, backend_type) -> None:
    pytest.importorskip("mujoco")
    protocol = load_manipulation_evaluation_protocol(
        ROOT / "configs" / protocol_name
    )
    scenario = replace(protocol, scenarios=(protocol.scenarios[0],)).scenarios[0]
    backend = backend_type(ROOT / "assets/mujoco/task_board_scene.xml")
    try:
        state = backend.reset(scenario)
        assert state.step_count == 0
        assert np.isfinite(state.task_error)
    finally:
        backend.close()
