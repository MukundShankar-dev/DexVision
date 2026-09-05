from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from dexvision.evaluation.level4_button_learning import (
    ButtonRolloutResult,
    diagnose_button_rollout_failures,
)
from dexvision.learning.level4_lowdim import (
    BUTTON_ACTION_VERSION,
    BUTTON_DELTA_NAMES,
    BUTTON_OBSERVATION_NAMES,
    BUTTON_OBSERVATION_VERSION,
    ButtonActionAdapter,
    ButtonDeltaMLP,
    ButtonLearningError,
    button_episode_specs,
    load_button_learning_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "level4_button_learning_pilot.yaml"


def test_button_lowdim_contract_freezes_one_state_only_recipe() -> None:
    config, digest = load_button_learning_config(CONFIG)
    specs = button_episode_specs(config)

    assert len(digest) == 64
    assert len(specs) == 20
    assert sum(item.split == "train" for item in specs) == 14
    assert sum(item.split == "validation" for item in specs) == 3
    assert sum(item.split == "test" for item in specs) == 3
    assert len({item.episode_id for item in specs}) == 20
    assert {
        item.session_id: item.split for item in specs
    } == {
        "level43g_button_train": "train",
        "level43g_button_validation": "validation",
        "level43g_button_test": "test",
    }
    assert config["observation"]["schema_version"] == BUTTON_OBSERVATION_VERSION
    assert tuple(config["observation"]["fields"]) == BUTTON_OBSERVATION_NAMES
    assert config["observation"]["simulator_state_only"] is True
    assert all("rgb" not in name and "image" not in name for name in BUTTON_OBSERVATION_NAMES)
    assert config["action"]["schema_version"] == BUTTON_ACTION_VERSION
    assert tuple(config["action"]["output_fields"]) == BUTTON_DELTA_NAMES
    assert config["model"] == {
        "recipe_count": 1,
        "class": "small_mlp",
        "hidden_dims": [64, 64],
        "activation": "tanh",
    }
    assert config["change_control"]["allow_action_chunking"] is False
    assert config["change_control"]["allow_image_input"] is False


def test_button_pilot_rejects_recipe_or_data_increase_before_run(
    tmp_path: Path,
) -> None:
    config, _ = load_button_learning_config(CONFIG)
    changed = deepcopy(config)
    changed["model"]["hidden_dims"] = [128, 128]
    changed_path = tmp_path / "changed_model.yaml"
    changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ButtonLearningError, match="64x64"):
        load_button_learning_config(changed_path)

    changed = deepcopy(config)
    changed["dataset"]["successful_episodes"] = 21
    changed_path = tmp_path / "changed_data.yaml"
    changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ButtonLearningError, match="exactly 20"):
        load_button_learning_config(changed_path)


def test_button_mlp_and_adapter_keep_only_xyz_learned() -> None:
    model = ButtonDeltaMLP(
        input_dim=len(BUTTON_OBSERVATION_NAMES),
        hidden_dims=(64, 64),
        activation="tanh",
    )
    prediction = model(torch.zeros((2, len(BUTTON_OBSERVATION_NAMES))))
    assert prediction.shape == (2, 3)

    adapter = ButtonActionAdapter(
        finger_targets={"finger_a": 0.1, "finger_b": 0.2},
        fixed_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        workspace_min_m=(-0.18, -0.18, 0.08),
        workspace_max_m=(0.22, 0.18, 0.24),
        maximum_absolute_delta_by_phase_m={
            "approach": (0.012, 0.012, 0.012),
            "fixture_contact": (0.0025, 0.0, 0.0),
            "retract": (0.006, 0.003, 0.003),
        },
    )
    action, violated = adapter.expand(
        (0.0, 0.0, 0.15), (0.01, -0.01, 0.02), phase="approach"
    )
    assert violated is False
    assert action.base_position == pytest.approx([0.01, -0.01, 0.162])
    assert action.base_orientation_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert action.finger_targets == {"finger_a": 0.1, "finger_b": 0.2}

    clipped, violated = adapter.expand(
        (0.21, 0.0, 0.15), (0.02, 0.0, 0.0), phase="approach"
    )
    assert violated is True
    assert clipped.base_position[0] == pytest.approx(0.22)


def test_button_failure_is_diagnosed_before_recipe_changes() -> None:
    failure = ButtonRolloutResult(
        rollout_id="failure",
        coverage_cell="press_011_left_offset_test",
        seed=1000,
        success=False,
        task_success_observed=False,
        terminal_reason="timeout",
        steps=180,
        final_press_depth_m=0.0,
        workspace_violation_count=0,
        joint_limit_violation_count=0,
        joint_limit_names=(),
        wrong_button_contact_count=0,
        unintended_contact_count=0,
        invalid_action_count=0,
        phase_counts={"approach": 180},
    )
    assert (
        diagnose_button_rollout_failures((failure,))
        == "closed_loop_approach_did_not_reach_button"
    )
