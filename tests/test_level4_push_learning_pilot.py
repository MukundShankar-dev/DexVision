from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dexvision.evaluation.level4_push_learning import (
    run_push_learning_pilot,
    save_push_learning_report,
)
from dexvision.learning.level4_lowdim import (
    LowDimDeltaMLP,
    TaskLocalDeltaActionAdapter,
)
from dexvision.learning.level4_push_lowdim import (
    LOWDIM_ACTION_VERSION,
    PUSH_DELTA_NAMES,
    PUSH_OBSERVATION_NAMES,
    PushActionAdapter,
    load_push_learning_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "level4_push_learning_pilot.yaml"
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"
RETARGETER_CONFIG = ROOT / "configs" / "level1_teleop.yaml"


def test_push_config_reuses_frozen_lowdim_interface_without_chunking() -> None:
    config, digest = load_push_learning_config(CONFIG)

    assert len(digest) == 64
    assert config["interface"]["action_schema"] == LOWDIM_ACTION_VERSION
    assert config["action"]["schema_version"] == LOWDIM_ACTION_VERSION
    assert tuple(config["action"]["output_fields"]) == PUSH_DELTA_NAMES
    assert len(PUSH_OBSERVATION_NAMES) == 32
    assert config["interface"]["control_sim_steps"] == 17
    assert config["rollout"]["sim_steps_per_action"] == 17
    assert config["model"] == {
        "recipe_count": 1,
        "class": "small_mlp",
        "hidden_dims": [64, 64],
        "activation": "tanh",
    }
    assert config["change_control"]["allow_action_chunking"] is False


def test_shared_model_and_adapter_accept_push_dimensions_and_task_frame() -> None:
    model = LowDimDeltaMLP(
        input_dim=len(PUSH_OBSERVATION_NAMES), hidden_dims=(64, 64), activation="tanh"
    )
    assert tuple(model(torch.zeros((2, 32), dtype=torch.float32)).shape) == (2, 3)
    assert issubclass(PushActionAdapter, TaskLocalDeltaActionAdapter)


def test_frozen_push_mlp_qualifies_on_twenty_held_out_resets(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    report = run_push_learning_pilot(
        config_path=CONFIG,
        dataset_config=DATASET_CONFIG,
        workcell_config=WORKCELL_CONFIG,
        retargeter_config=RETARGETER_CONFIG,
    )

    assert report.collected_successes == 20
    assert report.session_split_episode_counts == {
        "test": 3,
        "train": 14,
        "validation": 3,
    }
    assert report.held_out_rollout_count == 20
    assert report.held_out_success_rate >= 0.70
    assert report.held_out_success_count >= 14
    assert report.violation_totals == {
        "board_exit": 0,
        "neighbor_disturbance": 0,
        "workspace": 0,
        "joint_limit": 0,
        "unintended_contact": 0,
        "object_tipped": 0,
        "invalid_action": 0,
    }
    assert all(report.gate_results.values())
    assert report.passed is True
    assert report.failure_diagnosis is None
    assert report.recipe_change_count == 0
    assert report.data_increase_count == 0
    assert report.action_chunking_used is False
    assert report.action_chunking_evidence == (
        "none_single_step_interface_tested_directly"
    )
    assert all(item.success for item in report.rollouts)
    assert all(item.task_success_observed for item in report.rollouts)
    assert all(item.terminal_reason == "completed" for item in report.rollouts)
    assert {item.family for item in report.rollouts} == {"cuboid", "flat_puck"}
    assert all(
        set(item.phase_counts)
        == {"approach", "push_contact", "settle", "retract"}
        for item in report.rollouts
    )

    output = save_push_learning_report(report, tmp_path / "push_learning_report.json")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["config_digest"] == report.config_digest
    assert len(saved["rollouts"]) == 20
