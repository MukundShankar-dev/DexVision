from __future__ import annotations

import json
from pathlib import Path

import pytest

from dexvision.evaluation.level4_button_learning import (
    run_button_learning_pilot,
    save_button_learning_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "level4_button_learning_pilot.yaml"
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"
RETARGETER_CONFIG = ROOT / "configs" / "level1_teleop.yaml"


def test_frozen_button_mlp_qualifies_on_twenty_held_out_resets(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    report = run_button_learning_pilot(
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
    assert report.held_out_success_rate >= 0.80
    assert report.held_out_success_count >= 16
    assert report.violation_totals == {
        "workspace": 0,
        "joint_limit": 0,
        "wrong_button_contact": 0,
        "unintended_contact": 0,
        "invalid_action": 0,
    }
    assert all(report.gate_results.values())
    assert report.passed is True
    assert report.failure_diagnosis is None
    assert report.recipe_change_count == 0
    assert report.data_increase_count == 0
    assert all(item.success for item in report.rollouts)
    assert all(item.task_success_observed for item in report.rollouts)
    assert all(item.terminal_reason == "completed" for item in report.rollouts)
    assert all(
        set(item.phase_counts) == {"approach", "fixture_contact", "retract"}
        for item in report.rollouts
    )

    output = save_button_learning_report(
        report, tmp_path / "button_learning_report.json"
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["config_digest"] == report.config_digest
    assert len(saved["rollouts"]) == 20
