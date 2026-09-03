from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = ROOT / "docs/level3_temporal_baseline_decision.json"
LOCAL_SOURCE_PATHS = (
    ROOT / "outputs/level3/diagnostics_v1/report.json",
    ROOT / "outputs/level3/reach_rollout_v2/report.json",
    ROOT / "outputs/level3/button_rollout_v1/report.json",
    ROOT / "outputs/level3/push_rollout_v1/report.json",
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    assert isinstance(document, dict)
    return document


def test_temporal_trigger_is_not_promoted_from_level36_hypotheses() -> None:
    decision = _load_json(DECISION_PATH)

    assessment = decision["trigger_assessment"]
    assert decision["decision"] == "not_justified"
    assert assessment == {
        "measured_compounding_error": False,
        "measured_temporal_ambiguity": False,
        "missing_recovery_coverage_status": "unproven_hypothesis",
        "required_evidence": "measured_temporal_ambiguity_or_compounding_error",
        "trigger_met": False,
        "unproven_hypotheses": [
            "compounding_error",
            "missing_recovery_coverage",
        ],
    }


@pytest.mark.skipif(
    any(not path.is_file() for path in LOCAL_SOURCE_PATHS),
    reason="generated Level 3 source reports are not present in this checkout",
)
def test_decision_metrics_match_versioned_source_reports() -> None:
    decision = _load_json(DECISION_PATH)
    source_reports = decision["source_reports"]

    for source in source_reports.values():
        source_path = ROOT / source["path"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]

    diagnostics = _load_json(ROOT / source_reports["diagnostics"]["path"])
    hypotheses = " ".join(diagnostics["conclusions"]["hypotheses"]).lower()
    measured = " ".join(diagnostics["conclusions"]["measured"]).lower()
    assert "compounding error" in hypotheses
    assert "missing recovery coverage" in hypotheses
    assert "compounding error" not in measured
    assert "temporal ambiguity" not in measured

    rows = {row["experiment_id"]: row for row in diagnostics["summary_table"]}
    coupling = decision["measured_findings"]["action_space_coupling"]
    assert coupling["button_full_held_out_success_rate"] == rows[
        "button_press/full"
    ]["held_out_success_rate"]
    assert coupling["button_base_only_held_out_success_rate"] == rows[
        "button_press/base_only"
    ]["held_out_success_rate"]
    assert coupling["button_full_joint_limit_violation_count"] == rows[
        "button_press/full"
    ]["joint_limit_violation_count"]
    assert coupling["button_base_only_joint_limit_violation_count"] == rows[
        "button_press/base_only"
    ]["joint_limit_violation_count"]
    assert coupling["reach_full_mean_final_task_error_m"] == rows[
        "reach_touch_target/full"
    ]["mean_final_task_error"]
    assert coupling["reach_base_only_mean_final_task_error_m"] == rows[
        "reach_touch_target/base_only"
    ]["mean_final_task_error"]

    source_keys = {
        "reach_touch_target": "reach_rollout",
        "button_press": "button_rollout",
        "push_cube_to_target": "push_rollout",
    }
    for task_id, source_key in source_keys.items():
        rollout = _load_json(ROOT / source_reports[source_key]["path"])
        recorded_safety = decision["measured_findings"]["rollout_safety"][task_id]
        for metric_name, value in recorded_safety.items():
            assert value == rollout["metrics"][metric_name]

        recorded_mismatch = decision["measured_findings"][
            "offline_to_rollout_mismatch"
        ][task_id]
        assert recorded_mismatch["selected_validation_loss"] == rollout[
            "selected_validation_loss"
        ]
        success_key = (
            "training_target_success_rate"
            if task_id == "reach_touch_target"
            else "training_goal_success_rate"
        )
        held_out_key = (
            "held_out_target_success_rate"
            if task_id == "reach_touch_target"
            else "held_out_goal_success_rate"
        )
        assert recorded_mismatch["training_success_rate"] == rollout["metrics"][
            success_key
        ]
        assert recorded_mismatch["held_out_success_rate"] == rollout["metrics"][
            held_out_key
        ]


def test_not_justified_decision_adds_no_temporal_model() -> None:
    decision = _load_json(DECISION_PATH)

    assert decision["implementation"] == {
        "gru_added": False,
        "sequence_dataset_added": False,
        "temporal_training_run_added": False,
    }
    assert not (ROOT / "dexvision/learning/sequence_datasets.py").exists()
    assert not (ROOT / "dexvision/learning/temporal_models.py").exists()
