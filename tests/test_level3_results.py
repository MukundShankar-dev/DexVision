from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs/level3/feasibility_v1/summary.json"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    assert isinstance(document, dict)
    return document


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def test_feasibility_decision_and_provenance_are_complete() -> None:
    summary = _load_json(SUMMARY_PATH)

    assert summary["version"] == "level3/feasibility-report-v1"
    assert summary["checkpoint"] == "Level 3.8"
    assert summary["decision"] == {
        "learning_pipeline": "go",
        "level2_policies_for_level5_qualification": "no_go",
        "level4_collection_and_specification": "go",
        "summary": summary["decision"]["summary"],
    }

    release = summary["dataset_release"]
    manifest_path = ROOT / release["manifest_path"]
    manifest = _load_json(manifest_path)
    assert _sha256(manifest_path) == release["manifest_sha256"]
    assert manifest["archive_sha256"] == release["archive_sha256"]
    dataset_summary_path = ROOT / release["dataset_summary_path"]
    if dataset_summary_path.is_file():
        assert _sha256(dataset_summary_path) == release["dataset_summary_sha256"]

    experiments = summary["experiments"]
    assert {experiment["id"] for experiment in experiments} == {
        "reach_v1_final_epoch",
        "reach_v2_best_validation",
        "button_v1_best_validation",
        "push_v1_best_validation",
    }
    for experiment in experiments:
        assert experiment["passed"] is False
        assert _is_sha256(experiment["dataset_digest"])
        assert _is_sha256(experiment["split_manifest_digest"])
        assert _is_sha256(experiment["checkpoint"]["sha256"])
        assert _is_sha256(experiment["evaluation_config"]["digest"])
        assert _is_sha256(experiment["rollout_report"]["sha256"])
        assert _is_sha256(experiment["rollout_report"]["rollout_config_digest"])

        training_config = ROOT / experiment["training_config"]["path"]
        evaluation_config = ROOT / experiment["evaluation_config"]["path"]
        assert _sha256(training_config) == experiment["training_config"][
            "file_sha256"
        ]
        assert _sha256(evaluation_config) == experiment["evaluation_config"][
            "digest"
        ]


@pytest.mark.skipif(
    not (ROOT / "outputs/level3/reach_rollout_v1/report.json").is_file(),
    reason="generated Level 3 run artifacts are not present in this checkout",
)
def test_compact_metrics_match_generated_source_reports() -> None:
    summary = _load_json(SUMMARY_PATH)

    for experiment in summary["experiments"]:
        checkpoint = ROOT / experiment["checkpoint"]["path"]
        rollout_path = ROOT / experiment["rollout_report"]["path"]
        rollout = _load_json(rollout_path)

        assert _sha256(checkpoint) == experiment["checkpoint"]["sha256"]
        assert _sha256(rollout_path) == experiment["rollout_report"]["sha256"]
        assert rollout["dataset_digest"] == experiment["dataset_digest"]
        assert rollout["protocol_digest"] == experiment["evaluation_config"][
            "digest"
        ]
        assert rollout["checkpoint_digest"] == experiment["checkpoint"]["sha256"]

        metrics = rollout["metrics"]
        compact_metrics = experiment["metrics"]
        training_key = (
            "training_target_success_rate"
            if experiment["task_id"] == "reach_touch_target"
            else "training_goal_success_rate"
        )
        held_out_key = (
            "held_out_target_success_rate"
            if experiment["task_id"] == "reach_touch_target"
            else "held_out_goal_success_rate"
        )
        final_error_key = (
            "mean_final_distance_m"
            if experiment["task_id"] == "reach_touch_target"
            else "mean_final_task_error"
        )
        assert compact_metrics["training_success_rate"] == metrics[training_key]
        assert compact_metrics["held_out_success_rate"] == metrics[held_out_key]
        assert compact_metrics["mean_final_task_error"] == metrics[final_error_key]
        for key in (
            "scenario_count",
            "mean_normalized_action_jerk",
            "invalid_action_count",
            "workspace_violation_count",
            "joint_limit_violation_count",
        ):
            assert compact_metrics[key] == metrics[key]

    diagnostics = summary["diagnostics"]
    diagnostics_path = ROOT / diagnostics["report_path"]
    assert _sha256(diagnostics_path) == diagnostics["report_sha256"]


def test_level4_plan_is_specific_and_stays_in_scope() -> None:
    summary = _load_json(SUMMARY_PATH)
    requirements = summary["level4_requirements"]
    plan = requirements["collection_plan"]

    assert plan["status"] == "provisional_until_level4.0_and_level4.3_freeze"
    assert plan["minimum_genuine_sessions"] == 3
    assert plan["required_total_minimum"] == 250
    assert plan["required_total_target_range"] == [250, 350]
    assert sum(
        group["minimum_accepted_episodes"] for group in plan["groups"]
    ) == 250
    assert {group["name"] for group in plan["groups"]} == {
        "reach_object_or_fixture_approach",
        "complete_pick_place_sequence",
        "push_to_zone",
        "button_press",
        "ordinary_failures_and_safe_corrections",
    }
    assert "separate bounded residual heads" in requirements[
        "model_recommendation_for_level5"
    ]["recommended_candidate"]
    assert "short-history model only after" in requirements[
        "model_recommendation_for_level5"
    ]["temporal_condition"]


def test_readable_report_and_plots_are_versioned() -> None:
    report = (ROOT / "docs/level3_results.md").read_text(encoding="utf-8")
    summary = _load_json(SUMMARY_PATH)

    assert "**No-go** for qualifying or deploying any policy" in report
    assert "Cross-session, cross-operator" in report
    assert "250–350" in report
    assert "No scope correction to `docs/progress_level_4.md` is needed" in report
    for relative_path in summary["plots"]:
        plot_path = ROOT / relative_path
        plot = plot_path.read_text(encoding="utf-8")
        assert plot.startswith("<svg")
        assert "role=\"img\"" in plot


def test_level38_completion_remains_recorded_during_level43() -> None:
    status = (ROOT / "docs/CURRENT_STATUS.md").read_text(encoding="utf-8")
    progress = (ROOT / "docs/progress_level_3.md").read_text(encoding="utf-8")

    assert "Level 4 — Comprehensive Multi-Session Dataset Collection" in status
    assert "`docs/progress_level_4.md`" in status
    assert "Level 3.8 — Feasibility Report and Level 4 Data Requirements" in status
    assert "## Last Completed Checkpoint\n\nLevel 4.3H" in status
    assert "## Next Target Checkpoint\n\nLevel 4.3I" in status
    assert "[x] Feasibility report defines the Level 4 data haul" in progress
    assert progress.count("[x] Every result traces to a config") == 1
