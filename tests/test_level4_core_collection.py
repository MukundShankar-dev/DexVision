from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dexvision.apps import record_demo, run_level1_teleop
from dexvision.evaluation.dataset_coverage import summarize_level4_coverage
from dexvision.features.hand_features import no_hand_features
from dexvision.logging.level4_collection import (
    PilotReview,
    build_level4_core_collection_plan,
    save_pilot_review,
)
from dexvision.logging.session_manifest import RecordingSession, append_session_manifest
from dexvision.retargeting.curl_retargeter import CurlRetargeter, load_curl_retargeter_config
from dexvision.sim.level4_expert import SafeWaypointReachConfig, SafeWaypointReachExpert


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "level4_dataset.yaml"
PICK_PLACE_PHASES = (
    "approach",
    "acquire",
    "lift",
    "stabilize",
    "transport",
    "place",
    "release",
    "settle",
    "retract",
)


def _accepted_review(episode_id: str) -> PilotReview:
    return PilotReview(
        episode_id=episode_id,
        schema_validation=True,
        timestamp_alignment=True,
        headless_replay=True,
        terminal_metric_recomputation=True,
        recomputed_success=True,
        operator_label_agreement=True,
        quality_thresholds=True,
        coverage_assignment=True,
        split_session_leakage_audit=True,
        expert_accepted=True,
    )


def _typed_goal(cell: dict[str, object], skill: str) -> dict[str, object]:
    if skill == "reach_object":
        return {"entity_id": cell["entity_id"]}
    if skill == "push_object_to_target":
        return {"object_id": cell["object_id"], "target_zone": cell["target_id"]}
    return {
        "button_id": cell["button_id"],
        "target_press_depth_m": cell["target_depth_m"],
    }


def _write_complete_core_dataset(dataset_dir: Path) -> None:
    plan = build_level4_core_collection_plan(CONFIG_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in config["coverage_cells"]}
    slot_split = {
        "session_a": "train",
        "session_b": "train",
        "session_c": "validation",
        "session_d": "test",
    }
    for slot, split in slot_split.items():
        append_session_manifest(
            dataset_dir / "session_manifest.json",
            RecordingSession(
                recording_session_id=slot,
                operator_id="operator_local_01",
                split=split,
                process_start_timestamp=f"2026-09-05T1{len(slot)}:00:00Z",
                reset_seed=len(slot),
                calibration_record_digest=f"sha256:{slot}",
            ),
        )
    for item in plan:
        episode_id = f"level44_{item.sequence:06d}"
        episode_dir = (
            dataset_dir / item.session_slot / f"episode_{item.sequence:06d}"
        )
        episode_dir.mkdir(parents=True)
        phases = (
            ("approach", "push_contact", "settle", "retract")
            if item.skill_name == "push_object_to_target"
            else (
                ("approach", "fixture_contact", "retract")
                if item.skill_name == "press_button"
                else ("approach", "retract")
            )
        )
        goal = _typed_goal(cells[item.coverage_cell_id], item.skill_name)
        metadata = {
            "episode_schema_version": "level4/episode-v1",
            "episode_id": episode_id,
            "recording_session_id": item.session_slot,
            "skill_name": item.skill_name,
            "source": item.source,
            "goal_condition_id": item.coverage_cell_id,
            "typed_goal": goal,
            "object_instance_ids": (
                [goal["object_id"]] if "object_id" in goal else []
            ),
            "phase_intervals": [
                {"phase": phase, "start_frame": index, "end_frame": index + 1}
                for index, phase in enumerate(phases)
            ],
            "collection_duration_seconds": 1.0,
            "success": True,
        }
        (episode_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        np.save(episode_dir / "timestamps.npy", np.arange(len(phases), dtype=float))
        np.save(episode_dir / "online_phases.npy", np.asarray(phases))
        np.save(episode_dir / "audited_phases.npy", np.asarray(phases))
        save_pilot_review(episode_dir, _accepted_review(episode_id))


def test_core_plan_expands_only_reach_push_and_button_minima() -> None:
    plan = build_level4_core_collection_plan(CONFIG_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert len(plan) == 60
    assert {item.data_group for item in plan} == {"reach", "push", "button"}
    assert sum(item.source == "teleoperation" for item in plan) == 0
    assert sum(item.source == "scripted" for item in plan) == 60
    assert {item.session_slot for item in plan if item.split == "train"} == {
        "session_a",
        "session_b",
    }
    assert {item.session_slot for item in plan if item.split == "validation"} == {
        "session_c"
    }
    assert {item.session_slot for item in plan if item.split == "test"} == {
        "session_d"
    }
    assert all(item.data_group != "pick_place" for item in plan)
    assert config["pilot"]["scripted_push"]["push_step_m"] == pytest.approx(0.002)
    assert config["level4_4_core_collection"]["scripted_expert_overrides"][
        "scripted_push"
    ]["push_step_m"] == pytest.approx(0.001)
    assert config["level4_4_core_collection"]["required_source_policy"] == (
        "scripted_only"
    )
    assert {
        item.coverage_cell_id: item.seed
        for item in plan
        if item.coverage_cell_id
        in config["level4_4_core_collection"]["seed_override_by_cell"]
    } == config["level4_4_core_collection"]["seed_override_by_cell"]


def test_core_plan_uses_qualified_seeds_only_for_held_out_push_cells() -> None:
    plan = build_level4_core_collection_plan(CONFIG_PATH)
    by_sequence = {item.sequence: item for item in plan}

    assert [by_sequence[index].seed for index in range(15, 21)] == list(
        range(46000, 46006)
    )
    assert [by_sequence[index].seed for index in range(37, 41)] == [
        200,
        1004,
        201,
        202,
    ]
    assert [by_sequence[index].seed for index in range(57, 61)] == list(
        range(46010, 46014)
    )


def test_complete_core_dataset_passes_level4_4_gates(tmp_path: Path) -> None:
    _write_complete_core_dataset(tmp_path)

    report = summarize_level4_coverage(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )["level4_4_core_collection"]

    assert report["accepted_episode_count"] == 60
    assert report["coverage_matrix"]["cell_count"] == 32
    assert report["coverage_matrix"]["complete_cell_count"] == 32
    assert report["source_requirements"] == {
        "scripted": {"observed": 60, "minimum": 60, "passed": True},
    }
    assert report["session_balance_passed"] is True
    assert report["target_balance"]["passed"] is True
    assert report["test_isolation_passed"] is True
    assert report["issues"] == []
    assert report["checkpoint_complete"] is True


def test_optional_teleoperation_episode_is_preserved_but_not_required(
    tmp_path: Path,
) -> None:
    _write_complete_core_dataset(tmp_path)
    episode_id = "optional_teleoperation"
    episode_dir = tmp_path / "session_a" / "episode_999999"
    episode_dir.mkdir(parents=True)
    metadata = {
        "episode_schema_version": "level4/episode-v1",
        "episode_id": episode_id,
        "recording_session_id": "session_a",
        "skill_name": "reach_object",
        "source": "teleoperation",
        "goal_condition_id": "reach_block_small_interior",
        "typed_goal": {"entity_id": "block_small"},
        "object_instance_ids": [],
        "phase_intervals": [
            {"phase": "approach", "start_frame": 0, "end_frame": 1},
            {"phase": "retract", "start_frame": 1, "end_frame": 2},
        ],
        "collection_duration_seconds": 1.0,
        "success": True,
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    np.save(episode_dir / "timestamps.npy", np.arange(2, dtype=float))
    np.save(episode_dir / "online_phases.npy", np.asarray(("approach", "retract")))
    np.save(episode_dir / "audited_phases.npy", np.asarray(("approach", "retract")))
    save_pilot_review(episode_dir, _accepted_review(episode_id))

    report = summarize_level4_coverage(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )["level4_4_core_collection"]

    assert report["accepted_episode_count"] == 60
    assert report["nonrequired_source_episode_count"] == 1
    assert report["source_requirements"] == {
        "scripted": {"observed": 60, "minimum": 60, "passed": True}
    }
    assert report["issues"] == []
    assert report["checkpoint_complete"] is True


def test_core_recorder_can_enforce_the_frozen_source_before_writing(
    tmp_path: Path,
) -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "reach_object",
            "--source",
            "teleoperation",
            "--session-id",
            "level44_bad_source",
            "--operator-id",
            "scripted_expert_v1",
            "--session-split",
            "train",
            "--goal-condition-id",
            "reach_block_small_interior",
            "--level4-dataset-dir",
            str(tmp_path),
            "--level4-dataset-config",
            str(CONFIG_PATH),
            "--enforce-frozen-cell-owner",
        ]
    )

    with pytest.raises(ValueError, match="requires source 'scripted'"):
        record_demo._prepare_level4_workcell_recording(args)
    assert not (tmp_path / "session_manifest.json").exists()


def test_core_plan_cli_is_read_only(tmp_path: Path, capsys) -> None:
    result = record_demo.main(
        [
            "--task",
            "level4_workcell",
            "--print-level4-core-plan",
            "--level4-dataset-config",
            str(CONFIG_PATH),
            "--level4-dataset-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Total Level 4.4 accepted episodes required: 60" in output
    assert "pick_place_sequence" not in output
    assert list(tmp_path.iterdir()) == []


def test_scripted_fixture_reach_uses_a_safe_precontact_pose() -> None:
    pytest.importorskip("mujoco")
    with record_demo.WorkcellPilotTask(
        workcell_config=ROOT / "configs" / "workcell.yaml",
        dataset_config=CONFIG_PATH,
        skill_name="reach_object",
        goal_condition_id="reach_start_button_centered",
        seed=45000,
    ) as task:
        retargeter = CurlRetargeter.from_mapping(
            load_curl_retargeter_config(ROOT / "configs" / "level1_teleop.yaml")
        )
        open_targets = run_level1_teleop.build_full_hand_targets(
            retargeter, no_hand_features()
        )
        reach_config = record_demo._level4_scripted_expert_settings(
            SimpleNamespace(enforce_frozen_cell_owner=True),
            task,
            "scripted_reach",
        )
        finger_targets = record_demo._scripted_finger_synergy_targets(
            retargeter,
            open_targets,
            reach_config["fixed_finger_synergy_margin"],
        )
        expert = SafeWaypointReachExpert(
            finger_targets=finger_targets,
            config=SafeWaypointReachConfig.from_mapping(reach_config),
        )

        expert.reset(task, task.initial_world_state)

        assert task.goal["approach_pose"][:3] == pytest.approx(
            [-0.04, -0.121, 0.165]
        )
        assert expert.validation is not None
        assert expert.validation.valid is True
        assert expert.validation.reason is None
