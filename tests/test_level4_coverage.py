from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexvision.apps.summarize_level4_coverage import main as summarize_main
from dexvision.evaluation.dataset_coverage import summarize_level4_coverage
from dexvision.logging.level4_collection import (
    ManualReplayReview,
    PilotReview,
    append_manual_replay_review,
    save_pilot_review,
)
from dexvision.logging.session_manifest import RecordingSession, append_session_manifest


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


def _add_session(dataset_dir: Path, session_id: str) -> None:
    append_session_manifest(
        dataset_dir / "session_manifest.json",
        RecordingSession(
            recording_session_id=session_id,
            operator_id="operator_local_01",
            split="train",
            process_start_timestamp=f"2026-09-04T12:00:0{len(session_id)}Z",
            reset_seed=len(session_id),
            calibration_record_digest=f"sha256:{session_id}",
        ),
    )


def _write_episode(
    dataset_dir: Path,
    *,
    index: int,
    session_id: str,
    skill_name: str,
    cell_id: str,
    typed_goal: dict[str, object],
    source: str | None = None,
    manual: bool = False,
    success: bool = True,
) -> Path:
    episode_id = f"pilot_{index:06d}"
    path = dataset_dir / session_id / f"episode_{index:06d}"
    path.mkdir(parents=True)
    phases = PICK_PLACE_PHASES if skill_name == "pick_place_sequence" else ("approach", "retract")
    intervals = [
        {"phase": phase, "start_frame": frame, "end_frame": frame + 1}
        for frame, phase in enumerate(phases)
    ]
    metadata = {
        "episode_schema_version": "level4/episode-v1",
        "episode_id": episode_id,
        "recording_session_id": session_id,
        "skill_name": skill_name,
        "source": source
        or ("teleoperation" if skill_name == "reach_object" else "scripted"),
        "goal_condition_id": cell_id,
        "typed_goal": typed_goal,
        "object_instance_ids": (
            [typed_goal["object_id"]] if "object_id" in typed_goal else []
        ),
        "phase_intervals": intervals,
        "collection_duration_seconds": 60.0,
        "success": success,
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(path / "timestamps.npy", np.arange(len(phases), dtype=np.float64))
    np.save(path / "online_phases.npy", np.asarray(phases))
    np.save(path / "audited_phases.npy", np.asarray(phases))
    review = (
        _accepted_review(episode_id)
        if success
        else PilotReview(
            episode_id=episode_id,
            schema_validation=True,
            timestamp_alignment=True,
            headless_replay=True,
            terminal_metric_recomputation=True,
            recomputed_success=False,
            operator_label_agreement=True,
            quality_thresholds=False,
            coverage_assignment=True,
            split_session_leakage_audit=True,
            expert_accepted=False,
            rejection_reasons=("ordinary_task_failure",),
        )
    )
    save_pilot_review(path, review)
    if manual:
        skills = {
            "reach_object": ("reach_object",),
            "pick_place_sequence": ("pick_object", "place_held_object"),
            "push_object_to_target": ("push_object_to_target",),
            "press_button": ("press_button",),
        }[skill_name]
        append_manual_replay_review(
            dataset_dir,
            ManualReplayReview(
                episode_id=episode_id,
                verified_skills=skills,
                passed=True,
                notes="Synthetic visible-replay confirmation for coverage test.",
            ),
        )
    return path


def _write_complete_synthetic_pilot(dataset_dir: Path) -> None:
    _add_session(dataset_dir, "session_a")
    _add_session(dataset_dir, "session_b")
    index = 0
    for offset in range(5):
        index += 1
        _write_episode(
            dataset_dir,
            index=index,
            session_id="session_a" if offset < 3 else "session_b",
            skill_name="reach_object",
            cell_id="reach_block_small_interior",
            typed_goal={"entity_id": "block_small"},
            manual=offset == 0,
        )
    pick_place_goals = (
        ("block_small", "return_bin_left", "pp_block_small_return_bin_left"),
        ("cylinder_short", "inspection_pad", "pp_cylinder_short_inspection_pad"),
        ("puck_light", "setup_slot_a", "pp_puck_light_setup_slot_a"),
    )
    for offset in range(10):
        object_id, target_id, cell_id = pick_place_goals[offset % 3]
        index += 1
        _write_episode(
            dataset_dir,
            index=index,
            session_id="session_a" if offset < 5 else "session_b",
            skill_name="pick_place_sequence",
            cell_id=cell_id,
            typed_goal={"object_id": object_id, "target_id": target_id},
            manual=offset == 0,
        )
    for offset in range(5):
        index += 1
        _write_episode(
            dataset_dir,
            index=index,
            session_id="session_a" if offset < 3 else "session_b",
            skill_name="push_object_to_target",
            cell_id="push_cuboid_setup_slot_a_interior",
            typed_goal={"object_id": "block_small", "target_zone": "setup_slot_a"},
            manual=offset == 0,
        )
    for offset in range(5):
        index += 1
        _write_episode(
            dataset_dir,
            index=index,
            session_id="session_a" if offset < 3 else "session_b",
            skill_name="press_button",
            cell_id="press_008_centered_nominal",
            typed_goal={"button_id": "start_button"},
            manual=offset == 0,
        )
    index += 1
    _write_episode(
        dataset_dir,
        index=index,
        session_id="session_b",
        skill_name="reach_object",
        cell_id="reach_block_small_interior",
        typed_goal={"entity_id": "block_small"},
        success=False,
    )


def test_report_separates_episodes_segments_failures_and_manual_gate(
    tmp_path: Path,
) -> None:
    _write_complete_synthetic_pilot(tmp_path)

    report = summarize_level4_coverage(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )

    assert report["attempt_episode_count"] == 26
    assert report["expert_accepted_episode_count"] == 25
    assert report["ordinary_failure_episode_count"] == 1
    assert report["episode_counts_by_group"]["pick_place"]["accepted"] == 10
    assert report["segment_counts_by_skill"] == {
        "pick_object": 10,
        "place_held_object": 10,
        "press_button": 5,
        "push_object_to_target": 5,
        "reach_object": 15,
    }
    assert report["missing_object_families"] == []
    assert report["missing_target_types"] == []
    assert report["phase_label_agreement"]["disagreement_fraction"] == 0.0
    assert report["collection_time"]["minutes_per_accepted_episode"] == 1.0
    assert report["optional_dial_decision"] == "deferred"
    assert report["automated_pilot_requirements_passed"] is True
    assert report["manual_replay_gate_passed"] is True
    assert report["checkpoint_complete"] is True
    assert report["coverage_matrix"]["minimum_episode_total"] == 114
    assert report["coverage_matrix"]["fits_required_envelope"] is True
    assert report["source_mix"]["sources"]["teleoperation"] == {
        "observed": 5,
        "minimum": 13,
        "passed": False,
    }
    assert report["source_mix"]["sources"]["scripted"]["observed"] == 20
    assert report["storage"]["payload_handling"] == "git_lfs"


def test_cli_writes_incomplete_empty_report_without_mutating_dataset(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "reports" / "coverage.json"

    result = summarize_main(
        [
            "--config",
            str(CONFIG_PATH),
            "--dataset-dir",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pilot_status"] == "incomplete"
    assert report["storage"]["payload_handling"] == "undetermined_no_pilot_data"
    assert "missing session manifest" in report["issues"][0]
    assert "Pilot status: incomplete" in capsys.readouterr().out


def test_source_mismatch_cannot_satisfy_a_frozen_cell(tmp_path: Path) -> None:
    _add_session(tmp_path, "session_a")
    _write_episode(
        tmp_path,
        index=1,
        session_id="session_a",
        skill_name="reach_object",
        cell_id="reach_block_small_interior",
        typed_goal={"entity_id": "block_small"},
        source="scripted",
    )

    report = summarize_level4_coverage(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )

    assert any("does not match cell requirement" in issue for issue in report["issues"])
    assert report["source_mix"]["sources"]["scripted"]["observed"] == 0
    cell = next(
        row
        for row in report["coverage_matrix"]["cells"]
        if row["cell_id"] == "reach_block_small_interior"
    )
    assert cell["observed"] == 0
