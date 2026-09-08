from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from dexvision.apps import record_demo
from dexvision.evaluation.dataset_coverage import summarize_level4_coverage
from dexvision.logging.level4_collection import (
    ManualReplayReview,
    PilotReview,
    append_manual_replay_review,
    build_level4_pick_place_collection_plan,
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


def _add_sessions(dataset_dir: Path) -> None:
    for slot, split in (
        ("session_a", "train"),
        ("session_b", "train"),
        ("session_c", "validation"),
        ("session_d", "test"),
    ):
        append_session_manifest(
            dataset_dir / "session_manifest.json",
            RecordingSession(
                recording_session_id=slot,
                operator_id="scripted_pick_place_anchor_v1",
                split=split,
                process_start_timestamp=f"2026-09-08T0{len(slot)}:00:00Z",
                reset_seed=len(slot),
                calibration_record_digest=f"sha256:{slot}",
            ),
        )


def _write_episode(
    dataset_dir: Path,
    *,
    sequence: int,
    session_id: str,
    cell_id: str,
    object_id: str,
    target_id: str,
    accepted: bool = True,
) -> str:
    episode_id = f"level45a_{sequence:06d}"
    episode_dir = dataset_dir / session_id / f"episode_{sequence:06d}"
    episode_dir.mkdir(parents=True)
    metadata = {
        "episode_schema_version": "level4/episode-v1",
        "episode_id": episode_id,
        "recording_session_id": session_id,
        "skill_name": "pick_place_sequence",
        "source": "scripted",
        "goal_condition_id": cell_id,
        "typed_goal": {"object_id": object_id, "target_id": target_id},
        "object_instance_ids": [object_id],
        "phase_intervals": [
            {"phase": phase, "start_frame": index, "end_frame": index + 1}
            for index, phase in enumerate(PICK_PLACE_PHASES)
        ],
        "collection_duration_seconds": 1.0,
        "success": accepted,
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    np.save(episode_dir / "timestamps.npy", np.arange(len(PICK_PLACE_PHASES)))
    np.save(episode_dir / "online_phases.npy", np.asarray(PICK_PLACE_PHASES))
    np.save(episode_dir / "audited_phases.npy", np.asarray(PICK_PLACE_PHASES))
    review = (
        _accepted_review(episode_id)
        if accepted
        else PilotReview(
            episode_id=episode_id,
            schema_validation=True,
            timestamp_alignment=True,
            headless_replay=True,
            terminal_metric_recomputation=True,
            recomputed_success=False,
            operator_label_agreement=True,
            quality_thresholds=True,
            coverage_assignment=True,
            split_session_leakage_audit=True,
            expert_accepted=False,
            rejection_reasons=("failed_acquisition",),
        )
    )
    save_pilot_review(episode_dir, review)
    return episode_id


def _write_complete_anchor(dataset_dir: Path) -> dict[str, list[str]]:
    _add_sessions(dataset_dir)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in config["coverage_cells"]}
    episode_ids: dict[str, list[str]] = defaultdict(list)
    for item in build_level4_pick_place_collection_plan(CONFIG_PATH):
        cell = cells[item.coverage_cell_id]
        episode_ids[item.coverage_cell_id].append(
            _write_episode(
                dataset_dir,
                sequence=item.sequence,
                session_id=item.session_slot,
                cell_id=item.coverage_cell_id,
                object_id=cell["object_id"],
                target_id=cell["target_id"],
            )
        )
    return episode_ids


def test_pick_place_plan_expands_the_frozen_42_episode_anchor() -> None:
    plan = build_level4_pick_place_collection_plan(CONFIG_PATH)

    assert len(plan) == 42
    assert {item.data_group for item in plan} == {"pick_place"}
    assert {item.skill_name for item in plan} == {"pick_place_sequence"}
    assert {item.source for item in plan} == {"scripted"}
    assert Counter(item.split for item in plan) == {
        "train": 18,
        "validation": 6,
        "test": 18,
    }
    assert len({item.coverage_cell_id for item in plan}) == 30
    assert {item.session_slot for item in plan if item.split == "train"} == {
        "session_a",
        "session_b",
    }
    assert {item.seed for item in plan if item.repetition == 1} == {0}
    assert {item.seed for item in plan if item.repetition == 2} == {1}
    assert {item.seed for item in plan if item.split == "test"} == {0}


def test_complete_anchor_passes_automated_gates_but_waits_for_manual_replay(
    tmp_path: Path,
) -> None:
    _write_complete_anchor(tmp_path)

    report = summarize_level4_coverage(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )["level4_5a_pick_place_collection"]

    assert report["accepted_episode_count"] == 42
    assert report["coverage_matrix"]["cell_count"] == 30
    assert report["coverage_matrix"]["complete_cell_count"] == 30
    assert report["object_family_counts"].keys() == {
        "cuboid",
        "cylinder",
        "flat_puck",
    }
    assert len(report["object_instance_counts"]) == 6
    assert len(report["target_counts"]) == 5
    assert report["segment_counts_by_skill"] == {
        "pick_object": 42,
        "place_held_object": 42,
        "reach_object": 42,
    }
    assert report["session_balance_passed"] is True
    assert report["test_isolation_passed"] is True
    assert report["issues"] == []
    assert report["automated_requirements_passed"] is True
    assert report["manual_replay"]["passed"] is False
    assert report["status"] == "manual_verification_required"
    assert report["checkpoint_complete"] is False


def test_six_stratified_manual_replays_complete_the_anchor_gate(
    tmp_path: Path,
) -> None:
    episode_ids = _write_complete_anchor(tmp_path)
    reviewed_cells = (
        "pp_block_small_return_bin_left",
        "pp_cylinder_short_inspection_pad",
        "pp_puck_light_setup_slot_a",
        "pp_block_small_setup_slot_b",
        "pp_cylinder_tall_return_bin_right",
        "pp_puck_heavy_setup_slot_b",
    )
    for cell_id in reviewed_cells:
        append_manual_replay_review(
            tmp_path,
            ManualReplayReview(
                episode_id=episode_ids[cell_id][0],
                verified_skills=(
                    "reach_object",
                    "pick_object",
                    "place_held_object",
                ),
                passed=True,
                notes="Visible phases and final placement match saved labels.",
            ),
        )

    report = summarize_level4_coverage(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )["level4_5a_pick_place_collection"]

    assert report["manual_replay"]["reviewed_episode_count"] == 6
    assert report["manual_replay"]["object_families"] == [
        "cuboid",
        "cylinder",
        "flat_puck",
    ]
    assert report["manual_replay"]["target_types"] == [
        "placement_slot",
        "planar_zone",
        "receptacle",
    ]
    assert report["manual_replay"]["passed"] is True
    assert report["status"] == "complete"
    assert report["checkpoint_complete"] is True


def test_failed_pick_phase_never_adds_later_expert_segments(tmp_path: Path) -> None:
    _write_complete_anchor(tmp_path)
    _write_episode(
        tmp_path,
        sequence=999,
        session_id="session_a",
        cell_id="pp_block_small_inspection_pad",
        object_id="block_small",
        target_id="inspection_pad",
        accepted=False,
    )

    report = summarize_level4_coverage(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )["level4_5a_pick_place_collection"]

    assert report["attempt_episode_count"] == 43
    assert report["accepted_episode_count"] == 42
    assert report["rejected_or_failed_auditable_count"] == 1
    assert report["segment_counts_by_skill"] == {
        "pick_object": 42,
        "place_held_object": 42,
        "reach_object": 42,
    }


def test_pick_place_plan_cli_is_read_only(tmp_path: Path, capsys) -> None:
    result = record_demo.main(
        [
            "--task",
            "level4_workcell",
            "--print-level4-pick-place-plan",
            "--level4-dataset-config",
            str(CONFIG_PATH),
            "--level4-dataset-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Total Level 4.5A accepted episodes required: 42" in output
    assert "pick_place_sequence" in output
    assert "reach_object" not in output
    assert list(tmp_path.iterdir()) == []
