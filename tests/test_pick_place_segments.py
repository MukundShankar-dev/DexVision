from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo
from dexvision.apps.validate_level4_episode import validate_episode_directory
from dexvision.evaluation.level4_expert_audit import audit_scripted_episode
from dexvision.logging.demo_logger import (
    action_schema_from_metadata,
    load_logged_demo,
)
from dexvision.logging.phase_labels import (
    derive_pick_place_segments,
    validate_phase_intervals,
)
from dexvision.logging.level4_collection import WorkcellPilotTask
from test_level4_place_expert import record_pick_place


ROOT = Path(__file__).resolve().parents[1]


def test_complete_recording_yields_compatible_reach_pick_and_place_segments(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    episode_dir = record_pick_place(
        tmp_path,
        cell="pp_block_small_inspection_pad",
        seed=0,
        name="segment_contract",
    )
    episode = load_logged_demo(episode_dir)
    intervals = validate_phase_intervals(
        episode.metadata["phase_intervals"],
        frame_count=episode.actions.shape[0],
        phases=episode.online_phases.tolist(),
    )
    segments = derive_pick_place_segments(
        intervals, frame_count=episode.actions.shape[0]
    )
    report = validate_episode_directory(episode_dir)
    action_schema = action_schema_from_metadata(episode.metadata)

    assert [segment.skill_name for segment in segments] == [
        "reach_object",
        "pick_object",
        "place_held_object",
    ]
    assert report["derived_segments"] == [segment.to_dict() for segment in segments]
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == segments[1].start_frame
    assert segments[1].end_frame == segments[2].start_frame
    assert segments[2].end_frame <= episode.actions.shape[0]
    assert episode.actions.shape[1] == action_schema.action_dim
    for segment in segments:
        action_slice = episode.actions[segment.start_frame : segment.end_frame]
        assert action_slice.ndim == 2
        assert action_slice.shape[0] > 0
        assert action_slice.shape[1] == action_schema.action_dim
        assert np.all(np.isfinite(action_slice))


def test_pick_place_replay_keeps_task_geometry_isolated(
    tmp_path: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    episode_dir = tmp_path / "geometry_isolation"
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "pick_place_sequence",
            "--source",
            "scripted",
            "--session-id",
            "geometry_isolation",
            "--operator-id",
            "scripted_pick_place_anchor_v1",
            "--session-split",
            "train",
            "--goal-condition-id",
            "pp_block_small_inspection_pad",
            "--task-seed",
            "0",
            "--output",
            str(episode_dir),
            "--level4-dataset-dir",
            str(tmp_path / "dataset"),
            "--level4-dataset-config",
            str(ROOT / "configs" / "level4_dataset.yaml"),
            "--workcell-config",
            str(ROOT / "configs" / "workcell.yaml"),
            "--enforce-frozen-cell-owner",
        ]
    )
    assert record_demo.run_record_demo(args) == 0
    episode = load_logged_demo(episode_dir)

    assert episode.metadata["success"] is True
    assert episode.metadata["teleop_config"]["scripted_expert"][
        "maximum_placement_center_x_m"
    ] == pytest.approx(0.11)
    assert episode.metadata["teleop_config"]["scripted_expert"][
        "transport_step_m"
    ] == pytest.approx(0.005)
    assert episode.metadata["teleop_config"]["scripted_expert"]["grasp"][
        "family_templates"
    ]["cuboid"]["object_relative_position_m"] == pytest.approx(
        [0.0075, 0.0, 0.020]
    )
    assert episode.metadata["teleop_config"]["scripted_expert"][
        "family_target_offset_xy_m"
    ]["cuboid"] == pytest.approx([-0.01, 0.0])

    audit = audit_scripted_episode(
        episode_dir,
        config_path=ROOT / "configs" / "level4_dataset.yaml",
        workcell_config=ROOT / "configs" / "workcell.yaml",
    )
    assert audit.accepted is True
    assert audit.recomputed_success is True

    with WorkcellPilotTask(
        workcell_config=ROOT / "configs" / "workcell.yaml",
        dataset_config=ROOT / "configs" / "level4_dataset.yaml",
        skill_name="pick_place_sequence",
        goal_condition_id="pp_block_small_inspection_pad",
        seed=0,
    ) as task:
        for geom_name in ("fixture_wall_geom", "start_button_geom"):
            geom_id = mujoco.mj_name2id(
                task.env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            assert task.env.model.geom_contype[geom_id] == 0
            assert task.env.model.geom_conaffinity[geom_id] == 0
            assert task.env.model.geom_rgba[geom_id, 3] == pytest.approx(0.0)
        for geom_name in ("return_bin_left_wall", "return_bin_right_wall"):
            geom_id = mujoco.mj_name2id(
                task.env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            assert task.env.model.geom_contype[geom_id] == 0
            assert task.env.model.geom_conaffinity[geom_id] == 0
            assert task.env.model.geom_rgba[geom_id, 3] > 0.0
