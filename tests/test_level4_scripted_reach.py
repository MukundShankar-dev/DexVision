from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo, replay_demo as replay_app
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"


def _record_one(tmp_path: Path, seed: int) -> Path:
    output = tmp_path / f"episode_{seed:03d}"
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "reach_object",
            "--source",
            "scripted",
            "--session-id",
            f"scripted_train_{seed:03d}",
            "--operator-id",
            "scripted_expert_v1",
            "--session-split",
            "train",
            "--goal-condition-id",
            "reach_block_small_interior",
            "--task-seed",
            str(seed),
            "--output",
            str(output),
            "--level4-pilot-dataset-dir",
            str(tmp_path / "dataset"),
            "--level4-dataset-config",
            str(DATASET_CONFIG),
            "--workcell-config",
            str(WORKCELL_CONFIG),
        ]
    )
    assert record_demo.run_record_demo(args) == 0
    return output


def _recompute_replay_success(episode_dir: Path, seed: int) -> tuple[np.ndarray, float]:
    loaded = load_replay_demo(episode_dir)
    metrics = []
    hand_contacts: list[tuple[str, str]] = []
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="reach_object",
        goal_condition_id="reach_block_small_interior",
        seed=seed,
    ) as task:
        metric_task = task.workcell.create_task("reach_object", **task.goal)

        def observe(_step: object, _state: object) -> None:
            world = task.workcell.get_world_state()
            metrics.append(metric_task.evaluate(world))
            hand_contacts.extend(
                pair
                for pair in world.contacts
                if pair[0].startswith("rh_") or pair[1].startswith("rh_")
            )

        result = replay_loaded_demo(
            loaded,
            task.env,
            speed=1000.0,
            sim_steps_per_action=17,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
        final_qpos = task.env.get_state().qpos.copy()
        marker_position, _ = task.env.get_mocap_pose("pilot_goal_marker")

    assert result.steps_replayed == loaded.episode.actions.shape[0]
    assert metrics[-1].success
    assert metrics[-1].values["maximum_scene_disturbance_m"] <= 0.005
    assert not any("workcell_table" in pair for pair in hand_contacts)
    assert not any("start_button" in pair for pair in hand_contacts)
    assert marker_position == pytest.approx(task.goal["approach_pose"][:3])
    return final_qpos, float(metrics[-1].values["maximum_scene_disturbance_m"])


def test_five_randomized_scripted_reaches_validate_recompute_and_replay(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    episodes = [_record_one(tmp_path, seed) for seed in range(5)]

    for seed, episode_dir in enumerate(episodes):
        episode = load_logged_demo(episode_dir)
        assert episode.metadata["source"] == "scripted"
        assert episode.metadata["random_seed"] == seed
        assert episode.metadata["success"] is True
        assert episode.metadata["recording"]["sim_steps_per_frame"] == 17
        assert episode.request_sources is not None
        assert set(episode.request_sources.tolist()) == {"script"}
        assert episode.requested_actions is not None
        assert episode.commanded_actions is not None
        assert episode.applied_actions is not None
        assert np.array_equal(episode.actions, episode.applied_actions)
        assert np.array_equal(episode.requested_actions, episode.commanded_actions)
        assert np.array_equal(episode.commanded_actions, episode.applied_actions)
        assert episode.safety_masks is not None
        assert not np.any(episode.safety_masks)
        assert episode.safety_reasons is not None
        assert set(episode.safety_reasons.reshape(-1).tolist()) == {"none"}
        assert episode.task_states is not None
        assert bool(episode.task_states[-1, 4])
        assert episode.task_states[-1, 7] <= 0.005
        loaded = load_replay_demo(episode_dir)
        assert replay_app._resolve_sim_steps_per_action(loaded, None) == 17
        _recompute_replay_success(episode_dir, seed)

    first_qpos, first_disturbance = _recompute_replay_success(episodes[0], 0)
    second_qpos, second_disturbance = _recompute_replay_success(episodes[0], 0)
    assert second_qpos == pytest.approx(first_qpos, abs=1e-12)
    assert second_disturbance == pytest.approx(first_disturbance, abs=1e-12)


def test_scripted_source_does_not_enable_camera_or_viewer_preset() -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "reach_object",
            "--source",
            "scripted",
            "--goal-condition-id",
            "reach_block_small_interior",
            "--workcell-dry-run",
        ]
    )
    record_demo._prepare_level4_workcell_recording(args)
    record_demo._apply_recording_presets(args)

    assert not args.show_camera_window
    assert not args.viewer
    assert not args.require_hand_detected


def test_scripted_complete_pick_place_remains_explicitly_deferred() -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "pick_place_sequence",
            "--source",
            "scripted",
            "--goal-condition-id",
            "pp_block_small_inspection_pad",
            "--workcell-dry-run",
        ]
    )
    with pytest.raises(ValueError, match="complete pick/place is a separate checkpoint"):
        record_demo._prepare_level4_workcell_recording(args)
