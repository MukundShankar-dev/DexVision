from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo, replay_demo as replay_app
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.sim.level4_expert import (
    _has_joint_limit_violation,
    _unsafe_button_contact_reason,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"
GOAL_CELL = "press_014_centered_nominal"


def _record_one(tmp_path: Path, seed: int) -> Path:
    output = tmp_path / f"button_episode_{seed:03d}"
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "press_button",
            "--source",
            "scripted",
            "--session-id",
            f"scripted_button_train_{seed:03d}",
            "--operator-id",
            "scripted_button_expert_v1",
            "--session-split",
            "train",
            "--goal-condition-id",
            GOAL_CELL,
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


def _replay_and_recompute(
    episode_dir: Path, seed: int
) -> tuple[np.ndarray, tuple[float, ...]]:
    loaded = load_replay_demo(episode_dir)
    successes: list[bool] = []
    external_hand_contacts: list[tuple[str, str]] = []
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="press_button",
        goal_condition_id=GOAL_CELL,
        seed=seed,
    ) as task:
        initial_positions = {
            entity.object_id: np.asarray(entity.position)
            for entity in task.initial_world_state.entities
        }
        metric_task = task.workcell.create_task("press_button", **task.goal)

        def observe(_step: object, _state: object) -> None:
            world = task.workcell.get_world_state()
            successes.append(metric_task.evaluate(world).success)
            for pair in world.contacts:
                left_hand = pair[0].startswith("rh_")
                right_hand = pair[1].startswith("rh_")
                if not left_hand and not right_hand:
                    continue
                if left_hand and right_hand:
                    continue
                other = pair[1] if left_hand else pair[0]
                if other != "start_button":
                    external_hand_contacts.append(pair)

        result = replay_loaded_demo(
            loaded,
            task.env,
            speed=1000.0,
            sim_steps_per_action=17,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
        final_world = task.workcell.get_world_state()
        final_qpos = task.env.get_state().qpos.copy()
        disturbances = tuple(
            float(
                np.linalg.norm(
                    np.asarray(entity.position) - initial_positions[entity.object_id]
                )
            )
            for entity in final_world.entities
            if entity.object_id != "start_button"
        )
        marker_position, _ = task.env.get_mocap_pose("pilot_goal_marker")

    assert result.steps_replayed == loaded.episode.actions.shape[0]
    assert any(successes)
    assert not external_hand_contacts
    assert final_world.require_fixture("start_button").press_depth_m <= 0.002
    assert not any("start_button" in pair for pair in final_world.contacts)
    assert max(disturbances) <= 0.005
    assert marker_position == pytest.approx([0.11, -0.11, 0.20])
    return final_qpos, disturbances


def test_five_randomized_button_resets_recompute_and_replay(
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
        assert episode.intervention_flags is not None
        assert not np.any(episode.intervention_flags)
        assert episode.failure_reasons is not None
        assert set(episode.failure_reasons.tolist()) == {""}
        assert episode.online_phases is not None
        phases = [
            phase
            for index, phase in enumerate(episode.online_phases.tolist())
            if index == 0 or phase != episode.online_phases[index - 1]
        ]
        assert phases == ["approach", "fixture_contact", "retract"]
        assert np.all(episode.actions[:, 3:] == episode.actions[0, 3:])
        assert replay_app._resolve_sim_steps_per_action(
            load_replay_demo(episode_dir), None
        ) == 17
        _replay_and_recompute(episode_dir, seed)

    first_qpos, first_disturbances = _replay_and_recompute(episodes[0], 0)
    second_qpos, second_disturbances = _replay_and_recompute(episodes[0], 0)
    assert second_qpos == pytest.approx(first_qpos, abs=1e-12)
    assert second_disturbances == pytest.approx(first_disturbances, abs=1e-12)


def test_button_contact_and_joint_limit_rejections_are_explicit() -> None:
    pytest.importorskip("mujoco")
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="press_button",
        goal_condition_id=GOAL_CELL,
        seed=0,
    ) as task:
        world = task.initial_world_state
        assert (
            _unsafe_button_contact_reason(
                replace(world, contacts=(("rh_palm", "fixture_wall"),)),
                button_id="start_button",
            )
            == "wrong_fixture_contact"
        )
        assert (
            _unsafe_button_contact_reason(
                replace(world, contacts=(("rh_palm", "workcell_table"),)),
                button_id="start_button",
            )
            == "table_contact"
        )
        assert (
            _unsafe_button_contact_reason(
                replace(world, contacts=(("rh_mfdistal", "start_button"),)),
                button_id="start_button",
            )
            is None
        )
        joint_id = task.env._mujoco.mj_name2id(
            task.env.model,
            task.env._mujoco.mjtObj.mjOBJ_JOINT,
            "rh_WRJ2",
        )
        address = int(task.env.model.jnt_qposadr[joint_id])
        task.env.data.qpos[address] = task.env.model.jnt_range[joint_id, 1] + 0.01
        assert _has_joint_limit_violation(task.workcell, tolerance=0.0005)
