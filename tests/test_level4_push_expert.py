from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo, replay_demo as replay_app
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.sim.level4_expert import _unsafe_push_contact_reason


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"
CASES = (
    ("push_cuboid_setup_slot_a_interior", 0),
    ("push_cuboid_setup_slot_a_interior", 1),
    ("push_flat_puck_return_bin_left_interior", 2),
    ("push_flat_puck_return_bin_left_interior", 3),
    ("push_flat_puck_return_bin_left_interior", 4),
)


def _record_one(tmp_path: Path, cell: str, seed: int) -> Path:
    output = tmp_path / f"push_episode_{seed:03d}"
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "push_object_to_target",
            "--source",
            "scripted",
            "--session-id",
            f"scripted_push_train_{seed:03d}",
            "--operator-id",
            "scripted_push_expert_v1",
            "--session-split",
            "train",
            "--goal-condition-id",
            cell,
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
    episode_dir: Path, cell: str, seed: int
) -> tuple[np.ndarray, tuple[float, ...]]:
    loaded = load_replay_demo(episode_dir)
    successes: list[bool] = []
    tilts: list[float] = []
    unsafe_contacts: list[tuple[str, str]] = []
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="push_object_to_target",
        goal_condition_id=cell,
        seed=seed,
    ) as task:
        object_id = str(task.goal["object_id"])
        initial_positions = {
            entity.object_id: np.asarray(entity.position)
            for entity in task.initial_world_state.entities
        }
        metric_task = task.workcell.create_task("push_object_to_target", **task.goal)

        def observe(_step: object, _state: object) -> None:
            world = task.workcell.get_world_state()
            result = metric_task.evaluate(world)
            successes.append(result.success)
            tilts.append(float(result.values["object_upright_tilt_rad"]))
            reason = _unsafe_push_contact_reason(world, object_id=object_id)
            if reason is not None:
                unsafe_contacts.extend(world.contacts)

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
                    np.asarray(entity.position[:2])
                    - initial_positions[entity.object_id][:2]
                )
            )
            for entity in final_world.entities
            if entity.object_id != object_id
        )
        marker_position, _ = task.env.get_mocap_pose("pilot_goal_marker")
        target_position = final_world.require_entity(
            str(task.goal["target_zone"])
        ).position
        final_metric = metric_task.evaluate(final_world)

    assert result.steps_replayed == loaded.episode.actions.shape[0]
    assert any(successes)
    assert successes[-1]
    assert final_metric.success
    assert final_metric.values["object_on_board"]
    assert final_metric.values["object_supported"]
    assert final_metric.values["object_upright_tilt_rad"] <= 0.174534
    assert max(tilts) <= 0.174534
    assert not unsafe_contacts
    assert max(disturbances) <= 0.005
    assert marker_position == pytest.approx(target_position)
    return final_qpos, disturbances


def test_five_varied_pushes_recompute_replay_and_stay_task_axis(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    episodes = [
        (_record_one(tmp_path, cell, seed), cell, seed) for cell, seed in CASES
    ]
    families: set[str] = set()
    directions: list[np.ndarray] = []

    for episode_dir, cell, seed in episodes:
        episode = load_logged_demo(episode_dir)
        typed_goal = episode.metadata["typed_goal"]
        initial = episode.metadata["task_config"]["initial_state"][
            "entity_positions_m"
        ]
        object_id = typed_goal["object_id"]
        target_id = typed_goal["target_zone"]
        direction = np.asarray(initial[target_id][:2]) - np.asarray(
            initial[object_id][:2]
        )
        direction /= np.linalg.norm(direction)
        directions.append(direction)
        families.add("flat_puck" if object_id.startswith("puck") else "cuboid")

        assert episode.metadata["success"] is True
        assert episode.metadata["source"] == "scripted"
        assert episode.request_sources is not None
        assert set(episode.request_sources.tolist()) == {"script"}
        assert episode.requested_actions is not None
        assert episode.commanded_actions is not None
        assert episode.applied_actions is not None
        assert np.array_equal(episode.requested_actions, episode.commanded_actions)
        assert np.array_equal(episode.commanded_actions, episode.applied_actions)
        assert episode.safety_masks is not None and not np.any(episode.safety_masks)
        assert episode.intervention_flags is not None
        assert not np.any(episode.intervention_flags)
        assert episode.safety_reasons is not None
        assert set(episode.safety_reasons.reshape(-1).tolist()) == {"none"}
        assert episode.failure_reasons is not None
        assert set(episode.failure_reasons.tolist()) == {""}
        assert episode.online_phases is not None
        phases = [
            phase
            for index, phase in enumerate(episode.online_phases.tolist())
            if index == 0 or phase != episode.online_phases[index - 1]
        ]
        assert phases == ["approach", "push_contact", "settle", "retract"]

        contact_actions = episode.actions[
            episode.online_phases == "push_contact"
        ]
        assert np.all(contact_actions[:, 2:] == contact_actions[0, 2:])
        deltas = np.diff(contact_actions[:, :2], axis=0)
        moving = deltas[np.linalg.norm(deltas, axis=1) > 1e-12]
        assert np.allclose(moving[:, 0] * direction[1], moving[:, 1] * direction[0])
        assert np.all(moving @ direction > 0.0)
        assert replay_app._resolve_sim_steps_per_action(
            load_replay_demo(episode_dir), None
        ) == 17
        _replay_and_recompute(episode_dir, cell, seed)

    assert families == {"cuboid", "flat_puck"}
    assert not np.array_equal(directions[0], directions[-1])
    first = _replay_and_recompute(*episodes[0])
    second = _replay_and_recompute(*episodes[0])
    assert second[0] == pytest.approx(first[0], abs=1e-12)
    assert second[1] == pytest.approx(first[1], abs=1e-12)


def test_push_contact_rejects_every_non_target_body() -> None:
    pytest.importorskip("mujoco")
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="push_object_to_target",
        goal_condition_id="push_cuboid_setup_slot_a_interior",
        seed=0,
    ) as task:
        world = task.initial_world_state
        assert (
            _unsafe_push_contact_reason(
                replace(world, contacts=(("rh_ffdistal", "block_large"),)),
                object_id="block_small",
            )
            == "wrong_object_contact"
        )
        assert (
            _unsafe_push_contact_reason(
                replace(world, contacts=(("rh_ffdistal", "workcell_table"),)),
                object_id="block_small",
            )
            == "table_contact"
        )
        assert (
            _unsafe_push_contact_reason(
                replace(world, contacts=(("block_small", "rh_ffdistal"),)),
                object_id="block_small",
            )
            is None
        )
