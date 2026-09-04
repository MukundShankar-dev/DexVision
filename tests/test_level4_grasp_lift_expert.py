from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dexvision.apps import record_demo, replay_demo as replay_app
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.sim.level4_expert import _unsafe_grasp_contact_reason


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"
CASES = (
    ("cuboid", "pp_block_small_return_bin_left"),
    ("cylinder", "pp_cylinder_short_return_bin_left"),
    ("flat_puck", "pp_puck_light_return_bin_left"),
)


def _record_one(tmp_path: Path, family: str, cell: str, seed: int) -> Path:
    output = tmp_path / f"grasp_{family}_{seed:03d}"
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "pick_object",
            "--source",
            "scripted",
            "--session-id",
            f"scripted_grasp_{family}_{seed:03d}",
            "--operator-id",
            "scripted_grasp_expert_v1",
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
    metric_successes: list[bool] = []
    unsafe_after_lift: list[str] = []
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_object",
        goal_condition_id=cell,
        seed=seed,
    ) as task:
        object_id = str(task.goal["object_id"])
        initial_positions = {
            entity.object_id: np.asarray(entity.position)
            for entity in task.initial_world_state.entities
        }
        initial_z = task.initial_world_state.require_entity(object_id).position[2]
        assert (
            task.initial_world_state.relation_for(object_id).supported_by
            == "workcell_table"
        )
        metric_task = task.workcell.create_task("pick_object", **task.goal)

        def observe(_step: object, _state: object) -> None:
            world = task.workcell.get_world_state()
            metric = metric_task.evaluate(world)
            metric_successes.append(metric.success)
            lift_height = world.require_entity(object_id).position[2] - initial_z
            if lift_height >= 0.040:
                reason = _unsafe_grasp_contact_reason(
                    world,
                    object_id=object_id,
                    allow_table_contact=False,
                )
                if reason is not None:
                    unsafe_after_lift.append(reason)

        result = replay_loaded_demo(
            loaded,
            task.env,
            speed=1000.0,
            sim_steps_per_action=17,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
        final_world = task.workcell.get_world_state()
        final_object = final_world.require_entity(object_id)
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

    assert result.steps_replayed == loaded.episode.actions.shape[0]
    assert any(metric_successes)
    assert final_world.relation_for(object_id).held_by == "rh_palm"
    assert final_world.relation_for(object_id).supported_by is None
    assert final_object.position[2] - initial_z >= 0.040
    assert np.linalg.norm(final_object.linear_velocity) <= 0.020
    assert not unsafe_after_lift
    assert max(disturbances) <= 0.005
    return final_qpos, disturbances


def test_three_resets_per_family_recompute_replay_and_hold_stably(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    episodes = [
        (_record_one(tmp_path, family, cell, seed), cell, seed)
        for family, cell in CASES
        for seed in range(3)
    ]

    for episode_dir, cell, seed in episodes:
        episode = load_logged_demo(episode_dir)
        assert episode.metadata["source"] == "scripted"
        assert episode.metadata["skill_name"] == "pick_object"
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
        assert phases == ["approach", "acquire", "lift", "stabilize"]
        assert replay_app._resolve_sim_steps_per_action(
            load_replay_demo(episode_dir), None
        ) == 17
        _replay_and_recompute(episode_dir, cell, seed)

    first = _replay_and_recompute(*episodes[0])
    second = _replay_and_recompute(*episodes[0])
    assert second[0] == pytest.approx(first[0], abs=1e-12)
    assert second[1] == pytest.approx(first[1], abs=1e-12)


def test_grasp_templates_and_synergy_are_configuration_owned() -> None:
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    grasp = config["pilot"]["scripted_grasp"]

    assert grasp["hand_poses"] == {
        "open": "configured_retargeter_open",
        "closed": "configured_retargeter_full_flexion",
    }
    assert set(grasp["family_templates"]) == {"cuboid", "cylinder", "flat_puck"}
    for template in grasp["family_templates"].values():
        assert len(template["object_relative_position_m"]) == 3
        assert np.linalg.norm(template["wrist_orientation_wxyz"]) == pytest.approx(1.0)
        assert 0.0 < template["grasp_synergy"] <= 1.0
        assert template["lift_distance_m"] >= 0.08
