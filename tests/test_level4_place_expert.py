from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo, replay_demo as replay_app
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.sim.mujoco_env import MujocoEnv


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"
CASES = (
    ("cuboid", "pp_block_small_inspection_pad", 0),
    ("cuboid", "pp_block_small_inspection_pad", 1),
    ("cuboid", "pp_block_small_setup_slot_a", 2),
    ("cuboid", "pp_block_small_setup_slot_a", 3),
    ("cylinder", "pp_cylinder_short_inspection_pad", 0),
    ("cylinder", "pp_cylinder_short_inspection_pad", 1),
    ("cylinder", "pp_cylinder_short_setup_slot_a", 2),
    ("flat_puck", "pp_puck_light_inspection_pad", 0),
    ("flat_puck", "pp_puck_light_inspection_pad", 1),
    ("flat_puck", "pp_puck_light_setup_slot_a", 2),
)


def _quaternion_z_axis(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
    )


def record_pick_place(
    tmp_path: Path,
    *,
    cell: str,
    seed: int,
    name: str,
) -> Path:
    output = tmp_path / name
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "pick_place_sequence",
            "--source",
            "scripted",
            "--session-id",
            f"scripted_place_{name}",
            "--operator-id",
            "scripted_pick_place_expert_v1",
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
    episode_dir: Path,
    *,
    family: str,
    cell: str,
    seed: int,
) -> tuple[np.ndarray, tuple[float, ...]]:
    loaded = load_replay_demo(episode_dir)
    successes: list[bool] = []
    held_orientation_deviations: list[float] = []
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id=cell,
        seed=seed,
    ) as task:
        object_id = str(task.goal["object_id"])
        target_id = str(task.goal["target_id"])
        initial_positions = {
            entity.object_id: np.asarray(entity.position)
            for entity in task.initial_world_state.entities
        }
        assert task.initial_world_state.relation_for(object_id).supported_by == (
            "workcell_table"
        )
        initial_orientation = np.asarray(
            task.initial_world_state.require_entity(object_id).orientation_wxyz,
            dtype=np.float64,
        )
        metric_task = task.workcell.create_task(
            "place_held_object", object_id=object_id, target_id=target_id
        )

        def observe(step: object, _state: object) -> None:
            world = task.workcell.get_world_state()
            successes.append(metric_task.evaluate(world).success)
            phase = str(loaded.episode.online_phases[step.index])
            if phase in {"lift", "stabilize", "transport", "place"}:
                orientation = np.asarray(
                    world.require_entity(object_id).orientation_wxyz,
                    dtype=np.float64,
                )
                if family == "cuboid":
                    dot = abs(float(np.dot(initial_orientation, orientation)))
                    deviation = 2.0 * math.acos(
                        float(np.clip(dot, -1.0, 1.0))
                    )
                else:
                    initial_axis = _quaternion_z_axis(initial_orientation)
                    observed_axis = _quaternion_z_axis(orientation)
                    deviation = math.acos(
                        float(np.clip(np.dot(initial_axis, observed_axis), -1.0, 1.0))
                    )
                held_orientation_deviations.append(deviation)

        result = replay_loaded_demo(
            loaded,
            task.env,
            speed=1000.0,
            sim_steps_per_action=17,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
        final_world = task.workcell.get_world_state()
        final_metric = metric_task.evaluate(final_world)
        final_object = final_world.require_entity(object_id)
        final_relation = final_world.relation_for(object_id)
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
        source_displacement = float(
            np.linalg.norm(
                np.asarray(final_object.position[:2])
                - initial_positions[object_id][:2]
            )
        )

    assert result.steps_replayed == loaded.episode.actions.shape[0]
    assert held_orientation_deviations
    assert max(held_orientation_deviations) <= math.radians(5.0)
    assert any(successes)
    assert successes[-1]
    assert final_metric.success
    assert final_metric.values["object_to_target_distance_m"] <= 0.025
    assert final_metric.values["object_linear_speed_mps"] <= 0.020
    assert final_metric.values["object_angular_speed_radps"] <= 0.200
    assert final_metric.values["object_inside_target"] is True
    assert final_metric.values["object_supported"] is True
    assert final_relation.supported_by == "workcell_table"
    assert final_relation.held_by is None
    assert source_displacement >= 0.030
    assert max(disturbances) <= 0.005
    return final_qpos, disturbances


def test_ten_complete_pick_places_recompute_replay_and_settle(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    episodes = [
        (
            record_pick_place(
                tmp_path,
                cell=cell,
                seed=seed,
                name=f"{family}_{index:03d}",
            ),
            family,
            cell,
            seed,
        )
        for index, (family, cell, seed) in enumerate(CASES)
    ]

    for episode_dir, _family, cell, seed in episodes:
        episode = load_logged_demo(episode_dir)
        assert episode.metadata["source"] == "scripted"
        assert episode.metadata["skill_name"] == "pick_place_sequence"
        assert episode.metadata["success"] is True
        assert (
            episode.metadata["teleop_config"]["scripted_expert"]["grasp"]
            ["orientation_preservation_policy"]
            == "shape_aware_hammer_grip_with_world_orientation_hold"
        )
        contact = episode.metadata["task_config"]["contact_dynamics"]
        assert contact["table_condim"] == 6
        assert contact["geom_friction"]["puck_light_geom"] == [1.5, 0.01, 0.02]
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
        assert phases == [
            "approach",
            "acquire",
            "lift",
            "stabilize",
            "transport",
            "place",
            "release",
            "settle",
            "retract",
        ]
        assert replay_app._resolve_sim_steps_per_action(
            load_replay_demo(episode_dir), None
        ) == 17
        _replay_and_recompute(
            episode_dir, family=_family, cell=cell, seed=seed
        )

    assert {family for _path, family, _cell, _seed in episodes} == {
        "cuboid",
        "cylinder",
        "flat_puck",
    }
    first = _replay_and_recompute(
        episodes[0][0],
        family=episodes[0][1],
        cell=episodes[0][2],
        seed=episodes[0][3],
    )
    second = _replay_and_recompute(
        episodes[0][0],
        family=episodes[0][1],
        cell=episodes[0][2],
        seed=episodes[0][3],
    )
    assert second[0] == pytest.approx(first[0], abs=1e-12)
    assert second[1] == pytest.approx(first[1], abs=1e-12)

    loaded = load_replay_demo(episodes[0][0])
    with MujocoEnv(loaded.model_path) as env:
        raw_result = replay_loaded_demo(
            loaded,
            env,
            speed=1000.0,
            sim_steps_per_action=17,
            sleep_fn=lambda _delay: None,
        )
        raw_qpos = env.get_state().qpos.copy()
    assert raw_result.steps_replayed == loaded.episode.actions.shape[0]
    assert raw_qpos == pytest.approx(first[0], abs=1e-12)


def test_round_object_placement_uses_physical_rolling_resistance() -> None:
    mujoco = pytest.importorskip("mujoco")
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_puck_light_inspection_pad",
        seed=0,
    ) as task:
        model = task.env.model

        def geom_id(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        assert model.geom_condim[geom_id("workcell_table_geom")] == 6
        assert model.geom_friction[geom_id("cylinder_short_geom")] == pytest.approx(
            [0.75, 0.01, 0.05]
        )
        assert model.geom_friction[geom_id("puck_light_geom")] == pytest.approx(
            [1.50, 0.01, 0.02]
        )
