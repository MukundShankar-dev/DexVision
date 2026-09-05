from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo, run_level1_teleop
from dexvision.evaluation.level4_expert_audit import audit_expert_architecture
from dexvision.features.hand_features import no_hand_features
from dexvision.logging.demo_logger import DemoLoggerError
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    load_curl_retargeter_config,
)
from dexvision.sim.level4_expert import (
    Level4ExpertError,
    RequestedAction,
    SafeWaypointReachConfig,
    SafeWaypointReachExpert,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "level4_dataset.yaml"
WORKCELL_CONFIG = ROOT / "configs" / "workcell.yaml"
TELEOP_CONFIG = ROOT / "configs" / "level1_teleop.yaml"


def _neutral_targets() -> dict[str, float]:
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(TELEOP_CONFIG)
    )
    return run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )


def _task(seed: int = 0) -> WorkcellPilotTask:
    return WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="reach_object",
        goal_condition_id="reach_block_small_interior",
        seed=seed,
    )


def test_requested_action_requires_one_complete_finite_named_row() -> None:
    action = RequestedAction(("a", "b"), (1.0, 2.0))
    assert action.as_array().tolist() == [1.0, 2.0]

    with pytest.raises(Level4ExpertError, match="same non-zero length"):
        RequestedAction(("a",), ())
    with pytest.raises(Level4ExpertError, match="unique"):
        RequestedAction(("a", "a"), (1.0, 2.0))
    with pytest.raises(Level4ExpertError, match="finite"):
        RequestedAction(("a",), (float("nan"),))


def test_reset_and_step_are_deterministic_and_do_not_mutate_live_mjdata() -> None:
    pytest.importorskip("mujoco")
    targets = _neutral_targets()
    results = []

    for _ in range(2):
        with _task(seed=3) as task:
            before = task.env.get_state()
            expert = SafeWaypointReachExpert(finger_targets=targets)
            expert.reset(task, task.initial_world_state)
            after = task.env.get_state()
            first = expert.step(task.initial_world_state)
            results.append(
                (
                    tuple(tuple(point) for point in expert.waypoints),
                    first[0].names,
                    first[0].values,
                    first[1:],
                    expert.validation,
                )
            )
            assert after.time == before.time
            assert np.array_equal(after.qpos, before.qpos)
            assert np.array_equal(after.qvel, before.qvel)
            assert np.array_equal(after.ctrl, before.ctrl)

    assert results[0] == results[1]
    waypoints, names, _values, step_status, validation = results[0]
    assert len(waypoints) == 4
    assert names[:7] == (
        "base_position_target/x",
        "base_position_target/y",
        "base_position_target/z",
        "base_orientation_target/qw",
        "base_orientation_target/qx",
        "base_orientation_target/qy",
        "base_orientation_target/qz",
    )
    assert len(names) == 27
    assert step_status == ("approach", False, None)
    assert validation is not None and validation.valid
    assert validation.checked_actions > 0


def test_workspace_rejection_is_stable_and_prevents_execution() -> None:
    pytest.importorskip("mujoco")
    config = SafeWaypointReachConfig(
        transit_height_m=0.30,
        corridor_entry_height_m=0.18,
    )
    with _task() as task:
        expert = SafeWaypointReachExpert(
            finger_targets=_neutral_targets(), config=config
        )
        expert.reset(task, task.initial_world_state)
        assert expert.validation is not None
        assert not expert.validation.valid
        assert expert.validation.reason == "workspace_violation"
        first = expert.step(task.initial_world_state)
        second = expert.step(task.initial_world_state)

    assert first[2:] == (True, "workspace_violation")
    assert second[2:] == (True, "workspace_violation")
    assert first[0] == second[0]


def test_reach_expert_rejects_future_checkpoint_skills() -> None:
    pytest.importorskip("mujoco")
    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="press_button",
        goal_condition_id="press_008_centered_nominal",
        seed=0,
    ) as task:
        expert = SafeWaypointReachExpert(finger_targets=_neutral_targets())
        with pytest.raises(Level4ExpertError, match="only supports reach_object"):
            expert.reset(task, task.initial_world_state)


def test_target_cage_is_centered_around_the_selected_block() -> None:
    pytest.importorskip("mujoco")
    with _task() as task:
        model = task.env.model
        mujoco = task.env._mujoco
        outline_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            task.workcell.config.scene["pilot_target_outline"],
        )
        geom_ids = np.flatnonzero(model.geom_bodyid == outline_id)
        local_z = model.geom_pos[geom_ids, 2]
        half_z = model.geom_size[geom_ids, 2]
        cage_min_z = float(np.min(local_z - half_z))
        cage_max_z = float(np.max(local_z + half_z))
        block = next(
            spec
            for spec in task.workcell.config.objects
            if spec.object_id == "block_small"
        )
        block_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, block.geom
        )
        block_half_height = float(model.geom_size[block_geom_id, 2])

    assert cage_min_z < -block_half_height
    assert cage_max_z > block_half_height
    assert cage_min_z == pytest.approx(-cage_max_z)


def _record_audit_episode(
    tmp_path: Path,
    *,
    skill_name: str,
    cell_id: str,
    seed: int,
    label: str,
    maximum_frames: int = 0,
) -> Path:
    output = tmp_path / label
    arguments = [
        "--task",
        "level4_workcell",
        "--skill",
        skill_name,
        "--source",
        "scripted",
        "--episode-id",
        f"audit_{label}",
        "--session-id",
        f"audit_session_{label}",
        "--operator-id",
        "scripted_expert_audit_v1",
        "--session-split",
        "train",
        "--goal-condition-id",
        cell_id,
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
    if maximum_frames:
        arguments.extend(("--max-frames", str(maximum_frames)))
    args = record_demo.build_parser().parse_args(arguments)
    if maximum_frames:
        with pytest.raises(DemoLoggerError, match="did not satisfy"):
            record_demo.run_record_demo(args)
    else:
        assert record_demo.run_record_demo(args) == 0
    return output


def test_repeated_cross_skill_experts_replay_recompute_and_keep_failures(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    cases = (
        ("reach_object", "reach_block_small_interior"),
        ("press_button", "press_008_centered_nominal"),
        ("push_object_to_target", "push_cuboid_setup_slot_a_interior"),
        ("pick_object", "pp_block_small_return_bin_left"),
        ("pick_place_sequence", "pp_block_small_inspection_pad"),
    )
    episodes = [
        _record_audit_episode(
            tmp_path,
            skill_name=skill,
            cell_id=cell,
            seed=seed,
            label=f"{skill}_{seed}",
        )
        for skill, cell in cases
        for seed in range(2)
    ]
    failure = _record_audit_episode(
        tmp_path,
        skill_name="reach_object",
        cell_id="reach_block_small_interior",
        seed=7,
        label="ordinary_reach_failure",
        maximum_frames=1,
    )

    report = audit_expert_architecture(
        [*episodes, failure],
        config_path=DATASET_CONFIG,
        workcell_config=WORKCELL_CONFIG,
    )

    assert report["qualified"] is True
    assert report["episode_count"] == 11
    assert report["accepted_episode_count"] == 10
    assert report["ordinary_failure_count"] == 1
    assert report["unexpected_rejection_count"] == 0
    assert report["accepted_source_skill_counts"] == {
        skill: 2 for skill, _cell in cases
    }
    assert set(report["accepted_derived_skill_counts"]) == {
        "reach_object",
        "press_button",
        "push_object_to_target",
        "pick_object",
        "place_held_object",
    }
    assert report["safety_violation_episode_count"] == 0
    assert report["neighbor_disturbance_failure_count"] == 0
    failure_audit = report["episodes"][-1]
    assert failure_audit["accepted"] is False
    assert failure_audit["operator_success"] is False
    assert "ordinary_task_failure" in failure_audit["rejection_reasons"]
