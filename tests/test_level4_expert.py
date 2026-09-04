from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import run_level1_teleop
from dexvision.features.hand_features import no_hand_features
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
