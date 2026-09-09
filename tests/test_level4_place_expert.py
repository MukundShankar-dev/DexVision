from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dexvision.apps import record_demo, replay_demo as replay_app, run_level1_teleop
from dexvision.features.hand_features import no_hand_features
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import (
    WorkcellPilotTask,
    build_level4_procedural_expansion_plan,
    sample_level4_procedural_variation,
)
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    load_curl_retargeter_config,
)
from dexvision.sim.level4_expert import (
    DeterministicGraspLiftConfig,
    DeterministicPickPlaceExpert,
    DeterministicPlaceConfig,
    DeterministicPushConfig,
    DeterministicPushExpert,
    GraspFamilyTemplate,
    _copy_task_local_model_configuration,
    _conditioned_grasp_orientation,
)
from dexvision.sim.mujoco_env import MujocoEnv
from dexvision.sim.workcell import Workcell


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


def test_cuboid_grasp_orientation_conditions_both_seeded_yaw_signs() -> None:
    template = GraspFamilyTemplate(
        object_relative_position_m=(0.0, 0.0, 0.02),
        wrist_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        negative_object_yaw_to_wrist_yaw_gain=-1.0,
        orientation_symmetry="none",
        orientation_feedback_enabled=True,
        transport_orientation_feedback_enabled=True,
        grasp_synergy=1.0,
        lift_distance_m=0.08,
    )

    for yaw in (-0.2, 0.2):
        object_orientation = np.asarray(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        )
        conditioned = _conditioned_grasp_orientation(template, object_orientation)
        expected_yaw = -yaw
        expected = np.asarray(
            [
                math.cos(expected_yaw / 2.0),
                0.0,
                0.0,
                math.sin(expected_yaw / 2.0),
            ]
        )
        assert np.allclose(conditioned, expected)


def test_copied_validation_preserves_procedural_model_variation() -> None:
    pytest.importorskip("mujoco")
    with Workcell(WORKCELL_CONFIG) as live, Workcell(WORKCELL_CONFIG) as scratch:
        live.reset(seed=492404)
        scratch.reset(seed=492404)
        live.apply_object_variation(
            "puck_light",
            position_offset_xy_m=(0.00069, -0.00038),
            scale_multiplier=0.99836,
            mass_multiplier=0.99205,
            friction_multiplier=0.97003,
        )
        live.offset_static_entity(
            "inspection_pad",
            (0.0012, -0.0007, 0.0004),
        )

        _copy_task_local_model_configuration(live, scratch)

        assert np.array_equal(scratch.env.model.geom_size, live.env.model.geom_size)
        assert np.array_equal(
            scratch.env.model.geom_friction, live.env.model.geom_friction
        )
        assert np.array_equal(scratch.env.model.body_mass, live.env.model.body_mass)
        assert np.array_equal(
            scratch.env.model.body_inertia, live.env.model.body_inertia
        )
        assert np.array_equal(scratch.env.model.body_pos, live.env.model.body_pos)
        assert np.array_equal(
            scratch.env.model.actuator_gainprm, live.env.model.actuator_gainprm
        )
        assert np.array_equal(
            scratch.env.model.actuator_biasprm, live.env.model.actuator_biasprm
        )


def test_push_reapproaches_after_contact_loss_for_v13_diagnostic() -> None:
    pytest.importorskip("mujoco")
    raw_config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    seed = 1_394_802
    variation = sample_level4_procedural_variation(
        raw_config["level4_5b_procedural_expansion"],
        skill_name="push_object_to_target",
        seed=seed,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="push_object_to_target",
        goal_condition_id="push_cuboid_setup_slot_b_interior",
        seed=seed,
        procedural_variation=variation,
    ) as task:
        settings = record_demo._level4_scripted_expert_settings(
            args, task, "scripted_push"
        )
        push_config = DeterministicPushConfig.from_mapping(settings)
        expert = DeterministicPushExpert(
            finger_targets=record_demo._scripted_push_finger_targets(
                retargeter,
                open_targets,
                index_curl=float(
                    push_config.family_parameters["cuboid"]["index_curl"]
                ),
            ),
            config=push_config,
        )
        expert.reset(task, task.initial_world_state)
        assert expert.validation is not None
        assert expert.validation.valid is True

        terminal = task.current_state
        phases: list[str] = []
        for _ in range(500):
            requested, phase, done, reason = expert.step(terminal.world_state)
            phases.append(phase)
            assert reason is None
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=requested.base_position,
                orientation_quat=requested.base_orientation_wxyz,
            )
            task.env.set_joint_targets(requested.finger_targets)
            terminal = task.step(n_steps=push_config.sim_steps_per_action)
            if done and terminal.success:
                break

    assert terminal.success is True
    first_contact = phases.index("push_contact")
    assert "approach" in phases[first_contact + 1 :]


def test_push_selects_frozen_cuboid_contact_fallback_on_development_seed() -> None:
    pytest.importorskip("mujoco")
    assignment = next(
        item
        for item in build_level4_procedural_expansion_plan(DATASET_CONFIG)
        if item.sequence == 696
    )
    # Explicit development seed; intentionally independent of every frozen
    # held-out namespace and stable across future plan-version bumps.
    seed = 1_994_807
    raw_config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        raw_config["level4_5b_procedural_expansion"],
        skill_name=assignment.skill_name,
        seed=seed,
        boundary_case=assignment.repetition == 16,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name=assignment.skill_name,
        goal_condition_id=assignment.coverage_cell_id,
        seed=seed,
        procedural_variation=variation,
    ) as task:
        settings = record_demo._level4_scripted_expert_settings(
            args, task, "scripted_push"
        )
        push_config = DeterministicPushConfig.from_mapping(settings)
        expert = DeterministicPushExpert(
            finger_targets=record_demo._scripted_push_finger_targets(
                retargeter,
                open_targets,
                index_curl=float(
                    push_config.family_parameters["cuboid"]["index_curl"]
                ),
            ),
            config=push_config,
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.config.fingertip_lateral_offset_m == pytest.approx(0.0335)


def test_push_height_portfolio_resolves_v17_quarantined_diagnostic() -> None:
    pytest.importorskip("mujoco")
    raw_config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    seed = 1_794_808
    variation = sample_level4_procedural_variation(
        raw_config["level4_5b_procedural_expansion"],
        skill_name="push_object_to_target",
        seed=seed,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="push_object_to_target",
        goal_condition_id="push_cuboid_setup_slot_b_interior",
        seed=seed,
        procedural_variation=variation,
    ) as task:
        settings = record_demo._level4_scripted_expert_settings(
            args, task, "scripted_push"
        )
        push_config = DeterministicPushConfig.from_mapping(settings)
        expert = DeterministicPushExpert(
            finger_targets=record_demo._scripted_push_finger_targets(
                retargeter,
                open_targets,
                index_curl=float(
                    push_config.family_parameters["cuboid"]["index_curl"]
                ),
            ),
            config=push_config,
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None
    assert expert.config.fingertip_lateral_offset_m == pytest.approx(0.0335)
    assert expert.config.family_parameters["cuboid"][
        "control_height_m"
    ] == pytest.approx(0.08519952658523464)


def test_push_emits_contact_before_settle_for_v18_quarantined_seed() -> None:
    pytest.importorskip("mujoco")
    raw_config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    seed = 1_895_014
    variation = sample_level4_procedural_variation(
        raw_config["level4_5b_procedural_expansion"],
        skill_name="push_object_to_target",
        seed=seed,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="push_object_to_target",
        goal_condition_id="push_flat_puck_inspection_pad_interior",
        seed=seed,
        procedural_variation=variation,
    ) as task:
        settings = record_demo._level4_scripted_expert_settings(
            args, task, "scripted_push"
        )
        push_config = DeterministicPushConfig.from_mapping(settings)
        expert = DeterministicPushExpert(
            finger_targets=record_demo._scripted_push_finger_targets(
                retargeter,
                open_targets,
                index_curl=float(
                    push_config.family_parameters["flat_puck"]["index_curl"]
                ),
            ),
            config=push_config,
        )
        state = task.initial_world_state
        expert.reset(task, state)
        phases: list[str] = []
        done = False
        for _ in range(push_config.maximum_total_actions):
            action, phase, done, reason = expert.step(state)
            assert reason is None
            phases.append(phase)
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=action.base_position,
                orientation_quat=action.base_orientation_wxyz,
            )
            task.env.set_joint_targets(action.finger_targets)
            state = task.workcell.step(n_steps=push_config.sim_steps_per_action)
            if done:
                break
        result = task.workcell.create_task(
            "push_object_to_target", **task.goal
        ).evaluate(state)

    transitions = [
        phase
        for index, phase in enumerate(phases)
        if index == 0 or phase != phases[index - 1]
    ]
    assert done
    assert result.qualifies
    assert transitions[0] == "approach"
    assert transitions[-2:] == ["settle", "retract"]
    assert "push_contact" in transitions
    assert all(
        pair != ("approach", "settle")
        for pair in zip(transitions, transitions[1:], strict=False)
    )


def test_v5_puck_fix_resolves_quarantined_diagnostic_in_copied_preflight() -> None:
    pytest.importorskip("mujoco")
    variation = {
        "version": "level4/procedural-expansion-v1",
        "seed": 492404,
        "case_class": "nominal",
        "source_position_offset_xy_m": [
            0.0006935931111826921,
            -0.00037799968027056976,
        ],
        "goal_position_offset_m": [
            -0.00042735114578655695,
            -0.0001341133619360685,
            -0.00023571941227754848,
        ],
        "object_scale_multiplier": 0.9983558225998064,
        "object_mass_multiplier": 0.9920485161431748,
        "object_friction_multiplier": 0.9700267831895241,
        "controller_position_offset_m": [
            -0.00026703505061634813,
            0.00003877293910485753,
            -0.00011593404743786418,
        ],
    }
    raw_retargeter = load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    retargeter = CurlRetargeter.from_mapping(raw_retargeter)
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_puck_light_return_bin_right",
        seed=492404,
        procedural_variation=variation,
    ) as task:
        grasp = DeterministicGraspLiftConfig.from_mapping(
            record_demo._level4_scripted_expert_settings(
                args, task, "scripted_grasp"
            )
        )
        place = DeterministicPlaceConfig.from_mapping(
            record_demo._level4_scripted_expert_settings(
                args, task, "scripted_place"
            )
        )
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=grasp,
            place_config=place,
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None


def test_v5_puck_fix_selects_frozen_fallback_for_training_edge_case() -> None:
    pytest.importorskip("mujoco")
    assignment = next(
        item
        for item in build_level4_procedural_expansion_plan(DATASET_CONFIG)
        if item.sequence == 288
    )
    raw_retargeter = load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    retargeter = CurlRetargeter.from_mapping(raw_retargeter)
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name=assignment.skill_name,
        goal_condition_id=assignment.coverage_cell_id,
        seed=assignment.seed,
        procedural_variation=assignment.variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.grasp_config.family_templates[
        "flat_puck"
    ].object_relative_position_m == (-0.004, 0.0, 0.020)


def test_v6_puck_fix_resolves_second_quarantined_diagnostic() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=592410,
    )
    raw_retargeter = load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    retargeter = CurlRetargeter.from_mapping(raw_retargeter)
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_puck_light_return_bin_right",
        seed=592410,
        procedural_variation=variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.grasp_config.family_templates[
        "flat_puck"
    ].object_relative_position_m == (-0.006, 0.0, 0.020)


def test_v7_large_block_contact_fix_resolves_quarantined_diagnostic() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=692502,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_block_large_return_bin_left",
        seed=692502,
        procedural_variation=variation,
    ) as task:
        geom_id = task.workcell._require_mujoco_name("geom", "block_large_geom")
        assert task.env.model.geom_friction[geom_id, 0] > 2.5
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None


def test_v8_cuboid_fallback_resolves_quarantined_diagnostic() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=791402,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_block_small_return_bin_right",
        seed=791402,
        procedural_variation=variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.grasp_config.family_templates[
        "cuboid"
    ].negative_object_yaw_to_wrist_yaw_gain == -1.0


def test_v9_large_block_portfolio_resolves_quarantined_diagnostic() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=892504,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_block_large_return_bin_left",
        seed=892504,
        procedural_variation=variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None
    assert expert.grasp_config.family_templates[
        "cuboid"
    ].negative_object_yaw_to_wrist_yaw_gain == -0.25


def test_v9_heavy_puck_fallback_resolves_retired_boundary_diagnostic() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=893616,
        boundary_case=True,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_puck_heavy_inspection_pad",
        seed=893616,
        procedural_variation=variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None
    assert expert.grasp_config.family_templates[
        "flat_puck"
    ].object_relative_position_m == (-0.006, 0.0, 0.016)


def test_v10_puck_fallback_resolves_v9_quarantined_diagnostic() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=992408,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_puck_light_return_bin_right",
        seed=992408,
        procedural_variation=variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None
    assert expert.grasp_config.family_templates[
        "flat_puck"
    ].object_relative_position_m == (-0.004, 0.0, 0.016)


@pytest.mark.parametrize(
    ("cell_id", "seed", "expected_first_done_success"),
    (
        ("pp_puck_heavy_inspection_pad", 1093607, True),
        ("pp_block_large_return_bin_left", 1292513, None),
    ),
)
def test_recording_waits_for_recomputed_terminal_dwell(
    cell_id: str,
    seed: int,
    expected_first_done_success: bool | None,
) -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=seed,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id=cell_id,
        seed=seed,
        procedural_variation=variation,
    ) as task:
        place_config = DeterministicPlaceConfig.from_mapping(
            record_demo._level4_scripted_expert_settings(
                args, task, "scripted_place"
            )
        )
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=place_config,
        )
        terminal = task.current_state
        expert.reset(task, terminal.world_state)
        first_done_success: bool | None = None
        for _ in range(place_config.maximum_total_actions):
            requested, phase, done, reason = expert.step(terminal.world_state)
            assert reason is None
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=requested.base_position,
                orientation_quat=requested.base_orientation_wxyz,
            )
            task.env.set_joint_targets(requested.finger_targets)
            terminal = record_demo._step_scripted_workcell(
                task,
                phase=phase,
                n_steps=place_config.sim_steps_per_action,
                orientation_hold_chunk_steps=place_config.orientation_hold_chunk_steps,
            )
            if done and first_done_success is None:
                first_done_success = terminal.success
            if done and terminal.success:
                break

    if expected_first_done_success is not None:
        assert first_done_success is expected_first_done_success
    assert terminal.success is True


def test_v12_puck_portfolio_resolves_v11_quarantined_diagnostic() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=1192408,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_puck_light_return_bin_right",
        seed=1192408,
        procedural_variation=variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None
    assert expert.grasp_config.family_templates[
        "flat_puck"
    ].object_relative_position_m == (-0.006, 0.0, 0.020)


def test_pick_place_preflight_rejects_v16_nonreplayable_grasp() -> None:
    pytest.importorskip("mujoco")
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    variation = sample_level4_procedural_variation(
        config["level4_5b_procedural_expansion"],
        skill_name="pick_place_sequence",
        seed=1692715,
    )
    retargeter = CurlRetargeter.from_mapping(
        load_curl_retargeter_config(ROOT / "configs/level1_teleop.yaml")
    )
    open_targets = run_level1_teleop.build_full_hand_targets(
        retargeter, no_hand_features()
    )
    closed_targets = record_demo._scripted_closed_finger_targets(
        retargeter, open_targets
    )
    args = SimpleNamespace(enforce_frozen_cell_owner=True)

    with WorkcellPilotTask(
        workcell_config=WORKCELL_CONFIG,
        dataset_config=DATASET_CONFIG,
        skill_name="pick_place_sequence",
        goal_condition_id="pp_block_large_setup_slot_a",
        seed=1692715,
        procedural_variation=variation,
    ) as task:
        expert = DeterministicPickPlaceExpert(
            open_finger_targets=open_targets,
            closed_finger_targets=closed_targets,
            grasp_config=DeterministicGraspLiftConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_grasp"
                )
            ),
            place_config=DeterministicPlaceConfig.from_mapping(
                record_demo._level4_scripted_expert_settings(
                    args, task, "scripted_place"
                )
            ),
        )
        expert.reset(task, task.initial_world_state)

    assert expert.validation is not None
    assert expert.validation.valid is True
    assert expert.validation.reason is None
    assert expert.grasp_config.family_templates[
        "cuboid"
    ].object_relative_position_m == (0.0, 0.0, 0.024)


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
