from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo, run_level1_teleop
from dexvision.features.hand_features import no_hand_features
from dexvision.features.hand_base import ImagePalmCenterTarget
from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoStepData,
    build_level1_action_schema,
    build_level2_observation_schema,
)
from dexvision.logging.level4_collection import (
    Level4CollectionError,
    ManualReplayReview,
    PilotReview,
    WorkcellPilotTask,
    append_manual_replay_review,
    discover_pilot_episodes,
    load_level4_collection_config,
    load_manual_replay_reviews,
    load_pilot_review,
    save_pilot_review,
)
from dexvision.logging.session_manifest import load_session_manifest
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    load_curl_retargeter_config,
)
from dexvision.sim.hand_base_control import HandBaseControlConfig, WorkspaceLimits
from dexvision.sim.workcell_rate_control import (
    WorkcellRateControlConfig,
    WorkcellRateController,
    _apply_reach_virtual_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "level4_dataset.yaml"


class _FakeMocapEnv:
    def __init__(self) -> None:
        self.position = np.asarray([-0.16, 0.0, 0.20], dtype=np.float64)
        self.orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_mocap_pose(self, _name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.position.copy(), self.orientation.copy()

    def set_mocap_pose(
        self,
        _name: str,
        *,
        position: np.ndarray,
        orientation_quat: np.ndarray,
    ) -> None:
        self.position = np.asarray(position, dtype=np.float64)
        self.orientation = np.asarray(orientation_quat, dtype=np.float64)


def _image_target(
    palm_center: tuple[float, float], *, hand_scale: float = 0.20
) -> ImagePalmCenterTarget:
    return ImagePalmCenterTarget(
        palm_center=np.asarray(palm_center, dtype=np.float64),
        hand_scale=hand_scale,
        confidence=1.0,
        valid=True,
    )


def _write_episode(
    path: Path,
    *,
    episode_id: str = "pilot_000001",
    success: bool = True,
) -> None:
    path.mkdir(parents=True)
    metadata = {
        "episode_schema_version": "level4/episode-v1",
        "episode_id": episode_id,
        "recording_session_id": "session_a",
        "skill_name": "reach_object",
        "goal_condition_id": "reach_block_small_interior",
        "typed_goal": {"entity_id": "block_small"},
        "object_instance_ids": ["block_small"],
        "success": success,
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(path / "timestamps.npy", np.asarray([10.0, 12.0]))


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


def test_level4_config_freezes_pilot_protocol_and_defers_dial() -> None:
    config, protocol = load_level4_collection_config(CONFIG_PATH)

    assert protocol.minimum_genuine_sessions == 2
    assert protocol.accepted_by_group == {
        "reach": 5,
        "pick_place": 10,
        "push": 5,
        "button": 5,
    }
    assert protocol.optional_dial_decision == "deferred"
    assert config["pilot"]["expert_architecture_audit"] == {
        "version": "level4/expert-replay-audit-v1",
        "minimum_repeats_per_source_skill": 2,
        "required_source_skills": [
            "reach_object",
            "press_button",
            "push_object_to_target",
            "pick_object",
            "pick_place_sequence",
        ],
    }
    assert config["version"] == "level4/workcell-dataset-plan-v3"
    assert config["freeze"]["coverage_count_status"] == "final_accepted"
    assert config["source_mix"]["minimum_accepted_by_source"] == {
        "scripted": 111,
        "teleoperation": 0,
        "policy_rollout": 0,
        "corrective_intervention": 3,
    }


def test_pilot_review_is_append_only_and_discovery_keeps_failures(
    tmp_path: Path,
) -> None:
    accepted_dir = tmp_path / "session_a" / "episode_000001"
    failed_dir = tmp_path / "session_a" / "episode_000002"
    _write_episode(accepted_dir, episode_id="pilot_accepted")
    _write_episode(failed_dir, episode_id="pilot_failure", success=False)

    review_path = save_pilot_review(
        accepted_dir,
        _accepted_review("pilot_accepted"),
    )
    failed_review = PilotReview(
        episode_id="pilot_failure",
        schema_validation=True,
        timestamp_alignment=True,
        headless_replay=True,
        terminal_metric_recomputation=True,
        recomputed_success=False,
        operator_label_agreement=True,
        quality_thresholds=False,
        coverage_assignment=True,
        split_session_leakage_audit=True,
        expert_accepted=False,
        rejection_reasons=("recomputed_task_failure",),
    )
    save_pilot_review(failed_dir, failed_review)

    assert load_pilot_review(review_path).expert_accepted
    episodes = discover_pilot_episodes(tmp_path)
    assert [episode.episode_id for episode in episodes] == [
        "pilot_accepted",
        "pilot_failure",
    ]
    assert [episode.expert_accepted for episode in episodes] == [True, False]
    assert episodes[0].duration_seconds == pytest.approx(2.0)
    with pytest.raises(Level4CollectionError, match="append-only"):
        save_pilot_review(accepted_dir, _accepted_review("pilot_accepted"))


def test_expert_acceptance_cannot_hide_failed_gate_or_operator_failure(
    tmp_path: Path,
) -> None:
    invalid = _accepted_review("pilot")
    invalid = PilotReview(
        **{
            **invalid.__dict__,
            "quality_thresholds": False,
        }
    )
    with pytest.raises(Level4CollectionError, match="failed gates: quality_thresholds"):
        invalid.validate()

    episode_dir = tmp_path / "failure"
    _write_episode(episode_dir, episode_id="pilot", success=False)
    with pytest.raises(Level4CollectionError, match="operator success label"):
        save_pilot_review(episode_dir, _accepted_review("pilot"))


def test_manual_replay_manifest_appends_without_rewriting_episode_review(
    tmp_path: Path,
) -> None:
    review = ManualReplayReview(
        episode_id="pilot_000001",
        verified_skills=("pick_object", "place_held_object"),
        passed=True,
        notes="Visible phases and terminal placement matched.",
    )

    path = append_manual_replay_review(tmp_path, review)

    assert path.name == "manual_replay_manifest.json"
    assert load_manual_replay_reviews(tmp_path) == (review,)
    with pytest.raises(Level4CollectionError, match="already exists"):
        append_manual_replay_review(tmp_path, review)


@pytest.mark.parametrize(
    ("skill_name", "cell_id", "expected_goal"),
    (
        ("reach_object", "reach_block_small_interior", "entity_id"),
        ("pick_object", "pp_block_small_return_bin_left", "object_id"),
        (
            "pick_place_sequence",
            "pp_block_small_return_bin_left",
            "target_id",
        ),
        (
            "push_object_to_target",
            "push_cuboid_setup_slot_a_interior",
            "target_zone",
        ),
        ("press_button", "press_008_centered_nominal", "button_id"),
    ),
)
def test_workcell_pilot_adapter_resolves_every_recordable_skill(
    skill_name: str,
    cell_id: str,
    expected_goal: str,
) -> None:
    pytest.importorskip("mujoco")

    with WorkcellPilotTask(
        workcell_config=ROOT / "configs" / "workcell.yaml",
        dataset_config=CONFIG_PATH,
        skill_name=skill_name,
        goal_condition_id=cell_id,
        seed=0,
    ) as task:
        state = task.current_state
        outline_id = task.workcell.env._mujoco.mj_name2id(
            task.workcell.env.model,
            task.workcell.env._mujoco.mjtObj.mjOBJ_BODY,
            task.workcell.config.scene["pilot_target_outline"],
        )
        marker_id = task.workcell.env._mujoco.mj_name2id(
            task.workcell.env.model,
            task.workcell.env._mujoco.mjtObj.mjOBJ_BODY,
            task.workcell.config.scene["pilot_goal_marker"],
        )
        outline_position = task.workcell.env.data.xpos[outline_id].copy()
        marker_position = task.workcell.env.data.xpos[marker_id].copy()

    assert expected_goal in task.goal
    assert state.online_phase == "approach"
    assert state.as_task_state().shape == (8,)
    assert state.as_object_state().shape == (78,)
    assert outline_position[2] > -0.5
    assert marker_position[2] > -0.5


def test_workcell_dry_run_needs_no_session_and_writes_no_manifest(
    tmp_path: Path,
) -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "reach_object",
            "--goal-condition-id",
            "reach_block_small_interior",
            "--workcell-dry-run",
            "--level4-pilot-dataset-dir",
            str(tmp_path),
            "--level4-dataset-config",
            str(CONFIG_PATH),
            "--workcell-config",
            str(ROOT / "configs" / "workcell.yaml"),
        ]
    )

    record_demo._prepare_level4_workcell_recording(args)
    record_demo._apply_recording_presets(args)
    record_demo._validate_recording_args(args)

    assert args.model.name == "workcell_scene.xml"
    assert not (tmp_path / "session_manifest.json").exists()
    assert args.output == record_demo.DEFAULT_OUTPUT


def test_rate_control_stops_at_center_and_integrates_nonlinear_velocity() -> None:
    env = _FakeMocapEnv()
    base_config = HandBaseControlConfig(
        enabled=True,
        base_fixed_z=0.20,
        position_offset=np.asarray([-0.16, 0.0, 0.0]),
        enable_depth_control=True,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-0.18, -0.18, 0.08]),
            maximum=np.asarray([0.22, 0.18, 0.24]),
        ),
    )
    rate_config = WorkcellRateControlConfig(
        goal_position=np.asarray([-0.14, -0.09, 0.148]),
        control_rate_hz=10.0,
        max_velocity_m_s=np.asarray([0.10, 0.10, 0.10]),
    )
    controller = WorkcellRateController(  # type: ignore[arg-type]
        env, base_config, rate_config
    )

    assert controller.calibrate_image_2d(_image_target((0.5, 0.5)))
    initial = env.position.copy()
    centered = controller.apply_image_2d(_image_target((0.52, 0.48)))
    assert centered.applied_target.position == pytest.approx(initial)

    moving = controller.apply_image_2d(_image_target((0.28, 0.5)))
    assert moving.control_mode == "image_2d_rate"
    assert moving.applied_target.position[1] < initial[1]
    moved = env.position.copy()

    controller.apply_image_2d(_image_target((0.5, 0.5)))
    assert env.position == pytest.approx(moved)


def test_rate_control_virtual_fixture_requires_high_transit_and_target_descent() -> None:
    config = WorkcellRateControlConfig(
        goal_position=np.asarray([-0.14, -0.09, 0.148]),
        transit_height_m=0.19,
        descent_radius_m=0.035,
    )

    outside, outside_limited = _apply_reach_virtual_fixture(
        previous=np.asarray([-0.16, 0.0, 0.20]),
        candidate=np.asarray([-0.16, 0.0, 0.17]),
        config=config,
    )
    over_target, target_limited = _apply_reach_virtual_fixture(
        previous=np.asarray([-0.14, -0.09, 0.19]),
        candidate=np.asarray([-0.14, -0.09, 0.17]),
        config=config,
    )
    below_goal, below_limited = _apply_reach_virtual_fixture(
        previous=np.asarray([-0.14, -0.09, 0.16]),
        candidate=np.asarray([-0.14, -0.09, 0.13]),
        config=config,
    )

    assert outside_limited
    assert outside[2] == pytest.approx(0.19)
    assert not target_limited
    assert over_target[2] == pytest.approx(0.17)
    assert below_limited
    assert below_goal[2] == pytest.approx(0.148)


def test_record_demo_prepares_append_only_workcell_session_and_output(
    tmp_path: Path,
) -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "reach_object",
            "--session-id",
            "pilot_train_001",
            "--operator-id",
            "operator_local_01",
            "--session-split",
            "train",
            "--goal-condition-id",
            "reach_block_small_interior",
            "--level4-pilot-dataset-dir",
            str(tmp_path),
            "--level4-dataset-config",
            str(CONFIG_PATH),
            "--workcell-config",
            str(ROOT / "configs" / "workcell.yaml"),
        ]
    )

    record_demo._prepare_level4_workcell_recording(args)
    record_demo._apply_recording_presets(args)
    record_demo._validate_recording_args(args)

    assert args.output == tmp_path / "pilot_train_001" / "episode_000001"
    assert args.model.name == "workcell_scene.xml"
    assert args.show_camera_window is True
    assert args.viewer is True
    manifest = load_session_manifest(tmp_path / "session_manifest.json")
    assert manifest.sessions[0].recording_session_id == "pilot_train_001"
    assert manifest.sessions[0].split == "train"


def test_workcell_recorder_uses_workcell_neutral_and_prompts_for_label() -> None:
    pytest.importorskip("mujoco")
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "reach_object",
            "--session-id",
            "pilot_train_002",
            "--operator-id",
            "operator_local_01",
            "--session-split",
            "train",
            "--goal-condition-id",
            "reach_block_small_interior",
        ]
    )
    base_config = record_demo._base_config(
        load_curl_retargeter_config(args.config), args=args
    )
    with WorkcellPilotTask(
        workcell_config=ROOT / "configs" / "workcell.yaml",
        dataset_config=CONFIG_PATH,
        skill_name="reach_object",
        goal_condition_id="reach_block_small_interior",
        seed=0,
    ) as task:
        configured = record_demo._workcell_pilot_base_config(base_config, task=task)
        neutral = np.asarray(task.workcell.config.scene["hand_neutral_position_m"])
        marker_id = task.workcell.env._mujoco.mj_name2id(
            task.workcell.env.model,
            task.workcell.env._mujoco.mjtObj.mjOBJ_BODY,
            task.workcell.config.scene["pilot_goal_marker"],
        )
        marker_position = task.workcell.env.data.xpos[marker_id].copy()
        outline_id = task.workcell.env._mujoco.mj_name2id(
            task.workcell.env.model,
            task.workcell.env._mujoco.mjtObj.mjOBJ_BODY,
            task.workcell.config.scene["pilot_target_outline"],
        )
        outline_position = task.workcell.env.data.xpos[outline_id].copy()
        target_position = np.asarray(
            task.initial_world_state.require_entity("block_small").position
        )

    assert configured.neutral_base_position == pytest.approx(neutral)
    assert configured.enable_base_orientation is False
    assert configured.base_position_scale_x == pytest.approx(0.75)
    assert configured.base_position_scale_y == pytest.approx(0.70)
    assert configured.base_smoothing_alpha == pytest.approx(0.80)
    assert configured.depth_scale == pytest.approx(0.55)
    assert configured.depth_smoothing_alpha == pytest.approx(0.80)
    assert marker_position == pytest.approx(task.goal["approach_pose"][:3])
    assert outline_position == pytest.approx(target_position)
    assert (
        record_demo._resolve_operator_success_label(args, input_fn=lambda _prompt: "n")
        is False
    )


def test_reach_goal_is_collision_free_and_triggers_automatic_stop() -> None:
    pytest.importorskip("mujoco")

    with WorkcellPilotTask(
        workcell_config=ROOT / "configs" / "workcell.yaml",
        dataset_config=CONFIG_PATH,
        skill_name="reach_object",
        goal_condition_id="reach_block_small_interior",
        seed=0,
    ) as task:
        approach_pose = task.goal["approach_pose"]
        assert approach_pose[2] >= 0.14
        task.env.set_mocap_pose(
            "dexvision_hand_base_target",
            position=approach_pose[:3],
            orientation_quat=approach_pose[3:],
        )
        for _ in range(5):
            state = task.step(n_steps=5)

    dense_state = state.as_task_state()
    assert state.success
    assert state.dwell_steps == 5
    assert state.task_values["maximum_scene_disturbance_m"] < 0.005
    assert dense_state[7] == pytest.approx(
        state.task_values["maximum_scene_disturbance_m"]
    )
    assert record_demo._task_should_stop_recording(state)


def test_workcell_recorder_advances_physics_at_realtime_rate() -> None:
    pytest.importorskip("mujoco")
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--control-rate-hz",
            "30",
            "--sim-steps-per-frame",
            "1",
        ]
    )
    with WorkcellPilotTask(
        workcell_config=ROOT / "configs" / "workcell.yaml",
        dataset_config=CONFIG_PATH,
        skill_name="reach_object",
        goal_condition_id="reach_block_small_interior",
        seed=0,
    ) as task:
        timestep = float(task.env.model.opt.timestep)
        record_demo._ensure_realtime_sim_steps(
            args=args,
            env=task.env,
            label="test",
        )

    assert args.sim_steps_per_frame == 17
    assert args.sim_steps_per_frame * timestep >= 1.0 / args.control_rate_hz


def test_workcell_pilot_episode_metadata_and_dense_arrays_validate(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "level4_workcell",
            "--skill",
            "reach_object",
            "--session-id",
            "pilot_train_002",
            "--operator-id",
            "operator_local_01",
            "--session-split",
            "train",
            "--goal-condition-id",
            "reach_block_small_interior",
            "--level4-pilot-dataset-dir",
            str(tmp_path),
            "--level4-dataset-config",
            str(CONFIG_PATH),
            "--workcell-config",
            str(ROOT / "configs" / "workcell.yaml"),
        ]
    )
    record_demo._prepare_level4_workcell_recording(args)
    record_demo._apply_recording_presets(args)
    record_demo._validate_recording_args(args)
    raw_config = load_curl_retargeter_config(args.config)
    retargeter = CurlRetargeter.from_mapping(raw_config)
    target_names = run_level1_teleop.robot_target_names(retargeter)
    action_schema = build_level1_action_schema(target_names)

    with WorkcellPilotTask(
        workcell_config=args.workcell_config,
        dataset_config=args.level4_dataset_config,
        skill_name=args.skill_name,
        goal_condition_id=args.goal_condition_id,
        seed=args.task_seed,
    ) as task:
        env = task.env
        initial = env.get_state()
        names = record_demo.mujoco_observation_order(env)
        observation_schema = build_level2_observation_schema(
            robot_qpos_dim=initial.qpos.size,
            robot_qvel_dim=initial.qvel.size,
            finger_target_dim=initial.ctrl.size,
            tracking_quality_dim=len(record_demo.TRACKING_QUALITY_FIELDS),
            robot_qpos_names=names[0],
            robot_qvel_names=names[1],
            actuator_names=names[2],
            finger_joint_qpos_indices=names[3],
            finger_joint_qvel_indices=names[4],
            finger_joint_names=names[5],
            tracking_quality_names=record_demo.TRACKING_QUALITY_FIELDS,
            object_state_dim=78,
            task_state_dim=8,
            target_state_dim=7,
            success_metric_dim=8,
        )
        logger = DemoLogger(
            args.output,
            action_schema=action_schema,
            observation_schema=observation_schema,
        )
        logger.start_episode(
            record_demo._metadata(
                args=args,
                episode_id="pilot_test_episode",
                raw_config=raw_config,
                model_path=args.model,
                target_names=target_names,
                observation_schema=observation_schema,
                synthetic=False,
                workcell_pilot_task=task,
            )
        )
        targets = run_level1_teleop.build_full_hand_targets(
            retargeter, no_hand_features()
        )
        env.set_joint_targets(targets)
        task_state = task.step(n_steps=1)
        state = env.get_state()
        position, orientation = env.get_mocap_pose("dexvision_hand_base_target")
        action = record_demo.action_vector(
            base_position=position,
            base_orientation=orientation,
            targets=targets,
            target_names=target_names,
        )
        logger.append(
            DemoStepData(
                features=np.zeros(len(record_demo.FEATURE_FIELDS)),
                action=action,
                robot_state=record_demo.robot_state_vector(
                    state,
                    base_position=position,
                    base_orientation=orientation,
                ),
                tracking_quality=np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
                timestamp=1.0,
                object_state=task_state.as_object_state(),
                task_state=task_state.as_task_state(),
                online_phase=task_state.online_phase,
                action_timestamp=1.0,
                task_timestamp=1.0,
                state_timestamp=1.0,
            )
        )
        episode = logger.close(success=False)

    assert episode.object_states is not None
    assert episode.object_states.shape == (1, 78)
    assert episode.task_states is not None
    assert episode.task_states.shape == (1, 8)
    assert episode.online_phases.tolist() == ["approach"]
