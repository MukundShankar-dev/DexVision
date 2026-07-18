from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo
from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoLoggerError,
    DemoStepData,
    build_level2_observation_schema,
)
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.sim.mujoco_env import MujocoEnv
from dexvision.sim.tasks import (
    ReachTouchTargetConfig,
    ReachTouchTargetParameters,
    ReachTouchTargetTask,
    reach_touch_failure_reason,
)


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"
TASK_BOARD_MODEL = ROOT / "assets" / "mujoco" / "task_board_scene.xml"


def test_reach_touch_parser_accepts_checkpoint_command_and_target_selection() -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "reach_touch_target",
            "--retargeter",
            "curl",
            "--output",
            "data/demos/raw/reach_touch_target/2026-06-14_001",
            "--level1-13-full",
            "--target-site",
            "reach_target_right",
            "--task-seed",
            "7",
        ]
    )

    record_demo._apply_recording_presets(args)
    record_demo._validate_recording_args(args)

    assert args.target_site == "reach_target_right"
    assert args.task_seed == 7
    assert args.show_camera_window is True
    assert args.viewer is True
    assert args.start_on_calibration is True


def test_reach_touch_uses_task_board_model_by_default() -> None:
    args = record_demo.build_parser().parse_args(
        ["--task", "reach_touch_target", "--config", str(TELEOP_CONFIG_PATH)]
    )

    model_path = record_demo._resolve_recording_model_path(
        args=args,
        raw_config={"model_path": "assets/mujoco/hand_scene.xml"},
    )

    assert model_path == record_demo.DEFAULT_TASK_BOARD_MODEL


def test_reach_touch_rejects_synthetic_pilot_recording() -> None:
    args = record_demo.build_parser().parse_args(
        ["--task", "reach_touch_target", "--synthetic"]
    )

    with pytest.raises(ValueError, match="live task state"):
        record_demo._validate_recording_args(args)


def test_live_reach_touch_does_not_terminate_on_workspace_bounds() -> None:
    config = ReachTouchTargetConfig()

    failure = reach_touch_failure_reason(
        touch_position=(-0.18048895, -0.12265543, 0.44333925),
        step_count=189,
        max_episode_steps=config.max_episode_steps,
        workspace_min=config.workspace_min,
        workspace_max=config.workspace_max,
        enforce_workspace_bounds=config.terminate_on_workspace_bounds,
    )

    assert failure is None
    assert config.terminate_on_workspace_bounds is False
    assert (
        reach_touch_failure_reason(
            touch_position=(-0.18048895, -0.12265543, 0.44333925),
            step_count=189,
            max_episode_steps=config.max_episode_steps,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
        )
        == "workspace_bounds"
    )


def test_task_board_targets_are_in_front_of_neutral_palm() -> None:
    mujoco = pytest.importorskip("mujoco")

    with ReachTouchTargetTask(TASK_BOARD_MODEL) as task:
        task.reset(
            parameters=ReachTouchTargetParameters(target_site="reach_target_center")
        )
        palm_id = mujoco.mj_name2id(
            task.env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "rh_palm",
        )
        palm_rotation = np.asarray(
            task.env.data.xmat[palm_id],
            dtype=np.float64,
        ).reshape(3, 3)
        palm_outward_normal = palm_rotation @ np.asarray(
            [0.0, -1.0, 0.0],
            dtype=np.float64,
        )

        approaches = []
        for target_site in task.config.target_sites:
            state = task.reset(
                parameters=ReachTouchTargetParameters(target_site=target_site)
            )
            approaches.append(state.target_position - state.touch_position)

    center_approach = approaches[1] / np.linalg.norm(approaches[1])
    assert np.dot(center_approach, palm_outward_normal) > 0.95
    assert all(np.dot(approach, palm_outward_normal) > 0.08 for approach in approaches)


def test_reach_target_visuals_are_enabled_by_default() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(TASK_BOARD_MODEL))
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)

    active_target_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "active_reach_target",
    )
    assert active_target_id >= 0
    assert options.geomgroup[int(model.geom_group[active_target_id])] == 1

    for site_name in ReachTouchTargetConfig().target_sites:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        assert site_id >= 0
        assert options.sitegroup[int(model.site_group[site_id])] == 1


def test_task_board_is_visual_only_while_active_target_remains_collidable() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(TASK_BOARD_MODEL))

    board_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "task_board",
    )
    active_target_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "active_reach_target",
    )

    assert board_id >= 0
    assert model.geom_contype[board_id] == 0
    assert model.geom_conaffinity[board_id] == 0
    assert active_target_id >= 0
    assert model.geom_contype[active_target_id] != 0
    assert model.geom_conaffinity[active_target_id] != 0


@pytest.mark.parametrize(
    ("target_site", "base_position"),
    [
        ("reach_target_left", (0.10, -0.10, 0.113)),
        ("reach_target_center", (0.10, 0.0, 0.153)),
        ("reach_target_right", (0.10, 0.06, 0.173)),
    ],
)
def test_physical_palm_target_contact_accumulates_success_dwell(
    target_site: str,
    base_position: tuple[float, float, float],
) -> None:
    pytest.importorskip("mujoco")

    with ReachTouchTargetTask(TASK_BOARD_MODEL) as task:
        task.reset(
            parameters=ReachTouchTargetParameters(target_site=target_site)
        )
        task.env.set_mocap_pose(
            task.config.base_target_body,
            position=base_position,
            orientation_quat=(1.0, 0.0, 0.0, 0.0),
        )
        state = task.get_state()
        for _ in range(80):
            state = task.step(n_steps=5)
            if state.success:
                break

    assert state.palm_contact is True
    assert state.within_success_distance is True
    assert state.distance_to_target <= task.config.success_distance_m
    assert state.dwell_steps >= task.config.success_dwell_steps
    assert state.success is True


@pytest.mark.parametrize(
    ("response", "expected"),
    [("yes", True), ("s", True), ("no", False), ("failure", False)],
)
def test_reach_touch_operator_label_prompt(response: str, expected: bool) -> None:
    args = record_demo.build_parser().parse_args(["--task", "reach_touch_target"])

    result = record_demo._resolve_operator_success_label(
        args,
        input_fn=lambda _prompt: response,
    )

    assert result is expected


def test_reach_touch_operator_label_requires_clear_answer() -> None:
    args = record_demo.build_parser().parse_args(["--task", "reach_touch_target"])

    with pytest.raises(DemoLoggerError, match="operator label must be"):
        record_demo._resolve_operator_success_label(
            args,
            input_fn=lambda _prompt: "maybe",
        )


def test_reach_touch_episode_saves_task_state_goal_initial_state_and_label(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "reach_touch_target",
            "--config",
            str(TELEOP_CONFIG_PATH),
            "--target-site",
            "reach_target_left",
            "--failure",
        ]
    )
    record_demo._validate_recording_args(args)

    with ReachTouchTargetTask(TASK_BOARD_MODEL) as task:
        initial_task_state = task.reset(
            seed=args.task_seed,
            parameters=ReachTouchTargetParameters(target_site=args.target_site),
        )
        target_names = tuple(
            task.spec.action_schema.representation_notes["finger_target_names"]
        )
        initial_robot_state = task.env.get_state()
        (
            qpos_names,
            qvel_names,
            actuator_names,
            finger_qpos_indices,
            finger_qvel_indices,
            finger_joint_names,
        ) = record_demo.mujoco_observation_order(task.env)
        observation_schema = build_level2_observation_schema(
            robot_qpos_dim=initial_robot_state.qpos.size,
            robot_qvel_dim=initial_robot_state.qvel.size,
            finger_target_dim=initial_robot_state.ctrl.size,
            tracking_quality_dim=len(record_demo.TRACKING_QUALITY_FIELDS),
            robot_qpos_names=qpos_names,
            robot_qvel_names=qvel_names,
            actuator_names=actuator_names,
            finger_joint_qpos_indices=finger_qpos_indices,
            finger_joint_qvel_indices=finger_qvel_indices,
            finger_joint_names=finger_joint_names,
            tracking_quality_names=record_demo.TRACKING_QUALITY_FIELDS,
            object_state_dim=3,
            task_state_dim=initial_task_state.as_task_state().size,
            target_state_dim=3,
            success_metric_dim=8,
        )
        metadata = record_demo._metadata(
            args=args,
            episode_id="reach_touch_pilot_001",
            raw_config={"base_control": {"mocap_body": "dexvision_hand_base_target"}},
            model_path=TASK_BOARD_MODEL,
            target_names=target_names,
            observation_schema=observation_schema,
            synthetic=False,
            reach_touch_task=task,
            reach_touch_initial_state=initial_task_state,
        )
        logger = DemoLogger(
            tmp_path / "reach_touch_pilot_001",
            action_schema=task.spec.action_schema,
            observation_schema=observation_schema,
        )
        logger.start_episode(metadata)

        task_state = task.step(n_steps=1)
        robot_state = task.env.get_state()
        base_position, base_orientation = task.env.get_mocap_pose(
            task.config.base_target_body
        )
        targets = dict(zip(target_names, robot_state.ctrl, strict=True))
        logger.append(
            DemoStepData(
                features=np.zeros(len(record_demo.FEATURE_FIELDS), dtype=np.float64),
                action=record_demo.action_vector(
                    base_position=base_position,
                    base_orientation=base_orientation,
                    targets=targets,
                    target_names=target_names,
                ),
                robot_state=task.robot_state_vector(),
                tracking_quality=np.asarray(
                    [1.0, 1.0, 0.9, 0.9, 0.0, 0.0],
                    dtype=np.float64,
                ),
                timestamp=1.0,
                landmarks=np.zeros((21, 3), dtype=np.float64),
                object_state=task_state.target_position,
                task_state=task_state.as_task_state(),
            )
        )
        episode = logger.close(success=False)

    output = tmp_path / "reach_touch_pilot_001"
    saved_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    saved_object_states = np.load(output / "object_states.npy")
    saved_task_states = np.load(output / "task_states.npy")

    assert episode.success is False
    assert saved_metadata["success"] is False
    assert saved_metadata["robot_model"] == str(TASK_BOARD_MODEL)
    assert saved_metadata["task_config"]["requires_task_state"] is True
    assert saved_metadata["task_config"]["requires_success_metric_inputs"] is True
    assert saved_metadata["task_config"]["resolved_target_source"] == "reach_target_left"
    assert saved_metadata["task_config"]["target_position"] == pytest.approx(
        initial_task_state.target_position
    )
    assert saved_metadata["task_config"]["initial_robot_qpos"] == pytest.approx(
        initial_task_state.initial_robot_qpos
    )
    assert saved_metadata["action_schema"]["finger_actuator_targets"] == [
        7,
        7 + len(target_names),
    ]
    assert saved_task_states.shape == (1, task_state.as_task_state().size)
    assert saved_object_states == pytest.approx(task_state.target_position[None, :])

    loaded = load_replay_demo(output)
    with MujocoEnv(TASK_BOARD_MODEL) as replay_env:
        replay_result = replay_loaded_demo(loaded, replay_env)
        replay_target_position, replay_target_orientation = replay_env.get_mocap_pose(
            task.config.target_marker_body
        )
    assert replay_result.steps_replayed == 1
    assert replay_target_position == pytest.approx(initial_task_state.target_position)
    assert replay_target_orientation == pytest.approx([1.0, 0.0, 0.0, 0.0])
