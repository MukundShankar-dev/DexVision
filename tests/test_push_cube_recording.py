from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.apps import record_demo
from dexvision.logging.demo_logger import DemoLogger, DemoStepData, load_logged_demo
from dexvision.sim.tasks import (
    DEFAULT_TASK_BOARD_MODEL,
    PushCubeConfig,
    PushCubeParameters,
    PushCubeTask,
)


def test_push_cube_parser_accepts_typed_task_parameters() -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "push_cube_to_target",
            "--object-id",
            "push_cube",
            "--target-zone-id",
            "push_target_right",
            "--approach-side",
            "left",
        ]
    )

    record_demo._validate_recording_args(args)

    assert args.object_id == "push_cube"
    assert args.target_zone_id == "push_target_right"
    assert args.target_pose is None
    assert args.approach_side == "left"
    assert record_demo._minimum_realtime_sim_steps(
        simulation_timestep=0.002,
        control_rate_hz=30.0,
    ) == 17


def test_push_cube_target_pose_and_zone_are_mutually_exclusive() -> None:
    parser = record_demo.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--task",
                "push_cube_to_target",
                "--target-zone-id",
                "push_target_left",
                "--target-pose",
                "0.1",
                "0.0",
                "-0.015",
            ]
        )


@pytest.mark.parametrize(("response", "expected"), [("yes", True), ("no", False)])
def test_push_cube_pilot_requires_operator_label(
    response: str,
    expected: bool,
) -> None:
    args = record_demo.build_parser().parse_args(["--task", "push_cube_to_target"])

    assert (
        record_demo._resolve_operator_success_label(
            args,
            input_fn=lambda _prompt: response,
        )
        is expected
    )


def test_push_cube_recording_uses_reachable_neutral_and_has_no_timeout_stop() -> None:
    pytest.importorskip("mujoco")
    with PushCubeTask(DEFAULT_TASK_BOARD_MODEL, enforce_timeout=False) as task:
        initial = task.reset(
            seed=0,
            parameters=PushCubeParameters(target_zone_id="push_target_left"),
        )
        base_config = record_demo._push_cube_base_config(
            record_demo.HandBaseControlConfig(
                enabled=True,
                enable_base_orientation=True,
                enable_depth_control=True,
            ),
            initial_state=initial,
        )
        snapshot = record_demo._teleop_config_with_effective_base(
            {},
            base_config=base_config,
            neutral_orientation=initial.initial_base_orientation,
        )

    assert base_config.neutral_base_position == pytest.approx(
        initial.initial_base_position
    )
    assert base_config.workspace_limits.minimum == pytest.approx(
        [-0.22, -0.07, -0.24]
    )
    assert base_config.workspace_limits.maximum == pytest.approx(
        [0.04, -0.07, -0.24]
    )
    assert initial.initial_base_orientation == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert base_config.enable_base_orientation is False
    assert base_config.base_position_scale_x == pytest.approx(0.0)
    assert base_config.base_position_scale_y == pytest.approx(0.0)
    assert base_config.base_smoothing_alpha == pytest.approx(0.40)
    assert base_config.depth_scale == pytest.approx(0.35)
    assert base_config.depth_min == pytest.approx(-0.22)
    assert base_config.depth_max == pytest.approx(0.04)
    assert base_config.depth_smoothing_alpha == pytest.approx(0.40)
    assert base_config.depth_deadband == pytest.approx(0.01)
    assert base_config.max_position_step == pytest.approx(0.02)
    assert snapshot["base_control"]["workspace_limits"]["min"] == pytest.approx(
        [-0.22, -0.07, -0.24]
    )
    assert snapshot["base_control"]["task_override"] == "push_cube_to_target"
    assert snapshot["base_control"]["enable_base_control"] is True
    assert snapshot["base_control"]["enable_base_orientation"] is False
    assert snapshot["base_control"]["enable_depth_control"] is True
    assert snapshot["base_control"]["depth_scale"] == pytest.approx(0.35)
    assert snapshot["base_control"]["base_smoothing_alpha"] == pytest.approx(0.40)
    assert record_demo._task_should_stop_recording(
        SimpleNamespace(success=False, failure_reason="timeout")
    ) is False
    assert record_demo._task_should_stop_recording(
        SimpleNamespace(success=False, failure_reason="object_workspace_bounds")
    ) is True
    assert record_demo._task_should_stop_recording(
        SimpleNamespace(success=True, failure_reason=None)
    ) is True

    with PushCubeTask(
        DEFAULT_TASK_BOARD_MODEL,
        config=PushCubeConfig(max_episode_steps=1),
        enforce_timeout=False,
    ) as task:
        task.reset(
            parameters=PushCubeParameters(
                target_pose=(0.09, -0.07, -0.015),
            )
        )
        state_after_nominal_timeout = task.step()

    assert state_after_nominal_timeout.success is False
    assert state_after_nominal_timeout.failure_reason is None


def test_push_cube_aligned_reset_is_stationary_and_forward_motion_reaches_cube() -> None:
    pytest.importorskip("mujoco")
    with PushCubeTask(DEFAULT_TASK_BOARD_MODEL, enforce_timeout=False) as task:
        initial = task.reset(
            seed=0,
            parameters=PushCubeParameters(target_zone_id="push_target_left"),
        )
        start = initial.object_position.copy()
        task.env.step(n_steps=150)
        settled = task.get_state().object_position.copy()

        for x_position in np.linspace(
            initial.initial_base_position[0],
            initial.initial_base_position[0] + 0.22,
            50,
        ):
            task.env.set_mocap_pose(
                task.config.base_target_body,
                position=(
                    x_position,
                    initial.initial_base_position[1],
                    initial.initial_base_position[2],
                ),
                orientation_quat=initial.initial_base_orientation,
            )
            task.env.step(n_steps=8)
        pushed = task.get_state().object_position.copy()
        physical_orientation = task.env.data.qpos[3:7].copy()

    assert np.linalg.norm(settled - start) < 0.002
    assert pushed[0] > settled[0] + 0.05
    assert abs(pushed[2] - settled[2]) < 0.02
    assert abs(
        float(np.dot(physical_orientation, initial.initial_base_orientation))
    ) > 0.999


def test_push_cube_scene_keeps_palm_contacts_and_disables_below_table_arm() -> None:
    pytest.importorskip("mujoco")
    with PushCubeTask(DEFAULT_TASK_BOARD_MODEL, enforce_timeout=False) as task:
        mujoco = task.env._mujoco

        def geom_state(body_name: str) -> tuple[np.ndarray, np.ndarray]:
            body_id = mujoco.mj_name2id(
                task.env.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            geom_ids = np.flatnonzero(task.env.model.geom_bodyid == body_id)
            return (
                task.env.model.geom_contype[geom_ids],
                task.env.model.geom_rgba[geom_ids, 3],
            )

        forearm_flags, forearm_alpha = geom_state("rh_forearm")
        wrist_flags, wrist_alpha = geom_state("rh_wrist")
        palm_flags, palm_alpha = geom_state("rh_palm")

    assert np.all(forearm_flags == 0)
    assert np.all(wrist_flags == 0)
    assert np.all(forearm_alpha == 0)
    assert np.all(wrist_alpha == 0)
    assert np.any(palm_flags > 0)
    assert np.any(palm_alpha > 0)


def test_push_cube_viewer_uses_clear_three_quarter_camera() -> None:
    camera = SimpleNamespace(
        lookat=np.zeros(3, dtype=np.float64),
        distance=0.0,
        azimuth=0.0,
        elevation=0.0,
    )

    record_demo._configure_push_cube_viewer(SimpleNamespace(cam=camera))

    assert camera.lookat == pytest.approx([-0.02, 0.01, 0.03])
    assert camera.distance == pytest.approx(0.62)
    assert camera.azimuth == pytest.approx(35.0)
    assert camera.elevation == pytest.approx(-25.0)


def test_realtime_simulation_steps_remove_mocap_follow_lag() -> None:
    pytest.importorskip("mujoco")
    final_errors: dict[int, float] = {}
    for sim_steps in (2, 17):
        with PushCubeTask(DEFAULT_TASK_BOARD_MODEL, enforce_timeout=False) as task:
            initial = task.reset(seed=0)
            target = initial.initial_base_position.copy()
            for x_position in np.linspace(target[0], -0.40, 45):
                target[0] = x_position
                task.env.set_mocap_pose(
                    task.config.base_target_body,
                    position=target,
                    orientation_quat=initial.initial_base_orientation,
                )
                task.env.step(n_steps=sim_steps)
            final_errors[sim_steps] = float(
                np.linalg.norm(task.env.data.qpos[:3] - target)
            )

    assert final_errors[17] < 0.005
    assert final_errors[17] < final_errors[2] / 5.0


def test_push_cube_episode_logs_full_action_object_and_task_state(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "push_cube_to_target",
            "--object-id",
            "push_cube",
            "--target-zone-id",
            "push_target_center",
            "--approach-side",
            "front",
            "--output",
            str(tmp_path / "episode"),
        ]
    )
    record_demo._validate_recording_args(args)
    args.sim_steps_per_frame = 17

    with PushCubeTask(DEFAULT_TASK_BOARD_MODEL, enforce_timeout=False) as task:
        initial = task.reset(
            seed=7,
            parameters=PushCubeParameters(
                object_id=args.object_id,
                target_zone_id=args.target_zone_id,
                approach_side=args.approach_side,
            ),
        )
        target_names = tuple(
            task.spec.action_schema.representation_notes["finger_target_names"]
        )
        metadata = record_demo._metadata(
            args=args,
            episode_id="push-cube-pilot-001",
            raw_config={},
            model_path=DEFAULT_TASK_BOARD_MODEL,
            target_names=target_names,
            observation_schema=task.spec.observation_schema,
            synthetic=False,
            push_cube_task=task,
            push_cube_initial_state=initial,
        )
        base_position, base_orientation = task.env.get_mocap_pose(
            task.config.base_target_body
        )
        action = np.concatenate(
            (
                base_position,
                base_orientation,
                np.zeros(len(target_names), dtype=np.float64),
            )
        )
        logger = DemoLogger(
            args.output,
            action_schema=task.spec.action_schema,
            observation_schema=task.spec.observation_schema,
        )
        logger.start_episode(metadata)
        logger.append(
            DemoStepData(
                features=np.zeros(len(record_demo.FEATURE_FIELDS), dtype=np.float64),
                action=action,
                robot_state=task.robot_state_vector(),
                tracking_quality=np.asarray(
                    [1.0, 1.0, 0.95, 0.95, 0.0, 0.0],
                    dtype=np.float64,
                ),
                timestamp=0.0,
                object_state=initial.as_object_state(),
                task_state=initial.as_task_state(),
            )
        )
        logger.close(success=False)

    episode = load_logged_demo(args.output)
    task_config = episode.metadata["task_config"]
    split_action = task.spec.action_schema.split(episode.actions)

    assert episode.actions.shape == (1, task.spec.action_schema.action_dim)
    assert split_action["base_position_target"][0] == pytest.approx(base_position)
    assert split_action["base_orientation_target"][0] == pytest.approx(base_orientation)
    assert episode.object_states is not None
    assert episode.object_states.shape == (1, 13)
    assert episode.task_states is not None
    assert episode.task_states.shape[1] == initial.as_task_state().size
    assert task_config["resolved_object_id"] == "push_cube"
    assert task_config["resolved_target_source"] == "push_target_center"
    assert task_config["resolved_approach_side"] == "front"
    assert task_config["target_radius"] == pytest.approx(task.config.target_radius_m)
    assert task_config["success_dwell_steps"] == task.config.success_dwell_steps
    assert task_config["recording_timeout_enabled"] is False
    assert task_config["initial_object_position"] == pytest.approx(
        initial.initial_object_position
    )
    assert episode.metadata["recording"]["sim_steps_per_frame"] == 17
