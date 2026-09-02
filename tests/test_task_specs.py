from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dexvision.sim.tasks import (
    BUTTON_PRESS_TASK_ID,
    PUSH_CUBE_TASK_ID,
    REACH_TOUCH_TARGET_TASK_ID,
    ButtonPressConfig,
    ButtonPressParameters,
    ButtonPressTask,
    PushCubeConfig,
    PushCubeParameters,
    PushCubeTask,
    ReachTouchTargetConfig,
    ReachTouchTargetParameters,
    ReachTouchTargetTask,
    TaskError,
    is_reach_touch_success,
    is_push_cube_success,
    push_cube_distance,
    push_cube_failure_reason,
    reach_distance,
    reach_touch_failure_reason,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "mujoco" / "task_board_scene.xml"


def test_task_board_scene_loads_with_reach_fixtures() -> None:
    pytest.importorskip("mujoco")

    with ReachTouchTargetTask(MODEL_PATH) as task:
        state = task.reset(seed=3)
        cube_geom_id = task.env._mujoco.mj_name2id(
            task.env.model,
            task.env._mujoco.mjtObj.mjOBJ_GEOM,
            "push_cube_geom",
        )

    assert task.env.model.nq > 0
    assert task.env.model.nu > 0
    assert state.target_source in task.config.target_sites
    assert np.all(np.isfinite(state.as_task_state()))
    assert task.env.model.geom_rgba[cube_geom_id, 3] == pytest.approx(0.0)
    assert task.env.model.geom_contype[cube_geom_id] == 0


def test_reach_touch_spec_declares_typed_goal_state_and_metric_contracts() -> None:
    pytest.importorskip("mujoco")

    with ReachTouchTargetTask(MODEL_PATH) as task:
        spec = task.spec

    assert spec.task_id == REACH_TOUCH_TARGET_TASK_ID
    assert spec.skill_name == REACH_TOUCH_TARGET_TASK_ID
    assert spec.required_objects == ("reach_target_marker",)
    assert spec.parameter_type is ReachTouchTargetParameters
    assert spec.parameter_schema["target_pose"]["shape"] == (3,)
    assert spec.parameter_schema["target_pose"]["coordinate_frame"] == "MuJoCo world"
    assert spec.parameter_schema["target_site"]["named_id_source"] == task.config.target_sites
    assert spec.max_episode_steps == 240
    assert "target_position" in spec.state_fields
    assert "initial_robot_qpos" in spec.state_fields
    assert spec.success_metric_inputs == (
        "target_position",
        "touch_position",
        "distance_to_target",
        "palm_contact",
    )
    spec.action_schema.validate()
    spec.observation_schema.validate()


def test_reset_is_deterministic_and_saves_sampled_target_and_initial_state() -> None:
    pytest.importorskip("mujoco")

    with ReachTouchTargetTask(MODEL_PATH) as task:
        first = task.reset(seed=17)
        second = task.reset(seed=17)
        vector = second.as_task_state()

    assert first.target_source == second.target_source
    assert first.target_index == second.target_index
    assert first.target_position == pytest.approx(second.target_position)
    assert first.initial_robot_qpos == pytest.approx(second.initial_robot_qpos)
    assert first.initial_robot_qvel == pytest.approx(second.initial_robot_qvel)
    assert second.target_position == pytest.approx(vector[0:3])
    assert second.palm_contact is False
    assert second.initial_base_position == pytest.approx(vector[14:17])
    assert second.initial_base_orientation == pytest.approx(vector[17:21])
    assert second.initial_robot_qpos.size == task.env.model.nq
    assert second.initial_robot_qvel.size == task.env.model.nv
    qpos_stop = 21 + task.env.model.nq
    assert second.initial_robot_qpos == pytest.approx(vector[21:qpos_stop])
    assert second.initial_robot_qvel == pytest.approx(vector[qpos_stop:])
    assert vector.shape == (21 + task.env.model.nq + task.env.model.nv,)


def test_reset_accepts_named_site_and_explicit_world_target_pose() -> None:
    pytest.importorskip("mujoco")

    with ReachTouchTargetTask(MODEL_PATH) as task:
        named = task.reset(
            seed=0,
            parameters=ReachTouchTargetParameters(target_site="reach_target_right"),
        )
        explicit = task.reset(
            seed=0,
            parameters=ReachTouchTargetParameters(target_pose=(0.02, -0.1, 0.48)),
        )

    assert named.target_source == "reach_target_right"
    assert named.target_index == 2
    assert explicit.target_source == "target_pose"
    assert explicit.target_index == -1
    assert explicit.target_position == pytest.approx([0.02, -0.1, 0.48])


def test_reach_touch_parameters_reject_ambiguous_or_invalid_targets() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ReachTouchTargetParameters(
            target_pose=(0.0, 0.0, 0.0),
            target_site="reach_target_center",
        )
    with pytest.raises(ValueError, match="three finite"):
        ReachTouchTargetParameters(target_pose=(0.0, float("nan"), 0.0))


def test_synthetic_reach_success_requires_distance_and_dwell() -> None:
    assert reach_distance([0.0, 0.0, 0.0], [0.01, 0.0, 0.0]) == pytest.approx(0.01)
    assert not is_reach_touch_success(
        distance_m=0.02,
        dwell_steps=4,
        distance_threshold_m=0.03,
        required_dwell_steps=5,
    )
    assert is_reach_touch_success(
        distance_m=0.02,
        dwell_steps=5,
        distance_threshold_m=0.03,
        required_dwell_steps=5,
    )
    assert not is_reach_touch_success(
        distance_m=0.04,
        dwell_steps=5,
        distance_threshold_m=0.03,
        required_dwell_steps=5,
    )
    assert not is_reach_touch_success(
        distance_m=0.01,
        dwell_steps=5,
        distance_threshold_m=0.03,
        required_dwell_steps=5,
        palm_contact=False,
    )


def test_synthetic_failure_metrics_cover_workspace_and_timeout() -> None:
    config = ReachTouchTargetConfig()

    assert (
        reach_touch_failure_reason(
            touch_position=(0.0, 0.0, 0.48),
            step_count=10,
            max_episode_steps=config.max_episode_steps,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
        )
        is None
    )
    assert (
        reach_touch_failure_reason(
            touch_position=(0.5, 0.0, 0.48),
            step_count=10,
            max_episode_steps=config.max_episode_steps,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
        )
        == "workspace_bounds"
    )
    assert (
        reach_touch_failure_reason(
            touch_position=(0.0, 0.0, 0.48),
            step_count=config.max_episode_steps,
            max_episode_steps=config.max_episode_steps,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
        )
        == "timeout"
    )


def test_task_state_and_robot_state_extraction_match_declared_widths() -> None:
    pytest.importorskip("mujoco")

    with ReachTouchTargetTask(MODEL_PATH) as task:
        reset_state = task.reset(
            parameters=ReachTouchTargetParameters(target_site="reach_target_center")
        )
        stepped_state = task.step(n_steps=2)
        robot_state = task.robot_state_vector()
        task_state = task.task_state_vector()
        robot_width = max(
            layout.maximum_column
            for layout in task.spec.observation_schema.layouts.values()
            if layout.source_array == "robot_states"
        )

    assert reset_state.step_count == 0
    assert stepped_state.step_count == 1
    assert np.isfinite(stepped_state.distance_to_target)
    assert robot_state.shape == (robot_width,)
    assert task_state.shape == task.spec.observation_schema.shapes["task_state"]


def test_check_task_progress_command_runs_headlessly() -> None:
    pytest.importorskip("mujoco")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexvision.apps.check_task",
            "--task",
            REACH_TOUCH_TARGET_TASK_ID,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "DexVision task check" in result.stdout
    assert "Task board scene load and deterministic reset: PASS" in result.stdout
    assert "Viewer: off" in result.stdout


def test_task_board_scene_loads_with_button_fixtures() -> None:
    pytest.importorskip("mujoco")

    with ButtonPressTask(MODEL_PATH) as task:
        state = task.reset(seed=3)

    assert state.button_id in task.config.button_ids
    assert state.press_depth == pytest.approx(0.0)
    assert np.all(np.isfinite(state.as_task_state()))
    for button_id in task.config.button_ids:
        joint_id = task.env._mujoco.mj_name2id(
            task.env.model,
            task.env._mujoco.mjtObj.mjOBJ_JOINT,
            f"{button_id}_joint",
        )
        site_id = task.env._mujoco.mj_name2id(
            task.env.model,
            task.env._mujoco.mjtObj.mjOBJ_SITE,
            f"{button_id}_site",
        )
        assert joint_id >= 0
        assert site_id >= 0


def test_button_spec_declares_parameter_and_terminal_state_schemas() -> None:
    pytest.importorskip("mujoco")

    with ButtonPressTask(MODEL_PATH) as task:
        spec = task.spec

    assert spec.task_id == BUTTON_PRESS_TASK_ID
    assert spec.skill_name == BUTTON_PRESS_TASK_ID
    assert spec.required_objects == task.config.button_ids
    assert spec.parameter_type is ButtonPressParameters
    assert spec.parameter_schema["button_id"]["named_id_source"] == (
        "button_left",
        "button_center",
        "button_right",
    )
    assert spec.parameter_schema["target_press_depth"]["units"] == "metres"
    assert spec.parameter_schema["pressed_state_target"]["type"] == "boolean"
    assert spec.parameter_schema["approach_pose"]["shape"] == (3,)
    assert spec.success_metric_inputs == (
        "press_depth",
        "target_press_depth",
        "button_pressed",
        "target_pressed_state",
        "dwell_steps",
    )
    assert spec.terminal_state_schema["success"]["terminal"] is True
    assert spec.terminal_state_schema["press_depth"]["units"] == "metres"
    assert "failure_reason" in spec.terminal_state_schema
    spec.action_schema.validate()
    spec.observation_schema.validate()


def test_button_reset_is_deterministic_and_saves_goal_and_initial_state() -> None:
    pytest.importorskip("mujoco")

    parameters = ButtonPressParameters(
        target_press_depth=0.01,
        approach_pose=(0.10, -0.02, 0.40),
    )
    with ButtonPressTask(MODEL_PATH) as task:
        first = task.reset(seed=23, parameters=parameters)
        second = task.reset(seed=23, parameters=parameters)
        vector = second.as_task_state()

    assert first.button_id == second.button_id
    assert first.button_index == second.button_index
    assert first.button_position == pytest.approx(second.button_position)
    assert first.initial_robot_qpos == pytest.approx(second.initial_robot_qpos)
    assert first.initial_robot_qvel == pytest.approx(second.initial_robot_qvel)
    assert second.target_press_depth == pytest.approx(0.01)
    assert second.target_pressed_state is True
    assert second.approach_pose_present is True
    assert second.approach_pose == pytest.approx([0.10, -0.02, 0.40])
    assert second.initial_button_depth == pytest.approx(0.0)
    assert vector.shape == task.spec.observation_schema.shapes["task_state"]
    assert vector.shape == (25 + task.env.model.nq + task.env.model.nv,)


def test_button_reset_accepts_named_button_and_pressed_state_target() -> None:
    pytest.importorskip("mujoco")

    with ButtonPressTask(MODEL_PATH) as task:
        state = task.reset(
            seed=0,
            parameters=ButtonPressParameters(
                button_id="button_center",
                pressed_state_target=True,
            ),
        )

    assert state.button_id == "button_center"
    assert state.button_index == 1
    assert state.target_pressed_state is True
    assert state.target_press_depth == pytest.approx(
        ButtonPressConfig().default_target_press_depth_m
    )


def test_button_parameters_reject_ambiguous_or_invalid_goals() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ButtonPressParameters(
            target_press_depth=0.01,
            pressed_state_target=True,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        ButtonPressParameters(target_press_depth=0.0)
    with pytest.raises(ValueError, match="three finite"):
        ButtonPressParameters(approach_pose=(0.0, float("nan"), 0.4))


def test_button_reset_rejects_press_depth_outside_joint_range() -> None:
    pytest.importorskip("mujoco")

    with ButtonPressTask(MODEL_PATH) as task:
        with pytest.raises(TaskError, match="outside the selected button joint range"):
            task.reset(
                parameters=ButtonPressParameters(target_press_depth=0.021)
            )


def test_button_state_extracts_press_depth_and_matches_declared_widths() -> None:
    mujoco = pytest.importorskip("mujoco")

    with ButtonPressTask(MODEL_PATH) as task:
        task.reset(
            parameters=ButtonPressParameters(
                button_id="button_left",
                target_press_depth=0.01,
            )
        )
        joint_id = mujoco.mj_name2id(
            task.env.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "button_left_joint",
        )
        qpos_address = int(task.env.model.jnt_qposadr[joint_id])
        qvel_address = int(task.env.model.jnt_dofadr[joint_id])
        task.env.data.qpos[qpos_address] = 0.011
        mujoco.mj_forward(task.env.model, task.env.data)
        state = task.get_state()
        task_state = task.task_state_vector()
        robot_state = task.robot_state_vector()
        robot_width = max(
            layout.maximum_column
            for layout in task.spec.observation_schema.layouts.values()
            if layout.source_array == "robot_states"
        )
        terminal = state
        for _ in range(task.config.success_dwell_steps):
            task.env.data.qpos[qpos_address] = 0.02
            task.env.data.qvel[qvel_address] = 0.0
            terminal = task.step(n_steps=1)

    assert state.press_depth == pytest.approx(0.011)
    assert state.button_pressed is True
    assert state.within_target is True
    assert state.dwell_steps == 0
    assert state.success is False
    assert task_state[0] == pytest.approx(0.011)
    assert task_state.shape == task.spec.observation_schema.shapes["task_state"]
    assert robot_state.shape == (robot_width,)
    assert terminal.dwell_steps == task.config.success_dwell_steps
    assert terminal.success is True
    assert terminal.failure_reason is None


def test_button_terminal_state_reports_timeout_without_success() -> None:
    pytest.importorskip("mujoco")

    config = ButtonPressConfig(max_episode_steps=1)
    with ButtonPressTask(MODEL_PATH, config=config) as task:
        task.reset(
            parameters=ButtonPressParameters(button_id="button_center")
        )
        terminal = task.step(n_steps=1)

    assert terminal.success is False
    assert terminal.failure_reason == "timeout"
    assert terminal.step_count == 1


def test_task_board_scene_loads_with_push_cube_fixtures() -> None:
    mujoco = pytest.importorskip("mujoco")

    with PushCubeTask(MODEL_PATH) as task:
        state = task.reset(seed=3)
        object_vector = state.as_object_state()
        task_vector = state.as_task_state()
        cube_geom_id = mujoco.mj_name2id(
            task.env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "push_cube_geom",
        )
        wall_geom_id = mujoco.mj_name2id(
            task.env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "task_board",
        )
        button_geom_ids = tuple(
            mujoco.mj_name2id(
                task.env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{button_id}_geom",
            )
            for button_id in ("button_left", "button_center", "button_right")
        )

    assert state.object_id == "push_cube"
    assert state.target_source in task.config.target_zone_sites
    assert object_vector.shape == (13,)
    assert np.all(np.isfinite(object_vector))
    assert np.all(np.isfinite(task_vector))
    assert task.env.model.geom_rgba[cube_geom_id, 3] == pytest.approx(1.0)
    assert task.env.model.geom_contype[cube_geom_id] == 1
    assert task.env.model.geom_rgba[wall_geom_id, 3] == pytest.approx(0.0)
    assert task.env.model.geom_contype[wall_geom_id] == 0
    assert all(
        task.env.model.geom_rgba[geom_id, 3] == pytest.approx(0.0)
        for geom_id in button_geom_ids
    )
    for object_name, object_type in (
        ("push_cube_joint", mujoco.mjtObj.mjOBJ_JOINT),
        ("push_cube_geom", mujoco.mjtObj.mjOBJ_GEOM),
        ("push_cube_target_geom", mujoco.mjtObj.mjOBJ_GEOM),
    ):
        assert mujoco.mj_name2id(task.env.model, object_type, object_name) >= 0


def test_push_cube_spec_declares_parameter_and_terminal_state_schemas() -> None:
    pytest.importorskip("mujoco")

    with PushCubeTask(MODEL_PATH) as task:
        spec = task.spec

    assert spec.task_id == PUSH_CUBE_TASK_ID
    assert spec.skill_name == PUSH_CUBE_TASK_ID
    assert spec.required_objects == ("push_cube", "push_cube_target_marker")
    assert spec.parameter_type is PushCubeParameters
    assert spec.parameter_schema["object_id"]["named_id_source"] == ("push_cube",)
    assert spec.parameter_schema["target_pose"]["shape"] == (3,)
    assert spec.parameter_schema["target_pose"]["coordinate_frame"] == (
        "MuJoCo world cube centre"
    )
    assert spec.parameter_schema["target_zone_id"]["named_id_source"] == (
        "push_target_left",
        "push_target_center",
        "push_target_right",
    )
    assert spec.parameter_schema["approach_side"]["values"] == (
        "left",
        "right",
        "front",
        "back",
    )
    assert spec.success_metric_inputs == (
        "object_position",
        "target_position",
        "distance_to_target",
        "dwell_steps",
    )
    assert spec.terminal_state_schema["success"]["terminal"] is True
    assert spec.terminal_state_schema["object_position"]["shape"] == (3,)
    assert spec.terminal_state_schema["object_linear_velocity"]["shape"] == (3,)
    assert "failure_reason" in spec.terminal_state_schema
    spec.action_schema.validate()
    spec.observation_schema.validate()


def test_push_cube_reset_is_deterministic_and_saves_object_target_initial_state() -> None:
    pytest.importorskip("mujoco")

    parameters = PushCubeParameters(approach_side="left")
    with PushCubeTask(MODEL_PATH) as task:
        first = task.reset(seed=29, parameters=parameters)
        second = task.reset(seed=29, parameters=parameters)
        task_vector = second.as_task_state()
        object_vector = second.as_object_state()
        robot_vector = task.robot_state_vector()
        robot_width = max(
            layout.maximum_column
            for layout in task.spec.observation_schema.layouts.values()
            if layout.source_array == "robot_states"
        )

    assert first.object_position == pytest.approx(second.object_position)
    assert first.target_source == second.target_source
    assert first.target_index == second.target_index
    assert first.target_position == pytest.approx(second.target_position)
    assert first.initial_robot_qpos == pytest.approx(second.initial_robot_qpos)
    assert first.initial_robot_qvel == pytest.approx(second.initial_robot_qvel)
    assert second.approach_side == "left"
    assert second.approach_side_index == 0
    assert second.initial_object_position == pytest.approx(second.object_position)
    assert second.initial_object_orientation == pytest.approx(
        second.object_orientation
    )
    assert second.initial_object_linear_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert second.initial_object_angular_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert second.initial_base_position == pytest.approx(
        [
            task.config.initial_base_x,
            second.initial_object_position[1],
            task.config.initial_base_z,
        ]
    )
    assert second.initial_base_orientation == pytest.approx(
        task.config.initial_base_orientation
    )
    assert object_vector == pytest.approx(
        np.concatenate(
            (
                second.object_position,
                second.object_orientation,
                second.object_linear_velocity,
                second.object_angular_velocity,
            )
        )
    )
    assert task_vector.shape == task.spec.observation_schema.shapes["task_state"]
    assert task_vector.shape == (47 + task.env.model.nq + task.env.model.nv,)
    assert robot_vector.shape == (robot_width,)


def test_push_cube_reset_accepts_named_target_and_explicit_world_target() -> None:
    pytest.importorskip("mujoco")

    with PushCubeTask(MODEL_PATH) as task:
        named = task.reset(
            seed=1,
            parameters=PushCubeParameters(
                object_id="push_cube",
                target_zone_id="push_target_right",
                approach_side="front",
            ),
        )
        explicit = task.reset(
            seed=1,
            parameters=PushCubeParameters(target_pose=(0.04, -0.02, -0.015)),
        )

    assert named.target_source == "push_target_right"
    assert named.target_index == 2
    assert named.initial_object_position[1] == pytest.approx(
        named.target_position[1]
    )
    assert named.approach_side == "front"
    assert explicit.target_source == "target_pose"
    assert explicit.target_index == -1
    assert explicit.target_position == pytest.approx([0.04, -0.02, -0.015])


def test_push_cube_parameters_reject_ambiguous_or_invalid_goals() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        PushCubeParameters(
            target_pose=(0.0, 0.0, 0.0),
            target_zone_id="push_target_center",
        )
    with pytest.raises(ValueError, match="three finite"):
        PushCubeParameters(target_pose=(0.0, float("nan"), 0.0))
    with pytest.raises(ValueError, match="non-empty"):
        PushCubeParameters(approach_side="")


def test_synthetic_push_cube_metrics_cover_success_workspace_and_timeout() -> None:
    config = PushCubeConfig()

    assert push_cube_distance(
        [0.0, 0.0, -0.015],
        [0.03, 0.04, -0.015],
    ) == pytest.approx(0.05)
    assert is_push_cube_success(
        distance_m=0.02,
        dwell_steps=config.success_dwell_steps,
        distance_threshold_m=config.target_radius_m,
        required_dwell_steps=config.success_dwell_steps,
    )
    assert not is_push_cube_success(
        distance_m=0.04,
        dwell_steps=config.success_dwell_steps,
        distance_threshold_m=config.target_radius_m,
        required_dwell_steps=config.success_dwell_steps,
    )
    assert (
        push_cube_failure_reason(
            object_position=(0.0, 0.0, -0.015),
            step_count=1,
            max_episode_steps=config.max_episode_steps,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
        )
        is None
    )
    assert (
        push_cube_failure_reason(
            object_position=(0.3, 0.0, -0.015),
            step_count=1,
            max_episode_steps=config.max_episode_steps,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
        )
        == "object_workspace_bounds"
    )
    assert (
        push_cube_failure_reason(
            object_position=(0.0, 0.0, -0.015),
            step_count=config.max_episode_steps,
            max_episode_steps=config.max_episode_steps,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
        )
        == "timeout"
    )


def test_push_cube_state_extracts_pose_velocity_and_reaches_dwell_success() -> None:
    pytest.importorskip("mujoco")

    config = PushCubeConfig(success_dwell_steps=3)
    target = (0.04, 0.0, -0.015)
    with PushCubeTask(MODEL_PATH, config=config) as task:
        task.reset(parameters=PushCubeParameters(target_pose=target))
        terminal = task.get_state()
        for _ in range(config.success_dwell_steps):
            task._set_object_state(
                position=target,
                orientation=(1.0, 0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
            )
            terminal = task.step(n_steps=1)

    assert terminal.object_position == pytest.approx(target, abs=2e-3)
    assert np.all(np.isfinite(terminal.object_orientation))
    assert np.all(np.isfinite(terminal.object_linear_velocity))
    assert np.all(np.isfinite(terminal.object_angular_velocity))
    assert terminal.distance_to_target <= config.target_radius_m
    assert terminal.dwell_steps == config.success_dwell_steps
    assert terminal.success is True
    assert terminal.failure_reason is None


def test_push_cube_terminal_state_reports_timeout_without_success() -> None:
    pytest.importorskip("mujoco")

    config = PushCubeConfig(max_episode_steps=1)
    with PushCubeTask(MODEL_PATH, config=config) as task:
        task.reset(
            seed=0,
            parameters=PushCubeParameters(
                target_zone_id="push_target_right",
            ),
        )
        terminal = task.step(n_steps=1)

    assert terminal.success is False
    assert terminal.failure_reason == "timeout"
    assert terminal.step_count == 1
