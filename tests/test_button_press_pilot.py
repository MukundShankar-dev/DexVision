from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import record_demo
from dexvision.logging.dataset_summary import summarize_demo_dataset
from dexvision.logging.demo_logger import DemoLogger, DemoStepData
from dexvision.logging.quality_filters import (
    QualityThresholds,
    filter_demo_dataset,
    save_quality_report,
)
from dexvision.logging.relabel_success import (
    BUTTON_PRESS_RELABEL_REPORT_VERSION,
    relabel_demo_dataset,
    save_relabel_report,
)
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.sim.mujoco_env import MujocoEnv
from dexvision.sim.tasks import ButtonPressParameters, ButtonPressTask


ROOT = Path(__file__).resolve().parents[1]
TASK_BOARD_MODEL = ROOT / "assets" / "mujoco" / "task_board_scene.xml"


def test_button_pilot_parser_accepts_goal_parameters_and_uses_task_board() -> None:
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "button_press",
            "--button-id",
            "button_center",
            "--target-press-depth",
            "0.01",
            "--approach-pose",
            "0.1",
            "0.0",
            "0.4",
            "--level1-13-full",
        ]
    )

    record_demo._apply_recording_presets(args)
    record_demo._validate_recording_args(args)
    model_path = record_demo._resolve_recording_model_path(
        args=args,
        raw_config={"model_path": "assets/mujoco/hand_scene.xml"},
    )

    assert args.button_id == "button_center"
    assert args.target_press_depth == pytest.approx(0.01)
    assert args.approach_pose == pytest.approx([0.1, 0.0, 0.4])
    assert args.show_camera_window is True
    assert args.viewer is True
    assert args.start_on_calibration is True
    assert model_path == record_demo.DEFAULT_TASK_BOARD_MODEL


@pytest.mark.parametrize(("response", "expected"), [("yes", True), ("no", False)])
def test_button_pilot_requires_operator_label(
    response: str,
    expected: bool,
) -> None:
    args = record_demo.build_parser().parse_args(["--task", "button_press"])

    assert (
        record_demo._resolve_operator_success_label(
            args,
            input_fn=lambda _prompt: response,
        )
        is expected
    )


def test_button_task_hides_reach_fixtures_and_colors_selected_target() -> None:
    mujoco = pytest.importorskip("mujoco")

    with ButtonPressTask(TASK_BOARD_MODEL) as task:
        task.reset(
            parameters=ButtonPressParameters(button_id="button_center")
        )
        site_alphas = []
        for site_name in (
            "reach_target_left",
            "reach_target_center",
            "reach_target_right",
        ):
            site_id = mujoco.mj_name2id(
                task.env.model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_name,
            )
            site_alphas.append(float(task.env.model.site_rgba[site_id, 3]))
        geom_id = mujoco.mj_name2id(
            task.env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "active_reach_target",
        )
        geom_alpha = float(task.env.model.geom_rgba[geom_id, 3])
        geom_contype = int(task.env.model.geom_contype[geom_id])
        geom_conaffinity = int(task.env.model.geom_conaffinity[geom_id])
        button_colors = {}
        for button_id in task.config.button_ids:
            material_id = mujoco.mj_name2id(
                task.env.model,
                mujoco.mjtObj.mjOBJ_MATERIAL,
                f"{button_id}_material",
            )
            button_colors[button_id] = tuple(
                float(value) for value in task.env.model.mat_rgba[material_id]
            )

    untouched_model = mujoco.MjModel.from_xml_path(str(TASK_BOARD_MODEL))
    untouched_geom_id = mujoco.mj_name2id(
        untouched_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "active_reach_target",
    )

    assert site_alphas == [0.0, 0.0, 0.0]
    assert geom_alpha == 0.0
    assert geom_contype == 0
    assert geom_conaffinity == 0
    assert button_colors["button_center"] == pytest.approx((0.1, 1.0, 0.2, 1.0))
    assert button_colors["button_left"] == pytest.approx((0.22, 0.22, 0.22, 1.0))
    assert button_colors["button_right"] == pytest.approx((0.22, 0.22, 0.22, 1.0))
    assert untouched_model.geom_rgba[untouched_geom_id, 3] > 0.0
    assert untouched_model.geom_contype[untouched_geom_id] != 0
    assert untouched_model.geom_conaffinity[untouched_geom_id] != 0


def test_button_episode_passes_relabel_quality_summary_and_replay(
    tmp_path: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    episode_dir = tmp_path / "raw" / "button_press" / "pilot_001"
    args = record_demo.build_parser().parse_args(
        [
            "--task",
            "button_press",
            "--button-id",
            "button_left",
            "--target-press-depth",
            "0.01",
            "--success",
        ]
    )
    record_demo._validate_recording_args(args)

    with ButtonPressTask(TASK_BOARD_MODEL) as task:
        initial_state = task.reset(
            parameters=ButtonPressParameters(
                button_id=args.button_id,
                target_press_depth=args.target_press_depth,
            )
        )
        target_names = tuple(
            task.spec.action_schema.representation_notes["finger_target_names"]
        )
        metadata = record_demo._metadata(
            args=args,
            episode_id="button_press_pilot_001",
            raw_config=record_demo.load_curl_retargeter_config(
                record_demo.DEFAULT_CONFIG
            ),
            model_path=TASK_BOARD_MODEL,
            target_names=target_names,
            observation_schema=task.spec.observation_schema,
            synthetic=False,
            button_press_task=task,
            button_press_initial_state=initial_state,
        )
        logger = DemoLogger(
            episode_dir,
            action_schema=task.spec.action_schema,
            observation_schema=task.spec.observation_schema,
        )
        logger.start_episode(metadata)

        joint_id = mujoco.mj_name2id(
            task.env.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "button_left_joint",
        )
        qpos_address = int(task.env.model.jnt_qposadr[joint_id])
        for frame_index in range(task.config.success_dwell_steps):
            task.env.data.qpos[qpos_address] = 0.02
            mujoco.mj_forward(task.env.model, task.env.data)
            state = task.step(n_steps=1)
            base_position, base_orientation = task.env.get_mocap_pose(
                task.config.base_target_body
            )
            robot_state = task.env.get_state()
            targets = dict(zip(target_names, robot_state.ctrl, strict=True))
            logger.append(
                DemoStepData(
                    features=np.zeros(
                        len(record_demo.FEATURE_FIELDS),
                        dtype=np.float64,
                    ),
                    action=record_demo.action_vector(
                        base_position=base_position,
                        base_orientation=base_orientation,
                        targets=targets,
                        target_names=target_names,
                    ),
                    robot_state=task.robot_state_vector(),
                    tracking_quality=np.asarray(
                        [1.0, 1.0, 0.95, 0.95, 0.0, 0.0],
                        dtype=np.float64,
                    ),
                    timestamp=frame_index / 30.0,
                    object_state=state.as_object_state(),
                    task_state=state.as_task_state(),
                )
            )
        logger.close(success=True)

    relabel_report = relabel_demo_dataset(episode_dir.parent)
    save_relabel_report(
        relabel_report,
        episode_dir.parent / "relabel_report.json",
    )
    thresholds = QualityThresholds(
        max_feature_jitter_p95=1.0,
        max_action_jerk_p95=1.0,
        max_joint_limit_hit_fraction=1.0,
        max_workspace_limit_hit_fraction=1.0,
    )
    quality_report = filter_demo_dataset(
        episode_dir.parent,
        thresholds=thresholds,
    )
    save_quality_report(
        quality_report,
        episode_dir.parent / "quality_report.json",
    )
    summary = summarize_demo_dataset(tmp_path)
    loaded = load_replay_demo(episode_dir)
    with MujocoEnv(TASK_BOARD_MODEL) as replay_env:
        replay_result = replay_loaded_demo(
            loaded,
            replay_env,
            sleep_fn=lambda _delay: None,
        )
        active_target_id = mujoco.mj_name2id(
            replay_env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "active_reach_target",
        )
        replay_active_target_alpha = float(
            replay_env.model.geom_rgba[active_target_id, 3]
        )
        replay_button_colors = {}
        for button_id in ("button_left", "button_center", "button_right"):
            material_id = mujoco.mj_name2id(
                replay_env.model,
                mujoco.mjtObj.mjOBJ_MATERIAL,
                f"{button_id}_material",
            )
            replay_button_colors[button_id] = tuple(
                float(value) for value in replay_env.model.mat_rgba[material_id]
            )

    saved_metadata = json.loads(
        (episode_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["task_config"]["resolved_button_id"] == "button_left"
    assert saved_metadata["task_config"]["target_press_depth"] == pytest.approx(0.01)
    assert saved_metadata["task_config"]["target_visual_cue"] == "bright_green"
    assert saved_metadata["task_config"]["non_target_visual_cue"] == "dark_gray"
    assert relabel_report.version == BUTTON_PRESS_RELABEL_REPORT_VERSION
    assert relabel_report.recomputed_success_count == 1
    assert relabel_report.label_disagreement_count == 0
    assert quality_report.pass_count == 1
    assert replay_result.steps_replayed == task.config.success_dwell_steps
    assert replay_active_target_alpha == 0.0
    assert replay_button_colors["button_left"] == pytest.approx(
        (0.1, 1.0, 0.2, 1.0)
    )
    assert replay_button_colors["button_center"] == pytest.approx(
        (0.22, 0.22, 0.22, 1.0)
    )
    assert replay_button_colors["button_right"] == pytest.approx(
        (0.22, 0.22, 0.22, 1.0)
    )
    button_group = next(
        group for group in summary.groups if group.task_id == "button_press"
    )
    assert button_group.num_episodes == 1
    assert button_group.num_success == 1
    assert button_group.quality_pass_count == 1
    assert button_group.relabel_unreported_count == 0
