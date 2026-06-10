from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dexvision.apps import check_hand_actuation


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "mujoco" / "debug_hand_scene.xml"
GESTURE_CONFIG_PATH = ROOT / "configs" / "hand_gestures.yaml"

EXPECTED_ACTUATORS = {
    "thumb_abduction_position",
    "thumb_flexion_position",
    "index_mcp_position",
    "index_pip_position",
    "middle_mcp_position",
    "middle_pip_position",
    "ring_mcp_position",
    "ring_pip_position",
    "pinky_mcp_position",
    "pinky_pip_position",
}


def test_check_hand_actuation_help_runs_without_loading_model() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_hand_actuation", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "--gestures" in result.stdout
    assert "--headless" in result.stdout
    assert "--sequence" in result.stdout


def test_default_gesture_config_is_readable() -> None:
    library = check_hand_actuation.load_gesture_config(GESTURE_CONFIG_PATH)

    assert set(library.gestures) >= check_hand_actuation.REQUIRED_GESTURES
    assert set(library.sequence) <= set(library.gestures)
    assert library.steps_per_gesture > 0
    assert library.sequence[0] == "open_hand"
    for targets in library.gestures.values():
        assert set(targets) == EXPECTED_ACTUATORS


def test_default_gestures_respect_hand_model_limits() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    library = check_hand_actuation.load_gesture_config(GESTURE_CONFIG_PATH)
    with MujocoEnv(MODEL_PATH) as env:
        bindings = check_hand_actuation.resolve_actuator_bindings(env)
        check_hand_actuation.validate_gesture_library(library, bindings)

    assert set(bindings) == EXPECTED_ACTUATORS
    assert bindings["index_mcp_position"].target_minimum == pytest.approx(-0.15)
    assert bindings["index_mcp_position"].target_maximum == pytest.approx(1.45)


def test_default_gestures_use_visible_thumb_and_strong_fist() -> None:
    library = check_hand_actuation.load_gesture_config(GESTURE_CONFIG_PATH)
    thumb_abduction_targets = [
        targets["thumb_abduction_position"] for targets in library.gestures.values()
    ]

    assert max(thumb_abduction_targets) - min(thumb_abduction_targets) >= 1.0
    assert library.gestures["fist"]["thumb_flexion_position"] >= 1.1
    assert library.gestures["fist"]["index_mcp_position"] >= 1.4
    assert library.gestures["fist"]["index_pip_position"] >= 1.5
    assert library.gestures["fist"]["pinky_mcp_position"] >= 1.3
    assert library.gestures["fist"]["pinky_pip_position"] >= 1.4


def test_peace_sign_keeps_index_middle_open_and_curls_outer_fingers() -> None:
    library = check_hand_actuation.load_gesture_config(GESTURE_CONFIG_PATH)
    peace_sign = library.gestures["peace_sign"]

    assert peace_sign["index_mcp_position"] <= -0.1
    assert peace_sign["index_pip_position"] == pytest.approx(0.0)
    assert peace_sign["middle_mcp_position"] <= -0.1
    assert peace_sign["middle_pip_position"] == pytest.approx(0.0)
    assert peace_sign["ring_mcp_position"] >= 1.4
    assert peace_sign["ring_pip_position"] >= 1.55
    assert peace_sign["pinky_mcp_position"] >= 1.3
    assert peace_sign["pinky_pip_position"] >= 1.45


def test_open_thumb_pose_stays_outside_palm_box() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    library = check_hand_actuation.load_gesture_config(GESTURE_CONFIG_PATH)
    with MujocoEnv(MODEL_PATH) as env:
        env.reset()
        env.set_joint_targets(library.gestures["open_hand"])
        env.step(n_steps=120)

        mujoco_module = env._mujoco
        palm_geom_id = mujoco_module.mj_name2id(
            env.model,
            mujoco_module.mjtObj.mjOBJ_GEOM,
            "palm_geom",
        )
        palm_body_id = mujoco_module.mj_name2id(
            env.model,
            mujoco_module.mjtObj.mjOBJ_BODY,
            "palm",
        )
        thumb_body_id = mujoco_module.mj_name2id(
            env.model,
            mujoco_module.mjtObj.mjOBJ_BODY,
            "thumb_distal",
        )

        palm_left_edge_x = float(env.data.xpos[palm_body_id, 0] - env.model.geom_size[palm_geom_id, 0])
        thumb_distal_x = float(env.data.xpos[thumb_body_id, 0])

    assert thumb_distal_x < palm_left_edge_x


def test_limit_validation_rejects_unknown_actuator() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    library = check_hand_actuation.load_gesture_config(GESTURE_CONFIG_PATH)
    gestures = {name: dict(targets) for name, targets in library.gestures.items()}
    gestures["open_hand"]["missing_actuator_position"] = 0.0
    bad_library = check_hand_actuation.GestureLibrary(
        path=library.path,
        gestures=gestures,
        sequence=library.sequence,
        steps_per_gesture=library.steps_per_gesture,
    )

    with MujocoEnv(MODEL_PATH) as env:
        bindings = check_hand_actuation.resolve_actuator_bindings(env)
        with pytest.raises(check_hand_actuation.MujocoError, match="unknown actuators"):
            check_hand_actuation.validate_gesture_library(bad_library, bindings)


def test_check_hand_actuation_headless_command_runs() -> None:
    pytest.importorskip("mujoco")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexvision.apps.check_hand_actuation",
            "--model",
            str(MODEL_PATH),
            "--gestures",
            str(GESTURE_CONFIG_PATH),
            "--headless",
            "--steps-per-gesture",
            "3",
            "--print-interval",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "DexVision hand actuation check" in result.stdout
    assert "No camera, MediaPipe, hand tracking, or retargeting is used." in result.stdout
    assert "Validated 6 gestures against 10 actuator limits." in result.stdout
    assert "gesture=fist" in result.stdout
    assert "gesture=pinch" in result.stdout
    assert "Simulation stepped to" in result.stdout


def test_check_hand_actuation_reports_missing_config(capsys: pytest.CaptureFixture[str]) -> None:
    result = check_hand_actuation.main(
        [
            "--gestures",
            "configs/missing_gestures.yaml",
            "--headless",
            "--steps-per-gesture",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: Gesture config file does not exist" in captured.err


def test_mjpython_command_quotes_paths_with_spaces() -> None:
    command = check_hand_actuation._format_mjpython_command(
        model_path=Path("assets/mujoco/hand scene.xml"),
        gesture_config_path=Path("configs/hand gestures.yaml"),
        sequence_override=("open_hand", "fist"),
        steps_per_gesture_override=10,
        viewer_sleep=0.0,
        print_interval=5,
    )

    assert command.startswith("mjpython -m dexvision.apps.check_hand_actuation")
    assert "'assets/mujoco/hand scene.xml'" in command
    assert "'configs/hand gestures.yaml'" in command
    assert "--sequence open_hand fist" in command
    assert "--steps-per-gesture 10" in command
    assert "--viewer-sleep 0.0" in command
    assert "--print-interval 5" in command
