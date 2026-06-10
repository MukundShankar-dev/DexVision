from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import check_hand_model
from dexvision.sim.hand_model import (
    check_rest_stability,
    inspect_hand_model,
    require_controllable_hand,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "mujoco" / "hand_scene.xml"

EXPECTED_JOINTS = {
    "thumb_abduction",
    "thumb_flexion",
    "index_mcp",
    "index_pip",
    "middle_mcp",
    "middle_pip",
    "ring_mcp",
    "ring_pip",
    "pinky_mcp",
    "pinky_pip",
}


def test_hand_scene_file_exists() -> None:
    assert MODEL_PATH.is_file()


def test_hand_model_contains_named_limited_joints_and_actuators() -> None:
    pytest.importorskip("mujoco")

    info = inspect_hand_model(MODEL_PATH)
    require_controllable_hand(info)

    assert set(info.joint_names) == EXPECTED_JOINTS
    assert info.actuator_count == len(EXPECTED_JOINTS)
    assert len(set(info.actuator_names)) == info.actuator_count

    actuated_joints = {actuator.joint_name for actuator in info.actuators}
    assert actuated_joints == EXPECTED_JOINTS
    for joint in info.joints:
        assert joint.limit.minimum < joint.limit.maximum
    for actuator in info.actuators:
        assert actuator.control_range.minimum < actuator.control_range.maximum


def test_hand_model_loads_and_steps_at_rest() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    with MujocoEnv(MODEL_PATH) as env:
        initial = env.reset()
        stepped = env.step(n_steps=20)

    assert env.model.njnt == len(EXPECTED_JOINTS)
    assert env.model.nu == len(EXPECTED_JOINTS)
    assert initial.time == pytest.approx(0.0)
    assert stepped.time > initial.time
    assert stepped.qpos.shape == (env.model.nq,)
    assert stepped.qvel.shape == (env.model.nv,)
    assert stepped.ctrl.shape == (env.model.nu,)
    assert np.all(np.isfinite(stepped.qpos))
    assert np.all(np.isfinite(stepped.qvel))


def test_hand_model_rest_stability_result_passes() -> None:
    pytest.importorskip("mujoco")

    result = check_rest_stability(MODEL_PATH, steps=240)

    assert result.stable
    assert result.final_time > result.initial_time
    assert result.max_abs_qpos < 2.5
    assert result.max_abs_qvel < 25.0


def test_check_hand_model_help_runs_without_loading_model() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_hand_model", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--steps" in result.stdout


def test_check_hand_model_headless_command_runs() -> None:
    pytest.importorskip("mujoco")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexvision.apps.check_hand_model",
            "--model",
            str(MODEL_PATH),
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "DexVision hand model check" in result.stdout
    assert "Joints (10)" in result.stdout
    assert "Actuators (10)" in result.stdout
    assert "Rest stability:" in result.stdout
    assert "status=PASS" in result.stdout


def test_check_hand_model_reports_missing_model(capsys: pytest.CaptureFixture[str]) -> None:
    result = check_hand_model.main(["--model", "assets/mujoco/missing_hand.xml", "--steps", "1"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MuJoCo model file does not exist" in captured.err
