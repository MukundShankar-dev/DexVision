from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import check_mujoco


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "mujoco" / "simple_scene.xml"


def test_simple_scene_file_exists() -> None:
    assert MODEL_PATH.is_file()


def test_mujoco_env_loads_resets_and_steps() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    with MujocoEnv(MODEL_PATH) as env:
        initial = env.reset()
        stepped = env.step(n_steps=5)

    assert env.model.nq > 0
    assert env.model.nv > 0
    assert env.model.nu == 0
    assert initial.time == pytest.approx(0.0)
    assert stepped.time > initial.time
    assert stepped.qpos.shape == (env.model.nq,)
    assert stepped.qvel.shape == (env.model.nv,)
    assert stepped.ctrl.shape == (env.model.nu,)
    assert np.all(np.isfinite(stepped.qpos))
    assert np.all(np.isfinite(stepped.qvel))


def test_check_mujoco_help_runs_without_loading_model() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_mujoco", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--steps" in result.stdout
    assert "--viewer" in result.stdout


def test_check_mujoco_headless_command_runs() -> None:
    pytest.importorskip("mujoco")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexvision.apps.check_mujoco",
            "--model",
            str(MODEL_PATH),
            "--steps",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "DexVision MuJoCo smoke test" in result.stdout
    assert "No camera or MediaPipe is used." in result.stdout
    assert "Simulation stepped" in result.stdout


def test_macos_viewer_preflight_reports_mjpython_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    from mujoco import viewer

    monkeypatch.setattr(check_mujoco.sys, "platform", "darwin")
    monkeypatch.setattr(viewer, "_MJPYTHON", None)

    with pytest.raises(check_mujoco.MujocoError) as exc_info:
        check_mujoco._ensure_viewer_can_launch(
            MODEL_PATH,
            steps=600,
            viewer_sleep=check_mujoco.DEFAULT_VIEWER_SLEEP,
        )

    message = str(exc_info.value)
    assert "requires the mjpython launcher" in message
    assert "regular macOS Terminal or iTerm" in message
    assert "mjpython -m dexvision.apps.check_mujoco" in message
    assert "--viewer" in message
    assert "--steps 600" in message


def test_mjpython_command_quotes_paths_with_spaces() -> None:
    command = check_mujoco._format_mjpython_command(
        Path("assets/mujoco/simple scene.xml"),
        steps=10,
        viewer_sleep=0.0,
    )

    assert command.startswith("mjpython -m dexvision.apps.check_mujoco")
    assert "'assets/mujoco/simple scene.xml'" in command
    assert "--viewer-sleep 0.0" in command


def test_check_mujoco_reports_missing_model(capsys: pytest.CaptureFixture[str]) -> None:
    result = check_mujoco.main(["--model", "assets/mujoco/missing_scene.xml", "--steps", "1"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MuJoCo model file does not exist" in captured.err
