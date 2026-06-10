from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dexvision.apps import check_one_joint


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "mujoco" / "hand_scene.xml"


def test_check_one_joint_help_runs_without_loading_model() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_one_joint", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "--joint" in result.stdout
    assert "--headless" in result.stdout
    assert "--frequency-hz" in result.stdout


def test_resolves_joint_to_single_position_actuator() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    with MujocoEnv(MODEL_PATH) as env:
        binding = check_one_joint.resolve_one_joint_binding(env, "index_mcp")

    assert binding.joint_name == "index_mcp"
    assert binding.actuator_name == "index_mcp_position"
    assert binding.target_minimum == pytest.approx(-0.15)
    assert binding.target_maximum == pytest.approx(1.45)


def test_resolves_actuator_name_to_attached_joint() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    with MujocoEnv(MODEL_PATH) as env:
        binding = check_one_joint.resolve_one_joint_binding(env, "index_mcp_position")

    assert binding.joint_name == "index_mcp"
    assert binding.actuator_name == "index_mcp_position"


def test_periodic_target_stays_inside_joint_and_actuator_limits() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    with MujocoEnv(MODEL_PATH) as env:
        binding = check_one_joint.resolve_one_joint_binding(env, "index_mcp")

    targets = [
        check_one_joint.compute_periodic_target(binding, time_seconds=sample, frequency_hz=1.0)
        for sample in (0.0, 0.125, 0.25, 0.5, 0.75)
    ]

    assert min(targets) >= binding.target_minimum
    assert max(targets) <= binding.target_maximum
    assert targets[0] == pytest.approx(0.65)
    assert max(targets) > targets[0]
    assert min(targets) < targets[0]


def test_check_one_joint_headless_command_runs() -> None:
    pytest.importorskip("mujoco")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexvision.apps.check_one_joint",
            "--model",
            str(MODEL_PATH),
            "--joint",
            "index_mcp",
            "--headless",
            "--steps",
            "30",
            "--print-interval",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "DexVision one-joint MuJoCo check" in result.stdout
    assert "No camera, MediaPipe, or hand tracking is used." in result.stdout
    assert "Joint: index_mcp" in result.stdout
    assert "Actuator: index_mcp_position" in result.stdout
    assert "target=" in result.stdout
    assert "current=" in result.stdout


def test_check_one_joint_reports_missing_joint(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("mujoco")

    result = check_one_joint.main(
        [
            "--model",
            str(MODEL_PATH),
            "--joint",
            "missing_joint",
            "--headless",
            "--steps",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: Unknown hand joint or actuator 'missing_joint'" in captured.err
    assert "index_mcp" in captured.err


def test_macos_viewer_preflight_reports_mjpython_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    from mujoco import viewer

    monkeypatch.setattr(check_one_joint.sys, "platform", "darwin")
    monkeypatch.setattr(viewer, "_MJPYTHON", None)

    with pytest.raises(check_one_joint.MujocoError) as exc_info:
        check_one_joint._ensure_viewer_can_launch(
            model_path=Path("assets/mujoco/hand_scene.xml"),
            selected_joint="index_mcp",
            steps=600,
            frequency_hz=check_one_joint.DEFAULT_FREQUENCY_HZ,
            viewer_sleep=check_one_joint.DEFAULT_VIEWER_SLEEP,
            print_interval=check_one_joint.DEFAULT_PRINT_INTERVAL,
        )

    message = str(exc_info.value)
    assert "requires the mjpython launcher" in message
    assert "regular macOS Terminal or iTerm" in message
    assert "mjpython -m dexvision.apps.check_one_joint" in message
    assert "--joint index_mcp" in message
    assert "--steps 600" in message


def test_mjpython_command_quotes_paths_with_spaces() -> None:
    command = check_one_joint._format_mjpython_command(
        model_path=Path("assets/mujoco/hand scene.xml"),
        selected_joint="index_mcp",
        steps=10,
        frequency_hz=0.5,
        viewer_sleep=0.0,
        print_interval=5,
    )

    assert command.startswith("mjpython -m dexvision.apps.check_one_joint")
    assert "'assets/mujoco/hand scene.xml'" in command
    assert "--frequency-hz 0.5" in command
    assert "--viewer-sleep 0.0" in command
    assert "--print-interval 5" in command
