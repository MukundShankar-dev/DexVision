from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from dexvision.apps import check_camera
from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera


class FakeCapture:
    def __init__(self, *, opened: bool = True, frame: np.ndarray | None = None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False
        self.set_calls: list[tuple[int, float]] = []
        self.read_count = 0

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, Any]:
        self.read_count += 1
        if self.frame is None:
            return False, None
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True


def test_opencv_camera_reads_synthetic_frame_and_releases() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    fake_capture = FakeCapture(frame=frame)
    camera = OpenCVCamera(
        camera_id=2,
        width=320,
        height=240,
        capture_factory=lambda camera_id: fake_capture,
    )

    camera.open()
    result = camera.read()
    camera.release()

    assert result.success
    assert result.frame is not None
    assert result.frame.shape == (240, 320, 3)
    assert result.frame.dtype == np.uint8
    assert isinstance(result.timestamp, float)
    assert fake_capture.set_calls == [(3, 320.0), (4, 240.0)]
    assert fake_capture.read_count == 1
    assert fake_capture.released


def test_opencv_camera_context_manager_releases_capture() -> None:
    frame = np.full((16, 16, 3), 127, dtype=np.uint8)
    fake_capture = FakeCapture(frame=frame)

    with OpenCVCamera(capture_factory=lambda camera_id: fake_capture) as camera:
        assert camera.is_open

    assert fake_capture.released


def test_opencv_camera_reports_failed_read_without_crashing() -> None:
    fake_capture = FakeCapture(frame=None)
    camera = OpenCVCamera(capture_factory=lambda camera_id: fake_capture)

    camera.open()
    result = camera.read()
    camera.release()

    assert not result.success
    assert result.frame is None
    assert isinstance(result.timestamp, float)


def test_opencv_camera_raises_clear_error_when_device_missing() -> None:
    fake_capture = FakeCapture(opened=False)
    camera = OpenCVCamera(
        camera_id=9,
        capture_factory=lambda camera_id: fake_capture,
    )

    with pytest.raises(CameraOpenError, match="Could not open camera 9"):
        camera.open()

    assert fake_capture.released


def test_opencv_camera_requires_open_before_read() -> None:
    camera = OpenCVCamera(capture_factory=lambda camera_id: FakeCapture())

    with pytest.raises(CameraOpenError, match="Call open"):
        camera.read()


def test_opencv_camera_validates_requested_size() -> None:
    with pytest.raises(ValueError, match="width"):
        OpenCVCamera(width=0)

    with pytest.raises(ValueError, match="height"):
        OpenCVCamera(height=0)


def test_check_camera_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_camera", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--camera-id" in result.stdout
    assert "--width" in result.stdout
    assert "--height" in result.stdout


def test_check_camera_main_reports_camera_open_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_open(camera_id: int, width: int, height: int) -> int:
        raise CameraOpenError(f"Could not open camera {camera_id}.")

    monkeypatch.setattr(check_camera, "run_camera", fail_to_open)

    result = check_camera.main(["--camera-id", "9", "--width", "320", "--height", "240"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: Could not open camera 9." in captured.err
