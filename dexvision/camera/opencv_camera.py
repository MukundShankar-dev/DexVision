"""OpenCV camera input for DexVision."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

try:  # OpenCV is imported lazily enough to keep tests mockable.
    import cv2
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent.
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR: ImportError | None = exc
else:
    _CV2_IMPORT_ERROR = None


class CameraOpenError(RuntimeError):
    """Raised when a camera cannot be opened or read."""


@dataclass(frozen=True)
class CameraFrame:
    """One camera read result.

    Attributes:
        success: Whether OpenCV returned a valid frame.
        frame: BGR image with shape [H, W, 3] and dtype uint8 when successful.
        timestamp: Monotonic timestamp in seconds.
    """

    success: bool
    frame: np.ndarray | None
    timestamp: float


class VideoCaptureLike(Protocol):
    """Minimal OpenCV VideoCapture protocol used by this module."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...

    def set(self, prop_id: int, value: float) -> bool: ...


CaptureFactory = Callable[[int], VideoCaptureLike]


class OpenCVCamera:
    """Small wrapper around ``cv2.VideoCapture`` with clean open/read/release steps."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        if width <= 0:
            raise ValueError("width must be a positive integer.")
        if height <= 0:
            raise ValueError("height must be a positive integer.")

        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._capture_factory = capture_factory
        self._capture: VideoCaptureLike | None = None

    @property
    def is_open(self) -> bool:
        """Return whether a capture object is currently open."""

        return self._capture is not None and self._capture.isOpened()

    def open(self) -> "OpenCVCamera":
        """Open the configured camera and apply the requested frame size."""

        if self.is_open:
            return self

        capture = self._make_capture()
        self._set_requested_size(capture)

        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(
                "Could not open camera "
                f"{self.camera_id}. Confirm the device is connected and try "
                "--camera-id 1 or --camera-id 2 if needed."
            )

        self._capture = capture
        return self

    def read(self) -> CameraFrame:
        """Read one frame from the open camera."""

        if self._capture is None:
            raise CameraOpenError("Camera is not open. Call open() before read().")

        success, frame = self._capture.read()
        timestamp = time.monotonic()
        if not success or not isinstance(frame, np.ndarray):
            return CameraFrame(success=False, frame=None, timestamp=timestamp)

        return CameraFrame(success=True, frame=frame, timestamp=timestamp)

    def release(self) -> None:
        """Release the underlying OpenCV capture if it exists."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "OpenCVCamera":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _make_capture(self) -> VideoCaptureLike:
        if self._capture_factory is not None:
            return self._capture_factory(self.camera_id)

        if cv2 is None:
            raise CameraOpenError(
                "OpenCV is required for camera input. Install the package providing "
                f"'cv2' ({_CV2_IMPORT_ERROR})."
            )

        return cv2.VideoCapture(self.camera_id)

    def _set_requested_size(self, capture: VideoCaptureLike) -> None:
        width_prop = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3) if cv2 is not None else 3
        height_prop = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4) if cv2 is not None else 4
        capture.set(width_prop, float(self.width))
        capture.set(height_prop, float(self.height))
