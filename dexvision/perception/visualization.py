"""OpenCV drawing helpers for hand-tracking overlays."""

from __future__ import annotations

from typing import Final

import numpy as np

from dexvision.perception.hand_tracker import HandTrackingResult


HAND_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def draw_hand_tracking(
    frame: np.ndarray,
    result: HandTrackingResult,
    *,
    draw_no_hand_label: bool = True,
) -> np.ndarray:
    """Draw hand landmarks, skeleton, and status text onto a BGR frame."""

    _validate_frame(frame)
    cv2 = _load_cv2()

    if result.detected and result.image_landmarks is not None:
        points = _normalized_landmarks_to_pixels(result.image_landmarks, frame.shape[1], frame.shape[0])
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 180, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 80), -1, cv2.LINE_AA)

        handedness = result.handedness or "Unknown"
        cv2.putText(
            frame,
            f"{handedness} {result.confidence:.2f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )
    elif draw_no_hand_label:
        cv2.putText(
            frame,
            "Hand: not detected",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError(
            "OpenCV is required for hand-tracking visualization. Install the "
            f"package providing 'cv2' ({exc})."
        ) from exc
    return cv2


def _normalized_landmarks_to_pixels(
    image_landmarks: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    if image_landmarks.shape != (21, 3):
        raise ValueError(f"image_landmarks must have shape [21, 3], got {image_landmarks.shape}.")

    points: list[tuple[int, int]] = []
    for x, y, _z in image_landmarks:
        safe_x = 0.0 if not np.isfinite(x) else float(np.clip(x, 0.0, 1.0))
        safe_y = 0.0 if not np.isfinite(y) else float(np.clip(y, 0.0, 1.0))
        pixel_x = int(round(safe_x * (width - 1)))
        pixel_y = int(round(safe_y * (height - 1)))
        points.append((pixel_x, pixel_y))
    return points


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape [H, W, 3].")
