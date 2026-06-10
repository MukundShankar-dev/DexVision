"""Open a camera feed, run hand tracking, and draw landmarks."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path

from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.perception.hand_tracker import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    HandTracker,
    HandTrackerError,
)
from dexvision.perception.visualization import draw_hand_tracking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display MediaPipe hand landmarks on a camera feed.")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "MediaPipe Tasks hand-landmarker model path. Defaults to "
            f"{DEFAULT_HAND_LANDMARKER_MODEL} when legacy MediaPipe Hands is unavailable."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe tracking confidence.",
    )
    return parser


def _load_cv2_for_display():
    try:
        import cv2
    except ImportError as exc:
        raise CameraOpenError(
            "OpenCV is required for camera display. Install the package providing "
            f"'cv2' ({exc})."
        ) from exc
    return cv2


def _draw_fps(cv2_module, frame, fps: float) -> None:
    cv2_module.putText(
        frame,
        f"FPS: {fps:5.1f}",
        (16, 64),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )


def run_hand_tracking(
    camera_id: int,
    width: int,
    height: int,
    model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> int:
    cv2_module = _load_cv2_for_display()
    window_name = "DexVision Hand Landmark Tracking"
    last_frame_time = time.monotonic()
    fps = 0.0

    print("DexVision hand landmark tracking")
    print(
        "Opening "
        f"camera_id={camera_id}, width={width}, height={height}, "
        f"model_path={model_path or DEFAULT_HAND_LANDMARKER_MODEL}, "
        f"min_detection_confidence={min_detection_confidence}, "
        f"min_tracking_confidence={min_tracking_confidence}"
    )
    print("Press q to quit.")

    try:
        with (
            OpenCVCamera(camera_id=camera_id, width=width, height=height) as camera,
            HandTracker(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_path=model_path,
            ) as tracker,
        ):
            while True:
                camera_result = camera.read()
                if not camera_result.success or camera_result.frame is None:
                    print("WARNING: Camera read failed; waiting for the next frame.")
                    if cv2_module.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                tracking_result = tracker.process(
                    camera_result.frame,
                    timestamp=camera_result.timestamp,
                )

                now = time.monotonic()
                elapsed = max(now - last_frame_time, 1e-9)
                last_frame_time = now
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)

                draw_hand_tracking(camera_result.frame, tracking_result)
                _draw_fps(cv2_module, camera_result.frame, fps)
                cv2_module.imshow(window_name, camera_result.frame)

                if cv2_module.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        with suppress(Exception):
            cv2_module.destroyWindow(window_name)

    print("Hand tracking closed cleanly.")
    return 0


def _is_cv2_error(exc: Exception) -> bool:
    return exc.__class__.__module__.split(".", maxsplit=1)[0] == "cv2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_hand_tracking(
            args.camera_id,
            args.width,
            args.height,
            args.model_path,
            args.min_detection_confidence,
            args.min_tracking_confidence,
        )
    except (CameraOpenError, HandTrackerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Hand tracking closed cleanly.")
        return 130
    except Exception as exc:
        if _is_cv2_error(exc):
            print(
                "ERROR: OpenCV display failed. Ensure this is running in a desktop "
                f"session with GUI support. Details: {exc}",
                file=sys.stderr,
            )
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
