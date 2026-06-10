"""Open a camera feed with OpenCV and display live FPS."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import suppress

from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open an OpenCV camera smoke-test feed.")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
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
        (16, 32),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2_module.LINE_AA,
    )


def run_camera(camera_id: int, width: int, height: int) -> int:
    cv2_module = _load_cv2_for_display()
    window_name = "DexVision Camera Smoke Test"
    last_frame_time = time.monotonic()
    fps = 0.0

    print("DexVision camera smoke test")
    print(f"Opening camera_id={camera_id}, width={width}, height={height}")
    print("Press q to quit.")

    try:
        with OpenCVCamera(camera_id=camera_id, width=width, height=height) as camera:
            while True:
                result = camera.read()
                if not result.success or result.frame is None:
                    print("WARNING: Camera read failed; waiting for the next frame.")
                    if cv2_module.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                now = time.monotonic()
                elapsed = max(now - last_frame_time, 1e-9)
                last_frame_time = now
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)

                _draw_fps(cv2_module, result.frame, fps)
                cv2_module.imshow(window_name, result.frame)

                if cv2_module.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        with suppress(Exception):
            cv2_module.destroyWindow(window_name)

    print("Camera closed cleanly.")
    return 0


def _is_cv2_error(exc: Exception) -> bool:
    return exc.__class__.__module__.split(".", maxsplit=1)[0] == "cv2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_camera(args.camera_id, args.width, args.height)
    except CameraOpenError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Camera closed cleanly.")
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
