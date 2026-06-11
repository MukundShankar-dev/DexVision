"""Smoke-test Shadow Hand base/wrist pose control."""

from __future__ import annotations

import argparse
import sys

from dexvision.apps import run_level1_teleop
from dexvision.camera.opencv_camera import CameraOpenError
from dexvision.perception.hand_tracker import HandTrackerError
from dexvision.retargeting.curl_retargeter import CurlRetargeterError
from dexvision.sim.mujoco_env import MujocoError


def build_parser() -> argparse.ArgumentParser:
    """Build the Level 1.13 smoke-test parser."""

    parser = run_level1_teleop.build_parser()
    parser.prog = "python -m dexvision.apps.check_hand_base_control"
    parser.description = "Smoke-test DexVision hand base/wrist pose control."
    parser.set_defaults(
        show_camera_window=True,
        enable_base_control=True,
        enable_depth_control=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("DexVision Level 1.13 hand base control smoke test")
    print("Base control and the camera overlay are enabled by default in this app.")
    print("Default mode is calibrated image_2d translation plus hand-scale depth.")
    print("Orientation is still opt-in; Level 1.13B does not apply rotation by default.")
    print("Calibration treats your current palm center and scale as neutral.")
    print("Press c in the camera overlay to calibrate neutral; press r to reset neutral.")
    print("Move your hand before collecting Level 2 demos only after this manual check passes.")
    try:
        return run_level1_teleop.run_level1_teleop(
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            config_path=args.config,
            mujoco_model_path=args.model,
            hand_landmarker_model_path=args.hand_landmarker_model,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            assume_mirrored_input=args.assume_mirrored_input,
            smoothing_alpha=args.smoothing_alpha,
            min_smoothing_confidence=args.min_smoothing_confidence,
            low_confidence_behavior=args.low_confidence_behavior,
            decay_alpha=args.decay_alpha,
            sim_steps_per_frame=args.sim_steps_per_frame,
            viewer_sleep=args.viewer_sleep,
            print_interval=args.print_interval,
            show_camera_window=args.show_camera_window,
            enable_base_control=True,
            base_control_mode=args.base_control_mode,
            enable_base_orientation=args.enable_base_orientation,
            enable_depth_control=args.enable_depth_control,
            camera_window_name=args.camera_window_name,
        )
    except (
        CameraOpenError,
        CurlRetargeterError,
        HandTrackerError,
        MujocoError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Hand base control smoke test closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
