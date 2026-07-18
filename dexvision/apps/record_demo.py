"""Record Level 1 teleop runs as Level 2 demonstration episodes."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dexvision.apps import run_level1_teleop
from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.features.hand_base import (
    HandBaseTargetSmoother,
    extract_hand_base_target,
    extract_image_palm_center_target,
    normalize_quaternion,
)
from dexvision.features.hand_features import HandFeatures, extract_hand_features, no_hand_features
from dexvision.features.smoothing import FeatureSmoother
from dexvision.logging.dataset_schema import (
    FREE_SPACE_GESTURE_LABELS,
    ActionSchema,
    ObservationSchema,
)
from dexvision.logging.demo_logger import (
    DEFAULT_OBSERVATION_SCHEMA_VERSION,
    DemoLogger,
    DemoLoggerError,
    DemoStepData,
    build_level1_action_schema,
    build_level2_observation_schema,
)
from dexvision.perception.hand_tracker import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    HandTracker,
    HandTrackerError,
    HandTrackingResult,
)
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    CurlRetargeterError,
    load_curl_retargeter_config,
)
from dexvision.sim.hand_base_control import (
    HandBaseMocapController,
    HandBaseControlConfig,
    HandBaseControlStatus,
    format_hand_base_status,
    hand_base_config_from_teleop_config,
)
from dexvision.sim.mujoco_env import MujocoEnv, MujocoError, MujocoState


DEFAULT_OUTPUT = Path("data/demos/free_space_gesture")
DEFAULT_CONFIG = run_level1_teleop.DEFAULT_CONFIG
DEFAULT_CAMERA_WIDTH = run_level1_teleop.DEFAULT_CAMERA_WIDTH
DEFAULT_CAMERA_HEIGHT = run_level1_teleop.DEFAULT_CAMERA_HEIGHT
DEFAULT_CONTROL_RATE_HZ = 30.0
DEFAULT_PRINT_INTERVAL = 30
DEFAULT_CAMERA_WINDOW_NAME = "DexVision Demo Recorder"
FEATURE_FIELDS = (
    "thumb_curl",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "pinky_curl",
    "index_bend",
    "middle_bend",
    "ring_bend",
    "pinky_bend",
    "pinch_thumb_index",
    "palm_roll",
    "palm_pitch",
    "palm_yaw",
    "confidence",
)
TRACKING_QUALITY_FIELDS = (
    "detected",
    "handedness_code",
    "hand_tracking_confidence",
    "feature_confidence",
    "dropped_frame",
    "reacquired",
)
FREE_SPACE_GESTURE_INSTRUCTIONS = {
    "open_palm": "Hold an open palm toward the camera with fingers extended.",
    "fist": "Close all fingers into a fist, then hold it steady.",
    "point": "Extend the index finger while the other long fingers stay folded.",
    "pinch": "Touch or nearly touch thumb and index fingertips.",
    "peace_sign": "Extend index and middle fingers while ring and pinky stay folded.",
    "wave": "Use an open palm and wave left/right or up/down while staying in frame.",
}


@dataclass(frozen=True)
class RecordingSummary:
    """Small live-recording summary used for quality guards."""

    frames: int
    detected_frames: int
    stopped_by_preview: bool = False


BaseCommand = Literal["calibrate_base", "reset_base"]


@dataclass(frozen=True)
class RecordingPreviewEvent:
    """Keyboard commands emitted by the recording preview."""

    should_stop: bool = False
    base_commands: tuple[BaseCommand, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a Level 1 DexVision teleop run as a Level 2 demo episode."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task id to store in metadata, e.g. free_space_gesture.",
    )
    parser.add_argument(
        "--skill-name",
        default=None,
        help="Skill name to store in metadata. Defaults to --task.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Human-readable task name. Defaults to a title-cased --task.",
    )
    parser.add_argument(
        "--gesture-label",
        default=None,
        help=(
            "Optional free_space_gesture metadata label. Supported labels: "
            + ", ".join(FREE_SPACE_GESTURE_LABELS)
            + ". Spaces and hyphens are normalized to underscores."
        ),
    )
    parser.add_argument(
        "--retargeter",
        choices=("curl",),
        default="curl",
        help="Retargeter type to record. Level 2.2 supports the existing curl retargeter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Episode output directory. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Episode id. Defaults to a timestamped id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Level 1 teleop/retargeter YAML config. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override MuJoCo model XML path. Defaults to the config model_path.",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--hand-landmarker-model",
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
    parser.add_argument(
        "--assume-mirrored-input",
        action="store_true",
        help="Keep MediaPipe handedness labels for selfie-mirrored camera images.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=run_level1_teleop.DEFAULT_SMOOTHING_ALPHA,
        help="EMA smoothing alpha in (0.0, 1.0].",
    )
    parser.add_argument(
        "--min-smoothing-confidence",
        type=float,
        default=run_level1_teleop.DEFAULT_MIN_SMOOTHING_CONFIDENCE,
        help="Decay controls below this feature confidence.",
    )
    parser.add_argument(
        "--low-confidence-behavior",
        choices=("hold", "decay"),
        default=run_level1_teleop.DEFAULT_LOW_CONFIDENCE_BEHAVIOR,
        help="How smoothed controls behave when tracking confidence is low.",
    )
    parser.add_argument(
        "--decay-alpha",
        type=float,
        default=run_level1_teleop.DEFAULT_DECAY_ALPHA,
        help="EMA alpha used only when --low-confidence-behavior=decay.",
    )
    parser.add_argument(
        "--sim-steps-per-frame",
        type=int,
        default=run_level1_teleop.DEFAULT_SIM_STEPS_PER_FRAME,
        help="MuJoCo integration steps to run per camera frame.",
    )
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=DEFAULT_CONTROL_RATE_HZ,
        help="Nominal control rate recorded in metadata.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N recorded frames. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=DEFAULT_PRINT_INTERVAL,
        help="Print recording status every N frames.",
    )
    parser.add_argument(
        "--level1-13-full",
        action="store_true",
        help=(
            "Record with the full Level 1.13 teleop interface: camera preview, "
            "MuJoCo viewer, base x/y/depth/orientation control, finger control, "
            "hand-detection guard, and recording gated until c calibrates/centers."
        ),
    )
    parser.add_argument(
        "--show-camera-window",
        action="store_true",
        help="Show a live OpenCV preview while recording. Press q in the window to stop.",
    )
    parser.add_argument(
        "--auto-calibrate-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Automatically capture the first valid hand pose as the base/depth/"
            "orientation neutral pose. Intended for smoke tests, not manual datasets."
        ),
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--start-on-calibration",
        dest="start_on_calibration",
        action="store_true",
        default=None,
        help="Wait to save live frames until c successfully calibrates/centers the hand.",
    )
    start_group.add_argument(
        "--record-immediately",
        dest="start_on_calibration",
        action="store_false",
        help="Save live frames immediately instead of waiting for c.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a MuJoCo viewer while recording. On macOS, run with mjpython.",
    )
    parser.add_argument(
        "--viewer-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each MuJoCo viewer sync.",
    )
    parser.add_argument(
        "--camera-window-name",
        default=DEFAULT_CAMERA_WINDOW_NAME,
        help=f"OpenCV preview window title. Defaults to {DEFAULT_CAMERA_WINDOW_NAME!r}.",
    )
    parser.add_argument(
        "--save-landmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save normalized image landmarks when available. Defaults to true.",
    )
    parser.add_argument(
        "--require-hand-detected",
        action="store_true",
        help="Fail instead of saving when too few frames contain a detected hand.",
    )
    parser.add_argument(
        "--min-hand-detected-frames",
        type=int,
        default=1,
        help="Minimum detected-hand frames required with --require-hand-detected.",
    )
    parser.add_argument(
        "--enable-base-control",
        action="store_true",
        help="Record and apply the configured Level 1.13 hand-base control.",
    )
    parser.add_argument(
        "--enable-base-orientation",
        action="store_true",
        help="Record relative palm orientation when base control is enabled.",
    )
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument(
        "--enable-depth-control",
        dest="enable_depth_control",
        action="store_true",
        default=None,
        help="Enable monocular hand-scale depth/in-out base control.",
    )
    depth_group.add_argument(
        "--disable-depth-control",
        dest="enable_depth_control",
        action="store_false",
        help="Disable monocular hand-scale depth/in-out base control.",
    )
    success_group = parser.add_mutually_exclusive_group()
    success_group.add_argument(
        "--success",
        dest="success",
        action="store_true",
        default=None,
        help="Mark the saved episode as successful.",
    )
    success_group.add_argument(
        "--failure",
        dest="success",
        action="store_false",
        help="Mark the saved episode as failed.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Record a short synthetic episode for automated smoke tests; no camera or MuJoCo.",
    )
    return parser


def run_record_demo(args: argparse.Namespace) -> int:
    """Run synthetic or live demo recording from parsed CLI args."""

    _apply_recording_presets(args)
    _validate_recording_args(args)
    raw_config = load_curl_retargeter_config(args.config)
    retargeter = CurlRetargeter.from_mapping(raw_config)
    target_names = run_level1_teleop.robot_target_names(retargeter)
    action_schema = build_level1_action_schema(target_names)
    episode_id = args.episode_id or _default_episode_id(args.task)
    model_path = run_level1_teleop.resolve_mujoco_model_path(
        raw_config,
        config_path=args.config,
        override=args.model,
    )

    print("DexVision Level 2 demo recorder")
    print(f"Task: {args.task}")
    print(f"Skill: {args.skill_name or args.task}")
    if args.gesture_label is not None:
        print(f"Gesture label: {args.gesture_label}")
    print(f"Episode: {episode_id}")
    print(f"Output: {args.output}")
    print(f"Retargeter: {args.retargeter} ({args.config})")
    print(f"Action schema: {action_schema.version}, dim={action_schema.action_dim}")
    print(f"Robot targets: {', '.join(target_names)}")
    print("Video recording: off")
    if args.level1_13_full:
        print("Recording preset: full Level 1.13 teleop")
    if args.task == "free_space_gesture":
        _print_free_space_recording_guide(args.gesture_label)

    if args.synthetic:
        return _run_synthetic_recording(
            args=args,
            raw_config=raw_config,
            model_path=model_path,
            retargeter=retargeter,
            target_names=target_names,
            action_schema=action_schema,
            episode_id=episode_id,
        )

    return _run_live_recording(
        args=args,
        raw_config=raw_config,
        model_path=model_path,
        retargeter=retargeter,
        target_names=target_names,
        action_schema=action_schema,
        episode_id=episode_id,
    )


def _run_synthetic_recording(
    *,
    args: argparse.Namespace,
    raw_config: Mapping[str, Any],
    model_path: Path,
    retargeter: CurlRetargeter,
    target_names: tuple[str, ...],
    action_schema: ActionSchema,
    episode_id: str,
) -> int:
    max_frames = args.max_frames if args.max_frames > 0 else 5
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=4,
        robot_qvel_dim=4,
        finger_target_dim=len(target_names),
        tracking_quality_dim=len(TRACKING_QUALITY_FIELDS),
    )
    logger = DemoLogger(
        args.output,
        action_schema=action_schema,
        observation_schema=observation_schema,
        overwrite=args.overwrite,
    )
    logger.start_episode(
        _metadata(
            args=args,
            episode_id=episode_id,
            raw_config=raw_config,
            model_path=model_path,
            target_names=target_names,
            observation_schema=observation_schema,
            synthetic=True,
        )
    )

    base_position = np.asarray([0.0, 0.0, 0.14], dtype=np.float64)
    base_orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    start_time = time.monotonic()
    for frame_index in range(max_frames):
        phase = frame_index / max(max_frames - 1, 1)
        features = HandFeatures(
            thumb_curl=phase,
            index_curl=phase,
            middle_curl=phase,
            ring_curl=phase,
            pinky_curl=phase,
            confidence=1.0,
        )
        targets = run_level1_teleop.build_full_hand_targets(retargeter, features)
        action = action_vector(
            base_position=base_position,
            base_orientation=base_orientation,
            targets=targets,
            target_names=target_names,
        )
        state = MujocoState(
            time=frame_index / args.control_rate_hz,
            qpos=np.full(4, phase, dtype=np.float64),
            qvel=np.zeros(4, dtype=np.float64),
            ctrl=ordered_targets(targets, target_names),
        )
        logger.append(
            DemoStepData(
                features=feature_vector(features),
                action=action,
                robot_state=robot_state_vector(
                    state,
                    base_position=base_position,
                    base_orientation=base_orientation,
                ),
                tracking_quality=np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64),
                timestamp=start_time + (frame_index / args.control_rate_hz),
                landmarks=np.zeros((21, 3), dtype=np.float64) if args.save_landmarks else None,
            )
        )

    episode = logger.close(success=args.success)
    print(f"Saved synthetic demo with {episode.timestamps.shape[0]} frames: {args.output}")
    return 0


def _run_live_recording(
    *,
    args: argparse.Namespace,
    raw_config: Mapping[str, Any],
    model_path: Path,
    retargeter: CurlRetargeter,
    target_names: tuple[str, ...],
    action_schema: ActionSchema,
    episode_id: str,
) -> int:
    base_config = _base_config(raw_config, args=args)
    print(f"Camera: id={args.camera_id}, width={args.width}, height={args.height}")
    print(f"Hand tracker model: {args.hand_landmarker_model or DEFAULT_HAND_LANDMARKER_MODEL}")
    print(f"MuJoCo model: {model_path}")
    print(
        "Base control: "
        f"{'on' if base_config.enabled else 'off'} "
        f"(mode: {base_config.base_control_mode}, mocap body: {base_config.mocap_body_name})"
    )
    if base_config.enabled:
        print(
            "Calibration/start gate: put your neutral palm pose in frame, then press c "
            "in the camera preview to calibrate/center it. Recording starts after that succeeds; "
            "press r to reset base control."
        )
        print(f"Base auto-calibration: {'on' if args.auto_calibrate_base else 'off'}")
    elif args.start_on_calibration:
        print("Recording gate: press c in the camera preview before saving frames.")
    print(f"Camera preview: {'on' if args.show_camera_window else 'off'}")
    if args.show_camera_window:
        print("Preview keys: c calibrate/center, q stop/save, r reset base.")
    print(f"MuJoCo viewer: {'on' if args.viewer else 'off'}")
    if args.require_hand_detected:
        print(
            "Hand-detection guard: "
            f"requires at least {args.min_hand_detected_frames} detected frame(s)."
        )
    print("Stop with Ctrl-C, --max-frames, or --duration-seconds.")

    smoother = FeatureSmoother(
        alpha=args.smoothing_alpha,
        min_confidence=args.min_smoothing_confidence,
        low_confidence_behavior=args.low_confidence_behavior,
        decay_alpha=args.decay_alpha,
    )

    preview = (
        RecordingPreview(window_name=args.camera_window_name)
        if args.show_camera_window
        else None
    )
    try:
        with (
            OpenCVCamera(camera_id=args.camera_id, width=args.width, height=args.height) as camera,
            HandTracker(
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                model_path=args.hand_landmarker_model,
                assume_mirrored_input=args.assume_mirrored_input,
            ) as tracker,
            MujocoEnv(model_path) as env,
        ):
            env.reset()
            base_controller = (
                HandBaseMocapController(env, base_config) if base_config.enabled else None
            )
            base_smoother = (
                HandBaseTargetSmoother(
                    alpha=base_config.base_smoothing_alpha,
                    min_confidence=base_config.min_confidence,
                )
                if base_config.enabled
                and (
                    base_config.base_control_mode == "pose_3d"
                    or base_config.enable_base_orientation
                )
                else None
            )
            neutral_targets = run_level1_teleop.build_full_hand_targets(
                retargeter,
                no_hand_features(),
            )
            env.set_joint_targets(neutral_targets)
            env.step(n_steps=max(1, args.sim_steps_per_frame))
            initial_state = env.get_state()
            observation_schema = build_level2_observation_schema(
                robot_qpos_dim=initial_state.qpos.size,
                robot_qvel_dim=initial_state.qvel.size,
                finger_target_dim=len(target_names),
                tracking_quality_dim=len(TRACKING_QUALITY_FIELDS),
            )
            logger = DemoLogger(
                args.output,
                action_schema=action_schema,
                observation_schema=observation_schema,
                overwrite=args.overwrite,
            )
            logger.start_episode(
                _metadata(
                    args=args,
                    episode_id=episode_id,
                    raw_config=raw_config,
                    model_path=model_path,
                    target_names=target_names,
                    observation_schema=observation_schema,
                    synthetic=False,
                )
            )
            try:
                summary = _record_live_with_optional_viewer(
                    args=args,
                    env=env,
                    camera=camera,
                    tracker=tracker,
                    smoother=smoother,
                    retargeter=retargeter,
                    target_names=target_names,
                    base_config=base_config,
                    base_controller=base_controller,
                    base_smoother=base_smoother,
                    logger=logger,
                    preview=preview,
                )
            except KeyboardInterrupt:
                if logger.step_count == 0:
                    raise
                print("\nInterrupted; saving the recorded frames collected so far.")
                summary = RecordingSummary(
                    frames=logger.step_count,
                    detected_frames=_detected_frames_from_logger(logger),
                )

            if logger.step_count == 0:
                raise DemoLoggerError(
                    "recording stopped before any frames were saved. "
                    "Put your neutral hand pose in frame, press c to calibrate/center it, "
                    "then perform the gesture before pressing q."
                )
            _validate_detection_guard(args=args, summary=summary)
            if summary.detected_frames == 0:
                print(
                    "WARNING: Saved demo contains no detected-hand frames. "
                    "Use --show-camera-window and --require-hand-detected for manual demos."
                )
            episode = logger.close(success=args.success)
            print(f"Saved demo with {episode.timestamps.shape[0]} frames: {args.output}")
            return 0
    finally:
        if preview is not None:
            preview.close()


def _record_live_with_optional_viewer(
    *,
    args: argparse.Namespace,
    env: MujocoEnv,
    camera: OpenCVCamera,
    tracker: HandTracker,
    smoother: FeatureSmoother,
    retargeter: CurlRetargeter,
    target_names: tuple[str, ...],
    base_config: HandBaseControlConfig,
    base_controller: HandBaseMocapController | None,
    base_smoother: HandBaseTargetSmoother | None,
    logger: DemoLogger,
    preview: "RecordingPreview | None",
) -> RecordingSummary:
    if not args.viewer:
        return _record_live_loop(
            args=args,
            env=env,
            camera=camera,
            tracker=tracker,
            smoother=smoother,
            retargeter=retargeter,
            target_names=target_names,
            base_config=base_config,
            base_controller=base_controller,
            base_smoother=base_smoother,
            logger=logger,
            preview=preview,
            viewer_handle=None,
        )

    _ensure_mujoco_viewer_can_launch(args)
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    try:
        with viewer.launch_passive(env.model, env.data) as viewer_handle:
            return _record_live_loop(
                args=args,
                env=env,
                camera=camera,
                tracker=tracker,
                smoother=smoother,
                retargeter=retargeter,
                target_names=target_names,
                base_config=base_config,
                base_controller=base_controller,
                base_smoother=base_smoother,
                logger=logger,
                preview=preview,
                viewer_handle=viewer_handle,
            )
    except Exception as exc:  # pragma: no cover - requires desktop GUI to exercise.
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc


def _record_live_loop(
    *,
    args: argparse.Namespace,
    env: MujocoEnv,
    camera: OpenCVCamera,
    tracker: HandTracker,
    smoother: FeatureSmoother,
    retargeter: CurlRetargeter,
    target_names: tuple[str, ...],
    base_config: HandBaseControlConfig,
    base_controller: HandBaseMocapController | None,
    base_smoother: HandBaseTargetSmoother | None,
    logger: DemoLogger,
    preview: "RecordingPreview | None" = None,
    viewer_handle: object | None = None,
) -> RecordingSummary:
    start_time = time.monotonic()
    previous_detected = False
    recorded_frame_index = 0
    preview_frame_index = 0
    detected_frames = 0
    stopped_by_preview = False
    pending_base_commands: tuple[BaseCommand, ...] = ()
    auto_base_calibrated = False
    recording_started = not args.start_on_calibration
    recording_start_time = start_time if recording_started else None

    while True:
        if recording_started and args.max_frames > 0 and recorded_frame_index >= args.max_frames:
            break
        if (
            recording_started
            and recording_start_time is not None
            and args.duration_seconds > 0.0
            and (time.monotonic() - recording_start_time) >= args.duration_seconds
        ):
            break

        camera_result = camera.read()
        if not camera_result.success or camera_result.frame is None:
            print("WARNING: Camera read failed; waiting for the next frame.")
            continue

        tracking_result = tracker.process(
            camera_result.frame,
            timestamp=camera_result.timestamp,
        )
        raw_features = extract_hand_features(tracking_result)
        smoothed_features = smoother.update(raw_features)
        targets = run_level1_teleop.build_full_hand_targets(retargeter, smoothed_features)
        env.set_joint_targets(targets)
        applied_base_commands = pending_base_commands
        pending_base_commands = ()
        if (
            args.auto_calibrate_base
            and base_controller is not None
            and not auto_base_calibrated
            and "reset_base" not in applied_base_commands
            and "calibrate_base" not in applied_base_commands
        ):
            auto_base_calibrated = _maybe_auto_calibrate_base(
                tracking_result=tracking_result,
                base_controller=base_controller,
                base_smoother=base_smoother,
            )
        base_status = _apply_base_control_for_recording(
            tracking_result=tracking_result,
            base_controller=base_controller,
            base_smoother=base_smoother,
            commands=applied_base_commands,
        )
        if (
            not recording_started
            and _calibration_started_recording(
                commands=applied_base_commands,
                base_status=base_status,
                base_control_enabled=base_controller is not None,
            )
        ):
            recording_started = True
            recording_start_time = time.monotonic()
            print(
                "Recording started after successful c calibration. "
                f"{_format_recording_status(recording_started=True, recorded_frames=0, gesture_label=args.gesture_label)}"
            )
        if "reset_base" in applied_base_commands:
            auto_base_calibrated = False
        elif "calibrate_base" in applied_base_commands and base_status is not None:
            auto_base_calibrated = bool(base_status.neutral_captured)
        state = env.step(n_steps=args.sim_steps_per_frame)
        run_level1_teleop._raise_if_unstable(
            state,
            max_abs_qvel=(
                run_level1_teleop.MAX_ABS_BASE_CONTROL_QVEL
                if base_controller is not None
                else run_level1_teleop.MAX_ABS_QVEL
            ),
        )

        base_position, base_orientation = _recorded_base_pose(
            env=env,
            base_config=base_config,
            base_status=base_status,
        )
        if recording_started:
            logger.append(
                DemoStepData(
                    features=feature_vector(smoothed_features),
                    action=action_vector(
                        base_position=base_position,
                        base_orientation=base_orientation,
                        targets=targets,
                        target_names=target_names,
                    ),
                    robot_state=robot_state_vector(
                        state,
                        base_position=base_position,
                        base_orientation=base_orientation,
                    ),
                    tracking_quality=tracking_quality_vector(
                        tracking_result=tracking_result,
                        features=smoothed_features,
                        dropped_frame=False,
                        reacquired=tracking_result.detected and not previous_detected,
                    ),
                    timestamp=camera_result.timestamp,
                    landmarks=(
                        landmarks_array(tracking_result)
                        if args.save_landmarks
                        else None
                    ),
                )
            )
            if tracking_result.detected:
                detected_frames += 1
            previous_detected = tracking_result.detected
            recorded_frame_index += 1

        preview_frame_index += 1
        if recording_started and (
            recorded_frame_index == 1 or recorded_frame_index % args.print_interval == 0
        ):
            print(
                f"recorded={recorded_frame_index:05d} "
                f"detected={tracking_result.detected} "
                f"confidence={raw_features.confidence:.2f} "
                f"{_format_recording_status(recording_started=True, recorded_frames=recorded_frame_index, gesture_label=args.gesture_label)} "
                f"{run_level1_teleop._format_control_summary(smoothed_features)} "
                f"{format_hand_base_status(base_status)} "
                f"sim_t={state.time:.3f}s"
            )
        elif not recording_started and (
            preview_frame_index == 1 or preview_frame_index % args.print_interval == 0
        ):
            print(
                f"armed={preview_frame_index:05d} "
                f"detected={tracking_result.detected} "
                f"confidence={raw_features.confidence:.2f} "
                f"{_format_recording_status(recording_started=False, recorded_frames=0, gesture_label=args.gesture_label)} "
                f"{format_hand_base_status(base_status)} "
                f"sim_t={state.time:.3f}s"
            )
        if preview is not None:
            preview_event = preview.show(
                frame=camera_result.frame,
                tracking_result=tracking_result,
                raw_features=raw_features,
                features=smoothed_features,
                targets=targets,
                target_names=target_names,
                frame_index=preview_frame_index,
                state=state,
                base_status=base_status,
                recording_started=recording_started,
                recorded_frames=recorded_frame_index,
                gesture_label=args.gesture_label,
            )
            pending_base_commands = preview_event.base_commands
            if preview_event.should_stop:
                stopped_by_preview = True
                break
        if viewer_handle is not None:
            viewer_handle.sync()
            if args.viewer_sleep > 0.0:
                time.sleep(args.viewer_sleep)
            if _viewer_was_closed(viewer_handle):
                break

    return RecordingSummary(
        frames=recorded_frame_index,
        detected_frames=detected_frames,
        stopped_by_preview=stopped_by_preview,
    )


def _apply_base_control_for_recording(
    *,
    tracking_result: HandTrackingResult,
    base_controller: HandBaseMocapController | None,
    base_smoother: HandBaseTargetSmoother | None,
    commands: tuple[BaseCommand, ...] = (),
) -> HandBaseControlStatus | None:
    if base_controller is None:
        return None
    return run_level1_teleop._apply_hand_base_control(
        tracking_result=tracking_result,
        base_controller=base_controller,
        base_smoother=base_smoother,
        commands=commands,
    )


def _maybe_auto_calibrate_base(
    *,
    tracking_result: HandTrackingResult,
    base_controller: HandBaseMocapController,
    base_smoother: HandBaseTargetSmoother | None,
) -> bool:
    if base_controller.config.base_control_mode != "image_2d":
        return False
    image_target = extract_image_palm_center_target(
        tracking_result,
        depth_source=base_controller.config.depth_source,
    )
    orientation_target = extract_hand_base_target(
        tracking_result,
        position_source=base_controller.config.position_source,
    )
    if base_smoother is not None:
        base_smoother.reset()
    calibrated = base_controller.calibrate_image_2d(
        image_target,
        orientation_target=orientation_target,
    )
    if calibrated:
        print(
            "Base control auto-calibrated: first confident palm center, scale, "
            "and orientation now map to the robot neutral/base pose."
        )
    return calibrated


def _calibration_started_recording(
    *,
    commands: tuple[BaseCommand, ...],
    base_status: HandBaseControlStatus | None,
    base_control_enabled: bool,
) -> bool:
    if "calibrate_base" not in commands:
        return False
    if not base_control_enabled:
        return True
    return bool(base_status is not None and base_status.neutral_captured)


def _format_recording_status(
    *,
    recording_started: bool,
    recorded_frames: int,
    gesture_label: str | None,
) -> str:
    label = gesture_label or "unlabeled"
    if recording_started:
        return f"recording=on frames={recorded_frames} gesture={label}"
    return f"recording=armed press-c-to-calibrate gesture={label}"


def _print_free_space_recording_guide(gesture_label: str | None) -> None:
    print("Free-space gesture recording plan:")
    print("  Start each clip from a neutral upright palm in frame, press c to calibrate/center, perform/hold 3-5 seconds, then press q.")
    if gesture_label is not None:
        print(f"  This clip label: {gesture_label}")
        print(f"  What to record: {FREE_SPACE_GESTURE_INSTRUCTIONS[gesture_label]}")
        return
    print("  Suggested 10-demo set: open_palm x2, fist x2, pinch x2, wave x2, point x1, peace_sign x1.")
    print("  Use --gesture-label open_palm|fist|point|pinch|peace_sign|wave for labeled clips.")


class RecordingPreview:
    """Separate-process camera overlay for manual demo recording."""

    def __init__(self, *, window_name: str) -> None:
        if not window_name:
            raise DemoLoggerError("camera_window_name must be a non-empty string.")
        self._overlay = run_level1_teleop.CameraOverlayProcess(
            window_name=window_name,
        ).start()

    def show(
        self,
        *,
        frame: np.ndarray,
        tracking_result: HandTrackingResult,
        raw_features: HandFeatures,
        features: HandFeatures,
        targets: dict[str, float],
        target_names: tuple[str, ...],
        frame_index: int,
        state: MujocoState,
        base_status: HandBaseControlStatus | None,
        recording_started: bool,
        recorded_frames: int,
        gesture_label: str | None,
    ) -> RecordingPreviewEvent:
        """Send one preview frame and return queued keyboard commands."""

        del frame_index
        self._overlay.send(
            run_level1_teleop.CameraOverlayFrame(
                frame=frame.copy(),
                tracking_result=tracking_result,
                raw_features=raw_features,
                smoothed_features=features,
                targets=dict(targets),
                target_names=target_names,
                fps=0.0,
                simulation_time=state.time,
                status_message=run_level1_teleop._format_tracking_status(
                    detected=tracking_result.detected,
                    confidence=raw_features.confidence,
                    min_confidence=0.0,
                    low_confidence_behavior="decay",
                ),
                base_status_message=(
                    _format_recording_status(
                        recording_started=recording_started,
                        recorded_frames=recorded_frames,
                        gesture_label=gesture_label,
                    )
                    + " | "
                    + format_hand_base_status(base_status)
                ),
            )
        )
        commands = tuple(self._overlay.poll_commands())
        return RecordingPreviewEvent(
            should_stop=self._overlay.should_stop(),
            base_commands=commands,
        )

    def close(self) -> None:
        """Close the separate overlay process."""

        self._overlay.close()


def feature_vector(features: HandFeatures) -> np.ndarray:
    """Flatten ``HandFeatures`` into the Level 2.2 logged feature vector."""

    return np.asarray(
        [float(getattr(features, field)) for field in FEATURE_FIELDS],
        dtype=np.float64,
    )


def ordered_targets(targets: dict[str, float], target_names: tuple[str, ...]) -> np.ndarray:
    """Return actuator targets in the metadata/action-schema order."""

    return np.asarray([float(targets[name]) for name in target_names], dtype=np.float64)


def action_vector(
    *,
    base_position: np.ndarray,
    base_orientation: np.ndarray,
    targets: dict[str, float],
    target_names: tuple[str, ...],
) -> np.ndarray:
    """Build one full Level 1.13 action vector."""

    position = np.asarray(base_position, dtype=np.float64)
    orientation = normalize_quaternion(np.asarray(base_orientation, dtype=np.float64))
    if position.shape != (3,):
        raise DemoLoggerError(f"base_position must have shape [3], got {position.shape}.")
    if orientation.shape != (4,):
        raise DemoLoggerError(f"base_orientation must have shape [4], got {orientation.shape}.")
    return np.concatenate((position, orientation, ordered_targets(targets, target_names)))


def robot_state_vector(
    state: MujocoState,
    *,
    base_position: np.ndarray,
    base_orientation: np.ndarray,
) -> np.ndarray:
    """Flatten MuJoCo state plus commanded base target for recording."""

    return np.concatenate(
        (
            np.asarray(state.qpos, dtype=np.float64),
            np.asarray(state.qvel, dtype=np.float64),
            np.asarray(state.ctrl, dtype=np.float64),
            np.asarray(base_position, dtype=np.float64),
            normalize_quaternion(np.asarray(base_orientation, dtype=np.float64)),
        )
    )


def tracking_quality_vector(
    *,
    tracking_result: HandTrackingResult,
    features: HandFeatures,
    dropped_frame: bool,
    reacquired: bool,
) -> np.ndarray:
    """Flatten tracking quality into the Level 2.2 logged quality vector."""

    return np.asarray(
        [
            1.0 if tracking_result.detected else 0.0,
            _handedness_code(tracking_result.handedness),
            _clip01(tracking_result.confidence),
            _clip01(features.confidence),
            1.0 if dropped_frame else 0.0,
            1.0 if reacquired else 0.0,
        ],
        dtype=np.float64,
    )


def landmarks_array(tracking_result: HandTrackingResult) -> np.ndarray:
    """Return normalized image landmarks or finite zeros for no-hand frames."""

    if tracking_result.image_landmarks is None:
        return np.zeros((21, 3), dtype=np.float64)
    landmarks = np.asarray(tracking_result.image_landmarks, dtype=np.float64)
    if landmarks.shape != (21, 3):
        raise DemoLoggerError(f"image landmarks must have shape [21, 3], got {landmarks.shape}.")
    return landmarks


def _metadata(
    *,
    args: argparse.Namespace,
    episode_id: str,
    raw_config: Mapping[str, Any],
    model_path: Path,
    target_names: tuple[str, ...],
    observation_schema: ObservationSchema,
    synthetic: bool,
) -> dict[str, Any]:
    task_config = _default_task_config(args.task)
    metadata: dict[str, Any] = {
        "skill_name": args.skill_name or args.task,
        "task_name": args.task_name or _default_task_name(args.task),
        "task_id": args.task,
        "episode_id": episode_id,
        "action_schema_version": "level1.13/full-action-v1",
        "observation_schema_version": DEFAULT_OBSERVATION_SCHEMA_VERSION,
        "robot_model": str(model_path),
        "retargeter_config": str(args.config),
        "control_rate_hz": float(args.control_rate_hz),
        "teleop_config": raw_config,
        "task_config": task_config,
        "feature_fields": FEATURE_FIELDS,
        "tracking_quality_fields": TRACKING_QUALITY_FIELDS,
        "finger_target_names": target_names,
        "robot_state_layout": {
            "contents": (
                "qpos",
                "qvel",
                "ctrl",
                "base_position_target",
                "base_orientation_target",
            ),
            "note": "Flattened MuJoCo state followed by commanded Level 1.13 base target.",
        },
        "recording": {
            "synthetic": bool(synthetic),
            "camera_id": int(args.camera_id),
            "width": int(args.width),
            "height": int(args.height),
            "show_camera_window": bool(args.show_camera_window),
            "auto_calibrate_base": bool(args.auto_calibrate_base),
            "start_on_calibration": bool(args.start_on_calibration),
            "viewer": bool(args.viewer),
            "level1_13_full": bool(args.level1_13_full),
            "require_hand_detected": bool(args.require_hand_detected),
            "min_hand_detected_frames": int(args.min_hand_detected_frames),
            "save_landmarks": bool(args.save_landmarks),
            "max_frames": int(args.max_frames),
            "duration_seconds": float(args.duration_seconds),
            "observation_fields": observation_schema.fields,
        },
    }
    if args.gesture_label is not None:
        metadata["gesture_label"] = args.gesture_label
    return metadata


def _default_task_config(task_id: str) -> dict[str, Any]:
    if task_id == "free_space_gesture":
        return {
            "required_objects": (),
            "requires_task_state": False,
            "requires_success_metric_inputs": False,
            "required_observation_fields": (),
            "gesture_labels": FREE_SPACE_GESTURE_LABELS,
        }
    return {
        "required_objects": (),
        "requires_task_state": False,
        "requires_success_metric_inputs": False,
        "required_observation_fields": (),
        "note": "No task object/state provider is implemented in Level 2.2.",
    }


def _base_config(
    raw_config: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> HandBaseControlConfig:
    base_config = hand_base_config_from_teleop_config(
        raw_config,
        enable_override=args.enable_base_control,
    )
    if args.enable_base_orientation:
        base_config = replace(base_config, enable_base_orientation=True)
    if args.enable_depth_control is not None:
        base_config = replace(base_config, enable_depth_control=args.enable_depth_control)
    return base_config


def _recorded_base_pose(
    *,
    env: MujocoEnv,
    base_config: HandBaseControlConfig,
    base_status: HandBaseControlStatus | None,
) -> tuple[np.ndarray, np.ndarray]:
    if base_status is not None and base_status.applied_target.valid:
        return (
            base_status.applied_target.position.copy(),
            base_status.applied_target.orientation_quat.copy(),
        )
    try:
        return env.get_mocap_pose(base_config.mocap_body_name)
    except MujocoError:
        return (
            base_config.neutral_base_position.copy(),
            base_config.rotation_offset_quat.copy(),
        )


def _validate_recording_args(args: argparse.Namespace) -> None:
    args.gesture_label = _normalize_gesture_label(args.gesture_label)
    if args.gesture_label is not None and args.task != "free_space_gesture":
        raise ValueError("--gesture-label is only supported with --task free_space_gesture.")
    if args.start_on_calibration is None:
        args.start_on_calibration = False
    if args.start_on_calibration and not args.show_camera_window:
        raise ValueError("--start-on-calibration requires --show-camera-window so c can be pressed.")
    if args.max_frames < 0:
        raise ValueError("max_frames must be non-negative.")
    if args.duration_seconds < 0.0:
        raise ValueError("duration_seconds must be non-negative.")
    if args.control_rate_hz <= 0.0:
        raise ValueError("control_rate_hz must be positive.")
    if args.sim_steps_per_frame <= 0:
        raise ValueError("sim_steps_per_frame must be positive.")
    if args.viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")
    if args.print_interval <= 0:
        raise ValueError("print_interval must be positive.")
    if args.min_hand_detected_frames <= 0:
        raise ValueError("min_hand_detected_frames must be positive.")


def _apply_recording_presets(args: argparse.Namespace) -> None:
    if not args.level1_13_full:
        return
    args.show_camera_window = True
    args.viewer = True
    args.enable_base_control = True
    args.enable_base_orientation = True
    args.auto_calibrate_base = False
    if args.start_on_calibration is None:
        args.start_on_calibration = True
    if args.enable_depth_control is None:
        args.enable_depth_control = True
    args.require_hand_detected = True
    args.min_hand_detected_frames = max(args.min_hand_detected_frames, 10)


def _validate_detection_guard(
    *,
    args: argparse.Namespace,
    summary: RecordingSummary,
) -> None:
    if not args.require_hand_detected:
        return
    if summary.detected_frames >= args.min_hand_detected_frames:
        return
    raise DemoLoggerError(
        "recording did not contain enough detected-hand frames: "
        f"required {args.min_hand_detected_frames}, got {summary.detected_frames}. "
        "No demo was saved; put your hand in frame and rerun with --show-camera-window."
    )


def _detected_frames_from_logger(logger: DemoLogger) -> int:
    steps = getattr(logger, "_steps", ())
    return sum(1 for step in steps if step.tracking_quality.size and step.tracking_quality[0] >= 0.5)


def _preview_event_from_key(key: int) -> RecordingPreviewEvent:
    if key == ord("q"):
        return RecordingPreviewEvent(should_stop=True)
    if key == ord("c"):
        return RecordingPreviewEvent(base_commands=("calibrate_base",))
    if key == ord("r"):
        return RecordingPreviewEvent(base_commands=("reset_base",))
    return RecordingPreviewEvent()


def _ensure_mujoco_viewer_can_launch(args: argparse.Namespace) -> None:
    if sys.platform != "darwin":
        return

    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    mjpython_base = getattr(viewer, "_MjPythonBase", None)
    mjpython_dispatcher = getattr(viewer, "_MJPYTHON", None)
    if mjpython_base is not None and isinstance(mjpython_dispatcher, mjpython_base):
        return

    command = [
        "mjpython",
        "-m",
        "dexvision.apps.record_demo",
        "--task",
        args.task,
        "--retargeter",
        args.retargeter,
        "--output",
        str(args.output),
    ]
    if args.level1_13_full:
        command.append("--level1-13-full")
        if args.start_on_calibration is False:
            command.append("--record-immediately")
    else:
        command.append("--viewer")
        if args.show_camera_window:
            command.append("--show-camera-window")
        if args.start_on_calibration:
            command.append("--start-on-calibration")
        if args.enable_base_control:
            command.append("--enable-base-control")
        if args.enable_base_orientation:
            command.append("--enable-base-orientation")
        if args.enable_depth_control is True:
            command.append("--enable-depth-control")
        elif args.enable_depth_control is False:
            command.append("--disable-depth-control")
    if args.require_hand_detected:
        command.append("--require-hand-detected")
        command.extend(["--min-hand-detected-frames", str(args.min_hand_detected_frames)])
    if args.max_frames:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.gesture_label is not None:
        command.extend(["--gesture-label", args.gesture_label])
    raise MujocoError(
        "MuJoCo viewer on macOS requires the mjpython launcher.\n"
        "Run this from a regular macOS Terminal or iTerm session:\n"
        f"  {' '.join(command)}"
    )


def _viewer_was_closed(viewer_handle: object) -> bool:
    is_running = getattr(viewer_handle, "is_running", None)
    if not callable(is_running):
        return False
    return not bool(is_running())


def _default_episode_id(task_id: str) -> str:
    return f"{task_id}_{time.strftime('%Y%m%d_%H%M%S')}"


def _default_task_name(task_id: str) -> str:
    return task_id.replace("_", " ").title()


def _normalize_gesture_label(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = "_".join(str(label).strip().lower().replace("-", " ").split())
    if not normalized:
        raise ValueError("gesture_label must be non-empty when provided.")
    if normalized not in FREE_SPACE_GESTURE_LABELS:
        allowed = ", ".join(FREE_SPACE_GESTURE_LABELS)
        raise ValueError(f"gesture_label must be one of: {allowed}.")
    return normalized


def _handedness_code(handedness: str | None) -> float:
    if handedness == "Left":
        return -1.0
    if handedness == "Right":
        return 1.0
    return 0.0


def _clip01(value: float) -> float:
    return float(np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_record_demo(args)
    except KeyboardInterrupt:
        print("\nInterrupted before the episode could be closed.", file=sys.stderr)
        return 130
    except (
        CameraOpenError,
        CurlRetargeterError,
        DemoLoggerError,
        HandTrackerError,
        MujocoError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
