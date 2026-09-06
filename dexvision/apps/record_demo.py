"""Record Level 1 teleop runs as Level 2 demonstration episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from copy import deepcopy
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import yaml

from dexvision.apps import run_level1_teleop
from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.features.hand_base import (
    HandBaseTargetSmoother,
    extract_hand_base_target,
    extract_image_palm_center_target,
    normalize_quaternion,
)
from dexvision.features.hand_features import (
    HandFeatures,
    extract_hand_features,
    no_hand_features,
)
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
from dexvision.logging.phase_labels import DEFAULT_PICK_PLACE_TRANSITIONS
from dexvision.logging.level4_collection import (
    DEFAULT_PILOT_DATASET_DIR,
    WORKCELL_PILOT_TASK_ID,
    WorkcellPilotState,
    WorkcellPilotTask,
    build_level4_core_collection_plan,
    load_level4_collection_config,
)
from dexvision.logging.session_manifest import (
    RecordingSession,
    append_session_manifest,
    next_episode_directory,
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
    WorkspaceLimits,
    format_hand_base_status,
    hand_base_config_from_teleop_config,
)
from dexvision.sim.mujoco_env import MujocoEnv, MujocoError, MujocoState
from dexvision.sim.level4_expert import (
    DeterministicButtonPressConfig,
    DeterministicButtonPressExpert,
    DeterministicGraspLiftConfig,
    DeterministicGraspLiftExpert,
    DeterministicPickPlaceExpert,
    DeterministicPlaceConfig,
    DeterministicPushConfig,
    DeterministicPushExpert,
    SafeWaypointReachConfig,
    SafeWaypointReachExpert,
)
from dexvision.sim.workcell import (
    DEFAULT_WORKCELL_CONFIG,
    WorkcellError,
    load_workcell_config,
)
from dexvision.sim.workcell_rate_control import (
    WorkcellRateControlConfig,
    WorkcellRateController,
)
from dexvision.sim.tasks import (
    BUTTON_PRESS_TASK_ID,
    DEFAULT_TASK_BOARD_MODEL,
    PUSH_CUBE_TASK_ID,
    REACH_TOUCH_TARGET_TASK_ID,
    ButtonPressConfig,
    ButtonPressParameters,
    ButtonPressState,
    ButtonPressTask,
    PushCubeConfig,
    PushCubeParameters,
    PushCubeState,
    PushCubeTask,
    ReachTouchTargetConfig,
    ReachTouchTargetParameters,
    ReachTouchTargetState,
    ReachTouchTargetTask,
    TaskError,
)


DEFAULT_OUTPUT = Path("data/demos/free_space_gesture")
DEFAULT_CONFIG = run_level1_teleop.DEFAULT_CONFIG
DEFAULT_CAMERA_WIDTH = run_level1_teleop.DEFAULT_CAMERA_WIDTH
DEFAULT_CAMERA_HEIGHT = run_level1_teleop.DEFAULT_CAMERA_HEIGHT
DEFAULT_CONTROL_RATE_HZ = 30.0
DEFAULT_PRINT_INTERVAL = 30
DEFAULT_CAMERA_WINDOW_NAME = "DexVision Demo Recorder"
DEFAULT_LEVEL4_DATASET_CONFIG = Path("configs/level4_dataset.yaml")
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
REACH_TOUCH_TARGET_SITES = ReachTouchTargetConfig().target_sites
BUTTON_PRESS_IDS = ButtonPressConfig().button_ids
PUSH_CUBE_OBJECT_IDS = PushCubeConfig().object_ids
PUSH_CUBE_TARGET_ZONES = PushCubeConfig().target_zone_sites
PUSH_CUBE_APPROACH_SIDES = PushCubeConfig().approach_sides

TaskEpisodeState = (
    ReachTouchTargetState | ButtonPressState | PushCubeState | WorkcellPilotState
)
TaskEpisode = ReachTouchTargetTask | ButtonPressTask | PushCubeTask | WorkcellPilotTask


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


def _prepare_level4_workcell_recording(args: argparse.Namespace) -> None:
    """Resolve one append-only workcell attempt before opening devices."""

    if args.task != WORKCELL_PILOT_TASK_ID:
        return
    if args.synthetic:
        raise ValueError(
            "Level 4 workcell pilot episodes must be live; --synthetic is forbidden."
        )
    if args.overwrite:
        raise ValueError("Level 4 workcell pilot episodes are append-only.")
    if not args.workcell_dry_run and (
        args.session_id is None or not str(args.session_id).strip()
    ):
        raise ValueError("Level 4 workcell recording requires --session-id.")
    if not args.workcell_dry_run and (
        args.operator_id is None or not str(args.operator_id).strip()
    ):
        raise ValueError("Level 4 workcell recording requires --operator-id.")
    if not args.workcell_dry_run and args.session_split is None:
        raise ValueError("Level 4 workcell recording requires --session-split.")
    if args.skill_name not in {
        "reach_object",
        "pick_object",
        "pick_place_sequence",
        "push_object_to_target",
        "press_button",
    }:
        raise ValueError(
            "Level 4 workcell --skill must be reach_object, pick_object, "
            "pick_place_sequence, push_object_to_target, or press_button."
        )
    if args.source == "scripted" and args.skill_name not in {
        "reach_object",
        "pick_object",
        "pick_place_sequence",
        "press_button",
        "push_object_to_target",
    }:
        raise ValueError(
            "Level 4.3A-E scripted recording supports reach_object, pick_object, "
            "pick_place_sequence, press_button, and push_object_to_target."
        )
    if args.goal_condition_id is None or not str(args.goal_condition_id).strip():
        raise ValueError("Level 4 workcell recording requires --goal-condition-id.")
    config, _ = load_level4_collection_config(args.level4_dataset_config)
    cell = next(
        (
            item
            for item in config["coverage_cells"]
            if item.get("id") == args.goal_condition_id
        ),
        None,
    )
    if cell is None:
        raise ValueError(f"unknown Level 4 coverage cell: {args.goal_condition_id}")
    if not args.workcell_dry_run and cell.get("split_owner") != args.session_split:
        raise ValueError(
            f"coverage cell {args.goal_condition_id!r} is owned by split "
            f"{cell.get('split_owner')!r}, not {args.session_split!r}."
        )
    if (
        args.enforce_frozen_cell_owner
        and not args.workcell_dry_run
        and cell.get("required_source") != args.source
    ):
        raise ValueError(
            f"coverage cell {args.goal_condition_id!r} requires source "
            f"{cell.get('required_source')!r}, not {args.source!r}."
        )
    workcell_config = load_workcell_config(args.workcell_config)
    args.model = workcell_config.model_path
    args.level1_13_full = True
    if args.workcell_dry_run:
        return
    if args.output == DEFAULT_OUTPUT:
        args.output = next_episode_directory(
            args.level4_pilot_dataset_dir,
            recording_session_id=str(args.session_id),
        )
    process_start = datetime.now(UTC).isoformat()
    calibration_record = {
        "session_id": str(args.session_id),
        "process_start_timestamp": process_start,
        "camera_id": int(args.camera_id),
        "width": int(args.width),
        "height": int(args.height),
        "teleop_config": str(args.config),
        "workcell_config": str(args.workcell_config),
    }
    calibration_digest = hashlib.sha256(
        json.dumps(calibration_record, sort_keys=True).encode("utf-8")
    ).hexdigest()
    append_session_manifest(
        args.level4_pilot_dataset_dir / "session_manifest.json",
        RecordingSession(
            recording_session_id=str(args.session_id),
            operator_id=str(args.operator_id),
            split=str(args.session_split),
            process_start_timestamp=process_start,
            reset_seed=int(args.task_seed),
            calibration_record_digest=f"sha256:{calibration_digest}",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record DexVision teleoperation as a Level 2 episode or a session-aware "
            "Level 4 workcell pilot episode."
        )
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task id to store in metadata, e.g. free_space_gesture.",
    )
    parser.add_argument(
        "--skill-name",
        "--skill",
        dest="skill_name",
        default=None,
        help="Skill name to store in metadata. Defaults to --task.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Human-readable task name. Defaults to a title-cased --task.",
    )
    parser.add_argument(
        "--target-site",
        choices=REACH_TOUCH_TARGET_SITES,
        default=None,
        help=(
            "Configured target site for reach_touch_target. If omitted, --task-seed "
            "selects one deterministically."
        ),
    )
    parser.add_argument(
        "--button-id",
        choices=BUTTON_PRESS_IDS,
        default=None,
        help=(
            "Configured button for button_press. If omitted, --task-seed selects "
            "one deterministically."
        ),
    )
    parser.add_argument(
        "--target-press-depth",
        type=float,
        default=None,
        help=(
            "Button joint displacement in metres required for button_press success. "
            "Defaults to the task's configured depth."
        ),
    )
    parser.add_argument(
        "--approach-pose",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional button_press world-frame approach position in metres.",
    )
    parser.add_argument(
        "--object-id",
        choices=PUSH_CUBE_OBJECT_IDS,
        default=None,
        help="Movable object id for push_cube_to_target.",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-zone-id",
        choices=PUSH_CUBE_TARGET_ZONES,
        default=None,
        help=(
            "Configured target zone for push_cube_to_target. If omitted with "
            "--target-pose, --task-seed selects one deterministically."
        ),
    )
    target_group.add_argument(
        "--target-pose",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Custom push-cube target centre in MuJoCo world-frame metres.",
    )
    parser.add_argument(
        "--approach-side",
        choices=PUSH_CUBE_APPROACH_SIDES,
        default=None,
        help="Optional push_cube_to_target approach side.",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=0,
        help="Deterministic task reset seed. Defaults to 0.",
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
        "--session-id",
        default=None,
        help="Genuine Level 4 recording session id; enables the Level 4 episode schema.",
    )
    parser.add_argument(
        "--operator-id",
        default=None,
        help="Stable pseudonymous operator id required with --session-id.",
    )
    parser.add_argument(
        "--goal-condition-id",
        default=None,
        help="Frozen Level 4 coverage-cell id required with --session-id.",
    )
    parser.add_argument(
        "--source",
        choices=(
            "teleoperation",
            "scripted",
            "policy_rollout",
            "corrective_intervention",
        ),
        default="teleoperation",
        help="Level 4 request provenance. Defaults to teleoperation.",
    )
    parser.add_argument(
        "--level4-dataset-config",
        type=Path,
        default=DEFAULT_LEVEL4_DATASET_CONFIG,
        help="Frozen Level 4 dataset/schema config used when --session-id is set.",
    )
    parser.add_argument(
        "--level4-pilot-dataset-dir",
        "--level4-dataset-dir",
        dest="level4_pilot_dataset_dir",
        type=Path,
        default=DEFAULT_PILOT_DATASET_DIR,
        help="Root for append-only Level 4 sessions and episodes.",
    )
    parser.add_argument(
        "--print-level4-core-plan",
        action="store_true",
        help="Print the frozen Level 4.4 minimum-coverage recording plan and exit.",
    )
    parser.add_argument(
        "--enforce-frozen-cell-owner",
        action="store_true",
        help="Reject a Level 4 recording whose source differs from its frozen cell.",
    )
    parser.add_argument(
        "--workcell-config",
        type=Path,
        default=DEFAULT_WORKCELL_CONFIG,
        help="Runtime Level 4 workcell configuration.",
    )
    parser.add_argument(
        "--workcell-dry-run",
        action="store_true",
        help=(
            "Run an interactive Level 4 workcell control trial without keeping an "
            "episode or changing the session manifest."
        ),
    )
    parser.add_argument(
        "--session-split",
        choices=("train", "validation", "test"),
        default=None,
        help="Whole-session split required for a new Level 4 workcell pilot session.",
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

    _prepare_level4_workcell_recording(args)
    _apply_recording_presets(args)
    _validate_recording_args(args)
    raw_config = load_curl_retargeter_config(args.config)
    retargeter = CurlRetargeter.from_mapping(raw_config)
    target_names = run_level1_teleop.robot_target_names(retargeter)
    action_schema = build_level1_action_schema(target_names)
    episode_id = args.episode_id or _default_episode_id(args.task)
    model_path = _resolve_recording_model_path(
        args=args,
        raw_config=raw_config,
    )

    print(
        "DexVision Level 4 workcell pilot recorder"
        if args.task == WORKCELL_PILOT_TASK_ID
        else "DexVision Level 2 demo recorder"
    )
    print(f"Task: {args.task}")
    print(f"Skill: {args.skill_name or args.task}")
    if args.gesture_label is not None:
        print(f"Gesture label: {args.gesture_label}")
    print(f"Episode: {episode_id}")
    if args.workcell_dry_run:
        print("Mode: interactive dry run; all temporary frames will be discarded")
    if args.session_id is not None:
        print(f"Session: {args.session_id} (operator: {args.operator_id})")
        print(f"Goal condition: {args.goal_condition_id}")
    print(f"Output: {args.output}")
    print(f"Retargeter: {args.retargeter} ({args.config})")
    print(f"Action schema: {action_schema.version}, dim={action_schema.action_dim}")
    print(f"Robot targets: {', '.join(target_names)}")
    print("Video recording: off")
    if args.level1_13_full:
        print("Recording preset: full Level 1.13 teleop")
    if args.task == "free_space_gesture":
        _print_free_space_recording_guide(args.gesture_label)
    elif args.task == REACH_TOUCH_TARGET_TASK_ID:
        selected_target = (
            args.target_site or f"deterministic sample from seed {args.task_seed}"
        )
        print(f"Reach-touch target: {selected_target}")
        print(
            "Pilot guide: calibrate with c, make one reach-touch attempt, then press q. "
            "You will be asked for the operator success/failure label unless "
            "--success or --failure was supplied."
        )
    elif args.task == BUTTON_PRESS_TASK_ID:
        selected_button = args.button_id or (
            f"deterministic sample from seed {args.task_seed}"
        )
        target_depth = (
            f"{args.target_press_depth:.4f} m"
            if args.target_press_depth is not None
            else "task default"
        )
        print(f"Button-press target: {selected_button}, depth={target_depth}")
        print(
            "Pilot guide: calibrate with c, press only the bright green button "
            "until the task reports success, then provide the operator label. "
            "Non-target buttons are dark gray."
        )
    elif args.task == PUSH_CUBE_TASK_ID:
        selected_target = args.target_zone_id or (
            args.target_pose
            if args.target_pose is not None
            else f"deterministic sample from seed {args.task_seed}"
        )
        print(f"Push-cube target: {selected_target}")
        print(
            "Pilot guide: the orange cube starts directly in front of the "
            "vertical palm. Face your real palm toward the camera with fingers "
            "up when pressing c, then move the palm toward the camera to push "
            "the cube "
            "into the green floor target, and press q when finished. There is "
            "no recording timeout. Real-hand up/down is intentionally ignored "
            "to keep the push planar."
        )

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

    if args.task == WORKCELL_PILOT_TASK_ID and args.source == "scripted":
        return _run_scripted_workcell_recording(
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


def _run_scripted_workcell_recording(
    *,
    args: argparse.Namespace,
    raw_config: Mapping[str, Any],
    model_path: Path,
    retargeter: CurlRetargeter,
    target_names: tuple[str, ...],
    action_schema: ActionSchema,
    episode_id: str,
) -> int:
    """Record a qualified Level 4 scripted expert without camera input."""

    if args.skill_name not in {
        "reach_object",
        "pick_object",
        "pick_place_sequence",
        "press_button",
        "push_object_to_target",
    }:
        raise DemoLoggerError(
            "Level 4.3A-E scripted recording supports reach_object, pick_object, "
            "pick_place_sequence, press_button, and push_object_to_target."
        )
    with ExitStack() as stack:
        task = stack.enter_context(
            WorkcellPilotTask(
                workcell_config=args.workcell_config,
                dataset_config=args.level4_dataset_config,
                skill_name=args.skill_name,
                goal_condition_id=args.goal_condition_id,
                seed=args.task_seed,
            )
        )
        neutral_targets = run_level1_teleop.build_full_hand_targets(
            retargeter, no_hand_features()
        )
        if args.skill_name == "reach_object":
            print("Scripted expert: safe-waypoint reach (no webcam)")
            expert_settings_key = "scripted_reach"
            expert_settings = _level4_scripted_expert_settings(
                args, task, expert_settings_key
            )
            synergy_margin = expert_settings.get("fixed_finger_synergy_margin")
            if isinstance(synergy_margin, Mapping) and synergy_margin:
                neutral_targets = _scripted_finger_synergy_targets(
                    retargeter,
                    neutral_targets,
                    synergy_margin,
                )
            expert_config = SafeWaypointReachConfig.from_mapping(
                expert_settings
            )
            expert = SafeWaypointReachExpert(
                finger_targets=neutral_targets,
                config=expert_config,
            )
            controller_name = "safe_waypoint_reach"
        elif args.skill_name == "pick_object":
            print("Scripted expert: object-relative family grasp and lift (no webcam)")
            expert_settings_key = "scripted_grasp"
            expert_settings = _level4_scripted_expert_settings(
                args, task, expert_settings_key
            )
            closed_targets = _scripted_closed_finger_targets(
                retargeter, neutral_targets
            )
            expert_config = DeterministicGraspLiftConfig.from_mapping(
                expert_settings
            )
            expert = DeterministicGraspLiftExpert(
                open_finger_targets=neutral_targets,
                closed_finger_targets=closed_targets,
                config=expert_config,
            )
            controller_name = "object_relative_family_grasp_lift"
        elif args.skill_name == "pick_place_sequence":
            print("Scripted expert: composed grasp, transport, place, and release")
            expert_settings_key = "scripted_place"
            expert_settings = _level4_scripted_expert_settings(
                args, task, expert_settings_key
            )
            closed_targets = _scripted_closed_finger_targets(
                retargeter, neutral_targets
            )
            grasp_config = DeterministicGraspLiftConfig.from_mapping(
                task.collection_config["pilot"]["scripted_grasp"]
            )
            expert_config = DeterministicPlaceConfig.from_mapping(
                expert_settings
            )
            expert = DeterministicPickPlaceExpert(
                open_finger_targets=neutral_targets,
                closed_finger_targets=closed_targets,
                grasp_config=grasp_config,
                place_config=expert_config,
            )
            controller_name = "composed_grasp_transport_place_release"
        elif args.skill_name == "press_button":
            print("Scripted expert: fixed-posture normal button press (no webcam)")
            expert_settings_key = "scripted_button"
            expert_settings = _level4_scripted_expert_settings(
                args, task, expert_settings_key
            )
            expert_config = DeterministicButtonPressConfig.from_mapping(
                expert_settings
            )
            expert = DeterministicButtonPressExpert(
                finger_targets=neutral_targets,
                config=expert_config,
            )
            controller_name = "fixed_posture_normal_press"
        else:
            print("Scripted expert: fixed-index task-axis push (no webcam)")
            expert_settings_key = "scripted_push"
            expert_settings = _level4_scripted_expert_settings(
                args, task, expert_settings_key
            )
            expert_config = DeterministicPushConfig.from_mapping(
                expert_settings
            )
            object_id = str(task.goal["object_id"])
            object_family = next(
                spec.family
                for spec in task.workcell.config.objects
                if spec.object_id == object_id
            )
            neutral_targets = _scripted_push_finger_targets(
                retargeter,
                neutral_targets,
                index_curl=float(
                    expert_config.family_parameters[object_family]["index_curl"]
                ),
            )
            expert = DeterministicPushExpert(
                finger_targets=neutral_targets,
                config=expert_config,
            )
            controller_name = "fixed_index_task_axis_push"
        args.sim_steps_per_frame = expert_config.sim_steps_per_action
        expert.reset(task, task.initial_world_state)
        validation = expert.validation
        if validation is None or not validation.valid:
            reason = validation.reason if validation is not None else "unknown"
            raise DemoLoggerError(
                f"scripted {args.skill_name} candidate failed copied-state "
                f"validation: {reason}"
            )
        print(
            "Copied-state validation: pass "
            f"({validation.checked_actions} actions, "
            f"max neighbor disturbance="
            f"{validation.maximum_non_target_disturbance_m:.6f}m)"
        )

        initial_state = task.env.get_state()
        names = mujoco_observation_order(task.env)
        observation_schema = build_level2_observation_schema(
            robot_qpos_dim=initial_state.qpos.size,
            robot_qvel_dim=initial_state.qvel.size,
            finger_target_dim=initial_state.ctrl.size,
            tracking_quality_dim=len(TRACKING_QUALITY_FIELDS),
            robot_qpos_names=names[0],
            robot_qvel_names=names[1],
            actuator_names=names[2],
            finger_joint_qpos_indices=names[3],
            finger_joint_qvel_indices=names[4],
            finger_joint_names=names[5],
            tracking_quality_names=TRACKING_QUALITY_FIELDS,
            object_state_dim=task.current_state.object_state.size,
            task_state_dim=task.current_state.as_task_state().size,
            target_state_dim=7,
            success_metric_dim=8,
        )
        effective_config = deepcopy(dict(raw_config))
        effective_config["scripted_expert"] = {
            "controller": controller_name,
            "planning_point": "grasp_site",
            "waypoints_m": [point.tolist() for point in expert.waypoints],
            "copied_state_validation": {
                "checked_actions": validation.checked_actions,
                "maximum_non_target_disturbance_m": (
                    validation.maximum_non_target_disturbance_m
                ),
            },
            **expert_settings,
        }
        if args.skill_name == "pick_place_sequence":
            effective_config["scripted_expert"]["grasp"] = dict(
                task.collection_config["pilot"]["scripted_grasp"]
            )
        logger = DemoLogger(
            args.output,
            action_schema=action_schema,
            observation_schema=observation_schema,
            overwrite=False,
        )
        logger.start_episode(
            _metadata(
                args=args,
                episode_id=episode_id,
                raw_config=effective_config,
                model_path=model_path,
                target_names=target_names,
                observation_schema=observation_schema,
                synthetic=False,
                workcell_pilot_task=task,
            )
        )

        viewer_handle = None
        if args.viewer:
            _ensure_mujoco_viewer_can_launch(args)
            try:
                from mujoco import viewer
            except ImportError as exc:  # pragma: no cover - optional GUI path.
                raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc
            viewer_handle = stack.enter_context(
                viewer.launch_passive(task.env.model, task.env.data)
            )
            _configure_workcell_pilot_viewer(viewer_handle, task)

        default_limit = (
            700
            if args.skill_name == "pick_place_sequence"
            else (
                500
                if args.skill_name in {"push_object_to_target", "pick_object"}
                else 300
            )
        )
        limit = args.max_frames if args.max_frames > 0 else default_limit
        terminal = task.current_state
        achieved_success = False
        expert_done = False
        no_features = no_hand_features()
        for frame_index in range(limit):
            requested, phase, done, reason = expert.step(terminal.world_state)
            action = requested.as_array()
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=requested.base_position,
                orientation_quat=requested.base_orientation_wxyz,
            )
            task.env.set_joint_targets(requested.finger_targets)
            _apply_scripted_object_orientation_hold(task, phase=phase)
            terminal = task.step(n_steps=expert_config.sim_steps_per_action)
            achieved_success = achieved_success or terminal.success
            state = task.env.get_state()
            timestamp = float(state.time)
            logger.append(
                DemoStepData(
                    features=feature_vector(no_features),
                    action=action,
                    robot_state=robot_state_vector(
                        state,
                        base_position=requested.base_position,
                        base_orientation=requested.base_orientation_wxyz,
                    ),
                    tracking_quality=np.zeros(
                        len(TRACKING_QUALITY_FIELDS), dtype=np.float64
                    ),
                    timestamp=timestamp,
                    object_state=terminal.as_object_state(),
                    task_state=terminal.as_task_state(),
                    requested_action=action,
                    commanded_action=action,
                    applied_action=action,
                    safety_mask=np.zeros(action.size, dtype=np.uint8),
                    safety_reason=("none",) * action.size,
                    request_source="script",
                    online_phase=phase,
                    audited_phase="",
                    intervention=False,
                    failure_reason=reason or "",
                    action_timestamp=timestamp,
                    task_timestamp=timestamp,
                    state_timestamp=timestamp,
                )
            )
            if viewer_handle is not None:
                viewer_handle.sync()
                if args.viewer_sleep > 0.0:
                    time.sleep(args.viewer_sleep)
                if _viewer_was_closed(viewer_handle):
                    break
            if frame_index == 0 or (frame_index + 1) % args.print_interval == 0:
                print(
                    f"scripted={frame_index + 1:03d} {terminal.status_text} "
                    f"reason={reason or 'none'}"
                )
            if done:
                expert_done = True
                break

        metric_success = (
            terminal.success
            if args.skill_name in {"push_object_to_target", "pick_place_sequence"}
            else achieved_success
        )
        recording_success = metric_success and expert_done
        episode = logger.close(success=recording_success)
        print(
            f"Saved scripted {args.skill_name} with "
            f"{episode.timestamps.shape[0]} frames: "
            f"{args.output}"
        )
        print(
            "Terminal: "
            f"success={recording_success} {terminal.status_text} "
            f"failure={terminal.failure_reason or 'none'}"
        )
        if not recording_success:
            raise DemoLoggerError(
                f"scripted {args.skill_name} did not satisfy the recomputed "
                "terminal metric and complete its scripted terminal phase."
            )
        return 0


def _apply_scripted_object_orientation_hold(
    task: WorkcellPilotTask, *, phase: str
) -> None:
    """Apply the declared rotation-only grasp hold before one simulator step."""

    if task.skill_name not in {"pick_object", "pick_place_sequence"} or phase not in {
        "lift",
        "stabilize",
        "transport",
        "place",
    }:
        return
    grasp = task.collection_config["pilot"]["scripted_grasp"]
    if (
        grasp.get("orientation_preservation_policy")
        != "shape_aware_hammer_grip_with_world_orientation_hold"
    ):
        return
    object_id = str(task.goal["object_id"])
    orientation = task.initial_world_state.require_entity(object_id).orientation_wxyz
    task.workcell.preserve_object_orientation(object_id, orientation)


def _level4_scripted_expert_settings(
    args: argparse.Namespace,
    task: WorkcellPilotTask,
    settings_key: str,
) -> dict[str, Any]:
    """Resolve checkpoint-local expert overrides without changing older pilots."""

    settings = dict(task.collection_config["pilot"][settings_key])
    if not args.enforce_frozen_cell_owner:
        return settings
    core = task.collection_config.get("level4_4_core_collection", {})
    overrides = core.get("scripted_expert_overrides", {})
    if isinstance(overrides, Mapping):
        raw = overrides.get(settings_key, {})
        if isinstance(raw, Mapping):
            settings.update(raw)
    return settings


def _scripted_push_finger_targets(
    retargeter: CurlRetargeter,
    open_targets: Mapping[str, float],
    *,
    index_curl: float,
) -> dict[str, float]:
    """Partially flex the index and fully flex other fingers out of the path."""

    if not 0.0 <= index_curl <= 1.0:
        raise DemoLoggerError("scripted push index curl must be in [0, 1].")
    targets = {str(name): float(value) for name, value in open_targets.items()}
    for finger in retargeter.config.fingers:
        curl = index_curl if finger.name == "index" else 1.0
        for target in finger.targets:
            targets[target.name] = target.map_control(curl)
    return targets


def _scripted_finger_synergy_targets(
    retargeter: CurlRetargeter,
    open_targets: Mapping[str, float],
    synergy_by_finger: Mapping[str, object],
) -> dict[str, float]:
    """Move selected fingers just inside their limits for safe transit."""

    targets = {str(name): float(value) for name, value in open_targets.items()}
    known_fingers = {finger.name for finger in retargeter.config.fingers}
    if not synergy_by_finger or not set(synergy_by_finger) <= known_fingers:
        raise DemoLoggerError(
            "scripted finger synergy margin must name configured fingers."
        )
    for finger in retargeter.config.fingers:
        if finger.name not in synergy_by_finger:
            continue
        control = float(synergy_by_finger[finger.name])
        if not 0.0 <= control <= 1.0:
            raise DemoLoggerError(
                "scripted finger synergy margins must be in [0, 1]."
            )
        for target in finger.targets:
            targets[target.name] = target.map_control(control)
    return targets


def _scripted_closed_finger_targets(
    retargeter: CurlRetargeter,
    open_targets: Mapping[str, float],
) -> dict[str, float]:
    """Return the deterministic configured full-flexion grasp endpoint."""

    targets = {str(name): float(value) for name, value in open_targets.items()}
    for finger in retargeter.config.fingers:
        for target in finger.targets:
            targets[target.name] = target.map_control(1.0)
    return targets


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
    synthetic_joint_names = tuple(f"synthetic_joint:{name}" for name in target_names)
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=len(target_names),
        robot_qvel_dim=len(target_names),
        finger_target_dim=len(target_names),
        tracking_quality_dim=len(TRACKING_QUALITY_FIELDS),
        robot_qpos_names=synthetic_joint_names,
        robot_qvel_names=synthetic_joint_names,
        actuator_names=target_names,
        finger_joint_qpos_indices=tuple(range(len(target_names))),
        finger_joint_qvel_indices=tuple(range(len(target_names))),
        finger_joint_names=synthetic_joint_names,
        tracking_quality_names=TRACKING_QUALITY_FIELDS,
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
            qpos=np.full(len(target_names), phase, dtype=np.float64),
            qvel=np.zeros(len(target_names), dtype=np.float64),
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
                tracking_quality=np.asarray(
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64
                ),
                timestamp=start_time + (frame_index / args.control_rate_hz),
                landmarks=np.zeros((21, 3), dtype=np.float64)
                if args.save_landmarks
                else None,
            )
        )

    episode = logger.close(success=args.success)
    print(
        f"Saved synthetic demo with {episode.timestamps.shape[0]} frames: {args.output}"
    )
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
    effective_raw_config: Mapping[str, Any] = raw_config
    print(f"Camera: id={args.camera_id}, width={args.width}, height={args.height}")
    print(
        f"Hand tracker model: {args.hand_landmarker_model or DEFAULT_HAND_LANDMARKER_MODEL}"
    )
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
        with ExitStack() as stack:
            camera = stack.enter_context(
                OpenCVCamera(
                    camera_id=args.camera_id, width=args.width, height=args.height
                )
            )
            tracker = stack.enter_context(
                HandTracker(
                    min_detection_confidence=args.min_detection_confidence,
                    min_tracking_confidence=args.min_tracking_confidence,
                    model_path=args.hand_landmarker_model,
                    assume_mirrored_input=args.assume_mirrored_input,
                )
            )
            reach_touch_task: ReachTouchTargetTask | None = None
            reach_touch_initial_state: ReachTouchTargetState | None = None
            button_press_task: ButtonPressTask | None = None
            button_press_initial_state: ButtonPressState | None = None
            push_cube_task: PushCubeTask | None = None
            push_cube_initial_state: PushCubeState | None = None
            workcell_pilot_task: WorkcellPilotTask | None = None
            workcell_pilot_initial_state: WorkcellPilotState | None = None
            if args.task == WORKCELL_PILOT_TASK_ID:
                workcell_pilot_task = stack.enter_context(
                    WorkcellPilotTask(
                        workcell_config=args.workcell_config,
                        dataset_config=args.level4_dataset_config,
                        skill_name=str(args.skill_name),
                        goal_condition_id=str(args.goal_condition_id),
                        seed=args.task_seed,
                    )
                )
                workcell_pilot_initial_state = workcell_pilot_task.current_state
                env = workcell_pilot_task.env
                _ensure_realtime_sim_steps(
                    args=args,
                    env=env,
                    label="Level 4 workcell",
                )
                print(
                    "Resolved Level 4 workcell pilot: "
                    f"skill={args.skill_name} goal={workcell_pilot_task.goal}"
                )
                print(
                    "Human neutral: face your palm toward the webcam with fingers up. "
                    "Do not imitate the robot's palm-down pose; calibration treats your "
                    "comfortable pose as a translation clutch origin."
                )
                if args.skill_name == "reach_object":
                    print(
                        "Reach guide: the selected object has a bright magenta outline; "
                        "the cyan cross is the robot-palm goal. Your hand is a velocity "
                        "joystick: move away from center to move, return to center to stop. "
                        "Small offsets are deliberately precise. The controller travels "
                        "above the clutter and only descends over the outlined target."
                    )
                elif args.skill_name == "pick_place_sequence":
                    print(
                        "Pick/place guide: the magenta cage marks the object to pick; "
                        "the cyan cross marks its destination."
                    )
                elif args.skill_name == "push_object_to_target":
                    print(
                        "Push guide: the magenta cage marks the object to push; the "
                        "cyan cross marks the destination zone."
                    )
                elif args.skill_name == "press_button":
                    print(
                        "Press guide: the magenta cage and cyan cross identify the "
                        "button to press."
                    )
                print(
                    "Press c to calibrate/start. Press q after success or a clear failure."
                )
                base_config = _workcell_pilot_base_config(
                    base_config,
                    task=workcell_pilot_task,
                )
                effective_raw_config = _teleop_config_with_effective_base(
                    raw_config,
                    base_config=base_config,
                    neutral_orientation=np.asarray(
                        workcell_pilot_task.workcell.config.scene[
                            "hand_neutral_orientation_wxyz"
                        ],
                        dtype=np.float64,
                    ),
                    task_override=WORKCELL_PILOT_TASK_ID,
                )
            elif args.task == REACH_TOUCH_TARGET_TASK_ID:
                reach_touch_task = stack.enter_context(ReachTouchTargetTask(model_path))
                reach_touch_initial_state = reach_touch_task.reset(
                    seed=args.task_seed,
                    parameters=ReachTouchTargetParameters(target_site=args.target_site),
                )
                env = reach_touch_task.env
                expected_targets = tuple(
                    reach_touch_task.spec.action_schema.representation_notes[
                        "finger_target_names"
                    ]
                )
                if target_names != expected_targets:
                    raise DemoLoggerError(
                        "retargeter targets do not match the reach-touch task-board "
                        f"actuators: retargeter={target_names}, task={expected_targets}."
                    )
                print(
                    "Resolved reach-touch target: "
                    f"{reach_touch_initial_state.target_source} at "
                    f"{reach_touch_initial_state.target_position.tolist()}"
                )
            elif args.task == BUTTON_PRESS_TASK_ID:
                button_press_task = stack.enter_context(ButtonPressTask(model_path))
                button_press_initial_state = button_press_task.reset(
                    seed=args.task_seed,
                    parameters=ButtonPressParameters(
                        button_id=args.button_id,
                        target_press_depth=args.target_press_depth,
                        approach_pose=(
                            tuple(args.approach_pose)
                            if args.approach_pose is not None
                            else None
                        ),
                    ),
                )
                env = button_press_task.env
                expected_targets = tuple(
                    button_press_task.spec.action_schema.representation_notes[
                        "finger_target_names"
                    ]
                )
                if target_names != expected_targets:
                    raise DemoLoggerError(
                        "retargeter targets do not match the button task-board "
                        f"actuators: retargeter={target_names}, task={expected_targets}."
                    )
                print(
                    "Resolved button target: "
                    f"{button_press_initial_state.button_id} at "
                    f"{button_press_initial_state.button_position.tolist()}, "
                    f"depth={button_press_initial_state.target_press_depth:.4f} m"
                )
                print("Visual target cue: press only the bright green button.")
            elif args.task == PUSH_CUBE_TASK_ID:
                push_cube_task = stack.enter_context(
                    PushCubeTask(model_path, enforce_timeout=False)
                )
                push_cube_initial_state = push_cube_task.reset(
                    seed=args.task_seed,
                    parameters=PushCubeParameters(
                        object_id=args.object_id,
                        target_pose=(
                            tuple(args.target_pose)
                            if args.target_pose is not None
                            else None
                        ),
                        target_zone_id=args.target_zone_id,
                        approach_side=args.approach_side,
                    ),
                )
                env = push_cube_task.env
                _ensure_realtime_sim_steps(
                    args=args,
                    env=env,
                    label="Push-cube",
                )
                base_config = _push_cube_base_config(
                    base_config,
                    initial_state=push_cube_initial_state,
                )
                effective_raw_config = _teleop_config_with_effective_base(
                    raw_config,
                    base_config=base_config,
                    neutral_orientation=(
                        push_cube_initial_state.initial_base_orientation
                    ),
                )
                expected_targets = tuple(
                    push_cube_task.spec.action_schema.representation_notes[
                        "finger_target_names"
                    ]
                )
                if target_names != expected_targets:
                    raise DemoLoggerError(
                        "retargeter targets do not match the push-cube task-board "
                        f"actuators: retargeter={target_names}, task={expected_targets}."
                    )
                print(
                    "Resolved push-cube task: "
                    f"object={push_cube_initial_state.object_id} "
                    f"start={push_cube_initial_state.initial_object_position.tolist()} "
                    f"target={push_cube_initial_state.target_source} at "
                    f"{push_cube_initial_state.target_position.tolist()} "
                    f"radius={push_cube_initial_state.target_radius:.3f}m"
                )
                print("Visual target cue: push the cube into the green target zone.")
                print(
                    "Push-cube hand neutral: "
                    f"base={push_cube_initial_state.initial_base_position.tolist()} "
                    "(camera-facing vertical palm, fingers up, behind the cube)."
                )
                print("Recording timeout: off; press q when the attempt is finished.")
            else:
                env = stack.enter_context(MujocoEnv(model_path))
                env.reset()

            active_task: TaskEpisode | None = (
                workcell_pilot_task
                or reach_touch_task
                or button_press_task
                or push_cube_task
            )
            initial_task_state: TaskEpisodeState | None = (
                workcell_pilot_initial_state
                or reach_touch_initial_state
                or button_press_initial_state
                or push_cube_initial_state
            )
            if (
                base_config.enabled
                and workcell_pilot_task is not None
                and workcell_pilot_task.skill_name == "reach_object"
            ):
                rate_config = _workcell_rate_control_config(
                    workcell_pilot_task,
                    control_rate_hz=args.control_rate_hz,
                )
                base_controller = WorkcellRateController(
                    env,
                    base_config,
                    rate_config,
                )
                effective_raw_config = _teleop_config_with_workcell_rate_control(
                    effective_raw_config,
                    rate_config=rate_config,
                )
            else:
                base_controller = (
                    HandBaseMocapController(env, base_config)
                    if base_config.enabled
                    else None
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
            (
                qpos_names,
                qvel_names,
                actuator_names,
                finger_qpos_indices,
                finger_qvel_indices,
                finger_joint_names,
            ) = mujoco_observation_order(env)
            observation_schema = build_level2_observation_schema(
                robot_qpos_dim=initial_state.qpos.size,
                robot_qvel_dim=initial_state.qvel.size,
                finger_target_dim=initial_state.ctrl.size,
                tracking_quality_dim=len(TRACKING_QUALITY_FIELDS),
                robot_qpos_names=qpos_names,
                robot_qvel_names=qvel_names,
                actuator_names=actuator_names,
                finger_joint_qpos_indices=finger_qpos_indices,
                finger_joint_qvel_indices=finger_qvel_indices,
                finger_joint_names=finger_joint_names,
                tracking_quality_names=TRACKING_QUALITY_FIELDS,
                object_state_dim=(
                    workcell_pilot_initial_state.object_state.size
                    if workcell_pilot_initial_state is not None
                    else (
                        3
                        if reach_touch_initial_state is not None
                        else (
                            4
                            if button_press_initial_state is not None
                            else 13
                            if push_cube_initial_state is not None
                            else None
                        )
                    )
                ),
                task_state_dim=(
                    initial_task_state.as_task_state().size
                    if initial_task_state is not None
                    else None
                ),
                target_state_dim=(
                    7
                    if workcell_pilot_initial_state is not None
                    else (
                        3
                        if reach_touch_initial_state is not None
                        else (
                            13
                            if button_press_initial_state is not None
                            else 7
                            if push_cube_initial_state is not None
                            else None
                        )
                    )
                ),
                success_metric_dim=(
                    8
                    if workcell_pilot_initial_state is not None
                    else (
                        8
                        if reach_touch_initial_state is not None
                        else (
                            5
                            if button_press_initial_state is not None
                            else 8
                            if push_cube_initial_state is not None
                            else None
                        )
                    )
                ),
            )
            logger_output = args.output
            if args.workcell_dry_run:
                temporary_root = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="dexvision_level4_dry_run_")
                )
                logger_output = Path(temporary_root) / "episode"
            logger = DemoLogger(
                logger_output,
                action_schema=action_schema,
                observation_schema=observation_schema,
                overwrite=args.overwrite,
            )
            logger.start_episode(
                _metadata(
                    args=args,
                    episode_id=episode_id,
                    raw_config=effective_raw_config,
                    model_path=model_path,
                    target_names=target_names,
                    observation_schema=observation_schema,
                    synthetic=False,
                    reach_touch_task=reach_touch_task,
                    reach_touch_initial_state=reach_touch_initial_state,
                    button_press_task=button_press_task,
                    button_press_initial_state=button_press_initial_state,
                    push_cube_task=push_cube_task,
                    push_cube_initial_state=push_cube_initial_state,
                    workcell_pilot_task=workcell_pilot_task,
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
                    task=active_task,
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
            if reach_touch_task is not None:
                final_task_state = reach_touch_task.get_state()
                print(
                    "Reach-touch terminal state: "
                    f"distance={final_task_state.distance_to_target:.4f}m "
                    f"palm_contact={final_task_state.palm_contact} "
                    f"dwell={final_task_state.dwell_steps} "
                    f"computed_success={final_task_state.success} "
                    f"failure={final_task_state.failure_reason or 'none'}"
                )
                if args.success is None and preview is not None:
                    preview.close()
            elif button_press_task is not None:
                final_task_state = button_press_task.get_state()
                print(
                    "Button-press terminal state: "
                    f"button={final_task_state.button_id} "
                    f"depth={final_task_state.press_depth:.4f}m "
                    f"target={final_task_state.target_press_depth:.4f}m "
                    f"dwell={final_task_state.dwell_steps} "
                    f"computed_success={final_task_state.success} "
                    f"failure={final_task_state.failure_reason or 'none'}"
                )
                if args.success is None and preview is not None:
                    preview.close()
            elif push_cube_task is not None:
                final_task_state = push_cube_task.get_state()
                print(
                    "Push-cube terminal state: "
                    f"object={final_task_state.object_id} "
                    f"target={final_task_state.target_source} "
                    f"distance={final_task_state.distance_to_target:.4f}m "
                    f"dwell={final_task_state.dwell_steps} "
                    f"computed_success={final_task_state.success} "
                    f"failure={final_task_state.failure_reason or 'none'}"
                )
                if args.success is None and preview is not None:
                    preview.close()
            elif workcell_pilot_task is not None:
                final_task_state = workcell_pilot_task.current_state
                print(
                    "Level 4 workcell terminal state: "
                    f"{final_task_state.status_text} "
                    f"failure={final_task_state.failure_reason or 'none'}"
                )
                if args.success is None and preview is not None:
                    preview.close()
            operator_success = (
                final_task_state.success
                if args.workcell_dry_run and final_task_state is not None
                else _resolve_operator_success_label(args)
            )
            episode = logger.close(success=operator_success)
            if args.workcell_dry_run:
                print(
                    "Dry run complete: "
                    f"{episode.timestamps.shape[0]} temporary frames discarded; "
                    "session manifest and pilot dataset were not changed."
                )
            else:
                print(
                    f"Saved demo with {episode.timestamps.shape[0]} frames: {args.output}"
                )
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
    task: TaskEpisode | None = None,
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
            task=task,
            viewer_handle=None,
        )

    _ensure_mujoco_viewer_can_launch(args)
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    try:
        with viewer.launch_passive(env.model, env.data) as viewer_handle:
            if isinstance(task, PushCubeTask):
                _configure_push_cube_viewer(viewer_handle)
            elif isinstance(task, WorkcellPilotTask):
                _configure_workcell_pilot_viewer(viewer_handle, task)
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
                task=task,
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
    task: TaskEpisode | None = None,
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
        if (
            recording_started
            and args.max_frames > 0
            and recorded_frame_index >= args.max_frames
        ):
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
        targets = run_level1_teleop.build_full_hand_targets(
            retargeter, smoothed_features
        )
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
        if not recording_started and _calibration_started_recording(
            commands=applied_base_commands,
            base_status=base_status,
            base_control_enabled=base_controller is not None,
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
        task_step_state: TaskEpisodeState | None = None
        if task is not None and recording_started:
            task_step_state = task.step(n_steps=args.sim_steps_per_frame)
            state = env.get_state()
        else:
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
            recorded_action = action_vector(
                base_position=base_position,
                base_orientation=base_orientation,
                targets=targets,
                target_names=target_names,
            )
            logger.append(
                DemoStepData(
                    features=feature_vector(smoothed_features),
                    action=recorded_action,
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
                    task_state=(
                        task_step_state.as_task_state()
                        if task_step_state is not None
                        else None
                    ),
                    object_state=(
                        task_step_state.as_object_state()
                        if isinstance(task_step_state, WorkcellPilotState)
                        else (
                            task_step_state.target_position
                            if isinstance(task_step_state, ReachTouchTargetState)
                            else (
                                task_step_state.as_object_state()
                                if isinstance(
                                    task_step_state,
                                    (ButtonPressState, PushCubeState),
                                )
                                else None
                            )
                        )
                    ),
                    requested_action=recorded_action,
                    commanded_action=recorded_action,
                    applied_action=recorded_action,
                    safety_mask=np.zeros(recorded_action.size, dtype=np.uint8),
                    safety_reason=("none",) * recorded_action.size,
                    request_source="operator",
                    online_phase=(
                        task_step_state.online_phase
                        if isinstance(task_step_state, WorkcellPilotState)
                        else None
                    ),
                    audited_phase="",
                    intervention=False,
                    failure_reason=(
                        task_step_state.failure_reason
                        if isinstance(task_step_state, WorkcellPilotState)
                        else ""
                    )
                    or "",
                    action_timestamp=camera_result.timestamp,
                    task_timestamp=camera_result.timestamp,
                    state_timestamp=camera_result.timestamp,
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
                f"{_format_task_status(task_step_state)} "
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
                task_state=task_step_state,
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
        if task_step_state is not None and _task_should_stop_recording(task_step_state):
            terminal_label = (
                "success" if task_step_state.success else task_step_state.failure_reason
            )
            print(f"{args.task} reached terminal state: {terminal_label}.")
            break

    return RecordingSummary(
        frames=recorded_frame_index,
        detected_frames=detected_frames,
        stopped_by_preview=stopped_by_preview,
    )


def _task_should_stop_recording(state: TaskEpisodeState) -> bool:
    """Stop on success or safety failure, but never on a task timeout."""

    return bool(
        state.success
        or (state.failure_reason is not None and state.failure_reason != "timeout")
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


def _format_task_status(state: TaskEpisodeState | None) -> str:
    if state is None:
        return ""
    if isinstance(state, WorkcellPilotState):
        return state.status_text
    if isinstance(state, ReachTouchTargetState):
        return (
            f"palm_contact={'yes' if state.palm_contact else 'no'} "
            f"distance={state.distance_to_target:.3f}m "
            f"dwell={state.dwell_steps}"
        )
    if isinstance(state, ButtonPressState):
        return (
            f"button={state.button_id} "
            "target_color=bright-green "
            f"press_depth={state.press_depth:.4f}m/"
            f"{state.target_press_depth:.4f}m "
            f"dwell={state.dwell_steps}"
        )
    return (
        f"cube={state.object_id} "
        f"target={state.target_source} "
        f"distance={state.distance_to_target:.3f}m/"
        f"{state.target_radius:.3f}m "
        f"dwell={state.dwell_steps}"
    )


def _print_free_space_recording_guide(gesture_label: str | None) -> None:
    print("Free-space gesture recording plan:")
    print(
        "  Start each clip from a neutral upright palm in frame, press c to calibrate/center, perform/hold 3-5 seconds, then press q."
    )
    if gesture_label is not None:
        print(f"  This clip label: {gesture_label}")
        print(f"  What to record: {FREE_SPACE_GESTURE_INSTRUCTIONS[gesture_label]}")
        return
    print(
        "  Suggested 10-demo set: open_palm x2, fist x2, pinch x2, wave x2, point x1, peace_sign x1."
    )
    print(
        "  Use --gesture-label open_palm|fist|point|pinch|peace_sign|wave for labeled clips."
    )


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
        task_state: TaskEpisodeState | None = None,
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
                    + (
                        " | " + _format_task_status(task_state)
                        if task_state is not None
                        else ""
                    )
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


def ordered_targets(
    targets: dict[str, float], target_names: tuple[str, ...]
) -> np.ndarray:
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
        raise DemoLoggerError(
            f"base_position must have shape [3], got {position.shape}."
        )
    if orientation.shape != (4,):
        raise DemoLoggerError(
            f"base_orientation must have shape [4], got {orientation.shape}."
        )
    return np.concatenate(
        (position, orientation, ordered_targets(targets, target_names))
    )


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


def mujoco_observation_order(
    env: MujocoEnv,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    """Return named MuJoCo qpos/qvel order and scalar hand-joint selections."""

    model = env.model
    mujoco_module = env._mujoco
    qpos_names: list[str] = []
    qvel_names: list[str] = []
    actuator_names = tuple(
        (
            mujoco_module.mj_id2name(
                model,
                mujoco_module.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            or f"actuator_{actuator_id}"
        )
        for actuator_id in range(model.nu)
    )
    finger_qpos_indices: list[int] = []
    finger_qvel_indices: list[int] = []
    finger_joint_names: list[str] = []

    free_type = int(mujoco_module.mjtJoint.mjJNT_FREE)
    ball_type = int(mujoco_module.mjtJoint.mjJNT_BALL)
    for joint_id in range(model.njnt):
        joint_name = (
            mujoco_module.mj_id2name(
                model,
                mujoco_module.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            or f"joint_{joint_id}"
        )
        qpos_start = int(model.jnt_qposadr[joint_id])
        qpos_stop = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < model.njnt
            else int(model.nq)
        )
        qvel_start = int(model.jnt_dofadr[joint_id])
        qvel_stop = (
            int(model.jnt_dofadr[joint_id + 1])
            if joint_id + 1 < model.njnt
            else int(model.nv)
        )
        joint_type = int(model.jnt_type[joint_id])

        if joint_type == free_type:
            qpos_suffixes = ("x", "y", "z", "qw", "qx", "qy", "qz")
            qvel_suffixes = ("vx", "vy", "vz", "wx", "wy", "wz")
        elif joint_type == ball_type:
            qpos_suffixes = ("qw", "qx", "qy", "qz")
            qvel_suffixes = ("wx", "wy", "wz")
        else:
            qpos_suffixes = ()
            qvel_suffixes = ()

        if qpos_suffixes:
            qpos_names.extend(f"{joint_name}/{suffix}" for suffix in qpos_suffixes)
        else:
            qpos_names.append(joint_name)
        if qvel_suffixes:
            qvel_names.extend(f"{joint_name}/{suffix}" for suffix in qvel_suffixes)
        else:
            qvel_names.append(joint_name)

        if qpos_stop - qpos_start == 1 and qvel_stop - qvel_start == 1:
            finger_qpos_indices.append(qpos_start)
            finger_qvel_indices.append(qvel_start)
            finger_joint_names.append(joint_name)

    if len(qpos_names) != model.nq or len(qvel_names) != model.nv:
        raise DemoLoggerError(
            "failed to reconstruct the complete named MuJoCo qpos/qvel order."
        )
    return (
        tuple(qpos_names),
        tuple(qvel_names),
        actuator_names,
        tuple(finger_qpos_indices),
        tuple(finger_qvel_indices),
        tuple(finger_joint_names),
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
        raise DemoLoggerError(
            f"image landmarks must have shape [21, 3], got {landmarks.shape}."
        )
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
    reach_touch_task: ReachTouchTargetTask | None = None,
    reach_touch_initial_state: ReachTouchTargetState | None = None,
    button_press_task: ButtonPressTask | None = None,
    button_press_initial_state: ButtonPressState | None = None,
    push_cube_task: PushCubeTask | None = None,
    push_cube_initial_state: PushCubeState | None = None,
    workcell_pilot_task: WorkcellPilotTask | None = None,
) -> dict[str, Any]:
    if workcell_pilot_task is not None:
        task_config = workcell_pilot_task.metadata_task_config()
    elif reach_touch_task is not None and reach_touch_initial_state is not None:
        task_config = _reach_touch_task_config(
            args=args,
            task=reach_touch_task,
            initial_state=reach_touch_initial_state,
        )
    elif button_press_task is not None and button_press_initial_state is not None:
        task_config = _button_press_task_config(
            args=args,
            task=button_press_task,
            initial_state=button_press_initial_state,
        )
    elif push_cube_task is not None and push_cube_initial_state is not None:
        task_config = _push_cube_task_config(
            args=args,
            task=push_cube_task,
            initial_state=push_cube_initial_state,
        )
    else:
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
            "sim_steps_per_frame": int(args.sim_steps_per_frame),
            "observation_fields": observation_schema.fields,
        },
    }
    if args.gesture_label is not None:
        metadata["gesture_label"] = args.gesture_label
    if args.session_id is not None:
        metadata.update(
            _level4_metadata(
                args=args,
                task_config=task_config,
                observation_schema=observation_schema,
                action_dim=7 + len(target_names),
            )
        )
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


def _level4_metadata(
    *,
    args: argparse.Namespace,
    task_config: Mapping[str, Any],
    observation_schema: ObservationSchema,
    action_dim: int,
) -> dict[str, Any]:
    """Build the frozen Level 4 provenance and causal phase snapshot."""

    try:
        payload = yaml.safe_load(args.level4_dataset_config.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DemoLoggerError(
            f"could not read Level 4 dataset config {args.level4_dataset_config}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise DemoLoggerError(
            f"could not parse Level 4 dataset config {args.level4_dataset_config}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DemoLoggerError("Level 4 dataset config root must be a mapping.")
    schema_versions = payload.get("schema_versions")
    phase_config = payload.get("phase_contract")
    machine_config = payload.get("online_phase_state_machine")
    action_config = payload.get("action_contract")
    quality_config = payload.get("quality_thresholds")
    if not all(
        isinstance(value, Mapping)
        for value in (
            schema_versions,
            phase_config,
            machine_config,
            action_config,
            quality_config,
        )
    ):
        raise DemoLoggerError(
            "Level 4 config must define schema_versions, phase_contract, "
            "online_phase_state_machine, and action_contract mappings."
        )
    coverage_cells = payload.get("coverage_cells")
    if not isinstance(coverage_cells, list):
        raise DemoLoggerError("Level 4 config coverage_cells must be a list.")
    if not any(
        isinstance(cell, Mapping) and cell.get("id") == args.goal_condition_id
        for cell in coverage_cells
    ):
        raise DemoLoggerError(
            f"unknown Level 4 goal condition id: {args.goal_condition_id}"
        )

    skill_name = args.skill_name or {
        REACH_TOUCH_TARGET_TASK_ID: "reach_object",
        BUTTON_PRESS_TASK_ID: "press_button",
        PUSH_CUBE_TASK_ID: "push_object_to_target",
    }.get(args.task, args.task)
    machines = machine_config.get("machines")
    if not isinstance(machines, Mapping):
        raise DemoLoggerError("Level 4 online phase machines must be a mapping.")
    if skill_name == "pick_place_sequence":
        initial_phase = "approach"
        transitions = [
            {"from": item.source, "to": item.target, "predicate": item.predicate}
            for item in DEFAULT_PICK_PLACE_TRANSITIONS
        ]
    else:
        machine = machines.get(skill_name)
        if not isinstance(machine, Mapping):
            raise DemoLoggerError(
                f"Level 4 has no causal phase machine for skill '{skill_name}'."
            )
        initial_phase = machine.get("initial_phase")
        transitions = machine.get("transitions")
        if not isinstance(initial_phase, str) or not isinstance(transitions, list):
            raise DemoLoggerError(
                f"invalid causal phase machine for skill '{skill_name}'."
            )

    named_layout = action_config.get("named_layout")
    grouped_masks = phase_config.get("action_relevance_masks")
    if not isinstance(named_layout, list) or not isinstance(grouped_masks, Mapping):
        raise DemoLoggerError(
            "Level 4 action layout and phase relevance masks are required."
        )
    ordered_layout = sorted(named_layout, key=lambda field: int(field["index"]))
    if len(ordered_layout) != action_dim:
        raise DemoLoggerError(
            f"Level 4 action layout width {len(ordered_layout)} does not match {action_dim}."
        )
    relevance_masks: dict[str, list[int]] = {}
    for phase, group_mask in grouped_masks.items():
        if not isinstance(group_mask, Mapping):
            raise DemoLoggerError(f"phase mask for '{phase}' must be a mapping.")
        relevance_masks[str(phase)] = [
            int(bool(group_mask.get(str(field["group"]), False)))
            for field in ordered_layout
        ]
    raw_reason_codes = action_config.get("safety_reason_codes")
    if not isinstance(raw_reason_codes, list):
        raise DemoLoggerError("Level 4 safety_reason_codes must be a list.")
    reason_codes = [
        str(item["code"])
        for item in raw_reason_codes
        if isinstance(item, Mapping) and item.get("code")
    ]
    typed_goal = task_config.get("parameters", task_config)
    configured_objects = (
        payload.get("workcell", {}).get("objects", {})
        if isinstance(payload.get("workcell"), Mapping)
        else {}
    )
    goal_object_id = (
        typed_goal.get("object_id") if isinstance(typed_goal, Mapping) else None
    )
    goal_entity_id = (
        typed_goal.get("entity_id") if isinstance(typed_goal, Mapping) else None
    )
    object_ids = [
        str(value)
        for value in (args.object_id, goal_object_id, goal_entity_id)
        if isinstance(value, str) and value in configured_objects
    ]
    object_ids = list(dict.fromkeys(object_ids))
    dataset_config = payload.get("dataset")
    config_version = (
        str(dataset_config.get("name"))
        if isinstance(dataset_config, Mapping) and dataset_config.get("name")
        else "level4-dataset-v1"
    )
    return {
        "skill_name": skill_name,
        "episode_schema_version": str(schema_versions["episode"]),
        "recording_session_id": args.session_id,
        "operator_id": args.operator_id,
        "source": args.source,
        "typed_goal": typed_goal,
        "object_instance_ids": object_ids,
        "goal_condition_id": args.goal_condition_id,
        "reset_state": task_config.get(
            "initial_state", {"random_seed": args.task_seed}
        ),
        "random_seed": int(args.task_seed),
        "camera_or_render_config": None,
        "code_version": "working-tree",
        "config_version": config_version,
        "schema_versions": {
            **dict(schema_versions),
            "observation": observation_schema.version,
        },
        "phase_contract": {
            "version": phase_config.get("version"),
            "vocabulary": list(phase_config["vocabulary"]),
            "transitions": transitions,
            "action_relevance_masks": relevance_masks,
        },
        "action_contract": {
            "version": action_config.get("version"),
            "layout_version": action_config.get("layout_version"),
            "named_layout": ordered_layout,
            "safety_reason_codes": reason_codes,
            "nominal_control_interval_s": action_config.get(
                "nominal_control_interval_s"
            ),
            "max_state_action_timestamp_skew_s": quality_config.get(
                "max_state_action_timestamp_skew_s"
            ),
        },
        "initial_online_phase": initial_phase,
    }


def _reach_touch_task_config(
    *,
    args: argparse.Namespace,
    task: ReachTouchTargetTask,
    initial_state: ReachTouchTargetState,
) -> dict[str, Any]:
    """Return reconstructable reach-touch reset, goal, and metric metadata."""

    return {
        "required_objects": task.spec.required_objects,
        "requires_task_state": True,
        "requires_success_metric_inputs": True,
        "required_observation_fields": (
            "task_state",
            "target_state",
            "success_metric_inputs",
        ),
        "task_state_fields": task.spec.state_fields,
        "success_metric_inputs": task.spec.success_metric_inputs,
        "success_condition": task.spec.success_condition,
        "failure_conditions": task.spec.failure_conditions,
        "max_episode_steps": task.spec.max_episode_steps,
        "reset_seed": int(args.task_seed),
        "requested_target_site": args.target_site,
        "resolved_target_source": initial_state.target_source,
        "target_index": int(initial_state.target_index),
        "target_marker_body": task.config.target_marker_body,
        "target_position": initial_state.target_position,
        "target_position_units": "metres",
        "target_position_frame": "MuJoCo world",
        "initial_base_position": initial_state.initial_base_position,
        "initial_base_orientation": initial_state.initial_base_orientation,
        "initial_robot_qpos": initial_state.initial_robot_qpos,
        "initial_robot_qvel": initial_state.initial_robot_qvel,
        "operator_label": "metadata.success; supplied after the single task attempt",
    }


def _button_press_task_config(
    *,
    args: argparse.Namespace,
    task: ButtonPressTask,
    initial_state: ButtonPressState,
) -> dict[str, Any]:
    """Return reconstructable button reset, goal, and metric metadata."""

    return {
        "required_objects": task.spec.required_objects,
        "requires_task_state": True,
        "requires_success_metric_inputs": True,
        "required_observation_fields": (
            "task_state",
            "target_state",
            "success_metric_inputs",
        ),
        "task_state_fields": task.spec.state_fields,
        "success_metric_inputs": task.spec.success_metric_inputs,
        "success_condition": task.spec.success_condition,
        "failure_conditions": task.spec.failure_conditions,
        "max_episode_steps": task.spec.max_episode_steps,
        "success_dwell_steps": task.config.success_dwell_steps,
        "reset_seed": int(args.task_seed),
        "requested_button_id": args.button_id,
        "resolved_button_id": initial_state.button_id,
        "button_index": int(initial_state.button_index),
        "button_position": initial_state.button_position,
        "button_position_units": "metres",
        "button_position_frame": "MuJoCo world",
        "target_press_depth": initial_state.target_press_depth,
        "target_press_depth_units": "metres",
        "target_pressed_state": initial_state.target_pressed_state,
        "target_visual_cue": "bright_green",
        "non_target_visual_cue": "dark_gray",
        "approach_pose": (
            initial_state.approach_pose if initial_state.approach_pose_present else None
        ),
        "approach_pose_frame": "MuJoCo world",
        "initial_button_depth": initial_state.initial_button_depth,
        "initial_base_position": initial_state.initial_base_position,
        "initial_base_orientation": initial_state.initial_base_orientation,
        "initial_robot_qpos": initial_state.initial_robot_qpos,
        "initial_robot_qvel": initial_state.initial_robot_qvel,
        "operator_label": "metadata.success; supplied after the single task attempt",
    }


def _push_cube_task_config(
    *,
    args: argparse.Namespace,
    task: PushCubeTask,
    initial_state: PushCubeState,
) -> dict[str, Any]:
    """Return reconstructable cube reset, goal, and metric metadata."""

    return {
        "required_objects": task.spec.required_objects,
        "requires_task_state": True,
        "requires_success_metric_inputs": True,
        "required_observation_fields": (
            "object_state",
            "task_state",
            "target_state",
            "success_metric_inputs",
        ),
        "task_state_fields": task.spec.state_fields,
        "success_metric_inputs": task.spec.success_metric_inputs,
        "success_condition": task.spec.success_condition,
        "failure_conditions": task.spec.failure_conditions,
        "max_episode_steps": task.spec.max_episode_steps,
        "recording_timeout_enabled": task.enforce_timeout,
        "success_dwell_steps": task.config.success_dwell_steps,
        "reset_seed": int(args.task_seed),
        "requested_object_id": args.object_id,
        "resolved_object_id": initial_state.object_id,
        "object_index": int(initial_state.object_index),
        "requested_target_zone_id": args.target_zone_id,
        "requested_target_pose": args.target_pose,
        "resolved_target_source": initial_state.target_source,
        "target_index": int(initial_state.target_index),
        "target_marker_body": task.config.target_marker_body,
        "base_free_joint": task.config.base_free_joint,
        "target_position": initial_state.target_position,
        "target_position_units": "metres",
        "target_position_frame": "MuJoCo world cube centre",
        "target_radius": initial_state.target_radius,
        "target_radius_units": "metres planar distance",
        "requested_approach_side": args.approach_side,
        "resolved_approach_side": initial_state.approach_side,
        "initial_object_position": initial_state.initial_object_position,
        "initial_object_orientation": initial_state.initial_object_orientation,
        "initial_object_linear_velocity": initial_state.initial_object_linear_velocity,
        "initial_object_angular_velocity": initial_state.initial_object_angular_velocity,
        "initial_base_position": initial_state.initial_base_position,
        "initial_base_orientation": initial_state.initial_base_orientation,
        "initial_robot_qpos": initial_state.initial_robot_qpos,
        "initial_robot_qvel": initial_state.initial_robot_qvel,
        "operator_label": "metadata.success; supplied after the single task attempt",
    }


def _push_cube_base_config(
    config: HandBaseControlConfig,
    *,
    initial_state: PushCubeState,
) -> HandBaseControlConfig:
    """Align camera-facing palm depth motion with the cube push direction."""

    neutral = np.asarray(initial_state.initial_base_position, dtype=np.float64)
    return replace(
        config,
        enable_base_orientation=False,
        base_fixed_z=float(neutral[2]),
        position_offset=np.asarray([neutral[0], neutral[1], 0.0], dtype=np.float64),
        base_position_scale_x=0.0,
        base_position_scale_y=0.0,
        base_smoothing_alpha=0.40,
        depth_scale=0.35,
        depth_smoothing_alpha=0.40,
        depth_deadband=0.01,
        depth_min=-0.22,
        depth_max=0.04,
        orientation_smoothing_alpha=0.50,
        orientation_deadband_deg=0.25,
        max_position_step=0.02,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-0.22, neutral[1], -0.24], dtype=np.float64),
            maximum=np.asarray([0.04, neutral[1], -0.24], dtype=np.float64),
        ),
    )


def _workcell_pilot_base_config(
    config: HandBaseControlConfig,
    *,
    task: WorkcellPilotTask,
) -> HandBaseControlConfig:
    """Center calibrated control on the workcell's collision-free palm pose."""

    neutral = np.asarray(
        task.workcell.config.scene["hand_neutral_position_m"], dtype=np.float64
    )
    workspace = task.workcell.config.requirements["workcell"]["safe_workspace"]
    return replace(
        config,
        enable_base_orientation=(
            False
            if task.skill_name == "reach_object"
            else config.enable_base_orientation
        ),
        base_fixed_z=float(neutral[2]),
        position_offset=np.asarray([neutral[0], neutral[1], 0.0], dtype=np.float64),
        base_position_scale_x=0.75,
        base_position_scale_y=0.70,
        base_smoothing_alpha=0.80,
        depth_scale=0.55,
        depth_smoothing_alpha=0.80,
        depth_deadband=0.02,
        depth_min=float(workspace["min"][0]),
        depth_max=float(workspace["max"][0]),
        max_position_step=0.06,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray(workspace["min"], dtype=np.float64),
            maximum=np.asarray(workspace["max"], dtype=np.float64),
        ),
    )


def _workcell_rate_control_config(
    task: WorkcellPilotTask,
    *,
    control_rate_hz: float,
) -> WorkcellRateControlConfig:
    """Build the single Level 4.3 reach-trial rate controller."""

    raw = task.collection_config["pilot"]["reach_rate_control"]
    return WorkcellRateControlConfig(
        goal_position=np.asarray(task.goal["approach_pose"][:3], dtype=np.float64),
        control_rate_hz=float(control_rate_hz),
        image_deadband=float(raw["image_deadband"]),
        image_full_scale=float(raw["image_full_scale"]),
        depth_deadband=float(raw["depth_deadband"]),
        depth_full_scale=float(raw["depth_full_scale"]),
        response_exponent=float(raw["response_exponent"]),
        max_velocity_m_s=np.asarray(raw["max_velocity_m_s"], dtype=np.float64),
        transit_height_m=float(raw["transit_height_m"]),
        descent_radius_m=float(raw["descent_radius_m"]),
    )


def _teleop_config_with_workcell_rate_control(
    raw_config: Mapping[str, Any],
    *,
    rate_config: WorkcellRateControlConfig,
) -> dict[str, Any]:
    """Record the rate law and virtual fixture used by a retained episode."""

    snapshot = deepcopy(dict(raw_config))
    snapshot["workcell_rate_control"] = {
        "mapping": "centered_velocity_joystick",
        "goal_position": rate_config.goal_position,
        "control_rate_hz": rate_config.control_rate_hz,
        "image_deadband": rate_config.image_deadband,
        "image_full_scale": rate_config.image_full_scale,
        "depth_deadband": rate_config.depth_deadband,
        "depth_full_scale": rate_config.depth_full_scale,
        "response_exponent": rate_config.response_exponent,
        "max_velocity_m_s": rate_config.max_velocity_m_s,
        "transit_height_m": rate_config.transit_height_m,
        "descent_radius_m": rate_config.descent_radius_m,
    }
    return snapshot


def _minimum_realtime_sim_steps(
    *,
    simulation_timestep: float,
    control_rate_hz: float,
) -> int:
    """Return enough MuJoCo steps to cover one nominal control period."""

    if not np.isfinite(simulation_timestep) or simulation_timestep <= 0.0:
        raise ValueError("simulation_timestep must be finite and positive.")
    if not np.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
        raise ValueError("control_rate_hz must be finite and positive.")
    return max(1, math.ceil((1.0 / control_rate_hz) / simulation_timestep))


def _ensure_realtime_sim_steps(
    *,
    args: argparse.Namespace,
    env: MujocoEnv,
    label: str,
) -> None:
    """Keep physics time aligned with the nominal camera/control interval."""

    minimum_sim_steps = _minimum_realtime_sim_steps(
        simulation_timestep=float(env.model.opt.timestep),
        control_rate_hz=float(args.control_rate_hz),
    )
    if args.sim_steps_per_frame >= minimum_sim_steps:
        return
    print(
        f"{label} real-time simulation override: "
        f"{args.sim_steps_per_frame} -> {minimum_sim_steps} "
        "MuJoCo steps per camera frame."
    )
    args.sim_steps_per_frame = minimum_sim_steps


def _teleop_config_with_effective_base(
    raw_config: Mapping[str, Any],
    *,
    base_config: HandBaseControlConfig,
    neutral_orientation: np.ndarray,
    task_override: str = PUSH_CUBE_TASK_ID,
) -> dict[str, Any]:
    """Snapshot task-specific base overrides used by recording and quality checks."""

    snapshot = deepcopy(dict(raw_config))
    raw_base = snapshot.get("base_control")
    base_payload = dict(raw_base) if isinstance(raw_base, Mapping) else {}
    base_payload.update(
        {
            "enable_base_control": base_config.enabled,
            "enable_base_orientation": base_config.enable_base_orientation,
            "enable_depth_control": base_config.enable_depth_control,
            "base_fixed_z": base_config.base_fixed_z,
            "position_offset": base_config.position_offset,
            "base_position_scale_x": base_config.base_position_scale_x,
            "base_position_scale_y": base_config.base_position_scale_y,
            "base_smoothing_alpha": base_config.base_smoothing_alpha,
            "depth_scale": base_config.depth_scale,
            "depth_smoothing_alpha": base_config.depth_smoothing_alpha,
            "depth_deadband": base_config.depth_deadband,
            "depth_min": base_config.depth_min,
            "depth_max": base_config.depth_max,
            "orientation_smoothing_alpha": (base_config.orientation_smoothing_alpha),
            "orientation_deadband_deg": base_config.orientation_deadband_deg,
            "max_position_step": base_config.max_position_step,
            "workspace_limits": {
                "min": base_config.workspace_limits.minimum,
                "max": base_config.workspace_limits.maximum,
            },
            "neutral_mocap_orientation": neutral_orientation,
            "task_override": task_override,
        }
    )
    snapshot["base_control"] = base_payload
    return snapshot


def _resolve_recording_model_path(
    *,
    args: argparse.Namespace,
    raw_config: Mapping[str, Any],
) -> Path:
    """Select the task-board scene for implemented task pilots."""

    if (
        args.task
        in {
            REACH_TOUCH_TARGET_TASK_ID,
            BUTTON_PRESS_TASK_ID,
            PUSH_CUBE_TASK_ID,
        }
        and args.model is None
    ):
        return DEFAULT_TASK_BOARD_MODEL
    return run_level1_teleop.resolve_mujoco_model_path(
        raw_config,
        config_path=args.config,
        override=args.model,
    )


def _resolve_operator_success_label(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] = input,
) -> bool | None:
    """Return the saved operator label, prompting for live task attempts."""

    if (
        args.task
        not in {
            REACH_TOUCH_TARGET_TASK_ID,
            BUTTON_PRESS_TASK_ID,
            PUSH_CUBE_TASK_ID,
            WORKCELL_PILOT_TASK_ID,
        }
        or args.success is not None
    ):
        return args.success
    task_label = args.task.replace("_", "-")
    try:
        response = input_fn(
            f"Operator label — did this {task_label} attempt succeed? [y/n]: "
        )
    except EOFError as exc:
        raise DemoLoggerError(
            f"{args.task} requires an operator label. Rerun with "
            "--success or --failure when interactive input is unavailable."
        ) from exc
    normalized = response.strip().lower()
    if normalized in {"y", "yes", "success", "s"}:
        return True
    if normalized in {"n", "no", "failure", "fail", "f"}:
        return False
    raise DemoLoggerError(
        "operator label must be y/yes/success or n/no/failure. "
        "The episode was not saved; rerun the single attempt."
    )


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
        base_config = replace(
            base_config, enable_depth_control=args.enable_depth_control
        )
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
    if args.session_id is not None:
        if not str(args.session_id).strip():
            raise ValueError("--session-id must be non-empty.")
        if args.operator_id is None or not str(args.operator_id).strip():
            raise ValueError("--operator-id is required with --session-id.")
        if args.goal_condition_id is None or not str(args.goal_condition_id).strip():
            raise ValueError("--goal-condition-id is required with --session-id.")
        if args.overwrite:
            raise ValueError(
                "--overwrite is forbidden for append-only Level 4 episodes."
            )
        if args.source == "corrective_intervention":
            raise ValueError(
                "corrective_intervention recording is deferred to Level 4.6; "
                "record_demo cannot create it in Level 4.2."
            )
    elif (
        not args.workcell_dry_run
        and (args.operator_id is not None or args.goal_condition_id is not None)
    ):
        raise ValueError("--operator-id and --goal-condition-id require --session-id.")
    if args.workcell_dry_run and args.task != WORKCELL_PILOT_TASK_ID:
        raise ValueError("--workcell-dry-run requires --task level4_workcell.")
    if args.gesture_label is not None and args.task != "free_space_gesture":
        raise ValueError(
            "--gesture-label is only supported with --task free_space_gesture."
        )
    if args.target_site is not None and args.task != REACH_TOUCH_TARGET_TASK_ID:
        raise ValueError(
            "--target-site is only supported with --task reach_touch_target."
        )
    if args.button_id is not None and args.task != BUTTON_PRESS_TASK_ID:
        raise ValueError("--button-id is only supported with --task button_press.")
    if args.target_press_depth is not None and args.task != BUTTON_PRESS_TASK_ID:
        raise ValueError(
            "--target-press-depth is only supported with --task button_press."
        )
    if args.approach_pose is not None and args.task != BUTTON_PRESS_TASK_ID:
        raise ValueError("--approach-pose is only supported with --task button_press.")
    if args.object_id is not None and args.task != PUSH_CUBE_TASK_ID:
        raise ValueError(
            "--object-id is only supported with --task push_cube_to_target."
        )
    if args.target_zone_id is not None and args.task != PUSH_CUBE_TASK_ID:
        raise ValueError(
            "--target-zone-id is only supported with --task push_cube_to_target."
        )
    if args.target_pose is not None and args.task != PUSH_CUBE_TASK_ID:
        raise ValueError(
            "--target-pose is only supported with --task push_cube_to_target."
        )
    if args.approach_side is not None and args.task != PUSH_CUBE_TASK_ID:
        raise ValueError(
            "--approach-side is only supported with --task push_cube_to_target."
        )
    if args.target_press_depth is not None and args.target_press_depth <= 0.0:
        raise ValueError("--target-press-depth must be positive.")
    if (
        args.task
        in {
            REACH_TOUCH_TARGET_TASK_ID,
            BUTTON_PRESS_TASK_ID,
            PUSH_CUBE_TASK_ID,
        }
        and args.synthetic
    ):
        raise ValueError(
            f"--synthetic is not supported for {args.task} because pilot "
            "episodes must contain live task state and one operator-reviewed attempt."
        )
    if args.start_on_calibration is None:
        args.start_on_calibration = False
    if args.start_on_calibration and not args.show_camera_window:
        raise ValueError(
            "--start-on-calibration requires --show-camera-window so c can be pressed."
        )
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
    if args.task == WORKCELL_PILOT_TASK_ID and args.source == "scripted":
        args.show_camera_window = False
        args.enable_base_control = False
        args.enable_base_orientation = False
        args.auto_calibrate_base = False
        args.start_on_calibration = False
        args.enable_depth_control = False
        args.require_hand_detected = False
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
    return sum(
        1
        for step in steps
        if step.tracking_quality.size and step.tracking_quality[0] >= 0.5
    )


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
        command.extend(
            ["--min-hand-detected-frames", str(args.min_hand_detected_frames)]
        )
    if args.max_frames:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.gesture_label is not None:
        command.extend(["--gesture-label", args.gesture_label])
    if args.target_site is not None:
        command.extend(["--target-site", args.target_site])
    if args.button_id is not None:
        command.extend(["--button-id", args.button_id])
    if args.target_press_depth is not None:
        command.extend(["--target-press-depth", str(args.target_press_depth)])
    if args.approach_pose is not None:
        command.extend(
            ["--approach-pose", *(str(value) for value in args.approach_pose)]
        )
    if args.task_seed != 0:
        command.extend(["--task-seed", str(args.task_seed)])
    if args.success is True:
        command.append("--success")
    elif args.success is False:
        command.append("--failure")
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


def _configure_push_cube_viewer(viewer_handle: object) -> None:
    """Frame the upright palm, cube, and target from a clear three-quarter view."""

    camera = getattr(viewer_handle, "cam", None)
    if camera is None:
        return
    camera.lookat[:] = np.asarray([-0.02, 0.01, 0.03], dtype=np.float64)
    camera.distance = 0.62
    camera.azimuth = 35.0
    camera.elevation = -25.0


def _configure_workcell_pilot_viewer(
    viewer_handle: object,
    task: WorkcellPilotTask,
) -> None:
    """Use the verified Level 4.1 operator-facing workcell overview."""

    camera = getattr(viewer_handle, "cam", None)
    if camera is None:
        return
    viewer = task.workcell.config.scene["viewer"]
    camera.lookat[:] = np.asarray(viewer["lookat_m"], dtype=np.float64)
    camera.distance = float(viewer["distance_m"])
    camera.azimuth = float(viewer["azimuth_deg"])
    camera.elevation = float(viewer["elevation_deg"])


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


def _print_level4_core_plan(args: argparse.Namespace) -> int:
    if args.task != WORKCELL_PILOT_TASK_ID:
        raise ValueError("--print-level4-core-plan requires --task level4_workcell.")
    plan = build_level4_core_collection_plan(args.level4_dataset_config)
    print("sequence\tsession_slot\tsplit\tsource\tskill\tcoverage_cell\tseed")
    for item in plan:
        print(
            f"{item.sequence}\t{item.session_slot}\t{item.split}\t{item.source}\t"
            f"{item.skill_name}\t{item.coverage_cell_id}\t{item.seed}"
        )
    print(f"Total Level 4.4 accepted episodes required: {len(plan)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.print_level4_core_plan:
            return _print_level4_core_plan(args)
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
        TaskError,
        ValueError,
        WorkcellError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
