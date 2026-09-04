from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps.validate_level4_episode import main as validate_main
from dexvision.apps import record_demo
from dexvision.logging.dataset_schema import DemoSchemaError, validate_demo
from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoLoggerError,
    DemoStepData,
    build_level1_action_schema,
    build_level2_observation_schema,
    load_logged_demo,
)
from dexvision.logging.session_manifest import (
    RecordingSession,
    SessionManifestError,
    append_session_manifest,
    load_session_manifest,
    next_episode_directory,
)


PHASES = (
    "approach",
    "acquire",
    "lift",
    "stabilize",
    "transport",
    "place",
    "release",
    "settle",
    "retract",
)


def _schemas():
    action_schema = build_level1_action_schema(("a0", "a1"))
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=4,
        robot_qvel_dim=4,
        finger_target_dim=2,
        tracking_quality_dim=6,
    )
    return action_schema, observation_schema


def _action(index: int, *, clipped: bool = False) -> tuple[np.ndarray, np.ndarray]:
    commanded = np.zeros(9, dtype=np.float64)
    commanded[:3] = [0.01 * index, 0.0, 0.14]
    commanded[3] = 1.0
    commanded[7:] = [0.1, 0.2]
    applied = commanded.copy()
    if clipped:
        commanded[0] = 0.5
        applied[0] = 0.22
    return commanded, applied


def _metadata(initial_action: np.ndarray, observation_version: str) -> dict:
    transitions = [
        {"from": source, "to": target}
        for source, target in zip(PHASES, PHASES[1:])
    ]
    return {
        "skill_name": "pick_place_sequence",
        "task_name": "Pick Place Sequence",
        "task_id": "pick_place_sequence",
        "episode_id": "level4_000001",
        "robot_model": "assets/mujoco/workcell_scene.xml",
        "retargeter_config": "configs/level1_teleop.yaml",
        "control_rate_hz": 30.0,
        "teleop_config": {},
        "task_config": {
            "required_objects": (),
            "requires_task_state": False,
            "requires_success_metric_inputs": False,
            "required_observation_fields": (),
        },
        "episode_schema_version": "level4/episode-v1",
        "recording_session_id": "session_a_20260904",
        "operator_id": "operator_local_01",
        "source": "teleoperation",
        "typed_goal": {"object_id": "block_small", "target_id": "setup_slot_a"},
        "object_instance_ids": ["block_small"],
        "goal_condition_id": "pp_block_small_setup_slot_a",
        "reset_state": {"object_id": "block_small", "position_m": [-0.08, 0.0, 0.02]},
        "random_seed": 7,
        "camera_or_render_config": None,
        "code_version": "test-tree",
        "config_version": "level4/dataset-plan-v1",
        "schema_versions": {
            "episode": "level4/episode-v1",
            "observation": observation_version,
            "action": "level4/request-command-apply-v1",
            "world_state": "level4/world-state-v1",
            "phase": "level4/causal-phase-v1",
            "safety": "level4/action-safety-v1",
        },
        "phase_contract": {
            "version": "level4/causal-phase-v1",
            "vocabulary": list(PHASES),
            "transitions": transitions,
            "action_relevance_masks": {phase: [1] * 9 for phase in PHASES},
        },
        "action_contract": {
            "version": "level4/request-command-apply-v1",
            "safety_reason_codes": ["none", "workspace_clip"],
            "max_state_action_timestamp_skew_s": 0.005,
        },
        "initial_commanded_action": initial_action.tolist(),
        "initial_applied_action": initial_action.tolist(),
    }


def _write_level4_episode(path: Path, *, with_rgb: bool = False) -> None:
    action_schema, observation_schema = _schemas()
    initial_commanded, initial_applied = _action(0)
    logger = DemoLogger(
        path,
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    metadata = _metadata(initial_applied, observation_schema.version)
    if with_rgb:
        metadata["camera_or_render_config"] = {
            "version": "level4/fixed-camera-rgb-v1",
            "calibration_version": "workcell-fixed-v1",
            "max_rgb_state_timestamp_skew_s": 0.017,
        }
    logger.start_episode(metadata)
    prior_commanded = initial_commanded
    prior_applied = initial_applied
    for index, phase in enumerate(PHASES):
        commanded, applied = _action(index, clipped=index == 2)
        safety_mask = np.zeros(action_schema.action_dim, dtype=np.uint8)
        safety_reasons = ["none"] * action_schema.action_dim
        if index == 2:
            safety_mask[0] = 1
            safety_reasons[0] = "workspace_clip"
        timestamp = index / 30.0
        logger.append(
            DemoStepData(
                features=np.zeros(14, dtype=np.float64),
                action=applied,
                robot_state=np.zeros(17, dtype=np.float64),
                tracking_quality=np.asarray(
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64
                ),
                timestamp=timestamp,
                requested_action=commanded,
                commanded_action=commanded,
                applied_action=applied,
                prior_commanded_action=prior_commanded,
                prior_applied_action=prior_applied,
                safety_mask=safety_mask,
                safety_reason=safety_reasons,
                request_source="operator",
                online_phase=phase,
                audited_phase="approach" if index == 1 else phase,
                phase_relevance_mask=np.ones(action_schema.action_dim, dtype=np.uint8),
                intervention=False,
                failure_reason="",
                action_timestamp=timestamp,
                task_timestamp=timestamp,
                state_timestamp=timestamp,
                rgb_frame=(
                    np.full((4, 5, 3), index, dtype=np.uint8) if with_rgb else None
                ),
                rgb_timestamp=timestamp if with_rgb else None,
            )
        )
        prior_commanded = commanded
        prior_applied = applied
    logger.close(success=True)


def test_level4_episode_round_trip_and_cli_validation(tmp_path: Path, capsys) -> None:
    episode_dir = tmp_path / "episode"
    _write_level4_episode(episode_dir)

    result = validate_main(["--episode", str(episode_dir)])

    assert result == 0
    output = capsys.readouterr().out
    assert "validation: PASS" in output
    assert "Audited phase disagreement: 1/9" in output
    assert "Derived segments: reach_object, pick_object, place_held_object" in output
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["recording_session_id"] == "session_a_20260904"
    assert metadata["phase_intervals"][0] == {
        "end_frame": 1,
        "phase": "approach",
        "start_frame": 0,
    }
    assert np.array_equal(
        np.load(episode_dir / "actions.npy"),
        np.load(episode_dir / "applied_actions.npy"),
    )


def test_record_demo_accepts_session_provenance_and_snapshots_frozen_contract() -> None:
    parser = record_demo.build_parser()
    args = parser.parse_args(
        [
            "--task",
            "free_space_gesture",
            "--skill",
            "reach_object",
            "--session-id",
            "session_a_20260904",
            "--operator-id",
            "operator_local_01",
            "--goal-condition-id",
            "reach_block_small_interior",
            "--level4-dataset-config",
            str(Path(__file__).resolve().parents[1] / "configs" / "level4_dataset.yaml"),
        ]
    )
    record_demo._validate_recording_args(args)
    _, observation_schema = _schemas()

    metadata = record_demo._level4_metadata(
        args=args,
        task_config={"parameters": {"entity_id": "block_small"}},
        observation_schema=observation_schema,
        action_dim=27,
    )

    assert metadata["recording_session_id"] == "session_a_20260904"
    assert metadata["skill_name"] == "reach_object"
    assert metadata["schema_versions"]["action"] == "level4/request-command-apply-v1"
    assert len(metadata["phase_contract"]["action_relevance_masks"]["approach"]) == 27


def test_record_demo_requires_complete_session_provenance() -> None:
    args = record_demo.build_parser().parse_args(
        ["--task", "free_space_gesture", "--session-id", "session_a"]
    )
    with pytest.raises(ValueError, match="--operator-id is required"):
        record_demo._validate_recording_args(args)


def test_level4_missing_session_id_fails_clearly(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    _write_level4_episode(episode_dir)
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["recording_session_id"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    episode = load_logged_demo(episode_dir)
    action_schema, observation_schema = _schemas()

    with pytest.raises(DemoSchemaError, match="recording_session_id"):
        validate_demo(
            episode,
            action_schema=action_schema,
            observation_schema=observation_schema,
        )


def test_safety_mask_reproduces_commanded_to_applied_handling(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    _write_level4_episode(episode_dir)
    masks_path = episode_dir / "safety_masks.npy"
    masks = np.load(masks_path)
    masks[2, 0] = 0
    np.save(masks_path, masks)
    episode = load_logged_demo(episode_dir)
    action_schema, observation_schema = _schemas()

    with pytest.raises(DemoSchemaError, match="identified by safety_masks"):
        validate_demo(
            episode,
            action_schema=action_schema,
            observation_schema=observation_schema,
        )


def test_optional_rgb_is_aligned_and_requires_calibration(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    _write_level4_episode(episode_dir, with_rgb=True)

    assert np.load(episode_dir / "rgb_frames.npy").shape == (9, 4, 5, 3)
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["camera_or_render_config"] = {"version": "rgb-v1"}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    episode = load_logged_demo(episode_dir)
    action_schema, observation_schema = _schemas()
    with pytest.raises(DemoSchemaError, match="calibration_version"):
        validate_demo(
            episode,
            action_schema=action_schema,
            observation_schema=observation_schema,
        )


def test_level4_logger_is_append_only_even_with_overwrite(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    _write_level4_episode(episode_dir)
    action_schema, observation_schema = _schemas()
    logger = DemoLogger(
        episode_dir,
        action_schema=action_schema,
        observation_schema=observation_schema,
        overwrite=True,
    )
    initial = _action(0)[1]
    logger.start_episode(_metadata(initial, observation_schema.version))
    with pytest.raises(DemoLoggerError, match="append-only"):
        _append_one_level4_step(logger, initial, action_schema.action_dim)
        logger.close(success=True)


def _append_one_level4_step(logger: DemoLogger, action: np.ndarray, action_dim: int) -> None:
    logger.append(
        DemoStepData(
            features=np.zeros(14),
            action=action,
            robot_state=np.zeros(17),
            tracking_quality=np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            timestamp=0.0,
            requested_action=action,
            commanded_action=action,
            applied_action=action,
            prior_commanded_action=action,
            prior_applied_action=action,
            safety_mask=np.zeros(action_dim),
            safety_reason=["none"] * action_dim,
            request_source="operator",
            online_phase="approach",
            audited_phase="approach",
            phase_relevance_mask=np.ones(action_dim),
            intervention=False,
            failure_reason="",
            action_timestamp=0.0,
            task_timestamp=0.0,
            state_timestamp=0.0,
        )
    )


def test_session_manifest_rejects_duplicates_and_resume_skips_existing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sessions.json"
    session = RecordingSession(
        recording_session_id="session_a_20260904",
        operator_id="operator_local_01",
        split="train",
        process_start_timestamp="2026-09-04T12:00:00Z",
        reset_seed=7,
        calibration_record_digest="sha256:abc123",
    )
    append_session_manifest(manifest_path, session)
    with pytest.raises(SessionManifestError, match="already exists"):
        append_session_manifest(manifest_path, session)
    assert load_session_manifest(manifest_path).sessions == (session,)

    first = next_episode_directory(
        tmp_path / "dataset",
        recording_session_id=session.recording_session_id,
    )
    first.mkdir(parents=True)
    second = next_episode_directory(
        tmp_path / "dataset",
        recording_session_id=session.recording_session_id,
    )
    assert second.name == "episode_000002"


def test_legacy_level2_episode_loads_without_invented_level4_fields(tmp_path: Path) -> None:
    action_schema, observation_schema = _schemas()
    logger = DemoLogger(
        tmp_path / "legacy",
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    metadata = {
        "skill_name": "free_space_gesture",
        "task_name": "Free Space Gesture",
        "task_id": "free_space_gesture",
        "episode_id": "legacy_0001",
        "robot_model": "assets/mujoco/hand_scene.xml",
        "retargeter_config": "configs/level1_teleop.yaml",
        "control_rate_hz": 30.0,
        "teleop_config": {},
        "task_config": {},
    }
    logger.start_episode(metadata)
    action = _action(0)[1]
    logger.append(
        DemoStepData(
            features=np.zeros(14),
            action=action,
            robot_state=np.zeros(17),
            tracking_quality=np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            timestamp=0.0,
        )
    )
    logger.close(success=True)

    loaded = load_logged_demo(tmp_path / "legacy")
    validate_demo(
        loaded,
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    assert "recording_session_id" not in loaded.metadata
    assert loaded.requested_actions is None
