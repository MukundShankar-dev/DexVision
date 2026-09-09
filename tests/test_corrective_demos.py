from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexvision.logging.corrective_demos import (
    CorrectiveDemoError,
    append_manual_correction_replay,
    build_level4_6_plan,
    generate_scripted_correction,
    generate_scripted_failure,
    select_level4_training_streams,
    validate_level4_6_metadata,
    load_manual_correction_replays,
)
from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoStepData,
    build_level1_action_schema,
    build_level2_observation_schema,
    load_logged_demo,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "level4_dataset.yaml"
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


def _write_nominal(path: Path, *, episode_id: str = "nominal_001") -> None:
    action_schema = build_level1_action_schema(("finger_a", "finger_b"))
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=2,
        robot_qvel_dim=2,
        finger_target_dim=2,
        tracking_quality_dim=6,
        object_state_dim=3,
        task_state_dim=2,
        success_metric_dim=2,
    )
    mask = [1] * action_schema.action_dim
    metadata = {
        "skill_name": "pick_place_sequence",
        "task_name": "Pick Place Sequence",
        "task_id": "level4_workcell",
        "episode_id": episode_id,
        "robot_model": "assets/mujoco/workcell_scene.xml",
        "retargeter_config": "configs/level1_teleop.yaml",
        "control_rate_hz": 30.0,
        "teleop_config": {},
        "task_config": {
            "required_objects": ["block_small"],
            "requires_task_state": True,
            "requires_success_metric_inputs": True,
            "required_observation_fields": ["object_state", "task_state"],
            "initial_state": {"objects": {}},
        },
        "episode_schema_version": "level4/episode-v1",
        "recording_session_id": "nominal_session",
        "operator_id": "scripted_nominal",
        "source": "scripted",
        "data_stream": "expert_success",
        "typed_goal": {"object_id": "block_small", "target_id": "setup_slot_a"},
        "object_instance_ids": ["block_small"],
        "goal_condition_id": "pp_block_small_setup_slot_a",
        "reset_state": {},
        "random_seed": 123,
        "camera_or_render_config": None,
        "code_version": "test-tree",
        "config_version": "level4/workcell-dataset-plan-v19",
        "schema_versions": {
            "episode": "level4/episode-v1",
            "observation": observation_schema.version,
            "action": "level4/request-command-apply-v1",
            "world_state": "level4/world-state-v1",
            "phase": "level4/causal-phase-v1",
            "safety": "level4/action-safety-v1",
        },
        "phase_contract": {
            "version": "level4/causal-phase-v1",
            "vocabulary": list(PHASES),
            "transitions": [
                {"from": source, "to": target}
                for source, target in zip(PHASES, PHASES[1:])
            ],
            "action_relevance_masks": {phase: mask for phase in PHASES},
        },
        "action_contract": {
            "version": "level4/request-command-apply-v1",
            "safety_reason_codes": ["none", "workspace_clip", "joint_limit_clip"],
            "max_state_action_timestamp_skew_s": 0.005,
        },
    }
    logger = DemoLogger(
        path,
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    logger.start_episode(metadata)
    for index, phase in enumerate(PHASES):
        action = np.asarray(
            [0.01 * index, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2],
            dtype=np.float64,
        )
        logger.append(
            DemoStepData(
                features=np.zeros(14),
                action=action,
                robot_state=np.zeros(13),
                tracking_quality=np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
                object_state=np.full(3, index, dtype=float),
                task_state=np.asarray([index, 0.0]),
                timestamp=index / 30.0,
                request_source="script",
                online_phase=phase,
                audited_phase=phase,
            )
        )
    logger.close(success=True)


def test_frozen_plan_has_twelve_cells_and_120_unique_assignments() -> None:
    plan = build_level4_6_plan(CONFIG_PATH)

    assert len(plan) == 120
    assert len({item.coverage_cell_id for item in plan}) == 12
    assert len({item.generation_seed for item in plan}) == 120
    assert sum(item.is_correction for item in plan) == 30
    assert all(
        item.retryability == "abort"
        for item in plan
        if item.failure_class in {"workspace_violation", "joint_limit_violation"}
    )


def test_correction_preserves_failure_prefix_and_conditional_provenance(
    tmp_path: Path,
) -> None:
    nominal_dir = tmp_path / "nominal" / "episode_000001"
    _write_nominal(nominal_dir)
    plan = build_level4_6_plan(CONFIG_PATH)
    failure_assignment = next(
        item
        for item in plan
        if item.coverage_cell_id == "failure_approach_miss" and item.repetition == 1
    )
    correction_assignment = next(
        item
        for item in plan
        if item.coverage_cell_id == "correction_approach_miss" and item.repetition == 1
    )
    failure_dir = tmp_path / failure_assignment.session_id / "episode_000001"
    correction_dir = tmp_path / correction_assignment.session_id / "episode_000001"

    generate_scripted_failure(
        source_rollout=nominal_dir,
        output_dir=failure_dir,
        assignment=failure_assignment,
        dataset_dir=tmp_path,
    )
    generate_scripted_correction(
        source_rollout=failure_dir,
        output_dir=correction_dir,
        assignment=correction_assignment,
        dataset_dir=tmp_path,
    )

    failure = load_logged_demo(failure_dir)
    correction = load_logged_demo(correction_dir)
    validate_level4_6_metadata(failure.metadata)
    validate_level4_6_metadata(correction.metadata)
    boundary = correction.metadata["intervention_interval"][0]
    assert correction.metadata["source_episode_id"] == failure.metadata["episode_id"]
    assert correction.metadata["trigger_source"] == "scripted"
    assert correction.metadata["source_policy_checkpoint"] is None
    assert (
        correction.metadata["source_policy_checkpoint_not_applicable_reason"]
        == "trigger_not_policy_rollout"
    )
    assert np.array_equal(failure.actions, correction.actions[:boundary])
    assert np.all(correction.intervention_flags[:boundary] == 0)
    assert np.all(correction.intervention_flags[boundary:] == 1)
    assert correction.metadata["original_terminal_result"] == failure.metadata[
        "original_terminal_result"
    ]
    assert correction.metadata["final_outcome"] == "corrected_success"

    manifest = append_manual_correction_replay(
        tmp_path,
        episode_id=str(correction.metadata["episode_id"]),
        notes="Visible failure and deterministic correction boundary matched metadata.",
    )
    assert manifest.name == "correction_manual_replay_manifest.json"
    reviews = load_manual_correction_replays(tmp_path)
    assert reviews[0]["episode_id"] == correction.metadata["episode_id"]
    assert reviews[0]["passed"] is True

    assert select_level4_training_streams(
        (failure_dir, correction_dir), include_corrections=False
    ) == ()
    assert select_level4_training_streams(
        (failure_dir, correction_dir), include_corrections=True
    ) == (correction_dir,)


def test_workspace_violation_is_clipped_abort_only(tmp_path: Path) -> None:
    nominal_dir = tmp_path / "nominal" / "episode_000001"
    _write_nominal(nominal_dir)
    assignment = next(
        item
        for item in build_level4_6_plan(CONFIG_PATH)
        if item.coverage_cell_id == "failure_workspace_violation"
        and item.repetition == 1
    )
    output_dir = tmp_path / assignment.session_id / "episode_000001"

    generate_scripted_failure(
        source_rollout=nominal_dir,
        output_dir=output_dir,
        assignment=assignment,
        dataset_dir=tmp_path,
    )

    episode = load_logged_demo(output_dir)
    assert episode.metadata["retryability"] == "abort"
    assert episode.metadata["intervention_interval"] is None
    assert episode.metadata["final_outcome"] == "aborted_unsafe"
    assert episode.requested_actions[-1, 0] == pytest.approx(10.0)
    assert episode.applied_actions[-1, 0] != pytest.approx(10.0)
    assert episode.safety_masks[-1, 0] == 1
    assert episode.safety_reasons[-1, 0] == "workspace_clip"


def test_policy_provenance_is_conditional() -> None:
    metadata = {
        "level4_6": {"version": "level4/scripted-corrections-v1"},
        "data_stream": "corrective_intervention",
        "failure_class": "approach_miss",
        "retryability": "retry_once_after_safe_pose",
        "source_episode_id": "policy_failure_001",
        "trigger_source": "policy_rollout",
        "source_policy_checkpoint": None,
        "source_policy_checkpoint_not_applicable_reason": None,
        "intervention_interval": [2, 4],
        "final_outcome": "corrected_success",
    }
    with pytest.raises(CorrectiveDemoError, match="requires a checkpoint"):
        validate_level4_6_metadata(metadata)

    metadata["source_policy_checkpoint"] = "checkpoint_010.pt"
    validate_level4_6_metadata(metadata)
