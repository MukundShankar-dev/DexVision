"""Closed-loop qualification for the frozen Level 4.3H push probe."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.evaluation.level4_button_learning import _joint_limit_names
from dexvision.learning.level4_lowdim import (
    DEFAULT_LEVEL4_DATASET_CONFIG,
    DEFAULT_RETARGETER_CONFIG,
    DEFAULT_WORKCELL_CONFIG,
    LowDimDeltaMLP,
    LowDimDeltaPolicy,
)
from dexvision.learning.level4_push_lowdim import (
    DEFAULT_PUSH_PILOT_CONFIG,
    PushActionAdapter,
    PushLearningError,
    PushTrainingResult,
    PushTrajectory,
    collect_push_expert_trajectories,
    fixed_push_finger_targets,
    load_push_learning_config,
    prepare_push_learning_reset,
    push_observation,
    push_task_frame,
    train_push_delta_policy,
)
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.sim.level4_expert import (
    DeterministicPushConfig,
    DeterministicPushExpert,
    _object_upright_tilt_rad,
    _unsafe_push_contact_reason,
)


PUSH_REPORT_VERSION = "level4/push-learning-report-v1"


@dataclass(frozen=True)
class PushRolloutResult:
    """One held-out rollout with task, boundary, neighbor, and safety outcomes."""

    rollout_id: str
    coverage_cell: str
    seed: int
    family: str
    success: bool
    task_success_observed: bool
    terminal_reason: str
    steps: int
    final_object_to_target_distance_m: float
    maximum_object_tilt_rad: float
    board_exit_count: int
    neighbor_disturbance_count: int
    workspace_violation_count: int
    joint_limit_violation_count: int
    unintended_contact_count: int
    object_tipped_count: int
    invalid_action_count: int
    phase_counts: Mapping[str, int]


@dataclass(frozen=True)
class PushLearningReport:
    """Training provenance, closed-loop metrics, and frozen gate results."""

    version: str
    config_digest: str
    collected_successes: int
    session_split_episode_counts: Mapping[str, int]
    selected_epoch: int
    training_loss: float
    validation_loss: float
    untouched_test_loss: float
    held_out_rollout_count: int
    held_out_success_count: int
    held_out_success_rate: float
    violation_totals: Mapping[str, int]
    gate_results: Mapping[str, bool]
    passed: bool
    failure_diagnosis: str | None
    recipe_change_count: int
    data_increase_count: int
    action_chunking_used: bool
    action_chunking_evidence: str
    rollouts: tuple[PushRolloutResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rollouts"] = [asdict(item) for item in self.rollouts]
        return payload


def run_push_learning_pilot(
    *,
    config_path: str | Path = DEFAULT_PUSH_PILOT_CONFIG,
    dataset_config: str | Path = DEFAULT_LEVEL4_DATASET_CONFIG,
    workcell_config: str | Path = DEFAULT_WORKCELL_CONFIG,
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> PushLearningReport:
    """Collect, train once, and qualify the frozen Level 4.3H formulation."""

    config, digest = load_push_learning_config(config_path)
    trajectories = collect_push_expert_trajectories(
        config,
        dataset_config=dataset_config,
        workcell_config=workcell_config,
        retargeter_config=retargeter_config,
    )
    training = train_push_delta_policy(trajectories, config)
    rollouts = evaluate_push_policy(
        training.policy,
        config,
        dataset_config=dataset_config,
        workcell_config=workcell_config,
        retargeter_config=retargeter_config,
    )
    return build_push_learning_report(
        config,
        config_digest=digest,
        trajectories=trajectories,
        training=training,
        rollouts=rollouts,
    )


def evaluate_push_policy(
    policy: LowDimDeltaPolicy,
    config: Mapping[str, Any],
    *,
    dataset_config: str | Path = DEFAULT_LEVEL4_DATASET_CONFIG,
    workcell_config: str | Path = DEFAULT_WORKCELL_CONFIG,
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> tuple[PushRolloutResult, ...]:
    """Run the fixed held-out test-cell/seed matrix without expert action fallback."""

    rollout = _mapping(config, "rollout")
    cells = rollout.get("held_out_coverage_cells")
    seeds = rollout.get("held_out_seeds")
    if (
        not isinstance(cells, Sequence)
        or isinstance(cells, (str, bytes))
        or not cells
        or not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
    ):
        raise PushLearningError("held-out push rollout cells and seeds are required.")
    return tuple(
        _rollout_push_policy(
            policy,
            config=config,
            coverage_cell=str(cells[index % len(cells)]),
            seed=int(seed),
            dataset_config=dataset_config,
            workcell_config=workcell_config,
            retargeter_config=retargeter_config,
        )
        for index, seed in enumerate(seeds)
    )


def build_push_learning_report(
    config: Mapping[str, Any],
    *,
    config_digest: str,
    trajectories: Sequence[PushTrajectory],
    training: PushTrainingResult,
    rollouts: Sequence[PushRolloutResult],
) -> PushLearningReport:
    """Apply the frozen success and zero-violation gates."""

    rollout_config = _mapping(config, "rollout")
    safety = _mapping(config, "safety")
    successes = sum(item.success for item in rollouts)
    count = len(rollouts)
    success_rate = successes / count if count else 0.0
    totals = {
        "board_exit": sum(item.board_exit_count for item in rollouts),
        "neighbor_disturbance": sum(
            item.neighbor_disturbance_count for item in rollouts
        ),
        "workspace": sum(item.workspace_violation_count for item in rollouts),
        "joint_limit": sum(item.joint_limit_violation_count for item in rollouts),
        "unintended_contact": sum(
            item.unintended_contact_count for item in rollouts
        ),
        "object_tipped": sum(item.object_tipped_count for item in rollouts),
        "invalid_action": sum(item.invalid_action_count for item in rollouts),
    }
    safety_total = sum(
        totals[name]
        for name in (
            "workspace",
            "joint_limit",
            "unintended_contact",
            "object_tipped",
        )
    )
    change = _mapping(config, "change_control")
    gates = {
        "qualified_interface_reused": _interface_reused(config, training.policy),
        "exactly_twenty_scripted_successes": len(trajectories) == 20,
        "at_least_twenty_held_out_resets": count >= 20,
        "minimum_held_out_success_rate": success_rate
        >= float(rollout_config["minimum_success_rate"]),
        "zero_board_exits": totals["board_exit"]
        <= int(safety["maximum_board_exits"]),
        "zero_neighbor_disturbances": totals["neighbor_disturbance"]
        <= int(safety["maximum_neighbor_disturbances"]),
        "zero_safety_violations": safety_total
        <= int(safety["maximum_safety_violations"]),
        "zero_invalid_actions": totals["invalid_action"]
        <= int(safety["maximum_invalid_actions"]),
        "no_unsupported_action_chunking": change["allow_action_chunking"] is False,
    }
    passed = all(gates.values())
    return PushLearningReport(
        version=PUSH_REPORT_VERSION,
        config_digest=config_digest,
        collected_successes=len(trajectories),
        session_split_episode_counts=dict(
            sorted(Counter(item.spec.split for item in trajectories).items())
        ),
        selected_epoch=training.selected_epoch,
        training_loss=training.training_loss,
        validation_loss=training.validation_loss,
        untouched_test_loss=training.test_loss,
        held_out_rollout_count=count,
        held_out_success_count=successes,
        held_out_success_rate=success_rate,
        violation_totals=totals,
        gate_results=gates,
        passed=passed,
        failure_diagnosis=None if passed else diagnose_push_rollout_failures(rollouts),
        recipe_change_count=0,
        data_increase_count=0,
        action_chunking_used=False,
        action_chunking_evidence=str(change["action_chunking_evidence"]),
        rollouts=tuple(rollouts),
    )


def diagnose_push_rollout_failures(rollouts: Sequence[PushRolloutResult]) -> str:
    """Prioritize a measured failure mechanism before any scale/model change."""

    if not rollouts:
        return "evaluation_protocol_missing_rollouts"
    ordered = (
        ("invalid_action_count", "invalid_numeric_policy_output"),
        ("workspace_violation_count", "task_delta_scaling_or_workspace_adapter"),
        ("board_exit_count", "push_stop_or_board_boundary_constraint"),
        ("neighbor_disturbance_count", "wrong_object_or_scene_disturbance"),
        ("joint_limit_violation_count", "fixed_posture_or_joint_limit_adapter"),
        ("unintended_contact_count", "approach_geometry_or_contact_constraint"),
        ("object_tipped_count", "contact_height_or_push_axis_constraint"),
    )
    for field, reason in ordered:
        if any(getattr(item, field) for item in rollouts):
            return reason
    failures = [item for item in rollouts if not item.success]
    if any(item.phase_counts.get("push_contact", 0) == 0 for item in failures):
        return "closed_loop_approach_did_not_make_contact"
    if any(not item.task_success_observed for item in failures):
        return "single_step_push_did_not_reach_or_settle_at_goal"
    if failures:
        return "retract_or_release_completion_failure"
    return "no_rollout_failure_detected"


def save_push_learning_report(
    report: PushLearningReport, output_path: str | Path
) -> Path:
    """Atomically save the reproducible pilot report outside working episodes."""

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise PushLearningError("push learning report must use a .json extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _rollout_push_policy(
    policy: LowDimDeltaPolicy,
    *,
    config: Mapping[str, Any],
    coverage_cell: str,
    seed: int,
    dataset_config: str | Path,
    workcell_config: str | Path,
    retargeter_config: str | Path,
) -> PushRolloutResult:
    rollout = _mapping(config, "rollout")
    safety = _mapping(config, "safety")
    phase = "approach"
    phase_counts: Counter[str] = Counter()
    achieved_success = False
    terminal_reason = "timeout"
    counters = Counter()
    maximum_tilt = 0.0
    steps = 0
    with WorkcellPilotTask(
        workcell_config=workcell_config,
        dataset_config=dataset_config,
        skill_name="push_object_to_target",
        goal_condition_id=coverage_cell,
        seed=seed,
    ) as task:
        fingers = fixed_push_finger_targets(task, retargeter_config)
        world = prepare_push_learning_reset(
            task, config, seed=seed, fixed_fingers=fingers
        )
        initial_positions = {
            item.object_id: np.asarray(item.position, dtype=np.float64)
            for item in world.entities
        }
        expert_config = DeterministicPushConfig.from_mapping(
            task.collection_config["pilot"]["scripted_push"]
        )
        nominal = DeterministicPushExpert(
            finger_targets=fingers, config=expert_config
        )
        nominal.reset(task, world)
        if nominal.validation is None or not nominal.validation.valid:
            reason = nominal.validation.reason if nominal.validation else "missing_validation"
            raise PushLearningError(f"held-out nominal push is invalid: {reason}")
        frame = push_task_frame(task, world)
        previous_delta = np.zeros(3, dtype=np.float64)
        adapter = PushActionAdapter(
            finger_targets=fingers,
            initial_orientation_wxyz=world.robot.base_orientation_wxyz,
            target_orientation_wxyz=nominal.target_orientation_wxyz,
            task_to_world_rotation=frame,
            workspace_min_m=safety["workspace_min_m"],
            workspace_max_m=safety["workspace_max_m"],
            maximum_absolute_delta_by_phase_m=_mapping(config, "action")[
                "maximum_absolute_delta_by_phase_m"
            ],
            transit_height_m=expert_config.transit_height_m,
            orientation_step_rad=expert_config.orientation_step_rad,
        )
        object_id = str(task.goal["object_id"])
        target_id = str(task.goal["target_zone"])
        for step_index in range(int(rollout["maximum_steps"])):
            nominal_requested, phase, nominal_done, nominal_reason = nominal.step(
                world
            )
            phase_counts[phase] += 1
            if nominal_reason is not None:
                terminal_reason = nominal_reason
                break
            observation = push_observation(
                task,
                world,
                phase=phase,
                previous_applied_delta=previous_delta,
                target_orientation_wxyz=nominal.target_orientation_wxyz,
                task_to_world_rotation=frame,
                target_stop_distance_m=expert_config.target_stop_distance_m,
            )
            try:
                delta = policy.predict(observation)
                requested, workspace_violation = adapter.expand(
                    nominal_requested.base_position,
                    delta,
                    phase=phase,
                    nominal_orientation_wxyz=nominal_requested.base_orientation_wxyz,
                )
            except (PushLearningError, FloatingPointError, ValueError):
                counters["invalid_action"] += 1
                terminal_reason = "invalid_action"
                break
            if not np.all(np.isfinite(delta)):
                counters["invalid_action"] += 1
                terminal_reason = "invalid_action"
                break
            counters["workspace"] += int(workspace_violation)
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=requested.base_position,
                orientation_quat=requested.base_orientation_wxyz,
            )
            task.env.set_joint_targets(requested.finger_targets)
            state = task.step(n_steps=int(rollout["sim_steps_per_action"]))
            world = state.world_state
            steps = step_index + 1
            previous_delta = np.asarray(delta, dtype=np.float64)
            object_state = world.require_entity(object_id)
            tilt = _object_upright_tilt_rad(object_state.orientation_wxyz)
            maximum_tilt = max(maximum_tilt, tilt)
            counters["object_tipped"] += int(
                tilt > expert_config.maximum_object_tilt_rad
            )
            counters["board_exit"] += int(
                state.task_values.get("object_on_board") is False
            )
            maximum_neighbor = max(
                (
                    float(
                        np.linalg.norm(
                            np.asarray(item.position[:2])
                            - initial_positions[item.object_id][:2]
                        )
                    )
                    for item in world.entities
                    if item.object_id != object_id
                    and item.object_id in initial_positions
                ),
                default=0.0,
            )
            counters["neighbor_disturbance"] += int(
                maximum_neighbor > expert_config.maximum_non_target_disturbance_m
            )
            counters["joint_limit"] += int(
                bool(
                    _joint_limit_names(
                        task.workcell,
                        tolerance=float(safety["joint_limit_tolerance_rad"]),
                    )
                )
            )
            contact_reason = _unsafe_push_contact_reason(world, object_id=object_id)
            counters["unintended_contact"] += int(contact_reason is not None)
            if any(counters.values()):
                terminal_reason = contact_reason or "safety_violation"
                break
            achieved_success = achieved_success or state.success
            if nominal_done:
                terminal_reason = "completed"
                break
        final_distance = math.dist(
            world.require_entity(object_id).position[:2],
            world.require_entity(target_id).position[:2],
        )
    success = (
        achieved_success
        and terminal_reason == "completed"
        and not any(counters.values())
    )
    return PushRolloutResult(
        rollout_id=f"{coverage_cell}__seed_{seed}",
        coverage_cell=coverage_cell,
        seed=seed,
        family=str(task.coverage_cell["family"]),
        success=success,
        task_success_observed=achieved_success,
        terminal_reason=terminal_reason,
        steps=steps,
        final_object_to_target_distance_m=float(final_distance),
        maximum_object_tilt_rad=float(maximum_tilt),
        board_exit_count=counters["board_exit"],
        neighbor_disturbance_count=counters["neighbor_disturbance"],
        workspace_violation_count=counters["workspace"],
        joint_limit_violation_count=counters["joint_limit"],
        unintended_contact_count=counters["unintended_contact"],
        object_tipped_count=counters["object_tipped"],
        invalid_action_count=counters["invalid_action"],
        phase_counts=dict(sorted(phase_counts.items())),
    )


def _interface_reused(
    config: Mapping[str, Any], policy: LowDimDeltaPolicy
) -> bool:
    interface = _mapping(config, "interface")
    action = _mapping(config, "action")
    return bool(
        interface["action_schema"] == action["schema_version"]
        and interface["control_sim_steps"]
        == _mapping(config, "rollout")["sim_steps_per_action"]
        and tuple(action["output_fields"]) == ("dx", "dy", "dz")
        and isinstance(policy.model, LowDimDeltaMLP)
        and tuple(policy.action_normalization.mean.shape) == (3,)
    )


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = values.get(key)
    if not isinstance(result, Mapping):
        raise PushLearningError(f"push pilot {key} must be a mapping.")
    return result
