"""Closed-loop qualification for the frozen Level 4.3G button learning probe."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.learning.level4_lowdim import (
    DEFAULT_BUTTON_PILOT_CONFIG,
    DEFAULT_LEVEL4_DATASET_CONFIG,
    DEFAULT_RETARGETER_CONFIG,
    DEFAULT_WORKCELL_CONFIG,
    ButtonActionAdapter,
    ButtonDeltaPolicy,
    ButtonLearningError,
    ButtonTrainingResult,
    button_observation,
    collect_button_expert_trajectories,
    fixed_button_finger_targets,
    load_button_learning_config,
    prepare_button_learning_reset,
    train_button_delta_policy,
)
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.sim.level4_expert import (
    _unsafe_button_contact_reason,
)


BUTTON_REPORT_VERSION = "level4/button-learning-report-v1"


@dataclass(frozen=True)
class ButtonRolloutResult:
    """One held-out closed-loop rollout and all safety outcomes."""

    rollout_id: str
    coverage_cell: str
    seed: int
    success: bool
    task_success_observed: bool
    terminal_reason: str
    steps: int
    final_press_depth_m: float
    workspace_violation_count: int
    joint_limit_violation_count: int
    joint_limit_names: tuple[str, ...]
    wrong_button_contact_count: int
    unintended_contact_count: int
    invalid_action_count: int
    phase_counts: Mapping[str, int]


@dataclass(frozen=True)
class ButtonLearningReport:
    """Training provenance, rollout metrics, gates, and failure diagnosis."""

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
    rollouts: tuple[ButtonRolloutResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rollouts"] = [asdict(item) for item in self.rollouts]
        return payload


def evaluate_button_policy(
    policy: ButtonDeltaPolicy,
    config: Mapping[str, Any],
    *,
    dataset_config: str | Path = DEFAULT_LEVEL4_DATASET_CONFIG,
    workcell_config: str | Path = DEFAULT_WORKCELL_CONFIG,
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> tuple[ButtonRolloutResult, ...]:
    """Run the frozen held-out seed/cell matrix without expert fallback."""

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
        raise ButtonLearningError("held-out rollout cells and seeds are required.")
    fixed_fingers = fixed_button_finger_targets(config, retargeter_config)
    return tuple(
        _rollout_button_policy(
            policy,
            config=config,
            coverage_cell=str(cells[index % len(cells)]),
            seed=int(seed),
            fixed_fingers=fixed_fingers,
            dataset_config=dataset_config,
            workcell_config=workcell_config,
        )
        for index, seed in enumerate(seeds)
    )


def run_button_learning_pilot(
    *,
    config_path: str | Path = DEFAULT_BUTTON_PILOT_CONFIG,
    dataset_config: str | Path = DEFAULT_LEVEL4_DATASET_CONFIG,
    workcell_config: str | Path = DEFAULT_WORKCELL_CONFIG,
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> ButtonLearningReport:
    """Collect, train once, and qualify the frozen Level 4.3G formulation."""

    config, digest = load_button_learning_config(config_path)
    trajectories = collect_button_expert_trajectories(
        config,
        dataset_config=dataset_config,
        workcell_config=workcell_config,
        retargeter_config=retargeter_config,
    )
    training = train_button_delta_policy(trajectories, config)
    rollouts = evaluate_button_policy(
        training.policy,
        config,
        dataset_config=dataset_config,
        workcell_config=workcell_config,
        retargeter_config=retargeter_config,
    )
    return build_button_learning_report(
        config,
        config_digest=digest,
        trajectories=trajectories,
        training=training,
        rollouts=rollouts,
    )


def build_button_learning_report(
    config: Mapping[str, Any],
    *,
    config_digest: str,
    trajectories: Sequence[object],
    training: ButtonTrainingResult,
    rollouts: Sequence[ButtonRolloutResult],
) -> ButtonLearningReport:
    """Apply the frozen numerical and zero-violation gates."""

    rollout_config = _mapping(config, "rollout")
    safety = _mapping(config, "safety")
    successes = sum(item.success for item in rollouts)
    count = len(rollouts)
    success_rate = successes / count if count else 0.0
    totals = {
        "workspace": sum(item.workspace_violation_count for item in rollouts),
        "joint_limit": sum(item.joint_limit_violation_count for item in rollouts),
        "wrong_button_contact": sum(
            item.wrong_button_contact_count for item in rollouts
        ),
        "unintended_contact": sum(
            item.unintended_contact_count for item in rollouts
        ),
        "invalid_action": sum(item.invalid_action_count for item in rollouts),
    }
    gates = {
        "exactly_one_frozen_recipe": int(_mapping(config, "model")["recipe_count"])
        == 1,
        "exactly_twenty_scripted_successes": len(trajectories) == 20,
        "at_least_twenty_held_out_resets": count >= 20,
        "minimum_held_out_success_rate": success_rate
        >= float(rollout_config["minimum_success_rate"]),
        "zero_workspace_violations": totals["workspace"]
        <= int(safety["maximum_workspace_violations"]),
        "zero_joint_limit_violations": totals["joint_limit"]
        <= int(safety["maximum_joint_limit_violations"]),
        "zero_wrong_button_contacts": totals["wrong_button_contact"]
        <= int(safety["maximum_wrong_button_contacts"]),
        "zero_unintended_contacts": totals["unintended_contact"]
        <= int(safety["maximum_unintended_contacts"]),
        "zero_invalid_actions": totals["invalid_action"] == 0,
    }
    passed = all(gates.values())
    diagnosis = diagnose_button_rollout_failures(rollouts) if not passed else None
    split_counts = Counter(getattr(item, "spec").split for item in trajectories)
    return ButtonLearningReport(
        version=BUTTON_REPORT_VERSION,
        config_digest=config_digest,
        collected_successes=len(trajectories),
        session_split_episode_counts=dict(sorted(split_counts.items())),
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
        failure_diagnosis=diagnosis,
        recipe_change_count=0,
        data_increase_count=0,
        rollouts=tuple(rollouts),
    )


def diagnose_button_rollout_failures(
    rollouts: Sequence[ButtonRolloutResult],
) -> str:
    """Return the first causal category to inspect before changing scale/model."""

    if not rollouts:
        return "evaluation_protocol_missing_rollouts"
    if any(item.invalid_action_count for item in rollouts):
        return "invalid_numeric_policy_output"
    if any(item.workspace_violation_count for item in rollouts):
        return "task_delta_scaling_or_workspace_adapter"
    if any(item.joint_limit_violation_count for item in rollouts):
        return "fixed_posture_or_joint_limit_adapter"
    if any(item.wrong_button_contact_count for item in rollouts):
        return "button_identity_or_fixture_contact"
    if any(item.unintended_contact_count for item in rollouts):
        return "approach_geometry_or_contact_constraint"
    failures = [item for item in rollouts if not item.success]
    if any(not item.task_success_observed for item in failures):
        if any(item.phase_counts.get("fixture_contact", 0) == 0 for item in failures):
            return "closed_loop_approach_did_not_reach_button"
        return "fixture_contact_did_not_reach_press_metric"
    if failures:
        return "retract_or_release_completion_failure"
    return "no_rollout_failure_detected"


def save_button_learning_report(
    report: ButtonLearningReport, output_path: str | Path
) -> Path:
    """Atomically save the reproducible pilot result outside working episodes."""

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise ButtonLearningError("button learning report must use a .json extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _rollout_button_policy(
    policy: ButtonDeltaPolicy,
    *,
    config: Mapping[str, Any],
    coverage_cell: str,
    seed: int,
    fixed_fingers: Mapping[str, float],
    dataset_config: str | Path,
    workcell_config: str | Path,
) -> ButtonRolloutResult:
    rollout = _mapping(config, "rollout")
    safety = _mapping(config, "safety")
    phase = "approach"
    achieved_success = False
    release_dwell = 0
    workspace_violations = 0
    joint_violations = 0
    violated_joint_names: set[str] = set()
    wrong_button_contacts = 0
    unintended_contacts = 0
    invalid_actions = 0
    terminal_reason = "timeout"
    phase_counts: Counter[str] = Counter()
    steps = 0
    with WorkcellPilotTask(
        workcell_config=workcell_config,
        dataset_config=dataset_config,
        skill_name="press_button",
        goal_condition_id=coverage_cell,
        seed=seed,
    ) as task:
        if task.coverage_cell.get("split_owner") != "test":
            raise ButtonLearningError(
                f"held-out rollout cell {coverage_cell!r} must be test-owned."
            )
        world = task.current_state.world_state
        world = prepare_button_learning_reset(
            task,
            config,
            seed=seed,
            fixed_fingers=fixed_fingers,
        )
        target_ee_orientation = np.asarray(
            world.robot.end_effector_orientation_wxyz, dtype=np.float64
        )
        previous_position = np.asarray(
            world.robot.end_effector_position, dtype=np.float64
        )
        previous_delta = np.zeros(3, dtype=np.float64)
        adapter = ButtonActionAdapter(
            finger_targets=fixed_fingers,
            fixed_orientation_wxyz=world.robot.base_orientation_wxyz,
            workspace_min_m=safety["workspace_min_m"],
            workspace_max_m=safety["workspace_max_m"],
            maximum_absolute_delta_by_phase_m=_mapping(config, "action")[
                "maximum_absolute_delta_by_phase_m"
            ],
        )
        button_id = str(task.goal["button_id"])
        for step_index in range(int(rollout["maximum_steps"])):
            phase_counts[phase] += 1
            observation = button_observation(
                task,
                world,
                phase=phase,
                previous_applied_delta=previous_delta,
                target_orientation_wxyz=target_ee_orientation,
            )
            try:
                delta = policy.predict(observation)
            except (ButtonLearningError, FloatingPointError, ValueError):
                invalid_actions += 1
                terminal_reason = "invalid_action"
                break
            if not np.all(np.isfinite(delta)):
                invalid_actions += 1
                terminal_reason = "invalid_action"
                break
            requested, workspace_violation = adapter.expand(
                previous_position, delta, phase=phase
            )
            workspace_violations += int(workspace_violation)
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=requested.base_position,
                orientation_quat=requested.base_orientation_wxyz,
            )
            task.env.set_joint_targets(requested.finger_targets)
            state = task.step(n_steps=int(rollout["sim_steps_per_action"]))
            world = state.world_state
            steps = step_index + 1
            previous_position = requested.base_position
            previous_delta = delta
            step_joint_violations = _joint_limit_names(
                task.workcell,
                tolerance=float(safety["joint_limit_tolerance_rad"]),
            )
            if step_joint_violations:
                joint_violations += 1
                violated_joint_names.update(step_joint_violations)
            contact_reason = _unsafe_button_contact_reason(world, button_id=button_id)
            if contact_reason == "wrong_fixture_contact":
                wrong_button_contacts += 1
            elif contact_reason is not None:
                unintended_contacts += 1
            if (
                workspace_violations
                or joint_violations
                or wrong_button_contacts
                or unintended_contacts
            ):
                terminal_reason = contact_reason or "safety_violation"
                break
            button_contact = any(
                button_id in pair and any(name.startswith("rh_") for name in pair)
                for pair in world.contacts
            )
            if state.success:
                achieved_success = True
                phase = "retract"
            elif phase == "approach" and button_contact:
                phase = "fixture_contact"
            fixture = world.require_fixture(button_id)
            released = (
                achieved_success
                and fixture.press_depth_m <= 0.002
                and not button_contact
            )
            release_dwell = release_dwell + 1 if released else 0
            if release_dwell >= int(rollout["required_release_steps"]):
                terminal_reason = "completed"
                break
        final_depth = world.require_fixture(button_id).press_depth_m
    success = (
        achieved_success
        and terminal_reason == "completed"
        and not any(
            (
                workspace_violations,
                joint_violations,
                wrong_button_contacts,
                unintended_contacts,
                invalid_actions,
            )
        )
    )
    return ButtonRolloutResult(
        rollout_id=f"{coverage_cell}__seed_{seed}",
        coverage_cell=coverage_cell,
        seed=seed,
        success=success,
        task_success_observed=achieved_success,
        terminal_reason=terminal_reason,
        steps=steps,
        final_press_depth_m=float(final_depth),
        workspace_violation_count=workspace_violations,
        joint_limit_violation_count=joint_violations,
        joint_limit_names=tuple(sorted(violated_joint_names)),
        wrong_button_contact_count=wrong_button_contacts,
        unintended_contact_count=unintended_contacts,
        invalid_action_count=invalid_actions,
        phase_counts=dict(sorted(phase_counts.items())),
    )


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = values.get(key)
    if not isinstance(result, Mapping):
        raise ButtonLearningError(f"button pilot {key} must be a mapping.")
    return result


def _joint_limit_names(workcell: object, *, tolerance: float) -> tuple[str, ...]:
    model = workcell.env.model
    data = workcell.env.data
    mujoco = workcell.env._mujoco
    free_joint = int(mujoco.mjtJoint.mjJNT_FREE)
    ball_joint = int(mujoco.mjtJoint.mjJNT_BALL)
    violations: list[str] = []
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        if int(model.jnt_type[joint_id]) in {free_joint, ball_joint}:
            continue
        address = int(model.jnt_qposadr[joint_id])
        qpos = float(data.qpos[address])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        if qpos < lower - tolerance or qpos > upper + tolerance:
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            violations.append(str(name))
    return tuple(violations)
