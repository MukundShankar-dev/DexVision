"""Frozen Level 3 reach-policy closed-loop rollout evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from dexvision.evaluation.benchmark_retargeters import mean_action_jerk
from dexvision.learning.datasets import quaternion_wxyz_to_rotation_6d
from dexvision.learning.models import (
    BASE_ORIENTATION_ACTION_NAMES,
    BASE_POSITION_ACTION_NAMES,
    FINGER_ACTION_PREFIX,
)
from dexvision.learning.policies import RolloutPolicy
from dexvision.sim.hand_base_control import WorkspaceLimits
from dexvision.sim.tasks import (
    ButtonPressParameters,
    ButtonPressTask,
    DEFAULT_TASK_BOARD_MODEL,
    PushCubeParameters,
    PushCubeTask,
    ReachTouchTargetParameters,
    ReachTouchTargetTask,
)


class PolicyEvaluationError(RuntimeError):
    """Raised when a rollout protocol, backend, or policy is incompatible."""


@dataclass(frozen=True)
class ReachScenario:
    """One frozen target and initial-base-offset combination."""

    scenario_id: str
    target_group: str
    target_id: str
    target_position: tuple[float, float, float]
    offset_id: str
    initial_base_offset_m: tuple[float, float, float]
    repetition: int


@dataclass(frozen=True)
class ReachEvaluationProtocol:
    """Validated machine-readable reach rollout protocol."""

    version: str
    task_id: str
    skill_name: str
    scenarios: tuple[ReachScenario, ...]
    acceptance_gates: dict[str, float | int]
    source_path: str
    source_digest: str


@dataclass(frozen=True)
class BackendState:
    """Task terminal and metric state returned by a rollout backend."""

    success: bool
    failure_reason: str | None
    distance_to_target: float
    step_count: int


@dataclass(frozen=True)
class ActionBounds:
    """Bounds used both for clipping and normalized-jerk measurement."""

    lower: np.ndarray
    upper: np.ndarray
    workspace_indices: tuple[int, ...]
    joint_indices: tuple[int, ...]


class ReachRolloutBackend(Protocol):
    """Backend boundary allowing CPU-only deterministic rollout tests."""

    max_episode_steps: int
    observation_schema_version: str
    action_schema_version: str

    def reset(self, scenario: ReachScenario) -> BackendState: ...

    def observation(
        self,
        names: Sequence[str],
        *,
        previous_action: Mapping[str, float],
    ) -> np.ndarray: ...

    def initial_action(self, action_names: Sequence[str]) -> np.ndarray: ...

    def action_bounds(self, action_names: Sequence[str]) -> ActionBounds: ...

    def step(
        self,
        action_names: Sequence[str],
        action: np.ndarray,
        *,
        n_steps: int,
    ) -> BackendState: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RolloutResult:
    """Metrics and provenance for one saved rollout."""

    scenario_id: str
    target_group: str
    target_id: str
    offset_id: str
    repetition: int
    success: bool
    terminal_reason: str
    completion_steps: int
    final_distance_m: float
    normalized_action_jerk: float
    invalid_action_count: int
    workspace_violation_count: int
    joint_limit_violation_count: int
    trajectory_file: str


@dataclass(frozen=True)
class EvaluationReport:
    """Complete frozen-matrix results and numerical gate decisions."""

    report_version: str
    protocol_version: str
    protocol_digest: str
    rollout_config: dict[str, Any]
    rollout_config_digest: str
    checkpoint_digest: str
    dataset_digest: str
    split_manifest_digest: str
    training_config_digest: str
    schema_digest: str
    selected_epoch: int
    selected_validation_loss: float
    action_mode: str
    goal_input_mode: str
    results: tuple[RolloutResult, ...]
    metrics: dict[str, Any]
    acceptance_gates: dict[str, float | int]
    gate_results: dict[str, bool]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(result) for result in self.results]
        return payload


def load_reach_evaluation_protocol(path: str | Path) -> ReachEvaluationProtocol:
    """Load the frozen matrix without deriving or changing its conditions."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw)
    except OSError as exc:
        raise PolicyEvaluationError(f"cannot read evaluation config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PolicyEvaluationError(f"invalid evaluation YAML {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PolicyEvaluationError("evaluation config must contain a mapping.")
    version = _required_string(payload, "version")
    if version != "level3/reach-evaluation-v1":
        raise PolicyEvaluationError(f"unsupported reach evaluation version {version!r}.")
    if _required_string(payload, "task_id") != "reach_touch_target":
        raise PolicyEvaluationError("Level 3.4 supports only task_id='reach_touch_target'.")
    rollout = _required_mapping(payload, "rollout")
    repetitions = rollout.get("repetitions_per_scenario")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise PolicyEvaluationError("rollout.repetitions_per_scenario must be positive.")
    for key in ("use_task_success_condition", "stop_on_success", "stop_on_failure", "stop_on_timeout"):
        if rollout.get(key) is not True:
            raise PolicyEvaluationError(f"rollout.{key} must be true for explicit termination.")

    targets = (
        ("training", _required_mapping(payload, "training_targets")),
        ("held_out", _required_mapping(payload, "held_out_rollout_targets")),
    )
    offsets = _required_mapping(payload, "initial_base_position_offsets_m")
    scenarios: list[ReachScenario] = []
    for group, group_targets in targets:
        for target_id, target_value in group_targets.items():
            target = _vector3(target_value, label=f"target {target_id!r}")
            for offset_id, offset_value in offsets.items():
                offset = _vector3(offset_value, label=f"offset {offset_id!r}")
                for repetition in range(repetitions):
                    scenario_id = f"{group}__{target_id}__{offset_id}__r{repetition + 1}"
                    scenarios.append(
                        ReachScenario(
                            scenario_id=scenario_id,
                            target_group=group,
                            target_id=str(target_id),
                            target_position=target,
                            offset_id=str(offset_id),
                            initial_base_offset_m=offset,
                            repetition=repetition + 1,
                        )
                    )
    gates = dict(_required_mapping(payload, "acceptance_gates"))
    required_gates = {
        "minimum_training_target_success_rate",
        "minimum_held_out_target_success_rate",
        "maximum_mean_action_jerk",
        "maximum_invalid_action_count",
        "maximum_workspace_violation_count",
        "maximum_joint_limit_violation_count",
    }
    if set(gates) != required_gates or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in gates.values()
    ):
        raise PolicyEvaluationError("evaluation acceptance_gates are incomplete or invalid.")
    return ReachEvaluationProtocol(
        version=version,
        task_id="reach_touch_target",
        skill_name=_required_string(payload, "skill_name"),
        scenarios=tuple(scenarios),
        acceptance_gates=gates,
        source_path=str(config_path),
        source_digest=hashlib.sha256(raw).hexdigest(),
    )


def evaluate_policy(
    policy: RolloutPolicy,
    protocol: ReachEvaluationProtocol,
    *,
    output_dir: str | Path,
    backend_factory: Callable[[], ReachRolloutBackend] | None = None,
    model_path: str | Path = DEFAULT_TASK_BOARD_MODEL,
    sim_steps_per_action: int = 17,
    ablation_name: str | None = None,
    base_workspace_min: Sequence[float] | None = None,
    base_workspace_max: Sequence[float] | None = None,
) -> EvaluationReport:
    """Run and save every scenario in the frozen reach matrix."""

    if sim_steps_per_action <= 0:
        raise PolicyEvaluationError("sim_steps_per_action must be positive.")
    full_action = policy.output_action_names == policy.dataset_action_names
    if not full_action and not ablation_name:
        raise PolicyEvaluationError(
            "an action-subset checkpoint requires an explicit non-empty ablation name."
        )
    if full_action and ablation_name:
        raise PolicyEvaluationError("ablation_name is only valid for an action-subset policy.")
    action_mode = "full_level1.13" if full_action else f"ablation:{ablation_name}"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_dir = destination / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    default_workspace = WorkspaceLimits()
    workspace_min = np.asarray(
        default_workspace.minimum if base_workspace_min is None else base_workspace_min,
        dtype=np.float64,
    )
    workspace_max = np.asarray(
        default_workspace.maximum if base_workspace_max is None else base_workspace_max,
        dtype=np.float64,
    )
    if (
        workspace_min.shape != (3,)
        or workspace_max.shape != (3,)
        or not np.all(np.isfinite(workspace_min))
        or not np.all(np.isfinite(workspace_max))
        or np.any(workspace_min >= workspace_max)
    ):
        raise PolicyEvaluationError("base workspace bounds must be finite ordered vectors.")
    factory = backend_factory or (
        lambda: MujocoReachRolloutBackend(
            model_path,
            base_workspace_min=workspace_min,
            base_workspace_max=workspace_max,
        )
    )
    rollout_config = {
        "model_path": str(model_path),
        "sim_steps_per_action": sim_steps_per_action,
        "scenario_count": len(protocol.scenarios),
        "action_mode": action_mode,
        "action_subset_fill_strategy": (
            "not_applicable" if full_action else "hold_previous_applied_action"
        ),
        "goal_input_mode": str(
            getattr(policy, "goal_input_mode", "conditioned")
        ),
        "base_workspace_min": workspace_min.tolist(),
        "base_workspace_max": workspace_max.tolist(),
    }
    results: list[RolloutResult] = []
    for scenario in protocol.scenarios:
        backend = factory()
        try:
            result = _run_scenario(
                policy,
                backend,
                scenario,
                trajectory_dir=trajectory_dir,
                sim_steps_per_action=sim_steps_per_action,
            )
        finally:
            backend.close()
        results.append(result)

    metrics, gate_results = _summarize(results, protocol.acceptance_gates)
    report = EvaluationReport(
        report_version="dexvision/reach-rollout-report-v1",
        protocol_version=protocol.version,
        protocol_digest=protocol.source_digest,
        rollout_config=rollout_config,
        rollout_config_digest=_canonical_digest(rollout_config),
        checkpoint_digest=policy.checkpoint_digest,
        dataset_digest=policy.dataset_digest,
        split_manifest_digest=str(
            getattr(policy, "split_manifest_digest", "unavailable")
        ),
        training_config_digest=str(
            getattr(policy, "experiment_config_digest", "unavailable")
        ),
        schema_digest=str(getattr(policy, "schema_digest", "unavailable")),
        selected_epoch=int(getattr(policy, "selected_epoch", 0)),
        selected_validation_loss=float(
            getattr(policy, "selected_validation_loss", float("nan"))
        ),
        action_mode=action_mode,
        goal_input_mode=str(getattr(policy, "goal_input_mode", "conditioned")),
        results=tuple(results),
        metrics=metrics,
        acceptance_gates=protocol.acceptance_gates,
        gate_results=gate_results,
        passed=all(gate_results.values()),
    )
    report_path = destination / "report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _run_scenario(
    policy: RolloutPolicy,
    backend: ReachRolloutBackend,
    scenario: ReachScenario,
    *,
    trajectory_dir: Path,
    sim_steps_per_action: int,
) -> RolloutResult:
    if policy.observation_schema_version != backend.observation_schema_version:
        raise PolicyEvaluationError(
            "policy and rollout backend observation schema versions differ."
        )
    if policy.action_schema_version != backend.action_schema_version:
        raise PolicyEvaluationError(
            "policy and rollout backend action schema versions differ."
        )
    action_names = policy.dataset_action_names
    bounds = backend.action_bounds(action_names)
    _validate_bounds(bounds, len(action_names))
    state = backend.reset(scenario)
    applied = backend.initial_action(action_names)
    if applied.shape != (len(action_names),) or not np.all(np.isfinite(applied)):
        raise PolicyEvaluationError("backend initial action is invalid.")
    previous = dict(zip(action_names, applied, strict=True))
    observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    distances: list[float] = [state.distance_to_target]
    invalid_count = 0
    workspace_count = 0
    joint_count = 0
    terminal_reason = "timeout"

    while state.step_count < backend.max_episode_steps:
        observation = backend.observation(
            policy.observation_names, previous_action=previous
        )
        goal = _goal_vector(policy.goal_names, scenario.target_position)
        prediction = np.asarray(policy.predict(observation, goal), dtype=np.float64)
        full_prediction = applied.copy()
        if prediction.shape != (len(policy.output_action_names),):
            raise PolicyEvaluationError(
                "policy prediction shape does not match its declared output layout."
            )
        output_indices = [action_names.index(name) for name in policy.output_action_names]
        full_prediction[output_indices] = prediction
        observations.append(observation.copy())
        raw_actions.append(full_prediction.copy())
        if not np.all(np.isfinite(full_prediction)):
            invalid_count += int(np.count_nonzero(~np.isfinite(full_prediction)))
            terminal_reason = "invalid_action"
            break

        workspace_violations = np.logical_or(
            full_prediction[list(bounds.workspace_indices)]
            < bounds.lower[list(bounds.workspace_indices)],
            full_prediction[list(bounds.workspace_indices)]
            > bounds.upper[list(bounds.workspace_indices)],
        )
        joint_violations = np.logical_or(
            full_prediction[list(bounds.joint_indices)]
            < bounds.lower[list(bounds.joint_indices)],
            full_prediction[list(bounds.joint_indices)]
            > bounds.upper[list(bounds.joint_indices)],
        )
        workspace_count += int(np.count_nonzero(workspace_violations))
        joint_count += int(np.count_nonzero(joint_violations))
        applied = np.clip(full_prediction, bounds.lower, bounds.upper)
        orientation_slice = slice(3, 7)
        quaternion = applied[orientation_slice]
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1e-12:
            invalid_count += 1
            terminal_reason = "invalid_action"
            break
        applied[orientation_slice] = quaternion / norm
        applied_actions.append(applied.copy())
        previous = dict(zip(action_names, applied, strict=True))
        state = backend.step(
            action_names, applied, n_steps=sim_steps_per_action
        )
        distances.append(state.distance_to_target)
        if workspace_count:
            terminal_reason = "workspace_violation"
            break
        if joint_count:
            terminal_reason = "joint_limit_violation"
            break
        if state.success:
            terminal_reason = "success"
            break
        if state.failure_reason is not None:
            terminal_reason = state.failure_reason
            break

    if state.step_count >= backend.max_episode_steps and not state.success:
        terminal_reason = state.failure_reason or "timeout"
    raw_matrix = _rows(raw_actions, len(action_names))
    applied_matrix = _rows(applied_actions, len(action_names))
    normalized = (
        (applied_matrix - bounds.lower) / (bounds.upper - bounds.lower)
        if applied_matrix.size
        else applied_matrix
    )
    jerk = mean_action_jerk(normalized) if applied_matrix.size else 0.0
    trajectory_path = trajectory_dir / f"{scenario.scenario_id}.npz"
    np.savez_compressed(
        trajectory_path,
        observations=_rows(observations, len(policy.observation_names)),
        raw_actions=raw_matrix,
        applied_actions=applied_matrix,
        distances_m=np.asarray(distances, dtype=np.float64),
        action_names=np.asarray(action_names),
        observation_names=np.asarray(policy.observation_names),
        target_position=np.asarray(scenario.target_position, dtype=np.float64),
        initial_base_offset_m=np.asarray(scenario.initial_base_offset_m, dtype=np.float64),
        terminal_reason=np.asarray(terminal_reason),
    )
    return RolloutResult(
        scenario_id=scenario.scenario_id,
        target_group=scenario.target_group,
        target_id=scenario.target_id,
        offset_id=scenario.offset_id,
        repetition=scenario.repetition,
        success=terminal_reason == "success",
        terminal_reason=terminal_reason,
        completion_steps=state.step_count,
        final_distance_m=state.distance_to_target,
        normalized_action_jerk=jerk,
        invalid_action_count=invalid_count,
        workspace_violation_count=workspace_count,
        joint_limit_violation_count=joint_count,
        trajectory_file=str(trajectory_path),
    )


class MujocoReachRolloutBackend:
    """Headless adapter from named Level 3 vectors to the reach MuJoCo task."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TASK_BOARD_MODEL,
        *,
        base_workspace_min: Sequence[float] | None = None,
        base_workspace_max: Sequence[float] | None = None,
    ) -> None:
        self.task = ReachTouchTargetTask(model_path)
        self.max_episode_steps = self.task.config.max_episode_steps
        self.observation_schema_version = self.task.spec.observation_schema.version
        self.action_schema_version = self.task.spec.action_schema.version
        self._target = np.zeros(3, dtype=np.float64)
        self._qpos_names, self._qvel_names = _mujoco_state_names(self.task)
        defaults = WorkspaceLimits()
        self._base_workspace_min = np.asarray(
            defaults.minimum if base_workspace_min is None else base_workspace_min,
            dtype=np.float64,
        )
        self._base_workspace_max = np.asarray(
            defaults.maximum if base_workspace_max is None else base_workspace_max,
            dtype=np.float64,
        )

    def reset(self, scenario: ReachScenario) -> BackendState:
        self.task.reset(
            parameters=ReachTouchTargetParameters(target_pose=scenario.target_position)
        )
        offset = np.asarray(scenario.initial_base_offset_m, dtype=np.float64)
        base_position, base_orientation = self.task.env.get_mocap_pose(
            self.task.config.base_target_body
        )
        self.task.env.set_mocap_pose(
            self.task.config.base_target_body,
            position=base_position + offset,
            orientation_quat=base_orientation,
        )
        joint_id = self.task.env._mujoco.mj_name2id(
            self.task.env.model,
            self.task.env._mujoco.mjtObj.mjOBJ_JOINT,
            "rh_base_freejoint",
        )
        if joint_id < 0:
            raise PolicyEvaluationError("reach model is missing 'rh_base_freejoint'.")
        address = int(self.task.env.model.jnt_qposadr[joint_id])
        dof_address = int(self.task.env.model.jnt_dofadr[joint_id])
        self.task.env.data.qpos[address : address + 3] += offset
        self.task.env.data.qvel[dof_address : dof_address + 6] = 0.0
        self.task.env._mujoco.mj_forward(self.task.env.model, self.task.env.data)
        self._target = np.asarray(scenario.target_position, dtype=np.float64)
        return self._state(self.task.get_state())

    def observation(
        self,
        names: Sequence[str],
        *,
        previous_action: Mapping[str, float],
    ) -> np.ndarray:
        sim = self.task.env.get_state()
        base_position, base_quaternion = self.task.env.get_mocap_pose(
            self.task.config.base_target_body
        )
        rotation = quaternion_wxyz_to_rotation_6d(base_quaternion[None, :])[0]
        qpos = dict(zip(self._qpos_names, sim.qpos, strict=True))
        qvel = dict(zip(self._qvel_names, sim.qvel, strict=True))
        tracking = {
            "detected": 1.0,
            "handedness": 1.0,
            "handedness_code": 1.0,
            "tracking_confidence": 1.0,
            "hand_tracking_confidence": 1.0,
            "feature_confidence": 1.0,
            "dropped_frame": 0.0,
            "reacquired": 0.0,
        }
        values: list[float] = []
        for name in names:
            prefix, separator, field = name.partition("/")
            if not separator:
                raise PolicyEvaluationError(f"unsupported observation field {name!r}.")
            if prefix == "robot_qpos":
                value = qpos.get(field)
            elif prefix == "robot_qvel":
                value = qvel.get(field)
            elif prefix == "finger_joint_positions":
                value = qpos.get(field)
            elif prefix == "finger_joint_velocities":
                value = qvel.get(field)
            elif prefix == "base_position":
                value = dict(zip(("x", "y", "z"), base_position, strict=True)).get(field)
            elif prefix == "base_orientation":
                rotation_names = (
                    "rotation_col0/x", "rotation_col0/y", "rotation_col0/z",
                    "rotation_col1/x", "rotation_col1/y", "rotation_col1/z",
                )
                value = dict(zip(rotation_names, rotation, strict=True)).get(field)
            elif prefix == "object_state" and field.startswith("object_state["):
                try:
                    value = self._target[int(field[13:-1])]
                except (ValueError, IndexError):
                    value = None
            elif prefix == "tracking_quality":
                value = tracking.get(field)
            elif prefix == "previous_action":
                value = previous_action.get(field)
            else:
                value = None
            if value is None:
                raise PolicyEvaluationError(
                    f"policy observation {name!r} is unavailable from the reach backend."
                )
            values.append(float(value))
        result = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise PolicyEvaluationError("reach backend produced a non-finite observation.")
        return result

    def initial_action(self, action_names: Sequence[str]) -> np.ndarray:
        sim = self.task.env.get_state()
        base_position, base_orientation = self.task.env.get_mocap_pose(
            self.task.config.base_target_body
        )
        fingers = {
            name: float(sim.ctrl[index])
            for index, name in enumerate(_actuator_names(self.task))
        }
        return _assemble_named_action(action_names, base_position, base_orientation, fingers)

    def action_bounds(self, action_names: Sequence[str]) -> ActionBounds:
        if tuple(action_names[:7]) != BASE_POSITION_ACTION_NAMES + BASE_ORIENTATION_ACTION_NAMES:
            raise PolicyEvaluationError("policy action layout is not full Level 1.13 order.")
        actuators = _actuator_names(self.task)
        expected = tuple(f"{FINGER_ACTION_PREFIX}{name}" for name in actuators)
        if tuple(action_names[7:]) != expected:
            raise PolicyEvaluationError("policy finger action layout does not match the model.")
        lower = np.concatenate(
            (
                self._base_workspace_min,
                np.full(4, -1.0),
                np.asarray(self.task.env.model.actuator_ctrlrange[:, 0], dtype=np.float64),
            )
        )
        upper = np.concatenate(
            (
                self._base_workspace_max,
                np.full(4, 1.0),
                np.asarray(self.task.env.model.actuator_ctrlrange[:, 1], dtype=np.float64),
            )
        )
        return ActionBounds(
            lower=lower,
            upper=upper,
            workspace_indices=(0, 1, 2),
            joint_indices=tuple(range(7, len(action_names))),
        )

    def step(
        self,
        action_names: Sequence[str],
        action: np.ndarray,
        *,
        n_steps: int,
    ) -> BackendState:
        values = dict(zip(action_names, action, strict=True))
        self.task.env.set_mocap_pose(
            self.task.config.base_target_body,
            position=[values[name] for name in BASE_POSITION_ACTION_NAMES],
            orientation_quat=[values[name] for name in BASE_ORIENTATION_ACTION_NAMES],
        )
        self.task.env.set_joint_targets(
            {
                name.removeprefix(FINGER_ACTION_PREFIX): value
                for name, value in values.items()
                if name.startswith(FINGER_ACTION_PREFIX)
            }
        )
        return self._state(self.task.step(n_steps=n_steps))

    def close(self) -> None:
        self.task.close()

    @staticmethod
    def _state(state: Any) -> BackendState:
        return BackendState(
            success=bool(state.success),
            failure_reason=state.failure_reason,
            distance_to_target=float(state.distance_to_target),
            step_count=int(state.step_count),
        )


def _summarize(
    results: Sequence[RolloutResult], gates: Mapping[str, float | int]
) -> tuple[dict[str, Any], dict[str, bool]]:
    training = [result for result in results if result.target_group == "training"]
    held_out = [result for result in results if result.target_group == "held_out"]
    if not training or not held_out:
        raise PolicyEvaluationError("frozen evaluation requires training and held-out runs.")
    training_rate = sum(result.success for result in training) / len(training)
    held_out_rate = sum(result.success for result in held_out) / len(held_out)
    mean_jerk = float(np.mean([result.normalized_action_jerk for result in results]))
    invalid = sum(result.invalid_action_count for result in results)
    workspace = sum(result.workspace_violation_count for result in results)
    joint = sum(result.joint_limit_violation_count for result in results)
    metrics = {
        "scenario_count": len(results),
        "training_target_success_rate": training_rate,
        "held_out_target_success_rate": held_out_rate,
        "mean_final_distance_m": float(np.mean([result.final_distance_m for result in results])),
        "mean_completion_steps": float(np.mean([result.completion_steps for result in results])),
        "mean_normalized_action_jerk": mean_jerk,
        "invalid_action_count": invalid,
        "workspace_violation_count": workspace,
        "joint_limit_violation_count": joint,
        "terminal_reason_distribution": dict(sorted(Counter(result.terminal_reason for result in results).items())),
    }
    gate_results = {
        "training_target_success_rate": training_rate >= float(gates["minimum_training_target_success_rate"]),
        "held_out_target_success_rate": held_out_rate >= float(gates["minimum_held_out_target_success_rate"]),
        "mean_normalized_action_jerk": mean_jerk <= float(gates["maximum_mean_action_jerk"]),
        "invalid_action_count": invalid <= int(gates["maximum_invalid_action_count"]),
        "workspace_violation_count": workspace <= int(gates["maximum_workspace_violation_count"]),
        "joint_limit_violation_count": joint <= int(gates["maximum_joint_limit_violation_count"]),
        "explicit_terminal_reasons": all(bool(result.terminal_reason) for result in results),
    }
    return metrics, gate_results


def _goal_vector(names: Sequence[str], target: Sequence[float]) -> np.ndarray:
    mapping = dict(zip(("target_position/x", "target_position/y", "target_position/z"), target, strict=True))
    try:
        return np.asarray([mapping[name] for name in names], dtype=np.float64)
    except KeyError as exc:
        raise PolicyEvaluationError(f"unsupported reach policy goal field {exc.args[0]!r}.") from exc


def _assemble_named_action(
    names: Sequence[str],
    position: np.ndarray,
    orientation: np.ndarray,
    fingers: Mapping[str, float],
) -> np.ndarray:
    values = dict(zip(BASE_POSITION_ACTION_NAMES, position, strict=True))
    values.update(zip(BASE_ORIENTATION_ACTION_NAMES, orientation, strict=True))
    values.update({f"{FINGER_ACTION_PREFIX}{name}": value for name, value in fingers.items()})
    try:
        return np.asarray([values[name] for name in names], dtype=np.float64)
    except KeyError as exc:
        raise PolicyEvaluationError(f"unknown policy action field {exc.args[0]!r}.") from exc


def _mujoco_state_names(task: ReachTouchTargetTask) -> tuple[tuple[str, ...], tuple[str, ...]]:
    model = task.env.model
    mujoco = task.env._mujoco
    qpos_names: list[str] = []
    qvel_names: list[str] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise PolicyEvaluationError("reach model contains an unnamed joint.")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            qpos_names.extend(f"{name}/{suffix}" for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz"))
            qvel_names.extend(f"{name}/{suffix}" for suffix in ("vx", "vy", "vz", "wx", "wy", "wz"))
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            qpos_names.extend(f"{name}/{suffix}" for suffix in ("qw", "qx", "qy", "qz"))
            qvel_names.extend(f"{name}/{suffix}" for suffix in ("wx", "wy", "wz"))
        else:
            qpos_names.append(str(name))
            qvel_names.append(str(name))
    if len(qpos_names) != model.nq or len(qvel_names) != model.nv:
        raise PolicyEvaluationError("cannot reconstruct named MuJoCo state layout.")
    return tuple(qpos_names), tuple(qvel_names)


def _actuator_names(task: ReachTouchTargetTask) -> tuple[str, ...]:
    names = tuple(
        task.env._mujoco.mj_id2name(
            task.env.model, task.env._mujoco.mjtObj.mjOBJ_ACTUATOR, index
        )
        for index in range(task.env.model.nu)
    )
    if any(name is None for name in names):
        raise PolicyEvaluationError("reach model contains an unnamed actuator.")
    return tuple(str(name) for name in names)


def _validate_bounds(bounds: ActionBounds, size: int) -> None:
    if bounds.lower.shape != (size,) or bounds.upper.shape != (size,):
        raise PolicyEvaluationError("backend action bounds have the wrong shape.")
    if not np.all(np.isfinite(bounds.lower)) or not np.all(np.isfinite(bounds.upper)):
        raise PolicyEvaluationError("backend action bounds must be finite.")
    if np.any(bounds.lower >= bounds.upper):
        raise PolicyEvaluationError("backend lower action bounds must be below upper bounds.")


def _rows(rows: Sequence[np.ndarray], width: int) -> np.ndarray:
    return np.stack(rows) if rows else np.empty((0, width), dtype=np.float64)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise PolicyEvaluationError(f"evaluation config {key!r} must be a mapping.")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PolicyEvaluationError(f"evaluation config {key!r} must be a non-empty string.")
    return value


def _vector3(value: object, *, label: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyEvaluationError(f"{label} must be numeric.") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise PolicyEvaluationError(f"{label} must be a finite length-3 vector.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class ManipulationScenario:
    """One frozen button or push goal and reset perturbation."""

    scenario_id: str
    goal_group: str
    goal_id: str
    goal_parameters: dict[str, Any]
    offset_id: str
    initial_position_offset_m: tuple[float, float, float]
    repetition: int


@dataclass(frozen=True)
class ManipulationEvaluationProtocol:
    """Validated frozen button/push evaluation matrix."""

    version: str
    task_id: str
    skill_name: str
    scenarios: tuple[ManipulationScenario, ...]
    acceptance_gates: dict[str, float | int]
    task_geometry: dict[str, Any]
    sim_steps_per_action: int
    source_path: str
    source_digest: str


@dataclass(frozen=True)
class ManipulationBackendState:
    """Terminal state and scalar task metrics from a manipulation backend."""

    success: bool
    failure_reason: str | None
    task_error: float
    task_value: float
    step_count: int
    object_workspace_violation: bool = False


class ManipulationRolloutBackend(Protocol):
    """Backend boundary used by button/push CPU-only rollout tests."""

    max_episode_steps: int
    observation_schema_version: str
    action_schema_version: str

    def reset(self, scenario: ManipulationScenario) -> ManipulationBackendState: ...

    def observation(
        self,
        names: Sequence[str],
        *,
        previous_action: Mapping[str, float],
    ) -> np.ndarray: ...

    def initial_action(self, action_names: Sequence[str]) -> np.ndarray: ...

    def action_bounds(self, action_names: Sequence[str]) -> ActionBounds: ...

    def step(
        self,
        action_names: Sequence[str],
        action: np.ndarray,
        *,
        n_steps: int,
    ) -> ManipulationBackendState: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ManipulationRolloutResult:
    """Saved metrics for one frozen button or push rollout."""

    scenario_id: str
    goal_group: str
    goal_id: str
    offset_id: str
    repetition: int
    success: bool
    terminal_reason: str
    completion_steps: int
    final_task_error: float
    final_task_value: float
    normalized_action_jerk: float
    invalid_action_count: int
    workspace_violation_count: int
    object_workspace_violation_count: int
    joint_limit_violation_count: int
    trajectory_file: str


@dataclass(frozen=True)
class ManipulationEvaluationReport:
    """Cross-task report with offline selection and rollout provenance."""

    report_version: str
    task_id: str
    protocol_version: str
    protocol_digest: str
    rollout_config: dict[str, Any]
    rollout_config_digest: str
    checkpoint_digest: str
    dataset_digest: str
    split_manifest_digest: str
    training_config_digest: str
    schema_digest: str
    selected_epoch: int
    selected_validation_loss: float
    action_mode: str
    goal_input_mode: str
    results: tuple[ManipulationRolloutResult, ...]
    metrics: dict[str, Any]
    acceptance_gates: dict[str, float | int]
    gate_results: dict[str, bool]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(result) for result in self.results]
        return payload


def load_manipulation_evaluation_protocol(
    path: str | Path,
) -> ManipulationEvaluationProtocol:
    """Load a frozen Level 3.5A button or push rollout protocol."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw)
    except OSError as exc:
        raise PolicyEvaluationError(
            f"cannot read evaluation config {config_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise PolicyEvaluationError(
            f"invalid evaluation YAML {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PolicyEvaluationError("evaluation config must contain a mapping.")
    version = _required_string(payload, "version")
    task_id = _required_string(payload, "task_id")
    expected_versions = {
        "button_press": "level3/button-evaluation-v1",
        "push_cube_to_target": "level3/push-evaluation-v1",
    }
    if expected_versions.get(task_id) != version:
        raise PolicyEvaluationError(
            f"unsupported manipulation task/version {task_id!r}/{version!r}."
        )
    rollout = _required_mapping(payload, "rollout")
    repetitions = rollout.get("repetitions_per_scenario")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise PolicyEvaluationError("rollout.repetitions_per_scenario must be positive.")
    sim_steps = rollout.get("sim_steps_per_action", 17)
    if isinstance(sim_steps, bool) or not isinstance(sim_steps, int) or sim_steps <= 0:
        raise PolicyEvaluationError("rollout.sim_steps_per_action must be positive.")
    for key in (
        "use_task_success_condition",
        "stop_on_success",
        "stop_on_failure",
        "stop_on_timeout",
    ):
        if rollout.get(key) is not True:
            raise PolicyEvaluationError(f"rollout.{key} must be true.")
    offset_key = (
        "initial_base_position_offsets_m"
        if task_id == "button_press"
        else "initial_state_position_offsets_m"
    )
    offsets = _required_mapping(payload, offset_key)
    scenarios: list[ManipulationScenario] = []
    for group, goals_key in (
        ("training", "training_goals"),
        ("held_out", "held_out_rollout_goals"),
    ):
        goals = _required_mapping(payload, goals_key)
        for goal_id, parameters in goals.items():
            if not isinstance(parameters, Mapping):
                raise PolicyEvaluationError(f"goal {goal_id!r} must be a mapping.")
            for offset_id, raw_offset in offsets.items():
                offset = _vector3(raw_offset, label=f"offset {offset_id!r}")
                for repetition in range(1, repetitions + 1):
                    scenarios.append(
                        ManipulationScenario(
                            scenario_id=(
                                f"{group}__{goal_id}__{offset_id}__r{repetition}"
                            ),
                            goal_group=group,
                            goal_id=str(goal_id),
                            goal_parameters=dict(parameters),
                            offset_id=str(offset_id),
                            initial_position_offset_m=offset,
                            repetition=repetition,
                        )
                    )
    gates = dict(_required_mapping(payload, "acceptance_gates"))
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in gates.values()
    ):
        raise PolicyEvaluationError("evaluation acceptance_gates are invalid.")
    geometry = payload.get("task_geometry", {})
    if not isinstance(geometry, Mapping):
        raise PolicyEvaluationError("task_geometry must be a mapping when present.")
    return ManipulationEvaluationProtocol(
        version=version,
        task_id=task_id,
        skill_name=_required_string(payload, "skill_name"),
        scenarios=tuple(scenarios),
        acceptance_gates=gates,
        task_geometry=dict(geometry),
        sim_steps_per_action=sim_steps,
        source_path=str(config_path),
        source_digest=hashlib.sha256(raw).hexdigest(),
    )


def evaluate_manipulation_policy(
    policy: RolloutPolicy,
    protocol: ManipulationEvaluationProtocol,
    *,
    output_dir: str | Path,
    backend_factory: Callable[[], ManipulationRolloutBackend] | None = None,
    model_path: str | Path = DEFAULT_TASK_BOARD_MODEL,
    sim_steps_per_action: int | None = None,
    ablation_name: str | None = None,
) -> ManipulationEvaluationReport:
    """Evaluate button or push using its frozen Level 3.5A matrix."""

    steps_per_action = protocol.sim_steps_per_action if sim_steps_per_action is None else sim_steps_per_action
    if steps_per_action <= 0:
        raise PolicyEvaluationError("sim_steps_per_action must be positive.")
    full_action = policy.output_action_names == policy.dataset_action_names
    if not full_action and not ablation_name:
        raise PolicyEvaluationError(
            "an action-subset checkpoint requires an explicit non-empty ablation name."
        )
    if full_action and ablation_name:
        raise PolicyEvaluationError("ablation_name is only valid for an action-subset policy.")
    action_mode = "full_level1.13" if full_action else f"ablation:{ablation_name}"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_dir = destination / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    if backend_factory is None:
        if protocol.task_id == "button_press":
            def factory() -> ManipulationRolloutBackend:
                return MujocoButtonRolloutBackend(model_path)
        else:
            def factory() -> ManipulationRolloutBackend:
                return MujocoPushRolloutBackend(model_path)
    else:
        factory = backend_factory
    rollout_config = {
        "model_path": str(model_path),
        "sim_steps_per_action": steps_per_action,
        "scenario_count": len(protocol.scenarios),
        "action_mode": action_mode,
        "action_subset_fill_strategy": (
            "not_applicable" if full_action else "hold_previous_applied_action"
        ),
        "goal_input_mode": str(
            getattr(policy, "goal_input_mode", "conditioned")
        ),
        "task_id": protocol.task_id,
    }
    results: list[ManipulationRolloutResult] = []
    for scenario in protocol.scenarios:
        backend = factory()
        try:
            results.append(
                _run_manipulation_scenario(
                    policy,
                    backend,
                    protocol,
                    scenario,
                    trajectory_dir=trajectory_dir,
                    sim_steps_per_action=steps_per_action,
                )
            )
        finally:
            backend.close()
    metrics, gate_results = _summarize_manipulation(
        results, protocol.task_id, protocol.acceptance_gates
    )
    report = ManipulationEvaluationReport(
        report_version="dexvision/manipulation-rollout-report-v1",
        task_id=protocol.task_id,
        protocol_version=protocol.version,
        protocol_digest=protocol.source_digest,
        rollout_config=rollout_config,
        rollout_config_digest=_canonical_digest(rollout_config),
        checkpoint_digest=policy.checkpoint_digest,
        dataset_digest=policy.dataset_digest,
        split_manifest_digest=str(
            getattr(policy, "split_manifest_digest", "unavailable")
        ),
        training_config_digest=str(
            getattr(policy, "experiment_config_digest", "unavailable")
        ),
        schema_digest=str(getattr(policy, "schema_digest", "unavailable")),
        selected_epoch=int(getattr(policy, "selected_epoch", 0)),
        selected_validation_loss=float(
            getattr(policy, "selected_validation_loss", float("nan"))
        ),
        action_mode=action_mode,
        goal_input_mode=str(getattr(policy, "goal_input_mode", "conditioned")),
        results=tuple(results),
        metrics=metrics,
        acceptance_gates=protocol.acceptance_gates,
        gate_results=gate_results,
        passed=all(gate_results.values()),
    )
    (destination / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _run_manipulation_scenario(
    policy: RolloutPolicy,
    backend: ManipulationRolloutBackend,
    protocol: ManipulationEvaluationProtocol,
    scenario: ManipulationScenario,
    *,
    trajectory_dir: Path,
    sim_steps_per_action: int,
) -> ManipulationRolloutResult:
    if policy.observation_schema_version != backend.observation_schema_version:
        raise PolicyEvaluationError(
            "policy and rollout backend observation schema versions differ."
        )
    if policy.action_schema_version != backend.action_schema_version:
        raise PolicyEvaluationError(
            "policy and rollout backend action schema versions differ."
        )
    action_names = policy.dataset_action_names
    bounds = backend.action_bounds(action_names)
    _validate_bounds(bounds, len(action_names))
    state = backend.reset(scenario)
    applied = backend.initial_action(action_names)
    if applied.shape != (len(action_names),) or not np.all(np.isfinite(applied)):
        raise PolicyEvaluationError("backend initial action is invalid.")
    previous = dict(zip(action_names, applied, strict=True))
    goal = _manipulation_goal_vector(policy.goal_names, protocol, scenario)
    observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    task_errors = [state.task_error]
    task_values = [state.task_value]
    invalid_count = 0
    workspace_count = 0
    object_workspace_count = int(state.object_workspace_violation)
    joint_count = 0
    terminal_reason = "timeout"
    while state.step_count < backend.max_episode_steps:
        observation = backend.observation(
            policy.observation_names, previous_action=previous
        )
        prediction = np.asarray(policy.predict(observation, goal), dtype=np.float64)
        if prediction.shape != (len(policy.output_action_names),):
            raise PolicyEvaluationError(
                "policy prediction shape does not match its declared output layout."
            )
        full_prediction = applied.copy()
        indices = [action_names.index(name) for name in policy.output_action_names]
        full_prediction[indices] = prediction
        observations.append(observation.copy())
        raw_actions.append(full_prediction.copy())
        if not np.all(np.isfinite(full_prediction)):
            invalid_count += int(np.count_nonzero(~np.isfinite(full_prediction)))
            terminal_reason = "invalid_action"
            break
        workspace = np.logical_or(
            full_prediction[list(bounds.workspace_indices)]
            < bounds.lower[list(bounds.workspace_indices)],
            full_prediction[list(bounds.workspace_indices)]
            > bounds.upper[list(bounds.workspace_indices)],
        )
        joints = np.logical_or(
            full_prediction[list(bounds.joint_indices)]
            < bounds.lower[list(bounds.joint_indices)],
            full_prediction[list(bounds.joint_indices)]
            > bounds.upper[list(bounds.joint_indices)],
        )
        workspace_count += int(np.count_nonzero(workspace))
        joint_count += int(np.count_nonzero(joints))
        applied = np.clip(full_prediction, bounds.lower, bounds.upper)
        quaternion = applied[3:7]
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1e-12:
            invalid_count += 1
            terminal_reason = "invalid_action"
            break
        applied[3:7] = quaternion / norm
        applied_actions.append(applied.copy())
        previous = dict(zip(action_names, applied, strict=True))
        state = backend.step(action_names, applied, n_steps=sim_steps_per_action)
        task_errors.append(state.task_error)
        task_values.append(state.task_value)
        object_workspace_count += int(state.object_workspace_violation)
        if workspace_count:
            terminal_reason = "workspace_violation"
            break
        if joint_count:
            terminal_reason = "joint_limit_violation"
            break
        if state.success:
            terminal_reason = "success"
            break
        if state.failure_reason is not None:
            terminal_reason = state.failure_reason
            break
    if state.step_count >= backend.max_episode_steps and not state.success:
        terminal_reason = state.failure_reason or "timeout"
    applied_matrix = _rows(applied_actions, len(action_names))
    normalized = (
        (applied_matrix - bounds.lower) / (bounds.upper - bounds.lower)
        if applied_matrix.size
        else applied_matrix
    )
    jerk = mean_action_jerk(normalized) if applied_matrix.size else 0.0
    trajectory_path = trajectory_dir / f"{scenario.scenario_id}.npz"
    np.savez_compressed(
        trajectory_path,
        observations=_rows(observations, len(policy.observation_names)),
        raw_actions=_rows(raw_actions, len(action_names)),
        applied_actions=applied_matrix,
        task_errors=np.asarray(task_errors, dtype=np.float64),
        task_values=np.asarray(task_values, dtype=np.float64),
        goal=goal,
        goal_names=np.asarray(policy.goal_names),
        action_names=np.asarray(action_names),
        observation_names=np.asarray(policy.observation_names),
        initial_position_offset_m=np.asarray(
            scenario.initial_position_offset_m, dtype=np.float64
        ),
        terminal_reason=np.asarray(terminal_reason),
    )
    return ManipulationRolloutResult(
        scenario_id=scenario.scenario_id,
        goal_group=scenario.goal_group,
        goal_id=scenario.goal_id,
        offset_id=scenario.offset_id,
        repetition=scenario.repetition,
        success=terminal_reason == "success",
        terminal_reason=terminal_reason,
        completion_steps=state.step_count,
        final_task_error=state.task_error,
        final_task_value=state.task_value,
        normalized_action_jerk=jerk,
        invalid_action_count=invalid_count,
        workspace_violation_count=workspace_count,
        object_workspace_violation_count=object_workspace_count,
        joint_limit_violation_count=joint_count,
        trajectory_file=str(trajectory_path),
    )


class _MujocoManipulationBackend:
    """Shared named-vector adapter for Level 3.5B MuJoCo tasks."""

    task: Any

    def _initialize(self) -> None:
        self.max_episode_steps = self.task.config.max_episode_steps
        self.observation_schema_version = self.task.spec.observation_schema.version
        self.action_schema_version = self.task.spec.action_schema.version
        self._qpos_names, self._qvel_names = _mujoco_state_names(self.task)

    def observation(
        self,
        names: Sequence[str],
        *,
        previous_action: Mapping[str, float],
    ) -> np.ndarray:
        sim = self.task.env.get_state()
        base_position, base_quaternion = self.task.env.get_mocap_pose(
            self.task.config.base_target_body
        )
        rotation = quaternion_wxyz_to_rotation_6d(base_quaternion[None, :])[0]
        qpos = dict(zip(self._qpos_names, sim.qpos, strict=True))
        qvel = dict(zip(self._qvel_names, sim.qvel, strict=True))
        object_state = self.task.get_state().as_object_state()
        tracking = {
            "detected": 1.0,
            "handedness": 1.0,
            "handedness_code": 1.0,
            "tracking_confidence": 1.0,
            "hand_tracking_confidence": 1.0,
            "feature_confidence": 1.0,
            "dropped_frame": 0.0,
            "reacquired": 0.0,
        }
        rotation_names = (
            "rotation_col0/x",
            "rotation_col0/y",
            "rotation_col0/z",
            "rotation_col1/x",
            "rotation_col1/y",
            "rotation_col1/z",
        )
        values: list[float] = []
        for name in names:
            prefix, separator, field = name.partition("/")
            value: float | None
            if not separator:
                value = None
            elif prefix in {"robot_qpos", "finger_joint_positions"}:
                value = qpos.get(field)
            elif prefix in {"robot_qvel", "finger_joint_velocities"}:
                value = qvel.get(field)
            elif prefix == "base_position":
                value = dict(zip(("x", "y", "z"), base_position, strict=True)).get(field)
            elif prefix == "base_orientation":
                value = dict(zip(rotation_names, rotation, strict=True)).get(field)
            elif prefix == "object_state" and field.startswith("object_state["):
                try:
                    value = float(object_state[int(field[13:-1])])
                except (ValueError, IndexError):
                    value = None
            elif prefix == "tracking_quality":
                value = tracking.get(field)
            elif prefix == "previous_action":
                value = previous_action.get(field)
            else:
                value = None
            if value is None:
                raise PolicyEvaluationError(
                    f"policy observation {name!r} is unavailable from the "
                    f"{self.task.spec.task_id} backend."
                )
            values.append(float(value))
        result = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise PolicyEvaluationError("backend produced a non-finite observation.")
        return result

    def initial_action(self, action_names: Sequence[str]) -> np.ndarray:
        sim = self.task.env.get_state()
        position, orientation = self.task.env.get_mocap_pose(
            self.task.config.base_target_body
        )
        fingers = {
            name: float(sim.ctrl[index])
            for index, name in enumerate(_actuator_names(self.task))
        }
        return _assemble_named_action(action_names, position, orientation, fingers)

    def action_bounds(self, action_names: Sequence[str]) -> ActionBounds:
        if tuple(action_names[:7]) != BASE_POSITION_ACTION_NAMES + BASE_ORIENTATION_ACTION_NAMES:
            raise PolicyEvaluationError("policy action layout is not full Level 1.13 order.")
        actuators = _actuator_names(self.task)
        expected = tuple(f"{FINGER_ACTION_PREFIX}{name}" for name in actuators)
        if tuple(action_names[7:]) != expected:
            raise PolicyEvaluationError("policy finger action layout does not match the model.")
        defaults = WorkspaceLimits()
        minimum = np.asarray(defaults.minimum, dtype=np.float64)
        maximum = np.asarray(defaults.maximum, dtype=np.float64)
        if self.task.spec.task_id == "push_cube_to_target":
            minimum = np.minimum(minimum, np.asarray([-0.25, -0.18, -0.30]))
        return ActionBounds(
            lower=np.concatenate(
                (minimum, np.full(4, -1.0), self.task.env.model.actuator_ctrlrange[:, 0])
            ),
            upper=np.concatenate(
                (maximum, np.full(4, 1.0), self.task.env.model.actuator_ctrlrange[:, 1])
            ),
            workspace_indices=(0, 1, 2),
            joint_indices=tuple(range(7, len(action_names))),
        )

    def _apply_action(
        self, action_names: Sequence[str], action: np.ndarray, *, n_steps: int
    ) -> Any:
        values = dict(zip(action_names, action, strict=True))
        self.task.env.set_mocap_pose(
            self.task.config.base_target_body,
            position=[values[name] for name in BASE_POSITION_ACTION_NAMES],
            orientation_quat=[values[name] for name in BASE_ORIENTATION_ACTION_NAMES],
        )
        self.task.env.set_joint_targets(
            {
                name.removeprefix(FINGER_ACTION_PREFIX): value
                for name, value in values.items()
                if name.startswith(FINGER_ACTION_PREFIX)
            }
        )
        return self.task.step(n_steps=n_steps)

    def _set_base_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self.task.env.set_mocap_pose(
            self.task.config.base_target_body,
            position=position,
            orientation_quat=orientation,
        )
        joint_id = self.task.env._mujoco.mj_name2id(
            self.task.env.model,
            self.task.env._mujoco.mjtObj.mjOBJ_JOINT,
            "rh_base_freejoint",
        )
        if joint_id < 0:
            raise PolicyEvaluationError("task model is missing 'rh_base_freejoint'.")
        qpos_address = int(self.task.env.model.jnt_qposadr[joint_id])
        qvel_address = int(self.task.env.model.jnt_dofadr[joint_id])
        self.task.env.data.qpos[qpos_address : qpos_address + 3] = position
        self.task.env.data.qpos[qpos_address + 3 : qpos_address + 7] = orientation
        self.task.env.data.qvel[qvel_address : qvel_address + 6] = 0.0
        self.task.env._mujoco.mj_forward(self.task.env.model, self.task.env.data)

    def close(self) -> None:
        self.task.close()


class MujocoButtonRolloutBackend(_MujocoManipulationBackend):
    """Headless named-vector adapter for frozen button scenarios."""

    def __init__(self, model_path: str | Path = DEFAULT_TASK_BOARD_MODEL) -> None:
        self.task = ButtonPressTask(model_path)
        self._initialize()

    def reset(self, scenario: ManipulationScenario) -> ManipulationBackendState:
        goal = scenario.goal_parameters
        self.task.reset(
            parameters=ButtonPressParameters(
                button_id=str(goal["button_id"]),
                target_press_depth=float(goal["target_press_depth"]),
            )
        )
        position, orientation = self.task.env.get_mocap_pose(
            self.task.config.base_target_body
        )
        self._set_base_pose(
            position + np.asarray(scenario.initial_position_offset_m), orientation
        )
        return self._state(self.task.get_state())

    def step(
        self,
        action_names: Sequence[str],
        action: np.ndarray,
        *,
        n_steps: int,
    ) -> ManipulationBackendState:
        return self._state(self._apply_action(action_names, action, n_steps=n_steps))

    @staticmethod
    def _state(state: Any) -> ManipulationBackendState:
        return ManipulationBackendState(
            success=bool(state.success),
            failure_reason=state.failure_reason,
            task_error=max(float(state.target_press_depth - state.press_depth), 0.0),
            task_value=float(state.press_depth),
            step_count=int(state.step_count),
        )


class MujocoPushRolloutBackend(_MujocoManipulationBackend):
    """Headless named-vector adapter for frozen cube-push scenarios."""

    def __init__(self, model_path: str | Path = DEFAULT_TASK_BOARD_MODEL) -> None:
        self.task = PushCubeTask(model_path)
        self._initialize()

    def reset(self, scenario: ManipulationScenario) -> ManipulationBackendState:
        goal = scenario.goal_parameters
        self.task.reset(
            parameters=PushCubeParameters(
                object_id=str(goal["object_id"]),
                target_pose=tuple(float(value) for value in goal["target_position"]),
                approach_side=str(goal["approach_side"]),
            )
        )
        offset = np.asarray(scenario.initial_position_offset_m, dtype=np.float64)
        object_position = np.asarray(goal["initial_object_position"], dtype=np.float64) + offset
        self.task._set_object_state(
            position=object_position,
            orientation=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )
        base_position = np.asarray(
            [
                self.task.config.initial_base_x + offset[0],
                object_position[1],
                self.task.config.initial_base_z,
            ],
            dtype=np.float64,
        )
        orientation = np.asarray(
            self.task.config.initial_base_orientation, dtype=np.float64
        )
        self._set_base_pose(base_position, orientation)
        return self._state(self.task.get_state())

    def step(
        self,
        action_names: Sequence[str],
        action: np.ndarray,
        *,
        n_steps: int,
    ) -> ManipulationBackendState:
        return self._state(self._apply_action(action_names, action, n_steps=n_steps))

    @staticmethod
    def _state(state: Any) -> ManipulationBackendState:
        violation = state.failure_reason == "object_workspace_bounds"
        return ManipulationBackendState(
            success=bool(state.success),
            failure_reason=state.failure_reason,
            task_error=float(state.distance_to_target),
            task_value=float(state.distance_to_target),
            step_count=int(state.step_count),
            object_workspace_violation=violation,
        )


def _manipulation_goal_vector(
    names: Sequence[str],
    protocol: ManipulationEvaluationProtocol,
    scenario: ManipulationScenario,
) -> np.ndarray:
    goal = scenario.goal_parameters
    if protocol.task_id == "button_press":
        button_ids = ("button_left", "button_center", "button_right")
        button_id = str(goal["button_id"])
        if button_id not in button_ids:
            raise PolicyEvaluationError(f"unsupported button id {button_id!r}.")
        position = _vector3(goal["button_position"], label="button position")
        mapping = {
            "button_index": float(button_ids.index(button_id)),
            **dict(zip(("button_position/x", "button_position/y", "button_position/z"), position, strict=True)),
            "target_press_depth": float(goal["target_press_depth"]),
            "target_pressed_state": 1.0,
            "approach_pose_present": 0.0,
            "approach_pose/x": 0.0,
            "approach_pose/y": 0.0,
            "approach_pose/z": 0.0,
        }
    else:
        target = _vector3(goal["target_position"], label="push target")
        initial = np.asarray(goal["initial_object_position"], dtype=np.float64) + np.asarray(
            scenario.initial_position_offset_m, dtype=np.float64
        )
        target_source = str(goal["target_source"])
        target_ids = ("push_target_left", "push_target_center", "push_target_right")
        target_index = target_ids.index(target_source) if target_source in target_ids else -1
        approach = str(goal["approach_side"])
        mapping = {
            "object_index": 0.0,
            "target_index": float(target_index),
            **dict(zip(("target_position/x", "target_position/y", "target_position/z"), target, strict=True)),
            "target_radius": float(protocol.task_geometry["target_radius_m"]),
            **dict(zip(("initial_object_position/x", "initial_object_position/y", "initial_object_position/z"), initial, strict=True)),
            "approach_side/left": float(approach == "left"),
            "approach_side/front": float(approach == "front"),
            "approach_side/right": float(approach == "right"),
        }
    try:
        result = np.asarray([mapping[name] for name in names], dtype=np.float64)
    except KeyError as exc:
        raise PolicyEvaluationError(
            f"unsupported {protocol.task_id} policy goal field {exc.args[0]!r}."
        ) from exc
    if not np.all(np.isfinite(result)):
        raise PolicyEvaluationError("manipulation goal contains non-finite values.")
    return result


def _summarize_manipulation(
    results: Sequence[ManipulationRolloutResult],
    task_id: str,
    gates: Mapping[str, float | int],
) -> tuple[dict[str, Any], dict[str, bool]]:
    training = [result for result in results if result.goal_group == "training"]
    held_out = [result for result in results if result.goal_group == "held_out"]
    if not training or not held_out:
        raise PolicyEvaluationError("frozen evaluation requires training and held-out runs.")
    training_rate = sum(result.success for result in training) / len(training)
    held_out_rate = sum(result.success for result in held_out) / len(held_out)
    mean_error = float(np.mean([result.final_task_error for result in results]))
    mean_jerk = float(np.mean([result.normalized_action_jerk for result in results]))
    invalid = sum(result.invalid_action_count for result in results)
    workspace = sum(result.workspace_violation_count for result in results)
    object_workspace = sum(
        result.object_workspace_violation_count for result in results
    )
    joint = sum(result.joint_limit_violation_count for result in results)
    metrics = {
        "scenario_count": len(results),
        "training_goal_success_rate": training_rate,
        "held_out_goal_success_rate": held_out_rate,
        "mean_final_task_error": mean_error,
        "mean_completion_steps": float(
            np.mean([result.completion_steps for result in results])
        ),
        "mean_normalized_action_jerk": mean_jerk,
        "invalid_action_count": invalid,
        "workspace_violation_count": workspace,
        "object_workspace_violation_count": object_workspace,
        "joint_limit_violation_count": joint,
        "terminal_reason_distribution": dict(
            sorted(Counter(result.terminal_reason for result in results).items())
        ),
    }
    if task_id == "button_press":
        metrics["mean_final_press_depth_shortfall_m"] = mean_error
        error_gate = "maximum_mean_final_press_depth_shortfall_m"
        workspace_gate = "maximum_workspace_violation_count"
        workspace_value = workspace
    else:
        metrics["mean_final_planar_distance_m"] = mean_error
        error_gate = "maximum_mean_final_planar_distance_m"
        workspace_gate = "maximum_object_workspace_violation_count"
        workspace_value = object_workspace
    gate_results = {
        "training_goal_success_rate": training_rate
        >= float(gates["minimum_training_goal_success_rate"]),
        "held_out_goal_success_rate": held_out_rate
        >= float(gates["minimum_held_out_goal_success_rate"]),
        "mean_final_task_error": mean_error <= float(gates[error_gate]),
        "mean_normalized_action_jerk": mean_jerk
        <= float(gates["maximum_mean_action_jerk"]),
        "invalid_action_count": invalid
        <= int(gates["maximum_invalid_action_count"]),
        "workspace_violation_count": workspace_value
        <= int(gates[workspace_gate]),
        "joint_limit_violation_count": joint
        <= int(gates["maximum_joint_limit_violation_count"]),
        "explicit_terminal_reasons": all(bool(result.terminal_reason) for result in results),
    }
    return metrics, gate_results


def save_reach_v1_v2_comparison(
    *,
    v1_report_path: str | Path,
    v1_checkpoint_path: str | Path,
    v2_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare corrected reach against immutable Level 3.4 evidence."""

    import torch

    paths = (Path(v1_report_path), Path(v2_report_path))
    try:
        v1, v2 = (
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyEvaluationError(f"cannot load reach comparison report: {exc}") from exc
    if not isinstance(v1, Mapping) or not isinstance(v2, Mapping):
        raise PolicyEvaluationError("reach comparison reports must contain mappings.")
    if v1.get("protocol_digest") != v2.get("protocol_digest"):
        raise PolicyEvaluationError("v1 and v2 reach reports use different protocols.")
    v1_results = v1.get("results")
    v2_results = v2.get("results")
    if not isinstance(v1_results, list) or not isinstance(v2_results, list):
        raise PolicyEvaluationError("reach comparison reports are missing results.")
    v1_ids = [item.get("scenario_id") for item in v1_results if isinstance(item, Mapping)]
    v2_ids = [item.get("scenario_id") for item in v2_results if isinstance(item, Mapping)]
    if v1_ids != v2_ids:
        raise PolicyEvaluationError("v1 and v2 reach reports do not use identical scenarios.")
    checkpoint_path = Path(v1_checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise PolicyEvaluationError(f"cannot load v1 reach checkpoint: {exc}") from exc
    history = checkpoint.get("loss_history") if isinstance(checkpoint, Mapping) else None
    completed = checkpoint.get("completed_epochs") if isinstance(checkpoint, Mapping) else None
    if not isinstance(history, list) or not isinstance(completed, int) or not history:
        raise PolicyEvaluationError("v1 reach checkpoint is missing its loss history.")
    final_entry = history[-1]
    if not isinstance(final_entry, Mapping) or final_entry.get("epoch") != completed:
        raise PolicyEvaluationError("v1 reach checkpoint loss history is inconsistent.")
    v1_metrics = v1.get("metrics")
    v2_metrics = v2.get("metrics")
    if not isinstance(v1_metrics, Mapping) or not isinstance(v2_metrics, Mapping):
        raise PolicyEvaluationError("reach comparison reports are missing metrics.")
    compared_metrics = (
        "training_target_success_rate",
        "held_out_target_success_rate",
        "mean_final_distance_m",
        "mean_normalized_action_jerk",
        "invalid_action_count",
        "workspace_violation_count",
        "joint_limit_violation_count",
    )
    deltas = {
        name: float(v2_metrics[name]) - float(v1_metrics[name])
        for name in compared_metrics
    }
    payload = {
        "version": "level3/reach-v1-v2-comparison-v1",
        "identical_frozen_scenarios": True,
        "protocol_version": v2.get("protocol_version"),
        "protocol_digest": v2.get("protocol_digest"),
        "dataset_digest_matches": v1.get("dataset_digest") == v2.get("dataset_digest"),
        "v1": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_digest": v1.get("checkpoint_digest"),
            "selected_epoch": completed,
            "selected_validation_loss": float(final_entry["validation_loss"]),
            "metrics": dict(v1_metrics),
            "passed": bool(v1.get("passed")),
        },
        "v2": {
            "checkpoint_digest": v2.get("checkpoint_digest"),
            "selected_epoch": v2.get("selected_epoch"),
            "selected_validation_loss": v2.get("selected_validation_loss"),
            "split_manifest_digest": v2.get("split_manifest_digest"),
            "training_config_digest": v2.get("training_config_digest"),
            "schema_digest": v2.get("schema_digest"),
            "metrics": dict(v2_metrics),
            "passed": bool(v2.get("passed")),
        },
        "metric_delta_v2_minus_v1": deltas,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
