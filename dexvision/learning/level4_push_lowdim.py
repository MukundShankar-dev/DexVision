"""Frozen state-only low-dimensional learning probe for Level 4 push."""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dexvision.features.hand_features import no_hand_features
from dexvision.learning.level4_lowdim import (
    DEFAULT_LEVEL4_DATASET_CONFIG,
    DEFAULT_RETARGETER_CONFIG,
    DEFAULT_WORKCELL_CONFIG,
    ButtonLearningError,
    LowDimDeltaMLP,
    LowDimDeltaPolicy,
    TaskLocalDeltaActionAdapter,
    _apply_normalization,
    _base_velocity,
    _mapping,
    _normalization,
    _relative_quaternion,
    phase_balancing_weights,
)
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.retargeting.curl_retargeter import CurlRetargeter
from dexvision.sim.level4_expert import (
    DeterministicPushConfig,
    DeterministicPushExpert,
    RequestedAction,
    _bounded_quaternion_step,
)
from dexvision.sim.world_state import WorldState


PUSH_PILOT_VERSION = "level4/push-learning-pilot-v1"
PUSH_OBSERVATION_VERSION = "level4/push-lowdim-observation-v1"
LOWDIM_ACTION_VERSION = "level4/task-local-xyz-delta-v1"
LOWDIM_OBSERVATION_CONVENTION = "level4/causal-state-phase-previous-delta-v1"
PUSH_PHASES = ("approach", "push_contact", "settle", "retract")
PUSH_FAMILIES = ("cuboid", "flat_puck")
PUSH_DELTA_NAMES = ("dx", "dy", "dz")
PUSH_OBSERVATION_NAMES = (
    "ee_to_object_task/x",
    "ee_to_object_task/y",
    "ee_to_object_task/z",
    "object_to_target_task/x",
    "object_to_target_task/y",
    "object_to_target_task/z",
    "ee_to_target_orientation/qw",
    "ee_to_target_orientation/qx",
    "ee_to_target_orientation/qy",
    "ee_to_target_orientation/qz",
    "object_linear_velocity_task/x",
    "object_linear_velocity_task/y",
    "object_linear_velocity_task/z",
    "object_angular_velocity_task/x",
    "object_angular_velocity_task/y",
    "object_angular_velocity_task/z",
    "base_linear_velocity_task/x",
    "base_linear_velocity_task/y",
    "base_linear_velocity_task/z",
    "base_angular_velocity_task/x",
    "base_angular_velocity_task/y",
    "base_angular_velocity_task/z",
    "phase/approach",
    "phase/push_contact",
    "phase/settle",
    "phase/retract",
    "previous_applied_delta/dx",
    "previous_applied_delta/dy",
    "previous_applied_delta/dz",
    "family/cuboid",
    "family/flat_puck",
    "goal/target_stop_distance",
)
DEFAULT_PUSH_PILOT_CONFIG = Path("configs/level4_push_learning_pilot.yaml")


class PushLearningError(ButtonLearningError):
    """Raised when the frozen push learning protocol cannot be reproduced."""


@dataclass(frozen=True)
class PushEpisodeSpec:
    """One whole-session-owned scripted trajectory in the 20-episode set."""

    episode_id: str
    session_id: str
    split: str
    coverage_cell: str
    seed: int


@dataclass(frozen=True)
class PushTrajectory:
    """Causal state observations and task-local expert deltas for one run."""

    spec: PushEpisodeSpec
    observations: np.ndarray
    actions: np.ndarray
    phases: tuple[str, ...]
    success: bool
    terminal_reason: str


@dataclass(frozen=True)
class PushTrainingResult:
    """Single-recipe training result selected only on validation loss."""

    policy: LowDimDeltaPolicy
    selected_epoch: int
    training_loss: float
    validation_loss: float
    test_loss: float
    sample_counts: Mapping[str, int]
    phase_counts: Mapping[str, int]


class PushActionAdapter(TaskLocalDeltaActionAdapter):
    """Reuse the XYZ adapter while enforcing the qualified push constraint."""

    def __init__(
        self,
        *,
        finger_targets: Mapping[str, float],
        initial_orientation_wxyz: Sequence[float],
        target_orientation_wxyz: Sequence[float],
        task_to_world_rotation: Sequence[Sequence[float]],
        workspace_min_m: Sequence[float],
        workspace_max_m: Sequence[float],
        maximum_absolute_delta_by_phase_m: Mapping[str, Sequence[float]],
        transit_height_m: float,
        orientation_step_rad: float,
    ) -> None:
        super().__init__(
            finger_targets=finger_targets,
            fixed_orientation_wxyz=initial_orientation_wxyz,
            workspace_min_m=workspace_min_m,
            workspace_max_m=workspace_max_m,
            maximum_absolute_delta_by_phase_m=maximum_absolute_delta_by_phase_m,
            task_to_world_rotation=task_to_world_rotation,
            phases=PUSH_PHASES,
        )
        self.current_orientation = self.orientation.copy()
        self.target_orientation = _unit_quaternion(target_orientation_wxyz)
        self.transit_height_m = float(transit_height_m)
        self.orientation_step_rad = float(orientation_step_rad)
        if self.transit_height_m <= 0.0 or self.orientation_step_rad <= 0.0:
            raise PushLearningError("push orientation transition parameters must be positive.")

    def expand(
        self,
        previous_position: Sequence[float],
        task_local_delta: Sequence[float],
        *,
        phase: str,
        nominal_orientation_wxyz: Sequence[float] | None = None,
    ) -> tuple[RequestedAction, bool]:
        delta = constrain_push_delta(
            task_local_delta, phase=phase, phase_limits=self.phase_delta_limits
        )
        previous = np.asarray(previous_position, dtype=np.float64)
        if nominal_orientation_wxyz is not None:
            self.current_orientation = _unit_quaternion(nominal_orientation_wxyz)
        elif previous[2] >= self.transit_height_m - 1e-6:
            self.current_orientation = _bounded_quaternion_step(
                self.current_orientation,
                self.target_orientation,
                self.orientation_step_rad,
            )
        return super().expand(
            previous,
            delta,
            phase=phase,
            orientation_wxyz=self.current_orientation,
        )


def constrain_push_delta(
    task_local_delta: Sequence[float],
    *,
    phase: str,
    phase_limits: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Apply the deterministic contact-axis and magnitude residual bounds."""

    delta = np.asarray(task_local_delta, dtype=np.float64).copy()
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise PushLearningError("push task-local delta must be finite XYZ.")
    if phase not in phase_limits:
        raise PushLearningError(f"unsupported push phase {phase!r}.")
    limit = _nonnegative_vector3(phase_limits[phase], f"{phase} push delta limit")
    delta = np.clip(delta, -limit, limit)
    if phase == "push_contact":
        delta[:] = (max(0.0, float(delta[0])), 0.0, 0.0)
    elif phase == "settle":
        delta[:] = 0.0
    elif phase == "retract":
        delta[:] = (min(0.0, float(delta[0])), 0.0, 0.0)
    return delta


def load_push_learning_config(
    path: str | Path = DEFAULT_PUSH_PILOT_CONFIG,
) -> tuple[dict[str, Any], str]:
    """Load and strictly validate the one frozen Level 4.3H recipe."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw)
    except OSError as exc:
        raise PushLearningError(f"cannot read push pilot config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PushLearningError(f"invalid push pilot YAML {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PushLearningError("push pilot config must contain a mapping.")
    config = dict(payload)
    if config.get("version") != PUSH_PILOT_VERSION:
        raise PushLearningError(f"push pilot version must be {PUSH_PILOT_VERSION!r}.")
    interface = _mapping(config, "interface")
    if (
        interface.get("observation_convention") != LOWDIM_OBSERVATION_CONVENTION
        or interface.get("action_schema") != LOWDIM_ACTION_VERSION
        or interface.get("normalization") != "training-split-population-statistics"
        or interface.get("control_sim_steps") != 17
    ):
        raise PushLearningError("push pilot drifted from the qualified low-dimensional interface.")
    specs = push_episode_specs(config)
    if _mapping(config, "dataset").get("successful_episodes") != 20 or len(specs) != 20:
        raise PushLearningError("Level 4.3H must freeze exactly 20 expert successes.")
    if Counter(spec.split for spec in specs) != {"train": 14, "validation": 3, "test": 3}:
        raise PushLearningError("push pilot split counts must be 14/3/3.")
    if any(
        len({item.split for item in specs if item.session_id == session_id}) != 1
        for session_id in {item.session_id for item in specs}
    ):
        raise PushLearningError("a push session may belong to only one split.")
    cells_by_split = {
        split: {item.coverage_cell for item in specs if item.split == split}
        for split in ("train", "validation", "test")
    }
    if any(
        cells_by_split[left] & cells_by_split[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise PushLearningError("push pilot coverage cells must be disjoint by split.")
    observation = _mapping(config, "observation")
    if (
        observation.get("schema_version") != PUSH_OBSERVATION_VERSION
        or observation.get("simulator_state_only") is not True
        or tuple(observation.get("fields", ())) != PUSH_OBSERVATION_NAMES
    ):
        raise PushLearningError("push pilot observation contract is not frozen.")
    action = _mapping(config, "action")
    if (
        action.get("schema_version") != LOWDIM_ACTION_VERSION
        or tuple(action.get("output_fields", ())) != PUSH_DELTA_NAMES
        or action.get("coordinate_frame") != "initial_object_to_target"
        or action.get("nominal_contact_constraint") != "positive_task_x_only"
    ):
        raise PushLearningError("push pilot action contract is not frozen.")
    _positive_vector3(action.get("maximum_absolute_delta_m"), "maximum push delta")
    phase_limits = action.get("maximum_absolute_delta_by_phase_m")
    if not isinstance(phase_limits, Mapping) or set(phase_limits) != set(PUSH_PHASES):
        raise PushLearningError("push pilot must freeze delta limits by phase.")
    for phase, values in phase_limits.items():
        _nonnegative_vector3(values, f"{phase} push delta limit")
    model = _mapping(config, "model")
    if (
        model.get("recipe_count") != 1
        or model.get("class") != "small_mlp"
        or tuple(model.get("hidden_dims", ())) != (64, 64)
        or model.get("activation") != "tanh"
    ):
        raise PushLearningError("Level 4.3H permits exactly one 64x64 tanh MLP.")
    training = _mapping(config, "training")
    required_training = {
        "epochs": 220,
        "batch_size": 64,
        "learning_rate": 0.002,
        "weight_decay": 0.0,
        "seed": 20260905,
        "device": "cpu",
        "loss": "phase_balanced_mse",
    }
    if any(training.get(name) != value for name, value in required_training.items()):
        raise PushLearningError("push pilot training recipe differs from the freeze.")
    rollout = _mapping(config, "rollout")
    seeds = rollout.get("held_out_seeds")
    if (
        not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or len(seeds) < 20
        or len(set(seeds)) != len(seeds)
        or rollout.get("sim_steps_per_action") != 17
        or float(rollout.get("minimum_success_rate", -1.0)) != 0.70
    ):
        raise PushLearningError("push pilot held-out rollout contract is not frozen.")
    change = _mapping(config, "change_control")
    if (
        change.get("maximum_successes_before_diagnosis") != 20
        or change.get("require_failure_diagnosis_before_more_data") is not True
        or change.get("require_failure_diagnosis_before_model_change") is not True
        or change.get("allow_action_chunking") is not False
        or change.get("action_chunking_evidence")
        != "none_single_step_interface_tested_directly"
        or change.get("allow_image_input") is not False
    ):
        raise PushLearningError("push pilot change control is not frozen.")
    return config, hashlib.sha256(raw).hexdigest()


def push_episode_specs(config: Mapping[str, Any]) -> tuple[PushEpisodeSpec, ...]:
    """Expand session entries without permitting episode-level split drift."""

    sessions = _mapping(config, "dataset").get("sessions")
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise PushLearningError("push pilot sessions must be a sequence.")
    specs: list[PushEpisodeSpec] = []
    seen: set[str] = set()
    for raw_session in sessions:
        if not isinstance(raw_session, Mapping):
            raise PushLearningError("each push session must be a mapping.")
        session_id = _required_string(raw_session, "session_id")
        split = _required_string(raw_session, "split")
        seeds = raw_session.get("seeds")
        cells = raw_session.get("coverage_cells")
        if split not in {"train", "validation", "test"}:
            raise PushLearningError("push session split must be train/validation/test.")
        if (
            not isinstance(seeds, Sequence)
            or isinstance(seeds, (str, bytes))
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise PushLearningError("push session seeds must be non-empty integers.")
        if (
            not isinstance(cells, Sequence)
            or isinstance(cells, (str, bytes))
            or not cells
            or any(not isinstance(cell, str) or not cell for cell in cells)
        ):
            raise PushLearningError("push session coverage cells must be named.")
        for index, seed in enumerate(seeds):
            episode_id = f"{session_id}_{seed:04d}"
            if episode_id in seen:
                raise PushLearningError("push pilot episode ids must be unique.")
            seen.add(episode_id)
            specs.append(
                PushEpisodeSpec(
                    episode_id=episode_id,
                    session_id=session_id,
                    split=split,
                    coverage_cell=str(cells[index % len(cells)]),
                    seed=int(seed),
                )
            )
    return tuple(specs)


def fixed_push_finger_targets(
    task: WorkcellPilotTask,
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> dict[str, float]:
    """Resolve the same family posture used by the qualified scripted expert."""

    retargeter = CurlRetargeter.from_yaml(retargeter_config)
    targets = retargeter.map(no_hand_features())
    family = str(task.coverage_cell["family"])
    raw = task.collection_config["pilot"]["scripted_push"]["family_parameters"]
    index_curl = float(raw[family]["index_curl"])
    for finger in retargeter.config.fingers:
        curl = index_curl if finger.name == "index" else 1.0
        for target in finger.targets:
            targets[target.name] = target.map_control(curl)
    return targets


def prepare_push_learning_reset(
    task: WorkcellPilotTask,
    config: Mapping[str, Any],
    *,
    seed: int,
    fixed_fingers: Mapping[str, float],
) -> WorldState:
    """Apply deterministic hand-pose jitter while preserving task objects."""

    reset = _mapping(config, "reset")
    half_range = _nonnegative_vector3(
        reset.get("seed_jitter_half_range_m"), "reset jitter"
    )
    if not np.any(half_range):
        return task.current_state.world_state
    rng = np.random.default_rng(int(seed) + 4_300)
    jitter = rng.uniform(-half_range, half_range)
    initial = task.current_state.world_state
    position = np.asarray(initial.robot.base_position, dtype=np.float64) + jitter
    safety = _mapping(config, "safety")
    minimum = _finite_vector3(safety.get("workspace_min_m"), "workspace minimum")
    maximum = _finite_vector3(safety.get("workspace_max_m"), "workspace maximum")
    if np.any(position < minimum) or np.any(position > maximum):
        raise PushLearningError("frozen push reset escaped the safe workspace.")
    task.workcell.set_hand_base_reset_pose(position, initial.robot.base_orientation_wxyz)
    task.env.set_joint_targets(fixed_fingers)
    state = task.step(n_steps=int(reset["settle_simulation_steps"]))
    return state.world_state


def push_task_frame(task: WorkcellPilotTask, world_state: WorldState) -> np.ndarray:
    """Return right-handed columns mapping initial task coordinates to world."""

    source = np.asarray(
        world_state.require_entity(str(task.goal["object_id"])).position,
        dtype=np.float64,
    )
    target = np.asarray(
        world_state.require_entity(str(task.goal["target_zone"])).position,
        dtype=np.float64,
    )
    direction = target[:2] - source[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise PushLearningError("push source and target must be distinct.")
    x_axis = np.asarray([direction[0] / norm, direction[1] / norm, 0.0])
    y_axis = np.asarray([-x_axis[1], x_axis[0], 0.0])
    return np.column_stack((x_axis, y_axis, np.asarray([0.0, 0.0, 1.0])))


def push_observation(
    task: WorkcellPilotTask,
    world_state: WorldState,
    *,
    phase: str,
    previous_applied_delta: Sequence[float],
    target_orientation_wxyz: Sequence[float],
    task_to_world_rotation: np.ndarray,
    target_stop_distance_m: float,
) -> np.ndarray:
    """Build the causal state-only push observation in the frozen task frame."""

    if phase not in PUSH_PHASES:
        raise PushLearningError(f"unsupported push phase {phase!r}.")
    previous = _finite_vector3(previous_applied_delta, "previous push delta")
    rotation = np.asarray(task_to_world_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise PushLearningError("push task rotation must be 3x3.")
    world_to_task = rotation.T
    object_state = world_state.require_entity(str(task.goal["object_id"]))
    target_state = world_state.require_entity(str(task.goal["target_zone"]))
    ee = np.asarray(world_state.robot.end_effector_position, dtype=np.float64)
    object_position = np.asarray(object_state.position, dtype=np.float64)
    target_position = np.asarray(target_state.position, dtype=np.float64)
    object_linear = np.asarray(object_state.linear_velocity or (0.0, 0.0, 0.0))
    object_angular = np.asarray(object_state.angular_velocity or (0.0, 0.0, 0.0))
    base_velocity = _base_velocity(task)
    family = str(task.coverage_cell["family"])
    observation = np.concatenate(
        (
            world_to_task @ (object_position - ee),
            world_to_task @ (target_position - object_position),
            _relative_quaternion(
                target_orientation_wxyz,
                world_state.robot.end_effector_orientation_wxyz,
            ),
            world_to_task @ object_linear,
            world_to_task @ object_angular,
            world_to_task @ base_velocity[:3],
            world_to_task @ base_velocity[3:],
            np.asarray([float(name == phase) for name in PUSH_PHASES]),
            previous,
            np.asarray([float(name == family) for name in PUSH_FAMILIES]),
            np.asarray([float(target_stop_distance_m)]),
        )
    )
    if observation.shape != (len(PUSH_OBSERVATION_NAMES),) or not np.all(
        np.isfinite(observation)
    ):
        raise PushLearningError("constructed push observation is invalid.")
    return observation


def collect_push_expert_trajectories(
    config: Mapping[str, Any],
    *,
    dataset_config: str | Path = DEFAULT_LEVEL4_DATASET_CONFIG,
    workcell_config: str | Path = DEFAULT_WORKCELL_CONFIG,
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> tuple[PushTrajectory, ...]:
    """Collect exactly 20 successful qualified expert trajectories in memory."""

    trajectories = tuple(
        _collect_one_push_trajectory(
            spec,
            config=config,
            dataset_config=dataset_config,
            workcell_config=workcell_config,
            retargeter_config=retargeter_config,
        )
        for spec in push_episode_specs(config)
    )
    failed = [item.spec.episode_id for item in trajectories if not item.success]
    if failed:
        raise PushLearningError(
            "frozen scripted push collection produced failures: " + ", ".join(failed)
        )
    return trajectories


def train_push_delta_policy(
    trajectories: Sequence[PushTrajectory], config: Mapping[str, Any]
) -> PushTrainingResult:
    """Train the reused one-step MLP and select solely by validation loss."""

    if len(trajectories) != 20 or any(not item.success for item in trajectories):
        raise PushLearningError("training requires the frozen 20 push successes.")
    by_split = {
        split: tuple(item for item in trajectories if item.spec.split == split)
        for split in ("train", "validation", "test")
    }
    if tuple(len(by_split[name]) for name in by_split) != (14, 3, 3):
        raise PushLearningError("push trajectories must retain the 14/3/3 split.")
    arrays = {split: _stack_trajectories(items) for split, items in by_split.items()}
    train_x, train_y, train_phases = arrays["train"]
    validation_x, validation_y, _ = arrays["validation"]
    test_x, test_y, _ = arrays["test"]
    observation_stats = _normalization(train_x)
    action_stats = _normalization(train_y)
    normalized = {
        split: (
            _apply_normalization(values[0], observation_stats),
            _apply_normalization(values[1], action_stats),
        )
        for split, values in arrays.items()
    }
    model_config = _mapping(config, "model")
    training = _mapping(config, "training")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LowDimDeltaMLP(
        input_dim=len(PUSH_OBSERVATION_NAMES),
        hidden_dims=tuple(int(value) for value in model_config["hidden_dims"]),
        activation=str(model_config["activation"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_tensor = torch.as_tensor(normalized["train"][0], dtype=torch.float32)
    target_tensor = torch.as_tensor(normalized["train"][1], dtype=torch.float32)
    weights = torch.as_tensor(
        phase_balancing_weights(train_phases, PUSH_PHASES), dtype=torch.float32
    )
    validation_tensor = torch.as_tensor(
        normalized["validation"][0], dtype=torch.float32
    )
    validation_target = torch.as_tensor(
        normalized["validation"][1], dtype=torch.float32
    )
    batch_size = int(training["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation = float("inf")
    best_training = float("inf")
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_tensor), generator=generator)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(train_tensor[indices])
            per_sample = torch.mean((prediction - target_tensor[indices]) ** 2, dim=1)
            batch_weights = weights[indices]
            loss = torch.sum(per_sample * batch_weights) / torch.sum(batch_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(torch.sum(per_sample.detach() * batch_weights))
            total_weight += float(torch.sum(batch_weights))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.mean((model(validation_tensor) - validation_target) ** 2)
            )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_training = total_loss / total_weight
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise PushLearningError("push training did not produce a model snapshot.")
    model.load_state_dict(best_state)
    model.eval()
    test_tensor = torch.as_tensor(normalized["test"][0], dtype=torch.float32)
    test_target = torch.as_tensor(normalized["test"][1], dtype=torch.float32)
    with torch.no_grad():
        test_loss = float(torch.mean((model(test_tensor) - test_target) ** 2))
    policy = LowDimDeltaPolicy(
        model=model,
        observation_normalization=observation_stats,
        action_normalization=action_stats,
        maximum_absolute_delta_m=_mapping(config, "action")["maximum_absolute_delta_m"],
    )
    return PushTrainingResult(
        policy=policy,
        selected_epoch=best_epoch,
        training_loss=best_training,
        validation_loss=best_validation,
        test_loss=test_loss,
        sample_counts={split: int(arrays[split][0].shape[0]) for split in arrays},
        phase_counts=dict(sorted(Counter(train_phases).items())),
    )


def _collect_one_push_trajectory(
    spec: PushEpisodeSpec,
    *,
    config: Mapping[str, Any],
    dataset_config: str | Path,
    workcell_config: str | Path,
    retargeter_config: str | Path,
) -> PushTrajectory:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[str] = []
    with WorkcellPilotTask(
        workcell_config=workcell_config,
        dataset_config=dataset_config,
        skill_name="push_object_to_target",
        goal_condition_id=spec.coverage_cell,
        seed=spec.seed,
    ) as task:
        fingers = fixed_push_finger_targets(task, retargeter_config)
        world = prepare_push_learning_reset(
            task, config, seed=spec.seed, fixed_fingers=fingers
        )
        expert_config = DeterministicPushConfig.from_mapping(
            task.collection_config["pilot"]["scripted_push"]
        )
        expert = DeterministicPushExpert(finger_targets=fingers, config=expert_config)
        expert.reset(task, world)
        if expert.validation is None or not expert.validation.valid:
            reason = expert.validation.reason if expert.validation else "missing_validation"
            raise PushLearningError(
                f"push expert validation failed for {spec.episode_id}: {reason}"
            )
        frame = push_task_frame(task, world)
        previous_delta = np.zeros(3, dtype=np.float64)
        current = task.current_state
        achieved_success = False
        terminal_reason = "timeout"
        for _step in range(int(_mapping(config, "rollout")["maximum_steps"])):
            requested, phase, done, reason = expert.step(current.world_state)
            observations.append(
                push_observation(
                    task,
                    current.world_state,
                    phase=phase,
                    previous_applied_delta=previous_delta,
                    target_orientation_wxyz=expert.target_orientation_wxyz,
                    task_to_world_rotation=frame,
                    target_stop_distance_m=expert_config.target_stop_distance_m,
                )
            )
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=requested.base_position,
                orientation_quat=requested.base_orientation_wxyz,
            )
            task.env.set_joint_targets(requested.finger_targets)
            current = task.step(n_steps=expert_config.sim_steps_per_action)
            delta = constrain_push_delta(
                frame.T
                @ (
                    requested.base_position
                    - np.asarray(
                        current.world_state.robot.end_effector_position,
                        dtype=np.float64,
                    )
                ),
                phase=phase,
                phase_limits=_mapping(config, "action")[
                    "maximum_absolute_delta_by_phase_m"
                ],
            )
            actions.append(delta)
            phases.append(phase)
            achieved_success = achieved_success or current.success
            previous_delta = delta
            if reason is not None:
                terminal_reason = reason
                break
            if done:
                terminal_reason = "completed"
                break
    return PushTrajectory(
        spec=spec,
        observations=np.asarray(observations, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        phases=tuple(phases),
        success=achieved_success and terminal_reason == "completed",
        terminal_reason=terminal_reason,
    )


def _stack_trajectories(
    trajectories: Sequence[PushTrajectory],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not trajectories:
        raise PushLearningError("each push split must contain episodes.")
    return (
        np.concatenate([item.observations for item in trajectories], axis=0),
        np.concatenate([item.actions for item in trajectories], axis=0),
        tuple(phase for item in trajectories for phase in item.phases),
    )


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise PushLearningError(f"push pilot {key} must be a non-empty string.")
    return value


def _finite_vector3(values: object, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise PushLearningError(f"{name} must be a finite 3-vector.")
    return vector


def _positive_vector3(values: object, name: str) -> np.ndarray:
    vector = _finite_vector3(values, name)
    if np.any(vector <= 0.0):
        raise PushLearningError(f"{name} must contain positive values.")
    return vector


def _nonnegative_vector3(values: object, name: str) -> np.ndarray:
    vector = _finite_vector3(values, name)
    if np.any(vector < 0.0):
        raise PushLearningError(f"{name} must contain non-negative values.")
    return vector


def _unit_quaternion(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise PushLearningError("push orientation must be a finite wxyz quaternion.")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise PushLearningError("push orientation must be non-zero.")
    return quaternion / norm
