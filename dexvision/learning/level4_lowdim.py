"""Frozen state-only low-dimensional learning probe for Level 4 button press."""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from dexvision.features.hand_features import no_hand_features
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.retargeting.curl_retargeter import CurlRetargeter
from dexvision.sim.level4_expert import (
    DeterministicButtonPressConfig,
    DeterministicButtonPressExpert,
    RequestedAction,
    level4_action_names,
)
from dexvision.sim.world_state import WorldState


BUTTON_PILOT_VERSION = "level4/button-learning-pilot-v1"
BUTTON_OBSERVATION_VERSION = "level4/button-lowdim-observation-v1"
BUTTON_ACTION_VERSION = "level4/button-task-local-delta-v1"
BUTTON_PHASES = ("approach", "fixture_contact", "retract")
BUTTON_OBSERVATION_NAMES = (
    "ee_to_button_position/x",
    "ee_to_button_position/y",
    "ee_to_button_position/z",
    "ee_to_target_orientation/qw",
    "ee_to_target_orientation/qx",
    "ee_to_target_orientation/qy",
    "ee_to_target_orientation/qz",
    "button/press_depth",
    "button/pressed",
    "base_velocity/vx",
    "base_velocity/vy",
    "base_velocity/vz",
    "base_velocity/wx",
    "base_velocity/wy",
    "base_velocity/wz",
    "phase/approach",
    "phase/fixture_contact",
    "phase/retract",
    "previous_applied_delta/dx",
    "previous_applied_delta/dy",
    "previous_applied_delta/dz",
    "goal/target_press_depth",
)
BUTTON_DELTA_NAMES = ("dx", "dy", "dz")
DEFAULT_BUTTON_PILOT_CONFIG = Path("configs/level4_button_learning_pilot.yaml")
DEFAULT_LEVEL4_DATASET_CONFIG = Path("configs/level4_dataset.yaml")
DEFAULT_WORKCELL_CONFIG = Path("configs/workcell.yaml")
DEFAULT_RETARGETER_CONFIG = Path("configs/level1_teleop.yaml")


class ButtonLearningError(RuntimeError):
    """Raised when the frozen button learning protocol cannot be reproduced."""


@dataclass(frozen=True)
class ButtonEpisodeSpec:
    """One session-owned scripted trajectory in the frozen 20-episode set."""

    episode_id: str
    session_id: str
    split: str
    coverage_cell: str
    seed: int


@dataclass(frozen=True)
class ButtonTrajectory:
    """Simulator-state observations and task-local expert deltas for one run."""

    spec: ButtonEpisodeSpec
    observations: np.ndarray
    actions: np.ndarray
    phases: tuple[str, ...]
    success: bool
    terminal_reason: str


@dataclass(frozen=True)
class VectorNormalization:
    """Training-only population statistics with protected constant columns."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class ButtonTrainingResult:
    """The single trained recipe and its frozen selection evidence."""

    policy: "ButtonDeltaPolicy"
    selected_epoch: int
    training_loss: float
    validation_loss: float
    test_loss: float
    sample_counts: Mapping[str, int]
    phase_counts: Mapping[str, int]


class ButtonDeltaMLP(nn.Module):
    """Small MLP mapping normalized causal simulator state to XYZ delta."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ButtonLearningError("button MLP dimensions must be positive.")
        activation_type: type[nn.Module]
        if activation == "tanh":
            activation_type = nn.Tanh
        elif activation == "relu":
            activation_type = nn.ReLU
        else:
            raise ButtonLearningError("button MLP activation must be tanh or relu.")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, int(width)), activation_type()))
            previous = int(width)
        layers.append(nn.Linear(previous, len(BUTTON_DELTA_NAMES)))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != len(
            BUTTON_OBSERVATION_NAMES
        ):
            raise ButtonLearningError(
                "button observations must have shape [batch, "
                f"{len(BUTTON_OBSERVATION_NAMES)}]."
            )
        return self.network(observations)


class ButtonDeltaPolicy:
    """CPU inference wrapper retaining training-only normalization."""

    def __init__(
        self,
        *,
        model: ButtonDeltaMLP,
        observation_normalization: VectorNormalization,
        action_normalization: VectorNormalization,
        maximum_absolute_delta_m: Sequence[float],
    ) -> None:
        maximum = np.asarray(maximum_absolute_delta_m, dtype=np.float64)
        if maximum.shape != (3,) or not np.all(np.isfinite(maximum)) or np.any(maximum <= 0):
            raise ButtonLearningError(
                "maximum_absolute_delta_m must be a positive finite 3-vector."
            )
        self.model = model.eval()
        self.observation_normalization = observation_normalization
        self.action_normalization = action_normalization
        self.maximum_absolute_delta_m = maximum

    def predict(self, observation: Sequence[float]) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (len(BUTTON_OBSERVATION_NAMES),) or not np.all(
            np.isfinite(values)
        ):
            raise ButtonLearningError("button policy observation is invalid.")
        normalized = (
            values - self.observation_normalization.mean
        ) / self.observation_normalization.std
        with torch.no_grad():
            prediction = self.model(
                torch.as_tensor(normalized[None, :], dtype=torch.float32)
            )[0].cpu().numpy()
        delta = (
            prediction * self.action_normalization.std
            + self.action_normalization.mean
        )
        return np.clip(
            np.asarray(delta, dtype=np.float64),
            -self.maximum_absolute_delta_m,
            self.maximum_absolute_delta_m,
        )


class ButtonActionAdapter:
    """Expand one learned task-local XYZ delta to the full action layout."""

    def __init__(
        self,
        *,
        finger_targets: Mapping[str, float],
        fixed_orientation_wxyz: Sequence[float],
        workspace_min_m: Sequence[float],
        workspace_max_m: Sequence[float],
        maximum_absolute_delta_by_phase_m: Mapping[str, Sequence[float]],
    ) -> None:
        self.finger_targets = {
            str(name): float(value) for name, value in finger_targets.items()
        }
        self.orientation = np.asarray(fixed_orientation_wxyz, dtype=np.float64)
        self.workspace_min = np.asarray(workspace_min_m, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max_m, dtype=np.float64)
        if set(maximum_absolute_delta_by_phase_m) != set(BUTTON_PHASES):
            raise ButtonLearningError(
                "button adapter needs one delta limit for every causal phase."
            )
        self.phase_delta_limits = {
            phase: _nonnegative_vector3(values, f"{phase} delta limit")
            for phase, values in maximum_absolute_delta_by_phase_m.items()
        }
        if not self.finger_targets or any(
            not name or not np.isfinite(value)
            for name, value in self.finger_targets.items()
        ):
            raise ButtonLearningError("button adapter needs finite fixed finger targets.")
        if self.orientation.shape != (4,) or not np.all(np.isfinite(self.orientation)):
            raise ButtonLearningError("button adapter orientation must be finite wxyz.")
        norm = float(np.linalg.norm(self.orientation))
        if norm <= 0.0:
            raise ButtonLearningError("button adapter orientation must be non-zero.")
        self.orientation = self.orientation / norm
        if (
            self.workspace_min.shape != (3,)
            or self.workspace_max.shape != (3,)
            or np.any(self.workspace_min >= self.workspace_max)
        ):
            raise ButtonLearningError("button adapter workspace bounds are invalid.")
        self.names = level4_action_names(tuple(self.finger_targets))

    def expand(
        self,
        previous_position: Sequence[float],
        task_local_delta: Sequence[float],
        *,
        phase: str,
    ) -> tuple[RequestedAction, bool]:
        previous = np.asarray(previous_position, dtype=np.float64)
        delta = np.asarray(task_local_delta, dtype=np.float64)
        if previous.shape != (3,) or delta.shape != (3,) or not np.all(
            np.isfinite(np.concatenate((previous, delta)))
        ):
            raise ButtonLearningError("button adapter position and delta must be finite 3-vectors.")
        if phase not in self.phase_delta_limits:
            raise ButtonLearningError(f"unsupported button adapter phase {phase!r}.")
        limit = self.phase_delta_limits[phase]
        delta = np.clip(delta, -limit, limit)
        # The current fixture frame is axis-aligned: normal +x, lateral +y,
        # vertical +z. Keeping the conversion explicit prevents world-frame
        # action semantics from leaking into the learned output contract.
        raw_position = previous + delta
        workspace_violation = bool(
            np.any(raw_position < self.workspace_min)
            or np.any(raw_position > self.workspace_max)
        )
        position = np.clip(raw_position, self.workspace_min, self.workspace_max)
        values = (
            *position.tolist(),
            *self.orientation.tolist(),
            *(self.finger_targets[name] for name in self.finger_targets),
        )
        return RequestedAction(self.names, tuple(float(value) for value in values)), workspace_violation


def load_button_learning_config(
    path: str | Path = DEFAULT_BUTTON_PILOT_CONFIG,
) -> tuple[dict[str, Any], str]:
    """Load and strictly validate the one frozen Level 4.3G recipe."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw)
    except OSError as exc:
        raise ButtonLearningError(f"cannot read button pilot config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ButtonLearningError(f"invalid button pilot YAML {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ButtonLearningError("button pilot config must contain a mapping.")
    config = dict(payload)
    if config.get("version") != BUTTON_PILOT_VERSION:
        raise ButtonLearningError(
            f"button pilot version must be {BUTTON_PILOT_VERSION!r}."
        )
    dataset = _mapping(config, "dataset")
    sessions = dataset.get("sessions")
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise ButtonLearningError("button pilot sessions must be a sequence.")
    specs = button_episode_specs(config)
    if dataset.get("successful_episodes") != 20 or len(specs) != 20:
        raise ButtonLearningError("Level 4.3G must freeze exactly 20 expert successes.")
    session_splits: dict[str, str] = {}
    for spec in specs:
        previous = session_splits.setdefault(spec.session_id, spec.split)
        if previous != spec.split:
            raise ButtonLearningError("a button session may belong to only one split.")
    if Counter(spec.split for spec in specs) != {
        "train": 14,
        "validation": 3,
        "test": 3,
    }:
        raise ButtonLearningError("button pilot split counts must be 14/3/3.")

    observation = _mapping(config, "observation")
    if (
        observation.get("schema_version") != BUTTON_OBSERVATION_VERSION
        or observation.get("simulator_state_only") is not True
        or tuple(observation.get("fields", ())) != BUTTON_OBSERVATION_NAMES
    ):
        raise ButtonLearningError("button pilot observation contract is not frozen.")
    reset = _mapping(config, "reset")
    offsets = reset.get("approach_offset_by_class_m")
    expected_classes = {"centered", "oblique", "left_offset", "right_offset"}
    if not isinstance(offsets, Mapping) or set(offsets) != expected_classes:
        raise ButtonLearningError(
            "button pilot reset offsets must cover every approach class."
        )
    for approach_class, offset in offsets.items():
        _finite_vector3(offset, f"reset offset {approach_class}")
    _positive_vector(
        reset.get("seed_jitter_half_range_m"), "seed_jitter_half_range_m"
    )
    settle_steps = reset.get("settle_simulation_steps")
    if isinstance(settle_steps, bool) or not isinstance(settle_steps, int) or settle_steps <= 0:
        raise ButtonLearningError("button pilot settle_simulation_steps must be positive.")
    action = _mapping(config, "action")
    if (
        action.get("schema_version") != BUTTON_ACTION_VERSION
        or tuple(action.get("output_fields", ())) != BUTTON_DELTA_NAMES
        or action.get("coordinate_frame") != "button_fixture"
        or action.get("expansion") != "fixed_orientation_and_finger_posture"
    ):
        raise ButtonLearningError("button pilot action contract is not frozen.")
    _positive_vector(action.get("maximum_absolute_delta_m"), "maximum_absolute_delta_m")
    phase_limits = action.get("maximum_absolute_delta_by_phase_m")
    if not isinstance(phase_limits, Mapping) or set(phase_limits) != set(BUTTON_PHASES):
        raise ButtonLearningError("button pilot must freeze delta limits by phase.")
    for phase, limit in phase_limits.items():
        _nonnegative_vector3(limit, f"{phase} delta limit")
    synergy = action.get("fixed_finger_synergy")
    if not isinstance(synergy, Mapping) or set(synergy) != {
        "index",
        "middle",
        "ring",
        "pinky",
    }:
        raise ButtonLearningError("button pilot fixed finger synergy is incomplete.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
        for value in synergy.values()
    ):
        raise ButtonLearningError("button fixed finger synergies must be in [0, 1].")
    _finite_vector3(
        action.get("contact_posture_press_offset_adjustment_m"),
        "contact posture press offset adjustment",
    )
    model = _mapping(config, "model")
    if model.get("recipe_count") != 1 or model.get("class") != "small_mlp":
        raise ButtonLearningError("Level 4.3G permits exactly one small-MLP recipe.")
    if tuple(model.get("hidden_dims", ())) != (64, 64) or model.get("activation") != "tanh":
        raise ButtonLearningError("the Level 4.3G MLP architecture must remain 64x64 tanh.")
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
        raise ButtonLearningError("button pilot training recipe differs from the freeze.")
    rollout = _mapping(config, "rollout")
    held_out_seeds = rollout.get("held_out_seeds")
    if (
        not isinstance(held_out_seeds, Sequence)
        or isinstance(held_out_seeds, (str, bytes))
        or len(held_out_seeds) < 20
        or len(set(held_out_seeds)) != len(held_out_seeds)
    ):
        raise ButtonLearningError("button pilot needs at least 20 unique held-out seeds.")
    if float(rollout.get("minimum_success_rate", -1.0)) != 0.80:
        raise ButtonLearningError("button pilot minimum success rate must be 0.80.")
    change = _mapping(config, "change_control")
    if (
        change.get("maximum_successes_before_diagnosis") != 20
        or change.get("require_failure_diagnosis_before_more_data") is not True
        or change.get("require_failure_diagnosis_before_model_change") is not True
        or change.get("allow_action_chunking") is not False
        or change.get("allow_image_input") is not False
    ):
        raise ButtonLearningError("button pilot change control is not frozen.")
    return config, hashlib.sha256(raw).hexdigest()


def button_episode_specs(config: Mapping[str, Any]) -> tuple[ButtonEpisodeSpec, ...]:
    """Expand frozen session entries without allowing episode-level split drift."""

    dataset = _mapping(config, "dataset")
    sessions = dataset.get("sessions")
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise ButtonLearningError("button pilot sessions must be a sequence.")
    specs: list[ButtonEpisodeSpec] = []
    seen_ids: set[str] = set()
    for raw_session in sessions:
        if not isinstance(raw_session, Mapping):
            raise ButtonLearningError("each button pilot session must be a mapping.")
        session_id = _required_string(raw_session, "session_id")
        split = _required_string(raw_session, "split")
        if split not in {"train", "validation", "test"}:
            raise ButtonLearningError("button session split must be train/validation/test.")
        seeds = raw_session.get("seeds")
        cells = raw_session.get("coverage_cells")
        if (
            not isinstance(seeds, Sequence)
            or isinstance(seeds, (str, bytes))
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise ButtonLearningError("button session seeds must be non-empty integers.")
        if (
            not isinstance(cells, Sequence)
            or isinstance(cells, (str, bytes))
            or not cells
            or any(not isinstance(cell, str) or not cell for cell in cells)
        ):
            raise ButtonLearningError("button session coverage cells must be named.")
        for index, seed in enumerate(seeds):
            episode_id = f"{session_id}_{int(seed):04d}"
            if episode_id in seen_ids:
                raise ButtonLearningError("button pilot episode ids must be unique.")
            seen_ids.add(episode_id)
            specs.append(
                ButtonEpisodeSpec(
                    episode_id=episode_id,
                    session_id=session_id,
                    split=split,
                    coverage_cell=str(cells[index % len(cells)]),
                    seed=int(seed),
                )
            )
    return tuple(specs)


def collect_button_expert_trajectories(
    config: Mapping[str, Any],
    *,
    dataset_config: str | Path = DEFAULT_LEVEL4_DATASET_CONFIG,
    workcell_config: str | Path = DEFAULT_WORKCELL_CONFIG,
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> tuple[ButtonTrajectory, ...]:
    """Collect exactly the frozen 20 successful scripted trajectories in memory."""

    fixed_fingers = fixed_button_finger_targets(config, retargeter_config)
    trajectories = tuple(
        _collect_one_button_trajectory(
            spec,
            config=config,
            fixed_fingers=fixed_fingers,
            dataset_config=dataset_config,
            workcell_config=workcell_config,
        )
        for spec in button_episode_specs(config)
    )
    failed = [item.spec.episode_id for item in trajectories if not item.success]
    if failed:
        raise ButtonLearningError(
            "frozen scripted button collection produced failures: " + ", ".join(failed)
        )
    if len(trajectories) != 20:
        raise ButtonLearningError("button collection must contain exactly 20 successes.")
    return trajectories


def train_button_delta_policy(
    trajectories: Sequence[ButtonTrajectory],
    config: Mapping[str, Any],
) -> ButtonTrainingResult:
    """Train the single frozen recipe and select only by validation loss."""

    if len(trajectories) != 20 or any(not item.success for item in trajectories):
        raise ButtonLearningError("training requires the frozen 20 successful trajectories.")
    by_split = {
        split: tuple(item for item in trajectories if item.spec.split == split)
        for split in ("train", "validation", "test")
    }
    if tuple(len(by_split[name]) for name in by_split) != (14, 3, 3):
        raise ButtonLearningError("training trajectories must retain the 14/3/3 split.")
    arrays = {split: _stack_trajectories(items) for split, items in by_split.items()}
    train_x, train_y, train_phases = arrays["train"]
    validation_x, validation_y, _ = arrays["validation"]
    test_x, test_y, _ = arrays["test"]
    observation_stats = _normalization(train_x)
    action_stats = _normalization(train_y)
    normalized = {
        "train": (
            _apply_normalization(train_x, observation_stats),
            _apply_normalization(train_y, action_stats),
        ),
        "validation": (
            _apply_normalization(validation_x, observation_stats),
            _apply_normalization(validation_y, action_stats),
        ),
        "test": (
            _apply_normalization(test_x, observation_stats),
            _apply_normalization(test_y, action_stats),
        ),
    }
    model_config = _mapping(config, "model")
    training = _mapping(config, "training")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ButtonDeltaMLP(
        input_dim=len(BUTTON_OBSERVATION_NAMES),
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
    phase_weights = _phase_balancing_weights(train_phases)
    weight_tensor = torch.as_tensor(phase_weights, dtype=torch.float32)
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
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(train_tensor[indices])
            per_sample = torch.mean((prediction - target_tensor[indices]) ** 2, dim=1)
            weights = weight_tensor[indices]
            loss = torch.sum(per_sample * weights) / torch.sum(weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(torch.sum(per_sample.detach() * weights))
            epoch_weight += float(torch.sum(weights))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.mean((model(validation_tensor) - validation_target) ** 2)
            )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_training = epoch_loss / epoch_weight
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ButtonLearningError("button training did not produce a model snapshot.")
    model.load_state_dict(best_state)
    model.eval()
    test_tensor = torch.as_tensor(normalized["test"][0], dtype=torch.float32)
    test_target = torch.as_tensor(normalized["test"][1], dtype=torch.float32)
    with torch.no_grad():
        test_loss = float(torch.mean((model(test_tensor) - test_target) ** 2))
    action = _mapping(config, "action")
    policy = ButtonDeltaPolicy(
        model=model,
        observation_normalization=observation_stats,
        action_normalization=action_stats,
        maximum_absolute_delta_m=action["maximum_absolute_delta_m"],
    )
    return ButtonTrainingResult(
        policy=policy,
        selected_epoch=best_epoch,
        training_loss=best_training,
        validation_loss=best_validation,
        test_loss=test_loss,
        sample_counts={split: int(arrays[split][0].shape[0]) for split in arrays},
        phase_counts=dict(sorted(Counter(train_phases).items())),
    )


def button_observation(
    task: WorkcellPilotTask,
    world_state: WorldState,
    *,
    phase: str,
    previous_applied_delta: Sequence[float],
    target_orientation_wxyz: Sequence[float],
) -> np.ndarray:
    """Build the frozen causal state-only observation for one control sample."""

    if phase not in BUTTON_PHASES:
        raise ButtonLearningError(f"unsupported button phase {phase!r}.")
    previous = np.asarray(previous_applied_delta, dtype=np.float64)
    if previous.shape != (3,) or not np.all(np.isfinite(previous)):
        raise ButtonLearningError("previous applied button delta must be finite XYZ.")
    button_id = str(task.goal["button_id"])
    button = world_state.require_entity(button_id)
    fixture = world_state.require_fixture(button_id)
    relative_position = np.asarray(button.position, dtype=np.float64) - np.asarray(
        world_state.robot.end_effector_position, dtype=np.float64
    )
    relative_orientation = _relative_quaternion(
        target_orientation_wxyz,
        world_state.robot.end_effector_orientation_wxyz,
    )
    velocity = _base_velocity(task)
    one_hot = np.asarray([float(name == phase) for name in BUTTON_PHASES])
    observation = np.concatenate(
        (
            relative_position,
            relative_orientation,
            np.asarray([fixture.press_depth_m, float(fixture.pressed)]),
            velocity,
            one_hot,
            previous,
            np.asarray([float(task.goal["target_press_depth_m"])]),
        )
    )
    if observation.shape != (len(BUTTON_OBSERVATION_NAMES),) or not np.all(
        np.isfinite(observation)
    ):
        raise ButtonLearningError("constructed button observation is invalid.")
    return observation


def fixed_button_finger_targets(
    config: Mapping[str, Any],
    retargeter_config: str | Path = DEFAULT_RETARGETER_CONFIG,
) -> dict[str, float]:
    """Resolve the deterministic index-forward posture used by expert and policy."""

    retargeter = CurlRetargeter.from_yaml(retargeter_config)
    targets = retargeter.map(no_hand_features())
    synergy = _mapping(config, "action")["fixed_finger_synergy"]
    if not isinstance(synergy, Mapping):
        raise ButtonLearningError("button fixed finger synergy must be a mapping.")
    for finger in retargeter.config.fingers:
        if finger.name not in synergy:
            continue
        control = float(synergy[finger.name])
        for target in finger.targets:
            targets[target.name] = target.map_control(control)
    return targets


def prepare_button_learning_reset(
    task: WorkcellPilotTask,
    config: Mapping[str, Any],
    *,
    seed: int,
    fixed_fingers: Mapping[str, float],
) -> WorldState:
    """Apply a frozen approach-class offset plus deterministic seed jitter."""

    reset = _mapping(config, "reset")
    offsets = reset["approach_offset_by_class_m"]
    approach_class = str(task.coverage_cell.get("approach_class", ""))
    if not isinstance(offsets, Mapping) or approach_class not in offsets:
        raise ButtonLearningError(
            f"button coverage cell has unsupported approach class {approach_class!r}."
        )
    base_offset = _finite_vector3(
        offsets[approach_class], f"reset offset {approach_class}"
    )
    jitter_half_range = _positive_vector(
        reset["seed_jitter_half_range_m"], "seed_jitter_half_range_m"
    )
    rng = np.random.default_rng(int(seed) + 4_300)
    jitter = rng.uniform(-jitter_half_range, jitter_half_range)
    initial = task.current_state.world_state
    position = np.asarray(initial.robot.base_position, dtype=np.float64) + base_offset + jitter
    safety = _mapping(config, "safety")
    minimum = _finite_vector3(safety["workspace_min_m"], "workspace_min_m")
    maximum = _finite_vector3(safety["workspace_max_m"], "workspace_max_m")
    if np.any(position < minimum) or np.any(position > maximum):
        raise ButtonLearningError("frozen button reset escaped the safe workspace.")
    task.workcell.set_hand_base_reset_pose(
        position,
        initial.robot.base_orientation_wxyz,
    )
    task.env.set_joint_targets(fixed_fingers)
    state = task.step(n_steps=int(reset["settle_simulation_steps"]))
    return state.world_state


def _collect_one_button_trajectory(
    spec: ButtonEpisodeSpec,
    *,
    config: Mapping[str, Any],
    fixed_fingers: Mapping[str, float],
    dataset_config: str | Path,
    workcell_config: str | Path,
) -> ButtonTrajectory:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[str] = []
    with WorkcellPilotTask(
        workcell_config=workcell_config,
        dataset_config=dataset_config,
        skill_name="press_button",
        goal_condition_id=spec.coverage_cell,
        seed=spec.seed,
    ) as task:
        if task.coverage_cell.get("split_owner") != spec.split:
            raise ButtonLearningError(
                f"coverage cell {spec.coverage_cell!r} is not owned by {spec.split!r}."
            )
        prepare_button_learning_reset(
            task,
            config,
            seed=spec.seed,
            fixed_fingers=fixed_fingers,
        )
        expert_config = DeterministicButtonPressConfig.from_mapping(
            task.collection_config["pilot"]["scripted_button"]
        )
        press_adjustment = _finite_vector3(
            _mapping(config, "action")[
                "contact_posture_press_offset_adjustment_m"
            ],
            "contact posture press offset adjustment",
        )
        expert_config = replace(
            expert_config,
            press_offset_m=tuple(
                np.asarray(expert_config.press_offset_m, dtype=np.float64)
                + press_adjustment
            ),
        )
        expert = DeterministicButtonPressExpert(
            finger_targets=fixed_fingers,
            config=expert_config,
        )
        expert.reset(task, task.initial_world_state)
        if expert.validation is None or not expert.validation.valid:
            reason = expert.validation.reason if expert.validation else "missing_validation"
            raise ButtonLearningError(
                f"button expert validation failed for {spec.episode_id}: {reason}"
            )
        current = task.current_state
        target_orientation = np.asarray(
            current.world_state.robot.end_effector_orientation_wxyz,
            dtype=np.float64,
        )
        previous_position = np.asarray(
            current.world_state.robot.end_effector_position, dtype=np.float64
        )
        previous_delta = np.zeros(3, dtype=np.float64)
        achieved_success = False
        terminal_reason = "timeout"
        for _step in range(int(_mapping(config, "rollout")["maximum_steps"])):
            requested, phase, done, reason = expert.step(current.world_state)
            observations.append(
                button_observation(
                    task,
                    current.world_state,
                    phase=phase,
                    previous_applied_delta=previous_delta,
                    target_orientation_wxyz=target_orientation,
                )
            )
            delta = requested.base_position - previous_position
            actions.append(delta)
            phases.append(phase)
            task.env.set_mocap_pose(
                str(task.workcell.config.scene["hand_base_target"]),
                position=requested.base_position,
                orientation_quat=requested.base_orientation_wxyz,
            )
            task.env.set_joint_targets(requested.finger_targets)
            current = task.step(n_steps=expert_config.sim_steps_per_action)
            achieved_success = achieved_success or current.success
            previous_position = requested.base_position
            previous_delta = delta
            if reason is not None:
                terminal_reason = reason
                break
            if done:
                terminal_reason = "completed"
                break
        success = achieved_success and terminal_reason == "completed"
    return ButtonTrajectory(
        spec=spec,
        observations=np.asarray(observations, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        phases=tuple(phases),
        success=success,
        terminal_reason=terminal_reason,
    )


def _stack_trajectories(
    trajectories: Sequence[ButtonTrajectory],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not trajectories:
        raise ButtonLearningError("each button dataset split must contain episodes.")
    return (
        np.concatenate([item.observations for item in trajectories], axis=0),
        np.concatenate([item.actions for item in trajectories], axis=0),
        tuple(phase for item in trajectories for phase in item.phases),
    )


def _normalization(values: np.ndarray) -> VectorNormalization:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    return VectorNormalization(mean=mean, std=np.maximum(std, 1e-6))


def _apply_normalization(
    values: np.ndarray, stats: VectorNormalization
) -> np.ndarray:
    return (values - stats.mean) / stats.std


def _phase_balancing_weights(phases: Sequence[str]) -> np.ndarray:
    counts = Counter(phases)
    if set(counts) != set(BUTTON_PHASES):
        raise ButtonLearningError("training data must contain every causal button phase.")
    total = float(len(phases))
    return np.asarray(
        [total / (len(BUTTON_PHASES) * counts[phase]) for phase in phases],
        dtype=np.float64,
    )


def _base_velocity(task: WorkcellPilotTask) -> np.ndarray:
    joint_name = str(task.workcell.config.scene["hand_base_free_joint"])
    joint_id = int(
        task.env._mujoco.mj_name2id(
            task.env.model,
            task.env._mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
    )
    if joint_id < 0:
        raise ButtonLearningError(f"unknown hand base joint {joint_name!r}.")
    address = int(task.env.model.jnt_dofadr[joint_id])
    return np.asarray(task.env.data.qvel[address : address + 6], dtype=np.float64)


def _relative_quaternion(
    target_wxyz: Sequence[float], current_wxyz: Sequence[float]
) -> np.ndarray:
    target = np.asarray(target_wxyz, dtype=np.float64)
    current = np.asarray(current_wxyz, dtype=np.float64)
    if target.shape != (4,) or current.shape != (4,):
        raise ButtonLearningError("button orientations must be wxyz quaternions.")
    conjugate = current * np.asarray([1.0, -1.0, -1.0, -1.0])
    tw, tx, ty, tz = target
    cw, cx, cy, cz = conjugate
    result = np.asarray(
        [
            tw * cw - tx * cx - ty * cy - tz * cz,
            tw * cx + tx * cw + ty * cz - tz * cy,
            tw * cy - tx * cz + ty * cw + tz * cx,
            tw * cz + tx * cy - ty * cx + tz * cw,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ButtonLearningError("button relative orientation is invalid.")
    result /= norm
    if result[0] < 0.0:
        result *= -1.0
    return result


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = values.get(key)
    if not isinstance(result, Mapping):
        raise ButtonLearningError(f"button pilot {key} must be a mapping.")
    return result


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ButtonLearningError(f"button pilot {key} must be a non-empty string.")
    return value


def _positive_vector(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ButtonLearningError(f"button pilot {label} must be a positive 3-vector.")
    return result


def _finite_vector3(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ButtonLearningError(f"button pilot {label} must be a finite 3-vector.")
    return result


def _nonnegative_vector3(value: object, label: str) -> np.ndarray:
    result = _finite_vector3(value, label)
    if np.any(result < 0.0) or not np.any(result > 0.0):
        raise ButtonLearningError(
            f"button pilot {label} must be non-negative and not all zero."
        )
    return result
