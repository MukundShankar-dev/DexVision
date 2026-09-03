"""Schema-bound goal-conditioned models for Level 3 behavior cloning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn


BASE_POSITION_ACTION_NAMES = (
    "base_position_target/x",
    "base_position_target/y",
    "base_position_target/z",
)
BASE_ORIENTATION_ACTION_NAMES = (
    "base_orientation_target/qw",
    "base_orientation_target/qx",
    "base_orientation_target/qy",
    "base_orientation_target/qz",
)
FINGER_ACTION_PREFIX = "finger_actuator_targets/"


class LearningModelError(ValueError):
    """Raised when a learning model or its schema contract is invalid."""


@dataclass(frozen=True)
class PolicySchema:
    """Named input/output contract attached to a learned policy.

    ``dataset_action_names`` always describes the complete saved Level 1.13
    action vector. ``output_action_names`` either matches it exactly or names
    an explicit, order-preserving ablation subset.
    """

    observation_schema_version: str
    action_schema_version: str
    observation_names: tuple[str, ...]
    goal_names: tuple[str, ...]
    dataset_action_names: tuple[str, ...]
    output_action_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, version in (
            ("observation_schema_version", self.observation_schema_version),
            ("action_schema_version", self.action_schema_version),
        ):
            if not isinstance(version, str) or not version:
                raise LearningModelError(f"{label} must be a non-empty string.")
        for label, names in (
            ("observation_names", self.observation_names),
            ("goal_names", self.goal_names),
            ("dataset_action_names", self.dataset_action_names),
            ("output_action_names", self.output_action_names),
        ):
            _validate_names(names, label=label)

        expected_prefix = BASE_POSITION_ACTION_NAMES + BASE_ORIENTATION_ACTION_NAMES
        if self.dataset_action_names[: len(expected_prefix)] != expected_prefix:
            raise LearningModelError(
                "dataset_action_names must begin with the named Level 1.13 base "
                "position and quaternion orientation fields."
            )
        finger_names = self.dataset_action_names[len(expected_prefix) :]
        if not finger_names or any(
            not name.startswith(FINGER_ACTION_PREFIX) or name == FINGER_ACTION_PREFIX
            for name in finger_names
        ):
            raise LearningModelError(
                "dataset_action_names must end with one or more named finger "
                "actuator target fields."
            )

        missing = [
            name for name in self.output_action_names if name not in self.dataset_action_names
        ]
        if missing:
            raise LearningModelError(
                f"output_action_names contains fields absent from the dataset: {missing}."
            )
        output_indices = tuple(
            self.dataset_action_names.index(name) for name in self.output_action_names
        )
        if output_indices != tuple(sorted(output_indices)):
            raise LearningModelError(
                "output_action_names must preserve the dataset action field order."
            )

    @classmethod
    def from_episode(
        cls,
        episode: Any,
        *,
        output_action_names: Sequence[str] | None = None,
    ) -> PolicySchema:
        """Create a contract from a validated Level 3 dataset episode layout."""

        dataset_action_names = tuple(episode.action_names)
        return cls(
            observation_schema_version=episode.observation_schema_version,
            action_schema_version=episode.action_schema_version,
            observation_names=tuple(episode.observation_names),
            goal_names=tuple(episode.goal_names),
            dataset_action_names=dataset_action_names,
            output_action_names=(
                dataset_action_names
                if output_action_names is None
                else tuple(output_action_names)
            ),
        )

    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    @property
    def goal_dim(self) -> int:
        return len(self.goal_names)

    @property
    def dataset_action_dim(self) -> int:
        return len(self.dataset_action_names)

    @property
    def output_dim(self) -> int:
        return len(self.output_action_names)

    @property
    def is_action_subset(self) -> bool:
        return self.output_action_names != self.dataset_action_names

    @property
    def output_indices(self) -> tuple[int, ...]:
        return tuple(
            self.dataset_action_names.index(name) for name in self.output_action_names
        )

    def select_action_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Select the declared model targets from full dataset actions."""

        if actions.ndim < 1 or actions.shape[-1] != self.dataset_action_dim:
            raise LearningModelError(
                "actions must end with the full dataset action dimension "
                f"{self.dataset_action_dim}, got shape {tuple(actions.shape)}."
            )
        indices = torch.tensor(self.output_indices, device=actions.device)
        return torch.index_select(actions, dim=-1, index=indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_schema_version": self.observation_schema_version,
            "action_schema_version": self.action_schema_version,
            "observation_names": list(self.observation_names),
            "goal_names": list(self.goal_names),
            "dataset_action_names": list(self.dataset_action_names),
            "output_action_names": list(self.output_action_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PolicySchema:
        try:
            return cls(
                observation_schema_version=_required_string(
                    payload, "observation_schema_version"
                ),
                action_schema_version=_required_string(payload, "action_schema_version"),
                observation_names=_required_names(payload, "observation_names"),
                goal_names=_required_names(payload, "goal_names"),
                dataset_action_names=_required_names(payload, "dataset_action_names"),
                output_action_names=_required_names(payload, "output_action_names"),
            )
        except (KeyError, TypeError) as exc:
            raise LearningModelError(f"invalid policy schema metadata: {exc}") from exc


@dataclass(frozen=True)
class MLPConfig:
    """Architecture settings for the Level 3.2 baseline."""

    hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "relu"

    SUPPORTED_ACTIVATIONS: ClassVar[tuple[str, ...]] = ("relu", "tanh", "gelu")

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in self.hidden_dims
        ):
            raise LearningModelError("hidden_dims must contain positive integers.")
        if self.activation not in self.SUPPORTED_ACTIVATIONS:
            raise LearningModelError(
                f"activation must be one of {self.SUPPORTED_ACTIVATIONS}, "
                f"got {self.activation!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MLPConfig:
        try:
            hidden_dims = payload["hidden_dims"]
            activation = payload["activation"]
        except KeyError as exc:
            raise LearningModelError(f"invalid MLP config metadata: missing {exc}") from exc
        if not isinstance(hidden_dims, Sequence) or isinstance(hidden_dims, str):
            raise LearningModelError("MLP config hidden_dims must be a sequence.")
        return cls(hidden_dims=tuple(hidden_dims), activation=activation)


class GoalConditionedMLP(nn.Module):
    """Small MLP mapping normalized state and typed goal to normalized action."""

    CHECKPOINT_VERSION = "dexvision/goal-conditioned-mlp-v1"

    def __init__(self, schema: PolicySchema, config: MLPConfig | None = None) -> None:
        super().__init__()
        self.schema = schema
        self.config = config or MLPConfig()

        layers: list[nn.Module] = []
        input_dim = schema.observation_dim + schema.goal_dim
        previous_dim = input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(_activation(self.config.activation))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, schema.output_dim))
        self.network = nn.Sequential(*layers)

    @property
    def output_action_names(self) -> tuple[str, ...]:
        return self.schema.output_action_names

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Predict normalized actions for batches shaped ``[B, features]``."""

        _validate_model_input(
            observations,
            label="observations",
            expected_dim=self.schema.observation_dim,
        )
        _validate_model_input(goals, label="goals", expected_dim=self.schema.goal_dim)
        if observations.shape[0] != goals.shape[0]:
            raise LearningModelError(
                "observations and goals must have the same batch size, got "
                f"{observations.shape[0]} and {goals.shape[0]}."
            )
        if observations.device != goals.device:
            raise LearningModelError("observations and goals must use the same device.")
        if observations.dtype != goals.dtype:
            raise LearningModelError("observations and goals must use the same dtype.")
        return self.network(torch.cat((observations, goals), dim=-1))

    def save(self, output_path: str | Path) -> None:
        """Save weights together with the exact schema and architecture contract."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "schema": self.schema.to_dict(),
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> GoalConditionedMLP:
        """Load a schema-bound model checkpoint without executing pickled code."""

        path = Path(checkpoint_path)
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except (OSError, RuntimeError) as exc:
            raise LearningModelError(f"cannot load model checkpoint {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise LearningModelError(f"model checkpoint {path} must contain a mapping.")
        if payload.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise LearningModelError(
                f"model checkpoint {path} has unsupported checkpoint_version "
                f"{payload.get('checkpoint_version')!r}."
            )
        schema_payload = payload.get("schema")
        config_payload = payload.get("config")
        state_dict = payload.get("state_dict")
        if not isinstance(schema_payload, Mapping):
            raise LearningModelError(f"model checkpoint {path} is missing schema metadata.")
        if not isinstance(config_payload, Mapping):
            raise LearningModelError(f"model checkpoint {path} is missing config metadata.")
        if not isinstance(state_dict, Mapping):
            raise LearningModelError(f"model checkpoint {path} is missing model weights.")
        model = cls(
            schema=PolicySchema.from_dict(schema_payload),
            config=MLPConfig.from_dict(config_payload),
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise LearningModelError(
                f"model checkpoint {path} weights do not match its schema/config: {exc}"
            ) from exc
        return model


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise LearningModelError(f"unsupported activation {name!r}.")


def _validate_names(names: tuple[str, ...], *, label: str) -> None:
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise LearningModelError(f"{label} must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise LearningModelError(f"{label} must not contain duplicates.")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise LearningModelError(f"policy schema {key} must be a non-empty string.")
    return value


def _required_names(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise LearningModelError(f"policy schema {key} must be a sequence.")
    return tuple(value)


def _validate_model_input(
    values: torch.Tensor,
    *,
    label: str,
    expected_dim: int,
) -> None:
    if not isinstance(values, torch.Tensor):
        raise LearningModelError(f"{label} must be a torch.Tensor.")
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise LearningModelError(
            f"{label} must have shape [batch, {expected_dim}], got {tuple(values.shape)}."
        )
    if not values.is_floating_point():
        raise LearningModelError(f"{label} must use a floating-point dtype.")
