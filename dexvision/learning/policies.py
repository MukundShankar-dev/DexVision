"""Validated inference policies for Level 3 closed-loop evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from dexvision.learning.models import GoalConditionedMLP, PolicySchema
from dexvision.learning.train_bc import file_sha256


class PolicyError(RuntimeError):
    """Raised when a policy checkpoint or inference request is invalid."""


class RolloutPolicy(Protocol):
    """Small policy interface consumed by the rollout evaluator."""

    @property
    def observation_names(self) -> tuple[str, ...]: ...

    @property
    def observation_schema_version(self) -> str: ...

    @property
    def action_schema_version(self) -> str: ...

    @property
    def goal_names(self) -> tuple[str, ...]: ...

    @property
    def dataset_action_names(self) -> tuple[str, ...]: ...

    @property
    def output_action_names(self) -> tuple[str, ...]: ...

    @property
    def checkpoint_digest(self) -> str: ...

    @property
    def dataset_digest(self) -> str: ...

    def predict(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class _Stats:
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


class CheckpointPolicy:
    """CPU inference wrapper for a validated Level 3.3 training checkpoint."""

    def __init__(
        self,
        *,
        model: GoalConditionedMLP,
        observation_stats: _Stats,
        goal_stats: _Stats,
        action_stats: _Stats,
        checkpoint_digest: str,
        dataset_digest: str,
        normalization_digest: str,
        experiment_config_digest: str,
        split_manifest_digest: str,
        selected_epoch: int,
        selected_validation_loss: float,
    ) -> None:
        self.model = model.eval()
        self._observation_stats = observation_stats
        self._goal_stats = goal_stats
        self._action_stats = action_stats
        self._checkpoint_digest = checkpoint_digest
        self._dataset_digest = dataset_digest
        self.normalization_digest = normalization_digest
        self.experiment_config_digest = experiment_config_digest
        self.split_manifest_digest = split_manifest_digest
        self.selected_epoch = selected_epoch
        self.selected_validation_loss = selected_validation_loss
        self.schema_digest = _canonical_digest(model.schema.to_dict())

    @property
    def schema(self) -> PolicySchema:
        return self.model.schema

    @property
    def observation_names(self) -> tuple[str, ...]:
        return self.schema.observation_names

    @property
    def observation_schema_version(self) -> str:
        return self.schema.observation_schema_version

    @property
    def action_schema_version(self) -> str:
        return self.schema.action_schema_version

    @property
    def goal_names(self) -> tuple[str, ...]:
        return self.schema.goal_names

    @property
    def dataset_action_names(self) -> tuple[str, ...]:
        return self.schema.dataset_action_names

    @property
    def output_action_names(self) -> tuple[str, ...]:
        return self.schema.output_action_names

    @property
    def checkpoint_digest(self) -> str:
        return self._checkpoint_digest

    @property
    def dataset_digest(self) -> str:
        return self._dataset_digest

    def predict(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Predict one unnormalized action in the declared output layout."""

        observation_array = _finite_vector(
            observation, size=len(self.observation_names), label="observation"
        )
        goal_array = _finite_vector(goal, size=len(self.goal_names), label="goal")
        normalized_observation = (
            observation_array - self._observation_stats.mean
        ) / self._observation_stats.std
        normalized_goal = (goal_array - self._goal_stats.mean) / self._goal_stats.std
        with torch.no_grad():
            prediction = self.model(
                torch.as_tensor(normalized_observation[None, :], dtype=torch.float32),
                torch.as_tensor(normalized_goal[None, :], dtype=torch.float32),
            )[0].cpu().numpy().astype(np.float64, copy=False)
        indices = np.asarray(self.schema.output_indices, dtype=np.int64)
        result = (
            prediction * self._action_stats.std[indices]
            + self._action_stats.mean[indices]
        )
        return np.asarray(result, dtype=np.float64)


def load_checkpoint_policy(
    checkpoint_path: str | Path,
    *,
    expected_dataset_digest: str | None = None,
) -> CheckpointPolicy:
    """Load a checkpoint only after digest, schema, and normalization checks."""

    path = Path(checkpoint_path)
    digest = _verify_required_digest(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise PolicyError(f"cannot load policy checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PolicyError(f"policy checkpoint {path} must contain a mapping.")
    if payload.get("training_checkpoint_version") != "dexvision/bc-training-v1":
        raise PolicyError("policy checkpoint is not a supported Level 3.3 checkpoint.")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PolicyError("policy checkpoint is missing provenance metadata.")
    dataset_digest = _required_string(provenance, "dataset_digest")
    if expected_dataset_digest is not None and dataset_digest != expected_dataset_digest:
        raise PolicyError(
            "policy checkpoint dataset digest does not match the requested dataset."
        )
    normalization = provenance.get("normalization")
    if not isinstance(normalization, Mapping):
        raise PolicyError("policy checkpoint is missing normalization metadata.")
    if normalization.get("source_split") != "train":
        raise PolicyError("policy normalization must come from the training split.")
    if normalization.get("dataset_digest") != dataset_digest:
        raise PolicyError("policy normalization and checkpoint dataset digests differ.")
    normalization_digest = _required_string(provenance, "normalization_digest")
    if normalization_digest != _canonical_digest(normalization):
        raise PolicyError("policy normalization digest verification failed.")

    try:
        model = GoalConditionedMLP.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise PolicyError(f"cannot restore policy model: {exc}") from exc
    schema = model.schema
    observation_stats = _load_stats(
        normalization, "observation", expected_names=schema.observation_names
    )
    goal_stats = _load_stats(normalization, "goal", expected_names=schema.goal_names)
    action_stats = _load_stats(
        normalization, "action", expected_names=schema.dataset_action_names
    )
    experiment_config_digest = _optional_string(
        provenance, "experiment_config_digest", default="unavailable"
    )
    split_manifest_digest = _optional_string(
        provenance, "split_manifest_digest", default="unavailable"
    )
    selected_epoch = payload.get("selected_epoch", payload.get("completed_epochs", 0))
    selected_validation_loss = payload.get("selected_validation_loss")
    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int) or selected_epoch < 0:
        raise PolicyError("policy checkpoint has an invalid selected_epoch.")
    if selected_validation_loss is None and selected_epoch > 0:
        history = payload.get("loss_history")
        if not isinstance(history, list):
            raise PolicyError("policy checkpoint is missing loss history.")
        try:
            selected_validation_loss = next(
                float(item["validation_loss"])
                for item in history
                if item.get("epoch") == selected_epoch
            )
        except (StopIteration, TypeError, ValueError) as exc:
            raise PolicyError("policy checkpoint cannot resolve its selected validation loss.") from exc
    if selected_validation_loss is None:
        selected_validation_loss = float("nan")
    elif (
        isinstance(selected_validation_loss, bool)
        or not isinstance(selected_validation_loss, (int, float))
        or not np.isfinite(selected_validation_loss)
    ):
        raise PolicyError("policy checkpoint has an invalid selected validation loss.")
    return CheckpointPolicy(
        model=model,
        observation_stats=observation_stats,
        goal_stats=goal_stats,
        action_stats=action_stats,
        checkpoint_digest=digest,
        dataset_digest=dataset_digest,
        normalization_digest=normalization_digest,
        experiment_config_digest=experiment_config_digest,
        split_manifest_digest=split_manifest_digest,
        selected_epoch=selected_epoch,
        selected_validation_loss=float(selected_validation_loss),
    )


def _load_stats(
    normalization: Mapping[str, Any],
    key: str,
    *,
    expected_names: tuple[str, ...],
) -> _Stats:
    payload = normalization.get(key)
    if not isinstance(payload, Mapping):
        raise PolicyError(f"normalization is missing {key!r} statistics.")
    raw_names = payload.get("names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
        raise PolicyError(f"normalization {key} names must be a sequence.")
    names = tuple(raw_names)
    if names != expected_names:
        raise PolicyError(f"normalization {key} layout is incompatible with the policy.")
    mean = _finite_vector(payload.get("mean"), size=len(names), label=f"{key} mean")
    std = _finite_vector(payload.get("std"), size=len(names), label=f"{key} std")
    if np.any(std <= 0.0):
        raise PolicyError(f"normalization {key} std values must be positive.")
    count = payload.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PolicyError(f"normalization {key} count must be a positive integer.")
    return _Stats(names=names, mean=mean, std=std)


def _verify_required_digest(path: Path) -> str:
    if not path.is_file():
        raise PolicyError(f"policy checkpoint does not exist: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise PolicyError(f"cannot read required checkpoint digest {sidecar}: {exc}") from exc
    if len(fields) != 2 or fields[1] != path.name:
        raise PolicyError(f"invalid checkpoint digest sidecar {sidecar}.")
    actual = file_sha256(path)
    if fields[0] != actual:
        raise PolicyError(f"checkpoint SHA-256 verification failed for {path}.")
    return actual


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PolicyError(f"checkpoint provenance {key!r} must be a non-empty string.")
    return value


def _optional_string(
    payload: Mapping[str, Any], key: str, *, default: str
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise PolicyError(f"checkpoint provenance {key!r} must be a non-empty string.")
    return value


def _finite_vector(value: object, *, size: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"{label} must be numeric.") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise PolicyError(f"{label} must be a finite vector with shape [{size}].")
    return array
