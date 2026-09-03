"""Deterministic CPU behavior-cloning training for the Level 3 MLP baseline."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dexvision.learning.datasets import (
    DEFAULT_OBSERVATION_FIELDS,
    DatasetBundle,
    GoalConditionedSkillDataset,
    load_frozen_skill_datasets,
)
from dexvision.learning.models import GoalConditionedMLP, MLPConfig, PolicySchema


class BCTrainingError(RuntimeError):
    """Raised when a behavior-cloning experiment is invalid or incompatible."""


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and deterministic data-order settings."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 20260903
    num_workers: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name, value in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BCTrainingError(f"{name} must be a positive integer.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise BCTrainingError("learning_rate must be finite and greater than zero.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise BCTrainingError("weight_decay must be finite and non-negative.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise BCTrainingError("seed must be an integer.")
        if self.num_workers != 0:
            raise BCTrainingError(
                "num_workers must be 0 for deterministic, cross-platform Level 3.3 training."
            )
        if self.device != "cpu":
            raise BCTrainingError("Level 3.3 supports device='cpu' only.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingConfig:
        try:
            return cls(
                epochs=payload.get("epochs", 100),
                batch_size=payload.get("batch_size", 64),
                learning_rate=payload.get("learning_rate", 1e-3),
                weight_decay=payload.get("weight_decay", 0.0),
                seed=payload.get("seed", 20260903),
                num_workers=payload.get("num_workers", 0),
                device=payload.get("device", "cpu"),
            )
        except TypeError as exc:
            raise BCTrainingError(f"invalid training config: {exc}") from exc


@dataclass(frozen=True)
class BCExperimentConfig:
    """Validated file-backed configuration for one reach-policy experiment."""

    version: str
    skill_name: str
    dataset_root: Path
    evaluation_config: Path
    observation_fields: tuple[str, ...]
    include_previous_action: bool
    output_action_names: tuple[str, ...] | None
    model: MLPConfig
    training: TrainingConfig
    output_dir: Path
    checkpoint_name: str
    best_checkpoint_name: str
    source_digest: str

    SUPPORTED_VERSION: ClassVar[str] = "level3/bc-training-v2"
    SUPPORTED_VERSIONS: ClassVar[tuple[str, ...]] = (
        "level3/bc-training-v1",
        SUPPORTED_VERSION,
    )

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / self.best_checkpoint_name

    def compatibility_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "skill_name": self.skill_name,
            "dataset_root": str(self.dataset_root),
            "evaluation_config": str(self.evaluation_config),
            "observation_fields": list(self.observation_fields),
            "include_previous_action": self.include_previous_action,
            "output_action_names": (
                None
                if self.output_action_names is None
                else list(self.output_action_names)
            ),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class TrainingResult:
    """Paths and metrics produced by a completed training invocation."""

    checkpoint_path: Path
    digest_path: Path
    checkpoint_digest: str
    completed_epochs: int
    loss_history: tuple[dict[str, float | int], ...]
    best_checkpoint_path: Path
    best_digest_path: Path
    best_checkpoint_digest: str
    last_checkpoint_path: Path
    last_digest_path: Path
    last_checkpoint_digest: str
    selected_epoch: int
    selected_validation_loss: float


def load_experiment_config(path: str | Path) -> BCExperimentConfig:
    """Load and validate the Level 3.3 YAML experiment configuration."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw)
    except OSError as exc:
        raise BCTrainingError(f"cannot read training config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise BCTrainingError(f"invalid YAML in training config {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BCTrainingError(f"training config {config_path} must contain a YAML mapping.")

    version = _required_string(payload, "version")
    if version not in BCExperimentConfig.SUPPORTED_VERSIONS:
        raise BCTrainingError(
            f"unsupported training config version {version!r}; expected "
            f"one of {BCExperimentConfig.SUPPORTED_VERSIONS!r}."
        )
    dataset = _required_mapping(payload, "dataset")
    model = _required_mapping(payload, "model")
    training = _required_mapping(payload, "training")
    output = _required_mapping(payload, "output")
    observation_fields = dataset.get("observation_fields", DEFAULT_OBSERVATION_FIELDS)
    if not isinstance(observation_fields, Sequence) or isinstance(observation_fields, str):
        raise BCTrainingError("dataset.observation_fields must be a sequence.")
    observation_fields = tuple(observation_fields)
    if not observation_fields or any(
        not isinstance(name, str) or not name for name in observation_fields
    ):
        raise BCTrainingError("dataset.observation_fields must contain non-empty strings.")

    output_names_value = model.get("output_action_names")
    output_action_names: tuple[str, ...] | None
    if output_names_value is None:
        output_action_names = None
    elif isinstance(output_names_value, Sequence) and not isinstance(
        output_names_value, str
    ):
        output_action_names = tuple(output_names_value)
    else:
        raise BCTrainingError("model.output_action_names must be null or a sequence.")

    include_previous_action = dataset.get("include_previous_action", False)
    if not isinstance(include_previous_action, bool):
        raise BCTrainingError("dataset.include_previous_action must be boolean.")
    if version == "level3/bc-training-v1":
        checkpoint_name = _required_string(output, "checkpoint_name")
        best_checkpoint_name = f"{Path(checkpoint_name).stem}_best{Path(checkpoint_name).suffix}"
    else:
        checkpoint_name = _required_string(output, "last_checkpoint_name")
        best_checkpoint_name = _required_string(output, "best_checkpoint_name")
    for field_name, file_name in (
        ("last_checkpoint_name", checkpoint_name),
        ("best_checkpoint_name", best_checkpoint_name),
    ):
        if Path(file_name).name != file_name:
            raise BCTrainingError(f"output.{field_name} must be a file name, not a path.")
    if checkpoint_name == best_checkpoint_name:
        raise BCTrainingError("best and last checkpoint names must be distinct.")

    return BCExperimentConfig(
        version=version,
        skill_name=_required_string(payload, "skill_name"),
        dataset_root=Path(_required_string(dataset, "root")),
        evaluation_config=Path(_required_string(dataset, "evaluation_config")),
        observation_fields=observation_fields,
        include_previous_action=include_previous_action,
        output_action_names=output_action_names,
        model=MLPConfig.from_dict(model),
        training=TrainingConfig.from_dict(training),
        output_dir=Path(_required_string(output, "directory")),
        checkpoint_name=checkpoint_name,
        best_checkpoint_name=best_checkpoint_name,
        source_digest=hashlib.sha256(raw).hexdigest(),
    )


def run_experiment(
    config: BCExperimentConfig,
    *,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Load one frozen task split and train the shared corrected baseline."""

    protocol_versions = {
        "reach_touch_target": "level3/reach-evaluation-v1",
        "button_press": "level3/button-evaluation-v1",
        "push_cube_to_target": "level3/push-evaluation-v1",
    }
    try:
        protocol_version = protocol_versions[config.skill_name]
    except KeyError as exc:
        raise BCTrainingError(
            f"unsupported Level 3 training skill {config.skill_name!r}."
        ) from exc
    root = config.dataset_root if dataset_root is None else Path(dataset_root)
    destination = config.output_dir if output_dir is None else Path(output_dir)
    bundle = load_frozen_skill_datasets(
        root,
        evaluation_config_path=config.evaluation_config,
        expected_version=protocol_version,
        expected_skill_name=config.skill_name,
        observation_fields=config.observation_fields,
        include_previous_action=config.include_previous_action,
        normalize=True,
    )
    effective = config.compatibility_dict()
    effective["dataset_root"] = str(root)
    return train_behavior_cloning(
        bundle,
        output_path=destination / config.checkpoint_name,
        best_output_path=destination / config.best_checkpoint_name,
        training_config=config.training,
        model_config=config.model,
        output_action_names=config.output_action_names,
        experiment_config_version=config.version,
        experiment_config_digest=_canonical_digest(effective),
        source_config_digest=config.source_digest,
        resume_from=resume_from,
    )


def train_behavior_cloning(
    bundle: DatasetBundle,
    *,
    output_path: str | Path,
    training_config: TrainingConfig,
    model_config: MLPConfig | None = None,
    output_action_names: Sequence[str] | None = None,
    experiment_config_version: str = BCExperimentConfig.SUPPORTED_VERSION,
    experiment_config_digest: str = "in-memory-config",
    source_config_digest: str | None = None,
    resume_from: str | Path | None = None,
    best_output_path: str | Path | None = None,
) -> TrainingResult:
    """Train and validate a schema-bound MLP, saving last and validation-best.

    Data-loader order is derived solely from ``seed + epoch``. Consequently,
    an epoch-boundary resume uses the same batches as an uninterrupted run.
    """

    training_config.__post_init__()
    _validate_bundle(bundle)
    first_episode = bundle.train.episodes[0]
    schema = PolicySchema.from_episode(
        first_episode, output_action_names=output_action_names
    )
    architecture = model_config or MLPConfig()
    device = torch.device(training_config.device)
    _seed_everything(training_config.seed)
    model = GoalConditionedMLP(schema, architecture).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    selected_epoch = 0
    selected_validation_loss = math.inf
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None

    provenance = _provenance(
        bundle,
        experiment_config_version=experiment_config_version,
        experiment_config_digest=experiment_config_digest,
        source_config_digest=source_config_digest,
    )
    if resume_from is not None:
        start_epoch, history = _restore_training_state(
            Path(resume_from),
            model=model,
            optimizer=optimizer,
            schema=schema,
            model_config=architecture,
            training_config=training_config,
            provenance=provenance,
        )
        selected = _select_best_history_entry(history)
        selected_epoch = int(selected["epoch"])
        selected_validation_loss = float(selected["validation_loss"])
    if start_epoch >= training_config.epochs:
        raise BCTrainingError(
            f"checkpoint already completed {start_epoch} epochs; requested total is "
            f"{training_config.epochs}."
        )

    last_checkpoint_path = Path(output_path)
    best_checkpoint_path = (
        Path(best_output_path)
        if best_output_path is not None
        else last_checkpoint_path.with_name(
            f"{last_checkpoint_path.stem}_best{last_checkpoint_path.suffix}"
        )
    )
    if last_checkpoint_path == best_checkpoint_path:
        raise BCTrainingError("best and last checkpoint paths must be distinct.")
    if start_epoch:
        best_model_state, best_optimizer_state = _restore_best_snapshot(
            best_checkpoint_path,
            selected_epoch=selected_epoch,
            selected_validation_loss=selected_validation_loss,
            fallback_model=model,
            fallback_optimizer=optimizer,
            completed_epochs=start_epoch,
            schema=schema,
            model_config=architecture,
            provenance=provenance,
        )
    last_digest = ""
    for epoch in range(start_epoch, training_config.epochs):
        train_loss = _train_epoch(
            model,
            bundle.train,
            optimizer,
            schema=schema,
            config=training_config,
            epoch=epoch,
            device=device,
        )
        validation_loss = evaluate_loss(
            model,
            bundle.validation,
            schema=schema,
            batch_size=training_config.batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < selected_validation_loss:
            selected_epoch = epoch + 1
            selected_validation_loss = validation_loss
            best_model_state = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
        last_digest = _save_training_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            completed_epochs=epoch + 1,
            history=history,
            provenance=provenance,
            device=device,
            checkpoint_role="last",
            state_epoch=epoch + 1,
            selected_epoch=selected_epoch,
            selected_validation_loss=selected_validation_loss,
        )

    if best_model_state is None or best_optimizer_state is None:
        raise BCTrainingError("training did not produce a validation-selected checkpoint.")
    final_model_state = deepcopy(model.state_dict())
    final_optimizer_state = deepcopy(optimizer.state_dict())
    model.load_state_dict(best_model_state, strict=True)
    optimizer.load_state_dict(best_optimizer_state)
    best_digest = _save_training_checkpoint(
        best_checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        completed_epochs=training_config.epochs,
        history=history,
        provenance=provenance,
        device=device,
        checkpoint_role="best_validation",
        state_epoch=selected_epoch,
        selected_epoch=selected_epoch,
        selected_validation_loss=selected_validation_loss,
    )
    model.load_state_dict(final_model_state, strict=True)
    optimizer.load_state_dict(final_optimizer_state)

    return TrainingResult(
        checkpoint_path=last_checkpoint_path,
        digest_path=_digest_path(last_checkpoint_path),
        checkpoint_digest=last_digest,
        completed_epochs=training_config.epochs,
        loss_history=tuple(history),
        best_checkpoint_path=best_checkpoint_path,
        best_digest_path=_digest_path(best_checkpoint_path),
        best_checkpoint_digest=best_digest,
        last_checkpoint_path=last_checkpoint_path,
        last_digest_path=_digest_path(last_checkpoint_path),
        last_checkpoint_digest=last_digest,
        selected_epoch=selected_epoch,
        selected_validation_loss=selected_validation_loss,
    )


def evaluate_loss(
    model: GoalConditionedMLP,
    dataset: GoalConditionedSkillDataset,
    *,
    schema: PolicySchema,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> float:
    """Compute read-only mean validation MSE without changing model parameters."""

    if len(dataset) == 0:
        raise BCTrainingError("validation dataset must contain at least one sample.")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            observations = batch["obs"].to(device)
            goals = batch["goal"].to(device)
            actions = batch["action"].to(device)
            targets = schema.select_action_targets(actions)
            predictions = model(observations, goals)
            loss = nn.functional.mse_loss(predictions, targets, reduction="sum")
            total_loss += float(loss.item())
            total_samples += targets.numel()
    model.train(was_training)
    if total_samples == 0:
        raise BCTrainingError("validation dataset produced no action targets.")
    return total_loss / total_samples


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_epoch(
    model: GoalConditionedMLP,
    dataset: GoalConditionedSkillDataset,
    optimizer: torch.optim.Optimizer,
    *,
    schema: PolicySchema,
    config: TrainingConfig,
    epoch: int,
    device: torch.device,
) -> float:
    generator = torch.Generator()
    generator.manual_seed(config.seed + epoch)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    model.train()
    total_loss = 0.0
    total_values = 0
    for batch in loader:
        observations = batch["obs"].to(device)
        goals = batch["goal"].to(device)
        actions = batch["action"].to(device)
        targets = schema.select_action_targets(actions)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(observations, goals)
        loss = nn.functional.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise BCTrainingError(f"training loss became non-finite in epoch {epoch + 1}.")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * targets.numel()
        total_values += targets.numel()
    if total_values == 0:
        raise BCTrainingError("training dataset produced no action targets.")
    return total_loss / total_values


def _validate_bundle(bundle: DatasetBundle) -> None:
    if not bundle.train.episodes or len(bundle.train) == 0:
        raise BCTrainingError("training dataset must contain at least one episode and sample.")
    if not bundle.validation.episodes or len(bundle.validation) == 0:
        raise BCTrainingError("validation dataset must contain at least one episode and sample.")
    if bundle.train.normalization is None or bundle.validation.normalization is None:
        raise BCTrainingError(
            "training and validation datasets must use training-only normalization."
        )
    normalization = bundle.normalization
    if normalization.source_split != "train":
        raise BCTrainingError("normalization source_split must be 'train'.")
    if normalization.dataset_digest != bundle.manifest.dataset_digest:
        raise BCTrainingError("normalization and split manifest dataset digests differ.")
    for name, dataset in (("train", bundle.train), ("validation", bundle.validation)):
        if dataset.normalization.dataset_digest != normalization.dataset_digest:
            raise BCTrainingError(f"{name} dataset uses incompatible normalization metadata.")


def _provenance(
    bundle: DatasetBundle,
    *,
    experiment_config_version: str,
    experiment_config_digest: str,
    source_config_digest: str | None,
) -> dict[str, Any]:
    manifest = bundle.manifest.to_dict()
    normalization = bundle.normalization.to_dict()
    return {
        "experiment_config_version": experiment_config_version,
        "experiment_config_digest": experiment_config_digest,
        "source_config_digest": source_config_digest,
        "dataset_digest": bundle.manifest.dataset_digest,
        "split_manifest_digest": _canonical_digest(manifest),
        "split_manifest": manifest,
        "normalization_digest": _canonical_digest(normalization),
        "normalization": normalization,
    }


def _save_training_checkpoint(
    path: Path,
    *,
    model: GoalConditionedMLP,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    completed_epochs: int,
    history: list[dict[str, float | int]],
    provenance: Mapping[str, Any],
    device: torch.device,
    checkpoint_role: str,
    state_epoch: int,
    selected_epoch: int,
    selected_validation_loss: float,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        # Keep the Level 3.2 fields at top level so GoalConditionedMLP.load()
        # can load a trained checkpoint directly.
        "checkpoint_version": model.CHECKPOINT_VERSION,
        "schema": model.schema.to_dict(),
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "training_checkpoint_version": "dexvision/bc-training-v1",
        "checkpoint_selection_version": "dexvision/offline-validation-selection-v1",
        "checkpoint_role": checkpoint_role,
        "training_config": training_config.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epochs": completed_epochs,
        "state_epoch": state_epoch,
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss",
        "selected_validation_loss": selected_validation_loss,
        "selection_tie_break": "earliest_epoch",
        "loss_history": list(history),
        "provenance": dict(provenance),
        "environment": _environment_metadata(device),
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    digest_path = _digest_path(path)
    digest_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _restore_training_state(
    path: Path,
    *,
    model: GoalConditionedMLP,
    optimizer: torch.optim.Optimizer,
    schema: PolicySchema,
    model_config: MLPConfig,
    training_config: TrainingConfig,
    provenance: Mapping[str, Any],
) -> tuple[int, list[dict[str, float | int]]]:
    _verify_digest_sidecar_when_present(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise BCTrainingError(f"cannot load training checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BCTrainingError(f"training checkpoint {path} must contain a mapping.")
    if payload.get("training_checkpoint_version") != "dexvision/bc-training-v1":
        raise BCTrainingError(f"training checkpoint {path} has an unsupported version.")
    state_epoch = payload.get("state_epoch", payload.get("completed_epochs"))
    completed_for_run = payload.get("completed_epochs")
    if state_epoch != completed_for_run:
        raise BCTrainingError(
            "resume requires a last checkpoint whose state matches its completed epoch."
        )
    if payload.get("schema") != schema.to_dict():
        raise BCTrainingError("resume checkpoint policy schema is incompatible.")
    if payload.get("config") != model_config.to_dict():
        raise BCTrainingError("resume checkpoint model config is incompatible.")
    saved_training = payload.get("training_config")
    if not isinstance(saved_training, Mapping):
        raise BCTrainingError("resume checkpoint is missing training_config metadata.")
    for key, value in training_config.to_dict().items():
        if key != "epochs" and saved_training.get(key) != value:
            raise BCTrainingError(f"resume checkpoint training setting {key!r} differs.")
    saved_provenance = payload.get("provenance")
    if not isinstance(saved_provenance, Mapping):
        raise BCTrainingError("resume checkpoint is missing provenance metadata.")
    for key in (
        "experiment_config_version",
        "experiment_config_digest",
        "dataset_digest",
        "split_manifest_digest",
        "normalization_digest",
    ):
        if saved_provenance.get(key) != provenance.get(key):
            raise BCTrainingError(f"resume checkpoint provenance {key!r} differs.")

    state_dict = payload.get("state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    history = payload.get("loss_history")
    completed = payload.get("completed_epochs")
    if not isinstance(state_dict, Mapping) or not isinstance(optimizer_state, Mapping):
        raise BCTrainingError("resume checkpoint is missing model or optimizer state.")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise BCTrainingError("resume checkpoint has invalid completed_epochs.")
    if not isinstance(history, list) or len(history) != completed:
        raise BCTrainingError("resume checkpoint loss_history is inconsistent.")
    try:
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(optimizer_state)
    except (RuntimeError, ValueError) as exc:
        raise BCTrainingError(f"cannot restore training state: {exc}") from exc
    return completed, [dict(item) for item in history]


def _select_best_history_entry(
    history: Sequence[Mapping[str, float | int]],
) -> Mapping[str, float | int]:
    """Select the lowest offline validation loss, breaking ties by earliest epoch."""

    if not history:
        raise BCTrainingError("cannot select a checkpoint from empty loss history.")
    try:
        return min(
            history,
            key=lambda item: (float(item["validation_loss"]), int(item["epoch"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BCTrainingError("loss history is missing valid epoch/validation_loss values.") from exc


def _restore_best_snapshot(
    path: Path,
    *,
    selected_epoch: int,
    selected_validation_loss: float,
    fallback_model: GoalConditionedMLP,
    fallback_optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    schema: PolicySchema,
    model_config: MLPConfig,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the earlier best snapshot needed by an epoch-boundary resume."""

    if selected_epoch == completed_epochs and not path.is_file():
        return deepcopy(fallback_model.state_dict()), deepcopy(
            fallback_optimizer.state_dict()
        )
    _verify_digest_sidecar_when_present(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise BCTrainingError(
            f"cannot restore validation-best checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise BCTrainingError(f"validation-best checkpoint {path} must contain a mapping.")
    if payload.get("state_epoch") != selected_epoch:
        raise BCTrainingError(
            "validation-best checkpoint epoch does not match resumed loss history."
        )
    if payload.get("checkpoint_role") != "best_validation":
        raise BCTrainingError("validation-best checkpoint has the wrong role.")
    if payload.get("schema") != schema.to_dict():
        raise BCTrainingError("validation-best checkpoint policy schema is incompatible.")
    if payload.get("config") != model_config.to_dict():
        raise BCTrainingError("validation-best checkpoint model config is incompatible.")
    if payload.get("selected_validation_loss") != selected_validation_loss:
        raise BCTrainingError(
            "validation-best checkpoint metric does not match resumed loss history."
        )
    saved_provenance = payload.get("provenance")
    if not isinstance(saved_provenance, Mapping):
        raise BCTrainingError("validation-best checkpoint is missing provenance metadata.")
    for key in (
        "experiment_config_version",
        "experiment_config_digest",
        "dataset_digest",
        "split_manifest_digest",
        "normalization_digest",
    ):
        if saved_provenance.get(key) != provenance.get(key):
            raise BCTrainingError(
                f"validation-best checkpoint provenance {key!r} differs."
            )
    state_dict = payload.get("state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(state_dict, Mapping) or not isinstance(optimizer_state, Mapping):
        raise BCTrainingError("validation-best checkpoint is missing model or optimizer state.")
    return deepcopy(dict(state_dict)), deepcopy(dict(optimizer_state))


def _verify_digest_sidecar_when_present(path: Path) -> None:
    digest_path = _digest_path(path)
    if not digest_path.exists():
        return
    try:
        fields = digest_path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise BCTrainingError(f"cannot read checkpoint digest {digest_path}: {exc}") from exc
    if len(fields) != 2 or fields[1] != path.name:
        raise BCTrainingError(f"invalid checkpoint digest sidecar {digest_path}.")
    actual = file_sha256(path)
    if fields[0] != actual:
        raise BCTrainingError(f"checkpoint SHA-256 verification failed for {path}.")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "executable": Path(sys.executable).name,
    }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BCTrainingError(f"training config {key} must be a mapping.")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise BCTrainingError(f"training config {key} must be a non-empty string.")
    return value
