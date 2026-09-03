"""Controlled Level 3 data, action-space, and goal-input diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dexvision.evaluation.evaluate_policy import (
    evaluate_manipulation_policy,
    evaluate_policy,
    load_manipulation_evaluation_protocol,
    load_reach_evaluation_protocol,
)
from dexvision.learning.datasets import (
    ELIGIBILITY_QUALITY_PASSED_SUCCESS,
    ELIGIBILITY_RECOMPUTED_SUCCESS,
    GOAL_INPUT_CONDITIONED,
    GOAL_INPUT_FIXED_TRAINING_MEAN,
    load_frozen_skill_datasets,
    load_skill_episodes,
)
from dexvision.learning.models import (
    BASE_ORIENTATION_ACTION_NAMES,
    BASE_POSITION_ACTION_NAMES,
    FINGER_ACTION_PREFIX,
)
from dexvision.learning.policies import CheckpointPolicy, load_checkpoint_policy
from dexvision.learning.train_bc import (
    BCExperimentConfig,
    load_experiment_config,
    run_experiment,
)


class Level3DiagnosticsError(RuntimeError):
    """Raised when the diagnostic matrix is incomplete or not comparable."""


@dataclass(frozen=True)
class TaskDiagnosticSource:
    """Existing full-action baseline artifacts for one frozen task."""

    task_id: str
    training_config: Path
    baseline_checkpoint: Path
    baseline_report: Path


@dataclass(frozen=True)
class Level3DiagnosticsConfig:
    """Validated file-backed definition of the Level 3.6 matrix."""

    version: str
    dataset_root: Path
    output_directory: Path
    tasks: tuple[TaskDiagnosticSource, ...]
    source_path: Path
    source_digest: str


@dataclass(frozen=True)
class DiagnosticExperiment:
    """One controlled baseline or ablation in the diagnostic matrix."""

    experiment_id: str
    task_id: str
    comparison_family: str
    variant: str
    training_config: BCExperimentConfig
    checkpoint_path: Path
    rollout_report_path: Path
    output_action_names: tuple[str, ...]
    eligibility: str
    goal_input_mode: str
    is_existing_baseline: bool = False
    reference_split_assignments: tuple[tuple[str, str], ...] = ()


_PROTOCOL_VERSIONS = {
    "reach_touch_target": "level3/reach-evaluation-v1",
    "button_press": "level3/button-evaluation-v1",
    "push_cube_to_target": "level3/push-evaluation-v1",
}
_TASKS = tuple(_PROTOCOL_VERSIONS)


def load_level3_diagnostics_config(path: str | Path) -> Level3DiagnosticsConfig:
    """Load the versioned Level 3.6 matrix and its preserved baselines."""

    source_path = Path(path)
    try:
        raw = source_path.read_bytes()
        payload = yaml.safe_load(raw)
    except OSError as exc:
        raise Level3DiagnosticsError(
            f"cannot read diagnostics config {source_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise Level3DiagnosticsError(
            f"invalid diagnostics YAML {source_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Level3DiagnosticsError("diagnostics config must contain a mapping.")
    version = _required_string(payload, "version")
    if version != "level3/data-action-diagnostics-v1":
        raise Level3DiagnosticsError(
            f"unsupported diagnostics version {version!r}."
        )
    tasks_value = payload.get("tasks")
    if not isinstance(tasks_value, Mapping) or set(tasks_value) != set(_TASKS):
        raise Level3DiagnosticsError(
            f"diagnostics tasks must be exactly {list(_TASKS)!r}."
        )
    tasks: list[TaskDiagnosticSource] = []
    for task_id in _TASKS:
        item = tasks_value[task_id]
        if not isinstance(item, Mapping):
            raise Level3DiagnosticsError(f"task {task_id!r} must be a mapping.")
        tasks.append(
            TaskDiagnosticSource(
                task_id=task_id,
                training_config=Path(_required_string(item, "training_config")),
                baseline_checkpoint=Path(
                    _required_string(item, "baseline_checkpoint")
                ),
                baseline_report=Path(_required_string(item, "baseline_report")),
            )
        )
    return Level3DiagnosticsConfig(
        version=version,
        dataset_root=Path(_required_string(payload, "dataset_root")),
        output_directory=Path(_required_string(payload, "output_directory")),
        tasks=tuple(tasks),
        source_path=source_path,
        source_digest=hashlib.sha256(raw).hexdigest(),
    )


def build_diagnostic_matrix(
    config: Level3DiagnosticsConfig,
) -> tuple[DiagnosticExperiment, ...]:
    """Build explicit full/base/finger/goal/data experiments from baselines."""

    experiments: list[DiagnosticExperiment] = []
    for source in config.tasks:
        base_config = load_experiment_config(source.training_config)
        if base_config.skill_name != source.task_id:
            raise Level3DiagnosticsError(
                f"{source.training_config} targets {base_config.skill_name!r}, "
                f"not {source.task_id!r}."
            )
        action_names = _checkpoint_action_names(source.baseline_checkpoint)
        if base_config.output_action_names is not None:
            raise Level3DiagnosticsError(
                f"baseline {source.training_config} is not a full-action config."
            )
        baseline = DiagnosticExperiment(
            experiment_id=f"{source.task_id}/full",
            task_id=source.task_id,
            comparison_family="baseline",
            variant="full",
            training_config=base_config,
            checkpoint_path=source.baseline_checkpoint,
            rollout_report_path=source.baseline_report,
            output_action_names=action_names,
            eligibility=ELIGIBILITY_QUALITY_PASSED_SUCCESS,
            goal_input_mode=GOAL_INPUT_CONDITIONED,
            is_existing_baseline=True,
        )
        experiments.append(baseline)
        base_names = BASE_POSITION_ACTION_NAMES + BASE_ORIENTATION_ACTION_NAMES
        finger_names = tuple(
            name for name in action_names if name.startswith(FINGER_ACTION_PREFIX)
        )
        if action_names != base_names + finger_names:
            raise Level3DiagnosticsError(
                f"{source.baseline_checkpoint} has an incompatible action layout."
            )
        for variant, output_names in (
            ("base_only", base_names),
            ("finger_only", finger_names),
        ):
            experiments.append(
                _derived_experiment(
                    config,
                    source.task_id,
                    base_config,
                    action_names,
                    comparison_family="action_space",
                    variant=variant,
                    output_action_names=output_names,
                )
            )
        experiments.append(
            _derived_experiment(
                config,
                source.task_id,
                base_config,
                action_names,
                comparison_family="goal_input",
                variant="fixed_training_mean",
                output_action_names=action_names,
                goal_input_mode=GOAL_INPUT_FIXED_TRAINING_MEAN,
            )
        )
        if source.task_id == "reach_touch_target":
            reference_assignments = _checkpoint_split_assignments(
                source.baseline_checkpoint
            )
            experiments.append(
                _derived_experiment(
                    config,
                    source.task_id,
                    base_config,
                    action_names,
                    comparison_family="data_quality",
                    variant="broader_recomputed_success",
                    output_action_names=action_names,
                    eligibility=ELIGIBILITY_RECOMPUTED_SUCCESS,
                    reference_split_assignments=reference_assignments,
                )
            )
    _validate_matrix(experiments)
    return tuple(experiments)


def run_level3_diagnostics(
    config: Level3DiagnosticsConfig,
    *,
    model_path: str | Path = "assets/mujoco/task_board_scene.xml",
) -> dict[str, Any]:
    """Train/evaluate every new variant and save JSON/CSV diagnostic tables."""

    experiments = build_diagnostic_matrix(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    field_errors: dict[str, Any] = {}
    reports: dict[str, Mapping[str, Any]] = {}
    checkpoints: dict[str, Mapping[str, Any]] = {}

    for experiment in experiments:
        if not experiment.is_existing_baseline:
            run_experiment(
                experiment.training_config,
                reference_split_assignments=(
                    dict(experiment.reference_split_assignments)
                    if experiment.reference_split_assignments
                    else None
                ),
            )
            policy = load_checkpoint_policy(experiment.checkpoint_path)
            if experiment.task_id == "reach_touch_target":
                protocol = load_reach_evaluation_protocol(
                    experiment.training_config.evaluation_config
                )
                evaluate_policy(
                    policy,
                    protocol,
                    output_dir=experiment.rollout_report_path.parent,
                    model_path=model_path,
                    ablation_name=(
                        experiment.variant
                        if experiment.comparison_family == "action_space"
                        else None
                    ),
                )
            else:
                protocol = load_manipulation_evaluation_protocol(
                    experiment.training_config.evaluation_config
                )
                evaluate_manipulation_policy(
                    policy,
                    protocol,
                    output_dir=experiment.rollout_report_path.parent,
                    model_path=model_path,
                    ablation_name=(
                        experiment.variant
                        if experiment.comparison_family == "action_space"
                        else None
                    ),
                )
        report = _load_json(experiment.rollout_report_path, label="rollout report")
        checkpoint = _load_checkpoint(experiment.checkpoint_path)
        reports[experiment.experiment_id] = report
        checkpoints[experiment.experiment_id] = checkpoint
        policy = load_checkpoint_policy(experiment.checkpoint_path)
        bundle = _load_raw_validation_bundle(experiment)
        field_errors[experiment.experiment_id] = offline_action_errors(policy, bundle)
        rows.append(_table_row(experiment, report, checkpoint, bundle))

    fairness = _validate_comparability(experiments, reports, checkpoints)
    data_availability = _data_availability(config)
    payload = {
        "version": config.version,
        "config_path": str(config.source_path),
        "config_digest": config.source_digest,
        "dataset_root": str(config.dataset_root),
        "comparison_controls": fairness,
        "data_quality_availability": data_availability,
        "action_subset_reconstruction": {
            "strategy": "hold_previous_applied_action",
            "reversible": True,
            "description": (
                "Each subset keeps the complete saved Level 1.13 layout. Fields "
                "outside the declared output subset hold their prior applied value."
            ),
        },
        "summary_table": rows,
        "offline_error_by_action_field_and_goal": field_errors,
        "conclusions": _conclusions(rows, data_availability),
    }
    report_path = config.output_directory / "report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(config.output_directory / "summary.csv", rows)
    return payload


def offline_action_errors(
    policy: CheckpointPolicy, bundle: Any
) -> dict[str, Any]:
    """Measure raw validation action error by predicted field and goal id."""

    field_squares: dict[str, list[float]] = defaultdict(list)
    field_absolutes: dict[str, list[float]] = defaultdict(list)
    goal_squares: dict[str, list[float]] = defaultdict(list)
    indices = [
        policy.dataset_action_names.index(name) for name in policy.output_action_names
    ]
    for episode in bundle.validation.episodes:
        for observation, target in zip(
            episode.observations, episode.actions, strict=True
        ):
            prediction = policy.predict(observation, episode.goal)
            expected = target[indices]
            errors = prediction - expected
            for name, error in zip(policy.output_action_names, errors, strict=True):
                field_squares[name].append(float(error * error))
                field_absolutes[name].append(float(abs(error)))
            goal_squares[episode.goal_id].extend(float(value * value) for value in errors)
    return {
        "validation_episode_count": len(bundle.validation.episodes),
        "validation_sample_count": len(bundle.validation),
        "predicted_action_names": list(policy.output_action_names),
        "by_action_field": {
            name: {
                "rmse": float(np.sqrt(np.mean(field_squares[name]))),
                "mae": float(np.mean(field_absolutes[name])),
            }
            for name in policy.output_action_names
        },
        "by_goal_condition": {
            goal_id: {
                "rmse": float(np.sqrt(np.mean(values))),
                "value_count": len(values),
            }
            for goal_id, values in sorted(goal_squares.items())
        },
    }


def _derived_experiment(
    config: Level3DiagnosticsConfig,
    task_id: str,
    base_config: BCExperimentConfig,
    dataset_action_names: tuple[str, ...],
    *,
    comparison_family: str,
    variant: str,
    output_action_names: tuple[str, ...],
    eligibility: str = ELIGIBILITY_QUALITY_PASSED_SUCCESS,
    goal_input_mode: str = GOAL_INPUT_CONDITIONED,
    reference_split_assignments: tuple[tuple[str, str], ...] = (),
) -> DiagnosticExperiment:
    experiment_id = f"{task_id}/{variant}"
    checkpoint_dir = config.output_directory / "checkpoints" / task_id / variant
    rollout_dir = config.output_directory / "rollouts" / task_id / variant
    derived = replace(
        base_config,
        dataset_root=config.dataset_root,
        eligibility=eligibility,
        goal_input_mode=goal_input_mode,
        output_action_names=(
            None if output_action_names == dataset_action_names else output_action_names
        ),
        output_dir=checkpoint_dir,
    )
    return DiagnosticExperiment(
        experiment_id=experiment_id,
        task_id=task_id,
        comparison_family=comparison_family,
        variant=variant,
        training_config=derived,
        checkpoint_path=checkpoint_dir / derived.best_checkpoint_name,
        rollout_report_path=rollout_dir / "report.json",
        output_action_names=output_action_names,
        eligibility=eligibility,
        goal_input_mode=goal_input_mode,
        reference_split_assignments=reference_split_assignments,
    )


def _load_raw_validation_bundle(experiment: DiagnosticExperiment) -> Any:
    return load_frozen_skill_datasets(
        experiment.training_config.dataset_root,
        evaluation_config_path=experiment.training_config.evaluation_config,
        expected_version=_PROTOCOL_VERSIONS[experiment.task_id],
        expected_skill_name=experiment.task_id,
        observation_fields=experiment.training_config.observation_fields,
        include_previous_action=experiment.training_config.include_previous_action,
        normalize=False,
        eligibility=experiment.eligibility,
        goal_input_mode=GOAL_INPUT_CONDITIONED,
        reference_split_assignments=(
            dict(experiment.reference_split_assignments)
            if experiment.reference_split_assignments
            else None
        ),
    )


def _table_row(
    experiment: DiagnosticExperiment,
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    bundle: Any,
) -> dict[str, Any]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise Level3DiagnosticsError(
            f"{experiment.rollout_report_path} is missing metrics."
        )
    training_rate = metrics.get(
        "training_target_success_rate", metrics.get("training_goal_success_rate")
    )
    held_out_rate = metrics.get(
        "held_out_target_success_rate", metrics.get("held_out_goal_success_rate")
    )
    final_error = metrics.get("mean_final_distance_m", metrics.get("mean_final_task_error"))
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Level3DiagnosticsError(
            f"{experiment.checkpoint_path} is missing provenance."
        )
    return {
        "experiment_id": experiment.experiment_id,
        "task_id": experiment.task_id,
        "comparison_family": experiment.comparison_family,
        "variant": experiment.variant,
        "eligibility": experiment.eligibility,
        "goal_input_mode": experiment.goal_input_mode,
        "output_action_count": len(experiment.output_action_names),
        "output_action_names": list(experiment.output_action_names),
        "train_episode_count": len(bundle.train.episodes),
        "validation_episode_count": len(bundle.validation.episodes),
        "selected_epoch": int(report.get("selected_epoch", 0)),
        "selected_validation_loss": float(
            report.get("selected_validation_loss", float("nan"))
        ),
        "training_success_rate": float(training_rate),
        "held_out_success_rate": float(held_out_rate),
        "mean_final_task_error": float(final_error),
        "mean_normalized_action_jerk": float(
            metrics["mean_normalized_action_jerk"]
        ),
        "invalid_action_count": int(metrics["invalid_action_count"]),
        "workspace_violation_count": int(metrics["workspace_violation_count"]),
        "object_workspace_violation_count": int(
            metrics.get("object_workspace_violation_count", 0)
        ),
        "joint_limit_violation_count": int(
            metrics["joint_limit_violation_count"]
        ),
        "dataset_digest": str(report["dataset_digest"]),
        "split_manifest_digest": str(report["split_manifest_digest"]),
        "checkpoint_digest": str(report["checkpoint_digest"]),
        "protocol_digest": str(report["protocol_digest"]),
        "training_config_digest": str(report["training_config_digest"]),
        "model_hidden_dims": list(checkpoint["config"]["hidden_dims"]),
        "training_seed": int(checkpoint["training_config"]["seed"]),
        "split_seed": int(provenance["split_manifest"]["seed"]),
    }


def _validate_matrix(experiments: Sequence[DiagnosticExperiment]) -> None:
    ids = [item.experiment_id for item in experiments]
    if len(ids) != len(set(ids)):
        raise Level3DiagnosticsError("diagnostic experiment ids must be unique.")
    by_task: dict[str, set[str]] = defaultdict(set)
    for item in experiments:
        by_task[item.task_id].add(item.variant)
    expected = {"full", "base_only", "finger_only", "fixed_training_mean"}
    for task_id in _TASKS:
        if not expected.issubset(by_task[task_id]):
            raise Level3DiagnosticsError(
                f"diagnostic matrix for {task_id!r} is incomplete."
            )
    if "broader_recomputed_success" not in by_task["reach_touch_target"]:
        raise Level3DiagnosticsError("reach data-quality comparison is missing.")


def _validate_comparability(
    experiments: Sequence[DiagnosticExperiment],
    reports: Mapping[str, Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    by_task: dict[str, list[DiagnosticExperiment]] = defaultdict(list)
    for item in experiments:
        by_task[item.task_id].append(item)
    for task_id, items in by_task.items():
        baseline = next(item for item in items if item.variant == "full")
        base_report = reports[baseline.experiment_id]
        base_checkpoint = checkpoints[baseline.experiment_id]
        base_results = _scenario_ids(base_report)
        base_training = base_checkpoint.get("training_config")
        base_model = base_checkpoint.get("config")
        if not isinstance(base_training, Mapping) or not isinstance(base_model, Mapping):
            raise Level3DiagnosticsError("baseline checkpoint metadata is incomplete.")
        task_controls: dict[str, Any] = {}
        for item in items:
            report = reports[item.experiment_id]
            checkpoint = checkpoints[item.experiment_id]
            training = checkpoint.get("training_config")
            model = checkpoint.get("config")
            if training != base_training or model != base_model:
                raise Level3DiagnosticsError(
                    f"{item.experiment_id} changed model or training settings."
                )
            if report.get("protocol_digest") != base_report.get("protocol_digest"):
                raise Level3DiagnosticsError(
                    f"{item.experiment_id} changed the frozen protocol."
                )
            if _scenario_ids(report) != base_results:
                raise Level3DiagnosticsError(
                    f"{item.experiment_id} changed the frozen scenario matrix."
                )
            same_dataset = report.get("dataset_digest") == base_report.get(
                "dataset_digest"
            )
            if item.comparison_family != "data_quality" and not same_dataset:
                raise Level3DiagnosticsError(
                    f"{item.experiment_id} changed data outside the data comparison."
                )
            task_controls[item.variant] = {
                "same_training_settings": True,
                "same_hidden_layer_size": True,
                "same_frozen_protocol": True,
                "same_scenario_order": True,
                "same_dataset_and_split": same_dataset,
                "intentional_dataset_change": item.comparison_family == "data_quality",
                "shared_episode_assignments_preserved": _shared_assignments_preserved(
                    base_checkpoint, checkpoint
                ),
            }
        controls[task_id] = task_controls
    return controls


def _data_availability(config: Level3DiagnosticsConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id in _TASKS:
        clean = load_skill_episodes(
            config.dataset_root,
            skill_name=task_id,
            eligibility=ELIGIBILITY_QUALITY_PASSED_SUCCESS,
        )
        broader = load_skill_episodes(
            config.dataset_root,
            skill_name=task_id,
            eligibility=ELIGIBILITY_RECOMPUTED_SUCCESS,
        )
        available = len(broader) > len(clean)
        result[task_id] = {
            "quality_passed_success_count": len(clean),
            "recomputed_success_count": len(broader),
            "comparison_available": available,
            "limitation": (
                None
                if available
                else "No recomputed-success episodes exist outside the quality-passed set."
            ),
        }
    return result


def _conclusions(
    rows: Sequence[Mapping[str, Any]], data_availability: Mapping[str, Any]
) -> dict[str, list[str]]:
    by_task = defaultdict(dict)
    for row in rows:
        by_task[str(row["task_id"])][str(row["variant"])] = row
    measured: list[str] = []
    for task_id, variants in by_task.items():
        full = variants["full"]
        for variant in ("base_only", "finger_only", "fixed_training_mean"):
            item = variants[variant]
            measured.append(
                f"{task_id}: {variant} changed training/held-out success by "
                f"{item['training_success_rate'] - full['training_success_rate']:+.3f}/"
                f"{item['held_out_success_rate'] - full['held_out_success_rate']:+.3f} "
                "relative to full conditioned control."
            )
        broader = variants.get("broader_recomputed_success")
        if broader is not None:
            measured.append(
                f"{task_id}: adding recomputed-success, quality-failed episodes changed "
                f"training/held-out success by "
                f"{broader['training_success_rate'] - full['training_success_rate']:+.3f}/"
                f"{broader['held_out_success_rate'] - full['held_out_success_rate']:+.3f}."
            )
    for task_id, item in data_availability.items():
        if not item["comparison_available"]:
            measured.append(f"{task_id}: data-quality comparison unavailable; {item['limitation']}")
    hypotheses = [
        "Differences between low offline error and failed rollouts may indicate compounding error or missing recovery coverage, but this diagnostic does not prove either cause.",
        "Safety-limit terminations may reflect action-distribution tails or state-distribution shift; causal attribution requires a later targeted experiment.",
        "Level 2 has no genuine session identifiers, so none of these effects establish cross-session generalization.",
    ]
    return {"measured": measured, "hypotheses": hypotheses}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = [
        key
        for key, value in rows[0].items()
        if not isinstance(value, (list, dict))
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_keys})


def _checkpoint_action_names(path: Path) -> tuple[str, ...]:
    checkpoint = _load_checkpoint(path)
    schema = checkpoint.get("schema")
    names = schema.get("dataset_action_names") if isinstance(schema, Mapping) else None
    if not isinstance(names, Sequence) or isinstance(names, str):
        raise Level3DiagnosticsError(f"{path} is missing its dataset action layout.")
    return tuple(str(name) for name in names)


def _checkpoint_split_assignments(path: Path) -> tuple[tuple[str, str], ...]:
    checkpoint = _load_checkpoint(path)
    provenance = checkpoint.get("provenance")
    manifest = provenance.get("split_manifest") if isinstance(provenance, Mapping) else None
    assignments = manifest.get("assignments") if isinstance(manifest, Mapping) else None
    if not isinstance(assignments, list):
        raise Level3DiagnosticsError(f"{path} is missing split assignments.")
    result: list[tuple[str, str]] = []
    for item in assignments:
        if not isinstance(item, Mapping):
            raise Level3DiagnosticsError(f"{path} has an invalid split assignment.")
        episode_id = item.get("episode_id")
        split = item.get("split")
        if not isinstance(episode_id, str) or not isinstance(split, str):
            raise Level3DiagnosticsError(f"{path} has an invalid split assignment.")
        result.append((episode_id, split))
    return tuple(sorted(result))


def _shared_assignments_preserved(
    baseline: Mapping[str, Any], comparison: Mapping[str, Any]
) -> bool:
    def assignments(checkpoint: Mapping[str, Any]) -> dict[str, str]:
        provenance = checkpoint.get("provenance")
        manifest = provenance.get("split_manifest") if isinstance(provenance, Mapping) else None
        values = manifest.get("assignments") if isinstance(manifest, Mapping) else None
        if not isinstance(values, list):
            raise Level3DiagnosticsError("checkpoint is missing split assignments.")
        return {
            str(item["episode_id"]): str(item["split"])
            for item in values
            if isinstance(item, Mapping)
        }

    left = assignments(baseline)
    right = assignments(comparison)
    shared = set(left) & set(right)
    return bool(shared) and all(left[item] == right[item] for item in shared)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise Level3DiagnosticsError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise Level3DiagnosticsError(f"checkpoint {path} must contain a mapping.")
    return payload


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Level3DiagnosticsError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise Level3DiagnosticsError(f"{label} {path} must contain a mapping.")
    return payload


def _scenario_ids(report: Mapping[str, Any]) -> list[Any]:
    results = report.get("results")
    if not isinstance(results, list):
        raise Level3DiagnosticsError("rollout report is missing results.")
    return [item.get("scenario_id") for item in results if isinstance(item, Mapping)]


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise Level3DiagnosticsError(f"diagnostics config {key!r} must be a string.")
    return value
