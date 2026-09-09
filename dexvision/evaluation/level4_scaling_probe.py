"""Validation-only Level 4.5B nested data-scaling probe."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dexvision.logging.level4_collection import (
    LEVEL4_NOMINAL_GROUPS,
    PilotEpisode,
    WorkcellPilotTask,
    discover_pilot_episodes,
    load_level4_collection_config,
)
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.logging.session_manifest import load_session_manifest


SCALING_REPORT_VERSION = "level4/scaling-probe-report-v1"


class Level4ScalingProbeError(ValueError):
    """Raised when the frozen scaling probe cannot be evaluated."""


class _ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(width, int(hidden)), nn.Tanh()))
            width = int(hidden)
        layers.append(nn.Linear(width, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def run_level4_scaling_probe(
    *,
    config_path: str | Path,
    dataset_dir: str | Path,
    workcell_config: str | Path = "configs/workcell.yaml",
) -> dict[str, Any]:
    """Run the frozen 4/8/16 train-only comparison on one validation matrix."""

    config, _ = load_level4_collection_config(config_path)
    expansion = _mapping(config, "level4_5b_procedural_expansion")
    probe = _mapping(expansion, "scaling_probe")
    model_config = _mapping(probe, "model")
    gates = _mapping(probe, "readiness_gates")
    cells = {
        str(cell["id"]): cell
        for cell in config["coverage_cells"]
        if isinstance(cell, Mapping)
        and cell.get("data_group") in LEVEL4_NOMINAL_GROUPS
    }
    session_manifest = load_session_manifest(
        Path(dataset_dir) / "session_manifest.json"
    )
    sessions = {
        item.recording_session_id: item.split
        for item in session_manifest.sessions
    }
    accepted = [
        episode
        for episode in discover_pilot_episodes(dataset_dir)
        if episode.expert_accepted
        and episode.goal_condition_id in cells
        and sessions.get(episode.session_id) in {"train", "validation"}
    ]
    by_split_cell: dict[tuple[str, str], list[PilotEpisode]] = defaultdict(list)
    for episode in accepted:
        split = sessions[episode.session_id]
        cell = cells[episode.goal_condition_id]
        if split != cell["split_owner"]:
            raise Level4ScalingProbeError(
                f"episode {episode.episode_id!r} violates frozen split ownership."
            )
        by_split_cell[(split, episode.goal_condition_id)].append(episode)
    for episodes in by_split_cell.values():
        episodes.sort(key=_episode_order_key)

    train_cells = sorted(
        cell_id for cell_id, cell in cells.items() if cell["split_owner"] == "train"
    )
    validation_cells = sorted(
        cell_id
        for cell_id, cell in cells.items()
        if cell["split_owner"] == "validation"
    )
    tranches = tuple(int(value) for value in probe["nested_train_episodes_per_cell"])
    validation_count = int(probe["validation_rollouts_per_cell"])
    for cell_id in train_cells:
        if len(by_split_cell[("train", cell_id)]) < max(tranches):
            raise Level4ScalingProbeError(
                f"training cell {cell_id!r} has fewer than {max(tranches)} episodes."
            )
    validation_episodes: list[PilotEpisode] = []
    for cell_id in validation_cells:
        episodes = by_split_cell[("validation", cell_id)]
        if len(episodes) < validation_count:
            raise Level4ScalingProbeError(
                f"validation cell {cell_id!r} has fewer than {validation_count} episodes."
            )
        validation_episodes.extend(episodes[:validation_count])
    validation_ids = [episode.episode_id for episode in validation_episodes]

    results = []
    for tranche in tranches:
        training_episodes = [
            episode
            for cell_id in train_cells
            for episode in by_split_cell[("train", cell_id)][:tranche]
        ]
        model, mean, scale, train_loss = _train_residual_model(
            training_episodes,
            cells=cells,
            model_config=model_config,
        )
        rollout = _evaluate_validation_rollouts(
            model,
            mean=mean,
            scale=scale,
            validation_episodes=validation_episodes,
            cells=cells,
            config_path=config_path,
            workcell_config=workcell_config,
            residual_bound_m=float(model_config["residual_bound_m"]),
        )
        results.append(
            {
                "episodes_per_training_cell": tranche,
                "training_episode_count": len(training_episodes),
                "training_cell_count": len(train_cells),
                "model_seed": int(model_config["seed"]),
                "training_loss": train_loss,
                "normalization_digest": _array_digest(mean, scale),
                "validation_episode_ids": validation_ids,
                **rollout,
            }
        )

    largest = results[-1]
    improvement = float(
        results[-1]["aggregate_validation_success"]
        - results[-2]["aggregate_validation_success"]
    )
    dataset_sufficient = bool(
        largest["aggregate_validation_success"]
        >= float(gates["minimum_aggregate_validation_success"])
        and largest["worst_cell_validation_success"]
        >= float(gates["minimum_worst_cell_validation_success"])
        and largest["safety_violation_count"]
        <= int(gates["maximum_safety_violations"])
        and largest["invalid_action_count"]
        <= int(gates["maximum_invalid_actions"])
        and improvement < float(gates["material_improvement_fraction"])
    )
    return {
        "version": SCALING_REPORT_VERSION,
        "status": "sufficient" if dataset_sufficient else "larger_tranche_required",
        "config_version": config["version"],
        "config_digest": _file_digest(Path(config_path)),
        "dataset_dir": str(Path(dataset_dir)),
        "decision_frozen_at": datetime.now(UTC).isoformat(),
        "decision_uses_test_data": False,
        "test_episode_count_inspected": 0,
        "model_recipe": dict(model_config),
        "readiness_gates": dict(gates),
        "train_cell_ids": train_cells,
        "validation_cell_ids": validation_cells,
        "validation_episode_ids": validation_ids,
        "tranches": results,
        "improvement_8_to_16": improvement,
        "dataset_sufficient": dataset_sufficient,
    }


def save_scaling_probe_report(
    report: Mapping[str, Any], output_path: str | Path
) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise Level4ScalingProbeError("scaling probe report must use .json.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _train_residual_model(
    episodes: Sequence[PilotEpisode],
    *,
    cells: Mapping[str, Mapping[str, Any]],
    model_config: Mapping[str, Any],
) -> tuple[_ResidualMLP, np.ndarray, np.ndarray, float]:
    inputs = np.asarray(
        [_episode_feature(episode, cells[episode.goal_condition_id]) for episode in episodes],
        dtype=np.float32,
    )
    targets = np.asarray(
        [_controller_offset(episode) for episode in episodes], dtype=np.float32
    )
    mean = inputs.mean(axis=0)
    scale = inputs.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (inputs - mean) / scale
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = _ResidualMLP(normalized.shape[1], model_config["hidden_sizes"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_config["learning_rate"])
    )
    loss_function = nn.MSELoss()
    tensor_x = torch.from_numpy(normalized)
    tensor_y = torch.from_numpy(targets)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(model_config["batch_size"])
    final_loss = 0.0
    for _epoch in range(int(model_config["epochs"])):
        order = torch.randperm(tensor_x.shape[0], generator=generator)
        for start in range(0, tensor_x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            prediction = model(tensor_x[indices])
            loss = loss_function(prediction, tensor_y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    return model, mean, scale, final_loss


def _evaluate_validation_rollouts(
    model: _ResidualMLP,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    validation_episodes: Sequence[PilotEpisode],
    cells: Mapping[str, Mapping[str, Any]],
    config_path: str | Path,
    workcell_config: str | Path,
    residual_bound_m: float,
) -> Mapping[str, Any]:
    successes: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    safety_violations = 0
    invalid_actions = 0
    model.eval()
    for episode in validation_episodes:
        feature = _episode_feature(episode, cells[episode.goal_condition_id])
        normalized = ((feature - mean) / scale).astype(np.float32)
        with torch.no_grad():
            predicted = model(torch.from_numpy(normalized).unsqueeze(0)).numpy()[0]
        predicted = np.clip(predicted, -residual_bound_m, residual_bound_m)
        correction = np.clip(
            predicted - _controller_offset(episode),
            -residual_bound_m,
            residual_bound_m,
        )
        success, violations, invalid = _rollout_with_residual(
            episode,
            correction=correction,
            config_path=config_path,
            workcell_config=workcell_config,
        )
        totals[episode.goal_condition_id] += 1
        successes[episode.goal_condition_id] += int(success)
        safety_violations += violations
        invalid_actions += invalid
    rates = {
        cell_id: successes[cell_id] / totals[cell_id] for cell_id in sorted(totals)
    }
    total = sum(totals.values())
    return {
        "validation_rollout_count": total,
        "validation_success_count": sum(successes.values()),
        "aggregate_validation_success": (
            sum(successes.values()) / total if total else 0.0
        ),
        "worst_cell_validation_success": min(rates.values()) if rates else 0.0,
        "validation_success_by_cell": rates,
        "safety_violation_count": safety_violations,
        "invalid_action_count": invalid_actions,
    }


def _rollout_with_residual(
    episode: PilotEpisode,
    *,
    correction: np.ndarray,
    config_path: str | Path,
    workcell_config: str | Path,
) -> tuple[bool, int, int]:
    loaded = load_replay_demo(episode.path)
    actions = np.asarray(loaded.episode.actions, dtype=np.float64).copy()
    if not np.all(np.isfinite(correction)):
        return False, 0, 1
    actions[:, :3] += correction
    config, _ = load_level4_collection_config(config_path)
    workspace = _mapping(_mapping(config, "workcell"), "safe_workspace")
    low = np.asarray(workspace["min"], dtype=np.float64)
    high = np.asarray(workspace["max"], dtype=np.float64)
    violations = int(np.count_nonzero(np.any((actions[:, :3] < low) | (actions[:, :3] > high), axis=1)))
    if violations:
        return False, violations, 0
    adjusted = replace(loaded.episode, actions=actions)
    replay = replace(loaded, episode=adjusted)
    procedural = episode.metadata.get("procedural_expansion")
    variation = procedural.get("variation") if isinstance(procedural, Mapping) else None
    metric_successes: list[bool] = []
    pick_successes: list[bool] = []
    with WorkcellPilotTask(
        workcell_config=workcell_config,
        dataset_config=config_path,
        skill_name=episode.skill_name,
        goal_condition_id=episode.goal_condition_id,
        seed=int(episode.metadata["random_seed"]),
        procedural_variation=variation if isinstance(variation, Mapping) else None,
    ) as task:
        goal = task.goal
        if episode.skill_name == "pick_place_sequence":
            metric = task.workcell.create_task(
                "place_held_object",
                object_id=str(goal["object_id"]),
                target_id=str(goal["target_id"]),
            )
            pick = task.workcell.create_task(
                "pick_object", object_id=str(goal["object_id"])
            )
        else:
            metric = task.workcell.create_task(episode.skill_name, **goal)
            pick = None

        def observe(_step: object, _state: object) -> None:
            world = task.workcell.get_world_state()
            metric_successes.append(metric.evaluate(world).success)
            if pick is not None:
                pick_successes.append(pick.evaluate(world).success)

        recording = episode.metadata.get("recording")
        cadence = recording.get("sim_steps_per_frame") if isinstance(recording, Mapping) else None
        if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence <= 0:
            return False, 0, 1
        replay_loaded_demo(
            replay,
            task.env,
            speed=1000.0,
            sim_steps_per_action=cadence,
            reset_env=False,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
    if episode.skill_name in {"push_object_to_target", "pick_place_sequence"}:
        success = bool(metric_successes and metric_successes[-1])
    else:
        success = any(metric_successes)
    if episode.skill_name == "pick_place_sequence":
        success = success and any(pick_successes)
    return success, 0, 0


def _episode_feature(
    episode: PilotEpisode, cell: Mapping[str, Any]
) -> np.ndarray:
    reset = _mapping(episode.metadata, "reset_state")
    positions = _mapping(reset, "entity_positions_m")
    skill_index = {
        "reach_object": 0,
        "pick_place_sequence": 1,
        "push_object_to_target": 2,
        "press_button": 3,
    }[episode.skill_name]
    skill = np.zeros(4, dtype=np.float64)
    skill[skill_index] = 1.0
    source_id = str(
        cell.get("object_id") or cell.get("entity_id") or cell.get("button_id")
    )
    target_id = str(cell.get("target_id") or cell.get("entity_id") or cell.get("button_id"))
    source = np.asarray(positions[source_id], dtype=np.float64)
    target = np.asarray(positions[target_id], dtype=np.float64)
    procedural = episode.metadata.get("procedural_expansion")
    variation = procedural.get("variation") if isinstance(procedural, Mapping) else {}
    physical = np.asarray(
        [
            variation.get("object_scale_multiplier", 1.0),
            variation.get("object_mass_multiplier", 1.0),
            variation.get("object_friction_multiplier", 1.0),
        ],
        dtype=np.float64,
    )
    target_depth = float(cell.get("target_depth_m", 0.0))
    boundary = float(
        cell.get("approach_class") == "near_boundary"
        or "boundary" in str(cell.get("id", ""))
    )
    return np.concatenate((skill, source, target, physical, [target_depth, boundary]))


def _controller_offset(episode: PilotEpisode) -> np.ndarray:
    procedural = episode.metadata.get("procedural_expansion")
    variation = procedural.get("variation") if isinstance(procedural, Mapping) else {}
    return np.asarray(
        variation.get("controller_position_offset_m", (0.0, 0.0, 0.0)),
        dtype=np.float32,
    )


def _episode_order_key(episode: PilotEpisode) -> tuple[int, int, str]:
    procedural = episode.metadata.get("procedural_expansion")
    repetition = procedural.get("repetition") if isinstance(procedural, Mapping) else None
    if isinstance(repetition, int) and not isinstance(repetition, bool):
        return (1, repetition, episode.episode_id)
    return (0, int(episode.metadata.get("random_seed", 0)), episode.episode_id)


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
    return "sha256:" + digest.hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise Level4ScalingProbeError(f"{key} must be a mapping.")
    return value
