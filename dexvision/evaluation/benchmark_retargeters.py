"""Offline, repeatable metrics for comparing DexVision retargeters."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Final, TypeAlias

import numpy as np

from dexvision.features.hand_features import FingerState, HandFeatures
from dexvision.logging.replay_demo import LoadedReplayDemo, load_replay_demo, replay_loaded_demo
from dexvision.retargeting.curl_retargeter import CurlRetargeter, CurlRetargeterConfig
from dexvision.retargeting.fingertip_ik_retargeter import (
    FINGERTIP_NAMES,
    FingertipIKRetargeter,
)
from dexvision.retargeting.optimization_retargeter import OptimizationRetargeter


BENCHMARK_VERSION: Final[str] = "level2.10b/retargeting-benchmark-v2"
RETARGETER_NAMES: Final[tuple[str, ...]] = ("curl", "fingertip", "optimization")
_FINGER_CHAINS: Final[dict[str, tuple[int, int, int, int]]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_EPSILON: Final[float] = 1e-8


class RetargetingBenchmarkError(RuntimeError):
    """Raised when benchmark inputs or outputs are invalid."""


@dataclass(frozen=True)
class ConfidenceInterval:
    """Two-sided episode-bootstrap interval for an aggregate mean."""

    lower: float
    upper: float


@dataclass(frozen=True)
class TaskReplayMetrics:
    """Task and contact measurements from one counterfactual replay."""

    success: bool
    mean_fingertip_object_distance_m: float
    fingertip_contact_frame_rate: float


@dataclass(frozen=True)
class RetargeterMetrics:
    """Aggregate metrics for one retargeter over the same episode set."""

    retargeter: str
    episodes: int
    frames: int
    mean_latency_ms: float
    mean_action_jerk: float
    joint_limit_violation_rate: float
    mean_fingertip_error: float | None
    task_success_rate: float
    mean_fingertip_object_distance_m: float | None = None
    fingertip_contact_frame_rate: float | None = None
    confidence_intervals: Mapping[str, ConfidenceInterval] | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable Level 2.10 benchmark report."""

    benchmark_version: str
    generated_at_utc: str
    task_id: str
    episode_ids: tuple[str, ...]
    config_path: str
    success_evaluation: str
    bootstrap_samples: int
    bootstrap_seed: int
    action_jerk_units: str
    fingertip_error_units: str
    fingertip_object_distance_units: str
    metrics: tuple[RetargeterMetrics, ...]


SuccessEvaluator: TypeAlias = Callable[
    [LoadedReplayDemo, np.ndarray], bool | TaskReplayMetrics
]


def mean_action_jerk(actions: np.ndarray) -> float:
    """Return mean L2 third difference in normalized actuator units/frame^3."""

    values = _finite_matrix(actions, name="actions")
    if values.shape[0] < 4:
        return 0.0
    jerk = np.diff(values, n=3, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def joint_limit_violation_rate(
    actions: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> float:
    """Return the fraction of scalar action values outside inclusive limits."""

    values = _finite_matrix(actions, name="actions")
    lower = _finite_vector(lower_limits, name="lower_limits", size=values.shape[1])
    upper = _finite_vector(upper_limits, name="upper_limits", size=values.shape[1])
    if np.any(lower > upper):
        raise ValueError("lower_limits must not exceed upper_limits.")
    violations = np.logical_or(values < lower, values > upper)
    return float(np.count_nonzero(violations) / violations.size)


def mean_fingertip_error(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean Euclidean fingertip error for arrays shaped ``[..., 5, 3]``."""

    predicted_values = np.asarray(predicted, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if predicted_values.shape != target_values.shape:
        raise ValueError(
            "predicted and target fingertip arrays must have matching shapes."
        )
    if predicted_values.ndim < 2 or predicted_values.shape[-2:] != (5, 3):
        raise ValueError("fingertip arrays must end with shape [5, 3].")
    if not (
        np.all(np.isfinite(predicted_values))
        and np.all(np.isfinite(target_values))
    ):
        raise ValueError("fingertip arrays must contain only finite values.")
    return float(np.mean(np.linalg.norm(predicted_values - target_values, axis=-1)))


def bootstrap_confidence_interval(
    values: Sequence[float] | np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a deterministic percentile 95% CI by resampling episodes."""

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("bootstrap values must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(observations)):
        raise ValueError("bootstrap values must contain only finite values.")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer.")
    if observations.size == 1:
        value = float(observations[0])
        return ConfidenceInterval(value, value)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, observations.size, size=(samples, observations.size), endpoint=False
    )
    means = np.mean(observations[indices], axis=1)
    lower, upper = np.percentile(means, (2.5, 97.5))
    return ConfidenceInterval(float(lower), float(upper))


def discover_task_episodes(
    dataset_root: str | Path,
    *,
    task_id: str,
    episodes: int,
) -> tuple[Path, ...]:
    """Select a deterministic set of saved episode directories for one task."""

    if episodes <= 0:
        raise RetargetingBenchmarkError("episodes must be positive.")
    task_dir = Path(dataset_root) / task_id
    if not task_dir.is_dir():
        raise RetargetingBenchmarkError(f"task dataset directory does not exist: {task_dir}")
    candidates = tuple(
        path
        for path in sorted(task_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    if len(candidates) < episodes:
        raise RetargetingBenchmarkError(
            f"task '{task_id}' has {len(candidates)} saved episodes; "
            f"requested {episodes}."
        )
    return candidates[:episodes]


def run_benchmark(
    episode_dirs: Sequence[str | Path],
    *,
    task_id: str,
    config_path: str | Path,
    retargeter_names: Sequence[str] = RETARGETER_NAMES,
    success_evaluator: SuccessEvaluator | None = None,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> BenchmarkReport:
    """Benchmark retargeters on identical saved landmark and feature streams."""

    if not episode_dirs:
        raise RetargetingBenchmarkError("at least one episode is required.")
    if bootstrap_samples <= 0:
        raise RetargetingBenchmarkError("bootstrap_samples must be positive.")
    selected_names = tuple(retargeter_names)
    unknown = sorted(set(selected_names) - set(RETARGETER_NAMES))
    if unknown:
        raise RetargetingBenchmarkError(
            "unknown retargeters: " + ", ".join(unknown)
        )
    if len(set(selected_names)) != len(selected_names):
        raise RetargetingBenchmarkError("retargeter names must be unique.")
    if not {"curl", "fingertip"}.issubset(selected_names):
        raise RetargetingBenchmarkError(
            "Level 2.10 requires at least curl and fingertip retargeters."
        )

    loaded_episodes = tuple(load_replay_demo(path) for path in episode_dirs)
    for loaded in loaded_episodes:
        observed_task = loaded.episode.metadata.get("task_id")
        if observed_task != task_id:
            raise RetargetingBenchmarkError(
                f"episode {loaded.demo_dir} has task_id {observed_task!r}, "
                f"expected {task_id!r}."
            )
        if loaded.episode.landmarks is None:
            raise RetargetingBenchmarkError(
                f"episode {loaded.demo_dir} has no landmarks.npy; fingertip "
                "retargeters cannot be compared."
            )

    metrics = tuple(
        _benchmark_retargeter(
            name,
            loaded_episodes,
            config_path=Path(config_path),
            success_evaluator=success_evaluator,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for name in selected_names
    )
    success_description = (
        "counterfactual headless MuJoCo replay"
        if success_evaluator is not None
        else "recorded episode success labels"
    )
    return BenchmarkReport(
        benchmark_version=BENCHMARK_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        task_id=task_id,
        episode_ids=tuple(
            str(loaded.episode.metadata["episode_id"]) for loaded in loaded_episodes
        ),
        config_path=str(config_path),
        success_evaluation=success_description,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        action_jerk_units="normalized actuator units/frame^3",
        fingertip_error_units="palm-width-normalized Euclidean distance",
        fingertip_object_distance_units="metres, signed MuJoCo geom distance",
        metrics=metrics,
    )


def save_benchmark_report(
    report: BenchmarkReport,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Save the same benchmark metrics as JSON and tabular CSV."""

    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metric_names = (
        "mean_latency_ms",
        "mean_action_jerk",
        "joint_limit_violation_rate",
        "mean_fingertip_error",
        "task_success_rate",
        "mean_fingertip_object_distance_m",
        "fingertip_contact_frame_rate",
    )
    fieldnames = (
        "retargeter",
        "episodes",
        "frames",
        *metric_names,
        *(f"{name}_ci95_{bound}" for name in metric_names for bound in ("low", "high")),
    )
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in report.metrics:
            row = {
                key: value
                for key, value in asdict(metric).items()
                if key != "confidence_intervals"
            }
            intervals = metric.confidence_intervals or {}
            for metric_name in metric_names:
                interval = intervals.get(metric_name)
                row[f"{metric_name}_ci95_low"] = (
                    None if interval is None else interval.lower
                )
                row[f"{metric_name}_ci95_high"] = (
                    None if interval is None else interval.upper
                )
            writer.writerow(row)
    return json_output, csv_output


def save_benchmark_plot(report: BenchmarkReport, output_path: str | Path) -> Path:
    """Write a dependency-free SVG containing the four required metric plots."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [metric.retargeter for metric in report.metrics]
    panels = (
        ("Mean latency (ms)", [metric.mean_latency_ms for metric in report.metrics]),
        ("Mean action jerk", [metric.mean_action_jerk for metric in report.metrics]),
        (
            "Joint-limit violation rate",
            [metric.joint_limit_violation_rate for metric in report.metrics],
        ),
        ("Task success rate", [metric.task_success_rate for metric in report.metrics]),
        (
            "Fingertip-object distance (m)",
            [metric.mean_fingertip_object_distance_m or 0.0 for metric in report.metrics],
        ),
        (
            "Fingertip contact-frame rate",
            [metric.fingertip_contact_frame_rate or 0.0 for metric in report.metrics],
        ),
    )
    width, height = 960, 890
    colors = ("#2563eb", "#0f766e", "#b45309", "#7c3aed")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.panel{font-size:15px;font-weight:700}.label{font-size:12px}.value{font-size:11px}</style>',
        f'<text x="40" y="35" class="title">Retargeting benchmark — {_xml(report.task_id)}</text>',
    ]
    for panel_index, (title, values) in enumerate(panels):
        panel_x = 40 + (panel_index % 2) * 460
        panel_y = 70 + (panel_index // 2) * 270
        plot_width, plot_height = 400, 185
        maximum = max(max(values, default=0.0), 1e-12)
        svg.extend(
            (
                f'<text x="{panel_x}" y="{panel_y}" class="panel">{_xml(title)}</text>',
                f'<line x1="{panel_x}" y1="{panel_y + plot_height}" x2="{panel_x + plot_width}" y2="{panel_y + plot_height}" stroke="#9ca3af"/>',
            )
        )
        slot = plot_width / max(1, len(values))
        bar_width = min(72.0, slot * 0.55)
        for index, (name, value) in enumerate(zip(names, values, strict=True)):
            bar_height = 0.0 if value <= 0.0 else 145.0 * value / maximum
            x = panel_x + slot * index + (slot - bar_width) / 2
            y = panel_y + plot_height - bar_height
            svg.extend(
                (
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[index % len(colors)]}"/>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{panel_y + plot_height + 18}" text-anchor="middle" class="label">{_xml(name)}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{max(panel_y + 28, y - 5):.1f}" text-anchor="middle" class="value">{value:.4g}</text>',
                )
            )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return path


def replay_push_cube_metrics(
    loaded: LoadedReplayDemo, actions: np.ndarray
) -> TaskReplayMetrics:
    """Measure push-cube success and fingertip contact in headless MuJoCo."""

    if loaded.episode.metadata.get("task_id") != "push_cube_to_target":
        raise RetargetingBenchmarkError(
            "counterfactual replay success is currently defined for "
            "push_cube_to_target only."
        )
    from dexvision.sim.mujoco_env import MujocoEnv
    from dexvision.sim.tasks import push_cube_distance

    task_config = loaded.episode.metadata.get("task_config")
    if not isinstance(task_config, Mapping):
        raise RetargetingBenchmarkError("episode task_config must be a mapping.")
    object_id = str(task_config.get("resolved_object_id") or "")
    target_position = _finite_vector(
        task_config.get("target_position"), name="target_position", size=3
    )
    target_radius = float(task_config.get("target_radius", math.nan))
    required_dwell = int(task_config.get("success_dwell_steps", 0))
    if not object_id or not math.isfinite(target_radius) or target_radius <= 0.0:
        raise RetargetingBenchmarkError("push-cube success metadata is incomplete.")
    if required_dwell <= 0:
        raise RetargetingBenchmarkError("success_dwell_steps must be positive.")

    replay_episode = replace(loaded.episode, actions=np.asarray(actions, dtype=np.float64))
    replay_demo = replace(loaded, episode=replay_episode)
    dwell_steps = 0
    succeeded = False
    fingertip_distances: list[float] = []
    contact_frames = 0

    with MujocoEnv(loaded.model_path) as env:
        joint_id = env._mujoco.mj_name2id(
            env.model, env._mujoco.mjtObj.mjOBJ_JOINT, f"{object_id}_joint"
        )
        if joint_id < 0:
            raise RetargetingBenchmarkError(
                f"replay model is missing push-cube joint '{object_id}_joint'."
            )
        qpos_address = int(env.model.jnt_qposadr[joint_id])
        object_geom_id = env._mujoco.mj_name2id(
            env.model, env._mujoco.mjtObj.mjOBJ_GEOM, "push_cube_geom"
        )
        if object_geom_id < 0:
            raise RetargetingBenchmarkError(
                "replay model is missing named geom 'push_cube_geom'."
            )
        fingertip_geom_ids = _fingertip_collision_geom_ids(env)

        def observe(_step: object, _state: object) -> None:
            nonlocal dwell_steps, succeeded, contact_frames
            object_position = np.asarray(
                env.data.qpos[qpos_address : qpos_address + 3], dtype=np.float64
            )
            if push_cube_distance(object_position, target_position) <= target_radius:
                dwell_steps += 1
            else:
                dwell_steps = 0
            succeeded = succeeded or dwell_steps >= required_dwell
            distances = [
                float(
                    env._mujoco.mj_geomDistance(
                        env.model,
                        env.data,
                        geom_id,
                        object_geom_id,
                        1.0,
                        None,
                    )
                )
                for geom_id in fingertip_geom_ids
            ]
            minimum_distance = min(distances)
            fingertip_distances.append(minimum_distance)
            if minimum_distance <= 0.0:
                contact_frames += 1

        sim_steps = int(
            loaded.episode.metadata.get("recording", {}).get("sim_steps_per_frame", 1)
        )
        replay_loaded_demo(
            replay_demo,
            env,
            speed=1.0,
            sim_steps_per_action=sim_steps,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
    if not fingertip_distances:
        raise RetargetingBenchmarkError("push-cube replay produced no measured frames.")
    return TaskReplayMetrics(
        success=succeeded,
        mean_fingertip_object_distance_m=float(np.mean(fingertip_distances)),
        fingertip_contact_frame_rate=contact_frames / len(fingertip_distances),
    )


def replay_push_cube_success(loaded: LoadedReplayDemo, actions: np.ndarray) -> bool:
    """Compatibility wrapper returning only counterfactual task success."""

    return replay_push_cube_metrics(loaded, actions).success


def _benchmark_retargeter(
    name: str,
    episodes: Sequence[LoadedReplayDemo],
    *,
    config_path: Path,
    success_evaluator: SuccessEvaluator | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> RetargeterMetrics:
    episode_latencies_ms: list[float] = []
    episode_jerks: list[float] = []
    episode_limit_rates: list[float] = []
    episode_fingertip_errors: list[float] = []
    successes: list[bool] = []
    object_distances: list[float] = []
    contact_rates: list[float] = []
    frame_count = 0

    for loaded in episodes:
        retargeter = _make_retargeter(name, config_path)
        limits = _target_limits(retargeter.config, loaded.finger_target_names)
        feature_fields = _feature_field_indices(loaded.episode.metadata)
        landmarks = np.asarray(loaded.episode.landmarks, dtype=np.float64)
        finger_rows: list[np.ndarray] = []
        frame_latencies: list[float] = []
        frame_fingertip_errors: list[float] = []
        for index in range(landmarks.shape[0]):
            input_value: object
            if name == "curl":
                input_value = _hand_features(loaded.episode.features[index], feature_fields)
            else:
                input_value = landmarks[index]
            started = perf_counter()
            targets = retargeter.map(input_value)
            frame_latencies.append(perf_counter() - started)
            row = np.asarray(
                [targets[target_name] for target_name in loaded.finger_target_names],
                dtype=np.float64,
            )
            lower, upper = limits
            finger_rows.append(row)
            try:
                target_tips, predicted_tips = _fingertip_prediction(
                    landmarks[index], targets, retargeter.config
                )
            except ValueError:
                pass
            else:
                frame_fingertip_errors.append(
                    mean_fingertip_error(predicted_tips, target_tips)
                )

        finger_actions = np.stack(finger_rows, axis=0)
        normalized_actions = _normalize_actions(finger_actions, *limits)
        episode_latencies_ms.append(1000.0 * float(np.mean(frame_latencies)))
        episode_jerks.append(mean_action_jerk(normalized_actions))
        episode_limit_rates.append(
            joint_limit_violation_rate(finger_actions, *limits)
        )
        if frame_fingertip_errors:
            episode_fingertip_errors.append(float(np.mean(frame_fingertip_errors)))
        full_actions = np.asarray(loaded.episode.actions, dtype=np.float64).copy()
        finger_range = loaded.action_schema.finger_actuator_targets
        finger_slice = finger_range if isinstance(finger_range, slice) else slice(*finger_range)
        full_actions[:, finger_slice] = finger_actions
        replay_metrics = (
            success_evaluator(loaded, full_actions)
            if success_evaluator is not None
            else bool(loaded.episode.success)
        )
        if isinstance(replay_metrics, TaskReplayMetrics):
            successes.append(replay_metrics.success)
            object_distances.append(replay_metrics.mean_fingertip_object_distance_m)
            contact_rates.append(replay_metrics.fingertip_contact_frame_rate)
        else:
            successes.append(bool(replay_metrics))
        frame_count += finger_actions.shape[0]

    metric_samples: dict[str, Sequence[float]] = {
        "mean_latency_ms": episode_latencies_ms,
        "mean_action_jerk": episode_jerks,
        "joint_limit_violation_rate": episode_limit_rates,
        "task_success_rate": [float(value) for value in successes],
    }
    if episode_fingertip_errors:
        metric_samples["mean_fingertip_error"] = episode_fingertip_errors
    if object_distances:
        metric_samples["mean_fingertip_object_distance_m"] = object_distances
    if contact_rates:
        metric_samples["fingertip_contact_frame_rate"] = contact_rates
    confidence_intervals = {
        metric_name: bootstrap_confidence_interval(
            values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        for index, (metric_name, values) in enumerate(metric_samples.items())
    }
    return RetargeterMetrics(
        retargeter=name,
        episodes=len(episodes),
        frames=frame_count,
        mean_latency_ms=float(np.mean(episode_latencies_ms)),
        mean_action_jerk=float(np.mean(episode_jerks)),
        joint_limit_violation_rate=float(np.mean(episode_limit_rates)),
        mean_fingertip_error=(
            None
            if not episode_fingertip_errors
            else float(np.mean(episode_fingertip_errors))
        ),
        task_success_rate=float(np.mean(successes)),
        mean_fingertip_object_distance_m=(
            None if not object_distances else float(np.mean(object_distances))
        ),
        fingertip_contact_frame_rate=(
            None if not contact_rates else float(np.mean(contact_rates))
        ),
        confidence_intervals=confidence_intervals,
    )


def _make_retargeter(
    name: str, config_path: Path
) -> CurlRetargeter | FingertipIKRetargeter | OptimizationRetargeter:
    if name == "curl":
        return CurlRetargeter.from_yaml(config_path)
    if name == "fingertip":
        return FingertipIKRetargeter.from_yaml(config_path)
    if name == "optimization":
        return OptimizationRetargeter.from_yaml(config_path)
    raise RetargetingBenchmarkError(f"unknown retargeter: {name}")


def _target_limits(
    config: CurlRetargeterConfig, target_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    limits = {target.name: target.limit for target in config.static_targets}
    for finger in config.fingers:
        limits.update({target.name: target.limit for target in finger.targets})
    missing = [name for name in target_names if name not in limits]
    if missing:
        raise RetargetingBenchmarkError(
            "retargeter config is missing saved targets: " + ", ".join(missing)
        )
    return (
        np.asarray([limits[name].minimum for name in target_names], dtype=np.float64),
        np.asarray([limits[name].maximum for name in target_names], dtype=np.float64),
    )


def _feature_field_indices(metadata: Mapping[str, object]) -> dict[str, int]:
    raw_fields = metadata.get("feature_fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, str):
        raise RetargetingBenchmarkError("metadata.feature_fields must be a sequence.")
    fields = {str(name): index for index, name in enumerate(raw_fields)}
    required = {
        "thumb_curl",
        "index_curl",
        "middle_curl",
        "ring_curl",
        "pinky_curl",
        "index_bend",
        "middle_bend",
        "ring_bend",
        "pinky_bend",
        "pinch_thumb_index",
        "confidence",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RetargetingBenchmarkError(
            "saved feature layout is missing: " + ", ".join(missing)
        )
    return fields


def _hand_features(values: np.ndarray, fields: Mapping[str, int]) -> HandFeatures:
    row = np.asarray(values, dtype=np.float64)

    def finger(name: str, *, bend_name: str | None = None) -> FingerState:
        curl = float(row[fields[f"{name}_curl"]])
        bend = curl if bend_name is None else float(row[fields[bend_name]])
        extension = float(np.clip(1.0 - bend, 0.0, 1.0))
        return FingerState(curl, extension, None, extension >= 0.55, True)

    return HandFeatures(
        thumb=finger("thumb"),
        index=finger("index", bend_name="index_bend"),
        middle=finger("middle", bend_name="middle_bend"),
        ring=finger("ring", bend_name="ring_bend"),
        pinky=finger("pinky", bend_name="pinky_bend"),
        pinch_thumb_index=float(row[fields["pinch_thumb_index"]]),
        confidence=float(row[fields["confidence"]]),
    )


def _fingertip_prediction(
    landmarks: np.ndarray,
    targets: Mapping[str, float],
    config: CurlRetargeterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise ValueError("invalid landmarks")
    origin = points[0]
    across = points[5] - points[17]
    toward = points[9] - origin
    width = float(np.linalg.norm(across))
    if width <= _EPSILON:
        raise ValueError("zero palm width")
    x_axis = _unit(across)
    z_axis = _unit(np.cross(x_axis, toward))
    y_axis = _unit(np.cross(z_axis, x_axis))
    local = ((points - origin) @ np.stack((x_axis, y_axis, z_axis), axis=1)) / width
    target_tips = np.empty((5, 3), dtype=np.float64)
    predicted = np.empty_like(target_tips)
    controls = _controls_from_targets(targets, config)
    for index, finger_name in enumerate(FINGERTIP_NAMES):
        chain = _FINGER_CHAINS[finger_name]
        base = local[chain[0]]
        tip = local[chain[-1]]
        length = sum(
            float(np.linalg.norm(points[end] - points[start]))
            for start, end in zip(chain[:-1], chain[1:], strict=True)
        ) / width
        if length <= _EPSILON:
            raise ValueError("zero finger length")
        if finger_name == "thumb":
            direction = tip - base
            direction_norm = float(np.linalg.norm(direction))
            direction = (
                np.asarray([1.0, 0.0, 0.0])
                if direction_norm <= _EPSILON
                else direction / direction_norm
            )
        else:
            direction = np.asarray([0.0, 1.0, 0.0])
        target_tips[index] = tip
        predicted[index] = base + direction * length * (1.0 - controls[finger_name])
    return target_tips, predicted


def _controls_from_targets(
    targets: Mapping[str, float], config: CurlRetargeterConfig
) -> dict[str, float]:
    controls: dict[str, float] = {name: 0.0 for name in FINGERTIP_NAMES}
    for finger in config.fingers:
        estimates: list[float] = []
        for target in finger.targets:
            span = target.closed_value - target.open_value
            if abs(span) > _EPSILON:
                estimates.append((float(targets[target.name]) - target.open_value) / span)
        if estimates:
            controls[finger.name] = float(np.clip(np.mean(estimates), 0.0, 1.0))
    return controls


def _normalize_actions(
    actions: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    span = upper - lower
    safe_span = np.where(span > _EPSILON, span, 1.0)
    return (actions - lower) / safe_span


def _fingertip_collision_geom_ids(env: object) -> tuple[int, ...]:
    """Resolve one collidable distal geom for each Shadow Hand fingertip."""

    model = getattr(env, "model")
    mujoco_module = getattr(env, "_mujoco")
    geom_ids: list[int] = []
    for body_name in (
        "rh_thdistal",
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_lfdistal",
    ):
        body_id = mujoco_module.mj_name2id(
            model, mujoco_module.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise RetargetingBenchmarkError(
                f"replay model is missing fingertip body '{body_name}'."
            )
        candidates = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and int(model.geom_contype[geom_id]) != 0
        ]
        if len(candidates) != 1:
            raise RetargetingBenchmarkError(
                f"fingertip body '{body_name}' must have exactly one collidable geom; "
                f"found {len(candidates)}."
            )
        geom_ids.append(candidates[0])
    return tuple(geom_ids)


def _finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_vector(value: object, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values.")
    return array


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def _xml(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
