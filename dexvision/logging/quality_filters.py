"""Read-only quality filtering for saved Level 2 pilot demonstrations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.relabel_success import (
    BUTTON_PRESS_TASK_ID,
    REACH_TOUCH_TARGET_TASK_ID,
    SuccessRelabelError,
    relabel_button_press_episode,
    relabel_reach_touch_episode,
)


QUALITY_REPORT_VERSION = "level2/pilot-quality-report-v1"
QUALITY_THRESHOLDS_VERSION = "level2/pilot-quality-thresholds-v1"
DEFAULT_REPORT_NAME = "quality_report.json"


class QualityFilterError(RuntimeError):
    """Raised when a saved episode cannot be evaluated safely."""


@dataclass(frozen=True)
class QualityThresholds:
    """Versioned thresholds for the Level 2.7 pilot quality filters."""

    version: str = QUALITY_THRESHOLDS_VERSION
    min_mean_tracking_confidence: float = 0.75
    min_mean_feature_confidence: float = 0.75
    max_missing_frame_fraction: float = 0.10
    max_feature_jitter_p95: float = 0.20
    max_action_jerk_p95: float = 0.20
    max_joint_limit_hit_fraction: float = 0.20
    max_workspace_limit_hit_fraction: float = 0.10
    limit_margin_fraction: float = 0.01

    def validate(self) -> None:
        """Validate all thresholds before evaluating a dataset."""

        if not self.version:
            raise QualityFilterError("quality threshold version is required.")
        for field_name in (
            "min_mean_tracking_confidence",
            "min_mean_feature_confidence",
            "max_missing_frame_fraction",
            "max_joint_limit_hit_fraction",
            "max_workspace_limit_hit_fraction",
            "limit_margin_fraction",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise QualityFilterError(f"{field_name} must be in [0, 1], got {value}.")
        for field_name in ("max_feature_jitter_p95", "max_action_jerk_p95"):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise QualityFilterError(f"{field_name} must be non-negative, got {value}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable threshold snapshot."""

        return asdict(self)


@dataclass(frozen=True)
class EpisodeQualityResult:
    """Filter metrics and pass/fail reasons for one immutable raw episode."""

    episode_directory: str
    episode_id: str
    skill_name: str
    task_id: str
    passed: bool
    failed_filters: tuple[str, ...]
    metrics: dict[str, float | bool]


@dataclass(frozen=True)
class QualityGroupResult:
    """Quality counts for one skill/task pair."""

    skill_name: str
    task_id: str
    episode_count: int
    pass_count: int
    fail_count: int


@dataclass(frozen=True)
class QualityReport:
    """Dataset-level Level 2.7 quality report."""

    version: str
    threshold_version: str
    thresholds: dict[str, Any]
    dataset: str
    episode_count: int
    pass_count: int
    fail_count: int
    raw_episodes_modified: bool
    groups: tuple[QualityGroupResult, ...]
    episodes: tuple[EpisodeQualityResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def evaluate_episode_quality(
    episode_dir: str | Path,
    *,
    thresholds: QualityThresholds | None = None,
) -> EpisodeQualityResult:
    """Evaluate one saved supported pilot episode without modifying it."""

    limits = thresholds or QualityThresholds()
    limits.validate()
    path = Path(episode_dir)
    metadata = _load_metadata(path)
    skill_name = _required_string(metadata, "skill_name", path=path)
    task_id = _required_string(metadata, "task_id", path=path)
    if task_id not in {REACH_TOUCH_TARGET_TASK_ID, BUTTON_PRESS_TASK_ID}:
        raise QualityFilterError(
            f"{path / 'metadata.json'} has task_id={task_id!r}; Level 2.7 quality "
            "filtering supports only "
            f"{REACH_TOUCH_TARGET_TASK_ID!r} and {BUTTON_PRESS_TASK_ID!r} pilot demos."
        )

    tracking = _load_required_array(path, "tracking_quality.npy")
    features = _load_required_array(path, "features.npy")
    actions = _load_required_array(path, "actions.npy")
    _require_matching_frames(
        path,
        tracking_quality=tracking,
        features=features,
        actions=actions,
    )

    tracking_names = _required_name_list(
        metadata,
        "tracking_quality_fields",
        width=tracking.shape[1],
        path=path,
    )
    mean_tracking_confidence = _mean_named_column(
        tracking,
        tracking_names,
        "hand_tracking_confidence",
        path=path,
    )
    mean_feature_confidence = _mean_named_column(
        tracking,
        tracking_names,
        "feature_confidence",
        path=path,
    )
    detected = _named_column(tracking, tracking_names, "detected", path=path) >= 0.5
    missing = ~detected
    if "dropped_frame" in tracking_names:
        missing |= (
            _named_column(tracking, tracking_names, "dropped_frame", path=path) >= 0.5
        )
    missing_frame_fraction = float(np.mean(missing))

    feature_jitter_p95 = _difference_norm_percentile(features, order=1)
    action_jerk_p95 = _difference_norm_percentile(actions, order=3)
    joint_limit_hit_fraction = _joint_limit_hit_fraction(
        actions,
        metadata,
        margin_fraction=limits.limit_margin_fraction,
        path=path,
    )
    workspace_limit_hit_fraction = _workspace_limit_hit_fraction(
        actions,
        metadata,
        margin_fraction=limits.limit_margin_fraction,
        path=path,
    )
    try:
        if task_id == REACH_TOUCH_TARGET_TASK_ID:
            recomputed_success = relabel_reach_touch_episode(path).recomputed_success
        else:
            recomputed_success = relabel_button_press_episode(path).recomputed_success
    except SuccessRelabelError as exc:
        raise QualityFilterError(f"Could not recompute task success for {path}: {exc}") from exc

    failed_filters: list[str] = []
    if (
        mean_tracking_confidence < limits.min_mean_tracking_confidence
        or mean_feature_confidence < limits.min_mean_feature_confidence
    ):
        failed_filters.append("low_tracking_confidence")
    if missing_frame_fraction > limits.max_missing_frame_fraction:
        failed_filters.append("too_many_missing_frames")
    if feature_jitter_p95 > limits.max_feature_jitter_p95:
        failed_filters.append("high_feature_jitter")
    if action_jerk_p95 > limits.max_action_jerk_p95:
        failed_filters.append("high_action_jerk")
    if joint_limit_hit_fraction > limits.max_joint_limit_hit_fraction:
        failed_filters.append("too_many_joint_limit_hits")
    if not recomputed_success:
        failed_filters.append("recomputed_task_failure")
    if workspace_limit_hit_fraction > limits.max_workspace_limit_hit_fraction:
        failed_filters.append("workspace_limit_hits")

    return EpisodeQualityResult(
        episode_directory=path.name,
        episode_id=str(metadata.get("episode_id", path.name)),
        skill_name=skill_name,
        task_id=task_id,
        passed=not failed_filters,
        failed_filters=tuple(failed_filters),
        metrics={
            "mean_tracking_confidence": mean_tracking_confidence,
            "mean_feature_confidence": mean_feature_confidence,
            "missing_frame_fraction": missing_frame_fraction,
            "feature_jitter_p95": feature_jitter_p95,
            "action_jerk_p95": action_jerk_p95,
            "joint_limit_hit_fraction": joint_limit_hit_fraction,
            "workspace_limit_hit_fraction": workspace_limit_hit_fraction,
            "recomputed_task_success": recomputed_success,
        },
    )


def filter_demo_dataset(
    dataset_dir: str | Path,
    *,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Evaluate every immediate episode directory and group results by skill/task."""

    dataset = Path(dataset_dir)
    limits = thresholds or QualityThresholds()
    limits.validate()
    episodes = tuple(
        evaluate_episode_quality(path, thresholds=limits)
        for path in _episode_directories(dataset)
    )
    group_keys = sorted({(episode.skill_name, episode.task_id) for episode in episodes})
    groups = tuple(
        QualityGroupResult(
            skill_name=skill_name,
            task_id=task_id,
            episode_count=sum(
                episode.skill_name == skill_name and episode.task_id == task_id
                for episode in episodes
            ),
            pass_count=sum(
                episode.skill_name == skill_name
                and episode.task_id == task_id
                and episode.passed
                for episode in episodes
            ),
            fail_count=sum(
                episode.skill_name == skill_name
                and episode.task_id == task_id
                and not episode.passed
                for episode in episodes
            ),
        )
        for skill_name, task_id in group_keys
    )
    pass_count = sum(episode.passed for episode in episodes)
    return QualityReport(
        version=QUALITY_REPORT_VERSION,
        threshold_version=limits.version,
        thresholds=limits.to_dict(),
        dataset=str(dataset),
        episode_count=len(episodes),
        pass_count=pass_count,
        fail_count=len(episodes) - pass_count,
        raw_episodes_modified=False,
        groups=groups,
        episodes=episodes,
    )


def save_quality_report(report: QualityReport, output_path: str | Path) -> Path:
    """Save a JSON report atomically without rewriting raw episode files."""

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise QualityFilterError("quality report output must use a .json extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def _episode_directories(dataset: Path) -> tuple[Path, ...]:
    if not dataset.exists():
        raise QualityFilterError(f"Dataset directory does not exist: {dataset}")
    if not dataset.is_dir():
        raise QualityFilterError(f"Dataset path is not a directory: {dataset}")
    if (dataset / "metadata.json").is_file():
        return (dataset,)
    episodes = tuple(
        path
        for path in sorted(dataset.iterdir())
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    if not episodes:
        raise QualityFilterError(
            f"No episode directories containing metadata.json were found in: {dataset}"
        )
    return episodes


def _load_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise QualityFilterError(f"Missing metadata.json for quality filtering: {path}")
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityFilterError(f"Could not read valid JSON from {metadata_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise QualityFilterError(f"{metadata_path} must contain a JSON object.")
    return loaded


def _load_required_array(path: Path, name: str) -> np.ndarray:
    array_path = path / name
    if not array_path.is_file():
        raise QualityFilterError(f"Missing required quality input: {array_path}")
    try:
        array = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise QualityFilterError(f"Could not load {array_path}: {exc}") from exc
    if array.ndim != 2 or array.shape[0] == 0:
        raise QualityFilterError(
            f"{array_path} must be a non-empty 2D array; got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise QualityFilterError(f"{array_path} contains non-finite values.")
    return np.asarray(array, dtype=np.float64)


def _require_matching_frames(path: Path, **arrays: np.ndarray) -> None:
    lengths = {name: int(array.shape[0]) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise QualityFilterError(
            f"Quality input arrays in {path} have mismatched frame counts: {lengths}."
        )


def _required_string(metadata: dict[str, Any], name: str, *, path: Path) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise QualityFilterError(
            f"{path / 'metadata.json'} must declare non-empty {name!r}."
        )
    return value


def _required_name_list(
    metadata: dict[str, Any],
    name: str,
    *,
    width: int,
    path: Path,
) -> tuple[str, ...]:
    value = metadata.get(name)
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise QualityFilterError(
            f"{path / 'metadata.json'} must declare {name!r} as {width} unique names."
        )
    return tuple(value)


def _named_column(
    array: np.ndarray,
    names: tuple[str, ...],
    name: str,
    *,
    path: Path,
) -> np.ndarray:
    try:
        index = names.index(name)
    except ValueError as exc:
        raise QualityFilterError(
            f"{path / 'metadata.json'} tracking_quality_fields is missing {name!r}."
        ) from exc
    return array[:, index]


def _mean_named_column(
    array: np.ndarray,
    names: tuple[str, ...],
    name: str,
    *,
    path: Path,
) -> float:
    values = _named_column(array, names, name, path=path)
    if np.any((values < 0.0) | (values > 1.0)):
        raise QualityFilterError(f"{path / 'tracking_quality.npy'} {name} must be in [0, 1].")
    return float(np.mean(values))


def _difference_norm_percentile(values: np.ndarray, *, order: int) -> float:
    if values.shape[0] <= order:
        return 0.0
    differences = np.diff(values, n=order, axis=0)
    norms = np.linalg.norm(differences, axis=1)
    return float(np.percentile(norms, 95))


def _action_ranges(metadata: dict[str, Any], *, path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    action_schema = metadata.get("action_schema")
    if not isinstance(action_schema, dict):
        raise QualityFilterError(f"{path / 'metadata.json'} is missing action_schema.")
    return (
        _metadata_range(action_schema, "base_position_target", path=path),
        _metadata_range(action_schema, "finger_actuator_targets", path=path),
    )


def _metadata_range(
    mapping: dict[str, Any],
    name: str,
    *,
    path: Path,
) -> tuple[int, int]:
    value = mapping.get(name)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise QualityFilterError(
            f"{path / 'metadata.json'} action_schema.{name} must be [start, stop]."
        )
    return int(value[0]), int(value[1])


def _joint_limit_hit_fraction(
    actions: np.ndarray,
    metadata: dict[str, Any],
    *,
    margin_fraction: float,
    path: Path,
) -> float:
    _base_range, finger_range = _action_ranges(metadata, path=path)
    start, stop = finger_range
    if stop > actions.shape[1]:
        raise QualityFilterError(
            f"{path / 'actions.npy'} is too narrow for finger_actuator_targets {finger_range}."
        )
    names = _required_name_list(
        metadata,
        "finger_target_names",
        width=stop - start,
        path=path,
    )
    bounds = _actuator_bounds(metadata, path=path)
    missing = [name for name in names if name not in bounds]
    if missing:
        raise QualityFilterError(
            f"{path / 'metadata.json'} has no min/max limits for actuators: "
            + ", ".join(missing)
        )
    lower = np.asarray([bounds[name][0] for name in names], dtype=np.float64)
    upper = np.asarray([bounds[name][1] for name in names], dtype=np.float64)
    margin = np.maximum((upper - lower) * margin_fraction, 1e-9)
    targets = actions[:, start:stop]
    hits = (targets <= lower + margin) | (targets >= upper - margin)
    return float(np.mean(hits))


def _actuator_bounds(
    metadata: dict[str, Any],
    *,
    path: Path,
) -> dict[str, tuple[float, float]]:
    teleop = metadata.get("teleop_config")
    if not isinstance(teleop, dict):
        raise QualityFilterError(f"{path / 'metadata.json'} is missing teleop_config.")
    retargeting = teleop.get("retargeting")
    if not isinstance(retargeting, dict):
        raise QualityFilterError(
            f"{path / 'metadata.json'} teleop_config is missing retargeting."
        )
    target_groups: list[dict[str, Any]] = []
    static_targets = retargeting.get("static_targets")
    if isinstance(static_targets, dict):
        target_groups.append(static_targets)
    fingers = retargeting.get("fingers")
    if isinstance(fingers, dict):
        for finger in fingers.values():
            if isinstance(finger, dict) and isinstance(finger.get("targets"), dict):
                target_groups.append(finger["targets"])

    bounds: dict[str, tuple[float, float]] = {}
    for targets in target_groups:
        for actuator_name, config in targets.items():
            if not isinstance(actuator_name, str) or not isinstance(config, dict):
                continue
            minimum = config.get("min")
            maximum = config.get("max")
            if not isinstance(minimum, (int, float)) or not isinstance(
                maximum, (int, float)
            ):
                continue
            lower = float(minimum)
            upper = float(maximum)
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                raise QualityFilterError(
                    f"{path / 'metadata.json'} has invalid limits for {actuator_name!r}."
                )
            bounds[actuator_name] = (lower, upper)
    return bounds


def _workspace_limit_hit_fraction(
    actions: np.ndarray,
    metadata: dict[str, Any],
    *,
    margin_fraction: float,
    path: Path,
) -> float:
    base_range, _finger_range = _action_ranges(metadata, path=path)
    start, stop = base_range
    if stop - start != 3 or stop > actions.shape[1]:
        raise QualityFilterError(
            f"{path / 'metadata.json'} base_position_target must select 3 action columns."
        )
    teleop = metadata.get("teleop_config")
    base_control = teleop.get("base_control") if isinstance(teleop, dict) else None
    workspace = base_control.get("workspace_limits") if isinstance(base_control, dict) else None
    if not isinstance(workspace, dict):
        raise QualityFilterError(
            f"{path / 'metadata.json'} is missing teleop_config.base_control.workspace_limits."
        )
    lower = _three_vector(workspace.get("min"), "workspace minimum", path=path)
    upper = _three_vector(workspace.get("max"), "workspace maximum", path=path)
    if np.any(upper <= lower):
        raise QualityFilterError(f"{path / 'metadata.json'} workspace limits are invalid.")
    margin = np.maximum((upper - lower) * margin_fraction, 1e-9)
    positions = actions[:, start:stop]
    hit_frames = np.any(
        (positions <= lower + margin) | (positions >= upper - margin),
        axis=1,
    )
    return float(np.mean(hit_frames))


def _three_vector(value: Any, name: str, *, path: Path) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QualityFilterError(
            f"{path / 'metadata.json'} {name} must be a numeric length-3 vector."
        ) from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise QualityFilterError(
            f"{path / 'metadata.json'} {name} must be a finite length-3 vector."
        )
    return array
