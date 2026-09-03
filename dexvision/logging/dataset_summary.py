"""Read-only summaries for saved Level 2 demonstration datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexvision.logging.quality_filters import DEFAULT_REPORT_NAME as QUALITY_REPORT_NAME
from dexvision.logging.relabel_success import DEFAULT_REPORT_NAME as RELABEL_REPORT_NAME


DATASET_SUMMARY_VERSION = "level2/dataset-summary-v3"
DEFAULT_JSON_NAME = "dataset_summary.json"
DEFAULT_CSV_NAME = "dataset_summary.csv"
DEFAULT_REPORT_DIRECTORY = Path("reports") / "summaries"
DEFAULT_REACH_TOUCH_CONFIG = Path("configs/reach_touch_dataset.yaml")
DEFAULT_BUTTON_PRESS_CONFIG = Path("configs/button_press_dataset.yaml")
REACH_TOUCH_TASK_ID = "reach_touch_target"
BUTTON_PRESS_TASK_ID = "button_press"
PUSH_CUBE_TASK_ID = "push_cube_to_target"


class DatasetSummaryError(RuntimeError):
    """Raised when saved dataset inputs cannot be summarized safely."""


@dataclass(frozen=True)
class TargetDefinition:
    """One configured train or held-out target position."""

    target_id: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class ReachTouchDatasetConfig:
    """Versioned readiness and target-split contract."""

    version: str
    task_id: str
    minimum_clean_successful_episodes: int
    minimum_clean_per_training_target: int
    position_units: str
    coordinate_frame: str
    training_targets: tuple[TargetDefinition, ...]
    held_out_evaluation_targets: tuple[TargetDefinition, ...]


@dataclass(frozen=True)
class ButtonGoalDefinition:
    """One configured button identity and target-depth goal."""

    goal_id: str
    button_id: str
    button_position: tuple[float, float, float]
    target_press_depth: float


@dataclass(frozen=True)
class ButtonPressDatasetConfig:
    """Versioned button-press readiness and held-out-state contract."""

    version: str
    task_id: str
    minimum_clean_successful_episodes: int
    minimum_clean_per_training_goal: int
    position_units: str
    press_depth_units: str
    coordinate_frame: str
    training_goals: tuple[ButtonGoalDefinition, ...]
    held_out_evaluation_goals: tuple[ButtonGoalDefinition, ...]


@dataclass(frozen=True)
class TargetPositionSummary:
    """Recorded distribution and clean count for one training target."""

    target_id: str
    position: tuple[float, float, float]
    num_episodes: int
    num_recomputed_success: int
    quality_pass_count: int
    clean_success_count: int


@dataclass(frozen=True)
class ButtonGoalSummary:
    """Recorded distribution and clean count for one button goal."""

    goal_id: str
    button_id: str
    button_position: tuple[float, float, float]
    target_press_depth: float
    num_episodes: int
    num_recomputed_success: int
    quality_pass_count: int
    clean_success_count: int


@dataclass(frozen=True)
class ButtonInitialStateSummary:
    """Observed button/robot initial state and its episode counts."""

    button_id: str
    button_position: tuple[float, float, float]
    initial_button_depth: float
    initial_base_position: tuple[float, float, float]
    initial_base_orientation: tuple[float, float, float, float]
    num_episodes: int
    clean_success_count: int


@dataclass(frozen=True)
class QualityFailureSummary:
    """Quality-filter failure details for one episode."""

    episode_id: str
    episode_directory: str
    failed_filters: tuple[str, ...]


@dataclass(frozen=True)
class RelabelDisagreementSummary:
    """Operator/recomputed label disagreement for one episode."""

    episode_id: str
    episode_directory: str
    operator_success: bool
    recomputed_success: bool


@dataclass(frozen=True)
class SkillDatasetSummary:
    """Aggregate summary for one skill/task pair."""

    skill_name: str
    task_id: str
    num_episodes: int
    num_success: int
    num_unlabeled: int
    success_rate: float | None
    mean_episode_length: float
    mean_tracking_confidence: float
    quality_pass_count: int
    quality_fail_count: int
    quality_unreported_count: int
    relabel_disagreement_count: int
    relabel_unreported_count: int
    action_schema_version: str
    observation_schema_version: str
    action_schema_versions: tuple[str, ...]
    observation_schema_versions: tuple[str, ...]
    clean_success_count: int
    target_position_distribution: tuple[TargetPositionSummary, ...]
    button_goal_distribution: tuple[ButtonGoalSummary, ...]
    button_initial_state_distribution: tuple[ButtonInitialStateSummary, ...]
    held_out_evaluation_targets: tuple[TargetDefinition, ...]
    held_out_button_goals: tuple[ButtonGoalDefinition, ...]
    readiness_config_version: str | None
    minimum_clean_success_count: int | None
    minimum_clean_per_training_target: int | None
    minimum_clean_per_training_goal: int | None
    level3_ready: bool | None
    readiness_failures: tuple[str, ...]
    quality_failures: tuple[QualityFailureSummary, ...]
    relabel_disagreements: tuple[RelabelDisagreementSummary, ...]


@dataclass(frozen=True)
class DatasetSummaryReport:
    """Dataset-level summary grouped by skill and task."""

    version: str
    dataset: str
    num_groups: int
    num_episodes: int
    raw_episodes_modified: bool
    warnings: tuple[str, ...]
    groups: tuple[SkillDatasetSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class _EpisodeSummaryInput:
    path: Path
    episode_id: str
    skill_name: str
    task_id: str
    success: bool | None
    episode_length: int
    mean_tracking_confidence: float
    action_schema_version: str
    observation_schema_version: str
    target_source: str | None
    target_position: tuple[float, float, float] | None
    button_id: str | None
    button_position: tuple[float, float, float] | None
    target_press_depth: float | None
    initial_button_depth: float | None
    initial_base_position: tuple[float, float, float] | None
    initial_base_orientation: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class _QualityResult:
    passed: bool
    failed_filters: tuple[str, ...]


@dataclass(frozen=True)
class _RelabelResult:
    operator_success: bool | None
    recomputed_success: bool
    labels_agree: bool | None


@dataclass(frozen=True)
class _ReportIndex:
    quality_by_path: dict[Path, _QualityResult]
    quality_by_episode_id: dict[str, _QualityResult]
    relabel_by_path: dict[Path, _RelabelResult]
    relabel_by_episode_id: dict[str, _RelabelResult]


def summarize_demo_dataset(
    dataset_dir: str | Path,
    *,
    reach_touch_config: ReachTouchDatasetConfig | None = None,
    button_press_config: ButtonPressDatasetConfig | None = None,
) -> DatasetSummaryReport:
    """Summarize every saved episode below ``dataset_dir`` without modifying it."""

    dataset = Path(dataset_dir)
    warnings: list[str] = []
    search_root = _dataset_search_root(dataset)
    episode_dirs = _episode_directories(search_root, warnings=warnings)
    if not episode_dirs:
        return DatasetSummaryReport(
            version=DATASET_SUMMARY_VERSION,
            dataset=str(dataset),
            num_groups=0,
            num_episodes=0,
            raw_episodes_modified=False,
            warnings=tuple(warnings),
            groups=(),
        )

    episodes = tuple(_load_episode(path) for path in episode_dirs)
    reports = _load_report_index(search_root)
    group_keys = sorted({(episode.skill_name, episode.task_id) for episode in episodes})
    groups = tuple(
        _summarize_group(
            tuple(
                episode
                for episode in episodes
                if (episode.skill_name, episode.task_id) == group_key
            ),
            reports=reports,
            warnings=warnings,
            reach_touch_config=reach_touch_config,
            button_press_config=button_press_config,
        )
        for group_key in group_keys
    )
    return DatasetSummaryReport(
        version=DATASET_SUMMARY_VERSION,
        dataset=str(dataset),
        num_groups=len(groups),
        num_episodes=len(episodes),
        raw_episodes_modified=False,
        warnings=tuple(warnings),
        groups=groups,
    )


def save_dataset_summary(
    report: DatasetSummaryReport,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Save JSON and CSV summary outputs atomically."""

    json_output = Path(json_path)
    csv_output = Path(csv_path)
    if json_output.suffix.lower() != ".json":
        raise DatasetSummaryError("dataset summary JSON output must use a .json extension.")
    if csv_output.suffix.lower() != ".csv":
        raise DatasetSummaryError("dataset summary CSV output must use a .csv extension.")

    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = json_output.with_name(f".{json_output.name}.tmp")
    csv_temporary = csv_output.with_name(f".{csv_output.name}.tmp")
    json_temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_temporary.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "skill_name",
            "task_id",
            "num_episodes",
            "num_success",
            "num_unlabeled",
            "success_rate",
            "mean_episode_length",
            "mean_tracking_confidence",
            "quality_pass_count",
            "quality_fail_count",
            "quality_unreported_count",
            "relabel_disagreement_count",
            "relabel_unreported_count",
            "action_schema_version",
            "observation_schema_version",
            "action_schema_versions",
            "observation_schema_versions",
            "clean_success_count",
            "target_position_distribution",
            "button_goal_distribution",
            "button_initial_state_distribution",
            "held_out_evaluation_targets",
            "held_out_button_goals",
            "readiness_config_version",
            "minimum_clean_success_count",
            "minimum_clean_per_training_target",
            "minimum_clean_per_training_goal",
            "level3_ready",
            "readiness_failures",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in report.groups:
            row = asdict(group)
            writer.writerow(
                {
                    field: _csv_value(field, row[field])
                    for field in fieldnames
                }
            )
    json_temporary.replace(json_output)
    csv_temporary.replace(csv_output)
    return json_output, csv_output


def default_summary_paths(dataset_dir: str | Path) -> tuple[Path, Path]:
    """Return the standard JSON/CSV report paths for a dataset root."""

    report_dir = Path(dataset_dir) / DEFAULT_REPORT_DIRECTORY
    return report_dir / DEFAULT_JSON_NAME, report_dir / DEFAULT_CSV_NAME


def load_reach_touch_dataset_config(
    config_path: str | Path,
) -> ReachTouchDatasetConfig:
    """Load and validate the versioned reach-touch train/evaluation split."""

    path = Path(config_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetSummaryError(f"Reach-touch dataset config does not exist: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetSummaryError(
            f"Could not read valid YAML from reach-touch dataset config {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetSummaryError(f"{path} must contain a YAML mapping.")

    version = _config_string(payload, "version", path=path)
    task_id = _config_string(payload, "task_id", path=path)
    if task_id != REACH_TOUCH_TASK_ID:
        raise DatasetSummaryError(
            f"{path} task_id must be {REACH_TOUCH_TASK_ID!r}; got {task_id!r}."
        )
    minimum_clean = _config_positive_int(
        payload,
        "minimum_clean_successful_episodes",
        path=path,
    )
    minimum_per_target = _config_positive_int(
        payload,
        "minimum_clean_per_training_target",
        path=path,
    )
    units = _config_string(payload, "position_units", path=path)
    coordinate_frame = _config_string(payload, "coordinate_frame", path=path)
    training_targets = _config_targets(payload, "training_targets", path=path)
    held_out_targets = _config_targets(
        payload,
        "held_out_evaluation_targets",
        path=path,
    )
    if not training_targets:
        raise DatasetSummaryError(f"{path} must declare at least one training target.")
    if not held_out_targets:
        raise DatasetSummaryError(
            f"{path} must declare at least one held-out evaluation target."
        )
    training_ids = {target.target_id for target in training_targets}
    held_out_ids = {target.target_id for target in held_out_targets}
    overlap = training_ids & held_out_ids
    if overlap:
        raise DatasetSummaryError(
            f"{path} target ids cannot be both training and held-out: {sorted(overlap)}."
        )
    training_positions = {target.position for target in training_targets}
    held_out_positions = {target.position for target in held_out_targets}
    if training_positions & held_out_positions:
        raise DatasetSummaryError(
            f"{path} held-out positions must be distinct from training positions."
        )
    return ReachTouchDatasetConfig(
        version=version,
        task_id=task_id,
        minimum_clean_successful_episodes=minimum_clean,
        minimum_clean_per_training_target=minimum_per_target,
        position_units=units,
        coordinate_frame=coordinate_frame,
        training_targets=training_targets,
        held_out_evaluation_targets=held_out_targets,
    )


def load_button_press_dataset_config(
    config_path: str | Path,
) -> ButtonPressDatasetConfig:
    """Load and validate the versioned button-goal train/evaluation split."""

    path = Path(config_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetSummaryError(f"Button dataset config does not exist: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetSummaryError(
            f"Could not read valid YAML from button dataset config {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetSummaryError(f"{path} must contain a YAML mapping.")

    version = _config_string(payload, "version", path=path)
    task_id = _config_string(payload, "task_id", path=path)
    if task_id != BUTTON_PRESS_TASK_ID:
        raise DatasetSummaryError(
            f"{path} task_id must be {BUTTON_PRESS_TASK_ID!r}; got {task_id!r}."
        )
    minimum_clean = _config_positive_int(
        payload,
        "minimum_clean_successful_episodes",
        path=path,
    )
    minimum_per_goal = _config_positive_int(
        payload,
        "minimum_clean_per_training_goal",
        path=path,
    )
    units = _config_string(payload, "position_units", path=path)
    depth_units = _config_string(payload, "press_depth_units", path=path)
    coordinate_frame = _config_string(payload, "coordinate_frame", path=path)
    training_goals = _config_button_goals(payload, "training_goals", path=path)
    held_out_goals = _config_button_goals(
        payload,
        "held_out_evaluation_goals",
        path=path,
    )
    if not training_goals:
        raise DatasetSummaryError(f"{path} must declare at least one training goal.")
    if not held_out_goals:
        raise DatasetSummaryError(
            f"{path} must declare at least one held-out evaluation goal."
        )
    training_ids = {goal.goal_id for goal in training_goals}
    held_out_ids = {goal.goal_id for goal in held_out_goals}
    overlap = training_ids & held_out_ids
    if overlap:
        raise DatasetSummaryError(
            f"{path} goal ids cannot be both training and held-out: {sorted(overlap)}."
        )
    training_states = {
        (goal.button_id, goal.target_press_depth) for goal in training_goals
    }
    held_out_states = {
        (goal.button_id, goal.target_press_depth) for goal in held_out_goals
    }
    if training_states & held_out_states:
        raise DatasetSummaryError(
            f"{path} held-out button/depth states must be distinct from training states."
        )
    _validate_button_positions(training_goals + held_out_goals, path=path)
    return ButtonPressDatasetConfig(
        version=version,
        task_id=task_id,
        minimum_clean_successful_episodes=minimum_clean,
        minimum_clean_per_training_goal=minimum_per_goal,
        position_units=units,
        press_depth_units=depth_units,
        coordinate_frame=coordinate_frame,
        training_goals=training_goals,
        held_out_evaluation_goals=held_out_goals,
    )


def _episode_directories(dataset: Path, *, warnings: list[str]) -> tuple[Path, ...]:
    if not dataset.exists():
        warnings.append(f"Dataset directory does not exist: {dataset}")
        return ()
    if not dataset.is_dir():
        warnings.append(f"Dataset path is not a directory: {dataset}")
        return ()
    if (dataset / "metadata.json").is_file():
        return (dataset,)
    episodes = tuple(
        sorted(
            path.parent
            for path in dataset.rglob("metadata.json")
            if path.is_file()
        )
    )
    if not episodes:
        warnings.append(
            f"No episode directories containing metadata.json were found under: {dataset}"
        )
    return episodes


def _dataset_search_root(dataset: Path) -> Path:
    raw_root = dataset / "raw"
    return raw_root if raw_root.is_dir() else dataset


def _load_episode(path: Path) -> _EpisodeSummaryInput:
    metadata = _load_json_object(path / "metadata.json", label="episode metadata")
    skill_name = _required_string(metadata, "skill_name", path=path)
    task_id = _required_string(metadata, "task_id", path=path)
    episode_id = _required_string(metadata, "episode_id", path=path)
    success = metadata.get("success")
    if success is not None and not isinstance(success, bool):
        raise DatasetSummaryError(
            f"{path / 'metadata.json'} success must be true, false, or null."
        )
    timestamps = _load_array(path / "timestamps.npy", expected_dimensions=1)
    if timestamps.size == 0:
        raise DatasetSummaryError(f"{path / 'timestamps.npy'} contains no frames.")
    if np.any(np.diff(timestamps) < 0.0):
        raise DatasetSummaryError(f"{path / 'timestamps.npy'} must be monotonic.")

    tracking = _load_array(path / "tracking_quality.npy", expected_dimensions=2)
    if tracking.shape[0] != timestamps.shape[0]:
        raise DatasetSummaryError(
            f"{path} has mismatched timestamps/tracking frame counts: "
            f"{timestamps.shape[0]} and {tracking.shape[0]}."
        )
    tracking_names = _required_name_list(
        metadata,
        "tracking_quality_fields",
        width=tracking.shape[1],
        path=path,
    )
    try:
        confidence_index = tracking_names.index("hand_tracking_confidence")
    except ValueError as exc:
        raise DatasetSummaryError(
            f"{path / 'metadata.json'} tracking_quality_fields is missing "
            "'hand_tracking_confidence'."
        ) from exc
    confidence = tracking[:, confidence_index]
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise DatasetSummaryError(
            f"{path / 'tracking_quality.npy'} hand_tracking_confidence must be in [0, 1]."
        )

    action_version = _schema_version(metadata, "action_schema", path=path)
    observation_version = _schema_version(metadata, "observation_schema", path=path)
    target_source: str | None = None
    target_position: tuple[float, float, float] | None = None
    button_id: str | None = None
    button_position: tuple[float, float, float] | None = None
    target_press_depth: float | None = None
    initial_button_depth: float | None = None
    initial_base_position: tuple[float, float, float] | None = None
    initial_base_orientation: tuple[float, float, float, float] | None = None
    if task_id in {REACH_TOUCH_TASK_ID, PUSH_CUBE_TASK_ID}:
        task_config = metadata.get("task_config")
        if not isinstance(task_config, dict):
            raise DatasetSummaryError(
                f"{path / 'metadata.json'} must declare task_config as an object."
            )
        target_source = _required_string(
            task_config,
            "resolved_target_source",
            path=path,
        )
        target_position = _position_tuple(
            task_config.get("target_position"),
            label="task_config.target_position",
            path=path / "metadata.json",
        )
    elif task_id == BUTTON_PRESS_TASK_ID:
        task_config = metadata.get("task_config")
        if not isinstance(task_config, dict):
            raise DatasetSummaryError(
                f"{path / 'metadata.json'} must declare task_config as an object."
            )
        button_id = _required_string(task_config, "resolved_button_id", path=path)
        button_position = _position_tuple(
            task_config.get("button_position"),
            label="task_config.button_position",
            path=path / "metadata.json",
        )
        target_press_depth = _finite_scalar(
            task_config.get("target_press_depth"),
            label="task_config.target_press_depth",
            path=path / "metadata.json",
        )
        initial_button_depth = _finite_scalar(
            task_config.get("initial_button_depth"),
            label="task_config.initial_button_depth",
            path=path / "metadata.json",
        )
        initial_base_position = _position_tuple(
            task_config.get("initial_base_position"),
            label="task_config.initial_base_position",
            path=path / "metadata.json",
        )
        initial_base_orientation = _vector_tuple(
            task_config.get("initial_base_orientation"),
            length=4,
            label="task_config.initial_base_orientation",
            path=path / "metadata.json",
        )
    return _EpisodeSummaryInput(
        path=path.resolve(),
        episode_id=episode_id,
        skill_name=skill_name,
        task_id=task_id,
        success=success,
        episode_length=int(timestamps.shape[0]),
        mean_tracking_confidence=float(np.mean(confidence)),
        action_schema_version=action_version,
        observation_schema_version=observation_version,
        target_source=target_source,
        target_position=target_position,
        button_id=button_id,
        button_position=button_position,
        target_press_depth=target_press_depth,
        initial_button_depth=initial_button_depth,
        initial_base_position=initial_base_position,
        initial_base_orientation=initial_base_orientation,
    )


def _summarize_group(
    episodes: tuple[_EpisodeSummaryInput, ...],
    *,
    reports: _ReportIndex,
    warnings: list[str],
    reach_touch_config: ReachTouchDatasetConfig | None,
    button_press_config: ButtonPressDatasetConfig | None,
) -> SkillDatasetSummary:
    skill_name = episodes[0].skill_name
    task_id = episodes[0].task_id
    successes: list[bool | None] = []
    quality_results: list[_QualityResult | None] = []
    relabel_results: list[_RelabelResult | None] = []
    quality_failures: list[QualityFailureSummary] = []
    disagreements: list[RelabelDisagreementSummary] = []

    for episode in episodes:
        quality = reports.quality_by_path.get(episode.path)
        if quality is None:
            quality = reports.quality_by_episode_id.get(episode.episode_id)
        quality_results.append(quality)
        if quality is not None and not quality.passed:
            quality_failures.append(
                QualityFailureSummary(
                    episode_id=episode.episode_id,
                    episode_directory=episode.path.name,
                    failed_filters=quality.failed_filters,
                )
            )

        relabel = reports.relabel_by_path.get(episode.path)
        if relabel is None:
            relabel = reports.relabel_by_episode_id.get(episode.episode_id)
        relabel_results.append(relabel)
        successes.append(
            relabel.recomputed_success if relabel is not None else episode.success
        )
        if relabel is not None and relabel.labels_agree is False:
            if relabel.operator_success is None:
                raise DatasetSummaryError(
                    f"Relabel result for {episode.episode_id!r} reports disagreement "
                    "without an operator label."
                )
            disagreements.append(
                RelabelDisagreementSummary(
                    episode_id=episode.episode_id,
                    episode_directory=episode.path.name,
                    operator_success=relabel.operator_success,
                    recomputed_success=relabel.recomputed_success,
                )
            )

    quality_unreported_count = sum(result is None for result in quality_results)
    relabel_unreported_count = sum(result is None for result in relabel_results)
    if quality_unreported_count:
        warnings.append(
            f"{skill_name}/{task_id}: quality report coverage is missing for "
            f"{quality_unreported_count} of {len(episodes)} episodes."
        )
    if relabel_unreported_count:
        warnings.append(
            f"{skill_name}/{task_id}: relabel report coverage is missing for "
            f"{relabel_unreported_count} of {len(episodes)} episodes; operator labels "
            "are used when available."
        )

    labeled = tuple(value for value in successes if value is not None)
    num_success = sum(value is True for value in labeled)
    action_versions = tuple(sorted({episode.action_schema_version for episode in episodes}))
    observation_versions = tuple(
        sorted({episode.observation_schema_version for episode in episodes})
    )
    target_distribution = _summarize_target_distribution(
        episodes,
        quality_results=tuple(quality_results),
        relabel_results=tuple(relabel_results),
        config=reach_touch_config if task_id == REACH_TOUCH_TASK_ID else None,
    )
    button_goal_distribution = _summarize_button_goal_distribution(
        episodes,
        quality_results=tuple(quality_results),
        relabel_results=tuple(relabel_results),
        config=button_press_config if task_id == BUTTON_PRESS_TASK_ID else None,
    )
    button_initial_state_distribution = _summarize_button_initial_states(
        episodes,
        quality_results=tuple(quality_results),
        relabel_results=tuple(relabel_results),
    )
    clean_success_count = (
        sum(goal.clean_success_count for goal in button_goal_distribution)
        if task_id == BUTTON_PRESS_TASK_ID
        else sum(target.clean_success_count for target in target_distribution)
    )
    readiness_failures = _readiness_failures(
        task_id=task_id,
        reach_touch_config=reach_touch_config,
        button_press_config=button_press_config,
        target_distribution=target_distribution,
        button_goal_distribution=button_goal_distribution,
        clean_success_count=clean_success_count,
        quality_unreported_count=quality_unreported_count,
        relabel_unreported_count=relabel_unreported_count,
        disagreement_count=len(disagreements),
        action_versions=action_versions,
        observation_versions=observation_versions,
    )
    reach_readiness_applies = (
        task_id == REACH_TOUCH_TASK_ID and reach_touch_config is not None
    )
    button_readiness_applies = (
        task_id == BUTTON_PRESS_TASK_ID and button_press_config is not None
    )
    readiness_applies = reach_readiness_applies or button_readiness_applies
    return SkillDatasetSummary(
        skill_name=skill_name,
        task_id=task_id,
        num_episodes=len(episodes),
        num_success=num_success,
        num_unlabeled=len(episodes) - len(labeled),
        success_rate=(num_success / len(labeled) if labeled else None),
        mean_episode_length=float(
            np.mean([episode.episode_length for episode in episodes])
        ),
        mean_tracking_confidence=float(
            np.mean([episode.mean_tracking_confidence for episode in episodes])
        ),
        quality_pass_count=sum(
            result is not None and result.passed for result in quality_results
        ),
        quality_fail_count=sum(
            result is not None and not result.passed for result in quality_results
        ),
        quality_unreported_count=quality_unreported_count,
        relabel_disagreement_count=len(disagreements),
        relabel_unreported_count=relabel_unreported_count,
        action_schema_version=_display_schema_version(action_versions),
        observation_schema_version=_display_schema_version(observation_versions),
        action_schema_versions=action_versions,
        observation_schema_versions=observation_versions,
        clean_success_count=clean_success_count,
        target_position_distribution=target_distribution,
        button_goal_distribution=button_goal_distribution,
        button_initial_state_distribution=button_initial_state_distribution,
        held_out_evaluation_targets=(
            reach_touch_config.held_out_evaluation_targets
            if reach_readiness_applies
            else ()
        ),
        held_out_button_goals=(
            button_press_config.held_out_evaluation_goals
            if button_readiness_applies
            else ()
        ),
        readiness_config_version=(
            reach_touch_config.version
            if reach_readiness_applies
            else button_press_config.version
            if button_readiness_applies
            else None
        ),
        minimum_clean_success_count=(
            reach_touch_config.minimum_clean_successful_episodes
            if reach_readiness_applies
            else button_press_config.minimum_clean_successful_episodes
            if button_readiness_applies
            else None
        ),
        minimum_clean_per_training_target=(
            reach_touch_config.minimum_clean_per_training_target
            if reach_readiness_applies
            else None
        ),
        minimum_clean_per_training_goal=(
            button_press_config.minimum_clean_per_training_goal
            if button_readiness_applies
            else None
        ),
        level3_ready=(not readiness_failures if readiness_applies else None),
        readiness_failures=readiness_failures,
        quality_failures=tuple(quality_failures),
        relabel_disagreements=tuple(disagreements),
    )


def _summarize_target_distribution(
    episodes: tuple[_EpisodeSummaryInput, ...],
    *,
    quality_results: tuple[_QualityResult | None, ...],
    relabel_results: tuple[_RelabelResult | None, ...],
    config: ReachTouchDatasetConfig | None,
) -> tuple[TargetPositionSummary, ...]:
    if episodes[0].task_id not in {REACH_TOUCH_TASK_ID, PUSH_CUBE_TASK_ID}:
        return ()

    observed_positions: dict[str, tuple[float, float, float]] = {}
    counts: dict[str, list[int]] = {}
    for episode, quality, relabel in zip(
        episodes,
        quality_results,
        relabel_results,
        strict=True,
    ):
        if episode.target_source is None or episode.target_position is None:
            raise DatasetSummaryError(
                f"{episode.path} is missing its task target identity."
            )
        previous_position = observed_positions.setdefault(
            episode.target_source,
            episode.target_position,
        )
        if not np.allclose(
            previous_position,
            episode.target_position,
            rtol=0.0,
            atol=1e-9,
        ):
            raise DatasetSummaryError(
                f"Target {episode.target_source!r} has inconsistent saved positions."
            )
        values = counts.setdefault(episode.target_source, [0, 0, 0, 0])
        values[0] += 1
        values[1] += int(relabel is not None and relabel.recomputed_success)
        values[2] += int(quality is not None and quality.passed)
        values[3] += int(
            quality is not None
            and quality.passed
            and relabel is not None
            and relabel.recomputed_success
        )

    if config is not None:
        configured = {target.target_id: target.position for target in config.training_targets}
        unknown = set(observed_positions) - set(configured)
        if unknown:
            raise DatasetSummaryError(
                "Reach-touch dataset contains targets absent from the training split: "
                f"{sorted(unknown)}."
            )
        for target_id, position in observed_positions.items():
            if not np.allclose(position, configured[target_id], rtol=0.0, atol=1e-9):
                raise DatasetSummaryError(
                    f"Saved position for {target_id!r} does not match the dataset config."
                )
        ordered_targets = config.training_targets
    else:
        ordered_targets = tuple(
            TargetDefinition(target_id=target_id, position=observed_positions[target_id])
            for target_id in sorted(observed_positions)
        )

    return tuple(
        TargetPositionSummary(
            target_id=target.target_id,
            position=target.position,
            num_episodes=counts.get(target.target_id, [0, 0, 0, 0])[0],
            num_recomputed_success=counts.get(target.target_id, [0, 0, 0, 0])[1],
            quality_pass_count=counts.get(target.target_id, [0, 0, 0, 0])[2],
            clean_success_count=counts.get(target.target_id, [0, 0, 0, 0])[3],
        )
        for target in ordered_targets
    )


def _summarize_button_goal_distribution(
    episodes: tuple[_EpisodeSummaryInput, ...],
    *,
    quality_results: tuple[_QualityResult | None, ...],
    relabel_results: tuple[_RelabelResult | None, ...],
    config: ButtonPressDatasetConfig | None,
) -> tuple[ButtonGoalSummary, ...]:
    if episodes[0].task_id != BUTTON_PRESS_TASK_ID:
        return ()

    observed: dict[tuple[str, float], tuple[float, float, float]] = {}
    counts: dict[tuple[str, float], list[int]] = {}
    for episode, quality, relabel in zip(
        episodes,
        quality_results,
        relabel_results,
        strict=True,
    ):
        if (
            episode.button_id is None
            or episode.button_position is None
            or episode.target_press_depth is None
        ):
            raise DatasetSummaryError(f"{episode.path} is missing its button goal.")
        key = (episode.button_id, episode.target_press_depth)
        previous_position = observed.setdefault(key, episode.button_position)
        if not np.allclose(
            previous_position,
            episode.button_position,
            rtol=0.0,
            atol=1e-9,
        ):
            raise DatasetSummaryError(
                f"Button goal {key!r} has inconsistent saved positions."
            )
        values = counts.setdefault(key, [0, 0, 0, 0])
        values[0] += 1
        values[1] += int(relabel is not None and relabel.recomputed_success)
        values[2] += int(quality is not None and quality.passed)
        values[3] += int(
            quality is not None
            and quality.passed
            and relabel is not None
            and relabel.recomputed_success
        )

    if config is not None:
        configured = {
            (goal.button_id, goal.target_press_depth): goal
            for goal in config.training_goals
        }
        unknown = set(observed) - set(configured)
        if unknown:
            raise DatasetSummaryError(
                "Button dataset contains goals absent from the training split: "
                f"{sorted(unknown)}."
            )
        for key, position in observed.items():
            if not np.allclose(
                position,
                configured[key].button_position,
                rtol=0.0,
                atol=1e-9,
            ):
                raise DatasetSummaryError(
                    f"Saved position for button goal {key!r} does not match the "
                    "dataset config."
                )
        ordered_goals = config.training_goals
    else:
        ordered_goals = tuple(
            ButtonGoalDefinition(
                goal_id=f"{button_id}_depth_{depth:.3f}",
                button_id=button_id,
                button_position=observed[(button_id, depth)],
                target_press_depth=depth,
            )
            for button_id, depth in sorted(observed)
        )

    return tuple(
        ButtonGoalSummary(
            goal_id=goal.goal_id,
            button_id=goal.button_id,
            button_position=goal.button_position,
            target_press_depth=goal.target_press_depth,
            num_episodes=counts.get(
                (goal.button_id, goal.target_press_depth), [0, 0, 0, 0]
            )[0],
            num_recomputed_success=counts.get(
                (goal.button_id, goal.target_press_depth), [0, 0, 0, 0]
            )[1],
            quality_pass_count=counts.get(
                (goal.button_id, goal.target_press_depth), [0, 0, 0, 0]
            )[2],
            clean_success_count=counts.get(
                (goal.button_id, goal.target_press_depth), [0, 0, 0, 0]
            )[3],
        )
        for goal in ordered_goals
    )


def _summarize_button_initial_states(
    episodes: tuple[_EpisodeSummaryInput, ...],
    *,
    quality_results: tuple[_QualityResult | None, ...],
    relabel_results: tuple[_RelabelResult | None, ...],
) -> tuple[ButtonInitialStateSummary, ...]:
    if episodes[0].task_id != BUTTON_PRESS_TASK_ID:
        return ()

    counts: dict[
        tuple[
            str,
            tuple[float, float, float],
            float,
            tuple[float, float, float],
            tuple[float, float, float, float],
        ],
        list[int],
    ] = {}
    for episode, quality, relabel in zip(
        episodes,
        quality_results,
        relabel_results,
        strict=True,
    ):
        if (
            episode.button_id is None
            or episode.button_position is None
            or episode.initial_button_depth is None
            or episode.initial_base_position is None
            or episode.initial_base_orientation is None
        ):
            raise DatasetSummaryError(
                f"{episode.path} is missing its button initial state."
            )
        key = (
            episode.button_id,
            episode.button_position,
            episode.initial_button_depth,
            episode.initial_base_position,
            episode.initial_base_orientation,
        )
        values = counts.setdefault(key, [0, 0])
        values[0] += 1
        values[1] += int(
            quality is not None
            and quality.passed
            and relabel is not None
            and relabel.recomputed_success
        )
    return tuple(
        ButtonInitialStateSummary(
            button_id=key[0],
            button_position=key[1],
            initial_button_depth=key[2],
            initial_base_position=key[3],
            initial_base_orientation=key[4],
            num_episodes=values[0],
            clean_success_count=values[1],
        )
        for key, values in sorted(counts.items())
    )


def _readiness_failures(
    *,
    task_id: str,
    reach_touch_config: ReachTouchDatasetConfig | None,
    button_press_config: ButtonPressDatasetConfig | None,
    target_distribution: tuple[TargetPositionSummary, ...],
    button_goal_distribution: tuple[ButtonGoalSummary, ...],
    clean_success_count: int,
    quality_unreported_count: int,
    relabel_unreported_count: int,
    disagreement_count: int,
    action_versions: tuple[str, ...],
    observation_versions: tuple[str, ...],
) -> tuple[str, ...]:
    if task_id == REACH_TOUCH_TASK_ID and reach_touch_config is not None:
        minimum_clean = reach_touch_config.minimum_clean_successful_episodes
        minimum_distribution = (
            (
                target.target_id,
                target.clean_success_count,
                reach_touch_config.minimum_clean_per_training_target,
            )
            for target in target_distribution
        )
        held_out_present = bool(reach_touch_config.held_out_evaluation_targets)
        recorded_states: set[object] = {
            target.position for target in target_distribution if target.num_episodes
        }
        held_out_states: tuple[tuple[str, object], ...] = tuple(
            (target.target_id, target.position)
            for target in reach_touch_config.held_out_evaluation_targets
        )
    elif task_id == BUTTON_PRESS_TASK_ID and button_press_config is not None:
        minimum_clean = button_press_config.minimum_clean_successful_episodes
        minimum_distribution = (
            (
                goal.goal_id,
                goal.clean_success_count,
                button_press_config.minimum_clean_per_training_goal,
            )
            for goal in button_goal_distribution
        )
        held_out_present = bool(button_press_config.held_out_evaluation_goals)
        recorded_states = {
            (goal.button_id, goal.target_press_depth)
            for goal in button_goal_distribution
            if goal.num_episodes
        }
        held_out_states = tuple(
            (goal.goal_id, (goal.button_id, goal.target_press_depth))
            for goal in button_press_config.held_out_evaluation_goals
        )
    else:
        return ()

    failures: list[str] = []
    if quality_unreported_count:
        failures.append(f"quality coverage missing for {quality_unreported_count} episodes")
    if relabel_unreported_count:
        failures.append(f"relabel coverage missing for {relabel_unreported_count} episodes")
    if disagreement_count:
        failures.append(f"{disagreement_count} operator/recomputed label disagreements")
    if clean_success_count < minimum_clean:
        failures.append(
            f"clean successful episodes {clean_success_count} below required "
            f"{minimum_clean}"
        )
    for goal_id, goal_clean_count, required_count in minimum_distribution:
        if goal_clean_count < required_count:
            failures.append(
                f"{goal_id} has {goal_clean_count} clean successes; "
                f"requires {required_count}"
            )
    if not held_out_present:
        failures.append("no held-out evaluation states declared")
    contaminated = tuple(
        state_id for state_id, state in held_out_states if state in recorded_states
    )
    if contaminated:
        failures.append(
            f"held-out evaluation states overlap recorded training states: {contaminated}"
        )
    if len(action_versions) != 1:
        failures.append("mixed action schema versions")
    if len(observation_versions) != 1:
        failures.append("mixed observation schema versions")
    return tuple(failures)


def _load_report_index(dataset: Path) -> _ReportIndex:
    quality_by_path: dict[Path, _QualityResult] = {}
    quality_by_episode_id: dict[str, _QualityResult] = {}
    relabel_by_path: dict[Path, _RelabelResult] = {}
    relabel_by_episode_id: dict[str, _RelabelResult] = {}

    for report_path in _report_paths(dataset, QUALITY_REPORT_NAME):
        report = _load_json_object(report_path, label="quality report")
        episodes = _required_report_episodes(report, path=report_path)
        for entry in episodes:
            episode_directory = _required_string(
                entry,
                "episode_directory",
                path=report_path.parent,
            )
            episode_id = _required_string(entry, "episode_id", path=report_path.parent)
            passed = entry.get("passed")
            failed_filters = entry.get("failed_filters")
            if not isinstance(passed, bool):
                raise DatasetSummaryError(
                    f"{report_path} episode {episode_id!r} must declare boolean 'passed'."
                )
            if (
                not isinstance(failed_filters, list)
                or any(not isinstance(item, str) or not item for item in failed_filters)
            ):
                raise DatasetSummaryError(
                    f"{report_path} episode {episode_id!r} must declare "
                    "failed_filters as a list of strings."
                )
            result = _QualityResult(
                passed=passed,
                failed_filters=tuple(failed_filters),
            )
            quality_by_path[(report_path.parent / episode_directory).resolve()] = result
            _store_unique_episode_id(
                quality_by_episode_id,
                episode_id,
                result,
                report_path=report_path,
            )

    for report_path in _report_paths(dataset, RELABEL_REPORT_NAME):
        report = _load_json_object(report_path, label="relabel report")
        episodes = _required_report_episodes(report, path=report_path)
        for entry in episodes:
            episode_directory = _required_string(
                entry,
                "episode_directory",
                path=report_path.parent,
            )
            episode_id = _required_string(entry, "episode_id", path=report_path.parent)
            operator_success = entry.get("operator_success")
            recomputed_success = entry.get("recomputed_success")
            labels_agree = entry.get("labels_agree")
            if operator_success is not None and not isinstance(operator_success, bool):
                raise DatasetSummaryError(
                    f"{report_path} episode {episode_id!r} has invalid operator_success."
                )
            if not isinstance(recomputed_success, bool):
                raise DatasetSummaryError(
                    f"{report_path} episode {episode_id!r} must declare boolean "
                    "recomputed_success."
                )
            if labels_agree is not None and not isinstance(labels_agree, bool):
                raise DatasetSummaryError(
                    f"{report_path} episode {episode_id!r} has invalid labels_agree."
                )
            result = _RelabelResult(
                operator_success=operator_success,
                recomputed_success=recomputed_success,
                labels_agree=labels_agree,
            )
            relabel_by_path[(report_path.parent / episode_directory).resolve()] = result
            _store_unique_episode_id(
                relabel_by_episode_id,
                episode_id,
                result,
                report_path=report_path,
            )
    return _ReportIndex(
        quality_by_path=quality_by_path,
        quality_by_episode_id=quality_by_episode_id,
        relabel_by_path=relabel_by_path,
        relabel_by_episode_id=relabel_by_episode_id,
    )


def _report_paths(dataset: Path, report_name: str) -> tuple[Path, ...]:
    if not dataset.exists() or not dataset.is_dir():
        return ()
    if dataset.name == report_name and dataset.is_file():
        return (dataset,)
    return tuple(sorted(path for path in dataset.rglob(report_name) if path.is_file()))


def _required_report_episodes(
    report: dict[str, Any],
    *,
    path: Path,
) -> tuple[dict[str, Any], ...]:
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or any(
        not isinstance(entry, dict) for entry in episodes
    ):
        raise DatasetSummaryError(f"{path} must declare 'episodes' as a list of objects.")
    return tuple(episodes)


def _store_unique_episode_id(
    mapping: dict[str, Any],
    episode_id: str,
    result: Any,
    *,
    report_path: Path,
) -> None:
    if episode_id in mapping and mapping[episode_id] != result:
        raise DatasetSummaryError(
            f"Conflicting report entries found for episode_id {episode_id!r} "
            f"while reading {report_path}."
        )
    mapping[episode_id] = result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetSummaryError(f"Missing {label}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetSummaryError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DatasetSummaryError(f"{path} must contain a JSON object.")
    return loaded


def _load_array(path: Path, *, expected_dimensions: int) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise DatasetSummaryError(f"Missing summary input: {path}") from exc
    except (OSError, ValueError) as exc:
        raise DatasetSummaryError(f"Could not load {path}: {exc}") from exc
    if array.ndim != expected_dimensions:
        raise DatasetSummaryError(
            f"{path} must be {expected_dimensions}D; got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise DatasetSummaryError(f"{path} contains non-finite values.")
    return np.asarray(array, dtype=np.float64)


def _required_string(metadata: dict[str, Any], name: str, *, path: Path) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise DatasetSummaryError(
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
        raise DatasetSummaryError(
            f"{path / 'metadata.json'} must declare {name!r} as {width} unique names."
        )
    return tuple(value)


def _schema_version(metadata: dict[str, Any], schema_name: str, *, path: Path) -> str:
    top_level_name = f"{schema_name}_version"
    top_level = metadata.get(top_level_name)
    schema = metadata.get(schema_name)
    nested = schema.get("version") if isinstance(schema, dict) else None
    versions = tuple(
        version
        for version in (top_level, nested)
        if isinstance(version, str) and version
    )
    if not versions:
        raise DatasetSummaryError(
            f"{path / 'metadata.json'} must declare {top_level_name!r} or "
            f"{schema_name}.version."
        )
    if len(set(versions)) > 1:
        raise DatasetSummaryError(
            f"{path / 'metadata.json'} has inconsistent {schema_name} versions: "
            f"{versions}."
        )
    return versions[0]


def _display_schema_version(versions: tuple[str, ...]) -> str:
    return versions[0] if len(versions) == 1 else "mixed"


def _position_tuple(
    value: object,
    *,
    label: str,
    path: Path,
) -> tuple[float, float, float]:
    try:
        position = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DatasetSummaryError(
            f"{path} {label} must contain three finite numbers."
        ) from exc
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise DatasetSummaryError(
            f"{path} {label} must contain three finite numbers."
        )
    return tuple(float(item) for item in position)


def _vector_tuple(
    value: object,
    *,
    length: int,
    label: str,
    path: Path,
) -> tuple[float, ...]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DatasetSummaryError(
            f"{path} {label} must contain {length} finite numbers."
        ) from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise DatasetSummaryError(
            f"{path} {label} must contain {length} finite numbers."
        )
    return tuple(float(item) for item in vector)


def _finite_scalar(value: object, *, label: str, path: Path) -> float:
    if isinstance(value, bool):
        raise DatasetSummaryError(f"{path} {label} must be a finite number.")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetSummaryError(f"{path} {label} must be a finite number.") from exc
    if not np.isfinite(scalar):
        raise DatasetSummaryError(f"{path} {label} must be a finite number.")
    return scalar


def _config_string(payload: dict[str, Any], name: str, *, path: Path) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise DatasetSummaryError(f"{path} must declare non-empty {name!r}.")
    return value


def _config_positive_int(payload: dict[str, Any], name: str, *, path: Path) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DatasetSummaryError(f"{path} {name!r} must be a positive integer.")
    return value


def _config_targets(
    payload: dict[str, Any],
    name: str,
    *,
    path: Path,
) -> tuple[TargetDefinition, ...]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DatasetSummaryError(f"{path} {name!r} must be a target-id mapping.")
    targets: list[TargetDefinition] = []
    for target_id, position in value.items():
        if not isinstance(target_id, str) or not target_id:
            raise DatasetSummaryError(f"{path} {name!r} contains an invalid target id.")
        targets.append(
            TargetDefinition(
                target_id=target_id,
                position=_position_tuple(
                    position,
                    label=f"{name}.{target_id}",
                    path=path,
                ),
            )
        )
    positions = [target.position for target in targets]
    if len(set(positions)) != len(positions):
        raise DatasetSummaryError(f"{path} {name!r} contains duplicate positions.")
    return tuple(targets)


def _config_button_goals(
    payload: dict[str, Any],
    name: str,
    *,
    path: Path,
) -> tuple[ButtonGoalDefinition, ...]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DatasetSummaryError(f"{path} {name!r} must be a goal-id mapping.")
    goals: list[ButtonGoalDefinition] = []
    for goal_id, goal_payload in value.items():
        if not isinstance(goal_id, str) or not goal_id:
            raise DatasetSummaryError(f"{path} {name!r} contains an invalid goal id.")
        if not isinstance(goal_payload, dict):
            raise DatasetSummaryError(
                f"{path} {name}.{goal_id} must be a mapping."
            )
        button_id = _config_string(goal_payload, "button_id", path=path)
        position = _position_tuple(
            goal_payload.get("button_position"),
            label=f"{name}.{goal_id}.button_position",
            path=path,
        )
        depth = _finite_scalar(
            goal_payload.get("target_press_depth"),
            label=f"{name}.{goal_id}.target_press_depth",
            path=path,
        )
        if depth <= 0.0:
            raise DatasetSummaryError(
                f"{path} {name}.{goal_id}.target_press_depth must be positive."
            )
        goals.append(
            ButtonGoalDefinition(
                goal_id=goal_id,
                button_id=button_id,
                button_position=position,
                target_press_depth=depth,
            )
        )
    states = [(goal.button_id, goal.target_press_depth) for goal in goals]
    if len(set(states)) != len(states):
        raise DatasetSummaryError(
            f"{path} {name!r} contains duplicate button/depth states."
        )
    return tuple(goals)


def _validate_button_positions(
    goals: tuple[ButtonGoalDefinition, ...],
    *,
    path: Path,
) -> None:
    positions: dict[str, tuple[float, float, float]] = {}
    for goal in goals:
        previous = positions.setdefault(goal.button_id, goal.button_position)
        if previous != goal.button_position:
            raise DatasetSummaryError(
                f"{path} gives button {goal.button_id!r} inconsistent positions."
            )


def _csv_value(field: str, value: Any) -> Any:
    if field in {
        "action_schema_versions",
        "observation_schema_versions",
        "readiness_failures",
    }:
        return ";".join(value)
    if field in {"target_position_distribution", "held_out_evaluation_targets"}:
        return json.dumps(value, sort_keys=True)
    return value
