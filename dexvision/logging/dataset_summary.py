"""Read-only summaries for saved Level 2 demonstration datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.quality_filters import DEFAULT_REPORT_NAME as QUALITY_REPORT_NAME
from dexvision.logging.relabel_success import DEFAULT_REPORT_NAME as RELABEL_REPORT_NAME


DATASET_SUMMARY_VERSION = "level2/dataset-summary-v1"
DEFAULT_JSON_NAME = "dataset_summary.json"
DEFAULT_CSV_NAME = "dataset_summary.csv"
DEFAULT_REPORT_DIRECTORY = Path("reports") / "summaries"


class DatasetSummaryError(RuntimeError):
    """Raised when saved dataset inputs cannot be summarized safely."""


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


def summarize_demo_dataset(dataset_dir: str | Path) -> DatasetSummaryReport:
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
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in report.groups:
            row = asdict(group)
            writer.writerow(
                {
                    field: (
                        ";".join(row[field])
                        if field
                        in {"action_schema_versions", "observation_schema_versions"}
                        else row[field]
                    )
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
    )


def _summarize_group(
    episodes: tuple[_EpisodeSummaryInput, ...],
    *,
    reports: _ReportIndex,
    warnings: list[str],
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
        quality_failures=tuple(quality_failures),
        relabel_disagreements=tuple(disagreements),
    )


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
