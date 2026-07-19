"""Success relabeling for saved Level 2 task demonstrations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.sim.tasks import (
    BUTTON_PRESS_TASK_ID,
    REACH_TOUCH_TARGET_TASK_ID,
    ReachTouchTargetConfig,
    is_button_press_success,
)


RELABEL_REPORT_VERSION = "level2/reach-touch-success-v1"
BUTTON_PRESS_RELABEL_REPORT_VERSION = "level2/button-press-success-v1"
DEFAULT_REPORT_NAME = "relabel_report.json"
REQUIRED_METRIC_INPUTS = (
    "target_position",
    "touch_position",
    "distance_to_target",
    "palm_contact",
)
TARGET_POSITION_COLUMNS = slice(0, 3)
TOUCH_POSITION_COLUMNS = slice(3, 6)
DISTANCE_COLUMN = 6
PALM_CONTACT_COLUMN = 7
REQUIRED_TASK_STATE_WIDTH = 8
DISTANCE_TOLERANCE_M = 1e-6
BUTTON_REQUIRED_METRIC_INPUTS = (
    "press_depth",
    "target_press_depth",
    "button_pressed",
    "target_pressed_state",
    "dwell_steps",
)
BUTTON_REQUIRED_TASK_STATE_WIDTH = 5
BUTTON_PRESS_DEPTH_COLUMN = 0
BUTTON_TARGET_DEPTH_COLUMN = 1
BUTTON_PRESSED_COLUMN = 2
BUTTON_TARGET_PRESSED_COLUMN = 3
BUTTON_DWELL_COLUMN = 4


class SuccessRelabelError(RuntimeError):
    """Raised when saved inputs cannot produce an auditable success label."""


@dataclass(frozen=True)
class EpisodeRelabelResult:
    """Operator and recomputed labels for one immutable raw episode."""

    episode_directory: str
    episode_id: str
    task_id: str
    operator_success: bool | None
    recomputed_success: bool
    labels_agree: bool | None
    frame_count: int
    first_success_frame: int | None
    max_consecutive_contact_frames: int


@dataclass(frozen=True)
class RelabelReport:
    """Dataset-level audit report for reach-touch relabeling."""

    version: str
    task_id: str
    dataset: str
    distance_threshold_m: float
    required_dwell_frames: int
    episode_count: int
    recomputed_success_count: int
    operator_success_count: int
    label_disagreement_count: int
    raw_episodes_modified: bool
    episodes: tuple[EpisodeRelabelResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class ButtonPressEpisodeRelabelResult:
    """Operator and recomputed labels for one button-press episode."""

    episode_directory: str
    episode_id: str
    task_id: str
    operator_success: bool | None
    recomputed_success: bool
    labels_agree: bool | None
    frame_count: int
    first_success_frame: int | None
    max_consecutive_goal_frames: int
    resolved_button_id: str
    target_press_depth_m: float
    target_pressed_state: bool


@dataclass(frozen=True)
class ButtonPressRelabelReport:
    """Dataset-level audit report for button-press relabeling."""

    version: str
    task_id: str
    dataset: str
    required_dwell_frames: int
    episode_count: int
    recomputed_success_count: int
    operator_success_count: int
    label_disagreement_count: int
    raw_episodes_modified: bool
    episodes: tuple[ButtonPressEpisodeRelabelResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def relabel_reach_touch_episode(
    episode_dir: str | Path,
    *,
    config: ReachTouchTargetConfig | None = None,
) -> EpisodeRelabelResult:
    """Recompute one reach-touch label from its saved metric inputs."""

    path = Path(episode_dir)
    metadata = _load_metadata(path)
    _validate_reach_touch_metadata(metadata, path=path)
    task_states = _load_task_states(path)
    metric_inputs = task_states[:, :REQUIRED_TASK_STATE_WIDTH]
    _validate_metric_inputs(metric_inputs, path=path)

    target_positions = metric_inputs[:, TARGET_POSITION_COLUMNS]
    touch_positions = metric_inputs[:, TOUCH_POSITION_COLUMNS]
    recomputed_distances = np.linalg.norm(touch_positions - target_positions, axis=1)
    saved_distances = metric_inputs[:, DISTANCE_COLUMN]
    if not np.allclose(
        saved_distances,
        recomputed_distances,
        rtol=0.0,
        atol=DISTANCE_TOLERANCE_M,
    ):
        worst_error = float(np.max(np.abs(saved_distances - recomputed_distances)))
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has inconsistent distance_to_target values; "
            f"maximum position-derived error is {worst_error:.9f} m."
        )

    metric_config = config or ReachTouchTargetConfig()
    contact_flags = metric_inputs[:, PALM_CONTACT_COLUMN] > 0.5
    qualifying_frames = contact_flags & (
        recomputed_distances <= metric_config.success_distance_m
    )
    first_success_frame, max_dwell = _success_dwell(
        qualifying_frames,
        required_dwell_frames=metric_config.success_dwell_steps,
    )
    recomputed_success = first_success_frame is not None
    operator_success = _operator_success(metadata, path=path)

    return EpisodeRelabelResult(
        episode_directory=path.name,
        episode_id=str(metadata.get("episode_id", path.name)),
        task_id=REACH_TOUCH_TARGET_TASK_ID,
        operator_success=operator_success,
        recomputed_success=recomputed_success,
        labels_agree=(
            None if operator_success is None else operator_success == recomputed_success
        ),
        frame_count=int(task_states.shape[0]),
        first_success_frame=first_success_frame,
        max_consecutive_contact_frames=max_dwell,
    )


def relabel_reach_touch_dataset(
    dataset_dir: str | Path,
    *,
    config: ReachTouchTargetConfig | None = None,
) -> RelabelReport:
    """Relabel every reach-touch episode immediately below a dataset directory."""

    dataset = Path(dataset_dir)
    episode_dirs = _episode_directories(dataset)
    metric_config = config or ReachTouchTargetConfig()
    episodes = tuple(
        relabel_reach_touch_episode(path, config=metric_config)
        for path in episode_dirs
    )
    operator_success_count = sum(
        result.operator_success is True for result in episodes
    )
    disagreement_count = sum(result.labels_agree is False for result in episodes)
    return RelabelReport(
        version=RELABEL_REPORT_VERSION,
        task_id=REACH_TOUCH_TARGET_TASK_ID,
        dataset=str(dataset),
        distance_threshold_m=metric_config.success_distance_m,
        required_dwell_frames=metric_config.success_dwell_steps,
        episode_count=len(episodes),
        recomputed_success_count=sum(
            result.recomputed_success for result in episodes
        ),
        operator_success_count=operator_success_count,
        label_disagreement_count=disagreement_count,
        raw_episodes_modified=False,
        episodes=episodes,
    )


def relabel_button_press_episode(
    episode_dir: str | Path,
) -> ButtonPressEpisodeRelabelResult:
    """Recompute one button-press label from saved metric inputs."""

    path = Path(episode_dir)
    metadata = _load_metadata(path)
    task_config = _validate_button_press_metadata(metadata, path=path)
    raw_required_dwell = task_config.get("success_dwell_steps")
    if (
        not isinstance(raw_required_dwell, int)
        or isinstance(raw_required_dwell, bool)
        or raw_required_dwell <= 0
    ):
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} task_config.success_dwell_steps "
            "must be a positive integer."
        )
    required_dwell = raw_required_dwell

    task_states = _load_task_states(
        path,
        required_width=BUTTON_REQUIRED_TASK_STATE_WIDTH,
    )
    metrics = task_states[:, :BUTTON_REQUIRED_TASK_STATE_WIDTH]
    _validate_button_metric_inputs(metrics, path=path)

    target_depths = metrics[:, BUTTON_TARGET_DEPTH_COLUMN]
    target_states = metrics[:, BUTTON_TARGET_PRESSED_COLUMN] > 0.5
    if not np.allclose(target_depths, target_depths[0], rtol=0.0, atol=1e-12):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} target_press_depth must stay constant."
        )
    if not np.all(target_states == target_states[0]):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} target_pressed_state must stay constant."
        )

    press_depths = metrics[:, BUTTON_PRESS_DEPTH_COLUMN]
    button_states = metrics[:, BUTTON_PRESSED_COLUMN] > 0.5
    derived_button_states = press_depths >= target_depths
    if not np.all(button_states == derived_button_states):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has button_pressed values inconsistent "
            "with press_depth and target_press_depth."
        )
    goal_matches = (
        derived_button_states
        if bool(target_states[0])
        else ~derived_button_states
    )
    expected_dwell: list[int] = []
    dwell = 0
    for matches in goal_matches:
        dwell = dwell + 1 if bool(matches) else 0
        expected_dwell.append(dwell)
    saved_dwell = metrics[:, BUTTON_DWELL_COLUMN]
    if not np.array_equal(saved_dwell, np.asarray(expected_dwell, dtype=np.float64)):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has dwell_steps inconsistent with "
            "consecutive button goal frames."
        )

    qualifying = np.asarray(
        [
            is_button_press_success(
                press_depth_m=float(row[BUTTON_PRESS_DEPTH_COLUMN]),
                target_press_depth_m=float(row[BUTTON_TARGET_DEPTH_COLUMN]),
                button_pressed=bool(row[BUTTON_PRESSED_COLUMN] > 0.5),
                target_pressed_state=bool(row[BUTTON_TARGET_PRESSED_COLUMN] > 0.5),
                dwell_steps=int(round(row[BUTTON_DWELL_COLUMN])),
                required_dwell_steps=required_dwell,
            )
            for row in metrics
        ],
        dtype=bool,
    )
    first_success_frame = (
        int(np.flatnonzero(qualifying)[0]) if np.any(qualifying) else None
    )
    recomputed_success = first_success_frame is not None
    operator_success = _operator_success(metadata, path=path)
    resolved_button_id = task_config.get("resolved_button_id")
    if not isinstance(resolved_button_id, str) or not resolved_button_id:
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} task_config.resolved_button_id "
            "must be a non-empty string."
        )

    return ButtonPressEpisodeRelabelResult(
        episode_directory=path.name,
        episode_id=str(metadata.get("episode_id", path.name)),
        task_id=BUTTON_PRESS_TASK_ID,
        operator_success=operator_success,
        recomputed_success=recomputed_success,
        labels_agree=(
            None if operator_success is None else operator_success == recomputed_success
        ),
        frame_count=int(task_states.shape[0]),
        first_success_frame=first_success_frame,
        max_consecutive_goal_frames=int(np.max(metrics[:, BUTTON_DWELL_COLUMN])),
        resolved_button_id=resolved_button_id,
        target_press_depth_m=float(target_depths[0]),
        target_pressed_state=bool(target_states[0]),
    )


def relabel_button_press_dataset(
    dataset_dir: str | Path,
) -> ButtonPressRelabelReport:
    """Relabel every button-press episode immediately below a dataset directory."""

    dataset = Path(dataset_dir)
    episode_dirs = _episode_directories(dataset)
    episodes = tuple(
        relabel_button_press_episode(path) for path in episode_dirs
    )
    dwell_values = {
        int(
            _validate_button_press_metadata(_load_metadata(path), path=path)[
                "success_dwell_steps"
            ]
        )
        for path in episode_dirs
    }
    if len(dwell_values) != 1:
        raise SuccessRelabelError(
            "button_press episodes use inconsistent success_dwell_steps values."
        )
    return ButtonPressRelabelReport(
        version=BUTTON_PRESS_RELABEL_REPORT_VERSION,
        task_id=BUTTON_PRESS_TASK_ID,
        dataset=str(dataset),
        required_dwell_frames=dwell_values.pop(),
        episode_count=len(episodes),
        recomputed_success_count=sum(
            result.recomputed_success for result in episodes
        ),
        operator_success_count=sum(
            result.operator_success is True for result in episodes
        ),
        label_disagreement_count=sum(
            result.labels_agree is False for result in episodes
        ),
        raw_episodes_modified=False,
        episodes=episodes,
    )


def relabel_demo_dataset(
    dataset_dir: str | Path,
) -> RelabelReport | ButtonPressRelabelReport:
    """Dispatch one single-task dataset to its implemented relabeler."""

    dataset = Path(dataset_dir)
    episode_dirs = _episode_directories(dataset)
    task_id = _load_metadata(episode_dirs[0]).get("task_id")
    if task_id == REACH_TOUCH_TARGET_TASK_ID:
        return relabel_reach_touch_dataset(dataset)
    if task_id == BUTTON_PRESS_TASK_ID:
        return relabel_button_press_dataset(dataset)
    raise SuccessRelabelError(
        f"{episode_dirs[0] / 'metadata.json'} has unsupported task_id={task_id!r}; "
        f"expected {REACH_TOUCH_TARGET_TASK_ID!r} or {BUTTON_PRESS_TASK_ID!r}."
    )


def save_relabel_report(
    report: RelabelReport | ButtonPressRelabelReport,
    output_path: str | Path,
) -> Path:
    """Save a relabel report atomically without rewriting raw episodes."""

    path = Path(output_path)
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
        raise SuccessRelabelError(f"Dataset directory does not exist: {dataset}")
    if not dataset.is_dir():
        raise SuccessRelabelError(f"Dataset path is not a directory: {dataset}")
    if (dataset / "metadata.json").is_file():
        return (dataset,)
    episodes = tuple(
        path
        for path in sorted(dataset.iterdir())
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    if not episodes:
        raise SuccessRelabelError(
            f"No episode directories containing metadata.json were found in: {dataset}"
        )
    return episodes


def _load_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise SuccessRelabelError(
            f"Missing metadata.json for relabeling episode: {path}"
        )
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuccessRelabelError(
            f"Could not read valid JSON metadata from {metadata_path}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise SuccessRelabelError(
            f"{metadata_path} must contain a JSON object."
        )
    return loaded


def _validate_reach_touch_metadata(
    metadata: dict[str, Any],
    *,
    path: Path,
) -> None:
    task_id = metadata.get("task_id")
    if task_id != REACH_TOUCH_TARGET_TASK_ID:
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} has task_id={task_id!r}; only "
            f"{REACH_TOUCH_TARGET_TASK_ID!r} relabeling is supported."
        )
    task_config = metadata.get("task_config")
    if not isinstance(task_config, dict):
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} is missing the task_config object required "
            "for success relabeling."
        )
    metric_names = task_config.get("success_metric_inputs")
    if not isinstance(metric_names, list) or tuple(metric_names) != REQUIRED_METRIC_INPUTS:
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} is missing the required success_metric_inputs "
            f"{list(REQUIRED_METRIC_INPUTS)}."
        )


def _validate_button_press_metadata(
    metadata: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    task_id = metadata.get("task_id")
    if task_id != BUTTON_PRESS_TASK_ID:
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} has task_id={task_id!r}; only "
            f"{BUTTON_PRESS_TASK_ID!r} relabeling is supported."
        )
    task_config = metadata.get("task_config")
    if not isinstance(task_config, dict):
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} is missing the task_config object required "
            "for success relabeling."
        )
    metric_names = task_config.get("success_metric_inputs")
    if (
        not isinstance(metric_names, list)
        or tuple(metric_names) != BUTTON_REQUIRED_METRIC_INPUTS
    ):
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} is missing the required success_metric_inputs "
            f"{list(BUTTON_REQUIRED_METRIC_INPUTS)}."
        )
    return task_config


def _load_task_states(
    path: Path,
    *,
    required_width: int = REQUIRED_TASK_STATE_WIDTH,
) -> np.ndarray:
    task_state_path = path / "task_states.npy"
    if not task_state_path.is_file():
        raise SuccessRelabelError(
            f"Missing success metric inputs: {task_state_path} does not exist."
        )
    try:
        task_states = np.load(task_state_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SuccessRelabelError(
            f"Could not load success metric inputs from {task_state_path}: {exc}"
        ) from exc
    if task_states.ndim != 2:
        raise SuccessRelabelError(
            f"{task_state_path} must be a 2D array; got shape {task_states.shape}."
        )
    if task_states.shape[0] == 0:
        raise SuccessRelabelError(
            f"{task_state_path} contains no frames to relabel."
        )
    if task_states.shape[1] < required_width:
        raise SuccessRelabelError(
            f"{task_state_path} is missing success metric columns: expected at least "
            f"{required_width}, got {task_states.shape[1]}."
        )
    return np.asarray(task_states, dtype=np.float64)


def _validate_metric_inputs(metric_inputs: np.ndarray, *, path: Path) -> None:
    if not np.all(np.isfinite(metric_inputs)):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has non-finite success metric inputs."
        )
    if np.any(metric_inputs[:, DISTANCE_COLUMN] < 0.0):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has negative distance_to_target values."
        )
    contact_values = metric_inputs[:, PALM_CONTACT_COLUMN]
    if not np.all(np.isclose(contact_values, 0.0) | np.isclose(contact_values, 1.0)):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} palm_contact values must be binary 0 or 1."
        )


def _validate_button_metric_inputs(
    metric_inputs: np.ndarray,
    *,
    path: Path,
) -> None:
    if not np.all(np.isfinite(metric_inputs)):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has non-finite success metric inputs."
        )
    if np.any(metric_inputs[:, BUTTON_PRESS_DEPTH_COLUMN] < 0.0):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has negative press_depth values."
        )
    if np.any(metric_inputs[:, BUTTON_TARGET_DEPTH_COLUMN] <= 0.0):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} has non-positive target_press_depth values."
        )
    for column, name in (
        (BUTTON_PRESSED_COLUMN, "button_pressed"),
        (BUTTON_TARGET_PRESSED_COLUMN, "target_pressed_state"),
    ):
        values = metric_inputs[:, column]
        if not np.all(np.isclose(values, 0.0) | np.isclose(values, 1.0)):
            raise SuccessRelabelError(
                f"{path / 'task_states.npy'} {name} values must be binary 0 or 1."
            )
    dwell_values = metric_inputs[:, BUTTON_DWELL_COLUMN]
    if np.any(dwell_values < 0.0) or not np.allclose(
        dwell_values,
        np.round(dwell_values),
    ):
        raise SuccessRelabelError(
            f"{path / 'task_states.npy'} dwell_steps values must be "
            "non-negative integers."
        )


def _operator_success(metadata: dict[str, Any], *, path: Path) -> bool | None:
    value = metadata.get("success")
    if value is not None and not isinstance(value, bool):
        raise SuccessRelabelError(
            f"{path / 'metadata.json'} success must be true, false, or null."
        )
    return value


def _success_dwell(
    qualifying_frames: np.ndarray,
    *,
    required_dwell_frames: int,
) -> tuple[int | None, int]:
    dwell = 0
    max_dwell = 0
    first_success_frame: int | None = None
    for frame_index, qualifies in enumerate(qualifying_frames):
        dwell = dwell + 1 if bool(qualifies) else 0
        max_dwell = max(max_dwell, dwell)
        if dwell >= required_dwell_frames and first_success_frame is None:
            first_success_frame = frame_index
    return first_success_frame, max_dwell
