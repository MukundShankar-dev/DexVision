"""Goal-conditioned PyTorch datasets built from saved Level 2 episodes."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from dexvision.logging.dataset_schema import DemoSchemaError, extract_observations
from dexvision.logging.replay_demo import DemoReplayError, load_replay_demo
from dexvision.learning.splits import (
    EpisodeDescriptor,
    SplitConfig,
    SplitManifest,
    deterministic_episode_split,
    save_split_manifest,
    split_config_from_mapping,
)


DEFAULT_OBSERVATION_FIELDS = (
    "robot_qpos",
    "robot_qvel",
    "base_position",
    "base_orientation",
    "finger_joint_positions",
    "finger_joint_velocities",
    "object_state",
    "tracking_quality",
)
_EPISODE_DIGEST_SUFFIXES = {".json", ".npy"}
_PUSH_GOAL_IDS = {
    "push_target_left": "push_cube_left_lane",
    "push_target_center": "push_cube_center_lane",
    "push_target_right": "push_cube_right_lane",
}
_APPROACH_SIDES = ("left", "front", "right")


class LearningDatasetError(RuntimeError):
    """Raised when saved demonstrations cannot form a learning dataset."""


@dataclass(frozen=True)
class EpisodeData:
    """Validated, vectorized data for one complete demonstration episode."""

    episode_id: str
    goal_id: str
    recording_session_id: str | None
    action_schema_version: str
    observation_schema_version: str
    data_digest: str
    observations: np.ndarray
    goal: np.ndarray
    actions: np.ndarray
    observation_names: tuple[str, ...]
    goal_names: tuple[str, ...]
    action_names: tuple[str, ...]
    timestamps: np.ndarray
    tracking_quality: np.ndarray
    quality_passed: bool
    recomputed_success: bool

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])

    def descriptor(self) -> EpisodeDescriptor:
        return EpisodeDescriptor(
            episode_id=self.episode_id,
            goal_id=self.goal_id,
            data_digest=self.data_digest,
            action_schema_version=self.action_schema_version,
            observation_schema_version=self.observation_schema_version,
            recording_session_id=self.recording_session_id,
        )


@dataclass(frozen=True)
class VectorStats:
    """Per-column population statistics fitted on training timesteps only."""

    mean: np.ndarray
    std: np.ndarray
    count: int
    names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


@dataclass(frozen=True)
class NormalizationStats:
    """Observation, goal, and action statistics from the training partition."""

    source_split: str
    dataset_digest: str
    observation: VectorStats
    goal: VectorStats
    action: VectorStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_split": self.source_split,
            "dataset_digest": self.dataset_digest,
            "observation": self.observation.to_dict(),
            "goal": self.goal.to_dict(),
            "action": self.action.to_dict(),
        }


class GoalConditionedSkillDataset(Dataset[dict[str, Any]]):
    """Frame-level samples whose split membership is fixed by whole episode."""

    def __init__(
        self,
        episodes: Sequence[EpisodeData],
        *,
        normalization: NormalizationStats | None = None,
    ) -> None:
        self.episodes = tuple(episodes)
        self.normalization = normalization
        self._ends: list[int] = []
        total = 0
        for episode in self.episodes:
            total += episode.num_steps
            self._ends.append(total)
        if self.episodes:
            _validate_vector_layouts(self.episodes)
            if normalization is not None:
                _validate_normalization_layout(self.episodes[0], normalization)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._ends, index)
        start = 0 if episode_index == 0 else self._ends[episode_index - 1]
        timestep = index - start
        episode = self.episodes[episode_index]

        observation = episode.observations[timestep]
        goal = episode.goal
        action = episode.actions[timestep]
        if self.normalization is not None:
            observation = _normalize(observation, self.normalization.observation)
            goal = _normalize(goal, self.normalization.goal)
            action = _normalize(action, self.normalization.action)
        return {
            "obs": torch.as_tensor(observation.copy(), dtype=torch.float32),
            "goal": torch.as_tensor(goal.copy(), dtype=torch.float32),
            "action": torch.as_tensor(action.copy(), dtype=torch.float32),
            "episode_id": episode.episode_id,
            "demo_id": episode.episode_id,
            "goal_id": episode.goal_id,
            "timestep": timestep,
            "timestamp": float(episode.timestamps[timestep]),
            "tracking_quality": torch.as_tensor(
                episode.tracking_quality[timestep].copy(), dtype=torch.float32
            ),
            "quality_passed": episode.quality_passed,
            "recomputed_success": episode.recomputed_success,
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Whole-episode partitions sharing one training-only normalization."""

    train: GoalConditionedSkillDataset
    validation: GoalConditionedSkillDataset
    test: GoalConditionedSkillDataset
    manifest: SplitManifest
    normalization: NormalizationStats

    def save_manifest(self, output_path: str | Path) -> None:
        save_split_manifest(
            self.manifest,
            output_path,
            normalization=self.normalization.to_dict(),
        )


def load_skill_episodes(
    dataset_root: str | Path,
    *,
    skill_name: str,
    observation_fields: Sequence[str] = DEFAULT_OBSERVATION_FIELDS,
    include_previous_action: bool = False,
    require_clean: bool = True,
) -> tuple[EpisodeData, ...]:
    """Load one skill directly from an extracted Level 2 release.

    ``dataset_root`` may be the repository/extraction root, ``data/demos``,
    ``data/demos/raw``, or the task directory itself. Clean loading requires
    both the saved quality and relabel reports and selects only episodes that
    passed quality filtering and recomputed as successful.
    """

    if not isinstance(skill_name, str) or not skill_name:
        raise LearningDatasetError("skill_name must be a non-empty string.")
    requested_fields = tuple(observation_fields)
    if not requested_fields or any(not field for field in requested_fields):
        raise LearningDatasetError("observation_fields must contain non-empty names.")
    if len(set(requested_fields)) != len(requested_fields):
        raise LearningDatasetError("observation_fields must not contain duplicates.")

    skill_dir = _resolve_skill_dir(Path(dataset_root), skill_name)
    labels = _load_clean_labels(skill_dir) if require_clean else {}
    episode_dirs = tuple(
        path
        for path in sorted(skill_dir.iterdir())
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    if not episode_dirs:
        raise LearningDatasetError(f"no episode directories found in {skill_dir}.")

    loaded: list[EpisodeData] = []
    for episode_dir in episode_dirs:
        label = labels.get(episode_dir.name)
        if require_clean:
            if label is None:
                raise LearningDatasetError(
                    f"clean-label reports do not cover episode directory {episode_dir.name!r}."
                )
            quality_passed, recomputed_success = label
            if not quality_passed or not recomputed_success:
                continue
        else:
            quality_passed, recomputed_success = label or (False, False)
        loaded.append(
            _load_episode(
                episode_dir,
                skill_name=skill_name,
                observation_fields=requested_fields,
                include_previous_action=include_previous_action,
                quality_passed=quality_passed,
                recomputed_success=recomputed_success,
            )
        )
    if not loaded:
        raise LearningDatasetError(f"no eligible episodes found for skill {skill_name!r}.")
    _validate_vector_layouts(tuple(loaded))
    return tuple(loaded)


def build_skill_datasets(
    dataset_root: str | Path,
    *,
    skill_name: str,
    split_config: SplitConfig,
    observation_fields: Sequence[str] = DEFAULT_OBSERVATION_FIELDS,
    include_previous_action: bool = False,
    require_clean: bool = True,
    normalize: bool = True,
) -> DatasetBundle:
    """Load, split, fit training-only statistics, and construct datasets."""

    episodes = load_skill_episodes(
        dataset_root,
        skill_name=skill_name,
        observation_fields=observation_fields,
        include_previous_action=include_previous_action,
        require_clean=require_clean,
    )
    manifest = deterministic_episode_split(
        (episode.descriptor() for episode in episodes), split_config
    )
    normalization = fit_training_normalization(episodes, manifest)
    by_split = _episodes_by_split(episodes, manifest)
    applied_stats = normalization if normalize else None
    return DatasetBundle(
        train=GoalConditionedSkillDataset(by_split["train"], normalization=applied_stats),
        validation=GoalConditionedSkillDataset(
            by_split["validation"], normalization=applied_stats
        ),
        test=GoalConditionedSkillDataset(by_split["test"], normalization=applied_stats),
        manifest=manifest,
        normalization=normalization,
    )


def load_frozen_reach_datasets(
    dataset_root: str | Path,
    *,
    evaluation_config_path: str | Path = "configs/level3_evaluation.yaml",
    observation_fields: Sequence[str] = DEFAULT_OBSERVATION_FIELDS,
    include_previous_action: bool = False,
    normalize: bool = True,
) -> DatasetBundle:
    """Build the exact ``level3/reach-evaluation-v1`` offline split."""

    return load_frozen_skill_datasets(
        dataset_root,
        evaluation_config_path=evaluation_config_path,
        expected_version="level3/reach-evaluation-v1",
        expected_skill_name="reach_touch_target",
        observation_fields=observation_fields,
        include_previous_action=include_previous_action,
        normalize=normalize,
    )


def load_frozen_skill_datasets(
    dataset_root: str | Path,
    *,
    evaluation_config_path: str | Path,
    expected_version: str,
    expected_skill_name: str,
    observation_fields: Sequence[str] = DEFAULT_OBSERVATION_FIELDS,
    include_previous_action: bool = False,
    normalize: bool = True,
) -> DatasetBundle:
    """Build one task's exact frozen Level 3 offline split.

    The evaluator configuration is also the allow-list for demonstration goal
    ids.  Reserved rollout-only goals therefore cannot enter training merely
    because an episode was placed in the extracted dataset directory.
    """

    config_path = Path(evaluation_config_path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LearningDatasetError(
            f"cannot read Level 3 evaluation config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LearningDatasetError(f"{config_path} must contain a YAML mapping.")
    version = payload.get("version")
    if version != expected_version:
        raise LearningDatasetError(
            f"expected frozen split version {expected_version!r}, got {version!r}."
        )
    if payload.get("task_id") != expected_skill_name or payload.get(
        "skill_name"
    ) != expected_skill_name:
        raise LearningDatasetError(
            f"frozen config has an incompatible task_id/skill_name for "
            f"{expected_skill_name!r}."
        )
    offline = payload.get("offline_split")
    if not isinstance(offline, Mapping):
        raise LearningDatasetError("frozen config is missing offline_split.")
    if offline.get("group_by_episode") is not True:
        raise LearningDatasetError("frozen split must group by complete episode.")
    if offline.get("normalization_source") != "train_only":
        raise LearningDatasetError("frozen normalization_source must be train_only.")
    training_goals = payload.get("training_goals") or payload.get("training_targets")
    if not isinstance(training_goals, Mapping) or not training_goals:
        raise LearningDatasetError("frozen config must declare non-empty training goals.")
    split_config = split_config_from_mapping(offline, version=str(version))
    bundle = build_skill_datasets(
        dataset_root,
        skill_name=expected_skill_name,
        split_config=split_config,
        observation_fields=observation_fields,
        include_previous_action=include_previous_action,
        require_clean=True,
        normalize=normalize,
    )
    observed_goals = {
        episode.goal_id
        for dataset in (bundle.train, bundle.validation, bundle.test)
        for episode in dataset.episodes
    }
    configured_goals = set(training_goals)
    unexpected = sorted(observed_goals - configured_goals)
    missing = sorted(configured_goals - observed_goals)
    if unexpected or missing:
        raise LearningDatasetError(
            "saved demonstration goals do not match the frozen training-goal "
            f"declaration; unexpected={unexpected}, missing={missing}."
        )
    return bundle


def fit_training_normalization(
    episodes: Sequence[EpisodeData],
    manifest: SplitManifest,
) -> NormalizationStats:
    """Fit reproducible population statistics using only training episodes."""

    by_split = _episodes_by_split(episodes, manifest)
    training = by_split["train"]
    if not training:
        raise LearningDatasetError("cannot fit normalization without training episodes.")
    _validate_vector_layouts(training)
    observation_values = np.concatenate(
        [episode.observations for episode in training], axis=0
    )
    goal_values = np.concatenate(
        [np.repeat(episode.goal[None, :], episode.num_steps, axis=0) for episode in training],
        axis=0,
    )
    action_values = np.concatenate([episode.actions for episode in training], axis=0)
    reference = training[0]
    return NormalizationStats(
        source_split="train",
        dataset_digest=manifest.dataset_digest,
        observation=_fit_stats(observation_values, reference.observation_names),
        goal=_fit_stats(goal_values, reference.goal_names),
        action=_fit_stats(action_values, reference.action_names),
    )


def quaternion_wxyz_to_rotation_6d(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized wxyz quaternions to the first two rotation columns."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise LearningDatasetError(
            f"base orientation must have shape [T, 4], got {values.shape}."
        )
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(values)) or np.any(norms <= 1e-12):
        raise LearningDatasetError("base orientation contains invalid quaternions.")
    w, x, y, z = (values / norms[:, None]).T
    matrices = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices[:, :, :2].transpose(0, 2, 1).reshape(values.shape[0], 6)


def _load_episode(
    episode_dir: Path,
    *,
    skill_name: str,
    observation_fields: tuple[str, ...],
    include_previous_action: bool,
    quality_passed: bool,
    recomputed_success: bool,
) -> EpisodeData:
    try:
        loaded = load_replay_demo(episode_dir)
        extracted = extract_observations(
            loaded.episode, observation_schema=loaded.observation_schema
        )
    except (DemoReplayError, DemoSchemaError) as exc:
        raise LearningDatasetError(f"invalid learning episode {episode_dir}: {exc}") from exc
    metadata = loaded.episode.metadata
    if metadata.get("skill_name") != skill_name or metadata.get("task_id") != skill_name:
        raise LearningDatasetError(
            f"{episode_dir} belongs to skill/task {metadata.get('skill_name')!r}/"
            f"{metadata.get('task_id')!r}, expected {skill_name!r}."
        )
    if not loaded.observation_schema.executable:
        raise LearningDatasetError(
            f"{episode_dir} uses non-executable observation schema "
            f"{loaded.observation_schema.version!r}."
        )

    observations, observation_names = _vectorize_observations(
        extracted,
        loaded.observation_schema.layouts,
        observation_fields,
        task_config=metadata.get("task_config"),
    )
    split_actions = loaded.action_schema.split(loaded.episode.actions)
    actions = np.concatenate(
        [
            split_actions["base_position_target"],
            split_actions["base_orientation_target"],
            split_actions["finger_actuator_targets"],
        ],
        axis=1,
    ).astype(np.float64, copy=False)
    action_names = _action_names(loaded.finger_target_names)
    if include_previous_action:
        previous = np.zeros_like(actions)
        previous[1:] = actions[:-1]
        observations = np.concatenate((observations, previous), axis=1)
        observation_names += tuple(f"previous_action/{name}" for name in action_names)

    goal_id, goal, goal_names = _goal_from_metadata(metadata, path=episode_dir)
    session_id = metadata.get("recording_session_id")
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        raise LearningDatasetError(
            f"{episode_dir / 'metadata.json'} recording_session_id must be a "
            "non-empty string when present."
        )
    return EpisodeData(
        episode_id=str(metadata["episode_id"]),
        goal_id=goal_id,
        recording_session_id=session_id,
        action_schema_version=loaded.action_schema.version,
        observation_schema_version=loaded.observation_schema.version,
        data_digest=_episode_digest(episode_dir),
        observations=observations,
        goal=goal,
        actions=actions,
        observation_names=observation_names,
        goal_names=goal_names,
        action_names=action_names,
        timestamps=np.asarray(loaded.episode.timestamps, dtype=np.float64),
        tracking_quality=np.asarray(loaded.episode.tracking_quality, dtype=np.float64),
        quality_passed=quality_passed,
        recomputed_success=recomputed_success,
    )


def _vectorize_observations(
    extracted: Mapping[str, np.ndarray | None],
    layouts: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    task_config: object,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not isinstance(task_config, Mapping):
        raise LearningDatasetError("metadata task_config must be a mapping.")
    required_objects = task_config.get("required_objects", ())
    if not isinstance(required_objects, Sequence) or isinstance(required_objects, str):
        raise LearningDatasetError("task_config.required_objects must be a sequence.")
    object_names = tuple(
        name for name in required_objects if isinstance(name, str) and name
    )
    if len(object_names) != len(required_objects):
        raise LearningDatasetError(
            "task_config.required_objects must contain non-empty strings."
        )
    vectors: list[np.ndarray] = []
    names: list[str] = []
    for field_name in fields:
        if field_name not in extracted:
            raise LearningDatasetError(
                f"observation schema is missing requested field {field_name!r}."
            )
        value = extracted[field_name]
        if value is None:
            raise LearningDatasetError(
                f"requested observation field {field_name!r} is absent for this skill."
            )
        flattened = np.asarray(value, dtype=np.float64).reshape(value.shape[0], -1)
        if field_name == "base_orientation":
            flattened = quaternion_wxyz_to_rotation_6d(flattened)
            field_names = (
                "rotation_col0/x",
                "rotation_col0/y",
                "rotation_col0/z",
                "rotation_col1/x",
                "rotation_col1/y",
                "rotation_col1/z",
            )
        else:
            declared_names = tuple(layouts[field_name].names)
            field_names = declared_names or tuple(
                f"value[{index}]" for index in range(flattened.shape[1])
            )
            selected_indices = _task_relevant_named_indices(
                field_name,
                field_names,
                object_names=object_names,
            )
            if selected_indices is not None:
                flattened = flattened[:, selected_indices]
                field_names = tuple(field_names[index] for index in selected_indices)
        vectors.append(flattened)
        names.extend(f"{field_name}/{name}" for name in field_names)
    return np.concatenate(vectors, axis=1), tuple(names)


def _task_relevant_named_indices(
    field_name: str,
    names: tuple[str, ...],
    *,
    object_names: tuple[str, ...],
) -> tuple[int, ...] | None:
    """Select named robot/task columns without relying on packed offsets."""

    if field_name in {"finger_joint_positions", "finger_joint_velocities"}:
        selected = tuple(index for index, name in enumerate(names) if name.startswith("rh_"))
        if not selected:
            raise LearningDatasetError(
                f"observation field {field_name!r} has no named robot-hand joints."
            )
        return selected
    if field_name in {"robot_qpos", "robot_qvel"}:
        selected = tuple(
            index
            for index, name in enumerate(names)
            if name.startswith("rh_")
            or any(
                name == object_name or name.startswith(f"{object_name}_")
                for object_name in object_names
            )
        )
        if not selected:
            raise LearningDatasetError(
                f"observation field {field_name!r} has no named task-relevant state."
            )
        return selected
    return None


def _goal_from_metadata(
    metadata: Mapping[str, Any], *, path: Path
) -> tuple[str, np.ndarray, tuple[str, ...]]:
    task_id = metadata.get("task_id")
    config = metadata.get("task_config")
    if not isinstance(config, Mapping):
        raise LearningDatasetError(f"{path / 'metadata.json'} is missing task_config.")
    if task_id == "reach_touch_target":
        target = _finite_vector(config, "target_position", 3, path=path)
        goal_id = _required_string(config, "resolved_target_source", path=path)
        return goal_id, target, ("target_position/x", "target_position/y", "target_position/z")
    if task_id == "button_press":
        button_id = _required_string(config, "resolved_button_id", path=path)
        button_index = _finite_scalar(config, "button_index", path=path)
        position = _finite_vector(config, "button_position", 3, path=path)
        depth = _finite_scalar(config, "target_press_depth", path=path)
        pressed = config.get("target_pressed_state")
        if not isinstance(pressed, bool):
            raise LearningDatasetError(
                f"{path / 'metadata.json'} task_config.target_pressed_state must be boolean."
            )
        approach_raw = config.get("approach_pose")
        approach_present = approach_raw is not None
        approach = (
            np.zeros(3, dtype=np.float64)
            if approach_raw is None
            else _finite_array(approach_raw, "approach_pose", 3, path=path)
        )
        goal = np.concatenate(
            (
                np.asarray([button_index], dtype=np.float64),
                position,
                np.asarray([depth, float(pressed), float(approach_present)]),
                approach,
            )
        )
        goal_id = f"{button_id}_depth_{int(round(depth * 1000.0)):03d}"
        names = (
            "button_index",
            "button_position/x",
            "button_position/y",
            "button_position/z",
            "target_press_depth",
            "target_pressed_state",
            "approach_pose_present",
            "approach_pose/x",
            "approach_pose/y",
            "approach_pose/z",
        )
        return goal_id, goal, names
    if task_id == "push_cube_to_target":
        target_source = _required_string(config, "resolved_target_source", path=path)
        object_index = _finite_scalar(config, "object_index", path=path)
        target_index = _finite_scalar(config, "target_index", path=path)
        target = _finite_vector(config, "target_position", 3, path=path)
        radius = _finite_scalar(config, "target_radius", path=path)
        initial = _finite_vector(config, "initial_object_position", 3, path=path)
        side = _required_string(config, "resolved_approach_side", path=path)
        if side not in _APPROACH_SIDES:
            raise LearningDatasetError(
                f"{path / 'metadata.json'} has unsupported approach side {side!r}."
            )
        side_one_hot = np.asarray(
            [float(side == value) for value in _APPROACH_SIDES], dtype=np.float64
        )
        goal = np.concatenate(
            (
                np.asarray([object_index, target_index]),
                target,
                np.asarray([radius]),
                initial,
                side_one_hot,
            )
        )
        names = (
            "object_index",
            "target_index",
            "target_position/x",
            "target_position/y",
            "target_position/z",
            "target_radius",
            "initial_object_position/x",
            "initial_object_position/y",
            "initial_object_position/z",
            "approach_side/left",
            "approach_side/front",
            "approach_side/right",
        )
        return _PUSH_GOAL_IDS.get(target_source, target_source), goal, names
    raise LearningDatasetError(
        f"{path / 'metadata.json'} has unsupported Level 3 task_id {task_id!r}."
    )


def _resolve_skill_dir(root: Path, skill_name: str) -> Path:
    candidates = (
        root,
        root / skill_name,
        root / "raw" / skill_name,
        root / "data" / "demos" / "raw" / skill_name,
    )
    for candidate in candidates:
        if candidate.name == skill_name and candidate.is_dir():
            return candidate
    raise LearningDatasetError(
        f"could not find extracted skill directory {skill_name!r} below {root}."
    )


def _load_clean_labels(skill_dir: Path) -> dict[str, tuple[bool, bool]]:
    quality = _load_report_entries(skill_dir / "quality_report.json", "quality")
    relabel = _load_report_entries(skill_dir / "relabel_report.json", "relabel")
    if set(quality) != set(relabel):
        missing_quality = sorted(set(relabel) - set(quality))
        missing_relabel = sorted(set(quality) - set(relabel))
        raise LearningDatasetError(
            "quality/relabel report coverage differs; "
            f"missing quality={missing_quality}, missing relabel={missing_relabel}."
        )
    labels: dict[str, tuple[bool, bool]] = {}
    for directory in quality:
        quality_value = quality[directory].get("passed")
        recomputed_value = relabel[directory].get("recomputed_success")
        if not isinstance(quality_value, bool) or not isinstance(recomputed_value, bool):
            raise LearningDatasetError(
                f"reports for {directory!r} need boolean passed/recomputed_success labels."
            )
        labels[directory] = (quality_value, recomputed_value)
    return labels


def _load_report_entries(path: Path, label: str) -> dict[str, Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearningDatasetError(f"cannot read {label} report {path}: {exc}") from exc
    entries = payload.get("episodes") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        raise LearningDatasetError(f"{path} must contain an episodes list.")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise LearningDatasetError(f"{path} contains a non-object episode entry.")
        directory = entry.get("episode_directory")
        if not isinstance(directory, str) or not directory:
            raise LearningDatasetError(f"{path} entry is missing episode_directory.")
        if directory in result:
            raise LearningDatasetError(f"{path} repeats episode_directory {directory!r}.")
        result[directory] = entry
    return result


def _episode_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix in _EPISODE_DIGEST_SUFFIXES
    )
    if not files:
        raise LearningDatasetError(f"cannot digest empty episode directory {path}.")
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _action_names(finger_names: tuple[str, ...]) -> tuple[str, ...]:
    return (
        "base_position_target/x",
        "base_position_target/y",
        "base_position_target/z",
        "base_orientation_target/qw",
        "base_orientation_target/qx",
        "base_orientation_target/qy",
        "base_orientation_target/qz",
        *(f"finger_actuator_targets/{name}" for name in finger_names),
    )


def _episodes_by_split(
    episodes: Sequence[EpisodeData], manifest: SplitManifest
) -> dict[str, tuple[EpisodeData, ...]]:
    assignments = manifest.assignment_by_episode()
    episode_ids = {episode.episode_id for episode in episodes}
    if set(assignments) != episode_ids:
        raise LearningDatasetError(
            "split manifest episode ids do not match the loaded episode collection."
        )
    result: dict[str, list[EpisodeData]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for episode in episodes:
        split_name = assignments[episode.episode_id]
        if split_name not in result:
            raise LearningDatasetError(f"unknown split assignment {split_name!r}.")
        result[split_name].append(episode)
    return {name: tuple(values) for name, values in result.items()}


def _fit_stats(values: np.ndarray, names: tuple[str, ...]) -> VectorStats:
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(names):
        raise LearningDatasetError("cannot fit normalization for an invalid vector layout.")
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    return VectorStats(mean=mean, std=std, count=int(values.shape[0]), names=names)


def _normalize(values: np.ndarray, stats: VectorStats) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - stats.mean) / stats.std


def _validate_vector_layouts(episodes: Sequence[EpisodeData]) -> None:
    if not episodes:
        return
    reference = episodes[0]
    for episode in episodes[1:]:
        if episode.observation_names != reference.observation_names:
            raise LearningDatasetError(
                f"observation schema/layout mismatch in episode {episode.episode_id!r}."
            )
        if episode.goal_names != reference.goal_names:
            raise LearningDatasetError(
                f"goal schema/layout mismatch in episode {episode.episode_id!r}."
            )
        if episode.action_names != reference.action_names:
            raise LearningDatasetError(
                f"action schema/layout mismatch in episode {episode.episode_id!r}."
            )
        if episode.action_schema_version != reference.action_schema_version:
            raise LearningDatasetError("mixed action schema versions are not supported.")
        if episode.observation_schema_version != reference.observation_schema_version:
            raise LearningDatasetError("mixed observation schema versions are not supported.")


def _validate_normalization_layout(
    episode: EpisodeData, normalization: NormalizationStats
) -> None:
    expected = (
        (episode.observation_names, normalization.observation.names, "observation"),
        (episode.goal_names, normalization.goal.names, "goal"),
        (episode.action_names, normalization.action.names, "action"),
    )
    for actual, saved, label in expected:
        if actual != saved:
            raise LearningDatasetError(f"{label} normalization layout is incompatible.")


def _finite_vector(
    payload: Mapping[str, Any], key: str, length: int, *, path: Path
) -> np.ndarray:
    if key not in payload:
        raise LearningDatasetError(f"{path / 'metadata.json'} is missing task_config.{key}.")
    return _finite_array(payload[key], key, length, path=path)


def _finite_array(value: object, name: str, length: int, *, path: Path) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LearningDatasetError(
            f"{path / 'metadata.json'} task_config.{name} must be numeric."
        ) from exc
    if array.shape != (length,) or np.any(~np.isfinite(array)):
        raise LearningDatasetError(
            f"{path / 'metadata.json'} task_config.{name} must be a finite "
            f"length-{length} vector."
        )
    return array


def _finite_scalar(payload: Mapping[str, Any], key: str, *, path: Path) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise LearningDatasetError(
            f"{path / 'metadata.json'} task_config.{key} must be numeric."
        )
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise LearningDatasetError(
            f"{path / 'metadata.json'} task_config.{key} must be numeric."
        ) from exc
    if not np.isfinite(scalar):
        raise LearningDatasetError(
            f"{path / 'metadata.json'} task_config.{key} must be finite."
        )
    return scalar


def _required_string(payload: Mapping[str, Any], key: str, *, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise LearningDatasetError(
            f"{path / 'metadata.json'} task_config.{key} must be a non-empty string."
        )
    return value
