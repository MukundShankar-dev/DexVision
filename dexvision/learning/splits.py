"""Deterministic whole-episode splits for Level 3 learning datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SplitError(ValueError):
    """Raised when a requested dataset split is invalid or unsafe."""


@dataclass(frozen=True)
class EpisodeDescriptor:
    """Metadata needed to assign one complete episode to a partition."""

    episode_id: str
    goal_id: str
    data_digest: str
    action_schema_version: str
    observation_schema_version: str
    recording_session_id: str | None = None


@dataclass(frozen=True)
class SplitConfig:
    """Versioned split parameters.

    Fractions describe episode-level offline partitions. Session grouping takes
    precedence when a genuine ``recording_session_id`` is present.
    """

    version: str
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float = 0.0
    strategy: str = "stratified_episode_hash"

    def validate(self) -> None:
        if not self.version:
            raise SplitError("split version must be a non-empty string.")
        if self.strategy != "stratified_episode_hash":
            raise SplitError(
                "unsupported split strategy "
                f"{self.strategy!r}; expected 'stratified_episode_hash'."
            )
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise SplitError("split fractions must be finite and non-negative.")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise SplitError("train, validation, and test fractions must sum to 1.0.")
        if self.train_fraction <= 0.0:
            raise SplitError("train_fraction must be greater than zero.")


@dataclass(frozen=True)
class SplitAssignment:
    """Saved assignment and provenance for one episode."""

    episode_id: str
    goal_id: str
    split: str
    data_digest: str
    action_schema_version: str
    observation_schema_version: str
    recording_session_id: str | None


@dataclass(frozen=True)
class SplitManifest:
    """Reproducible split result for a single skill dataset."""

    version: str
    seed: int
    strategy: str
    dataset_digest: str
    assignments: tuple[SplitAssignment, ...]
    action_schema_versions: tuple[str, ...]
    observation_schema_versions: tuple[str, ...]
    recording_session_ids_available: bool

    def assignment_by_episode(self) -> dict[str, str]:
        return {item.episode_id: item.split for item in self.assignments}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "strategy": self.strategy,
            "dataset_digest": self.dataset_digest,
            "action_schema_versions": list(self.action_schema_versions),
            "observation_schema_versions": list(self.observation_schema_versions),
            "recording_session_ids_available": self.recording_session_ids_available,
            "assignments": [asdict(item) for item in self.assignments],
        }


def deterministic_episode_split(
    episodes: Iterable[EpisodeDescriptor],
    config: SplitConfig,
) -> SplitManifest:
    """Assign complete episodes deterministically, stratified by goal.

    The frozen reach protocol orders each goal group by the SHA-256 digest of
    ``"<seed>:<episode_id>"``. When session ids exist, a session is treated as
    one indivisible unit. Sessions spanning more than one goal are assigned by
    their session hash so no session can leak across partitions.
    """

    config.validate()
    descriptors = tuple(episodes)
    if not descriptors:
        raise SplitError("cannot split an empty episode collection.")
    _validate_descriptors(descriptors)

    session_groups: dict[str, list[EpisodeDescriptor]] = defaultdict(list)
    ungrouped_by_goal: dict[str, list[EpisodeDescriptor]] = defaultdict(list)
    for episode in descriptors:
        if episode.recording_session_id is None:
            ungrouped_by_goal[episode.goal_id].append(episode)
        else:
            session_groups[episode.recording_session_id].append(episode)

    single_goal_sessions: dict[str, list[tuple[str, tuple[EpisodeDescriptor, ...]]]] = (
        defaultdict(list)
    )
    mixed_goal_sessions: list[tuple[str, tuple[EpisodeDescriptor, ...]]] = []
    for session_id, members in session_groups.items():
        unit = (f"session:{session_id}", tuple(members))
        goals = {member.goal_id for member in members}
        if len(goals) == 1:
            single_goal_sessions[next(iter(goals))].append(unit)
        else:
            mixed_goal_sessions.append(unit)

    assignments: dict[str, str] = {}
    all_goals = set(ungrouped_by_goal) | set(single_goal_sessions)
    for goal_id in sorted(all_goals):
        units: list[tuple[str, tuple[EpisodeDescriptor, ...]]] = [
            (episode.episode_id, (episode,))
            for episode in ungrouped_by_goal.get(goal_id, ())
        ]
        units.extend(single_goal_sessions.get(goal_id, ()))
        units.sort(key=lambda unit: (_hash_key(config.seed, unit[0]), unit[0]))
        partition_names = _partition_names(len(units), config)
        for (_unit_id, members), split_name in zip(units, partition_names, strict=True):
            for member in members:
                assignments[member.episode_id] = split_name

    # A genuine session can cover multiple goals. Hash the indivisible session
    # unit directly; exact per-goal counts are impossible without leakage.
    for unit_id, members in sorted(
        mixed_goal_sessions,
        key=lambda unit: (_hash_key(config.seed, unit[0]), unit[0]),
    ):
        split_name = _hash_partition(unit_id, config)
        for member in members:
            assignments[member.episode_id] = split_name

    if set(assignments) != {episode.episode_id for episode in descriptors}:
        raise SplitError("internal split error: not every episode received an assignment.")
    _validate_session_isolation(descriptors, assignments)

    saved_assignments = tuple(
        SplitAssignment(
            episode_id=episode.episode_id,
            goal_id=episode.goal_id,
            split=assignments[episode.episode_id],
            data_digest=episode.data_digest,
            action_schema_version=episode.action_schema_version,
            observation_schema_version=episode.observation_schema_version,
            recording_session_id=episode.recording_session_id,
        )
        for episode in sorted(descriptors, key=lambda item: item.episode_id)
    )
    return SplitManifest(
        version=config.version,
        seed=config.seed,
        strategy=config.strategy,
        dataset_digest=_dataset_digest(saved_assignments),
        assignments=saved_assignments,
        action_schema_versions=tuple(
            sorted({episode.action_schema_version for episode in descriptors})
        ),
        observation_schema_versions=tuple(
            sorted({episode.observation_schema_version for episode in descriptors})
        ),
        recording_session_ids_available=any(
            episode.recording_session_id is not None for episode in descriptors
        ),
    )


def split_config_from_mapping(payload: Mapping[str, Any], *, version: str) -> SplitConfig:
    """Build a validated split config from an evaluation-config mapping."""

    try:
        config = SplitConfig(
            version=version,
            seed=int(payload["seed"]),
            strategy=str(payload["strategy"]),
            train_fraction=float(payload["train_fraction"]),
            validation_fraction=float(payload["validation_fraction"]),
            test_fraction=float(payload.get("test_fraction", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitError(f"invalid split configuration: {exc}") from exc
    config.validate()
    return config


def save_split_manifest(
    manifest: SplitManifest,
    output_path: str | Path,
    *,
    normalization: Mapping[str, Any] | None = None,
) -> None:
    """Save assignments and optional training-only normalization metadata."""

    payload = manifest.to_dict()
    if normalization is not None:
        payload["normalization"] = dict(normalization)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _partition_names(count: int, config: SplitConfig) -> tuple[str, ...]:
    if count == 0:
        return ()
    if config.validation_fraction > 0.0 and count < 2:
        raise SplitError(
            "each goal needs at least two independent episode/session groups "
            "when validation_fraction is non-zero."
        )

    train_count = int(math.floor(count * config.train_fraction))
    if config.test_fraction == 0.0:
        # The frozen reach protocol assigns 80% to training and explicitly
        # assigns the entire remainder to validation.
        test_count = 0
        validation_count = count - train_count
    elif config.validation_fraction == 0.0:
        validation_count = 0
        test_count = count - train_count
    else:
        validation_count = int(math.floor(count * config.validation_fraction))
        test_count = count - train_count - validation_count

    if config.validation_fraction > 0.0 and validation_count == 0:
        validation_count = 1
    if config.test_fraction > 0.0 and test_count == 0:
        test_count = 1
    train_count = count - validation_count - test_count
    if train_count <= 0:
        raise SplitError("split fractions leave no training groups for a goal.")
    return (
        ("train",) * train_count
        + ("validation",) * validation_count
        + ("test",) * test_count
    )


def _hash_partition(unit_id: str, config: SplitConfig) -> str:
    digest = _hash_key(config.seed, unit_id)
    value = int(digest[:16], 16) / float(16**16)
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def _hash_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def _validate_descriptors(episodes: tuple[EpisodeDescriptor, ...]) -> None:
    ids: set[str] = set()
    for episode in episodes:
        for field_name in (
            "episode_id",
            "goal_id",
            "data_digest",
            "action_schema_version",
            "observation_schema_version",
        ):
            value = getattr(episode, field_name)
            if not isinstance(value, str) or not value:
                raise SplitError(f"episode {field_name} must be a non-empty string.")
        if episode.episode_id in ids:
            raise SplitError(f"duplicate episode_id: {episode.episode_id!r}.")
        ids.add(episode.episode_id)
        if episode.recording_session_id == "":
            raise SplitError("recording_session_id must be non-empty when present.")


def _validate_session_isolation(
    episodes: tuple[EpisodeDescriptor, ...],
    assignments: Mapping[str, str],
) -> None:
    seen: dict[str, str] = {}
    for episode in episodes:
        if episode.recording_session_id is None:
            continue
        split_name = assignments[episode.episode_id]
        prior = seen.setdefault(episode.recording_session_id, split_name)
        if prior != split_name:
            raise SplitError(
                f"recording session {episode.recording_session_id!r} leaked across splits."
            )


def _dataset_digest(assignments: tuple[SplitAssignment, ...]) -> str:
    source = [
        {
            "episode_id": item.episode_id,
            "goal_id": item.goal_id,
            "data_digest": item.data_digest,
            "action_schema_version": item.action_schema_version,
            "observation_schema_version": item.observation_schema_version,
            "recording_session_id": item.recording_session_id,
        }
        for item in assignments
    ]
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
