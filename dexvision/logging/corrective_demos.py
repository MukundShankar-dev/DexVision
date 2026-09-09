"""Deterministic Level 4.6 failure and corrective-demonstration generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoStepData,
    action_schema_from_metadata,
    load_logged_demo,
    observation_schema_from_metadata,
)
from dexvision.logging.level4_collection import (
    DEFAULT_LEVEL4_CONFIG,
    LEVEL4_NOMINAL_GROUPS,
    PilotEpisode,
    discover_pilot_episodes,
    initial_state_digest,
    load_level4_collection_config,
)
from dexvision.logging.session_manifest import (
    RecordingSession,
    append_session_manifest,
    load_session_manifest,
)


LEVEL4_6_VERSION = "level4/scripted-corrections-v1"
LEVEL4_6_EPISODES_PER_CELL = 10
LEVEL4_6_OPERATOR_ID = "deterministic_scripted_correction_v1"
LEVEL4_6_QUARANTINE_VERSION = "level4/correction-quarantine-v1"
LEVEL4_6_QUARANTINE_FILENAME = "correction_quarantine_manifest.json"
LEVEL4_6_MANUAL_REPLAY_VERSION = "level4/correction-manual-replay-v1"
LEVEL4_6_MANUAL_REPLAY_FILENAME = "correction_manual_replay_manifest.json"
LEVEL4_6_STREAMS = (
    "expert_success",
    "ordinary_failure",
    "policy_rollout",
    "corrective_intervention",
)
UNSAFE_FAILURE_CLASSES = frozenset(
    {"workspace_violation", "joint_limit_violation"}
)
CORRECTABLE_FAILURE_CLASSES = frozenset(
    {"approach_miss", "failed_acquisition", "placement_miss"}
)
FAILURE_SOURCE_SKILLS = {
    "approach_miss": "pick_place_sequence",
    "wrong_contact": "reach_object",
    "failed_acquisition": "pick_place_sequence",
    "slip_drop": "pick_place_sequence",
    "placement_miss": "pick_place_sequence",
    "premature_release": "pick_place_sequence",
    "timeout": "reach_object",
    "workspace_violation": "reach_object",
    "joint_limit_violation": "pick_place_sequence",
}
CORRECTION_SOURCE_CELL = {
    "correction_approach_miss": "failure_approach_miss",
    "correction_failed_acquisition": "failure_failed_acquisition",
    "correction_placement_miss": "failure_placement_miss",
}
_PLACEMENT_BRIDGE_STEPS = 90
_SEED_BASES = {"train": 4_600_000, "validation": 4_610_000, "test": 4_620_000}


class CorrectiveDemoError(ValueError):
    """Raised when a Level 4.6 plan or derived episode is invalid."""


@dataclass(frozen=True)
class CorrectionAssignment:
    """One frozen Level 4.6 cell/repetition assignment."""

    sequence: int
    coverage_cell_id: str
    failure_class: str
    retryability: str
    source: str
    split: str
    repetition: int
    generation_seed: int
    source_failure_cell_id: str | None

    @property
    def is_correction(self) -> bool:
        return self.source == "corrective_intervention"

    @property
    def episode_id(self) -> str:
        return f"level46_{self.sequence:03d}"

    @property
    def session_id(self) -> str:
        return (
            f"level46_{self.split}_{self.sequence:03d}_"
            f"{self.coverage_cell_id}"
        )


def build_level4_6_plan(
    config_path: str | Path = DEFAULT_LEVEL4_CONFIG,
) -> tuple[CorrectionAssignment, ...]:
    """Expand the frozen 12 cells into 120 deterministic assignments."""

    config, _ = load_level4_collection_config(config_path)
    contract = _mapping(config, "failure_and_correction_contract")
    required_classes = tuple(str(item) for item in contract["required_failure_classes"])
    cells = [
        cell
        for cell in config["coverage_cells"]
        if isinstance(cell, Mapping) and cell.get("data_group") == "failure_correction"
    ]
    if len(cells) != 12:
        raise CorrectiveDemoError(
            f"Level 4.6 requires exactly 12 failure/correction cells, got {len(cells)}."
        )
    failure_cells = [cell for cell in cells if cell.get("required_source") == "scripted"]
    correction_cells = [
        cell for cell in cells if cell.get("required_source") == "corrective_intervention"
    ]
    observed_classes = tuple(str(cell.get("failure_class")) for cell in failure_cells)
    if len(failure_cells) != 9 or set(observed_classes) != set(required_classes):
        raise CorrectiveDemoError(
            "Level 4.6 requires one scripted failure cell for every frozen failure class."
        )
    if len(correction_cells) != 3 or {
        str(cell.get("failure_class")) for cell in correction_cells
    } != set(CORRECTABLE_FAILURE_CLASSES):
        raise CorrectiveDemoError(
            "Level 4.6 requires corrections for approach miss, failed acquisition, "
            "and placement miss."
        )

    assignments: list[CorrectionAssignment] = []
    for cell_index, cell in enumerate(cells):
        cell_id = _required_string(cell, "id")
        source = _required_string(cell, "required_source")
        split = _required_string(cell, "split_owner")
        failure_class = _required_string(cell, "failure_class")
        retryability = _required_string(cell, "retryability")
        minima = _mapping(cell, "minimum_accepted_by_split")
        if int(minima[split]) != 1 or sum(int(value) for value in minima.values()) != 1:
            raise CorrectiveDemoError(
                f"frozen anchor cell {cell_id!r} must own exactly one split minimum."
            )
        if failure_class in UNSAFE_FAILURE_CLASSES and retryability != "abort":
            raise CorrectiveDemoError(
                f"unsafe failure cell {cell_id!r} must be abort-only."
            )
        for repetition in range(1, LEVEL4_6_EPISODES_PER_CELL + 1):
            assignments.append(
                CorrectionAssignment(
                    sequence=len(assignments) + 1,
                    coverage_cell_id=cell_id,
                    failure_class=failure_class,
                    retryability=retryability,
                    source=source,
                    split=split,
                    repetition=repetition,
                    generation_seed=_SEED_BASES[split] + cell_index * 100 + repetition,
                    source_failure_cell_id=CORRECTION_SOURCE_CELL.get(cell_id),
                )
            )
    if len(assignments) != 120 or len({item.generation_seed for item in assignments}) != 120:
        raise CorrectiveDemoError("Level 4.6 plan must contain 120 unique assignments.")
    return tuple(assignments)


def generate_level4_6_dataset(
    *,
    config_path: str | Path,
    dataset_dir: str | Path,
    max_assignments: int = 0,
) -> tuple[Path, ...]:
    """Append every missing frozen Level 4.6 assignment in plan order."""

    if max_assignments < 0:
        raise CorrectiveDemoError("max_assignments must be non-negative.")
    root = Path(dataset_dir)
    plan = build_level4_6_plan(config_path)
    existing = discover_level4_6_episodes(root)
    by_slot = {
        (_level4_6_payload(item.metadata).get("coverage_cell_id"),
         _level4_6_payload(item.metadata).get("repetition")): item.path
        for item in existing
    }
    missing = [
        item
        for item in plan
        if (item.coverage_cell_id, item.repetition) not in by_slot
    ]
    if max_assignments:
        missing = missing[:max_assignments]

    nominal_sources = _nominal_source_pool(root, config_path=config_path)
    all_level4_6 = tuple(
        item
        for item in discover_pilot_episodes(root)
        if isinstance(item.metadata.get("level4_6"), Mapping)
    )
    used_nominal_ids = {
        str(item.metadata.get("source_episode_id"))
        for item in all_level4_6
        if item.metadata.get("data_stream") == "ordinary_failure"
    }
    generated: list[Path] = []
    for assignment in missing:
        assignment, output_dir = _next_available_assignment(root, assignment)
        if assignment.is_correction:
            key = (assignment.source_failure_cell_id, assignment.repetition)
            source_path = by_slot.get(key)
            if source_path is None:
                raise CorrectiveDemoError(
                    f"correction {assignment.coverage_cell_id!r} requires missing "
                    f"source slot {key}."
                )
        else:
            source = _take_nominal_source(
                nominal_sources,
                split=assignment.split,
                skill_name=FAILURE_SOURCE_SKILLS[assignment.failure_class],
                used_episode_ids=used_nominal_ids,
            )
            source_path = source.path
            used_nominal_ids.add(source.episode_id)

        append_level4_6_session(
            dataset_dir=root,
            assignment=assignment,
            source_dir=source_path,
        )
        if assignment.is_correction:
            result = generate_scripted_correction(
                source_rollout=source_path,
                output_dir=output_dir,
                assignment=assignment,
                dataset_dir=root,
            )
        else:
            result = generate_scripted_failure(
                source_rollout=source_path,
                output_dir=output_dir,
                assignment=assignment,
                dataset_dir=root,
            )
        by_slot[(assignment.coverage_cell_id, assignment.repetition)] = result
        generated.append(result)
    return tuple(generated)


def append_level4_6_session(
    *,
    dataset_dir: str | Path,
    assignment: CorrectionAssignment,
    source_dir: str | Path,
) -> None:
    """Append one generated-process session record before writing its episode."""

    root = Path(dataset_dir)
    manifest_path = root / "session_manifest.json"
    if manifest_path.exists():
        ids = {
            item.recording_session_id
            for item in load_session_manifest(manifest_path).sessions
        }
        if assignment.session_id in ids:
            return
    source_metadata = load_logged_demo(source_dir).metadata
    reset_seed = int(source_metadata["random_seed"])
    digest_input = (
        f"{LEVEL4_6_VERSION}|{assignment.generation_seed}|"
        f"{source_metadata['episode_id']}"
    ).encode()
    append_session_manifest(
        manifest_path,
        RecordingSession(
            recording_session_id=assignment.session_id,
            operator_id=LEVEL4_6_OPERATOR_ID,
            split=assignment.split,
            process_start_timestamp=datetime.now(timezone.utc).isoformat(),
            reset_seed=reset_seed,
            calibration_record_digest="sha256:" + hashlib.sha256(digest_input).hexdigest(),
        ),
    )


def generate_scripted_failure(
    *,
    source_rollout: str | Path,
    output_dir: str | Path,
    assignment: CorrectionAssignment,
    dataset_dir: str | Path | None = None,
) -> Path:
    """Create one deterministic ordinary-failure episode from a scripted rollout."""

    if assignment.is_correction:
        raise CorrectiveDemoError("generate_scripted_failure requires a failure assignment.")
    source_path = Path(source_rollout)
    source = load_logged_demo(source_path)
    if source.metadata.get("source") != "scripted" or source.success is not True:
        raise CorrectiveDemoError("failure sources must be successful scripted episodes.")
    selection, phase_override = _failure_selection(
        np.asarray(source.online_phases), assignment.failure_class
    )
    actions, requested, commanded, safety_masks, safety_reasons = _failure_actions(
        source,
        selection=selection,
        phases=phase_override,
        failure_class=assignment.failure_class,
    )
    metadata = _derived_metadata(
        source.metadata,
        assignment=assignment,
        source_episode_id=str(source.metadata["episode_id"]),
        source_episode_path=_relative_source_path(source_path, dataset_dir),
        intervention_interval=None,
        original_terminal_result={
            "success": False,
            "failure_class": assignment.failure_class,
            "outcome": (
                "aborted_unsafe"
                if assignment.failure_class in UNSAFE_FAILURE_CLASSES
                else "failed_unrecovered"
            ),
        },
        final_outcome=(
            "aborted_unsafe"
            if assignment.failure_class in UNSAFE_FAILURE_CLASSES
            else "failed_unrecovered"
        ),
    )
    _write_derived_episode(
        source,
        output_dir=Path(output_dir),
        metadata=metadata,
        source_indices=selection,
        phases=phase_override,
        actions=actions,
        requested_actions=requested,
        commanded_actions=commanded,
        safety_masks=safety_masks,
        safety_reasons=safety_reasons,
        intervention_start=None,
        success=False,
    )
    return Path(output_dir)


def generate_scripted_correction(
    *,
    source_rollout: str | Path,
    output_dir: str | Path,
    assignment: CorrectionAssignment,
    dataset_dir: str | Path | None = None,
) -> Path:
    """Append a deterministic expert intervention to a retryable failure episode."""

    if not assignment.is_correction:
        raise CorrectiveDemoError("generate_scripted_correction requires a correction assignment.")
    failure_path = Path(source_rollout)
    failure = load_logged_demo(failure_path)
    validate_level4_6_metadata(failure.metadata)
    if failure.metadata.get("data_stream") != "ordinary_failure":
        raise CorrectiveDemoError("correction source must be an ordinary failure episode.")
    if failure.metadata.get("failure_class") != assignment.failure_class:
        raise CorrectiveDemoError("correction and source failure classes must match.")
    if assignment.failure_class not in CORRECTABLE_FAILURE_CLASSES:
        raise CorrectiveDemoError("unsafe or non-retryable failures cannot become targets.")
    root = Path(dataset_dir) if dataset_dir is not None else failure_path.parents[2]
    nominal_path = _find_episode_by_id(
        root, str(failure.metadata["source_episode_id"])
    )
    nominal = load_logged_demo(nominal_path)
    suffix_indices = _correction_suffix_indices(
        np.asarray(nominal.online_phases), assignment.failure_class
    )
    prefix_count = int(failure.timestamps.shape[0])
    intervention_interval = [prefix_count, prefix_count + len(suffix_indices)]
    metadata = _derived_metadata(
        nominal.metadata,
        assignment=assignment,
        source_episode_id=str(failure.metadata["episode_id"]),
        source_episode_path=_relative_source_path(failure_path, dataset_dir),
        intervention_interval=intervention_interval,
        original_terminal_result=dict(failure.metadata["original_terminal_result"]),
        final_outcome="corrected_success",
    )
    metadata["trigger_episode_source_id"] = str(nominal.metadata["episode_id"])
    _write_correction_episode(
        failure,
        nominal,
        output_dir=Path(output_dir),
        metadata=metadata,
        suffix_indices=suffix_indices,
        failure_class=assignment.failure_class,
    )
    return Path(output_dir)


def validate_level4_6_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate stream separation and conditional correction provenance."""

    payload = _level4_6_payload(metadata)
    if payload.get("version") != LEVEL4_6_VERSION:
        raise CorrectiveDemoError(f"Level 4.6 metadata version must be {LEVEL4_6_VERSION}.")
    if metadata.get("data_stream") not in LEVEL4_6_STREAMS:
        raise CorrectiveDemoError("data_stream must use the frozen four-stream vocabulary.")
    failure_class = metadata.get("failure_class")
    if not isinstance(failure_class, str) or not failure_class:
        raise CorrectiveDemoError("failure_class must be a non-empty string.")
    retryability = metadata.get("retryability")
    if not isinstance(retryability, str) or not retryability:
        raise CorrectiveDemoError("retryability must be a non-empty string.")
    if failure_class in UNSAFE_FAILURE_CLASSES:
        if retryability != "abort" or metadata.get("data_stream") != "ordinary_failure":
            raise CorrectiveDemoError("workspace/joint-limit failures must be abort-only.")
        if metadata.get("intervention_interval") not in (None, []):
            raise CorrectiveDemoError("unsafe failures cannot contain intervention targets.")
    source_episode_id = metadata.get("source_episode_id")
    if not isinstance(source_episode_id, str) or not source_episode_id:
        raise CorrectiveDemoError("source_episode_id must be a non-empty string.")
    trigger_source = metadata.get("trigger_source")
    if trigger_source not in {"teleoperation", "scripted", "policy_rollout"}:
        raise CorrectiveDemoError("trigger_source is invalid.")
    checkpoint = metadata.get("source_policy_checkpoint")
    na_reason = metadata.get("source_policy_checkpoint_not_applicable_reason")
    if trigger_source == "policy_rollout":
        if not isinstance(checkpoint, str) or not checkpoint or na_reason is not None:
            raise CorrectiveDemoError(
                "policy-triggered provenance requires a checkpoint and no N/A reason."
            )
    elif checkpoint is not None or na_reason != "trigger_not_policy_rollout":
        raise CorrectiveDemoError(
            "non-policy provenance requires null checkpoint and stable N/A reason."
        )
    if metadata.get("data_stream") == "corrective_intervention":
        interval = metadata.get("intervention_interval")
        if (
            not isinstance(interval, Sequence)
            or isinstance(interval, str)
            or len(interval) != 2
            or not all(isinstance(value, int) for value in interval)
            or interval[1] <= interval[0]
        ):
            raise CorrectiveDemoError("corrections require a valid intervention interval.")
        if metadata.get("final_outcome") != "corrected_success":
            raise CorrectiveDemoError("correction outcome must remain distinct and explicit.")


def discover_level4_6_episodes(dataset_dir: str | Path) -> tuple[PilotEpisode, ...]:
    """Discover only generated Level 4.6 episodes without accepting nominal data."""

    root = Path(dataset_dir)
    quarantined = load_level4_6_quarantine(root)
    return tuple(
        item
        for item in discover_pilot_episodes(root)
        if isinstance(item.metadata.get("level4_6"), Mapping)
        and item.episode_id not in quarantined
    )


def load_level4_6_quarantine(dataset_dir: str | Path) -> frozenset[str]:
    """Return append-only diagnostic exclusions from active Level 4.6 coverage."""

    path = Path(dataset_dir) / LEVEL4_6_QUARANTINE_FILENAME
    if not path.exists():
        return frozenset()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectiveDemoError(f"could not read {path}: {exc}") from exc
    if payload.get("version") != LEVEL4_6_QUARANTINE_VERSION:
        raise CorrectiveDemoError("Level 4.6 quarantine manifest version is invalid.")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise CorrectiveDemoError("Level 4.6 quarantine entries must be a list.")
    result: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CorrectiveDemoError("Level 4.6 quarantine entries must be mappings.")
        episode_id = entry.get("episode_id")
        reason = entry.get("reason")
        if not isinstance(episode_id, str) or not episode_id:
            raise CorrectiveDemoError("quarantine episode_id must be non-empty.")
        if not isinstance(reason, str) or not reason:
            raise CorrectiveDemoError("quarantine reason must be non-empty.")
        result.add(episode_id)
    return frozenset(result)


def quarantine_level4_6_episodes(
    dataset_dir: str | Path,
    episode_ids: Sequence[str],
    *,
    reason: str,
) -> Path:
    """Append immutable diagnostic exclusions without changing episode payloads."""

    if not reason.strip():
        raise CorrectiveDemoError("quarantine reason must be non-empty.")
    root = Path(dataset_dir)
    path = root / LEVEL4_6_QUARANTINE_FILENAME
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != LEVEL4_6_QUARANTINE_VERSION:
            raise CorrectiveDemoError("Level 4.6 quarantine manifest version is invalid.")
    else:
        payload = {"version": LEVEL4_6_QUARANTINE_VERSION, "entries": []}
    entries = payload["entries"]
    assert isinstance(entries, list)
    existing = {
        str(entry.get("episode_id"))
        for entry in entries
        if isinstance(entry, Mapping)
    }
    for episode_id in episode_ids:
        if not isinstance(episode_id, str) or not episode_id:
            raise CorrectiveDemoError("quarantine episode ids must be non-empty strings.")
        if episode_id not in existing:
            entries.append({"episode_id": episode_id, "reason": reason})
            existing.add(episode_id)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def append_manual_correction_replay(
    dataset_dir: str | Path,
    *,
    episode_id: str,
    notes: str,
) -> Path:
    """Record one user-confirmed visible correction replay append-only."""

    if not episode_id.strip() or not notes.strip():
        raise CorrectiveDemoError("manual replay episode_id and notes must be non-empty.")
    root = Path(dataset_dir)
    matches = [
        item for item in discover_level4_6_episodes(root) if item.episode_id == episode_id
    ]
    if len(matches) != 1:
        raise CorrectiveDemoError(
            f"manual replay must reference one active Level 4.6 episode, found {len(matches)}."
        )
    episode = matches[0]
    if episode.metadata.get("data_stream") != "corrective_intervention":
        raise CorrectiveDemoError("manual replay must reference a correction episode.")
    path = root / LEVEL4_6_MANUAL_REPLAY_FILENAME
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != LEVEL4_6_MANUAL_REPLAY_VERSION:
            raise CorrectiveDemoError("Level 4.6 manual replay manifest version is invalid.")
    else:
        payload = {"version": LEVEL4_6_MANUAL_REPLAY_VERSION, "entries": []}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise CorrectiveDemoError("Level 4.6 manual replay entries must be a list.")
    if any(
        isinstance(entry, Mapping) and entry.get("episode_id") == episode_id
        for entry in entries
    ):
        raise CorrectiveDemoError(
            f"manual replay approval already exists for episode {episode_id!r}."
        )
    entries.append(
        {
            "episode_id": episode_id,
            "failure_class": episode.metadata["failure_class"],
            "intervention_interval": episode.metadata["intervention_interval"],
            "passed": True,
            "notes": notes,
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def load_manual_correction_replays(
    dataset_dir: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Load validated Level 4.6 visible-replay approvals."""

    path = Path(dataset_dir) / LEVEL4_6_MANUAL_REPLAY_FILENAME
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectiveDemoError(f"could not read {path}: {exc}") from exc
    if payload.get("version") != LEVEL4_6_MANUAL_REPLAY_VERSION:
        raise CorrectiveDemoError("Level 4.6 manual replay manifest version is invalid.")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise CorrectiveDemoError("Level 4.6 manual replay entries must be a list.")
    seen: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CorrectiveDemoError("manual replay entries must be mappings.")
        episode_id = entry.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in seen:
            raise CorrectiveDemoError("manual replay episode ids must be unique strings.")
        if entry.get("passed") is not True:
            raise CorrectiveDemoError("saved manual replay entries must be passed approvals.")
        if not isinstance(entry.get("notes"), str) or not entry["notes"].strip():
            raise CorrectiveDemoError("manual replay notes must be non-empty.")
        seen.add(episode_id)
        validated.append(dict(entry))
    return tuple(validated)


def select_level4_training_streams(
    episode_dirs: Sequence[str | Path],
    *,
    include_corrections: bool,
) -> tuple[Path, ...]:
    """Select expert successes and optionally corrections in stable path order."""

    selected: list[Path] = []
    for raw_path in sorted((Path(path) for path in episode_dirs), key=lambda item: str(item)):
        metadata = load_logged_demo(raw_path).metadata
        stream = metadata.get("data_stream")
        expert_success = stream == "expert_success" or (
            stream is None
            and metadata.get("source") == "scripted"
            and metadata.get("success") is True
        )
        if expert_success or (
            include_corrections and stream == "corrective_intervention"
        ):
            selected.append(raw_path)
    return tuple(selected)


def _failure_selection(
    phases: np.ndarray, failure_class: str
) -> tuple[np.ndarray, np.ndarray]:
    if phases.ndim != 1 or phases.size < 2:
        raise CorrectiveDemoError("source rollout must contain at least two phase samples.")
    values = phases.astype(str)
    phase_targets = {
        "approach_miss": {"approach"},
        "wrong_contact": {"approach"},
        "failed_acquisition": {"approach", "acquire"},
        "slip_drop": {"approach", "acquire", "lift", "stabilize"},
        "placement_miss": {"approach", "acquire", "lift", "stabilize", "transport", "place"},
        "premature_release": {"approach", "acquire", "lift", "stabilize", "transport"},
        "timeout": {"approach"},
        "workspace_violation": {"approach"},
        "joint_limit_violation": {"approach"},
    }
    allowed = phase_targets[failure_class]
    matching = np.flatnonzero(np.isin(values, tuple(allowed)))
    stop = int(matching[-1] + 1) if matching.size else min(2, values.size)
    stop = max(2, stop)
    indices = np.arange(stop, dtype=int)
    selected_phases = values[indices].copy()
    if failure_class == "placement_miss":
        place = np.flatnonzero(values == "place")
        if not place.size:
            raise CorrectiveDemoError("placement failure source has no place phase.")
        stop = min(values.size, int(place[0]) + 6)
        indices = np.arange(stop, dtype=int)
        selected_phases = values[indices].copy()
    elif failure_class == "timeout":
        count = max(20, min(60, values.size))
        indices = np.zeros(count, dtype=int)
        selected_phases = np.full(count, "approach", dtype="<U32")
    elif failure_class in UNSAFE_FAILURE_CLASSES:
        indices = np.asarray([0, min(1, values.size - 1)], dtype=int)
        selected_phases = np.full(2, "approach", dtype="<U32")
    return indices, selected_phases


def _failure_actions(
    source: Any,
    *,
    selection: np.ndarray,
    phases: np.ndarray,
    failure_class: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(source.applied_actions)[selection].copy()
    requested = actions.copy()
    commanded = actions.copy()
    masks = np.zeros(actions.shape, dtype=np.uint8)
    reasons = np.full(actions.shape, "none", dtype="<U32")
    open_fingers = np.asarray(source.applied_actions)[0, 7:].copy()

    if failure_class == "approach_miss":
        actions[:, 0] += 0.050
    elif failure_class == "wrong_contact":
        actions[:, 1] += 0.060
    elif failure_class == "failed_acquisition":
        actions[:, 7:] = open_fingers
    elif failure_class == "slip_drop":
        active = np.isin(phases, ("lift", "stabilize"))
        actions[active, 7:] = open_fingers
    elif failure_class == "placement_miss":
        active = phases == "place"
        actions[active, 1] += 0.050
    elif failure_class == "premature_release":
        active = phases == "transport"
        actions[active, 7:] = open_fingers
    elif failure_class == "timeout":
        actions[:] = actions[0]
    elif failure_class == "workspace_violation":
        requested = actions.copy()
        commanded = actions.copy()
        requested[-1, 0] = 10.0
        commanded[-1, 0] = 10.0
        masks[-1, 0] = 1
        reasons[-1, 0] = "workspace_clip"
        return actions, requested, commanded, masks, reasons
    elif failure_class == "joint_limit_violation":
        requested = actions.copy()
        commanded = actions.copy()
        requested[-1, -1] = 10.0
        commanded[-1, -1] = 10.0
        masks[-1, -1] = 1
        reasons[-1, -1] = "joint_limit_clip"
        return actions, requested, commanded, masks, reasons

    requested = actions.copy()
    commanded = actions.copy()
    return actions, requested, commanded, masks, reasons


def _correction_suffix_indices(phases: np.ndarray, failure_class: str) -> np.ndarray:
    values = phases.astype(str)
    if failure_class in {"approach_miss", "failed_acquisition"}:
        return np.arange(values.size, dtype=int)
    if failure_class == "placement_miss":
        place_candidates = np.flatnonzero(values == "place")
        release_candidates = np.flatnonzero(values == "release")
        if not place_candidates.size or not release_candidates.size:
            raise CorrectiveDemoError(
                "placement correction source requires place and release phases."
            )
        bridge = np.full(_PLACEMENT_BRIDGE_STEPS, int(place_candidates[0]), dtype=int)
        tail = np.arange(int(place_candidates[0]), values.size, dtype=int)
        return np.concatenate((bridge, tail))
    raise CorrectiveDemoError(f"unsupported correction class: {failure_class}")


def _write_correction_episode(
    failure: Any,
    nominal: Any,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    suffix_indices: np.ndarray,
    failure_class: str,
) -> None:
    metadata["initial_commanded_action"] = np.asarray(
        failure.commanded_actions[0]
    ).tolist()
    metadata["initial_applied_action"] = np.asarray(
        failure.applied_actions[0]
    ).tolist()
    logger = DemoLogger(
        output_dir,
        action_schema=action_schema_from_metadata(metadata),
        observation_schema=observation_schema_from_metadata(metadata),
    )
    prefix_count = int(failure.timestamps.shape[0])
    phases = [str(item) for item in np.asarray(failure.online_phases)] + [
        str(np.asarray(nominal.online_phases)[index]) for index in suffix_indices
    ]
    _allow_observed_phase_edges(metadata, phases)
    logger.start_episode(metadata)
    dt = _control_interval(nominal.timestamps)
    for output_index in range(prefix_count + len(suffix_indices)):
        if output_index < prefix_count:
            episode = failure
            source_index = output_index
            intervention = False
            failure_reason = ""
            request_source = str(np.asarray(failure.request_sources)[source_index])
        else:
            episode = nominal
            source_index = int(suffix_indices[output_index - prefix_count])
            intervention = True
            failure_reason = failure_class
            request_source = "script"
        timestamp = output_index * dt
        relative_suffix_index = output_index - prefix_count
        placement_bridge = (
            episode is nominal
            and failure_class == "placement_miss"
            and 0 <= relative_suffix_index < _PLACEMENT_BRIDGE_STEPS
        )
        if placement_bridge:
            alpha = float(relative_suffix_index + 1) / float(
                _PLACEMENT_BRIDGE_STEPS
            )
            start_action = np.asarray(failure.applied_actions[-1], dtype=np.float64)
            target_action = np.asarray(
                nominal.applied_actions[source_index], dtype=np.float64
            )
            action = (1.0 - alpha) * start_action + alpha * target_action
            requested_action = action
            commanded_action = action
            safety_mask = np.zeros(action.shape, dtype=np.uint8)
            safety_reason = np.full(action.shape, "none", dtype="<U32")
        else:
            action = np.asarray(episode.applied_actions)[source_index]
            requested_action = np.asarray(episode.requested_actions)[source_index]
            commanded_action = np.asarray(episode.commanded_actions)[source_index]
            safety_mask = np.asarray(episode.safety_masks)[source_index]
            safety_reason = np.asarray(episode.safety_reasons)[source_index]
        logger.append(
            _step_from_episode(
                episode,
                source_index=source_index,
                timestamp=timestamp,
                action=action,
                requested_action=requested_action,
                commanded_action=commanded_action,
                safety_mask=safety_mask,
                safety_reason=safety_reason,
                phase=phases[output_index],
                intervention=intervention,
                failure_reason=failure_reason,
                request_source=request_source,
            )
        )
    logger.close(success=True)


def _write_derived_episode(
    source: Any,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    source_indices: np.ndarray,
    phases: np.ndarray,
    actions: np.ndarray,
    requested_actions: np.ndarray,
    commanded_actions: np.ndarray,
    safety_masks: np.ndarray,
    safety_reasons: np.ndarray,
    intervention_start: int | None,
    success: bool,
) -> None:
    metadata["initial_commanded_action"] = np.asarray(
        commanded_actions[0]
    ).tolist()
    metadata["initial_applied_action"] = np.asarray(actions[0]).tolist()
    _allow_observed_phase_edges(metadata, phases.tolist())
    logger = DemoLogger(
        output_dir,
        action_schema=action_schema_from_metadata(metadata),
        observation_schema=observation_schema_from_metadata(metadata),
    )
    logger.start_episode(metadata)
    dt = _control_interval(source.timestamps)
    for output_index, source_index in enumerate(source_indices.tolist()):
        intervention = intervention_start is not None and output_index >= intervention_start
        logger.append(
            _step_from_episode(
                source,
                source_index=source_index,
                timestamp=output_index * dt,
                action=actions[output_index],
                requested_action=requested_actions[output_index],
                commanded_action=commanded_actions[output_index],
                safety_mask=safety_masks[output_index],
                safety_reason=safety_reasons[output_index],
                phase=str(phases[output_index]),
                intervention=intervention,
                failure_reason=(metadata["failure_class"] if intervention else ""),
                request_source="script",
            )
        )
    logger.close(success=success)


def _step_from_episode(
    episode: Any,
    *,
    source_index: int,
    timestamp: float,
    action: np.ndarray,
    requested_action: np.ndarray,
    commanded_action: np.ndarray,
    safety_mask: np.ndarray,
    safety_reason: Sequence[str],
    phase: str,
    intervention: bool,
    failure_reason: str,
    request_source: str,
) -> DemoStepData:
    def optional(name: str) -> np.ndarray | None:
        value = getattr(episode, name)
        return None if value is None else np.asarray(value)[source_index]

    return DemoStepData(
        features=np.asarray(episode.features)[source_index],
        action=np.asarray(action),
        robot_state=np.asarray(episode.robot_states)[source_index],
        tracking_quality=np.asarray(episode.tracking_quality)[source_index],
        timestamp=timestamp,
        landmarks=optional("landmarks"),
        object_state=optional("object_states"),
        task_state=optional("task_states"),
        requested_action=np.asarray(requested_action),
        commanded_action=np.asarray(commanded_action),
        applied_action=np.asarray(action),
        safety_mask=np.asarray(safety_mask),
        safety_reason=tuple(str(item) for item in safety_reason),
        request_source=request_source,
        online_phase=phase,
        audited_phase=phase,
        intervention=intervention,
        failure_reason=failure_reason,
        action_timestamp=timestamp,
        task_timestamp=timestamp,
        state_timestamp=timestamp,
    )


def _derived_metadata(
    source: Mapping[str, Any],
    *,
    assignment: CorrectionAssignment,
    source_episode_id: str,
    source_episode_path: str,
    intervention_interval: list[int] | None,
    original_terminal_result: Mapping[str, Any],
    final_outcome: str,
) -> dict[str, Any]:
    metadata = json.loads(json.dumps(source))
    metadata.update(
        {
            "episode_id": assignment.episode_id,
            "recording_session_id": assignment.session_id,
            "operator_id": LEVEL4_6_OPERATOR_ID,
            "source": assignment.source,
            "data_stream": (
                "corrective_intervention"
                if assignment.is_correction
                else "ordinary_failure"
            ),
            "goal_condition_id": assignment.coverage_cell_id,
            "source_episode_id": source_episode_id,
            "source_episode_path": source_episode_path,
            "trigger_source": "scripted",
            "source_policy_checkpoint": None,
            "source_policy_checkpoint_not_applicable_reason": (
                "trigger_not_policy_rollout"
            ),
            "failure_class": assignment.failure_class,
            "failure_reason": assignment.failure_class,
            "retryability": assignment.retryability,
            "intervention_interval": intervention_interval,
            "original_terminal_result": dict(original_terminal_result),
            "final_outcome": final_outcome,
            "random_seed": int(source["random_seed"]),
            "level4_6": {
                "version": LEVEL4_6_VERSION,
                "coverage_cell_id": assignment.coverage_cell_id,
                "repetition": assignment.repetition,
                "generation_seed": assignment.generation_seed,
                "generator": "deterministic_failure_injection_and_scripted_expert_v1",
                "pre_intervention_frames_preserved": assignment.is_correction,
                "unsafe_intervals_are_targets": False,
            },
        }
    )
    metadata.pop("procedural_expansion", None)
    action_contract = dict(metadata["action_contract"])
    reason_codes = list(action_contract["safety_reason_codes"])
    for reason in ("workspace_clip", "joint_limit_clip"):
        if reason not in reason_codes:
            reason_codes.append(reason)
    action_contract["safety_reason_codes"] = reason_codes
    metadata["action_contract"] = action_contract
    if not metadata.get("initial_state_digest"):
        task_config = _mapping(metadata, "task_config")
        initial_state = _mapping(task_config, "initial_state")
        variation = task_config.get("procedural_variation")
        if not isinstance(variation, Mapping):
            variation = {}
        metadata["initial_state_digest"] = initial_state_digest(
            initial_state=initial_state,
            procedural_variation=variation,
        )
    return metadata


def _allow_observed_phase_edges(metadata: dict[str, Any], phases: Sequence[str]) -> None:
    contract = dict(metadata["phase_contract"])
    transitions = [dict(item) for item in contract["transitions"]]
    allowed = {(str(item["from"]), str(item["to"])) for item in transitions}
    for source, target in zip(phases, phases[1:]):
        if source != target and (source, target) not in allowed:
            transitions.append({"from": source, "to": target})
            allowed.add((source, target))
    contract["transitions"] = transitions
    metadata["phase_contract"] = contract


def _nominal_source_pool(
    dataset_dir: Path, *, config_path: str | Path
) -> dict[tuple[str, str], list[PilotEpisode]]:
    config, _ = load_level4_collection_config(config_path)
    active_test_prefixes = tuple(
        str(item)
        for item in _mapping(config, "level4_5b_procedural_expansion")
        ["session_slots_by_split"]["test"]
    )
    sessions = {
        item.recording_session_id: item.split
        for item in load_session_manifest(dataset_dir / "session_manifest.json").sessions
    }
    pools: dict[tuple[str, str], list[PilotEpisode]] = defaultdict(list)
    for episode in discover_pilot_episodes(dataset_dir):
        if not episode.expert_accepted or episode.source != "scripted":
            continue
        group = {
            "reach_object": "reach",
            "pick_place_sequence": "pick_place",
            "push_object_to_target": "push",
            "press_button": "button",
        }.get(episode.skill_name)
        if group not in LEVEL4_NOMINAL_GROUPS:
            continue
        split = sessions.get(episode.session_id)
        if split is None:
            continue
        if split == "test" and not episode.session_id.startswith(active_test_prefixes):
            continue
        pools[(split, episode.skill_name)].append(episode)
    for values in pools.values():
        values.sort(key=lambda item: (item.episode_id, str(item.path)))
    return pools


def _take_nominal_source(
    pools: Mapping[tuple[str, str], Sequence[PilotEpisode]],
    *,
    split: str,
    skill_name: str,
    used_episode_ids: set[str],
) -> PilotEpisode:
    for episode in pools.get((split, skill_name), ()):
        if episode.episode_id not in used_episode_ids:
            return episode
    raise CorrectiveDemoError(
        f"not enough unique accepted {split} {skill_name} scripted sources."
    )


def _replacement_assignment(
    assignment: CorrectionAssignment, *, generation: int = 1
) -> CorrectionAssignment:
    return CorrectionAssignment(
        sequence=assignment.sequence + generation * 1000,
        coverage_cell_id=assignment.coverage_cell_id,
        failure_class=assignment.failure_class,
        retryability=assignment.retryability,
        source=assignment.source,
        split=assignment.split,
        repetition=assignment.repetition,
        generation_seed=assignment.generation_seed + generation * 1_000_000,
        source_failure_cell_id=assignment.source_failure_cell_id,
    )


def _next_available_assignment(
    dataset_root: Path, assignment: CorrectionAssignment
) -> tuple[CorrectionAssignment, Path]:
    candidate = assignment
    generation = 0
    while True:
        output = dataset_root / candidate.session_id / "episode_000001"
        if not output.exists():
            return candidate, output
        generation += 1
        candidate = _replacement_assignment(assignment, generation=generation)


def _find_episode_by_id(dataset_dir: Path, episode_id: str) -> Path:
    matches: list[Path] = []
    for metadata_path in dataset_dir.glob("**/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("episode_id") == episode_id:
            matches.append(metadata_path.parent)
    if len(matches) != 1:
        raise CorrectiveDemoError(
            f"expected one source episode {episode_id!r}, found {len(matches)}."
        )
    return matches[0]


def _relative_source_path(path: Path, dataset_dir: str | Path | None) -> str:
    if dataset_dir is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(Path(dataset_dir).resolve()))
    except ValueError:
        return str(path)


def _control_interval(timestamps: np.ndarray) -> float:
    values = np.asarray(timestamps, dtype=np.float64)
    positive = np.diff(values)
    positive = positive[positive > 0.0]
    return float(np.median(positive)) if positive.size else 1.0 / 30.0


def _level4_6_payload(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = metadata.get("level4_6")
    if not isinstance(payload, Mapping):
        raise CorrectiveDemoError("metadata is missing the Level 4.6 payload.")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CorrectiveDemoError(f"{key} must be a mapping.")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CorrectiveDemoError(f"{key} must be a non-empty string.")
    return value
