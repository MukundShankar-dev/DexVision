"""Read-only Level 4 pilot coverage and count-freeze reporting."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.level4_collection import (
    LEVEL4_CORE_GROUPS,
    LEVEL4_EPISODE_SOURCES,
    Level4CollectionError,
    PilotEpisode,
    discover_pilot_episodes,
    load_level4_collection_config,
    load_manual_replay_reviews,
    rejection_reason_counts,
)
from dexvision.logging.phase_labels import phase_disagreement_report
from dexvision.logging.session_manifest import SessionManifestError, load_session_manifest


COVERAGE_REPORT_VERSION = "level4/pilot-coverage-report-v3"
DEFAULT_SESSION_MANIFEST = "session_manifest.json"
DEFAULT_REPORT_NAME = "level4_coverage_report.json"
GROUP_BY_SKILL = {
    "reach_object": "reach",
    "pick_place_sequence": "pick_place",
    "push_object_to_target": "push",
    "press_button": "button",
    "rotate_dial": "dial",
}
REQUIRED_SKILLS = (
    "reach_object",
    "pick_object",
    "place_held_object",
    "push_object_to_target",
    "press_button",
)


class DatasetCoverageError(ValueError):
    """Raised when Level 4 coverage evidence is missing or inconsistent."""


def summarize_level4_coverage(
    *,
    config_path: str | Path,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    """Summarize pilot attempts without rewriting episodes or acceptance evidence."""

    try:
        config, protocol = load_level4_collection_config(config_path)
        episodes = discover_pilot_episodes(dataset_dir)
    except Level4CollectionError as exc:
        raise DatasetCoverageError(str(exc)) from exc
    root = Path(dataset_dir)
    cells = _coverage_cells(config)
    sessions, session_issues = _session_splits(root)
    issues = list(session_issues)
    accepted = tuple(episode for episode in episodes if episode.expert_accepted)
    accepted_by_group: Counter[str] = Counter()
    accepted_by_source: Counter[str] = Counter()
    accepted_by_cell_split: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    target_type_counts: Counter[str] = Counter()
    object_specs = _mapping(_mapping(config, "workcell"), "objects")
    target_specs = _mapping(_mapping(config, "workcell"), "targets")

    for episode in accepted:
        cell = cells.get(episode.goal_condition_id)
        if cell is None:
            issues.append(
                f"episode {episode.episode_id} references unknown coverage cell "
                f"{episode.goal_condition_id!r}"
            )
            continue
        group = GROUP_BY_SKILL.get(episode.skill_name)
        if group is None:
            issues.append(
                f"episode {episode.episode_id} has unsupported pilot skill "
                f"{episode.skill_name!r}"
            )
            continue
        if cell.get("data_group") != group:
            issues.append(
                f"episode {episode.episode_id} skill maps to {group!r}, but cell "
                f"{episode.goal_condition_id!r} belongs to {cell.get('data_group')!r}"
            )
            continue
        session_split = sessions.get(episode.session_id)
        if session_split is None:
            issues.append(
                f"episode {episode.episode_id} session {episode.session_id!r} is absent "
                f"from {DEFAULT_SESSION_MANIFEST}"
            )
            continue
        if session_split != cell.get("split_owner"):
            issues.append(
                f"episode {episode.episode_id} session split {session_split!r} does not "
                f"match cell owner {cell.get('split_owner')!r}"
            )
            continue
        source = episode.source
        if source not in LEVEL4_EPISODE_SOURCES:
            issues.append(
                f"episode {episode.episode_id} has unknown source {source!r}"
            )
            continue
        if source != cell.get("required_source"):
            issues.append(
                f"episode {episode.episode_id} source {source!r} does not match "
                f"cell requirement {cell.get('required_source')!r}"
            )
            continue
        accepted_by_group[group] += 1
        accepted_by_source[source] += 1
        accepted_by_cell_split[episode.goal_condition_id][session_split] += 1
        object_id = _episode_object_id(episode)
        if object_id is not None and object_id in object_specs:
            family_counts[str(object_specs[object_id]["family"])] += 1
        target_id = _episode_target_id(episode)
        if target_id is not None and target_id in target_specs:
            target_type_counts[str(target_specs[target_id]["target_type"])] += 1

    segment_counts = _segment_counts(accepted, issues=issues)
    replayed_skills = _replayed_success_skills(accepted)
    manual_reviews = load_manual_replay_reviews(root)
    manual_skills = _manual_replay_skills(
        accepted,
        manual_reviews=manual_reviews,
        issues=issues,
    )
    phase_summary = _phase_agreement(episodes)
    episode_count_requirements = {
        group: {
            "accepted": int(accepted_by_group[group]),
            "minimum": int(minimum),
            "passed": int(accepted_by_group[group]) >= int(minimum),
        }
        for group, minimum in protocol.accepted_by_group.items()
    }
    accepted_session_ids = sorted({episode.session_id for episode in accepted})
    required_families = sorted(_mapping(_mapping(config, "workcell"), "object_families"))
    required_target_types = sorted(
        {str(spec["target_type"]) for spec in target_specs.values()}
    )
    matrix = _coverage_matrix_summary(
        config,
        accepted_by_cell_split=accepted_by_cell_split,
    )
    storage = _storage_summary(config, episodes=episodes, accepted=accepted)
    source_mix = _source_mix_summary(config, accepted_by_source=accepted_by_source)
    core_collection = _core_collection_summary(
        config,
        episodes=episodes,
        sessions=sessions,
        cells=cells,
    )
    phase_limit = float(
        _mapping(config, "quality_thresholds")[
            "max_phase_annotation_disagreement_fraction"
        ]
    )
    phase_passed = (
        phase_summary["audited_frame_count"] > 0
        and phase_summary["disagreement_fraction"] <= phase_limit
    )
    protocol_passed = bool(
        all(item["passed"] for item in episode_count_requirements.values())
        and len(accepted_session_ids) >= protocol.minimum_genuine_sessions
        and set(required_families) <= set(family_counts)
        and set(required_target_types) <= set(target_type_counts)
        and set(REQUIRED_SKILLS) <= set(replayed_skills)
        and phase_passed
        and matrix["fits_required_envelope"]
        and storage["payload_handling"] != "undetermined_no_pilot_data"
        and not issues
    )
    manual_passed = set(REQUIRED_SKILLS) <= set(manual_skills)
    return {
        "version": COVERAGE_REPORT_VERSION,
        "config_path": str(Path(config_path)),
        "dataset_dir": str(root),
        "pilot_status": (
            "manual_verification_required"
            if protocol_passed and not manual_passed
            else "complete" if protocol_passed and manual_passed else "incomplete"
        ),
        "attempt_episode_count": len(episodes),
        "expert_accepted_episode_count": len(accepted),
        "ordinary_failure_episode_count": sum(
            1 for episode in episodes if episode.metadata.get("success") is False
        ),
        "unreviewed_episode_count": sum(1 for episode in episodes if episode.review is None),
        "episode_counts_by_group": episode_count_requirements,
        "segment_counts_by_skill": dict(sorted(segment_counts.items())),
        "accepted_session_ids": accepted_session_ids,
        "genuine_session_requirement": {
            "observed": len(accepted_session_ids),
            "minimum": protocol.minimum_genuine_sessions,
            "passed": len(accepted_session_ids) >= protocol.minimum_genuine_sessions,
        },
        "object_family_counts": dict(sorted(family_counts.items())),
        "missing_object_families": sorted(set(required_families) - set(family_counts)),
        "target_type_counts": dict(sorted(target_type_counts.items())),
        "missing_target_types": sorted(set(required_target_types) - set(target_type_counts)),
        "replayed_recomputed_success_skills": replayed_skills,
        "missing_replayed_success_skills": sorted(set(REQUIRED_SKILLS) - set(replayed_skills)),
        "manual_replay_verified_skills": manual_skills,
        "missing_manual_replay_skills": sorted(set(REQUIRED_SKILLS) - set(manual_skills)),
        "phase_label_agreement": {
            **phase_summary,
            "maximum_disagreement_fraction": phase_limit,
            "passed": phase_passed,
        },
        "rejection_reasons": rejection_reason_counts(episodes),
        "observed_safety_problems": _safety_reason_counts(episodes),
        "acceptance_evidence": _acceptance_evidence(episodes),
        "collection_time": _collection_time(accepted),
        "storage": storage,
        "source_mix": source_mix,
        "coverage_matrix": matrix,
        "level4_4_core_collection": core_collection,
        "optional_dial_decision": protocol.optional_dial_decision,
        "issues": sorted(set(issues)),
        "automated_pilot_requirements_passed": protocol_passed,
        "manual_replay_gate_passed": manual_passed,
        "checkpoint_complete": protocol_passed and manual_passed,
    }


def save_coverage_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically save a generated report outside immutable episode contents."""

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise DatasetCoverageError("coverage report output must use a .json extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _coverage_cells(config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    raw = config.get("coverage_cells")
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise DatasetCoverageError("coverage_cells must be a sequence.")
    return {str(cell["id"]): cell for cell in raw if isinstance(cell, Mapping)}


def _session_splits(root: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    path = root / DEFAULT_SESSION_MANIFEST
    if not path.exists():
        return {}, (f"missing session manifest: {path}",)
    try:
        manifest = load_session_manifest(path)
    except SessionManifestError as exc:
        return {}, (str(exc),)
    return (
        {session.recording_session_id: session.split for session in manifest.sessions},
        (),
    )


def _segment_counts(
    accepted: Sequence[PilotEpisode],
    *,
    issues: list[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for episode in accepted:
        if episode.skill_name == "pick_place_sequence":
            phases = [
                str(item.get("phase"))
                for item in episode.metadata.get("phase_intervals", [])
                if isinstance(item, Mapping)
            ]
            required = {
                "approach",
                "acquire",
                "lift",
                "stabilize",
                "transport",
                "place",
                "release",
                "settle",
            }
            if not required <= set(phases):
                issues.append(
                    f"accepted pick/place episode {episode.episode_id} lacks complete phases"
                )
                continue
            counts.update(("reach_object", "pick_object", "place_held_object"))
        elif episode.skill_name in REQUIRED_SKILLS:
            counts[episode.skill_name] += 1
    return counts


def _replayed_success_skills(accepted: Sequence[PilotEpisode]) -> list[str]:
    skills: set[str] = set()
    for episode in accepted:
        review = episode.review
        if review is None or not review.headless_replay or not review.recomputed_success:
            continue
        if episode.skill_name == "pick_place_sequence":
            skills.update(("reach_object", "pick_object", "place_held_object"))
        elif episode.skill_name in REQUIRED_SKILLS:
            skills.add(episode.skill_name)
    return sorted(skills)


def _manual_replay_skills(
    accepted: Sequence[PilotEpisode],
    *,
    manual_reviews: Sequence[Any],
    issues: list[str],
) -> list[str]:
    accepted_ids = {episode.episode_id for episode in accepted}
    skills: set[str] = set()
    for review in manual_reviews:
        if review.episode_id not in accepted_ids:
            issues.append(
                f"manual replay references non-accepted episode {review.episode_id!r}"
            )
            continue
        if review.passed:
            skills.update(review.verified_skills)
    return sorted(skills)


def _phase_agreement(episodes: Sequence[PilotEpisode]) -> Mapping[str, Any]:
    audited = 0
    disagreements = 0
    for episode in episodes:
        online_path = episode.path / "online_phases.npy"
        audited_path = episode.path / "audited_phases.npy"
        if not online_path.exists() or not audited_path.exists():
            continue
        try:
            online = np.load(online_path, allow_pickle=False).astype(str).tolist()
            audited_values = np.load(audited_path, allow_pickle=False).astype(str).tolist()
            report = phase_disagreement_report(online, audited_values)
        except (OSError, ValueError) as exc:
            raise DatasetCoverageError(
                f"could not compute phase agreement for {episode.path}: {exc}"
            ) from exc
        audited += int(report["audited_frame_count"])
        disagreements += int(report["disagreement_count"])
    return {
        "audited_frame_count": audited,
        "disagreement_count": disagreements,
        "disagreement_fraction": disagreements / audited if audited else 0.0,
    }


def _coverage_matrix_summary(
    config: Mapping[str, Any],
    *,
    accepted_by_cell_split: Mapping[str, Counter[str]],
) -> Mapping[str, Any]:
    cells = _coverage_cells(config)
    rows = []
    frozen_total = 0
    for cell_id, cell in cells.items():
        minima = _mapping(cell, "minimum_accepted_by_split")
        minimum_total = sum(int(minima.get(split, 0)) for split in ("train", "validation", "test"))
        frozen_total += minimum_total
        observed = accepted_by_cell_split.get(cell_id, Counter())
        rows.append(
            {
                "cell_id": cell_id,
                "data_group": cell.get("data_group"),
                "required_source": cell.get("required_source"),
                "split_owner": cell.get("split_owner"),
                "minimum": minimum_total,
                "observed": sum(observed.values()),
                "complete": all(
                    observed[split] >= int(minima.get(split, 0))
                    for split in ("train", "validation", "test")
                ),
            }
        )
    budget = _mapping(config, "episode_budget")
    minimum = int(budget["required_total_minimum"])
    maximum = int(budget["required_total_planning_maximum"])
    return {
        "status": _mapping(config, "freeze").get("coverage_count_status"),
        "cell_count": len(rows),
        "minimum_episode_total": frozen_total,
        "required_envelope": [minimum, maximum],
        "fits_required_envelope": minimum <= frozen_total <= maximum,
        "complete_cell_count": sum(1 for row in rows if row["complete"]),
        "cells": rows,
    }


def _core_collection_summary(
    config: Mapping[str, Any],
    *,
    episodes: Sequence[PilotEpisode],
    sessions: Mapping[str, str],
    cells: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Evaluate only the Level 4.4 reach/push/button haul."""

    core = _mapping(config, "level4_4_core_collection")
    minimum_sessions = _mapping(core, "minimum_sessions_by_split")
    accepted_by_cell_split: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_by_session: Counter[str] = Counter()
    accepted_by_group: Counter[str] = Counter()
    accepted_by_source: Counter[str] = Counter()
    observed_targets: dict[str, Counter[str]] = defaultdict(Counter)
    frozen_targets: dict[str, Counter[str]] = defaultdict(Counter)
    issues: list[str] = []
    attempt_count = 0
    rejected_or_failed_count = 0
    nonrequired_source_count = 0

    core_cells = {
        cell_id: cell
        for cell_id, cell in cells.items()
        if cell.get("data_group") in LEVEL4_CORE_GROUPS
    }
    for cell in core_cells.values():
        group = str(cell["data_group"])
        split = str(cell["split_owner"])
        minimum = int(_mapping(cell, "minimum_accepted_by_split")[split])
        frozen_targets[group][_core_target_id(cell)] += minimum

    held_out_objects = set(_mapping(config, "split_policy")["held_out_object_instances"])
    held_out_goals = set(_mapping(config, "split_policy")["held_out_goal_regions"])
    for episode in episodes:
        group = GROUP_BY_SKILL.get(episode.skill_name)
        if group not in LEVEL4_CORE_GROUPS:
            continue
        attempt_count += 1
        cell = core_cells.get(episode.goal_condition_id)
        if cell is None:
            issues.append(
                f"core episode {episode.episode_id} references non-core or unknown cell "
                f"{episode.goal_condition_id!r}"
            )
            continue
        split = sessions.get(episode.session_id)
        if split is None:
            issues.append(
                f"core episode {episode.episode_id} session {episode.session_id!r} "
                "is absent from the session manifest"
            )
            continue
        if split != cell.get("split_owner"):
            issues.append(
                f"core episode {episode.episode_id} session split {split!r} does not "
                f"match cell owner {cell.get('split_owner')!r}"
            )
            continue
        if episode.source != cell.get("required_source"):
            # Source requirements are minima, not a reason for preserved optional
            # evidence to invalidate the haul. Do not count the episode toward
            # its cell, source minimum, or accepted core total.
            nonrequired_source_count += 1
            continue
        object_id = _episode_object_id(episode)
        target_id = _episode_target_id(episode)
        if split != "test" and (
            object_id in held_out_objects or target_id in held_out_goals
        ):
            issues.append(
                f"core episode {episode.episode_id} leaks a held-out object or goal "
                f"into split {split!r}"
            )
            continue
        if episode.review is None:
            issues.append(
                f"core episode {episode.episode_id} has no append-only review evidence"
            )
            continue
        if not episode.expert_accepted:
            rejected_or_failed_count += 1
            continue
        accepted_by_cell_split[episode.goal_condition_id][split] += 1
        accepted_by_session[episode.session_id] += 1
        accepted_by_group[group] += 1
        accepted_by_source[episode.source] += 1
        observed_targets[group][_core_target_id(cell)] += 1

    matrix = _core_matrix_summary(
        core_cells,
        accepted_by_cell_split=accepted_by_cell_split,
    )
    accepted_total = sum(accepted_by_group.values())
    session_ids_by_split = {
        split: sorted(
            session_id
            for session_id in accepted_by_session
            if sessions.get(session_id) == split
        )
        for split in ("train", "validation", "test")
    }
    session_requirement = {
        split: {
            "observed": len(session_ids_by_split[split]),
            "minimum": int(minimum_sessions[split]),
            "passed": len(session_ids_by_split[split]) >= int(minimum_sessions[split]),
        }
        for split in ("train", "validation", "test")
    }
    maximum_session_fraction = float(core["maximum_single_session_fraction"])
    session_shares = {
        session_id: count / accepted_total if accepted_total else 0.0
        for session_id, count in sorted(accepted_by_session.items())
    }
    session_balance_passed = bool(
        accepted_total
        and all(share <= maximum_session_fraction for share in session_shares.values())
    )
    target_balance = _core_target_balance(
        observed=observed_targets,
        frozen=frozen_targets,
        tolerance=float(core["maximum_target_share_delta_from_frozen_minimum"]),
    )
    expected_source = Counter()
    for cell in core_cells.values():
        split = str(cell["split_owner"])
        expected_source[str(cell["required_source"])] += int(
            _mapping(cell, "minimum_accepted_by_split")[split]
        )
    source_requirements = {
        source: {
            "observed": int(accepted_by_source[source]),
            "minimum": int(expected_source[source]),
            "passed": int(accepted_by_source[source]) >= int(expected_source[source]),
        }
        for source in LEVEL4_EPISODE_SOURCES
        if expected_source[source]
    }
    restrictions = _mapping(config, "split_policy")
    test_isolation_passed = bool(
        restrictions.get("normalization_source") == "train_only"
        and restrictions.get("test_influences_tuning") is False
        and restrictions.get("test_influences_thresholds") is False
        and restrictions.get("test_influences_checkpoint_selection") is False
        and not any("held-out" in issue for issue in issues)
    )
    complete = bool(
        accepted_total >= int(core["required_accepted_episodes"])
        and matrix["complete"]
        and all(item["passed"] for item in session_requirement.values())
        and all(item["passed"] for item in source_requirements.values())
        and session_balance_passed
        and target_balance["passed"]
        and test_isolation_passed
        and not issues
    )
    return {
        "version": core["version"],
        "status": "complete" if complete else "incomplete",
        "attempt_episode_count": attempt_count,
        "accepted_episode_count": accepted_total,
        "required_accepted_episodes": int(core["required_accepted_episodes"]),
        "accepted_by_group": dict(sorted(accepted_by_group.items())),
        "source_requirements": source_requirements,
        "coverage_matrix": matrix,
        "session_ids_by_split": session_ids_by_split,
        "session_requirements": session_requirement,
        "session_shares": session_shares,
        "maximum_single_session_fraction": maximum_session_fraction,
        "session_balance_passed": session_balance_passed,
        "target_balance": target_balance,
        "test_isolation_passed": test_isolation_passed,
        "rejected_or_failed_auditable_count": rejected_or_failed_count,
        "nonrequired_source_episode_count": nonrequired_source_count,
        "issues": sorted(set(issues)),
        "checkpoint_complete": complete,
    }


def _core_matrix_summary(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    accepted_by_cell_split: Mapping[str, Counter[str]],
) -> Mapping[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell_id, cell in cells.items():
        split = str(cell["split_owner"])
        minimum = int(_mapping(cell, "minimum_accepted_by_split")[split])
        observed = int(accepted_by_cell_split.get(cell_id, Counter())[split])
        rows.append(
            {
                "cell_id": cell_id,
                "data_group": cell["data_group"],
                "required_source": cell["required_source"],
                "split_owner": split,
                "minimum": minimum,
                "observed": observed,
                "complete": observed >= minimum,
            }
        )
    return {
        "cell_count": len(rows),
        "complete_cell_count": sum(1 for row in rows if row["complete"]),
        "minimum_episode_total": sum(row["minimum"] for row in rows),
        "complete": all(row["complete"] for row in rows),
        "cells": rows,
    }


def _core_target_balance(
    *,
    observed: Mapping[str, Counter[str]],
    frozen: Mapping[str, Counter[str]],
    tolerance: float,
) -> Mapping[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in LEVEL4_CORE_GROUPS:
        frozen_total = sum(frozen[group].values())
        observed_total = sum(observed[group].values())
        for target_id, frozen_count in sorted(frozen[group].items()):
            frozen_share = frozen_count / frozen_total
            observed_share = (
                observed[group][target_id] / observed_total if observed_total else 0.0
            )
            rows.append(
                {
                    "data_group": group,
                    "target_id": target_id,
                    "frozen_minimum_share": frozen_share,
                    "observed_share": observed_share,
                    "maximum_share": min(1.0, frozen_share + tolerance),
                    "passed": observed_total > 0
                    and observed_share <= min(1.0, frozen_share + tolerance),
                }
            )
    return {"tolerance": tolerance, "passed": all(row["passed"] for row in rows), "rows": rows}


def _core_target_id(cell: Mapping[str, Any]) -> str:
    for field in ("target_id", "entity_id", "button_id"):
        value = cell.get(field)
        if isinstance(value, str) and value:
            return value
    raise DatasetCoverageError(
        f"core coverage cell {cell.get('id')!r} has no target identity"
    )


def _source_mix_summary(
    config: Mapping[str, Any],
    *,
    accepted_by_source: Counter[str],
) -> Mapping[str, Any]:
    source_mix = _mapping(config, "source_mix")
    minima = _mapping(source_mix, "minimum_accepted_by_source")
    rows = {
        source: {
            "observed": int(accepted_by_source[source]),
            "minimum": int(minima[source]),
            "passed": int(accepted_by_source[source]) >= int(minima[source]),
        }
        for source in LEVEL4_EPISODE_SOURCES
    }
    return {
        "version": source_mix.get("version"),
        "provenance_must_remain_separate": bool(
            source_mix.get("provenance_must_remain_separate")
        ),
        "sources": rows,
        "complete": all(item["passed"] for item in rows.values()),
    }


def _storage_summary(
    config: Mapping[str, Any],
    *,
    episodes: Sequence[PilotEpisode],
    accepted: Sequence[PilotEpisode],
) -> Mapping[str, Any]:
    pilot = _mapping(config, "pilot")
    storage = _mapping(pilot, "storage_projection")
    sample = accepted if accepted else episodes
    total = sum(episode.size_bytes for episode in sample)
    average = total / len(sample) if sample else 0.0
    planning_maximum = int(_mapping(config, "episode_budget")["required_total_planning_maximum"])
    projected = int(round(average * planning_maximum))
    threshold = int(storage["git_lfs_max_projected_payload_bytes"])
    return {
        "sample_episode_count": len(sample),
        "sample_total_bytes": total,
        "mean_bytes_per_episode": average,
        "bytes_by_episode_id": {
            episode.episode_id: episode.size_bytes for episode in sample
        },
        "planning_maximum_episode_count": planning_maximum,
        "projected_payload_bytes": projected,
        "git_lfs_max_projected_payload_bytes": threshold,
        "payload_handling": (
            "undetermined_no_pilot_data"
            if not sample
            else "git_lfs" if projected <= threshold else "external_object_storage"
        ),
    }


def _collection_time(accepted: Sequence[PilotEpisode]) -> Mapping[str, Any]:
    total_minutes = sum(episode.duration_seconds for episode in accepted) / 60.0
    return {
        "accepted_episode_minutes": total_minutes,
        "minutes_per_accepted_episode": (
            total_minutes / len(accepted) if accepted else None
        ),
    }


def _acceptance_evidence(episodes: Sequence[PilotEpisode]) -> Mapping[str, int]:
    return {
        "headless_replay_pass_count": sum(
            1
            for episode in episodes
            if episode.review is not None and episode.review.headless_replay
        ),
        "terminal_metric_recomputation_pass_count": sum(
            1
            for episode in episodes
            if episode.review is not None
            and episode.review.terminal_metric_recomputation
            and episode.review.recomputed_success
        ),
    }


def _safety_reason_counts(episodes: Sequence[PilotEpisode]) -> Mapping[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        path = episode.path / "safety_reasons.npy"
        if not path.exists():
            continue
        try:
            reasons = np.load(path, allow_pickle=False).astype(str).ravel()
        except (OSError, ValueError) as exc:
            raise DatasetCoverageError(f"could not read safety reasons {path}: {exc}") from exc
        counts.update(reason for reason in reasons if reason not in {"", "none"})
    return dict(sorted(counts.items()))


def _episode_object_id(episode: PilotEpisode) -> str | None:
    goal = episode.metadata.get("typed_goal")
    if isinstance(goal, Mapping) and isinstance(goal.get("object_id"), str):
        return str(goal["object_id"])
    ids = episode.metadata.get("object_instance_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, str) and ids:
        return str(ids[0])
    return None


def _episode_target_id(episode: PilotEpisode) -> str | None:
    goal = episode.metadata.get("typed_goal")
    if not isinstance(goal, Mapping):
        return None
    for key in ("target_id", "target_zone", "entity_id"):
        value = goal.get(key)
        if isinstance(value, str):
            return value
    return None


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DatasetCoverageError(f"Level 4 config {key} must be a mapping.")
    return value
