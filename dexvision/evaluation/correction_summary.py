"""Read-only Level 4.6 failure/correction coverage and provenance audit."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.corrective_demos import (
    LEVEL4_6_EPISODES_PER_CELL,
    UNSAFE_FAILURE_CLASSES,
    CorrectiveDemoError,
    build_level4_6_plan,
    discover_level4_6_episodes,
    load_level4_6_quarantine,
    load_manual_correction_replays,
    select_level4_training_streams,
    validate_level4_6_metadata,
)
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.logging.session_manifest import load_session_manifest
from dexvision.sim.workcell import DEFAULT_WORKCELL_CONFIG


CORRECTION_SUMMARY_VERSION = "level4/correction-summary-v1"
DEFAULT_REPORT_NAME = "level4_correction_summary.json"
_PREFIX_ARRAYS = (
    "features.npy",
    "actions.npy",
    "robot_states.npy",
    "object_states.npy",
    "task_states.npy",
    "tracking_quality.npy",
    "timestamps.npy",
    "requested_actions.npy",
    "commanded_actions.npy",
    "applied_actions.npy",
    "prior_commanded_actions.npy",
    "prior_applied_actions.npy",
    "safety_masks.npy",
    "safety_reasons.npy",
    "request_sources.npy",
    "online_phases.npy",
    "audited_phases.npy",
    "phase_relevance_masks.npy",
    "intervention_flags.npy",
    "failure_reasons.npy",
    "action_timestamps.npy",
    "task_timestamps.npy",
    "state_timestamps.npy",
)


def summarize_corrections(
    *,
    config_path: str | Path,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    """Audit all Level 4.6 slots without mutating episode or review evidence."""

    root = Path(dataset_dir)
    plan = build_level4_6_plan(config_path)
    expected = {
        (item.coverage_cell_id, item.repetition): item for item in plan
    }
    episodes = discover_level4_6_episodes(root)
    sessions = {
        item.recording_session_id: item.split
        for item in load_session_manifest(root / "session_manifest.json").sessions
    }
    issues: list[str] = []
    observed: dict[tuple[str, int], Any] = {}
    by_cell: Counter[str] = Counter()
    by_stream: Counter[str] = Counter()
    by_failure_class: Counter[str] = Counter()
    generation_seeds: list[int] = []
    source_ids_by_cell: dict[str, set[str]] = defaultdict(set)
    initial_digests_by_cell: dict[str, set[str]] = defaultdict(set)
    provenance_failures = 0
    episode_by_id = {item.episode_id: item for item in episodes}

    for episode in episodes:
        try:
            validate_level4_6_metadata(episode.metadata)
        except CorrectiveDemoError as exc:
            provenance_failures += 1
            issues.append(f"episode {episode.episode_id}: {exc}")
            continue
        payload = episode.metadata["level4_6"]
        assert isinstance(payload, Mapping)
        cell_id = str(payload["coverage_cell_id"])
        repetition = payload.get("repetition")
        if not isinstance(repetition, int) or isinstance(repetition, bool):
            issues.append(f"episode {episode.episode_id}: repetition must be an integer")
            continue
        key = (cell_id, repetition)
        assignment = expected.get(key)
        if assignment is None:
            issues.append(f"episode {episode.episode_id}: unexpected Level 4.6 slot {key}")
            continue
        if key in observed:
            issues.append(f"duplicate Level 4.6 slot {key}")
            continue
        observed[key] = episode
        if episode.metadata.get("source") != assignment.source:
            issues.append(f"episode {episode.episode_id}: source does not match cell")
        if episode.metadata.get("failure_class") != assignment.failure_class:
            issues.append(f"episode {episode.episode_id}: failure class does not match cell")
        if episode.metadata.get("retryability") != assignment.retryability:
            issues.append(f"episode {episode.episode_id}: retryability does not match cell")
        if sessions.get(episode.session_id) != assignment.split:
            issues.append(f"episode {episode.episode_id}: session split does not match cell")
        if episode.metadata.get("success") is not assignment.is_correction:
            issues.append(f"episode {episode.episode_id}: success conflates stream outcome")
        by_cell[cell_id] += 1
        by_stream[str(episode.metadata["data_stream"])] += 1
        by_failure_class[str(episode.metadata["failure_class"])] += 1
        generation_seeds.append(int(payload["generation_seed"]))
        source_ids_by_cell[cell_id].add(str(episode.metadata["source_episode_id"]))
        initial_digests_by_cell[cell_id].add(
            str(episode.metadata.get("initial_state_digest", ""))
        )

    missing = sorted(set(expected) - set(observed))
    for key in missing:
        issues.append(f"missing Level 4.6 slot {key}")

    prefix_checks = 0
    prefix_failures = 0
    split_alignment_failures = 0
    outcome_checks = 0
    outcome_failures = 0
    for key, episode in observed.items():
        assignment = expected[key]
        if not assignment.is_correction:
            continue
        source_id = str(episode.metadata["source_episode_id"])
        source = episode_by_id.get(source_id)
        if source is None:
            issues.append(f"correction {episode.episode_id}: source failure {source_id!r} missing")
            continue
        if source.metadata.get("data_stream") != "ordinary_failure":
            issues.append(f"correction {episode.episode_id}: source is not ordinary failure")
        if source.metadata.get("failure_class") != assignment.failure_class:
            issues.append(f"correction {episode.episode_id}: source failure class mismatch")
        if sessions.get(source.session_id) != sessions.get(episode.session_id):
            split_alignment_failures += 1
            issues.append(f"correction {episode.episode_id}: source/correction split mismatch")
        interval = episode.metadata["intervention_interval"]
        assert isinstance(interval, list)
        prefix_checks += 1
        if not _prefix_matches(source.path, episode.path, int(interval[0])):
            prefix_failures += 1
            issues.append(f"correction {episode.episode_id}: pre-intervention prefix changed")
        outcome_checks += 1
        outcome = recompute_corrected_pick_place_outcome(
            episode.path,
            dataset_dir=root,
            config_path=config_path,
        )
        if not outcome["corrected_success"]:
            outcome_failures += 1
            issues.append(
                f"correction {episode.episode_id}: independent final outcome failed"
            )

    unsafe_abort_only = True
    for episode in observed.values():
        if episode.metadata.get("failure_class") not in UNSAFE_FAILURE_CLASSES:
            continue
        if (
            episode.metadata.get("retryability") != "abort"
            or episode.metadata.get("intervention_interval") is not None
            or episode.metadata.get("success") is not False
        ):
            unsafe_abort_only = False
            issues.append(f"unsafe episode {episode.episode_id}: not abort-only")

    paths = tuple(item.path for item in episodes)
    exclude_first = select_level4_training_streams(paths, include_corrections=False)
    exclude_second = select_level4_training_streams(paths, include_corrections=False)
    include_first = select_level4_training_streams(paths, include_corrections=True)
    include_second = select_level4_training_streams(paths, include_corrections=True)
    deterministic_selection = (
        exclude_first == exclude_second and include_first == include_second
    )
    complete_cells = sum(
        count == LEVEL4_6_EPISODES_PER_CELL for count in by_cell.values()
    )
    per_cell_unique_sources = all(
        len(source_ids_by_cell.get(item.coverage_cell_id, set()))
        == LEVEL4_6_EPISODES_PER_CELL
        for item in plan[::LEVEL4_6_EPISODES_PER_CELL]
    )
    per_cell_unique_initial_states = all(
        "" not in initial_digests_by_cell.get(item.coverage_cell_id, set())
        and len(initial_digests_by_cell.get(item.coverage_cell_id, set()))
        == LEVEL4_6_EPISODES_PER_CELL
        for item in plan[::LEVEL4_6_EPISODES_PER_CELL]
    )
    quarantined = load_level4_6_quarantine(root)
    manual_reviews = load_manual_correction_replays(root)
    active_correction_ids = {
        item.episode_id
        for item in observed.values()
        if item.metadata.get("data_stream") == "corrective_intervention"
    }
    manual_verified_ids = sorted(
        str(item["episode_id"])
        for item in manual_reviews
        if item.get("passed") is True and item.get("episode_id") in active_correction_ids
    )
    manual_passed = bool(manual_verified_ids)
    automated = bool(
        len(observed) == 120
        and complete_cells == 12
        and by_stream == {"ordinary_failure": 90, "corrective_intervention": 30}
        and len(set(generation_seeds)) == 120
        and per_cell_unique_sources
        and per_cell_unique_initial_states
        and prefix_checks == 30
        and prefix_failures == 0
        and outcome_checks == 30
        and outcome_failures == 0
        and split_alignment_failures == 0
        and unsafe_abort_only
        and deterministic_selection
        and not issues
    )
    checkpoint_complete = automated and manual_passed
    return {
        "version": CORRECTION_SUMMARY_VERSION,
        "config_path": str(Path(config_path)),
        "dataset_dir": str(root),
        "status": (
            "complete"
            if checkpoint_complete
            else "manual_verification_required" if automated else "incomplete"
        ),
        "episode_count": len(observed),
        "required_episode_count": 120,
        "complete_cell_count": complete_cells,
        "required_cell_count": 12,
        "episodes_by_cell": dict(sorted(by_cell.items())),
        "episodes_by_stream": dict(sorted(by_stream.items())),
        "episodes_by_failure_class": dict(sorted(by_failure_class.items())),
        "unique_generation_seed_count": len(set(generation_seeds)),
        "per_cell_unique_source_episodes": per_cell_unique_sources,
        "per_cell_unique_initial_states": per_cell_unique_initial_states,
        "conditional_provenance_passed": provenance_failures == 0,
        "quarantined_diagnostic_episode_count": len(quarantined),
        "quarantined_diagnostic_episode_ids": sorted(quarantined),
        "pre_intervention_prefix": {
            "checked": prefix_checks,
            "failures": prefix_failures,
            "passed": prefix_checks == 30 and prefix_failures == 0,
        },
        "independent_outcome_recomputation": {
            "checked": outcome_checks,
            "failures": outcome_failures,
            "passed": outcome_checks == 30 and outcome_failures == 0,
        },
        "unsafe_failures_abort_only": unsafe_abort_only,
        "deterministic_include_exclude": deterministic_selection,
        "identical_split_comparison_ready": split_alignment_failures == 0,
        "issues": sorted(set(issues)),
        "automated_requirements_passed": automated,
        "manual_replay_verified_episode_ids": manual_verified_ids,
        "manual_replay_gate_passed": manual_passed,
        "manual_replay_required": not manual_passed,
        "checkpoint_complete": checkpoint_complete,
    }


def recompute_corrected_pick_place_outcome(
    episode_dir: str | Path,
    *,
    dataset_dir: str | Path,
    config_path: str | Path,
    workcell_config: str | Path = DEFAULT_WORKCELL_CONFIG,
) -> dict[str, bool]:
    """Replay one correction and independently recompute pick/place success."""

    root = Path(dataset_dir)
    correction = load_logged_demo(episode_dir)
    source_failure = _episode_path_by_id(
        root, str(correction.metadata["source_episode_id"])
    )
    failure = load_logged_demo(source_failure)
    nominal_path = _episode_path_by_id(
        root, str(failure.metadata["source_episode_id"])
    )
    nominal = load_logged_demo(nominal_path)
    metadata = nominal.metadata
    procedural = metadata.get("task_config", {}).get("procedural_variation")
    loaded = load_replay_demo(episode_dir)
    pick_successes: list[bool] = []
    place_successes: list[bool] = []
    with WorkcellPilotTask(
        workcell_config=workcell_config,
        dataset_config=config_path,
        skill_name=str(metadata["skill_name"]),
        goal_condition_id=str(metadata["goal_condition_id"]),
        seed=int(metadata["random_seed"]),
        procedural_variation=procedural,
    ) as task:
        if task._pick_task is None or task._place_task is None:
            raise CorrectiveDemoError(
                "Level 4.6 corrective outcome audit requires pick/place tasks."
            )

        def observe(_step: object, _state: object) -> None:
            world = task.workcell.get_world_state()
            assert task._pick_task is not None and task._place_task is not None
            pick_successes.append(task._pick_task.evaluate(world).success)
            place_successes.append(task._place_task.evaluate(world).success)

        recording = metadata.get("recording")
        sim_steps = (
            int(recording["sim_steps_per_frame"])
            if isinstance(recording, Mapping)
            else 1
        )
        replay_loaded_demo(
            loaded,
            task.env,
            speed=1000.0,
            sim_steps_per_action=sim_steps,
            reset_env=False,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
    pick_success = any(pick_successes)
    final_place_success = bool(place_successes and place_successes[-1])
    return {
        "pick_success": pick_success,
        "final_place_success": final_place_success,
        "corrected_success": pick_success and final_place_success,
    }


def save_correction_summary(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write a generated report outside immutable episode directories."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _prefix_matches(source_dir: Path, correction_dir: Path, prefix_count: int) -> bool:
    if prefix_count <= 0:
        return False
    for filename in _PREFIX_ARRAYS:
        source_path = source_dir / filename
        correction_path = correction_dir / filename
        if source_path.exists() != correction_path.exists():
            return False
        if not source_path.exists():
            continue
        source = np.load(source_path, allow_pickle=False)
        correction = np.load(correction_path, allow_pickle=False)
        if correction.shape[0] < prefix_count or source.shape[0] != prefix_count:
            return False
        if not np.array_equal(source, correction[:prefix_count]):
            return False
    return True


def _episode_path_by_id(dataset_dir: Path, episode_id: str) -> Path:
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
            f"expected one linked episode {episode_id!r}, found {len(matches)}."
        )
    return matches[0]
