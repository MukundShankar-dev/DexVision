"""Cross-skill replay qualification for Level 4 scripted experts."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexvision.logging.level4_collection import (
    GROUP_BY_PILOT_SKILL,
    WorkcellPilotTask,
    load_level4_collection_config,
)
from dexvision.logging.replay_demo import load_replay_demo, replay_loaded_demo
from dexvision.sim.workcell import WorkcellTask


EXPERT_AUDIT_VERSION = "level4/expert-replay-audit-v1"
REQUIRED_SOURCE_SKILLS = (
    "reach_object",
    "press_button",
    "push_object_to_target",
    "pick_object",
    "pick_place_sequence",
)
REQUIRED_DERIVED_SKILLS = (
    "reach_object",
    "press_button",
    "push_object_to_target",
    "pick_object",
    "place_held_object",
)
EXPECTED_PHASES = {
    "reach_object": ("approach",),
    "press_button": ("approach", "fixture_contact", "retract"),
    "push_object_to_target": ("approach", "push_contact", "settle", "retract"),
    "pick_object": ("approach", "acquire", "lift", "stabilize"),
    "pick_place_sequence": (
        "approach",
        "acquire",
        "lift",
        "stabilize",
        "transport",
        "place",
        "release",
        "settle",
        "retract",
    ),
}
REQUIRED_PROVENANCE_FIELDS = (
    "action_contract",
    "episode_id",
    "episode_schema_version",
    "goal_condition_id",
    "object_instance_ids",
    "operator_id",
    "phase_contract",
    "phase_intervals",
    "random_seed",
    "recording_session_id",
    "reset_state",
    "schema_versions",
    "source",
    "task_config",
    "typed_goal",
)


class Level4ExpertAuditError(ValueError):
    """Raised when an expert audit request is malformed."""


@dataclass(frozen=True)
class ExpertEpisodeAudit:
    """Replay and provenance evidence for one immutable episode."""

    episode_id: str
    episode_path: str
    source_skill: str
    derived_skills: tuple[str, ...]
    operator_success: bool
    schema_validation: bool
    complete_provenance: bool
    causal_phase_contract: bool
    timestamp_alignment: bool
    coverage_assignment: bool
    reset_metadata_match: bool
    headless_replay: bool
    terminal_metric_recomputation: bool
    recomputed_success: bool
    operator_label_agreement: bool
    safety_violation_count: int
    maximum_neighbor_disturbance_m: float
    neighbor_disturbance_limit_m: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    replayed_steps: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["derived_skills"] = list(self.derived_skills)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload


def audit_scripted_episode(
    episode_dir: str | Path,
    *,
    config_path: str | Path,
    workcell_config: str | Path,
) -> ExpertEpisodeAudit:
    """Replay one Level 4 episode and recompute its task result from metadata."""

    loaded = load_replay_demo(episode_dir)
    episode = loaded.episode
    metadata = episode.metadata
    episode_id = str(metadata.get("episode_id", ""))
    source_skill = str(metadata.get("skill_name", ""))
    operator_success = metadata.get("success") is True
    reasons: list[str] = []

    source_is_scripted = metadata.get("source") == "scripted"
    if not source_is_scripted:
        reasons.append("source_not_scripted")
    if source_skill not in REQUIRED_SOURCE_SKILLS:
        reasons.append("unsupported_source_skill")

    complete_provenance = _complete_provenance(episode)
    if not complete_provenance:
        reasons.append("incomplete_provenance")
    timestamp_alignment = _timestamp_alignment_passes(
        episode,
        config_path=config_path,
    )
    if not timestamp_alignment:
        reasons.append("timestamp_alignment")
    safety_violation_count = _safety_violation_count(episode)
    if safety_violation_count:
        reasons.append("safety_violation")
    causal_phase_contract = _causal_phase_contract_passes(episode, source_skill)
    if operator_success and not causal_phase_contract:
        reasons.append("causal_phase_contract")

    if source_skill not in REQUIRED_SOURCE_SKILLS:
        return ExpertEpisodeAudit(
            episode_id=episode_id,
            episode_path=str(Path(episode_dir)),
            source_skill=source_skill,
            derived_skills=(),
            operator_success=operator_success,
            schema_validation=True,
            complete_provenance=complete_provenance,
            causal_phase_contract=causal_phase_contract,
            timestamp_alignment=timestamp_alignment,
            coverage_assignment=False,
            reset_metadata_match=False,
            headless_replay=False,
            terminal_metric_recomputation=False,
            recomputed_success=False,
            operator_label_agreement=not operator_success,
            safety_violation_count=safety_violation_count,
            maximum_neighbor_disturbance_m=0.0,
            neighbor_disturbance_limit_m=0.0,
            accepted=False,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
            replayed_steps=0,
        )

    seed = metadata.get("random_seed")
    goal_condition_id = metadata.get("goal_condition_id")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Level4ExpertAuditError(
            f"episode {episode_id!r} random_seed must be an integer."
        )
    if not isinstance(goal_condition_id, str) or not goal_condition_id:
        raise Level4ExpertAuditError(
            f"episode {episode_id!r} goal_condition_id must be a string."
        )

    config, _ = load_level4_collection_config(config_path)
    disturbance_limit = _disturbance_limit(config, source_skill)
    cells = {
        str(cell["id"]): cell
        for cell in config["coverage_cells"]
        if isinstance(cell, Mapping)
    }
    coverage_assignment = _coverage_assignment_passes(
        metadata,
        source_skill=source_skill,
        cell=cells.get(goal_condition_id),
    )
    if not coverage_assignment:
        reasons.append("coverage_assignment")

    metric_successes: list[bool] = []
    derived_pick_successes: list[bool] = []
    maximum_disturbance = 0.0
    replayed_steps = 0
    headless_replay = False
    terminal_metric_recomputation = False
    with WorkcellPilotTask(
        workcell_config=workcell_config,
        dataset_config=config_path,
        skill_name=source_skill,
        goal_condition_id=goal_condition_id,
        seed=seed,
    ) as task:
        reset_metadata_match = _reset_matches_metadata(task, metadata)
        if not reset_metadata_match:
            reasons.append("reset_metadata_mismatch")
        metric, pick_metric = _metric_tasks(task, source_skill)
        target_object = _target_object_id(metadata)
        initial_positions = {
            spec.object_id: np.asarray(
                task.initial_world_state.require_entity(spec.object_id).position,
                dtype=np.float64,
            )
            for spec in task.workcell.config.objects
        }

        def observe(_step: object, _state: object) -> None:
            nonlocal maximum_disturbance
            world = task.workcell.get_world_state()
            metric_successes.append(metric.evaluate(world).success)
            if pick_metric is not None:
                derived_pick_successes.append(pick_metric.evaluate(world).success)
            for spec in task.workcell.config.objects:
                if spec.object_id == target_object:
                    continue
                position = np.asarray(
                    world.require_entity(spec.object_id).position,
                    dtype=np.float64,
                )
                maximum_disturbance = max(
                    maximum_disturbance,
                    float(np.linalg.norm(position[:2] - initial_positions[spec.object_id][:2])),
                )

        sim_steps = _saved_sim_steps(metadata)
        result = replay_loaded_demo(
            loaded,
            task.env,
            speed=1000.0,
            sim_steps_per_action=sim_steps,
            sleep_fn=lambda _delay: None,
            progress_callback=observe,
        )
        replayed_steps = result.steps_replayed
        headless_replay = replayed_steps == int(episode.timestamps.shape[0])
        terminal_metric_recomputation = bool(metric_successes)

    recomputed_success = _recomputed_source_success(
        source_skill,
        metric_successes,
        derived_pick_successes,
    )
    if not headless_replay:
        reasons.append("headless_replay")
    if not terminal_metric_recomputation or not recomputed_success:
        reasons.append("recomputed_task_failure")
    operator_label_agreement = operator_success == recomputed_success
    if not operator_label_agreement:
        reasons.append("operator_label_disagreement")
    if maximum_disturbance > disturbance_limit:
        reasons.append("neighbor_disturbance")
    if not operator_success:
        reasons.append("ordinary_task_failure")

    accepted = operator_success and not reasons
    derived_skills = _derived_skills(source_skill, derived_pick_successes)
    return ExpertEpisodeAudit(
        episode_id=episode_id,
        episode_path=str(Path(episode_dir)),
        source_skill=source_skill,
        derived_skills=derived_skills if accepted else (),
        operator_success=operator_success,
        schema_validation=True,
        complete_provenance=complete_provenance,
        causal_phase_contract=causal_phase_contract,
        timestamp_alignment=timestamp_alignment,
        coverage_assignment=coverage_assignment,
        reset_metadata_match=reset_metadata_match,
        headless_replay=headless_replay,
        terminal_metric_recomputation=terminal_metric_recomputation,
        recomputed_success=recomputed_success,
        operator_label_agreement=operator_label_agreement,
        safety_violation_count=safety_violation_count,
        maximum_neighbor_disturbance_m=maximum_disturbance,
        neighbor_disturbance_limit_m=disturbance_limit,
        accepted=accepted,
        rejection_reasons=() if accepted else tuple(dict.fromkeys(reasons)),
        replayed_steps=replayed_steps,
    )


def audit_expert_architecture(
    episode_dirs: Sequence[str | Path],
    *,
    config_path: str | Path,
    workcell_config: str | Path,
    minimum_repeats_per_source_skill: int | None = None,
) -> dict[str, Any]:
    """Audit repeated cross-skill episodes while retaining ordinary failures."""

    config, _ = load_level4_collection_config(config_path)
    frozen_audit = config["pilot"]["expert_architecture_audit"]
    frozen_minimum = int(frozen_audit["minimum_repeats_per_source_skill"])
    if minimum_repeats_per_source_skill is None:
        minimum_repeats_per_source_skill = frozen_minimum
    if minimum_repeats_per_source_skill != frozen_minimum:
        raise Level4ExpertAuditError(
            "minimum_repeats_per_source_skill must match the frozen config value "
            f"of {frozen_minimum}."
        )
    if not episode_dirs:
        raise Level4ExpertAuditError("expert architecture audit needs episode paths.")
    audits = tuple(
        audit_scripted_episode(
            path,
            config_path=config_path,
            workcell_config=workcell_config,
        )
        for path in episode_dirs
    )
    episode_ids = [item.episode_id for item in audits]
    if len(set(episode_ids)) != len(episode_ids):
        raise Level4ExpertAuditError(
            "expert architecture audit episode ids must be unique."
        )
    accepted = tuple(item for item in audits if item.accepted)
    source_counts = Counter(item.source_skill for item in accepted)
    derived_counts: Counter[str] = Counter()
    for item in accepted:
        derived_counts.update(item.derived_skills)
    missing_repeats = {
        skill: minimum_repeats_per_source_skill - source_counts[skill]
        for skill in REQUIRED_SOURCE_SKILLS
        if source_counts[skill] < minimum_repeats_per_source_skill
    }
    missing_derived = [
        skill for skill in REQUIRED_DERIVED_SKILLS if derived_counts[skill] == 0
    ]
    ordinary_failures = tuple(
        item for item in audits if not item.operator_success and not item.accepted
    )
    unexpected_rejections = tuple(
        item for item in audits if item.operator_success and not item.accepted
    )
    qualified = not missing_repeats and not missing_derived and not unexpected_rejections
    return {
        "version": EXPERT_AUDIT_VERSION,
        "minimum_repeats_per_source_skill": minimum_repeats_per_source_skill,
        "required_source_skills": list(REQUIRED_SOURCE_SKILLS),
        "required_derived_skills": list(REQUIRED_DERIVED_SKILLS),
        "episode_count": len(audits),
        "accepted_episode_count": len(accepted),
        "ordinary_failure_count": len(ordinary_failures),
        "unexpected_rejection_count": len(unexpected_rejections),
        "accepted_source_skill_counts": dict(sorted(source_counts.items())),
        "accepted_derived_skill_counts": dict(sorted(derived_counts.items())),
        "missing_source_skill_repeats": missing_repeats,
        "missing_derived_skills": missing_derived,
        "safety_violation_episode_count": sum(
            item.safety_violation_count > 0 for item in accepted
        ),
        "neighbor_disturbance_failure_count": sum(
            item.maximum_neighbor_disturbance_m
            > item.neighbor_disturbance_limit_m
            for item in accepted
        ),
        "ordinary_failure_episode_ids": [item.episode_id for item in ordinary_failures],
        "qualified": qualified,
        "episodes": [item.to_dict() for item in audits],
    }


def save_expert_audit_report(
    report: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Atomically save a generated audit report outside immutable episodes."""

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise Level4ExpertAuditError("expert audit report must use a .json extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _metric_tasks(
    task: WorkcellPilotTask, source_skill: str
) -> tuple[WorkcellTask, WorkcellTask | None]:
    goal = task.goal
    if source_skill == "pick_place_sequence":
        object_id = str(goal["object_id"])
        place = task.workcell.create_task(
            "place_held_object",
            object_id=object_id,
            target_id=str(goal["target_id"]),
        )
        pick = task.workcell.create_task("pick_object", object_id=object_id)
        return place, pick
    return task.workcell.create_task(source_skill, **goal), None


def _recomputed_source_success(
    source_skill: str,
    successes: Sequence[bool],
    derived_pick_successes: Sequence[bool],
) -> bool:
    if not successes:
        return False
    if source_skill in {"push_object_to_target", "pick_place_sequence"}:
        source_success = bool(successes[-1])
    else:
        source_success = any(successes)
    if source_skill == "pick_place_sequence":
        return source_success and any(derived_pick_successes)
    return source_success


def _derived_skills(
    source_skill: str, derived_pick_successes: Sequence[bool]
) -> tuple[str, ...]:
    if source_skill == "pick_place_sequence":
        if not any(derived_pick_successes):
            return ()
        return ("reach_object", "pick_object", "place_held_object")
    return (source_skill,)


def _complete_provenance(episode: object) -> bool:
    metadata = episode.metadata
    if any(field not in metadata for field in REQUIRED_PROVENANCE_FIELDS):
        return False
    if metadata.get("source") != "scripted":
        return False
    request_sources = episode.request_sources
    if request_sources is None or set(request_sources.astype(str).tolist()) != {
        "script"
    }:
        return False
    action_arrays = (
        episode.requested_actions,
        episode.commanded_actions,
        episode.applied_actions,
        episode.prior_commanded_actions,
        episode.prior_applied_actions,
    )
    if any(array is None for array in action_arrays):
        return False
    return bool(
        np.array_equal(episode.actions, episode.applied_actions)
        and np.array_equal(episode.requested_actions, episode.commanded_actions)
        and np.array_equal(episode.commanded_actions, episode.applied_actions)
    )


def _causal_phase_contract_passes(episode: object, source_skill: str) -> bool:
    expected = EXPECTED_PHASES.get(source_skill)
    phases = episode.online_phases
    if expected is None or phases is None or len(phases) == 0:
        return False
    observed = tuple(
        phase
        for index, phase in enumerate(phases.astype(str).tolist())
        if index == 0 or phase != str(phases[index - 1])
    )
    return observed == expected


def _timestamp_alignment_passes(
    episode: object, *, config_path: str | Path
) -> bool:
    config, _ = load_level4_collection_config(config_path)
    threshold = float(
        config["quality_thresholds"]["max_state_action_timestamp_skew_s"]
    )
    timestamps = np.asarray(episode.timestamps, dtype=np.float64)
    aligned = (
        episode.action_timestamps,
        episode.task_timestamps,
        episode.state_timestamps,
    )
    if any(values is None for values in aligned):
        return False
    return all(
        values.shape == timestamps.shape
        and np.all(np.isfinite(values))
        and float(np.max(np.abs(values - timestamps))) <= threshold
        for values in aligned
    )


def _safety_violation_count(episode: object) -> int:
    count = 0
    if episode.safety_masks is None:
        count += 1
    else:
        count += int(np.count_nonzero(episode.safety_masks))
    if episode.intervention_flags is None:
        count += 1
    else:
        count += int(np.count_nonzero(episode.intervention_flags))
    if episode.safety_reasons is None:
        count += 1
    else:
        count += sum(
            reason not in {"", "none"}
            for reason in episode.safety_reasons.astype(str).ravel()
        )
    return count


def _coverage_assignment_passes(
    metadata: Mapping[str, Any],
    *,
    source_skill: str,
    cell: Mapping[str, Any] | None,
) -> bool:
    if cell is None or cell.get("data_group") != GROUP_BY_PILOT_SKILL[source_skill]:
        return False
    goal = metadata.get("typed_goal")
    if not isinstance(goal, Mapping):
        return False
    comparisons = {
        "reach_object": (("entity_id", "entity_id"),),
        "pick_object": (("object_id", "object_id"),),
        "pick_place_sequence": (
            ("object_id", "object_id"),
            ("target_id", "target_id"),
        ),
        "push_object_to_target": (
            ("object_id", "object_id"),
            ("target_zone", "target_id"),
        ),
        "press_button": (("button_id", "button_id"),),
    }[source_skill]
    return all(goal.get(goal_key) == cell.get(cell_key) for goal_key, cell_key in comparisons)


def _reset_matches_metadata(
    task: WorkcellPilotTask, metadata: Mapping[str, Any]
) -> bool:
    reset_state = metadata.get("reset_state")
    if not isinstance(reset_state, Mapping):
        return False
    objects = reset_state.get("objects")
    if not isinstance(objects, Mapping):
        return False
    for spec in task.workcell.config.objects:
        raw = objects.get(spec.object_id)
        if not isinstance(raw, Mapping):
            return False
        entity = task.initial_world_state.require_entity(spec.object_id)
        if not np.allclose(raw.get("position_m"), entity.position, atol=1e-12, rtol=0.0):
            return False
        if not np.allclose(
            raw.get("orientation_wxyz"),
            entity.orientation_wxyz,
            atol=1e-12,
            rtol=0.0,
        ):
            return False
    return True


def _target_object_id(metadata: Mapping[str, Any]) -> str | None:
    goal = metadata.get("typed_goal")
    if isinstance(goal, Mapping) and isinstance(goal.get("object_id"), str):
        return str(goal["object_id"])
    return None


def _disturbance_limit(config: Mapping[str, Any], source_skill: str) -> float:
    key = {
        "reach_object": "scripted_reach",
        "press_button": "scripted_button",
        "push_object_to_target": "scripted_push",
        "pick_object": "scripted_grasp",
        "pick_place_sequence": "scripted_place",
    }[source_skill]
    return float(config["pilot"][key]["maximum_non_target_disturbance_m"])


def _saved_sim_steps(metadata: Mapping[str, Any]) -> int:
    recording = metadata.get("recording")
    value = recording.get("sim_steps_per_frame") if isinstance(recording, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Level4ExpertAuditError(
            "scripted expert metadata needs a positive recording sim cadence."
        )
    return value
