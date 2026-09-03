"""Export policy-free Level 2 metadata stubs for future skill cards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dexvision.sim.tasks import TaskSpec


SKILL_METADATA_SCHEMA_VERSION = "level2/skill-card-metadata-v1"
DEFAULT_SKILL_VERSION = "0.1.0"
DEFAULT_DATASET_SUMMARY = Path("data/demos/reports/summaries/dataset_summary.json")


class SkillMetadataError(ValueError):
    """Raised when a skill metadata stub cannot be built safely."""


@dataclass(frozen=True)
class SkillMetadataStub:
    """Policy-free task metadata that Level 3 can extend after training."""

    metadata_schema_version: str
    skill_name: str
    skill_version: str
    task_id: str
    observation_schema_version: str
    action_schema_version: str
    action_schema: Mapping[str, Mapping[str, Any]]
    parameter_schema: Mapping[str, Mapping[str, Any]]
    preconditions: tuple[str, ...]
    success_condition: str
    failure_conditions: tuple[str, ...]
    timeout: Mapping[str, Any]
    terminal_state_fields: Mapping[str, Mapping[str, Any]]
    dataset_summary_path: str
    dataset_summary: Mapping[str, Any]
    known_limitations: tuple[str, ...]
    policy_checkpoint: None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this stub."""

        return asdict(self)


_TASK_PRECONDITIONS: Mapping[str, tuple[str, ...]] = {
    "reach_touch_target": (
        "The configured reach target is available in the MuJoCo task scene.",
        "The robot begins from a reset state inside the configured workspace.",
        "Valid Level 1.13 hand-base and finger control observations are available.",
    ),
    "button_press": (
        "The selected button and target press state are available in the task scene.",
        "The button and robot begin from their saved reset states.",
        "Valid Level 1.13 hand-base and finger control observations are available.",
    ),
    "push_cube_to_target": (
        "The selected cube and target zone are available in the MuJoCo task scene.",
        "The cube and robot begin from their saved reset states.",
        "Valid Level 1.13 hand-base and finger control observations are available.",
    ),
}

_TASK_LIMITATIONS: Mapping[str, tuple[str, ...]] = {
    "reach_touch_target": (
        "The success metric is specific to simulated palm contact and target dwell.",
    ),
    "button_press": (
        "The success metric assumes the configured simulated button joint model.",
    ),
    "push_cube_to_target": (
        "The success metric uses planar cube distance and does not assess grasping.",
    ),
}

_COMMON_LIMITATIONS = (
    "This is a Level 2 metadata stub; no trained policy checkpoint is attached.",
    "The contract is validated for the local MuJoCo task board, not real hardware.",
)


def build_skill_metadata(
    task_spec: TaskSpec,
    *,
    dataset_summary_path: str | Path = DEFAULT_DATASET_SUMMARY,
    skill_version: str = DEFAULT_SKILL_VERSION,
    preconditions: Sequence[str] | None = None,
    known_limitations: Sequence[str] | None = None,
) -> SkillMetadataStub:
    """Build one validated metadata stub from a task spec and dataset summary."""

    if not skill_version.strip():
        raise SkillMetadataError("skill_version must be a non-empty string.")
    task_spec.action_schema.validate()
    summary_path = Path(dataset_summary_path)
    summary = _load_matching_summary(summary_path, task_spec=task_spec)
    action_schema = _action_schema_metadata(task_spec)
    parameter_schema = _parameter_schema_metadata(task_spec.parameter_schema)
    terminal_state_fields = _json_mapping(
        task_spec.terminal_state_schema,
        field_name="terminal_state_schema",
    )

    resolved_preconditions = tuple(
        preconditions
        if preconditions is not None
        else _TASK_PRECONDITIONS.get(
            task_spec.task_id,
            ("The task environment and required objects are reset and available.",),
        )
    )
    resolved_limitations = tuple(
        known_limitations
        if known_limitations is not None
        else (
            *_TASK_LIMITATIONS.get(task_spec.task_id, ()),
            *_COMMON_LIMITATIONS,
        )
    )
    _validate_text_items(resolved_preconditions, field_name="preconditions")
    _validate_text_items(resolved_limitations, field_name="known_limitations")
    if not task_spec.success_condition:
        raise SkillMetadataError("task spec must declare a success condition.")
    if not task_spec.failure_conditions:
        raise SkillMetadataError("task spec must declare failure conditions.")
    if task_spec.max_episode_steps <= 0:
        raise SkillMetadataError("task timeout must be a positive number of steps.")
    if "success" not in terminal_state_fields or "failure_reason" not in terminal_state_fields:
        raise SkillMetadataError(
            "terminal state must declare success and failure_reason fields."
        )

    return SkillMetadataStub(
        metadata_schema_version=SKILL_METADATA_SCHEMA_VERSION,
        skill_name=task_spec.skill_name,
        skill_version=skill_version,
        task_id=task_spec.task_id,
        observation_schema_version=task_spec.observation_schema.version,
        action_schema_version=task_spec.action_schema.version,
        action_schema=action_schema,
        parameter_schema=parameter_schema,
        preconditions=resolved_preconditions,
        success_condition=task_spec.success_condition,
        failure_conditions=tuple(task_spec.failure_conditions),
        timeout={
            "max_episode_steps": task_spec.max_episode_steps,
            "units": "control steps",
        },
        terminal_state_fields=terminal_state_fields,
        dataset_summary_path=str(summary_path),
        dataset_summary=summary,
        known_limitations=resolved_limitations,
    )


def save_skill_metadata(
    metadata: SkillMetadataStub,
    output_path: str | Path,
) -> Path:
    """Save one metadata stub as formatted JSON using an atomic replacement."""

    output = Path(output_path)
    if output.suffix.lower() != ".json":
        raise SkillMetadataError("skill metadata output must use a .json extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _load_matching_summary(
    path: Path,
    *,
    task_spec: TaskSpec,
) -> dict[str, Any]:
    if not path.is_file():
        raise SkillMetadataError(f"dataset summary does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillMetadataError(f"dataset summary is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise SkillMetadataError("dataset summary must contain a groups list.")
    matches = [
        group
        for group in payload["groups"]
        if isinstance(group, dict)
        and group.get("task_id") == task_spec.task_id
        and group.get("skill_name") == task_spec.skill_name
    ]
    if len(matches) != 1:
        raise SkillMetadataError(
            "dataset summary must contain exactly one group for "
            f"{task_spec.skill_name}/{task_spec.task_id}; found {len(matches)}."
        )
    group = matches[0]
    expected_versions = {
        "action_schema_version": task_spec.action_schema.version,
        "observation_schema_version": task_spec.observation_schema.version,
    }
    for field_name, expected in expected_versions.items():
        actual = group.get(field_name)
        if actual != expected:
            raise SkillMetadataError(
                f"dataset summary {field_name} '{actual}' does not match task spec "
                f"'{expected}'."
            )
    return {
        "version": payload.get("version"),
        "dataset": payload.get("dataset"),
        "num_episodes": group.get("num_episodes"),
        "clean_success_count": group.get("clean_success_count"),
        "level3_ready": group.get("level3_ready"),
    }


def _action_schema_metadata(task_spec: TaskSpec) -> dict[str, dict[str, Any]]:
    schema = task_spec.action_schema
    finger_start, finger_stop = schema.finger_actuator_targets
    finger_names = tuple(schema.representation_notes.get("finger_target_names", ()))
    if len(finger_names) != finger_stop - finger_start:
        raise SkillMetadataError(
            "action schema finger_target_names must match finger actuator columns."
        )
    return {
        "base_position_target": {
            "column_range": list(schema.base_position_target),
            "shape": [3],
            "type": "float64",
            "units": "metres",
            "coordinate_frame": "MuJoCo world",
        },
        "base_orientation_target": {
            "column_range": list(schema.base_orientation_target),
            "shape": [4],
            "type": "float64",
            "units": "unit quaternion",
            "coordinate_frame": "MuJoCo world, wxyz",
        },
        "finger_actuator_targets": {
            "column_range": list(schema.finger_actuator_targets),
            "shape": [finger_stop - finger_start],
            "type": "float64",
            "units": "model-defined actuator control units",
            "coordinate_frame": "MuJoCo actuator order",
            "names": list(finger_names),
        },
    }


def _parameter_schema_metadata(
    schema: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not schema:
        raise SkillMetadataError("task spec must declare a parameter schema.")
    result: dict[str, dict[str, Any]] = {}
    for parameter_name, raw_contract in schema.items():
        contract = _json_mapping(raw_contract, field_name=parameter_name)
        for required_field in ("type", "shape", "required"):
            if required_field not in contract:
                raise SkillMetadataError(
                    f"parameter '{parameter_name}' is missing {required_field}."
                )
        contract.setdefault("units", "unitless")
        contract.setdefault("coordinate_frame", "not_applicable")
        result[parameter_name] = contract
    return result


def _json_mapping(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise SkillMetadataError(f"{field_name} must be JSON serializable.") from exc


def _validate_text_items(values: Sequence[str], *, field_name: str) -> None:
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise SkillMetadataError(f"{field_name} must contain non-empty strings.")
