"""Shared frozen Level 4 scripted-expert settings; collection and approach setup agree."""
from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from dexvision.logging.level4_collection import WorkcellPilotTask


def resolve_scripted_expert_settings(
    task: WorkcellPilotTask,
    settings_key: str,
    *,
    enforce_frozen_cell_owner: bool = True,
) -> dict[str, Any]:
    """Resolve checkpoint-local expert overrides without changing older pilots."""

    settings = dict(task.collection_config["pilot"][settings_key])
    if not enforce_frozen_cell_owner:
        return settings
    checkpoint_blocks = [task.collection_config.get("level4_4_core_collection", {})]
    if task.skill_name == "pick_place_sequence":
        checkpoint_blocks.append(
            task.collection_config.get("level4_5a_pick_place_collection", {})
        )
    if task.procedural_variation:
        checkpoint_blocks.append(
            task.collection_config.get("level4_5b_procedural_expansion", {})
        )
    for checkpoint in checkpoint_blocks:
        if not isinstance(checkpoint, Mapping):
            continue
        overrides = checkpoint.get("scripted_expert_overrides", {})
        if isinstance(overrides, Mapping):
            raw = overrides.get(settings_key, {})
            if isinstance(raw, Mapping):
                settings = _merge_nested_settings(settings, raw)
    if settings_key == "scripted_grasp":
        settings = _resolve_instance_acquisition_hold_steps(settings, task)
        settings = _resolve_instance_grasp_template(settings, task)
        if task.procedural_variation:
            settings = _resolve_procedural_grasp_yaw_gain(settings, task)
        settings = _resolve_family_fallback_templates(settings)
    elif settings_key == "scripted_place":
        settings = _resolve_instance_place_target_offset(settings, task)
    if task.procedural_variation:
        settings = _apply_procedural_controller_offset(settings, task, settings_key)
    return settings



def _resolve_instance_acquisition_hold_steps(
    settings: Mapping[str, Any], task: WorkcellPilotTask
) -> dict[str, Any]:
    """Resolve an optional object-specific stable-contact dwell."""

    resolved = deepcopy(dict(settings))
    raw_instances = resolved.pop("instance_acquisition_hold_steps", {})
    if not isinstance(raw_instances, Mapping):
        raise ValueError(
            "scripted_grasp.instance_acquisition_hold_steps must be a mapping."
        )
    object_id = str(task.goal.get("object_id", ""))
    raw_steps = raw_instances.get(object_id)
    if raw_steps is not None:
        if isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps <= 0:
            raise ValueError(
                f"scripted grasp acquisition dwell for {object_id!r} must be positive."
            )
        resolved["required_acquisition_hold_steps"] = raw_steps
    return resolved



def _resolve_family_fallback_templates(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand partial fallback overrides against each resolved family template."""

    resolved = deepcopy(dict(settings))
    raw_fallbacks = resolved.pop("family_fallback_overrides", {})
    if not isinstance(raw_fallbacks, Mapping):
        raise ValueError("scripted_grasp.family_fallback_overrides must be a mapping.")
    raw_families = resolved.get("family_templates")
    if not isinstance(raw_families, Mapping):
        raise ValueError("scripted_grasp.family_templates must be a mapping.")
    fallback_templates: dict[str, list[dict[str, Any]]] = {}
    for family, raw_overrides in raw_fallbacks.items():
        base = raw_families.get(family)
        if not isinstance(base, Mapping):
            raise ValueError(f"missing scripted grasp family template {family!r}.")
        if not isinstance(raw_overrides, Sequence) or isinstance(raw_overrides, str):
            raise ValueError(
                f"scripted grasp fallbacks for {family!r} must be a sequence."
            )
        expanded: list[dict[str, Any]] = []
        for raw_override in raw_overrides:
            if not isinstance(raw_override, Mapping):
                raise ValueError(
                    f"scripted grasp fallback for {family!r} must be a mapping."
                )
            expanded.append(_merge_nested_settings(base, raw_override))
        fallback_templates[str(family)] = expanded
    resolved["family_fallback_templates"] = fallback_templates
    return resolved



def _resolve_procedural_grasp_yaw_gain(
    settings: Mapping[str, Any], task: WorkcellPilotTask
) -> dict[str, Any]:
    """Apply the train-qualified cuboid wrist correction schedule."""

    resolved = deepcopy(dict(settings))
    object_id = str(task.goal.get("object_id", ""))
    spec = next(
        item for item in task.workcell.config.objects if item.object_id == object_id
    )
    if spec.family != "cuboid":
        return resolved
    orientation = np.asarray(
        task.initial_world_state.require_entity(object_id).orientation_wxyz,
        dtype=float,
    )
    w, x, y, z = orientation
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    procedural = task.collection_config["level4_5b_procedural_expansion"]
    yaw_gain = procedural["cuboid_yaw_gain_schedule"]
    threshold = float(yaw_gain["high_absolute_yaw_threshold_rad"])
    if abs(yaw) >= threshold:
        ratio = float(yaw_gain["high_absolute_gain_ratio"])
    elif yaw > 0.0:
        ratio = float(yaw_gain["moderate_positive_gain_ratio"])
    else:
        return resolved
    families = deepcopy(dict(resolved["family_templates"]))
    cuboid = dict(families["cuboid"])
    cuboid["negative_object_yaw_to_wrist_yaw_gain"] = (
        float(cuboid["negative_object_yaw_to_wrist_yaw_gain"]) * ratio
    )
    families["cuboid"] = cuboid
    resolved["family_templates"] = families
    return resolved



def _apply_procedural_controller_offset(
    settings: Mapping[str, Any],
    task: WorkcellPilotTask,
    settings_key: str,
) -> dict[str, Any]:
    """Apply the saved sub-millimetre controller perturbation to one expert."""

    resolved = deepcopy(dict(settings))
    offset = np.asarray(
        task.procedural_variation["controller_position_offset_m"], dtype=float
    )
    if settings_key == "scripted_button":
        for key in ("precontact_offset_m", "press_offset_m"):
            resolved[key] = (np.asarray(resolved[key], dtype=float) + offset).tolist()
    elif settings_key == "scripted_push":
        resolved["approach_gap_m"] = float(resolved["approach_gap_m"]) + float(
            offset[0]
        )
        families = deepcopy(dict(resolved["family_parameters"]))
        for name, raw in families.items():
            values = dict(raw)
            values["fingertip_forward_offset_m"] = float(
                values["fingertip_forward_offset_m"]
            ) + float(offset[0])
            values["control_height_m"] = float(values["control_height_m"]) + float(
                offset[2]
            )
            families[name] = values
        resolved["family_parameters"] = families
        resolved["fingertip_lateral_offset_m"] = float(
            resolved["fingertip_lateral_offset_m"]
        ) + float(offset[1])
    elif settings_key == "scripted_place":
        offsets = deepcopy(dict(resolved["family_target_offset_xy_m"]))
        for name, raw in offsets.items():
            offsets[name] = (np.asarray(raw, dtype=float) + offset[:2]).tolist()
        resolved["family_target_offset_xy_m"] = offsets
    return resolved



def _merge_nested_settings(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge checkpoint-local expert settings."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_nested_settings(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged



def _resolve_instance_grasp_template(
    settings: Mapping[str, Any], task: WorkcellPilotTask
) -> dict[str, Any]:
    """Merge an optional object-specific grasp point into its family template."""

    resolved = deepcopy(dict(settings))
    raw_instances = resolved.pop("instance_templates", {})
    if not isinstance(raw_instances, Mapping):
        raise ValueError("scripted_grasp.instance_templates must be a mapping.")
    goal = task.goal
    object_id = str(goal.get("object_id", ""))
    raw_override = raw_instances.get(object_id)
    if raw_override is None:
        return resolved
    if not isinstance(raw_override, Mapping):
        raise ValueError(
            f"scripted grasp override for {object_id!r} must be a mapping."
        )
    family = next(
        spec.family for spec in task.workcell.config.objects if spec.object_id == object_id
    )
    raw_families = resolved.get("family_templates")
    if not isinstance(raw_families, Mapping):
        raise ValueError("scripted_grasp.family_templates must be a mapping.")
    family_template = raw_families.get(family)
    if not isinstance(family_template, Mapping):
        raise ValueError(f"missing scripted grasp family template {family!r}.")
    family_templates = deepcopy(dict(raw_families))
    family_templates[family] = _merge_nested_settings(
        family_template, raw_override
    )
    resolved["family_templates"] = family_templates
    return resolved



def _resolve_instance_place_target_offset(
    settings: Mapping[str, Any], task: WorkcellPilotTask
) -> dict[str, Any]:
    """Resolve an optional object-specific placement-center correction."""

    resolved = deepcopy(dict(settings))
    raw_instances = resolved.pop("instance_target_offset_xy_m", {})
    if not isinstance(raw_instances, Mapping):
        raise ValueError(
            "scripted_place.instance_target_offset_xy_m must be a mapping."
        )
    object_id = str(task.goal.get("object_id", ""))
    raw_offset = raw_instances.get(object_id)
    if raw_offset is None:
        return resolved
    family = next(
        spec.family for spec in task.workcell.config.objects if spec.object_id == object_id
    )
    raw_offsets = resolved.get("family_target_offset_xy_m")
    if not isinstance(raw_offsets, Mapping):
        raise ValueError(
            "scripted_place.family_target_offset_xy_m must be a mapping."
        )
    offsets = deepcopy(dict(raw_offsets))
    offsets[family] = deepcopy(raw_offset)
    resolved["family_target_offset_xy_m"] = offsets
    return resolved

