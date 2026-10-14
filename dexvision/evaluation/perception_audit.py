"""Evaluation-only truth adapter and exact accepted-bundle observability audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dexvision.logging.visual_stream import digest_file
from dexvision.perception.object_observations import SIMULATOR_GROUND_TRUTH, make_object_observation


def truth_observation(annotation: dict, timestamp: float):
    """Retain invisible truth labels ONLY for evaluation/schema parity.

    Inference never calls this function or accepts its output as input.
    """
    quaternion = Rotation.from_matrix(annotation["pose_6d"]["rotation_matrix"]).as_quat()
    return make_object_observation(object_id=annotation["object_id"], class_id=annotation["class_id"],
        position=annotation["pose_6d"]["translation_m"], orientation_wxyz=quaternion[[3, 0, 1, 2]],
        linear_velocity=None, source=SIMULATOR_GROUND_TRUTH, confidence=1.,
        timestamp=timestamp, frame="mujoco_world")


def accepted_inventory(recipe: dict) -> dict:
    """Every selected neural field plus runtime/context/preview dependencies.

    Source categories describe candidate producers; conditional robot telemetry
    is not claimed to exist on hardware. All five perception tracks are blocked.
    """
    import torch

    result = {}
    for skill, reference in recipe["accepted_bundles"].items():
        receipt = json.loads(Path(reference["receipt"]).read_text(encoding="utf-8"))
        fields, schemas = [], []
        for run in receipt["runs"]:
            path = run.get("checkpoint_path", run.get("selected_checkpoint"))
            expected = run.get("checkpoint_sha256", run.get("selected_checkpoint_sha256"))
            if digest_file(Path(path)) != expected:
                raise ValueError(f"Accepted {skill} weights changed: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            schemas.append(checkpoint["schema"])
        if any(schema != schemas[0] for schema in schemas):
            raise ValueError(f"Accepted {skill} seeds have different feature schemas")
        for name in schemas[0]["input_names"]:
            fields.append(field_source(skill, name))
        dependencies = recipe["runtime_dependencies"][skill]
        fields.extend(dependencies)
        result[skill] = {"receipt": reference["receipt"], "selected_input_names": schemas[0]["input_names"],
                         "fields": fields, "compatibility": "unsupported",
                         "reason": reference["reason"],
                         "control_cadence_s": reference["control_cadence_s"],
                         "qualified_closed_loop": False}
    return result


def field_source(skill: str, name: str) -> dict:
    """Conservative lineage for each exact selected normalized input name."""
    source = "unsupported"
    reason = "Requires certified semantic body pose/entry context, unavailable from this RGB baseline"
    units = "dimensionless_after_exact_accepted_transform"
    frame = "world_then_accepted_task_or_hand_transform"
    if any(token in name for token in ("phase", "held", "supported", "contact", "succeeded", "pressed_once")):
        reason = "Physical relation/contact or causal terminal/dwell needs a declared measured producer; simulator truth prohibited"
    elif any(token in name for token in ("angular", "orientation_error", "entity.rotation")):
        reason = "Full orientation/angular motion unobservable with unmarked symmetric bodies and unknown tilt"
    elif any(token in name for token in ("command_height", "command_lift", "closure_fraction",
                                         "prior_applied_actions", "finger_joint", "robot.rotation")):
        source = "declared_robot_sensor_telemetry"
        reason = "Conditional on actual named encoder/command feedback, calibrated body origin and aligned timestamps; no deployed sensor established"
        frame = "robot_command_or_named_joint"
    elif name.startswith("class/") or name.startswith("entity.class."):
        source = "inferred_vision"
        reason = "Colour class hypothesis; semantic instance mapping and confidence calibration unsupported"
        frame = "semantic"
    elif "velocity" in name or "speed" in name:
        source = "inferred_vision"
        reason = "Elapsed-time finite-difference position hypothesis; uncertainty and size/occlusion bias remain, angular motion unsupported"
    elif name.startswith("goal.target_"):
        source = "frozen_calibration_geometry"
        reason = "Only explicit request/static target calibration may supply this; entity-derived goal requires certified perception"
    elif name.startswith("tracking"):
        source = "declared_robot_sensor_telemetry"
        reason = "Named forearm pose plus pinned weld/command feedback; entry-relative predicates reset per request, no sensor established"
    elif name.startswith("fixture_depth"):
        reason = "Measured +X fixture slide displacement needs fixture sensor; 2mm release rule not supplied by nominal RGB box"
    return {"field": name, "consumer": f"{skill}/selected_feature_transform",
            "producer": source, "availability": "unsupported_or_conditional",
            "units": units, "frame_body_origin": frame,
            "timestamp_rule": "acquisition-time pre-action; never saved future outcome",
            "confidence_validity": "require available field, confidence>=0.8, age<=0.1s; otherwise reject",
            "reset_history_rule": "reset at independent request; causal current/past only",
            "reason": reason}


def feature_parity_diagnostics(recipe: dict) -> dict:
    """Offline oracle tests exact accepted transforms, not inferred skill rollout.

    Compare batched and scalar paths on explicit synthetic allowed telemetry and
    separately supplied evaluation-only physical inputs. These oracles are never
    policy-visible inferred features. Missing fields block actual integration.
    """
    from dexvision.learning.contact_policy import contact_features
    from dexvision.learning.pick_policy import boundary_context, pick_features
    from dexvision.learning.place_policy import place_features
    from dexvision.learning.push_predicates import push_features
    from dexvision.learning.reach_repair import calibrated_features, transform_features
    from dexvision.learning.skill_datasets import feature_schema

    plan = yaml.safe_load(Path("configs/level5/learning.yaml").read_text(encoding="utf-8"))
    result = {}
    for skill, reference in recipe["accepted_bundles"].items():
        skill_config = yaml.safe_load(Path(reference["skill_config"]).read_text(encoding="utf-8"))
        names = list(feature_schema(plan, skill_config)[0])
        spec = yaml.safe_load(Path(reference["feature_recipe"]).read_text(encoding="utf-8"))
        source = np.zeros(len(names))
        for prefix in ("robot.rotation6d", "entity.rotation_6d", "goal.target_rotation"):
            for i in (0, 4):
                key = f"{prefix}.{i}"
                if key in names:
                    source[names.index(key)] = 1.
        source[names.index("prior_applied_actions.base_orientation_target/qw")] = 1.
        for j, value in enumerate((.10, .0, .02)):
            source[names.index(f"entity.position_m.{j}")] = value
        for prefix in ("goal.target_position", "goal.approach_position"):
            for j, value in enumerate((.15, .05, .06)):
                key = f"{prefix}.{j}"
                if key in names:
                    source[names.index(key)] = value
        context = boundary_context(source, names, spec) if skill == "pick" else None

        def transform(values):
            if skill == "reach":
                calibrated = calibrated_features(values, names, spec, {"calibrate_measured_frame": True})
                return transform_features(calibrated, names, "goal_plane_height")[0]
            if skill == "pick":
                return pick_features(values, names, context, spec,
                                     history={"closed_contact_dwell": 0.})[0]
            if skill == "place":
                return place_features(values, names, spec)[0]
            if skill == "push":
                return push_features(values, names, spec, fixture_rest=np.array([.1, 0, .02]),
                                     entry_source=source, qualified_once=False)[0]
            return contact_features(values, names, spec, fixture_rest=np.array([.11, -.11, .2]),
                                    pressed_once=False)[0]

        scalar = transform(source)
        batch = transform(np.stack((source, source)))
        error = float(np.max(np.abs(batch[0]-scalar)))
        sensitivity = {}
        for delta in recipe["diagnostics"]["position_biases_m"]:
            perturbed = source.copy()
            perturbed[names.index("entity.position_m.0")] += delta
            sensitivity[str(delta)] = float(np.max(np.abs(transform(perturbed)-scalar)))
        result[skill] = {"exact_transform_scalar_batch_max_error": error,
                         "feature_count": len(scalar), "position_bias_feature_max_change": sensitivity,
                         "inferred_feature_parity": "blocked_unavailable_fields",
                         "claim": "synthetic exact-transform evaluation oracle only"}
    # Explicit boundaries, not a claim of sensor accuracy.
    result["predicate_sensitivity"] = {
        "press_release_depth_m": .002, "push_tracking_envelope_m": .010,
        "push_speed_limit_mps": .020,
        "press_depth_bias_m": recipe["diagnostics"]["position_biases_m"],
        "fixture_depth_error": "unsupported_no_sensor_or_certified_fixture_pose",
        "contact_phase_disagreement": "unsupported_no_measured_contact_relations",
        "uncertainty_at_decision_boundaries": "unsupported_no_calibrated_uncertainty",
        "release_depth_bias_sweep": [
            {"true_depth_m": depth, "bias_m": bias,
             "disagrees": bool((depth <= .002) != (depth+bias <= .002))}
            for depth in (.001, .002, .003) for bias in (-.001, .001, -.010, .010, -.025, .025)],
        "push_speed_bias_sweep": [
            {"true_speed_mps": speed, "bias_mps": bias,
             "disagrees": bool((speed <= .020) != (abs(speed+bias) <= .020))}
            for speed in (.019, .020, .021) for bias in (-.005, .005)],
        "push_tracking_bias_sweep": [
            {"true_tracking_m": lag, "bias_m": bias,
             "disagrees": bool((lag <= .010) != (abs(lag+bias) <= .010))}
            for lag in (.009, .010, .011) for bias in (-.001, .001, -.010, .010, -.025, .025)],
        "sweep_claim": "evaluation-only synthetic boundary sensitivity, not measured sensor reliability",
    }
    return result
