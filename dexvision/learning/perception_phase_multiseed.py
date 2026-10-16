"""Frozen coverage amendment; reuse exact prior TRAIN features, never tests."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from dexvision.learning.perception_metric_network import fit_network
from dexvision.learning.perception_motion_examples import (
    load_context_recipe,
)
from dexvision.evaluation.perception_phase_queries import score_phase_queries
from dexvision.learning.perception_context_phase import strict_validation
from dexvision.learning.perception_development import (
    development_rows,
    load_repair_recipe,
)
from dexvision.evaluation.perception_motion import score_context
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.perception.context_motion import FEATURE_VERSION
from dexvision.perception.perception_baseline import (
    load_model,
    model_digest,
    save_artifact_index,
)
from dexvision.logging.visual_stream import digest_file, write_json


def verify_reused_fit(recipe, parent):
    """Reuse one completed full-budget fit only with exactly pinned TRAIN/transform."""
    if recipe["train_feature_cache"] != str(
        Path(parent["model_dir"]) / "train_examples.npz"
    ):
        raise ValueError("Reused candidate TRAIN cache mismatch.")
    for key in (
        "engine",
        "feature_count",
        "visual_root",
        "development_protocol",
        "source_manifest",
        "amendment_root",
        "amendment_protocol",
        "amendment_manifest",
        "predictive_tracker",
        "detector_fragment_gap_px",
        "minimum_confidence",
        "maximum_age_s",
        "detector_recipe",
        "cuboid_amendment_fraction",
        "network_input",
        "network_target",
        "training_objective",
    ):
        if recipe[key] != parent[key]:
            raise ValueError("Reused candidate contract mismatch: " + key)
    selection = json.loads((Path(parent["model_dir"]) / "selection.json").read_text())
    if (
        selection["test_data_used_for_fitting_or_selection"]
        or selection["recipe_sha256"] != parent["_recipe_sha256"]
    ):
        raise ValueError("Reused fit binding/test-isolation mismatch.")
    candidate = json.loads(Path(recipe["reused_candidate"]).read_text())
    spec = next(
        s for s in recipe["network_candidates"] if s["seed"] == recipe["reused_seed"]
    )
    if (
        spec not in parent["network_candidates"]
        or len(candidate["train_loss"]) != spec["epochs"]
    ):
        raise ValueError("Reused candidate did not complete the exact budget.")
    if len({s["seed"] for s in recipe["network_candidates"]}) != 3:
        raise ValueError("Exactly three distinct frozen training seeds required.")
    for model in candidate["model"]["body_center_models"].values():
        if any(model[k] != spec[k] for k in ("seed", "epochs", "hidden")):
            raise ValueError("Reused candidate weights differ from frozen fit.")
    return candidate


def train_coverage(config):
    recipe = load_context_recipe(config)
    rows = development_rows(recipe)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    parent = load_context_recipe(recipe["reused_recipe"])
    reuse = verify_reused_fit(recipe, parent)
    with np.load(recipe["train_feature_cache"], allow_pickle=False) as cached:
        examples = {
            c: tuple(
                cached[f"{c}_{name}"].copy()
                for name in ("features", "targets", "weights")
            )
            for c in CORE_CLASSES
        }
    base = load_repair_recipe(recipe["detector_recipe"])
    detector = {k: v for k, v in load_model(base).items() if k != "body_center_models"}
    for c, rule in detector["fragment_merge"].items():
        rule["maximum_gap_px"] = recipe["detector_fragment_gap_px"][c]
    current = {
        **recipe,
        "tracking_mode": "predictive",
        "tracker": detector["tracker_config"],
    }
    amended = {
        **current,
        "visual_root": recipe["amendment_root"],
        "development_protocol": recipe["amendment_protocol"],
        "source_manifest": recipe["amendment_manifest"],
    }
    write_json(
        output / "training_audit.json",
        {
            "exact_frozen_TRAIN_cache_sha256": digest_file(
                Path(recipe["train_feature_cache"])
            ),
            "no_validation_or_test_fit": True,
            "no_second_amendment_merge": True,
            "reused_candidate": recipe["reused_candidate"],
            "three_seeds": [s["seed"] for s in recipe["network_candidates"]],
        },
    )
    calibration = json.loads(
        (Path(recipe["visual_root"]) / "calibration.json").read_text()
    )
    candidates = []
    for spec in recipe["network_candidates"]:
        if spec["seed"] == recipe["reused_seed"]:
            model, history = reuse["model"], reuse["train_loss"]
        else:
            models, history = fit_network(examples, calibration=calibration, **spec)
            model = {
                **detector,
                "body_center_models": models,
                "feature_version": FEATURE_VERSION,
                "tracking_mode": "predictive",
                "method": "rgb_motion_world_height_mlp_metric_loss",
            }
        report = score_context(current, model, rows, "validation")
        auxiliary = score_phase_queries(amended, model)
        report["additional_phase_validation"] = auxiliary
        passes = strict_validation(report) and auxiliary["passed"]
        name = f"mlp_{'_'.join(map(str, spec['hidden']))}_s{spec['seed']}"
        write_json(
            output / f"{name}.json",
            {
                "model": model,
                "report": report,
                "strict_pass": passes,
                "train_loss": history,
            },
        )
        print(
            name,
            "strict=",
            passes,
            "primary_p95=",
            report["primary_object_metrics"]["position_error_p95_m"],
            flush=True,
        )
        worst = max(
            g["position_error_p95_m"] or 1.0
            for g in report["primary_class_phase"].values()
            if g["visible_eligible_instances"] >= 8
        )
        candidates.append(
            (
                int(passes),
                -worst,
                -report["primary_object_metrics"]["position_error_p95_m"],
                -len(candidates),
                model,
                report,
            )
        )
    selected = max(candidates, key=lambda r: r[:4])
    model, report = selected[-2:]
    report["strict_validation_pass"] = (
        strict_validation(report) and report["additional_phase_validation"]["passed"]
    )
    write_json(output / "selected.json", model)
    write_json(output / "validation.json", report)
    write_json(
        output / "selection.json",
        {
            "rule": recipe["candidate_selection"],
            "model_sha256": model_digest(model),
            "recipe_sha256": digest_file(Path(config)),
            "recipe_input_digests": recipe["input_digests"],
            "implementation_digests": recipe["implementation_digests"],
            "test_data_used_for_fitting_or_selection": False,
            "strict_validation_pass": report["strict_validation_pass"],
            "candidate_count": len(candidates),
        },
    )
    save_artifact_index(output)
    load_model(recipe)
    return report
