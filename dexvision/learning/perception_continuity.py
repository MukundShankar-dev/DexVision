"""One predeclared identity amendment; frozen selected position weights."""

from pathlib import Path
import json

from dexvision.learning.perception_development import load_repair_recipe, development_rows
from dexvision.learning.perception_context import strict_validation
from dexvision.evaluation.perception_dual_queries import score_query_cache
from dexvision.logging.visual_stream import write_json, digest_file
from dexvision.perception.continuity_localization import ENGINE
from dexvision.perception.perception_baseline import model_digest, save_artifact_index


def load_identity_recipe(path):
    recipe = load_repair_recipe(path)
    if recipe["engine"] != ENGINE or recipe["feature_count"] != 706:
        raise ValueError("Frozen current-RGB identity contract required.")
    return recipe


def select_identity(config):
    from dexvision.evaluation.perception_continuity import score_context

    recipe = load_identity_recipe(config)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    parent = json.loads(Path(recipe["frozen_selected_model"]).read_text())
    model = {
        **parent, "feature_version": ENGINE, "inference_variant": ENGINE,
        "identity_method": "existing_track_continuity; reject uncertain same-class births; reacquire after frozen gap",
    }
    assert model_digest(model["body_center_models"]) == recipe["active_weights_sha256"]
    rows = development_rows(recipe)
    if {r["split"] for r in rows} != {"train", "validation"}:
        raise ValueError("Repair accepts separately frozen TRAIN/validation only.")
    print("V27 exact V25 weights; full ordered validation with identity coverage", flush=True)
    report = score_context(
        {**recipe, "tracker": model["tracker_config"]}, model, rows, "validation"
    )
    query = score_query_cache(
        recipe, {**model, "current_frame_models": model["body_center_models"]},
        force_current=True,
    )
    passed = strict_validation(report) and query["passed"] and report["identity_coverage"]["passed"]
    report.update(
        strict_validation_pass=passed, current_frame_phase_validation=query,
        physical_motion_available=False,
    )
    write_json(output / "selected.json", model)
    write_json(output / "validation.json", report)
    write_json(output / "selection.json", {
        "model_sha256": model_digest(model), "recipe_sha256": digest_file(Path(config)),
        "recipe_input_digests": recipe["input_digests"],
        "implementation_digests": recipe["implementation_digests"],
        "strict_validation_pass": passed, "candidate_count": 1,
        "rule": recipe["candidate_selection"], "neural_weight_updates": 0,
        "active_weights_sha256": model_digest(model["body_center_models"]),
        "parent_selected_model_sha256": model_digest(parent),
        "test_data_used_for_fitting_or_selection": False,
        "physical_motion_available": False,
    })
    save_artifact_index(output)
    print("V27 validation strict:", passed, "identity:", report["identity_coverage"], flush=True)
    return report
