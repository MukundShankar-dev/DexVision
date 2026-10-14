"""Deterministic training-only palette fit and validation-only selection."""

from __future__ import annotations

from pathlib import Path

from dexvision.evaluation.evaluate_perception import score_split
from dexvision.evaluation.perception_audit import accepted_inventory, feature_parity_diagnostics
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.perception_baseline import (
    load_recipe, model_digest, save_artifact_index, training_palette, visual_rows,
)


def train_perception(config: str | Path) -> dict:
    """No neural training/randomness: one median fit, all frozen hue candidates."""
    recipe = load_recipe(config)
    if recipe["version"] in {"level5/perception-repair-v3", "level5/perception-repair-v4", "level5/perception-repair-v5", "level5/perception-repair-v6", "level5/perception-repair-v7", "level5/perception-repair-v8"}:
        from dexvision.learning.perception_repair import train_repair
        return train_repair(config, recipe)
    rows = visual_rows(recipe)
    directory = Path(recipe["model_dir"])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/"field_source_inventory.json", accepted_inventory(recipe))
    write_json(directory/"feature_diagnostics.json", feature_parity_diagnostics(recipe))
    fitted = training_palette(recipe, rows)
    candidates = []
    for margin in recipe["candidate_hue_margins"]:
        model = {**fitted, "hue_margin": margin, "method": "training_hue_saturation_background_histogram"}
        report = score_split(recipe, model, rows, "validation", detector_only=True)
        write_json(directory/f"candidate_margin_{margin}.json", {"model": model, "report": report})
        worst = min(report["breakdowns"]["class_id"][c]["recall"] or 0.
                    for c in recipe["core_classes"])
        candidates.append((worst, report["recall"], -report["false_positive_components"], -margin, model))
    selected = max(candidates, key=lambda item: item[:4])[-1]
    write_json(directory/"selected.json", selected)
    write_json(directory/"selection.json", {"rule": recipe["candidate_selection"],
        "model_sha256": model_digest(selected), "recipe_input_digests": recipe["input_digests"],
        "recipe_sha256": digest_file(Path(config)), "implementation_digests": recipe["implementation_digests"],
        "test_data_used_for_fitting_or_selection": False, "training_seeds": "not_applicable_deterministic_median_no_optimizer"})
    report = score_split(recipe, selected, rows, "validation")
    write_json(directory/"validation.json", report)
    save_artifact_index(directory)
    return {"model_dir": directory.as_posix(), "selected_hue_margin": selected["hue_margin"],
            "validation_numerical_gates": report["numerical_gates"],
            "test_started": False, "baseline_qualified": False}
