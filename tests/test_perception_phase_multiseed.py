"""Reusing a completed fit must preserve TRAIN data, transform and full budget."""

import json
from pathlib import Path
import pytest
from dexvision.learning.perception_phase_multiseed import verify_reused_fit


def contracts(tmp_path):
    parent = {
        k: "same"
        for k in (
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
        )
    }
    parent.update(
        model_dir=str(tmp_path),
        _recipe_sha256="frozen",
        network_candidates=[
            {"seed": 3, "epochs": 3, "hidden": [2], "learning_rate": 0.01}
        ],
    )
    recipe = {
        **parent,
        "train_feature_cache": str(tmp_path / "train_examples.npz"),
        "reused_seed": 3,
        "reused_candidate": str(tmp_path / "candidate.json"),
        "network_candidates": [
            {**parent["network_candidates"][0], "seed": seed} for seed in (3, 7, 11)
        ],
    }
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "test_data_used_for_fitting_or_selection": False,
                "recipe_sha256": "frozen",
            }
        )
    )
    (tmp_path / "candidate.json").write_text(
        json.dumps(
            {
                "model": {
                    "body_center_models": {
                        "rigid_cuboid": {"seed": 3, "epochs": 3, "hidden": [2]}
                    }
                },
                "train_loss": [3, 2, 1],
            }
        )
    )
    return recipe, parent


def test_completed_fit_reuses_exact_full_budget(tmp_path):
    recipe, parent = contracts(tmp_path)
    assert len(verify_reused_fit(recipe, parent)["train_loss"]) == 3


@pytest.mark.parametrize(
    "failure",
    [
        "different_cache",
        "different_transform",
        "short_budget",
        "test_exposed",
        "duplicate_seed",
    ],
)
def test_reused_fit_rejects_invalid_binding(tmp_path, failure):
    recipe, parent = contracts(tmp_path)
    if failure == "different_cache":
        recipe["train_feature_cache"] = "other.npz"
    if failure == "different_transform":
        recipe["network_target"] = "other"
    if failure == "short_budget":
        p = Path(recipe["reused_candidate"])
        c = json.loads(p.read_text())
        c["train_loss"].pop()
        p.write_text(json.dumps(c))
    if failure == "test_exposed":
        p = tmp_path / "selection.json"
        c = json.loads(p.read_text())
        c["test_data_used_for_fitting_or_selection"] = True
        p.write_text(json.dumps(c))
    if failure == "duplicate_seed":
        recipe["network_candidates"][-1]["seed"] = 7
    with pytest.raises(ValueError):
        verify_reused_fit(recipe, parent)
