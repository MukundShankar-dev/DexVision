"""Require both full-sequence and dense stabilization validation before a test."""

from __future__ import annotations
import json
from pathlib import Path
import yaml
from dexvision.evaluation.perception_motion import (
    selected_context,
    freeze_context_confirmation,
)
from dexvision.learning.perception_context_phase import strict_validation


def freeze_phase_confirmation(config, output_dir):
    spec = yaml.safe_load(Path(config).read_text())
    recipe, _ = selected_context(spec["selected_recipe"])
    report = json.loads((Path(recipe["model_dir"]) / "validation.json").read_text())
    if not strict_validation(report) or not report.get(
        "additional_phase_validation", {}
    ).get("passed", False):
        raise ValueError(
            "Full-sequence and dense stabilization validation must pass before confirmation."
        )
    return freeze_context_confirmation(config, output_dir)
