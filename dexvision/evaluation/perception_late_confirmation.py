"""Require late-phase and stabilization validation before another fresh test."""

from pathlib import Path
import json
import yaml
from dexvision.evaluation.perception_motion import selected_context
from dexvision.evaluation.perception_phase_confirmation import freeze_phase_confirmation


def freeze_late_confirmation(config, output_dir):
    spec = yaml.safe_load(Path(config).read_text())
    recipe, _ = selected_context(spec["selected_recipe"])
    report = json.loads((Path(recipe["model_dir"]) / "validation.json").read_text())
    if not report.get("late_phase_validation", {}).get("passed", False):
        raise ValueError(
            "Late release/settle validation must pass before confirmation."
        )
    return freeze_phase_confirmation(config, output_dir)
