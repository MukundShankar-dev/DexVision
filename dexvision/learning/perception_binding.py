"""Add a redundant input binding omitted by the frozen neural report writer.

No weights, inference, ranking or validation metrics change. The original report
is retained before the strict shared loader's required metadata is added.
"""

from __future__ import annotations
import json
from pathlib import Path
from dexvision.learning.perception_development import load_repair_recipe
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.perception_baseline import (
    load_model,
    model_digest,
    save_artifact_index,
)


def bind_selection(config: str | Path) -> dict:
    recipe = load_repair_recipe(config)
    root = Path(recipe["model_dir"])
    path = root / "selection.json"
    original = path.read_bytes()
    selection = json.loads(original)
    model = json.loads((root / "selected.json").read_text())
    if (
        selection["recipe_sha256"] != recipe["_recipe_sha256"]
        or selection["implementation_digests"] != recipe["implementation_digests"]
        or selection["model_sha256"] != model_digest(model)
    ):
        raise ValueError(
            "Frozen model/recipe/code binding mismatch; cannot repair metadata."
        )
    if "recipe_input_digests" in selection:
        load_model(recipe)
        receipt = root / "selection_binding.json"
        if not receipt.exists():
            raise ValueError("Already bound; no metadata repair required.")
        return json.loads(receipt.read_text())
    archive = root / "selection_before_binding.json"
    with archive.open("xb") as handle:
        handle.write(original)
    selection["recipe_input_digests"] = recipe["input_digests"]
    write_json(path, selection)
    loaded = load_model(recipe)
    assert model_digest(loaded) == model_digest(model)
    receipt = {
        "metadata_only": True,
        "weights_unchanged": True,
        "ranking_and_validation_unchanged": True,
        "model_sha256": model_digest(model),
        "recipe_sha256": recipe["_recipe_sha256"],
        "original_selection_sha256": digest_file(archive),
        "bound_selection_sha256": digest_file(path),
        "binding_implementation_sha256": digest_file(Path(__file__)),
        "fitting_or_reselection": False,
        "test_data_used": False,
    }
    write_json(root / "selection_binding.json", receipt)
    save_artifact_index(root)
    return receipt
