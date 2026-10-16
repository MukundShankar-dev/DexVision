"""Metadata completion preserves selected weights, metrics and original report."""

import json
import pytest
from dexvision.perception.perception_baseline import model_digest
from dexvision.learning.perception_binding import bind_selection


def setup_binding(tmp_path, monkeypatch):
    import dexvision.learning.perception_binding as binding

    model = {"immutable_weights": [1.0, 2.0, 3.0]}
    recipe = {
        "model_dir": str(tmp_path),
        "_recipe_sha256": "recipe",
        "implementation_digests": {"source": "digest"},
        "input_digests": {"data": "digest"},
    }
    selection = {
        "model_sha256": model_digest(model),
        "recipe_sha256": "recipe",
        "implementation_digests": {"source": "digest"},
    }
    (tmp_path / "selected.json").write_text(json.dumps(model))
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (tmp_path / "validation.json").write_text('{"unchanged":true}')
    monkeypatch.setattr(binding, "load_repair_recipe", lambda _: recipe)
    return model, recipe, selection


def test_binding_preserves_weights_and_original_and_is_idempotent(
    tmp_path, monkeypatch
):
    model, _, _ = setup_binding(tmp_path, monkeypatch)
    original = (tmp_path / "selection.json").read_bytes()
    weights = (tmp_path / "selected.json").read_bytes()
    metrics = (tmp_path / "validation.json").read_bytes()
    result = bind_selection("frozen")
    assert result["metadata_only"] and result["weights_unchanged"]
    assert (tmp_path / "selection_before_binding.json").read_bytes() == original
    assert (tmp_path / "selected.json").read_bytes() == weights and (
        tmp_path / "validation.json"
    ).read_bytes() == metrics
    assert (
        result["model_sha256"] == model_digest(model)
        and bind_selection("frozen") == result
    )


@pytest.mark.parametrize(
    "field", ["recipe_sha256", "implementation_digests", "model_sha256"]
)
def test_binding_refuses_changed_selection_before_writes(tmp_path, monkeypatch, field):
    _, _, selection = setup_binding(tmp_path, monkeypatch)
    selection[field] = "tampered"
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="mismatch"):
        bind_selection("frozen")
    assert not (tmp_path / "selection_before_binding.json").exists()
