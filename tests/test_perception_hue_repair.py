"""TRAIN hue supervision and independent late-phase qualification gates."""

from pathlib import Path
import json
import numpy as np
import pytest
from dexvision.learning.perception_hue_repair import hue_interval, pose_weights_digest
from dexvision.evaluation.perception_late_queries import late_phase_pass


def test_hue_interval_and_pose_weight_isolation():
    hist = np.zeros(180, int)
    hist[89:92] = [1, 100, 1]
    assert hue_interval(hist, 0.0, 1.0, 1) == [-2, 2]
    m = {"body_center_models": {"cube": {"layers": [1.0, 2.0]}}}
    before = pose_weights_digest(m)
    m["class_hue_offsets"] = {"rigid_cuboid": [-2, 2]}
    assert before == pose_weights_digest(m)
    with pytest.raises(ValueError):
        hue_interval(np.zeros(180), 0.0, 1.0, 1)


def group():
    return {
        "visible_queries": 20,
        "matches": 20,
        "poses": 20,
        "visible_physical_source_episodes": 5,
        "median_m": 0.002,
        "p95_m": 0.006,
    }


def test_late_groups_cannot_pool_recall_or_drop_missing_phase():
    assert late_phase_pass({"release": group(), "settle": group()})
    assert not late_phase_pass({"release": group()})
    bad = group()
    bad.update(visible_queries=9, matches=7, poses=7)
    assert not late_phase_pass({"release": group(), "settle": bad})


def test_new_freeze_rejects_missing_late_gate_before_delegating(tmp_path, monkeypatch):
    import dexvision.evaluation.perception_late_confirmation as module

    spec = tmp_path / "config.yaml"
    spec.write_text("selected_recipe: unused\n")
    (tmp_path / "validation.json").write_text(
        json.dumps({"late_phase_validation": {"passed": False}})
    )
    monkeypatch.setattr(
        module, "selected_context", lambda _: ({"model_dir": str(tmp_path)}, None)
    )
    monkeypatch.setattr(
        module,
        "freeze_phase_confirmation",
        lambda *_: pytest.fail("Must reject before freezing"),
    )
    with pytest.raises(ValueError, match="Late release/settle"):
        module.freeze_late_confirmation(spec, tmp_path / "protocol")


def test_validation_pixels_cannot_enter_hue_fit(tmp_path, monkeypatch):
    import dexvision.learning.perception_hue_repair as module

    monkeypatch.setattr(
        module,
        "read_rgb",
        lambda *_args, **_kwargs: pytest.fail("Validation pixels must not be read"),
    )
    with pytest.raises(ValueError, match="hue support"):
        module.fit_cube_hue(
            {
                "cube_hue_fit": {
                    "lower_quantile": 0.0,
                    "upper_quantile": 1.0,
                    "padding": 1,
                }
            },
            [(Path(tmp_path), [{"split": "validation"}])],
            11.0,
        )


def test_train_fitted_hue_separates_tiny_cube_from_reddish_background(monkeypatch):
    import dexvision.learning.perception_hue_repair as module
    from dexvision.perception.object_detector import ColorDetector

    rgb = np.zeros((10, 10, 3), np.uint8)
    rgb[4, 2:5] = [97, 43, 12]
    rgb[4, 5] = [83, 34, 26]
    rgb[4:6, 6] = [88, 35, 30]
    mask = np.zeros((10, 10), np.uint8)
    mask[4, 2:5] = 1
    monkeypatch.setattr(module, "read_rgb", lambda *_args, **_kwargs: (rgb, mask))
    recipe = {
        "minimum_saturation": 60,
        "minimum_value": 10,
        "cube_hue_fit": {"lower_quantile": 0.0, "upper_quantile": 1.0, "padding": 1},
    }
    row = {
        "split": "train",
        "annotations": [{"visible": True, "class_id": "rigid_cuboid", "mask_value": 1}],
    }
    interval, _ = module.fit_cube_hue(recipe, [(".", [row])], 11.0)
    args = {
        "palette": {
            "rigid_cuboid": 11.0,
            "rigid_cylinder": 68.0,
            "rigid_flat_puck": 107.0,
        },
        "hue_margin": 8.0,
        "minimum_pixels": 1,
        "minimum_saturation": 60,
        "minimum_value": 10,
        "button_roi": (0, 0, 0, 0),
        "fragment_merge": {
            "rigid_cuboid": {"maximum_gap_px": 24, "maximum_extent_px": 64}
        },
    }
    broad = ColorDetector(**args).detect(rgb)
    narrow = ColorDetector(**args, class_hue_offsets={"rigid_cuboid": interval}).detect(
        rgb
    )
    assert broad[0].box == (2, 4, 7, 6)
    assert narrow[0].box == (2, 4, 5, 5) and narrow[0].mask.sum() == 3
