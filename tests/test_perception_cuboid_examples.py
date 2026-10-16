"""A cuboid-only coverage parent need not supply other TRAIN classes."""

import json
from types import SimpleNamespace
import numpy as np
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_tracker import Track


def test_cuboid_only_harvest_uses_exact_motion_features_without_other_classes(
    tmp_path, monkeypatch
):
    import dexvision.learning.perception_cuboid_examples as module

    rgb = np.zeros((480, 640, 3), np.uint8)
    mask = np.zeros((480, 640), bool)
    mask[238:242, 318:322] = True
    rgb[mask] = [30, 150, 60]
    d = Detection("rigid_cuboid", (318, 238, 322, 242), mask, 1.0)
    fake = SimpleNamespace(
        reset=lambda: None,
        detector=SimpleNamespace(detect=lambda _: (d,)),
        tracker=SimpleNamespace(
            update=lambda ds, t: tuple(
                Track("visual/1", v, t, False, False) for v in ds
            )
        ),
    )
    monkeypatch.setattr(module, "ContextBaseline", lambda *_: fake)
    monkeypatch.setattr(module, "read_rgb", lambda *_: rgb)
    cal = {
        "intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "camera_from_world": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    recipe = {"visual_root": str(tmp_path), "tracker": {"maximum_gap_s": 0.51}}
    owner = {
        "entity_id": "block",
        "phase_intervals": [{"start_frame": 0, "end_frame": 1, "phase": "place"}],
    }
    row = {
        "split": "train",
        "source_episode_id": "cube_train",
        "timestamp": 0.0,
        "source_frame_index": 0,
        "evaluation_only_source_owner": owner,
        "annotations": [
            {
                "class_id": "rigid_cuboid",
                "object_id": "block",
                "visible": True,
                "box_xyxy": [318, 238, 322, 242],
                "pose_6d": {"translation_m": [0.0, 0.0, 0.02]},
            }
        ],
    }
    result, audit = module.training_examples(recipe, [row], {})
    assert set(result) == {"rigid_cuboid"} and set(audit) == {"rigid_cuboid"}
    x, y, w = result["rigid_cuboid"]
    assert x.shape == (1, 706) and y.shape == (1, 3) and w.tolist() == [1.0]
    assert np.isclose(y[0, 2], 0.2) and not x[0, 370:].any()
