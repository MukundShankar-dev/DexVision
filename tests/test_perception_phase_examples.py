"""Context labels never fit; weighting uses original physical source ownership."""

import json
from types import SimpleNamespace
import numpy as np
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_tracker import Track


def test_query_only_fit_and_physical_source_balancing(tmp_path, monkeypatch):
    import dexvision.learning.perception_phase_examples as module

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
    rows = []
    for window, parent in [
        ("a1", "physical_a"),
        ("a2", "physical_a"),
        ("b1", "physical_b"),
    ]:
        owner = {
            "entity_id": "block",
            "evaluation_only_parent_episode": parent,
            "phase_intervals": [
                {"start_frame": 0, "end_frame": 1, "phase": "lift"},
                {"start_frame": 5, "end_frame": 6, "phase": "stabilize"},
            ],
        }
        for query in [False, True]:
            rows.append(
                {
                    "split": "train",
                    "source_episode_id": window,
                    "source_frame_index": 5 if query else 0,
                    "timestamp": 0.17 if query else 0.0,
                    "phase_window_query": query,
                    "evaluation_only_source_owner": owner,
                    "annotations": [
                        {
                            "class_id": "rigid_cuboid",
                            "object_id": "block",
                            "visible": True,
                            "box_xyxy": [318, 238, 322, 242],
                            "pose_6d": {
                                "translation_m": [0.0, 0.0, 0.02]
                                if query
                                else [999.0, 999.0, 999.0]
                            },
                        }
                    ],
                }
            )
    result, _ = module.training_examples(
        {"visual_root": str(tmp_path), "tracker": {"maximum_gap_s": 0.51}}, rows, {}
    )
    x, y, w = result["rigid_cuboid"]
    assert x.shape == (3, 706)
    np.testing.assert_allclose(y[:, 2], 0.2)
    np.testing.assert_allclose(x[:, 16], 1.0)
    assert np.isclose(w[:2].sum(), w[2])
