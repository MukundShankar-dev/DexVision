"""Correct action counts without changing immutable video pixels on disk."""

import numpy as np

from dexvision.evaluation.instrumented_review_overlay import annotate_review_frame


def test_review_header_uses_recorded_actions_and_keeps_input_frame():
    image = np.full((480, 640, 3), 80, np.uint8)
    before = image.copy()
    actions = np.zeros((7, 27))
    actions[:, 0] = np.arange(7) / 100
    trace = {"applied_actions": actions, "state_timestamps": np.arange(8) * 0.034}
    observations = {
        "timestamps": np.arange(8) * 0.034,
        "phases": np.array(["transport"] * 8),
    }
    result = {
        "terminal_result": "failed",
        "terminal_reason": "perception_unavailable",
        "condition": "nominal",
    }
    for index, expected in enumerate([1, 4, 7, 7]):
        panel, meta = annotate_review_frame(
            image,
            index,
            skill="place",
            result=result,
            trace=trace,
            observations=observations,
        )
        assert meta["applied"] == expected and meta["terminal_frame"] == (index == 3)
        assert np.array_equal(image, before)
        assert np.array_equal(panel[45:], before[45:])
    assert meta["xyz"] == actions[-1, :3].tolist()


def test_zero_action_review_does_not_fabricate_pose_or_action():
    _, meta = annotate_review_frame(
        np.zeros((480, 640, 3), np.uint8),
        0,
        skill="push",
        result={
            "terminal_result": "rejected_before_execution",
            "terminal_reason": "missing pose",
            "condition": "nominal",
        },
        trace={},
        observations={},
    )
    assert (
        meta["applied"] == 0 and meta["xyz"] is None and meta["phase"] == "unavailable"
    )
    assert meta["terminal_frame"]
