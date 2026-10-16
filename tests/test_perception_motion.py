"""Causal RGB motion features and portable inference have identical resets/scales."""

import numpy as np
import pytest
from dataclasses import replace
from dexvision.perception.context_motion import (
    context_features,
    NeuralContextPose,
    FEATURE_VERSION,
)
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_tracker import Track
from dexvision.learning.perception_motion import fit_network
from dexvision.evaluation.perception_metrics import CORE_CLASSES


def sample():
    rgb = np.zeros((80, 100, 3), np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[38:42, 48:52] = True
    rgb[mask] = [200, 90, 20]
    return rgb, Detection("rigid_cuboid", (48, 38, 52, 42), mask, 1.0)


def test_motion_is_previous_acquisition_rgb_only_and_expires():
    rgb, d = sample()
    a, stored = context_features(rgb, d, 1.0, None, maximum_gap_s=0.51)
    assert a.shape == (706,) and stored.shape == (370,) and not a[370:].any()
    changed = rgb.copy()
    changed[4:24, 10:35] = 100
    b, current = context_features(changed, d, 1.17, (1.0, stored), maximum_gap_s=0.51)
    np.testing.assert_allclose(b[370:], (current[34:] - stored[34:]) / 0.17)
    assert b[16] == 1 and np.any(b[370:])
    expired, _ = context_features(changed, d, 2.0, (1.0, stored), maximum_gap_s=0.51)
    assert not expired[370:].any() and expired[16] == 0
    alternate, _ = context_features(
        changed,
        replace(d, class_id="withheld-semantic"),
        1.17,
        (1.0, stored),
        maximum_gap_s=0.51,
    )
    np.testing.assert_array_equal(alternate, b)


def test_motion_network_full_budget_deterministic_numpy_parity():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 706))
    y = x[:, :3] * 0.01 + [0, 0, 0.4]
    e = {c: (x, y, np.ones(len(x))) for c in CORE_CLASSES}
    models, history = fit_network(e, hidden=[8], seed=5, epochs=2, learning_rate=0.001)
    repeat, other = fit_network(e, hidden=[8], seed=5, epochs=2, learning_rate=0.001)
    assert models == repeat and history == other and len(history) == 2


def test_compiled_pose_does_not_mutate_portable_model_and_reset_clears_rgb():
    import copy
    from dexvision.perception.world_estimate import inferred_world_estimate
    from dexvision.perception.object_observations import ObjectObservationError

    model = {
        "kind": "rgb_motion_world_height_mlp",
        "feature_version": FEATURE_VERSION,
        "mean": [0.0] * 706,
        "spread": [1.0] * 706,
        "class_index": 0,
        "layers": [{"weight": np.zeros((3, 710)).tolist(), "bias": [0.0, 0.0, 1.0]}],
        "target_mean": [0.0, 0.0, 0.0],
        "target_spread": [1.0, 1.0, 1.0],
    }
    frozen = copy.deepcopy(model)
    cal = {"intrinsics": np.eye(3).tolist(), "camera_from_world": np.eye(4).tolist()}
    pose = NeuralContextPose(cal, {"rigid_cuboid": model}, maximum_gap_s=0.51)
    rgb, d = sample()
    result = pose.estimate_rgb(rgb, Track("visual/1", d, 1.0, False, False))
    assert model == frozen and result.position[2] == pytest.approx(0.1)
    with pytest.raises(ObjectObservationError):
        inferred_world_estimate((result,), 1.0).require_control_state(now=1.0)
    pose.reset()
    assert not pose.feature_history and pose._context_maps is None


def test_motion_consumed_attempt_blocks_before_data_access(tmp_path, monkeypatch):
    import dexvision.logging.perception_development as logging
    from dexvision.evaluation.perception_motion import confirm_context

    monkeypatch.setattr(
        logging,
        "load_protocol",
        lambda _: {"dataset_role": "confirmation", "selected_engine": FEATURE_VERSION},
    )
    monkeypatch.setattr(
        logging, "verify_dataset", lambda *_: pytest.fail("Consumed set remains closed")
    )
    (tmp_path / "confirmation_attempt.json").write_text("consumed")
    with pytest.raises(ValueError, match="consumed"):
        confirm_context(
            tmp_path / "protocol.json", tmp_path / "absent", tmp_path / "new-output"
        )
