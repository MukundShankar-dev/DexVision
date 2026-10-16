"""Neural transforms are TRAIN-only, deterministic and portable to inference."""

import numpy as np
import pytest
from dexvision.learning.perception_neural_v15 import fit_network
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.perception.context_network import predict_network


def test_neural_full_budget_and_portable_numpy_prediction():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 370))
    y = x[:, :3] * 0.01 + np.array([0.0, 0.0, 0.4])
    examples = {c: (x, y, np.ones(len(x))) for c in CORE_CLASSES}
    models, history = fit_network(
        examples, hidden=[8], seed=3, epochs=3, learning_rate=0.001
    )
    assert len(history) == 3 and len(models) == 4
    for c, m in models.items():
        np.testing.assert_allclose(m["mean"], x.mean(0))
        assert m["training_rows"] == 48 and m["epochs"] == 3
        assert predict_network(m, x[0]).shape == (3,)
        with pytest.raises(ValueError, match="finite context"):
            predict_network(m, np.full(370, np.nan))
    repeat, loss = fit_network(
        examples, hidden=[8], seed=3, epochs=3, learning_rate=0.001
    )
    assert history == loss and models == repeat


def test_neural_pose_preserves_partial_schema_and_clears_history():
    from dexvision.perception.neural_context import NeuralContextPose
    from dexvision.perception.object_detector import Detection
    from dexvision.perception.object_tracker import Track
    from dexvision.perception.world_estimate import inferred_world_estimate
    from dexvision.perception.object_observations import ObjectObservationError

    model = {
        "kind": "rgb_context_world_height_mlp",
        "feature_version": "level5/rgb-context370-v1",
        "mean": [0.0] * 370,
        "spread": [1.0] * 370,
        "class_index": 0,
        "layers": [{"weight": np.zeros((3, 374)).tolist(), "bias": [0.0, 0.0, 1.0]}],
        "target_mean": [0.0, 0.0, 0.0],
        "target_spread": [1.0, 1.0, 1.0],
    }
    calibration = {
        "intrinsics": np.eye(3).tolist(),
        "camera_from_world": np.eye(4).tolist(),
    }
    estimator = NeuralContextPose(
        calibration, {"rigid_cuboid": model}, maximum_gap_s=0.51
    )
    rgb = np.zeros((80, 100, 3), np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[38:42, 48:52] = True
    rgb[mask] = [200, 90, 20]
    track = Track(
        "visual/1",
        Detection("rigid_cuboid", (48, 38, 52, 42), mask, 1.0),
        1.0,
        False,
        False,
    )
    pose = estimator.estimate_rgb(rgb, track)
    assert pose.position[2] == pytest.approx(0.1)
    assert pose.orientation_wxyz is None and pose.angular_velocity is None
    world = inferred_world_estimate((pose,), 1.0)
    with pytest.raises(ObjectObservationError):
        world.require_control_state(now=1.0)
    estimator.reset()
    assert (
        not estimator.previous
        and not estimator.feature_history
        and estimator._context_maps is None
    )


def test_neural_consumed_attempt_rejects_before_dataset_access(tmp_path, monkeypatch):
    import dexvision.logging.perception_development as logging
    from dexvision.evaluation.perception_neural import confirm_context

    monkeypatch.setattr(
        logging,
        "load_protocol",
        lambda _: {
            "dataset_role": "confirmation",
            "selected_engine": "level5/rgb-context370-v1",
        },
    )
    monkeypatch.setattr(
        logging,
        "verify_dataset",
        lambda *_: pytest.fail("Consumed set must remain closed"),
    )
    (tmp_path / "confirmation_attempt.json").write_text("consumed")
    with pytest.raises(ValueError, match="consumed"):
        confirm_context(
            tmp_path / "protocol.json", tmp_path / "absent", tmp_path / "new-output"
        )
