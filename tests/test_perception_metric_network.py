"""Metric TRAIN loss matches the inference decoder and explicit output units."""

import numpy as np
import pytest
import torch
from dexvision.learning.perception_metric_network import metric_world, fit_network
from dexvision.perception.height_model import decode_height
from dexvision.evaluation.perception_metrics import CORE_CLASSES


def calibration():
    return {
        "intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "camera_from_world": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_metric_loss_decoder_matches_frozen_numpy_and_has_gradients():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.7, (10, 706))
    y = rng.normal(0, 0.1, (10, 3))
    y[:, 2] = rng.uniform(0.2, 1.0, 10)
    predictions = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    actual = metric_world(
        predictions, torch.tensor(x[:, :2] * [640, 480]), calibration()
    )
    expected = np.stack([decode_height(a, b, calibration()) for a, b in zip(y, x)])
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-12)
    loss = ((actual - torch.tensor(expected + 0.01)) / 0.01).square().mean()
    loss.backward()
    assert torch.isfinite(predictions.grad).all() and torch.all(
        predictions.grad.abs().sum(0) > 0
    )


def test_metric_fit_uses_class_weighted_centers_and_fixed_physical_scales():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, (8, 706))
    x[:, :2] = rng.uniform(0.4, 0.6, (8, 2))
    weights = np.arange(1, 9, dtype=float)
    examples = {
        c: (x, np.c_[x[:, :2] * 0.01, np.full(8, 0.3 + i * 0.2)], weights)
        for i, c in enumerate(CORE_CLASSES)
    }
    models, history = fit_network(
        examples,
        hidden=[8],
        seed=4,
        epochs=2,
        learning_rate=0.001,
        calibration=calibration(),
    )
    repeat, other = fit_network(
        examples,
        hidden=[8],
        seed=4,
        epochs=2,
        learning_rate=0.001,
        calibration=calibration(),
    )
    assert models == repeat and history == other and len(history) == 2
    for c, model in models.items():
        np.testing.assert_allclose(
            model["target_mean"], np.average(examples[c][1], axis=0, weights=weights)
        )
        assert model["target_spread"] == [0.25, 0.25, 0.1]
        assert "world-body-origin XYZ" in model["training_objective"]


def test_parallel_training_ray_rejects():
    cal = calibration()
    cal["camera_from_world"] = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="parallel"):
        metric_world(
            torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64),
            torch.tensor([[320.0, 240.0]], dtype=torch.float64),
            cal,
        )
