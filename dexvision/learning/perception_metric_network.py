"""Calibrated body-origin metric loss; height is not diluted by target variance."""

from __future__ import annotations
import numpy as np
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.perception.context_motion import FEATURE_VERSION


def metric_world(prediction, pixel_origin, calibration):
    """Differentiable equivalent of the frozen UV/world-height ray decoder."""
    import torch

    inverse_intrinsics = torch.as_tensor(
        np.linalg.inv(np.asarray(calibration["intrinsics"])),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    camera_to_world = torch.as_tensor(
        np.linalg.inv(np.asarray(calibration["camera_from_world"])),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    pixel = torch.cat(
        (pixel_origin + prediction[:, :2] * 32.0, torch.ones_like(prediction[:, :1])),
        dim=1,
    )
    direction = pixel @ inverse_intrinsics.T @ camera_to_world[:3, :3].T
    if torch.any(direction[:, 2].abs() < 1e-8):
        raise ValueError("Metric TRAIN ray is parallel to the height plane.")
    depth = (prediction[:, 2] * 0.1 - camera_to_world[2, 3]) / direction[:, 2]
    return camera_to_world[:3, 3] + direction * depth[:, None]


def fit_network(examples, *, hidden, seed, epochs, learning_rate, calibration):
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    classes = tuple(CORE_CLASSES)
    all_x = np.concatenate([examples[c][0] for c in classes])
    all_y = np.concatenate([examples[c][1] for c in classes])
    # Equal visual-class totals, retaining phase/episode balance within classes.
    weights = np.concatenate([examples[c][2] / examples[c][2].sum() for c in classes])
    weights /= weights.sum()
    mean, spread = all_x.mean(0), np.maximum(all_x.std(0), 0.02)
    labels = np.concatenate(
        [np.full(len(examples[c][0]), i) for i, c in enumerate(classes)]
    )
    class_mean = np.stack(
        [np.average(examples[c][1], axis=0, weights=examples[c][2]) for c in classes]
    )
    output_scale = np.array([0.25, 0.25, 0.1])
    target_mean, target_spread = (
        class_mean[labels],
        np.broadcast_to(output_scale, all_y.shape),
    )
    inputs = np.c_[(all_x - mean) / spread, np.eye(4)[labels]]
    x = torch.tensor(inputs, dtype=torch.float32)
    offsets = torch.tensor(target_mean, dtype=torch.float32)
    scales = torch.tensor(output_scale, dtype=torch.float32)
    raw_y = torch.tensor(all_y, dtype=torch.float32)
    pixel_origin = torch.tensor(
        all_x[:, :2] * np.array([640.0, 480.0]), dtype=torch.float32
    )
    target_world = metric_world(raw_y, pixel_origin, calibration).detach()
    w = torch.tensor(weights, dtype=torch.float32)
    layers = []
    for a, b in zip([710] + list(hidden), list(hidden) + [3]):
        if layers:
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(a, b))
    net = torch.nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Full weighted TRAIN objective: no rare-phase sampling loss or early stop.
        predicted_world = metric_world(
            net(x) * scales + offsets, pixel_origin, calibration
        )
        loss = (((predicted_world - target_world) / 0.01).square().mean(1) * w).sum()
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append(float(loss.detach()))
        if (epoch + 1) % 100 == 0:
            print("epoch", epoch + 1, "TRAIN loss", history[-1], flush=True)
    portable = [
        {
            "weight": m.weight.detach().numpy().tolist(),
            "bias": m.bias.detach().numpy().tolist(),
        }
        for m in net
        if isinstance(m, torch.nn.Linear)
    ]
    common = {
        "kind": "rgb_motion_world_height_mlp",
        "feature_version": FEATURE_VERSION,
        "mean": mean.tolist(),
        "spread": spread.tolist(),
        "layers": portable,
        "seed": seed,
        "epochs": epochs,
        "hidden": hidden,
        "training_rows": len(x),
        "target_units": "pixel_offsets/32_world_height/0.1m",
    }
    models = {
        c: {
            **common,
            "class_index": i,
            "target_mean": class_mean[i].tolist(),
            "target_spread": output_scale.tolist(),
            "training_objective": "weighted world-body-origin XYZ error/0.01m; exact calibrated ray decoder",
        }
        for i, c in enumerate(classes)
    }
    from dexvision.perception.context_motion_network import predict_network

    with torch.no_grad():
        expected = net(x[:4]).numpy() * target_spread[:4] + target_mean[:4]
    actual = np.stack(
        [predict_network(models[classes[labels[i]]], all_x[i]) for i in range(4)]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return models, history
