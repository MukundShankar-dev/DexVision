"""Full-budget joint visual-class context regression on frozen TRAIN only."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from dexvision.learning.perception_motion_examples import (
    load_context_recipe,
    training_examples,
)
from dexvision.learning.perception_context_phase import strict_validation
from dexvision.learning.perception_development import (
    development_rows,
    load_repair_recipe,
)
from dexvision.perception.context_motion import FEATURE_VERSION
from dexvision.perception.perception_baseline import (
    load_model,
    save_artifact_index,
    model_digest,
)
from dexvision.logging.visual_stream import write_json, digest_file
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.evaluation.perception_motion import score_context


def fit_network(examples, *, hidden, seed, epochs, learning_rate):
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
    target_mean, target_spread = all_y.mean(0), np.maximum(all_y.std(0), 0.05)
    labels = np.concatenate(
        [np.full(len(examples[c][0]), i) for i, c in enumerate(classes)]
    )
    inputs = np.c_[(all_x - mean) / spread, np.eye(4)[labels]]
    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor((all_y - target_mean) / target_spread, dtype=torch.float32)
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
        loss = (((net(x) - y) ** 2).mean(1) * w).sum()
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
        "target_mean": target_mean.tolist(),
        "target_spread": target_spread.tolist(),
        "layers": portable,
        "seed": seed,
        "epochs": epochs,
        "hidden": hidden,
        "training_rows": len(x),
        "target_units": "pixel_offsets/32_world_height/0.1m",
    }
    models = {c: {**common, "class_index": i} for i, c in enumerate(classes)}
    from dexvision.perception.context_motion_network import predict_network

    with torch.no_grad():
        expected = net(x[:4]).numpy() * target_spread + target_mean
    actual = np.stack(
        [predict_network(models[classes[labels[i]]], all_x[i]) for i in range(4)]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return models, history


def train_context(config):
    recipe = load_context_recipe(config)
    rows = development_rows(recipe)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    parent = load_repair_recipe(recipe["detector_recipe"])
    detector = {
        k: v for k, v in load_model(parent).items() if k != "body_center_models"
    }
    for c, rule in detector["fragment_merge"].items():
        rule["maximum_gap_px"] = recipe["detector_fragment_gap_px"][c]
    current = {
        **recipe,
        "tracking_mode": "predictive",
        "tracker": detector["tracker_config"],
    }
    examples, audit = training_examples(current, rows, detector)
    write_json(output / "training_audit.json", audit)
    np.savez_compressed(
        output / "train_examples.npz",
        **{
            f"{c}_{name}": values[i]
            for c, values in examples.items()
            for i, name in enumerate(("features", "targets", "weights"))
        },
    )

    candidates = []
    for spec in recipe["network_candidates"]:
        models, history = fit_network(examples, **spec)
        model = {
            **detector,
            "body_center_models": models,
            "feature_version": FEATURE_VERSION,
            "tracking_mode": "predictive",
            "method": "rgb_motion_world_height_mlp",
        }
        report = score_context(current, model, rows, "validation")
        passes = strict_validation(report)
        name = f"mlp_{'_'.join(map(str, spec['hidden']))}_s{spec['seed']}"
        write_json(
            output / f"{name}.json",
            {
                "model": model,
                "report": report,
                "strict_pass": passes,
                "train_loss": history,
            },
        )
        print(
            name,
            "strict=",
            passes,
            "primary_p95=",
            report["primary_object_metrics"]["position_error_p95_m"],
            flush=True,
        )
        worst = max(
            g["position_error_p95_m"] or 1.0
            for g in report["primary_class_phase"].values()
            if g["visible_eligible_instances"] >= 8
        )
        candidates.append(
            (
                int(passes),
                -worst,
                -report["primary_object_metrics"]["position_error_p95_m"],
                -len(candidates),
                model,
                report,
            )
        )
    selected = max(candidates, key=lambda r: r[:4])
    model, report = selected[-2:]
    report["strict_validation_pass"] = strict_validation(report)
    write_json(output / "selected.json", model)
    write_json(output / "validation.json", report)
    write_json(
        output / "selection.json",
        {
            "rule": recipe["candidate_selection"],
            "model_sha256": model_digest(model),
            "recipe_sha256": digest_file(Path(config)),
            "implementation_digests": recipe["implementation_digests"],
            "test_data_used_for_fitting_or_selection": False,
            "strict_validation_pass": strict_validation(report),
            "candidate_count": len(candidates),
        },
    )
    save_artifact_index(output)
    return report
