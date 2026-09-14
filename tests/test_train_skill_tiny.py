"""Tiny CPU overfit, exact mid-epoch resume and validation-only selection."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dexvision.learning.skill_datasets import SkillDataset, SkillEpisode, fit_normalization
from dexvision.learning.skill_models import SkillMLP, grouped_loss
from dexvision.learning.train_skill import load_checkpoint, select_checkpoint, train_skill
from dexvision.learning.training_manifest import content_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_problem(epochs=12):
    plan = yaml.safe_load((ROOT / "configs/level5/learning.yaml").read_text())
    settings = copy.deepcopy(plan["training"])
    settings.update(max_epochs=epochs, batch_size=4)
    settings["early_stop"]["patience_epochs"] = epochs + 1
    settings["selection"]["candidate_epochs"] = list(range(2, epochs + 1, 2))
    architecture = copy.deepcopy(plan["models"]["baseline"])
    architecture["hidden_sizes"] = [16, 16]
    schema = {"version": "level5/policy-schema-v1", "input_names": ["x", "y", "z"],
              "output_names": [f"synthetic_target_{i}" for i in range(26)],
              "model": "baseline", "architecture": architecture}
    rng = np.random.default_rng(11)
    episodes = {}
    for split in ("train", "validation"):
        x = np.zeros((16, 3))
        x[:, 0] = np.linspace(-0.5, 0.5, 16) if split == "train" else rng.uniform(-0.45, 0.45, 16)
        target = np.zeros((16, 26))
        target[:, :3] = x * 0.3
        target[:, 6:] = x[:, :1] * 0.15
        episodes[split] = SkillEpisode(split, split, split, "synthetic", split, x, target,
            np.ones((16, 26)), np.arange(16), np.arange(16) * 0.034, ("acquire",) * 16,
            (0, 16) if split == "train" else None, {})
    norm = fit_normalization([episodes["train"]], tuple(schema["input_names"]), np.ones(3, bool), {})
    datasets = {s: SkillDataset([e], norm) for s, e in episodes.items()}
    manifest = {"version": "level5/training-manifest-v1", "seed": 5001, "device": "cpu",
                "validation_matrix_sha256": content_digest("synthetic-validation"),
                "validation_rollout_count": 16,
                "purpose": "synthetic CPU acceptance only; no real skill qualification"}
    manifest["manifest_digest"] = content_digest(manifest)
    return datasets, schema, settings, manifest


def run(problem, path, **kwargs):
    datasets, schema, settings, manifest = problem
    return train_skill(datasets, schema=schema, settings=settings, manifest=manifest,
                       output_dir=path, config_snapshot={"synthetic": True}, **kwargs)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def synthetic_reports(path):
    last = load_checkpoint(path / "last.pt")
    return [{"epoch": int(epoch), "split": "validation", "seed": last["manifest"]["seed"],
             "manifest_digest": last["manifest"]["manifest_digest"],
             "checkpoint_sha256": candidate["checkpoint_sha256"],
             "validation_matrix_sha256": last["manifest"]["validation_matrix_sha256"],
             "rollout_count": last["manifest"]["validation_rollout_count"],
             "safety_violation_count": 0, "success_rate": 0.0, "worst_cell_success_rate": 0.0,
             "normalized_terminal_error": candidate["offline_validation_loss"],
             "evidence_kind": "synthetic acceptance stub, not a MuJoCo rollout"}
            for epoch, candidate in last["training_state"]["candidates"].items()]


def test_mid_epoch_resume_is_bitwise_identical(tmp_path):
    problem = tiny_problem(6)
    uninterrupted = run(problem, tmp_path / "full")
    interrupted = run(problem, tmp_path / "resumed", stop_after_steps=3)
    assert interrupted["completed_epochs"] == 0 and interrupted["step_in_epoch"] == 3
    resumed = run(problem, tmp_path / "resumed", resume=True)
    assert resumed["history"] == uninterrupted["history"]
    left = load_checkpoint(tmp_path / "full/last.pt")
    right = load_checkpoint(tmp_path / "resumed/last.pt")
    for field in ("model", "optimizer", "scheduler", "rng", "training_state", "normalization"):
        assert_nested_equal(left[field], right[field])
    assert left["training_state"]["sampling_digest"] == right["training_state"]["sampling_digest"]
    assert not (tmp_path / "full/selected.pt").exists()


def test_tiny_cpu_overfits_and_selected_checkpoint_reloads(tmp_path):
    problem = tiny_problem(90)
    result = run(problem, tmp_path / "tiny")
    assert result["history"][-1]["training_loss"] < 0.0005
    assert result["best_offline_loss"] < 0.001
    selected = select_checkpoint(tmp_path / "tiny", synthetic_reports(tmp_path / "tiny"))
    saved = load_checkpoint(tmp_path / "tiny/selected.pt")
    model = SkillMLP(saved["schema"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    dataset = problem[0]["validation"]
    with torch.no_grad():
        loss = grouped_loss(model(torch.from_numpy(dataset.inputs)), torch.from_numpy(dataset.targets),
                            torch.from_numpy(dataset.masks), problem[2]["loss"]["group_weights"])
    assert loss.item() == pytest.approx(saved["training_state"]["history"][-1]["validation_offline_loss"])
    assert selected["epoch"] == result["best_offline_epoch"]
    assert json.loads((tmp_path / "tiny/selection.json").read_text())["qualification_claim"] is None
    assert run(problem, tmp_path / "tiny", resume=True)["selection_status"] == "selected_from_validation"
    for name in ("training_manifest.json", "normalization.json", "environment.json", "config_snapshot.yaml", "SHA256SUMS"):
        assert (tmp_path / "tiny" / name).is_file()


@pytest.mark.parametrize("change", ["seed", "config", "tensor", "schema", "normalization", "cell", "tamper"])
def test_resume_rejects_changed_contract(tmp_path, change):
    problem = tiny_problem(2)
    directory = tmp_path / "run"
    run(problem, directory, stop_after_steps=1)
    datasets, schema, settings, manifest = problem
    if change == "seed":
        manifest["seed"] += 1
    elif change == "config":
        settings["optimizer"]["learning_rate"] *= 2
    elif change == "tensor":
        datasets["train"].inputs[0, 0] += 1
    elif change == "schema":
        schema["input_names"][0] = "wrong"
    elif change == "normalization":
        datasets["train"].normalization["mean"][0] += 1
    elif change == "cell":
        from dataclasses import replace

        episode = replace(datasets["train"].episodes[0], cell_id="changed-cell")
        datasets["train"] = SkillDataset([episode], datasets["train"].normalization)
    else:
        path = directory / "last.pt"
        path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="changed|Resume|checksum"):
        run(problem, directory, resume=True)


def test_reject_overwrite_and_held_out_selection(tmp_path):
    problem = tiny_problem(2)
    directory = tmp_path / "run"
    run(problem, directory)
    with pytest.raises(FileExistsError):
        run(problem, directory)
    reports = synthetic_reports(directory)
    reports[0]["split"] = "test"
    with pytest.raises(ValueError, match="held-out"):
        select_checkpoint(directory, reports)
    reports = synthetic_reports(directory)
    reports[0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        select_checkpoint(directory, reports)
    reports = synthetic_reports(directory)
    reports[0]["rollout_count"] = 1
    with pytest.raises(ValueError, match="provenance"):
        select_checkpoint(directory, reports)
    reports = synthetic_reports(directory)
    for report in reports:
        report["normalized_terminal_error"] = None
        report["terminal_error_unavailable_reason"] = "rejected_precondition"
    assert select_checkpoint(directory, reports)["status"] == "selected_from_validation"
    datasets, schema, settings, manifest = problem
    datasets["test"] = datasets["validation"]
    with pytest.raises(ValueError, match="never test"):
        run((datasets, schema, settings, manifest), tmp_path / "invalid")


def test_group_loss_masks_and_group_width_normalization():
    pred = torch.ones((2, 26), requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.zeros_like(pred)
    mask[0, :3] = 1
    mask[1, 8:] = 1
    weights = dict.fromkeys(("base_position", "base_orientation", "wrist", "fingers"), 1.0)
    loss = grouped_loss(pred, target, mask, weights)
    assert loss.item() == 1.0
    assert grouped_loss(pred, target, mask, dict.fromkeys(weights, 0.25)).item() == 1.0
    loss.backward()
    assert torch.equal(pred.grad[mask == 0], torch.zeros_like(pred.grad[mask == 0]))


def test_cli_help_and_missing_config(capsys, tmp_path):
    from dexvision.apps.train_skill import main

    assert main(["--config", str(tmp_path / "missing.yaml"), "--dry-run"]) == 2
    assert "ERROR:" in capsys.readouterr().err
