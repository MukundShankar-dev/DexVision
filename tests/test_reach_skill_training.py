"""Reach-shaped CPU training, reload, evidence selection and held-out guards."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from dexvision.evaluation.evaluate_skill import ReachPolicy, evaluate_checkpoint, select_reach_candidates
from dexvision.learning.skill_datasets import SkillDataset, SkillEpisode, feature_schema, fit_normalization
from dexvision.learning.skill_models import model_schema
from dexvision.learning.train_skill import load_checkpoint, train_skill
from dexvision.learning.training_manifest import build_training_manifest, load_training_inputs


@pytest.fixture
def problem():
    inputs = copy.deepcopy(load_training_inputs("configs/level5/skills/reach_object.yaml"))
    # Explicitly synthetic, small CPU test; real CLI only accepts checksum-locked settings.
    inputs.plan["training"]["max_epochs"] = 2
    inputs.plan["training"]["batch_size"] = 4
    inputs.plan["training"]["selection"]["candidate_epochs"] = [1, 2]
    manifest = build_training_manifest(inputs, seed=5001, device="cpu", model="baseline")
    names, continuous = feature_schema(inputs.plan, inputs.skill)
    episodes = {}
    for split in ("train", "validation"):
        rows = manifest["assignments"][split]
        episodes[split] = [SkillEpisode(r["episode_id"], split, r["session_id"], r["cell_id"], r["episode_digest"],
            np.zeros((1, 192)), np.zeros((1, 26)), np.ones((1, 26)), np.array([0]), np.array([0.034]),
            ("approach",), (0, 1) if split == "train" else None, {}) for r in rows]
    norm = fit_normalization(episodes["train"], names, continuous, inputs.digests)
    datasets = {s: SkillDataset(e, norm) for s, e in episodes.items()}
    return inputs, manifest, datasets


def test_reach_named_heads_cpu_training_and_reload_retain_frozen_ownership(problem, tmp_path):
    inputs, manifest, datasets = problem
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        directory = tmp_path / "run"
        result = train_skill(datasets, schema=model_schema(inputs.plan, inputs.skill), settings=inputs.plan["training"],
            manifest=manifest, output_dir=directory, config_snapshot={"purpose": "synthetic reach interface test"})
        assert result["finished"] and result["completed_epochs"] == 2
        checkpoint = load_checkpoint(directory / "last.pt")
        policy = ReachPolicy(checkpoint, inputs)
        prior = np.zeros(27)
        prior[2:4] = [0.2, 1]
        action, prediction = policy(np.zeros(192), prior, "approach")
        assert action.shape == (27,) and prediction.shape == (26,)
        assert np.isfinite(action).all()
        np.testing.assert_array_equal(action[9:], prior[9:])
        assert not (directory / "selected.pt").exists()
        assert checkpoint["manifest"]["test_data_used_for_training_or_selection"] is False
        with pytest.raises(ValueError, match="validation-selected"):
            evaluate_checkpoint(inputs, directory / "last.pt", split="test", output_dir=tmp_path / "test")
        assert not (tmp_path / "test").exists()
        changed = copy.deepcopy(checkpoint)
        changed["schema"]["input_names"][0] = "future_result"
        with pytest.raises(ValueError, match="incompatible"):
            ReachPolicy(changed, inputs)
        changed = copy.deepcopy(checkpoint)
        changed["normalization"]["mean"][0] = 99
        with pytest.raises(ValueError, match="Normalization digest"):
            ReachPolicy(changed, inputs)
    finally:
        torch.set_num_threads(threads)


def test_training_rejects_test_tensors_and_session_leakage(problem, tmp_path):
    inputs, manifest, datasets = problem
    kwargs = dict(schema=model_schema(inputs.plan, inputs.skill), settings=inputs.plan["training"],
                  manifest=manifest, output_dir=tmp_path / "run", config_snapshot={})
    with pytest.raises(ValueError, match="never test"):
        train_skill({**datasets, "test": datasets["validation"]}, **kwargs)
    episode = replace(datasets["validation"].episodes[0], session_id=datasets["train"].episodes[0].session_id)
    with pytest.raises(ValueError, match="session leakage"):
        train_skill({**datasets, "validation": SkillDataset([episode], datasets["train"].normalization)}, **kwargs)


def test_candidate_selection_requires_finished_training(problem, tmp_path):
    inputs, manifest, datasets = problem
    directory = tmp_path / "run"
    train_skill(datasets, schema=model_schema(inputs.plan, inputs.skill), settings=inputs.plan["training"],
        manifest=manifest, output_dir=directory, config_snapshot={}, stop_after_steps=1)
    with pytest.raises(ValueError, match="finished"):
        select_reach_candidates(inputs, directory, output_dir=tmp_path / "validation")


def test_cli_reports_missing_checkpoint_and_future_skill_cleanly(capsys):
    from dexvision.apps.evaluate_skill import main

    assert main(["--config", "configs/level5/skills/reach_object.yaml", "--output-dir", "outputs/unused"]) == 2
    assert "requires --checkpoint" in capsys.readouterr().err
    assert main(["--config", "configs/level5/skills/pick_object.yaml", "--checkpoint", str(Path("absent.pt")), "--output-dir", "outputs/unused"]) == 2
    assert "5.2" in capsys.readouterr().err


def test_selected_run_cannot_repeat_test_even_with_new_output_directory(problem, tmp_path):
    from dexvision.learning.train_skill import _checksum_index, select_checkpoint

    inputs, manifest, datasets = problem
    directory = tmp_path / "run"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_skill(datasets, schema=model_schema(inputs.plan, inputs.skill), settings=inputs.plan["training"],
            manifest=manifest, output_dir=directory, config_snapshot={"purpose": "synthetic selection guard"})
        checkpoint = load_checkpoint(directory / "last.pt")
        reports = [{"epoch": int(epoch), "split": "validation", "seed": 5001,
                    "manifest_digest": manifest["manifest_digest"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "validation_matrix_sha256": manifest["validation_matrix_sha256"],
                    "rollout_count": manifest["validation_rollout_count"], "safety_violation_count": 0,
                    "success_rate": 0, "worst_cell_success_rate": 0, "normalized_terminal_error": 1}
                   for epoch, candidate in checkpoint["training_state"]["candidates"].items()]
        select_checkpoint(directory, reports)
        (directory / "test_state_attempt.json").write_text('{"purpose": "synthetic prior reservation"}')
        _checksum_index(directory)
        with pytest.raises(ValueError, match="single test"):
            evaluate_checkpoint(inputs, directory / "selected.pt", split="test", output_dir=tmp_path / "new_test")
        assert not (tmp_path / "new_test").exists()
    finally:
        torch.set_num_threads(threads)


def test_comparison_requires_both_frozen_models_and_all_seeds(tmp_path):
    from dexvision.evaluation.evaluate_skill import compare_reach_runs

    with pytest.raises(ValueError, match="both frozen models"):
        compare_reach_runs(Path("configs/level5/skills/reach_object.yaml"), [], tmp_path / "comparison", Path("absent.pt"))
    assert not (tmp_path / "comparison").exists()
