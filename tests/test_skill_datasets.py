"""Causality, named fields, immutable release smoke and normalization ownership."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from dexvision.learning.skill_datasets import (
    SkillDataset, SkillEpisode, action_targets, feature_schema, fit_normalization, vectorize_episode,
)
from dexvision.learning.skill_models import SkillMLP, decode_delta, model_schema
from dexvision.learning.training_manifest import ReleaseSource, load_training_inputs, select_streams


@pytest.fixture(scope="module")
def inputs():
    return {skill: load_training_inputs(f"configs/level5/skills/{skill}.yaml") for skill in (
        "reach_object", "pick_object", "place_held_object", "push_object_to_target", "press_button")}


@pytest.fixture(scope="module")
def released_examples(inputs):
    rows = {}
    for data in inputs.values():
        for split in ("train", "validation"):
            row = select_streams(data)["expert_success"][split][0]
            rows[row["episode_id"]] = row
    data = inputs["reach_object"]
    archive = data.root / data.plan["inputs"]["archive"]["path"]
    if not archive.is_file() or archive.stat().st_size < 1000:
        pytest.skip("Immutable Level 4 archive requires git lfs pull")
    records = ReleaseSource(data).read_episode_records(list(rows.values()))
    return rows, records


@pytest.mark.parametrize("skill", ["reach_object", "pick_object", "place_held_object", "push_object_to_target", "press_button"])
def test_all_five_skills_read_named_causal_release_tensors(inputs, released_examples, skill):
    _, records = released_examples
    data = inputs[skill]
    episodes = []
    names, continuous = feature_schema(data.plan, data.skill)
    for split in ("train", "validation"):
        row = select_streams(data)["expert_success"][split][0]
        e = vectorize_episode(data, row, records[row["episode_id"]])
        assert e.inputs.shape[1] == 183 + data.skill["numeric_goal_width"]
        assert e.targets.shape == e.masks.shape == (len(e.frames), 26)
        assert np.isfinite(e.inputs).all() and abs(e.targets).max() <= 1 + 1e-5
        assert not any("start_button_joint" in n or "block_large" in n for n in names)
        assert e.normalization_interval is None if split == "validation" else e.normalization_interval is not None
        episodes.append(e)
    norm = fit_normalization(episodes[:1], names, continuous, data.digests)
    assert norm["training_episode_ids"] == [episodes[0].episode_id]
    assert np.all(np.asarray(norm["mean"])[~continuous] == 0)
    assert np.all(np.asarray(norm["std"])[~continuous] == 1)
    for e in episodes:
        assert np.isfinite(SkillDataset([e], norm)[0]["inputs"]).all()
    with pytest.raises(ValueError, match="Only training"):
        fit_normalization(episodes, names, continuous, data.digests)


def test_current_and_future_state_and_annotations_cannot_change_current_input(inputs, released_examples):
    _, records = released_examples
    data = inputs["reach_object"]
    row = select_streams(data)["expert_success"]["train"][0]
    record = records[row["episode_id"]]
    original = vectorize_episode(data, row, record)
    changed = copy.deepcopy(record)
    changed["robot_states"][2:, 0] += 0.001
    changed["online_phases"][:] = "retract"
    changed["audited_phases"][:] = "retract"
    changed["task_states"][:] = 0
    altered = vectorize_episode(data, row, changed)
    np.testing.assert_array_equal(original.inputs[:3], altered.inputs[:3])
    assert not np.array_equal(original.inputs[3], altered.inputs[3])
    # Frame zero is reconstructed reset state, not first resulting robot row.
    assert not np.array_equal(original.inputs[0, :3], record["robot_states"][0, :3])
    assert original.records["saved_phase_disagreement_count"] != altered.records["saved_phase_disagreement_count"]


def test_skew_and_prior_history_are_rejected(inputs, released_examples):
    _, records = released_examples
    data = inputs["reach_object"]
    row = select_streams(data)["expert_success"]["train"][0]
    record = copy.deepcopy(records[row["episode_id"]])
    record["action_timestamps"][0] += 0.01
    with pytest.raises(ValueError, match="timestamp"):
        vectorize_episode(data, row, record)
    record = copy.deepcopy(records[row["episode_id"]])
    record["prior_applied_actions"][1, 0] += 0.01
    with pytest.raises(ValueError, match="Prior action"):
        vectorize_episode(data, row, record)


def test_rotation_bounds_roundtrip_and_relevance(inputs):
    import torch

    data = inputs["reach_object"]
    prior = np.zeros(27)
    prior[3] = 1
    prediction = np.full(26, 0.1)
    applied = decode_delta(prediction, prior, "acquire", data.plan)
    targets, mask = action_targets(applied[None], prior[None], np.zeros((1, 27)), ("acquire",), data.plan, "baseline")
    np.testing.assert_allclose(targets[0], prediction, atol=1e-12)
    assert mask.all()
    negative = applied.copy()
    negative[3:7] *= -1
    t, _ = action_targets(negative[None], prior[None], np.zeros((1, 27)), ("acquire",), data.plan, "baseline")
    np.testing.assert_allclose(t, targets)
    assert np.array_equal(decode_delta(prediction, prior, "settle", data.plan), prior)
    model = SkillMLP(model_schema(data.plan, data.skill))
    output = model(torch.full((8, 192), 1e6)).detach().numpy()
    assert abs(output).max() <= 1
    assert np.all(np.linalg.norm(output[:, 3:6], axis=1) <= 1 + 1e-6)


def synthetic_episode(eid="train-a", split="train", count=4, offset=0):
    values = np.arange(count * 2, dtype=float).reshape(count, 2) + offset
    return SkillEpisode(eid, split, eid, "cell-a", eid, values, np.zeros((count, 26)),
                        np.ones((count, 26)), np.arange(count), np.arange(count) * 0.034,
                        ("approach",) * count, (1, count), {})


def test_normalization_intersection_and_unscaled_columns():
    episode = synthetic_episode()
    norm = fit_normalization([episode], ("x", "binary"), np.array([True, False]), {})
    assert norm["count"] == 3
    assert norm["mean"] == [4.0, 0.0]
    assert norm["std"][0] == pytest.approx(np.std([2, 4, 6]))
    assert norm["std"][1] == 1
    excluded = replace(synthetic_episode("excluded", offset=1e9), normalization_interval=None)
    assert fit_normalization([episode, excluded], ("x", "binary"), np.array([True, False]), {}) == norm


def test_sampling_balances_cells_then_episodes_then_frames():
    a = synthetic_episode(count=2)
    b = replace(synthetic_episode("train-b", count=100), cell_id="cell-b")
    c = replace(synthetic_episode("train-c", count=2), cell_id="cell-b")
    norm = {"mean": [0, 0], "std": [1, 1]}
    dataset = SkillDataset([a, b, c], norm)
    first = dataset.sample_indices(np.random.default_rng(42), 12000)
    second = dataset.sample_indices(np.random.default_rng(42), 12000)
    assert np.array_equal(first, second)
    ids = [dataset.identities[i][0] for i in first]
    assert ids.count("train-a") / len(ids) == pytest.approx(0.5, abs=0.025)
    assert ids.count("train-b") / len(ids) == pytest.approx(0.25, abs=0.025)
    assert ids.count("train-c") / len(ids) == pytest.approx(0.25, abs=0.025)
