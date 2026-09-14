"""Frozen rollout matrices and independent immutable expert-statistic readback."""
from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ["reach_object", "pick_object", "place_held_object", "push_object_to_target", "press_button"]


def load(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    return json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)


@pytest.fixture(scope="module")
def documents():
    return (
        load("configs/level5/learning.yaml"),
        load("configs/level5/evaluation_common.yaml"),
        {s: load(f"configs/level5/skills/{s}.yaml") for s in SKILLS},
        {s: json.loads((ROOT / f"datasets/level4-v1/splits/{s}.json").read_text())
         for s in ["train", "validation", "test"]},
    )


def expected_interval(skill, row):
    segment = next(s for s in row["segments"] if s["skill_name"] == skill)
    start = segment["start_frame"]
    phase = {"press_button": "fixture_contact", "push_object_to_target": "push_contact"}.get(skill)
    if phase:
        start = next(p["start_frame"] for p in row["phase_intervals"] if p["phase"] == phase)
    return [start, segment["end_frame"]]


def test_every_required_cell_has_explicit_paired_resets_and_frozen_goals(documents):
    plan, common, skills, splits = documents
    base = load("configs/level4_dataset.yaml")
    assert set(skills) == set(plan["required_skills"])
    conditions = common["rollouts"]["perception_conditions"]
    assert conditions == list(load("configs/level4_visual_dataset.yaml")["conditions"])
    for skill, config in skills.items():
        assert config["contract"] == base["skills"][skill]
        assert config["numeric_goal_width"] == sum(f["shape"][0] for f in config["numeric_goal_layout"])
        assert config["observation_width"] == plan["observations"]["common_width"]
        assert config["goal_binding"] and config["goal_error_terms"]
        for term in config["goal_error_terms"]:
            assert term["metric"] in config["contract"]["terminal_state_fields"]
            assert term["operation"] in {"nonnegative_value", "positive_shortfall"}
            assert math.isfinite(term["scale"]) and term["scale"] > 0
            if term["operation"] == "positive_shortfall":
                target = term["target"]
                if isinstance(target, dict):
                    assert target["goal_field"] in config["contract"]["goal_fields"]
                else:
                    assert math.isfinite(target)
        for split in ["validation", "test"]:
            matrix = config["evaluation"][split]
            rows = {r["episode_id"]: r for r in splits[split]["episodes"]
                    if r["skill"] == config["source_skill"] and r["stream"] == "expert_success"}
            required = {r["cell_id"] for r in rows.values()}
            assert {c["cell_id"] for c in matrix["cells"]} == required
            assert len(matrix["cells"]) == len(required)
            count = max(3, math.ceil(30 / len(required)))
            chosen = []
            for cell in matrix["cells"]:
                expected = sorted(r["episode_id"] for r in rows.values() if r["cell_id"] == cell["cell_id"])[:count]
                assert cell["episode_ids"] == expected
                assert len(cell["reset_seeds"]) == len(cell["start_frames"]) == count
                for eid, start in zip(cell["episode_ids"], cell["start_frames"], strict=True):
                    row = rows[eid]
                    assert row["split"] == split and row["source"] == "scripted"
                    assert start == expected_interval(skill, row)[0]
                chosen.extend(expected)
            assert len(chosen) == len(set(chosen)) == matrix["unique_reset_count"] >= 30
            assert matrix["rollouts_per_track_per_training_seed"] == len(chosen) * len(conditions)
    assert common["reset_policy"]["same_resets_all_models_seeds_tracks"]
    assert not common["reset_policy"]["hidden_repairs"]
    assert not common["reset_policy"]["prefix_is_policy_success"]
    assert not common["perception"]["simulator_truth_fallback"]


def test_qualification_gates_cannot_be_relaxed_or_pooled(documents):
    _, common, skills, _ = documents
    minima = {"reach_object": (0.80, 0.65), "pick_object": (0.75, 0.60),
              "place_held_object": (0.75, 0.60), "push_object_to_target": (0.80, 0.65),
              "press_button": (0.90, 0.75)}
    for skill, (aggregate, worst) in minima.items():
        gate = skills[skill]["gates"]
        assert gate["state_success_min"] >= aggregate
        assert gate["state_worst_family_success_min"] >= worst
    assert skills["pick_object"]["gates"]["drop_rate_max"] <= 0.10
    assert skills["place_held_object"]["gates"]["premature_release_rate_max"] <= 0.10
    assert skills["place_held_object"]["gates"]["settled_among_nominally_placed_min"] >= 0.90
    assert skills["press_button"]["gates"]["wrong_button_activation_count_max"] == 0
    assert skills["push_object_to_target"]["gates"]["object_off_board_count_max"] == 0
    gates = common["common_gates"]
    for key in ["invalid_actions_max", "workspace_violations_max", "joint_limit_violations_max", "object_workspace_violations_max"]:
        assert gates[key] == 0
    assert gates["perception_success_min"] >= 0.65
    assert gates["perception_worst_family_success_min"] >= 0.50
    assert gates["perception_success_degradation_max"] <= 0.15
    assert gates["explicit_terminal_result_fraction_min"] == 1.0
    assert common["status"]["default_enabled"] == ["qualified"]
    assert common["status"]["state_only"] == "experimental"
    assert common["rollouts"]["test_once"] and common["rollouts"]["save_every_attempt"]
    assert "each_seed_must_pass" in common["jerk"]["aggregation"]
    assert set(common["terminal_results"]) == {
        "succeeded", "failed", "timed_out", "cancelled", "rejected_before_execution"
    }


def test_jerk_population_uses_only_eligible_training_experts(documents):
    _, common, skills, splits = documents
    train = {r["episode_id"]: r for r in splits["train"]["episodes"]}
    normalization = {r["episode_id"]: r for r in splits["train"]["normalization_inputs"]}
    for skill, config in skills.items():
        stats = config["expert_jerk"]
        expected = {eid for eid, r in train.items() if r["stream"] == "expert_success" and r["skill"] == config["source_skill"]}
        assert {s["episode_id"] for s in stats["samples"]} == expected
        assert len(stats["samples"]) == stats["episode_count"] == len(expected)
        for sample in stats["samples"]:
            eid = sample["episode_id"]
            assert sample["frame_interval"] == expected_interval(skill, train[eid])
            lo, hi = normalization[eid]["frame_interval"]
            start, end = sample["frame_interval"]
            assert lo <= start < end <= hi
            assert math.isfinite(sample["mean_jerk"]) and sample["mean_jerk"] >= 0
        p95 = float(np.percentile([s["mean_jerk"] for s in stats["samples"]], 95, method="linear"))
        assert stats["percentile_95"] == pytest.approx(p95, rel=1e-12)
        assert common["jerk"]["gate_multiplier"] <= 1.25
        assert config["gates"]["mean_action_jerk_max"] == pytest.approx(p95 * 1.25, rel=1e-12)


def test_release_archive_reproduces_jerk_and_reset_metadata(documents):
    """Read immutable bytes, not ignored working episodes or saved metrics."""
    plan, _, skills, splits = documents
    archive = ROOT / plan["inputs"]["archive"]["path"]
    if not archive.is_file() or archive.stat().st_size < 1024:
        pytest.skip("Level 4 LFS payload unavailable; pull it for independent archive readback")
    rows = {r["episode_id"]: r for s in splits.values() for r in s["episodes"]}
    requests = {}
    for skill, config in skills.items():
        for sample in config["expert_jerk"]["samples"]:
            row = rows[sample["episode_id"]]
            for suffix in ["applied_actions.npy", "action_timestamps.npy"]:
                path = f"data/demos/level4/{row['source_path']}/{suffix}"
                requests.setdefault(path, []).append((skill, sample, row))
        for split in ["validation", "test"]:
            for cell in config["evaluation"][split]["cells"]:
                for eid, seed in zip(cell["episode_ids"], cell["reset_seeds"], strict=True):
                    row = rows[eid]
                    path = f"data/demos/level4/{row['source_path']}/metadata.json"
                    requests.setdefault(path, []).append((skill, seed, row))
    ranges = np.array([f["upper"] - f["lower"] for f in plan["action"]["named_layout"]])
    seen = Counter()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name not in requests:
                continue
            seen[member.name] += 1
            data = stream.extractfile(member).read()
            suffix = Path(member.name).name
            for skill, expected, row in requests[member.name]:
                assert hashlib.sha256(data).hexdigest() == row["file_sha256"][suffix]
                if suffix == "metadata.json":
                    meta = json.loads(data)
                    assert meta["random_seed"] == expected
                    assert meta["episode_id"] == row["episode_id"]
                    assert meta["action_contract"]["named_layout"] == plan["action"]["named_layout"]
                else:
                    values = np.load(io.BytesIO(data), allow_pickle=False)
                    if suffix == "action_timestamps.npy":
                        assert np.allclose(np.diff(values), 0.034, atol=1e-8, rtol=0)
                    else:
                        start, end = expected["frame_interval"]
                        q = values[:, 3:7]
                        assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)
                        assert np.all(np.sum(q[1:] * q[:-1], axis=1) >= 0)
                        segment = values[start:end] / ranges
                        third = np.diff(segment, n=3, axis=0)
                        measured = float(np.mean(np.sqrt(np.sum(third * third, axis=1)))) if len(third) else 0.0
                        assert measured == pytest.approx(expected["mean_jerk"], rel=1e-10, abs=1e-14)
    assert set(seen) == set(requests)
    assert all(count == 1 for count in seen.values())
