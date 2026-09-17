"""Reach reset/feature parity, real headless physics and fail-closed accounting."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.evaluate_skill import (
    apply_action, check_preconditions, create_reach_task, frozen_scenarios,
    level3_compatibility, reach_features, run_evaluation_command, run_reach_rollout,
    safe_action, summarize_rollouts,
)
from dexvision.learning.skill_datasets import vectorize_episode
from dexvision.learning.training_manifest import ReleaseSource, load_training_inputs


@pytest.fixture(scope="module")
def inputs():
    return load_training_inputs("configs/level5/skills/reach_object.yaml")


@pytest.fixture(scope="module")
def example(inputs):
    row = frozen_scenarios(inputs, "validation")[0][0]
    archive = inputs.root / inputs.plan["inputs"]["archive"]["path"]
    if not archive.exists() or archive.stat().st_size < 1000:
        pytest.skip("Release integration requires the immutable LFS archive")
    record = ReleaseSource(inputs).read_episode_records([row])[row["episode_id"]]
    return row, record


def test_live_features_match_saved_causal_features_through_replayed_actions(inputs, example):
    row, record = example
    episode = vectorize_episode(inputs, row, record)
    with create_reach_task(inputs, row, record["metadata"]) as task:
        for frame in range(8):
            world = task.workcell.get_world_state()
            features = reach_features(task, world, row, inputs,
                record["prior_commanded_actions"][frame], record["prior_applied_actions"][frame],
                record["safety_masks"][frame - 1] if frame else np.zeros(27), task._phase)
            np.testing.assert_allclose(features, episode.inputs[frame], atol=1e-8, rtol=1e-8)
            apply_action(task, record["applied_actions"][frame], inputs.plan["action"]["named_layout"])
            task.step(n_steps=17)


@pytest.mark.parametrize("field,value,key", [(0, 1.0, "workspace_violations"), (7, 2.0, "joint_limit_violations"), (2, np.nan, "invalid_actions")])
def test_unsafe_requests_keep_counts_and_finite_applied_action(inputs, field, value, key):
    prior = np.zeros(27)
    prior[2:4] = [0.2, 1]
    raw = prior.copy()
    raw[field] = value
    applied, mask, count, reason = safe_action(raw, prior, inputs.plan)
    assert count[key] > 0 and reason and mask.any()
    assert np.isfinite(applied).all()
    if key == "invalid_actions":
        np.testing.assert_array_equal(applied, prior)


def test_quaternion_normalization_and_zero_rejection(inputs):
    prior = np.zeros(27)
    prior[2:4] = [0.2, 1]
    raw = prior.copy()
    raw[3:7] = [-0.5, -0.5, 0, 0]
    applied, _, count, reason = safe_action(raw, prior, inputs.plan)
    assert np.linalg.norm(applied[3:7]) == pytest.approx(1)
    assert applied[3:7] @ prior[3:7] >= 0 and not reason and not sum(count.values())
    raw[3:7] = 0
    assert safe_action(raw, prior, inputs.plan)[3] == "invalid_action"


def test_frozen_matrix_has_30_unique_resets_and_four_conditions(inputs):
    for split in ("validation", "test"):
        scenarios = frozen_scenarios(inputs, split)
        assert len(scenarios) == 120
        assert len({r["episode_id"] for r, _, _ in scenarios}) == 30
        assert {r["split"] for r, _, _ in scenarios} == {split}
    with pytest.raises(ValueError, match="frozen"):
        frozen_scenarios(inputs, "train")


def test_missing_stale_and_unsafe_preconditions_are_rejected(inputs, example):
    row, record = example
    with create_reach_task(inputs, row, record["metadata"]) as task:
        world = task.workcell.get_world_state()
        assert all(check_preconditions(world, row, inputs.skill).values())
        stale = replace(world, timestamp=world.timestamp + 1)
        assert not all(check_preconditions(stale, row, inputs.skill).values())
        wrong = copy.deepcopy(row)
        wrong["typed_goal"]["entity_id"] = "missing"
        assert not all(check_preconditions(world, wrong, inputs.skill).values())
        unsafe = replace(world, robot=replace(world.robot, safe_neutral=False))
        assert not all(check_preconditions(unsafe, row, inputs.skill).values())


def test_real_rollout_records_unsafe_attempt_and_terminal_state(inputs, example, tmp_path):
    row, record = example

    def unsafe(features, prior, phase):
        raw = prior.copy()
        raw[0] = 0.5
        return raw, np.zeros(26)

    result = run_reach_rollout(inputs, row, record["metadata"], unsafe, condition="nominal", output_dir=tmp_path / "unsafe")
    assert result["terminal_result"] == "failed"
    assert result["terminal_reason"] == "workspace_violation"
    assert result["safety_counters"]["workspace_violations"] == 1
    data = np.load(tmp_path / "unsafe/trajectory.npz")
    assert len(data["qpos"]) == 2 and len(data["observations"]) == 1
    assert data["raw_actions"][0, 0] == 0.5 and data["applied_actions"][0, 0] == 0.22
    assert data["state_timestamps"][-1] == pytest.approx(0.034)
    assert (tmp_path / "unsafe/SHA256SUMS").is_file()
    with pytest.raises(FileExistsError):
        run_reach_rollout(inputs, row, record["metadata"], unsafe, condition="nominal", output_dir=tmp_path / "unsafe")


def test_real_success_requires_five_new_physical_dwell_samples(inputs, example, tmp_path):
    row, record = example

    def goal_policy(features, prior, phase):
        raw = prior.copy()
        raw[:7] = row["typed_goal"]["approach_pose"]
        return raw, np.zeros(26)

    result = run_reach_rollout(inputs, row, record["metadata"], goal_policy, condition="nominal", output_dir=tmp_path / "goal")
    assert result["terminal_result"] == "succeeded"
    assert result["final_metrics"]["dwell_steps"] == 5
    assert result["end_timestamp"] == pytest.approx(5 * 0.034)


def test_rejection_remains_saved_and_in_denominator(inputs, example, tmp_path, monkeypatch):
    row, record = example
    monkeypatch.setattr("dexvision.evaluation.evaluate_skill.check_preconditions", lambda *args: {"robot.safe_neutral": False})

    def never(*args):
        raise AssertionError("A rejected request must not call its policy")

    result = run_reach_rollout(inputs, row, record["metadata"], never, condition="nominal", output_dir=tmp_path / "reject")
    assert result["terminal_result"] == "rejected_before_execution"
    data = np.load(tmp_path / "reject/trajectory.npz")
    assert data["applied_actions"].shape == (0, 27) and len(data["qpos"]) == 1
    assert result["normalized_terminal_error"] is None


def synthetic_results(inputs):
    return [{"episode_id": row["episode_id"], "render_condition": condition, "cell_id": row["cell_id"],
             "object_family": "cuboid", "object_instance": row["entity_id"], "target_id": row["entity_id"],
             "target_type": "cuboid", "session": row["recording_session_id"], "boundary_class": "interior",
             "terminal_result": "succeeded", "terminal_reason": "pose_and_dwell_passed",
             "safety_counters": dict.fromkeys(("invalid_actions", "workspace_violations", "joint_limit_violations", "object_workspace_violations"), 0),
             "mean_action_jerk": 0, "normalized_terminal_error": 0} for row, _, condition in frozen_scenarios(inputs, "test")]


def test_incomplete_evidence_rejected_and_state_only_pass_never_qualified(inputs):
    results = synthetic_results(inputs)
    report = summarize_rollouts(results, inputs, "test")
    assert report["state_numerical_gates_passed"] and not report["state_qualified"]
    assert report["status"] == "experimental" and report["manual_acceptance"] == "pending"
    with pytest.raises(ValueError, match="Missing"):
        summarize_rollouts(results[:-1], inputs, "test")
    for result in results[:40]:
        result.update(terminal_result="rejected_before_execution", terminal_reason="rejected_precondition", normalized_terminal_error=None)
    report = summarize_rollouts(results, inputs, "test")
    assert report["success_rate"] == pytest.approx(2 / 3)
    assert report["normalized_terminal_error"] is None and not report["state_numerical_gates_passed"]
    results[0]["safety_counters"]["joint_limit_violations"] = 1
    assert summarize_rollouts(results, inputs, "test")["status"] == "failed"


def test_legacy_contract_incompatibility_is_explicit(inputs):
    path = Path("outputs/level3/reach_bc_v2/policy_best.pt")
    if not path.exists():
        pytest.skip("Historical checkpoint is an optional local artifact")
    report = level3_compatibility(path, inputs)
    assert not report["compatible"] and report["matching_full_action_layout"]
    assert report["compatible_level4_scenario_count"] == 0
    assert report["historical_level3_4"]["workspace_violations"] == 7


def test_future_track_rejected_before_any_rollout(tmp_path):
    with pytest.raises(ValueError, match="5.2"):
        run_evaluation_command(Path("configs/level5/skills/reach_object.yaml"), checkpoint=None, output_dir=tmp_path / "future", track="perception")
    assert not (tmp_path / "future").exists()
