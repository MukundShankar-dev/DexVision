"""Executable acceptance checks for the Level 5.0 configuration-only freeze."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    text = (ROOT / path).read_text(encoding="utf-8")
    return json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)


@pytest.fixture(scope="module")
def plan():
    return load("configs/level5/learning.yaml")


def assert_readiness(probe: dict, audit: dict) -> None:
    """Evaluate numerical evidence, never just a saved 'sufficient' label."""
    assert probe["dataset_sufficient"] and probe["status"] == "sufficient"
    assert not probe["decision_uses_test_data"]
    assert probe["test_episode_count_inspected"] == 0
    tranches = probe["tranches"]
    assert [t["episodes_per_training_cell"] for t in tranches] == [4, 8, 16]
    gates = probe["readiness_gates"]
    latest = tranches[-1]
    for tranche in tranches:
        assert tranche["training_episode_count"] == (
            tranche["episodes_per_training_cell"] * tranche["training_cell_count"]
        )
        assert tranche["validation_rollout_count"] == len(tranche["validation_episode_ids"])
        assert tranche["validation_episode_ids"] == probe["validation_episode_ids"]
        assert tranche["aggregate_validation_success"] == pytest.approx(
            tranche["validation_success_count"] / tranche["validation_rollout_count"]
        )
        assert tranche["worst_cell_validation_success"] == min(
            tranche["validation_success_by_cell"].values()
        )
    assert latest["aggregate_validation_success"] >= gates["minimum_aggregate_validation_success"]
    assert latest["worst_cell_validation_success"] >= gates["minimum_worst_cell_validation_success"]
    assert latest["invalid_action_count"] <= gates["maximum_invalid_actions"] == 0
    assert latest["safety_violation_count"] <= gates["maximum_safety_violations"] == 0
    delta = latest["aggregate_validation_success"] - tranches[-2]["aggregate_validation_success"]
    assert delta == pytest.approx(probe["improvement_8_to_16"])
    assert delta < gates["material_improvement_fraction"]
    assert audit["passed"] and not audit["issues"]
    nominal = audit["nominal_coverage"]
    assert nominal["automated_requirements_passed"]
    similarity = nominal["similarity_audit"]
    assert similarity["passed"] and not similarity["issues"]
    assert similarity["duplicate_exact_action_trajectory_count"] == 0
    assert similarity["minimum_descriptor_l2_distance"] >= (
        similarity["required_minimum_descriptor_l2_distance"]
    )


def test_readiness_uses_measured_scaling_and_final_audit(plan):
    assert_readiness(plan["readiness"]["scaling_report_embedded"],
                     load("datasets/level4-v1/audit_report.json"))
    assert "not a standalone" in plan["readiness"]["historical_membership_note"]


@pytest.mark.parametrize("failure", ["still_improving", "low_success", "safety", "test_tuning", "duplicates"])
def test_readiness_fails_closed_on_bad_evidence(plan, failure):
    probe = copy.deepcopy(plan["readiness"]["scaling_report_embedded"])
    audit = load("datasets/level4-v1/audit_report.json")
    if failure == "still_improving":
        previous = probe["tranches"][1]
        previous["validation_success_count"] = 66
        previous["aggregate_validation_success"] = 66 / 72
        first_cell = next(iter(previous["validation_success_by_cell"]))
        previous["validation_success_by_cell"][first_cell] = 0.25
        previous["worst_cell_validation_success"] = 0.25
        probe["improvement_8_to_16"] = 5 / 72
    elif failure == "low_success":
        latest = probe["tranches"][-1]
        latest["validation_success_count"] = 36
        latest["aggregate_validation_success"] = 0.5
        latest["validation_success_by_cell"] = dict.fromkeys(
            latest["validation_success_by_cell"], 0.5
        )
        latest["worst_cell_validation_success"] = 0.5
        probe["improvement_8_to_16"] = -0.5
    elif failure == "safety":
        probe["tranches"][-1]["safety_violation_count"] = 1
    elif failure == "test_tuning":
        probe["decision_uses_test_data"] = True
    else:
        audit["nominal_coverage"]["similarity_audit"]["minimum_descriptor_l2_distance"] = 0
    with pytest.raises(AssertionError):
        assert_readiness(probe, audit)


def test_input_digests_and_config_lock(plan):
    def check(pin):
        p = Path(pin["path"])
        assert not p.is_absolute() and ".." not in p.parts
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == pin["sha256"]

    for key, pin in plan["inputs"].items():
        if key == "splits":
            for item in pin.values():
                check(item)
        elif key not in {"archive", "dataset_digest"}:
            check(pin)
    manifest = load("datasets/level4-v1/manifest.json")
    assert plan["inputs"]["archive"]["sha256"] == manifest["archive"]["sha256"]
    assert plan["inputs"]["dataset_digest"] == manifest["dataset_digest"]
    probe = plan["readiness"]["scaling_report"]
    # Original probe need not be unpacked in a metadata-only checkout.
    if (ROOT / probe["path"]).is_file():
        check(probe)
        assert load(probe["path"]) == plan["readiness"]["scaling_report_embedded"]
    checked = set()
    for line in (ROOT / "configs/level5/SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        check({"path": relative, "sha256": digest})
        assert relative not in checked
        checked.add(relative)
    expected = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs/level5").rglob("*.yaml")}
    assert checked == expected | {"docs/level5_learning_plan.md"}


def test_split_ownership_normalization_and_replacement_membership(plan):
    seen_ids, sessions = set(), set()
    for split, pin in plan["inputs"]["splits"].items():
        document = json.loads((ROOT / pin["path"]).read_text())
        payload = {k: v for k, v in document.items() if k != "manifest_digest"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert document["manifest_digest"] == pin["manifest_digest"] == digest
        assert document["status"] == "frozen" and document["dataset_audit_passed"]
        rows = document["episodes"]
        ids = {r["episode_id"] for r in rows}
        owners = {r["recording_session_id"] for r in rows}
        assert not ids & seen_ids and not owners & sessions
        seen_ids |= ids
        sessions |= owners
        assert all(r["split"] == split and r["audit_passed"] for r in rows)
        by_id = {r["episode_id"]: r for r in rows}
        if split != "train":
            assert not document["normalization_inputs"]
        else:
            assert len(document["normalization_inputs"]) == 416
            for entry in document["normalization_inputs"]:
                row = by_id[entry["episode_id"]]
                assert row["stream"] == "expert_success" and row["source"] == "scripted"
                assert entry["frame_interval"] == row["training_target_interval"]
                assert entry["episode_digest"] == row["episode_digest"]
    assert len(seen_ids) == 1112


def test_feature_names_are_resolvable_and_exclude_scene_joints(plan):
    rows = load("datasets/level4-v1/splits/train.json")["episodes"]
    groups = plan["observations"]["feature_groups"]
    assert sum(g["shape"][0] for g in groups) == plan["observations"]["common_width"] == 183
    names = [name for g in groups for name in g["names"]]
    assert len(names) == len(set(names))
    assert not any("start_button_joint" in name or "block_large" in name for name in names)
    hand_groups = {g["group"]: g for g in groups}
    for row in rows:
        layouts = row["observation_schema"]["layouts"]
        for field in ["finger_joint_positions", "finger_joint_velocities"]:
            assert hand_groups[field]["names"] == [
                f"{field}.{name}" for name in layouts[field]["names"] if name.startswith("rh_")
            ]
            assert hand_groups[field]["shape"] == [24]
        assert len([n for n in layouts["robot_qvel"]["names"] if n.startswith("rh_base_freejoint/")]) == 6
    assert not plan["observations"]["background_state_allowed"]
    assert not plan["observations"]["tracking_quality_allowed"]
    assert "t_minus_1" in plan["observations"]["time_alignment"]
    assert "t-1" in plan["observations"]["previous_safety_masks"]
    assert plan["data"]["baseline_streams"] == ["expert_success"]
    assert plan["data"]["normalization"]["owners"] == "train.json:normalization_inputs"


def test_action_heads_training_selection_and_claims_are_bounded(plan):
    base = load("configs/level4_dataset.yaml")
    action = plan["action"]
    assert action["named_layout"] == base["action_contract"]["named_layout"]
    assert action["phase_relevance"] == base["phase_contract"]["action_relevance_masks"]
    assert action["control_interval_s"] == pytest.approx(
        action["simulation_steps_per_action"] * action["simulation_timestep_s"]
    )
    widths = plan["models"]["baseline"]["heads"]
    assert sum(widths.values()) == 26 and widths["base_rotation_vector"] == 3
    assert not plan["models"]["baseline"]["expert_at_execution"]
    assert plan["models"]["expert_residual"]["enabled"] is False
    # Saturated heads remain within declared rate bounds before workspace checks.
    limits = np.array([f["max_change_per_sample"] for f in action["named_layout"]])
    assert np.all(np.abs(np.tanh(np.array([-1e9, 0, 1e9]))[:, None] * limits) <= limits)
    training = plan["training"]
    assert len(set(training["seeds"])) == 3
    assert training["max_epochs"] == 100 and training["batch_size"] == 256
    select = training["selection"]
    assert select["rollout_split"] == "validation" and select["no_test_selection"]
    assert select["test_evaluations_per_selected_checkpoint_per_track"] == 1
    assert select["sort_keys"][-1] == "epoch_ascending"
    assert "test" not in training["early_stop"]["monitor"]
    ablation = plan["ablations"]
    assert len(set(ablation["arms"])) == 4
    assert ablation["residual_never_claims_standalone"]
    assert ablation["material_improvement"]["required_splits"] == ["validation", "single_predeclared_test"]
    assert plan["artifacts"]["existing_output"] == "refuse_overwrite"
    assert not plan["optional_dial"]["enabled"]
    assert not (ROOT / "configs/level5/skills/rotate_dial.yaml").exists()
