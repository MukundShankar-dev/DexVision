from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "level4_dataset.yaml"
PLAN_PATH = ROOT / "docs" / "level4_dataset_plan.md"
REQUIRED_SKILLS = {
    "reach_object",
    "pick_object",
    "place_held_object",
    "push_object_to_target",
    "press_button",
}
REQUIRED_PHASES = {
    "approach",
    "acquire",
    "lift",
    "stabilize",
    "transport",
    "place",
    "release",
    "settle",
    "push_contact",
    "fixture_contact",
    "retract",
}
SPLITS = {"train", "validation", "test"}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_plan_files_freeze_scope_and_sources() -> None:
    config = load_config()
    plan = PLAN_PATH.read_text(encoding="utf-8")

    assert config["version"] == "level4/workcell-dataset-plan-v2"
    assert config["freeze"]["status"] == "requirements_frozen"
    assert config["freeze"]["collection_started"] is False
    assert "No full-scale Level 4 collection has started under this schema" in (
        " ".join(plan.split())
    )
    for source in (
        "docs/level3_results.md",
        "docs/level3_evaluation_protocol.md",
        "datasets/dexvision_level2_v1_manifest.json",
        "docs/task_environment.md",
    ):
        assert source in config["evidence_sources"]
        assert source in plan


def test_required_skills_have_typed_contracts_and_executable_metrics() -> None:
    skills = load_config()["skills"]

    assert set(skills) == REQUIRED_SKILLS
    for skill_name, skill in skills.items():
        assert skill["enabled"] is True, skill_name
        assert skill["preconditions"], skill_name
        assert skill["goal_fields"], skill_name
        assert skill["terminal_state_fields"], skill_name
        assert skill["success_metric"]["conditions"], skill_name
        assert skill["success_metric"]["required_consecutive_samples"] > 0
        assert skill["failure_rules"], skill_name
        assert skill["timeout_steps"] > 0
        for field_name, field in skill["goal_fields"].items():
            assert field["type"], (skill_name, field_name)
            assert "required" in field, (skill_name, field_name)
            assert field["units"], (skill_name, field_name)
            assert field["coordinate_frame"], (skill_name, field_name)
            assert "allowed_ids" in field or "range" in field, (
                skill_name,
                field_name,
            )

    assert load_config()["optional_skills"]["rotate_dial"]["enabled"] is False


def test_workcell_vocabulary_and_reset_bounds_are_frozen() -> None:
    config = load_config()
    workcell = config["workcell"]
    objects = workcell["objects"]

    assert set(workcell["object_families"]) == {"cuboid", "cylinder", "flat_puck"}
    assert len(objects) == 6
    assert Counter(item["family"] for item in objects.values()) == {
        "cuboid": 2,
        "cylinder": 2,
        "flat_puck": 2,
    }
    assert {item["split_role"] for item in objects.values()} == {
        "training_pool",
        "held_out_test_only",
    }
    assert all(item["geometry"] and item["mass_kg"] > 0 for item in objects.values())
    assert set(workcell["fixtures"]) == {"start_button"}
    assert set(workcell["targets"]) == {
        "return_bin_left",
        "return_bin_right",
        "inspection_pad",
        "setup_slot_a",
        "setup_slot_b",
    }
    assert workcell["targets"]["return_bin_right"]["split_role"] == "held_out_test_only"
    bounds = workcell["safe_workspace"]
    assert bounds["coordinate_frame"] == "mujoco_world"
    assert bounds["units"] == "m"
    assert bounds["margin_m"] > 0
    assert all(low < high for low, high in zip(bounds["min"], bounds["max"]))
    assert all(item["position_range_m"] for item in config["reset_ranges"].values())


def test_coverage_cells_have_exclusive_split_minima_and_match_budget() -> None:
    config = load_config()
    coverage = config["coverage_cells"]
    budgets = config["episode_budget"]["groups"]

    assert len(coverage) == 74
    assert len({cell["id"] for cell in coverage}) == len(coverage)
    totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for cell in coverage:
        minima = cell["minimum_accepted_by_split"]
        assert set(minima) == SPLITS
        assert cell["split_owner"] in SPLITS
        positive = {split for split, value in minima.items() if value > 0}
        assert positive == {cell["split_owner"]}, cell["id"]
        assert all(isinstance(value, int) and value >= 0 for value in minima.values())
        assert cell["required_source"] in config["source_mix"]["categories"]
        totals[cell["data_group"]] += sum(minima.values())
        counts[cell["data_group"]] += 1

    assert counts == {
        "reach": 10,
        "pick_place": 30,
        "push": 12,
        "button": 10,
        "failure_correction": 12,
    }
    assert totals == {
        group: values["minimum_new_accepted"] for group, values in budgets.items()
    }
    assert sum(totals.values()) == config["episode_budget"]["required_total_minimum"] == 114
    assert config["episode_budget"]["required_total_planning_maximum"] == 140
    source_totals: Counter[str] = Counter()
    for cell in coverage:
        source_totals[cell["required_source"]] += sum(
            cell["minimum_accepted_by_split"].values()
        )
    assert {
        source: source_totals[source]
        for source in config["source_mix"]["categories"]
    } == config["source_mix"]["minimum_accepted_by_source"]
    excluded = config["coverage_exclusions"]["cells"]
    assert len(excluded) == 8
    assert {cell["id"] for cell in excluded}.isdisjoint(
        {cell["id"] for cell in coverage}
    )
    assert config["episode_budget"]["optional_dial_minimum"] == 0
    assert config["counting_rules"]["pick_place_sequence_is_one_episode"] is True
    assert config["counting_rules"]["report_episode_and_segment_counts"] is True


def test_sessions_and_held_out_conditions_cannot_leak() -> None:
    config = load_config()
    sessions = config["sessions"]

    assert sessions["minimum_genuine_sessions"] == 4
    assert [slot["split"] for slot in sessions["required_slots"]] == [
        "train",
        "train",
        "validation",
        "test",
    ]
    assert sessions["whole_session_split_only"] is True
    assert sessions["fresh_process_and_calibration_required"] is True
    assert sessions["additional_session_assignment"] == "before_collection_or_inspection"
    assert config["split_policy"]["normalization_source"] == "train_only"
    assert config["split_policy"]["test_influences_tuning"] is False
    assert config["workcell"]["targets"]["setup_slot_b"]["split_role"] == (
        "validation_and_test"
    )

    held_out_objects = {
        object_id
        for object_id, item in config["workcell"]["objects"].items()
        if item["split_role"] == "held_out_test_only"
    }
    for cell in config["coverage_cells"]:
        if cell.get("object_id") in held_out_objects:
            assert cell["split_owner"] == "test"
        if cell.get("target_id") == "return_bin_right":
            assert cell["split_owner"] == "test"


def test_action_stages_are_reconstructable_and_safety_bounded() -> None:
    action = load_config()["action_contract"]
    fields = action["named_layout"]

    assert action["layout_version"] == "level1.13/full-action-v1"
    assert len(fields) == 27
    assert [field["index"] for field in fields] == list(range(27))
    assert action["stages"]["requested"]["request_sources"] == [
        "operator",
        "script",
        "policy",
    ]
    assert action["stages"]["commanded"]["before_safety_handling"] is True
    assert action["stages"]["applied"]["after_safety_handling"] is True
    assert action["record_prior_commanded"] is True
    assert action["record_prior_applied"] is True
    assert action["residual_target_derivation"]["deterministic"] is True
    assert all(field["lower"] < field["upper"] for field in fields)
    assert all(field["max_change_per_sample"] > 0 for field in fields)
    assert action["quaternion"]["normalization_required"] is True
    assert action["quaternion"]["sign_continuity_required"] is True
    assert action["quaternion"]["first_sample_canonical_rule"]
    reason_codes = {item["code"] for item in action["safety_reason_codes"]}
    assert {"none", "workspace_clip", "actuator_clip", "task_abort"} <= reason_codes
    assert action["per_field_safety_mask"] is True
    assert action["unsafe_motion_is_expert_target"] is False


def test_online_phases_are_causal_and_mask_every_action_group() -> None:
    config = load_config()
    phases = config["phase_contract"]
    state_machine = config["online_phase_state_machine"]

    assert set(phases["vocabulary"]) == REQUIRED_PHASES
    assert set(phases["action_relevance_masks"]) == REQUIRED_PHASES
    assert phases["irrelevant_action_behavior"] == "hold_previous_applied"
    for mask in phases["action_relevance_masks"].values():
        assert set(mask) == {"base_position", "base_orientation", "wrist", "fingers"}
        assert all(isinstance(value, bool) for value in mask.values())

    assert state_machine["causal"] is True
    assert state_machine["future_frames_allowed"] is False
    assert state_machine["audited_annotation_used_online"] is False
    assert set(state_machine["machines"]) == REQUIRED_SKILLS
    for skill_name, machine in state_machine["machines"].items():
        assert machine["initial_phase"] in REQUIRED_PHASES, skill_name
        assert machine["transitions"], skill_name
        assert all(rule["from"] in REQUIRED_PHASES for rule in machine["transitions"])
        assert all(rule["to"] in REQUIRED_PHASES for rule in machine["transitions"])
        assert all(rule["predicate"] for rule in machine["transitions"])


def test_stream_quality_visual_and_acceptance_contracts_are_explicit() -> None:
    config = load_config()
    streams = config["streams"]
    visual = config["visual_claim"]

    assert {"state", "action", "task_metrics", "phase_safety", "rgb"} <= set(streams)
    assert all(stream["required_for_release"] for stream in streams.values())
    assert streams["rgb"]["alignment"] == "episode_frame_and_state_timestamp"
    assert visual["camera_count"] == 1
    assert visual["fixed_pose_and_intrinsics"] is True
    assert visual["cross_camera_claim"] is False
    assert visual["real_world_transfer_claim"] is False
    assert set(visual["conditions"]) == {
        "nominal",
        "mild_illumination",
        "partial_occlusion",
        "bounded_distractors",
    }
    for condition in visual["conditions"].values():
        assert set(condition["minimum_source_episodes_by_split"]) == SPLITS
        assert all(value > 0 for value in condition["minimum_source_episodes_by_split"].values())
        assert condition["entity_coverage"]
    assert visual["source_episode_minima_are_additive_to_episode_budget"] is False

    storage = config["pilot"]["storage_projection"]
    assert storage["frozen_payload_handling"] == "git_lfs"
    assert storage["working_data_git_policy"] == "ignored_never_force_added"
    assert storage["existing_release_overwrite_allowed"] is False
    assert storage["release_artifacts"] == ["immutable_tar_gz", "sha256", "manifest"]

    thresholds = config["quality_thresholds"]
    assert thresholds["min_mean_tracking_confidence"] >= 0.75
    assert thresholds["max_missing_frame_fraction"] <= 0.10
    assert thresholds["require_recomputed_terminal_result"] is True
    assert thresholds["require_operator_recomputed_label_agreement"] is True
    assert config["acceptance_workflow"]["accepted_episode_mutable"] is False
    assert config["acceptance_workflow"]["append_only"] is True


def test_every_level3_failure_category_has_a_disposition_and_requirement() -> None:
    traceability = load_config()["level3_failure_traceability"]
    required_categories = {
        "task_coverage",
        "session_and_operator_provenance",
        "object_and_fixture_diversity",
        "goal_and_reset_coverage",
        "outcome_and_failure_coverage",
        "observation_and_world_state",
        "action_layout_and_safety",
        "temporal_and_distribution_hypotheses",
        "unsupported_generalization_claims",
    }

    assert set(traceability) == required_categories
    for category, entry in traceability.items():
        assert entry["disposition"] in {
            "accepted",
            "deferred_pending_evidence",
            "intentionally_unsupported",
        }, category
        assert entry["level3_evidence"], category
        assert entry["level4_requirements"], category
    temporal = traceability["temporal_and_distribution_hypotheses"]
    assert temporal["disposition"] == "deferred_pending_evidence"
    assert temporal["reconsideration_trigger"]
