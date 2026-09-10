"""Read-only selection, reporting, CLI failure and saved-file validation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dexvision.apps.audit_level4_dataset import main
from dexvision.evaluation import dataset_audit
from dexvision.evaluation.dataset_audit import (
    audit_level4_dataset, select_audit_episodes, summarize_records,
)
from dexvision.evaluation.split_audit import file_inventory

ROOT = Path(__file__).resolve().parents[1]


def test_quarantine_and_failed_reviews_cannot_enter_active_streams():
    requirements = yaml.safe_load((ROOT / "configs/level4_dataset.yaml").read_text())
    def episode(eid, **changes):
        return SimpleNamespace(**{"episode_id": eid, "session_id": eid,
            "expert_accepted": True, "source": "scripted", "metadata": {}, "review": None,
            **changes})
    prefix = requirements["level4_5b_procedural_expansion"]["diagnostic_quarantine"]["excluded_session_prefixes"][0]
    candidates = [episode("nominal"), episode("diagnostic", session_id=prefix + "session"),
                  episode("failed", expert_accepted=False), episode("legacy", source="teleoperation"),
                  episode("correction", metadata={"level4_6": {"version": "v1"}}),
                  episode("superseded", metadata={"level4_6": {"version": "v1"}})]
    active, excluded = select_audit_episodes(candidates, requirements, frozenset({"superseded"}))
    assert [e.episode_id for e in active] == ["nominal", "correction"]
    assert len(excluded) == 4


def test_segment_counts_do_not_inflate_recorded_episode_counts():
    row = {"skill": "pick_place_sequence", "recording_session_id": "a", "entity_id": "block",
           "target_id": "bin", "outcome": "success",
           "phase_intervals": [{"phase": "approach"}, {"phase": "acquire"}],
           "segments": [{"skill_name": "reach_object", "source_phases": ["approach"]},
                        {"skill_name": "pick_object", "source_phases": ["acquire"]}]}
    counts = summarize_records([row])
    assert sum(counts["episodes"]["skill"].values()) == 1
    assert sum(counts["segments"]["skill"].values()) == 2
    assert counts["phase_intervals"]["phase"] == {"acquire": 1, "approach": 1}


def test_cli_missing_dataset_is_actionable_and_nonzero(tmp_path, capsys):
    assert main(["--dataset-dir", str(tmp_path / "missing"), "--output-dir",
                 str(tmp_path / "audit")]) == 2
    assert "session manifest does not exist" in capsys.readouterr().err


@pytest.mark.parametrize("output", ["data/demos/level4/audit", "data/visual/level4/audit", "datasets/audit"])
def test_output_cannot_modify_preserved_sources(output):
    with pytest.raises(ValueError, match="outside source"):
        audit_level4_dataset(config_path="configs/level4_dataset.yaml",
                            splits_path="configs/level4_splits.yaml",
                            dataset_dir="data/demos/level4", output_dir=output)


def test_existing_output_is_never_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    with pytest.raises(ValueError, match="already exists"):
        audit_level4_dataset(config_path="configs/level4_dataset.yaml",
                            splits_path="configs/level4_splits.yaml",
                            dataset_dir="data/demos/level4", output_dir=tmp_path)
    assert marker.read_text() == "original"


def test_schema_corruption_is_detected_without_writing_source(tmp_path):
    from test_level4_episode_schema import _write_level4_episode

    path = tmp_path / "episode"
    _write_level4_episode(path)
    episode = SimpleNamespace(path=path)
    actions = np.load(path / "applied_actions.npy")
    actions[0, 0] = np.nan
    np.save(path / "applied_actions.npy", actions)
    before = file_inventory(path)
    with pytest.raises(ValueError):
        dataset_audit.audit_episode(episode, root=tmp_path,
            config_path=ROOT / "configs/level4_dataset.yaml",
            workcell_config=ROOT / "configs/workcell.yaml", requirements={})
    assert file_inventory(path) == before


def test_recomputed_failure_overrides_saved_acceptance(tmp_path, monkeypatch):
    from test_level4_episode_schema import _write_level4_episode

    path = tmp_path / "episode"
    _write_level4_episode(path)
    # Stop immediately after the independent replay returns: this tests that
    # acceptance is freshly recomputed, not copied from a review sidecar.
    called = []
    def reject(*args, **kwargs):
        called.append(args[0])
        raise ValueError("independent replay rejected")
    monkeypatch.setattr(dataset_audit, "audit_scripted_episode", reject)
    before = file_inventory(path)
    with pytest.raises(ValueError, match="independent replay rejected"):
        dataset_audit.audit_episode(SimpleNamespace(path=path), root=tmp_path,
            config_path=ROOT / "configs/level4_dataset.yaml",
            workcell_config=ROOT / "configs/workcell.yaml", requirements={})
    assert called == [path]
    assert file_inventory(path) == before


def test_plausible_but_wrong_pose_and_class_fail_truth_audit():
    from dexvision.evaluation.dataset_audit import audit_annotation_truth

    annotation = {"object_id": "block", "class_id": "rigid_cuboid", "category": "cuboid",
                  "pose_6d": {"translation_m": [0, 0, 0], "rotation_matrix": np.eye(3).tolist()},
                  "visible_pixels": 0, "visibility": "occluded_or_out_of_frame",
                  "source": "simulator_ground_truth", "confidence": 1.0}
    truth = {"block": {"class_id": "rigid_cuboid", "category": "cuboid",
                       "position": [0, 0, 0], "rotation": np.eye(3)}}
    assert not audit_annotation_truth({"annotations": [annotation]}, truth)
    annotation["pose_6d"]["translation_m"] = [0.01, 0, 0]
    annotation["class_id"] = "wrong"
    issues = audit_annotation_truth({"annotations": [annotation]}, truth)
    assert any("pose" in issue for issue in issues)
    assert any("class_id" in issue for issue in issues)
