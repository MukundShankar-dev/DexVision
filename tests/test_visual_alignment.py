"""Source alignment, deterministic ownership, and export boundary tests."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.logging.visual_stream import (
    audit_frame, build_visual_report, configure_render_model, export_visual_dataset,
    load_visual_config, named_joint_values, restore_visual_state, select_visual_sources,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/level4_visual_dataset.yaml"


def sources_fixture():
    config, workcell = load_visual_config(CONFIG)
    episodes, sessions = [], {}
    cells = workcell.requirements["coverage_cells"]
    objects = workcell.requirements["workcell"]["objects"]
    for split in ("train", "validation", "test"):
        for category in config["selection"]["primary_categories"]:
            cell = next(c for c in cells if c["split_owner"] == split and (
                objects.get(c.get("object_id", c.get("entity_id")), {}).get("family") == category
                or (category == "start_button" and c["data_group"] == "button")))
            entity = cell.get("object_id", cell.get("entity_id", "start_button"))
            for index in range(12):
                episode_id = f"{split}_{category}_{index:02d}"
                sessions[episode_id] = split
                episodes.append(SimpleNamespace(
                    episode_id=episode_id, session_id=episode_id, source="scripted",
                    expert_accepted=True, goal_condition_id=cell["id"],
                    metadata={"typed_goal": {"entity_id": entity}}))
    return config, workcell, episodes, sessions


def test_selection_is_deterministic_whole_session_and_condition_owned():
    config, workcell, episodes, sessions = sources_fixture()
    selected = select_visual_sources(episodes, sessions, config, workcell.requirements)
    reversed_selection = select_visual_sources(episodes[::-1], sessions, config, workcell.requirements)
    assert selected == reversed_selection
    assert len(selected) == 64
    assert len({item[3].session_id for item in selected}) == 64
    for condition, split, category, episode in selected:
        assert sessions[episode.session_id] == split
        assert category in config["selection"]["primary_categories"]


def test_selection_rejects_leakage_shortages_and_duplicate_ids():
    config, workcell, episodes, sessions = sources_fixture()
    sessions[episodes[0].session_id] = "test"
    with pytest.raises(ValueError, match="split mismatch"):
        select_visual_sources(episodes, sessions, config, workcell.requirements)
    with pytest.raises(ValueError, match="Missing source cell"):
        select_visual_sources([], {}, config, workcell.requirements)
    with pytest.raises(ValueError, match="Duplicate source"):
        select_visual_sources([episodes[0]]*2, {episodes[0].session_id: "train"},
                              config, workcell.requirements)


def test_quarantined_and_unaccepted_sources_are_never_selected():
    config, workcell, episodes, sessions = sources_fixture()
    rejected = deepcopy(episodes[0])
    rejected.episode_id = "000_rejected"
    rejected.expert_accepted = False
    quarantined = deepcopy(episodes[0])
    quarantined.episode_id = "000_quarantined"
    quarantined.session_id = (workcell.requirements["level4_5b_procedural_expansion"]
                             ["diagnostic_quarantine"]["excluded_session_prefixes"][0]+"x")
    selected = select_visual_sources([rejected, quarantined, *episodes], sessions,
                                     config, workcell.requirements)
    assert not any(item[3].episode_id.startswith("000") for item in selected)


def test_exact_named_saved_state_restoration_and_bad_timestamps():
    import mujoco

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="one" type="slide" axis="1 0 0"/><joint name="two" type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    position = SimpleNamespace(source_array="robot_states", column_range=(0, 2),
                               column_indices=(), names=("two", "one"))
    velocity = SimpleNamespace(source_array="robot_states", column_range=(2, 4),
                               column_indices=(), names=("two", "one"))
    row = np.array([.2, .1, .4, .3])
    np.testing.assert_array_equal(named_joint_values(model, position, row, velocity=False), [.1, .2])
    loaded = SimpleNamespace(observation_schema=SimpleNamespace(layouts={
        "robot_qpos": position, "robot_qvel": velocity}), episode=SimpleNamespace(
        robot_states=row[None], state_timestamps=np.array([1.5]), timestamps=np.array([1.5])))
    assert restore_visual_state(model, data, loaded, 0) == 1.5
    np.testing.assert_array_equal(data.qpos, [.1, .2])
    np.testing.assert_array_equal(data.qvel, [.3, .4])
    loaded.episode.state_timestamps[0] = 2.
    with pytest.raises(ValueError, match="alignment"):
        restore_visual_state(model, data, loaded, 0)
    position.names = ("unknown", "one")
    with pytest.raises(ValueError, match="layout disagrees"):
        named_joint_values(model, position, row, velocity=False)


def frame_fixture():
    masks = np.array([[0, 1], [0, 1]], np.uint16)
    frame = {"source_frame_index": 1, "timestamp": .2, "split": "train",
             "rgb_pixel_sha256": "pixel_digest", "annotations": [{"mask_value": 1,
             "box_xyxy": [1, 0, 2, 2], "visible_pixels": 2,
             "pose_6d": {"translation_m": [0., 0., 0.], "rotation_matrix": np.eye(3).tolist()}}]}
    return frame, masks


def test_frame_audit_rejects_cross_split_duplicate_and_annotation_misalignment():
    frame, masks = frame_fixture()
    timestamps = np.array([.1, .2])
    owners = {}
    audit_frame(frame, masks, source_timestamps=timestamps, image_owners=owners)
    frame["split"] = "test"
    with pytest.raises(ValueError, match="Duplicate image"):
        audit_frame(frame, masks, source_timestamps=timestamps, image_owners=owners)
    frame["split"] = "train"
    frame["annotations"][0]["box_xyxy"] = [0, 0, 2, 2]
    with pytest.raises(ValueError, match="disagreement"):
        audit_frame(frame, masks, source_timestamps=timestamps, image_owners={})
    frame["timestamp"] = .3
    with pytest.raises(ValueError, match="alignment"):
        audit_frame(frame, masks, source_timestamps=timestamps, image_owners={})


def test_report_exposes_missing_cells_and_never_claims_manual_completion():
    config, workcell = load_visual_config(CONFIG)
    report = build_visual_report(config, workcell, [], [])
    assert len(report["missing_cells"]) == 12
    assert not report["automated_passed"]
    assert not report["checkpoint_complete"]


def test_render_model_hides_held_out_background_and_restores_geometry():
    import mujoco

    config, workcell = load_visual_config(CONFIG)
    model = mujoco.MjModel.from_xml_path(str(workcell.model_path))
    initial = {"entity_positions_m": {name: model.body(name).pos.tolist()
               for name in (*workcell.targets, *workcell.fixtures)}}
    initial["entity_positions_m"]["start_button"] = [.13, -.1, .2]
    metadata = {"task_config": {"initial_state": initial, "procedural_variation": {
        "object_scale_multiplier": 1.05}}, "typed_goal": {"object_id": "block_small"}}
    size = model.geom("block_small_geom").size.copy()
    entities, hidden = configure_render_model(model, metadata, workcell, config, "train")
    assert "block_large" in hidden and "return_bin_right" in hidden
    assert not set(hidden) & {e.object_id for e in entities}
    assert model.geom("block_large_geom").rgba[3] == 0
    np.testing.assert_allclose(model.geom("block_small_geom").size, size*1.05)
    np.testing.assert_allclose(model.body("start_button").pos, [.13, -.1, .2])


def test_export_refuses_existing_output_before_touching_sources(tmp_path):
    output = tmp_path / "visual"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged")
    with pytest.raises(ValueError, match="already exists"):
        export_visual_dataset(config_path=CONFIG, dataset_dir=tmp_path / "missing", output_dir=output)
    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize(("field", "value"), [("sampling_stride", 0), ("sampling_stride", True)])
def test_visual_config_rejects_invalid_sampling(tmp_path, field, value):
    import yaml

    config, _ = load_visual_config(CONFIG)
    config["workcell_config"] = str(ROOT / "configs/workcell.yaml")
    config[field] = value
    path = tmp_path / "visual.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="sampling_stride"):
        load_visual_config(path)


def test_visual_config_rejects_weakened_coverage_and_excess_occlusion(tmp_path):
    import yaml

    config, _ = load_visual_config(CONFIG)
    config["workcell_config"] = str(ROOT / "configs/workcell.yaml")
    path = tmp_path / "visual.yaml"
    config["selection"]["minimum_episodes_by_condition_split"]["test"] = 1
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="quotas"):
        load_visual_config(path)
    config["selection"]["minimum_episodes_by_condition_split"]["test"] = 4
    config["conditions"]["partial_occlusion"]["maximum_target_mask_occlusion_fraction"] = .5
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Occlusion"):
        load_visual_config(path)


def test_report_does_not_count_sources_without_exported_frames():
    config, workcell = load_visual_config(CONFIG)
    sources = [{"episode_id": str(i), "condition": condition, "split": split}
               for condition in config["conditions"]
               for split in ("train", "validation", "test") for i in range(8)]
    report = build_visual_report(config, workcell, sources, [])
    assert all(cell["source_episodes"] == 0 for cell in report["cells"])
