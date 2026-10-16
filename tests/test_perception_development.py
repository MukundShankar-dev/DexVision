"""Separate repair split ownership, immutable releases and one-use confirmation."""
from __future__ import annotations

import json

import pytest

from dexvision.evaluation.perception_development import evaluate_confirmation
from dexvision.learning.perception_development import development_rows
from dexvision.logging.perception_development import load_protocol, package_dataset, verify_dataset
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.perception_baseline import save_artifact_index


def protocol(tmp_path, role="development"):
    p = tmp_path/"protocol.json"
    version = "level5/perception-development-v1" if role == "development" else "level5/perception-confirmation-v3"
    write_json(p, {"version": version, "pins": {}, "dataset_role": role,
                   "splits": {"train": {}, "validation": {}} if role == "development" else {"test": {}}})
    p.with_suffix(".sha256").write_text(f"{digest_file(p)}  {p.name}\n")
    return p


def dataset(tmp_path, p, rows, *, collision=False):
    root = tmp_path/"dataset"
    root.mkdir()
    write_json(root/"manifest.json", {"protocol_sha256": digest_file(p), "training_eligible": True,
        "splits": {"train": {}, "validation": {}}, "historical_pixel_collisions": 0,
        "cross_split_pixel_collisions": int(collision), "no_filtered_attempts": True})
    write_json(root/"owners.json", {r["source_episode_id"]: {} for r in rows})
    (root/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    save_artifact_index(root)
    return root


def row(split, episode, session):
    return {"split": split, "source_episode_id": episode, "recording_session_id": session}


def test_fresh_train_validation_ownership_and_release_no_overwrite(tmp_path):
    p = protocol(tmp_path)
    root = dataset(tmp_path, p, [row("train", "a", "sa"), row("validation", "b", "sb")])
    rows = development_rows({"development_protocol": str(p), "visual_root": str(root)})
    assert [r["split"] for r in rows] == ["train", "validation"]
    archive = tmp_path/"development-v1.tar.gz"
    info = package_dataset(p, root, archive)
    assert digest_file(archive) == info["archive_sha256"]
    with pytest.raises(ValueError, match="exists"):
        package_dataset(p, root, archive)


@pytest.mark.parametrize("rows", [
    [row("test", "a", "sa")],
    [row("train", "a", "sa"), row("validation", "a", "sb")],
    [row("train", "a", "same"), row("validation", "b", "same")],
])
def test_test_or_cross_split_ownership_is_never_fitted(tmp_path, rows):
    p = protocol(tmp_path)
    root = dataset(tmp_path, p, rows)
    with pytest.raises(ValueError, match="Test row|Cross-split"):
        development_rows({"development_protocol": str(p), "visual_root": str(root)})


def test_consumed_confirmation_rejected_before_rows_are_opened(tmp_path):
    p = protocol(tmp_path, "confirmation")
    with pytest.raises(ValueError, match="development"):
        development_rows({"development_protocol": str(p), "visual_root": str(tmp_path/"absent")})
    (tmp_path/"confirmation_attempt.json").write_text("consumed")
    with pytest.raises(ValueError, match="consumed"):
        evaluate_confirmation(p, tmp_path/"absent", tmp_path/"different-output")
    assert not (tmp_path/"different-output").exists()


def test_cross_split_pixels_and_changed_payload_fail_closed(tmp_path):
    p = protocol(tmp_path)
    root = dataset(tmp_path, p, [], collision=True)
    with pytest.raises(ValueError, match="independence"):
        verify_dataset(root, p)
    (root/"frames.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_dataset(root, p)


def test_changed_source_invalidates_frozen_protocol(tmp_path):
    p = protocol(tmp_path)
    source = tmp_path/"source.py"
    source.write_text("original")
    spec = json.loads(p.read_text())
    spec["pins"] = {str(source): digest_file(source)}
    write_json(p, spec)
    p.with_suffix(".sha256").write_text(f"{digest_file(p)}  {p.name}\n")
    load_protocol(p)
    source.write_text("different")
    with pytest.raises(ValueError, match="changed"):
        load_protocol(p)


def test_raster_amendment_rejects_consumed_source_without_reading_pixels(tmp_path, monkeypatch):
    import yaml
    import dexvision.logging.perception_raster as module
    config = tmp_path/"raster.yaml"
    config.write_text(yaml.safe_dump({"role": "development", "rasterization": {"offscreen_samples": 0},
        "source_protocol": "consumed/protocol.json", "source_dataset": "consumed/data"}))
    config.with_suffix(".sha256").write_text(f"{digest_file(config)}  {config.name}\n")
    monkeypatch.setattr(module, "load_protocol", lambda _: {"dataset_role": "confirmation"})
    monkeypatch.setattr(module, "verify_dataset", lambda *_: {"splits": {"test": {}}})
    with pytest.raises(ValueError, match="unconsumed"):
        module.freeze_development(config, tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_pixel_center_samples_set_before_renderer_without_gpu(tmp_path, monkeypatch):
    import numpy as np
    import mujoco
    from types import SimpleNamespace
    import dexvision.logging.perception_raster as module
    import dexvision.perception.render_annotations as annotations
    p = tmp_path/"p.json"
    p.write_text("frozen")
    calibration = tmp_path/"camera.json"
    calibration.write_text('{"version": "level4/visual-camera-v1"}')
    index = tmp_path/"prior.jsonl"
    index.write_text("")
    case = {"episode_id": "fresh", "session_id": "session", "split": "train",
            "skill": "press_button", "entity_id": "button", "condition": "nominal"}
    spec = {"cases": [case], "case_count": 1, "role": "development", "version": "test",
            "visual_config": "unused", "camera_calibration": str(calibration),
            "prior_frame_index": str(index), "prior_frame_indices": [], "sampling_stride": 1,
            "rasterization": {"offscreen_samples": 0}, "splits": {"train": {}, "validation": {}},
            "disclosure": "unit fixture"}
    monkeypatch.setattr(module, "load_protocol", lambda _: spec)
    monkeypatch.setattr(module, "load_visual_config", lambda _: (
        {"camera": {"id": "fixed"}, "conditions": {"nominal": {}}}, SimpleNamespace(model_path="model")))
    monkeypatch.setattr(module, "configure_render_model", lambda *_: ((), []))

    def source(protocol, case, child):
        child.mkdir(parents=True)
        np.savez(child/"states.npz", qpos=[[0.]], qvel=[[0.]], timestamps=[0.], phases=["fixture_contact"])
        return {"source_reason": "scripted_complete", "source_success": True, "frame_count": 1}

    class Model:
        @staticmethod
        def from_xml_path(_):
            return SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))

    class Renderer:
        def __init__(self, model, **kwargs):
            assert model.vis.quality.offsamples == 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(module, "generate_source", source)
    monkeypatch.setattr(mujoco, "MjModel", Model)
    monkeypatch.setattr(mujoco, "MjData", lambda _: SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1)))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *_: None)
    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    monkeypatch.setattr(annotations, "camera_calibration", lambda *_: {})
    monkeypatch.setattr(annotations, "render_frame", lambda *_args, **_kwargs: (
        np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.uint16), [], {}))
    result = module.collect_dataset(p, tmp_path/"out")
    assert result["frame_count"] == 1 and result["no_filtered_attempts"]
    assert result["rasterization"]["offscreen_samples"] == 0
