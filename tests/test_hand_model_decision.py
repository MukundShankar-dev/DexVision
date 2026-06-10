from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dexvision.apps import check_hand_actuation, check_hand_model
from dexvision.sim.hand_model import inspect_hand_model, require_controllable_hand


ROOT = Path(__file__).resolve().parents[1]
DEBUG_MODEL_PATH = ROOT / "assets" / "mujoco" / "debug_hand_scene.xml"
FINAL_MODEL_PATH = ROOT / "assets" / "mujoco" / "hand_scene.xml"
SHADOW_HAND_PATH = ROOT / "assets" / "mujoco" / "menagerie" / "shadow_hand"
DECISION_DOC_PATH = ROOT / "docs" / "robot_hand_model.md"
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"


def test_debug_and_final_model_paths_are_separate() -> None:
    assert check_hand_model.DEFAULT_MODEL == Path("assets/mujoco/hand_scene.xml")
    assert check_hand_actuation.DEFAULT_MODEL == Path("assets/mujoco/hand_scene.xml")
    assert DEBUG_MODEL_PATH.is_file()
    assert FINAL_MODEL_PATH.is_file()
    assert DEBUG_MODEL_PATH != FINAL_MODEL_PATH


def test_final_hand_scene_loads_shadow_hand_adapter() -> None:
    root = ET.parse(FINAL_MODEL_PATH).getroot()

    assert root.tag == "mujoco"
    assert root.attrib["model"] == "dexvision_shadow_hand_scene"
    include = root.find("include")
    assert include is not None
    assert include.attrib["file"] == "menagerie/shadow_hand/right_hand_dexvision.xml"


def test_debug_hand_scene_remains_controllable_smoke_test_model() -> None:
    root = ET.parse(DEBUG_MODEL_PATH).getroot()
    named_joints = [joint for joint in root.findall(".//joint") if joint.get("name")]

    assert root.tag == "mujoco"
    assert root.attrib["model"] == "dexvision_debug_hand_scene"
    assert len(named_joints) == 10
    assert len(root.findall("./actuator/position")) == 10


def test_decision_doc_records_debug_only_state_without_todos() -> None:
    text = DECISION_DOC_PATH.read_text(encoding="utf-8")

    assert "assets/mujoco/debug_hand_scene.xml" in text
    assert "assets/mujoco/hand_scene.xml" in text
    assert "Shadow Hand E3M5" in text
    assert "manual visual verification: pending" in text
    assert "Apache-2.0" in text
    assert "configs/level1_teleop.yaml" in text
    assert "TODO" not in text


def test_future_teleop_config_must_not_point_to_debug_hand() -> None:
    if not TELEOP_CONFIG_PATH.exists():
        return

    text = TELEOP_CONFIG_PATH.read_text(encoding="utf-8")
    assert "assets/mujoco/hand_scene.xml" in text
    assert "debug_hand_scene.xml" not in text


def test_shadow_hand_source_and_license_are_vendored() -> None:
    assert (SHADOW_HAND_PATH / "README.md").is_file()
    assert (SHADOW_HAND_PATH / "LICENSE").is_file()
    assert (SHADOW_HAND_PATH / "right_hand.xml").is_file()
    assert (SHADOW_HAND_PATH / "right_hand_dexvision.xml").is_file()
    assert (SHADOW_HAND_PATH / "assets" / "palm.obj").is_file()

    readme = (SHADOW_HAND_PATH / "README.md").read_text(encoding="utf-8")
    license_text = (SHADOW_HAND_PATH / "LICENSE").read_text(encoding="utf-8")

    assert "Shadow Hand E3M5" in readme
    assert "Apache-2.0" in readme
    assert "Copyright 2022 Shadow Robot Company Ltd" in license_text


def test_final_shadow_hand_passes_controllable_validation() -> None:
    pytest.importorskip("mujoco")

    info = inspect_hand_model(FINAL_MODEL_PATH)
    require_controllable_hand(info)

    assert info.joint_count == 24
    assert info.actuator_count == 20
    assert "rh_palm" in info.body_names
    assert "rh_ffdistal" in info.body_names
    assert "rh_thdistal" in info.body_names
    assert any(actuator.target_type == "tendon" for actuator in info.actuators)
