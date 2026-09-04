from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dexvision.apps import inspect_workcell
from dexvision.perception.object_observations import SIMULATOR_GROUND_TRUTH
from dexvision.sim.workcell import (
    OPTIONAL_DIAL_SKILL,
    REQUIRED_SKILLS,
    Workcell,
    WorkcellError,
    load_workcell_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "workcell.yaml"
FROZEN_PATH = ROOT / "configs" / "level4_dataset.yaml"


def test_runtime_config_matches_frozen_level4_vocabulary() -> None:
    config = load_workcell_config(CONFIG_PATH)
    frozen = yaml.safe_load(FROZEN_PATH.read_text(encoding="utf-8"))

    assert config.version == "level4/workcell-v1"
    assert config.world_state_version == frozen["schema_versions"]["world_state"]
    assert config.coordinate_frame == frozen["workcell"]["coordinate_frame"]
    assert config.length_units == frozen["workcell"]["length_units"]
    assert set(config.object_ids) == set(frozen["workcell"]["objects"])
    assert set(config.fixture_ids) == {"start_button"}
    assert set(config.target_ids) == {
        "return_bin_left",
        "return_bin_right",
        "inspection_pad",
        "setup_slot_a",
        "setup_slot_b",
    }
    assert set(frozen["skills"]) == set(REQUIRED_SKILLS)
    assert frozen["optional_skills"]["rotate_dial"]["enabled"] is False


def test_scene_loads_every_required_entity_in_one_model() -> None:
    pytest.importorskip("mujoco")

    with Workcell(CONFIG_PATH) as workcell:
        state = workcell.reset(seed=0)
        model = workcell.env.model

    assert model.nq > 0
    assert model.nu > 0
    assert len(state.entities) == 12
    assert {item.object_id for item in state.entities} == {
        *workcell.config.object_ids,
        *workcell.config.fixture_ids,
        *workcell.config.target_ids,
    }
    for entity in state.entities:
        assert entity.source == SIMULATOR_GROUND_TRUTH
        assert entity.confidence == 1.0
        assert entity.frame == "mujoco_world"
        assert entity.valid
    assert state.require_fixture("start_button").press_depth_m == pytest.approx(0.0)
    assert state.require_fixture("start_button").pressed is False
    label_sites = {
        workcell.env._mujoco.mj_id2name(
            model, workcell.env._mujoco.mjtObj.mjOBJ_SITE, site_id
        )
        for site_id in range(model.nsite)
    }
    assert {
        *workcell.config.object_ids,
        *workcell.config.fixture_ids,
        *workcell.config.target_ids,
        "clearing_region",
    } <= label_sites
    for site_name in {
        *workcell.config.object_ids,
        *workcell.config.fixture_ids,
        *workcell.config.target_ids,
        "clearing_region",
    }:
        site_id = workcell.env._mujoco.mj_name2id(
            model, workcell.env._mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        assert model.site_rgba[site_id, 3] > 0.0
        assert model.site_group[site_id] == 0
    helper_site_id = workcell.env._mujoco.mj_name2id(
        model,
        workcell.env._mujoco.mjtObj.mjOBJ_SITE,
        "dexvision_hand_base_target_site",
    )
    assert model.site_rgba[helper_site_id, 3] == pytest.approx(0.0)


def test_reset_is_deterministic_bounded_and_collision_free() -> None:
    pytest.importorskip("mujoco")

    frozen = yaml.safe_load(FROZEN_PATH.read_text(encoding="utf-8"))
    with Workcell(CONFIG_PATH) as workcell:
        first = workcell.reset(seed=17)
        second = workcell.reset(seed=17)
        different = workcell.reset(seed=18)
        clearance = float(workcell.config.scene["object_clearance_m"])

        first_poses = {
            object_id: (
                first.require_entity(object_id).position,
                first.require_entity(object_id).orientation_wxyz,
            )
            for object_id in workcell.config.object_ids
        }
        second_poses = {
            object_id: (
                second.require_entity(object_id).position,
                second.require_entity(object_id).orientation_wxyz,
            )
            for object_id in workcell.config.object_ids
        }
        different_poses = {
            object_id: different.require_entity(object_id).position
            for object_id in workcell.config.object_ids
        }

        for spec in workcell.config.objects:
            position = np.asarray(second.require_entity(spec.object_id).position)
            reset_range = frozen["reset_ranges"][spec.family]["position_range_m"]
            assert np.all(position >= np.asarray(reset_range["min"]))
            assert np.all(position <= np.asarray(reset_range["max"]))
            relation = second.relation_for(spec.object_id)
            assert relation.supported_by == "workcell_table"
            assert relation.held_by is None
            for target_id in ("setup_slot_a", "setup_slot_b"):
                target_position = np.asarray(
                    second.require_entity(target_id).position[:2]
                )
                minimum = (
                    spec.footprint_radius_m
                    + float(workcell.config.scene["setup_slot_visual_radius_m"])
                )
                assert np.linalg.norm(position[:2] - target_position) >= minimum

        for index, first_spec in enumerate(workcell.config.objects):
            for second_spec in workcell.config.objects[index + 1 :]:
                first_xy = np.asarray(second.require_entity(first_spec.object_id).position[:2])
                second_xy = np.asarray(
                    second.require_entity(second_spec.object_id).position[:2]
                )
                minimum = (
                    first_spec.footprint_radius_m
                    + second_spec.footprint_radius_m
                    + clearance
                )
                assert np.linalg.norm(first_xy - second_xy) >= minimum

    assert first_poses == second_poses
    assert different_poses != {
        object_id: pose[0] for object_id, pose in first_poses.items()
    }


def test_scene_remains_finite_and_objects_stay_on_board_headlessly() -> None:
    pytest.importorskip("mujoco")

    with Workcell(CONFIG_PATH) as workcell:
        workcell.reset(seed=4)
        final = workcell.step(n_steps=180)
        qpos = workcell.env.data.qpos.copy()
        qvel = workcell.env.data.qvel.copy()
        board = workcell.config.requirements["workcell"]["board_workspace"]

    assert np.all(np.isfinite(qpos))
    assert np.all(np.isfinite(qvel))
    for object_id in workcell.config.object_ids:
        position = np.asarray(final.require_entity(object_id).position)
        assert np.all(position[:2] >= np.asarray(board["min_xy_m"]))
        assert np.all(position[:2] <= np.asarray(board["max_xy_m"]))
        assert position[2] >= 0.0


def test_task_factory_rejects_unknown_ids_and_disabled_dial() -> None:
    pytest.importorskip("mujoco")

    with Workcell(CONFIG_PATH) as workcell:
        workcell.reset(seed=0)
        with pytest.raises(WorkcellError, match="unsupported value"):
            workcell.create_task(
                "reach_object",
                entity_id="not_an_entity",
                approach_pose=(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0),
            )
        with pytest.raises(WorkcellError, match="disabled"):
            workcell.create_task(OPTIONAL_DIAL_SKILL, dial_id="mode_dial")
        with pytest.raises(WorkcellError, match="Unsupported Level 4"):
            workcell.create_task("future_skill")


def test_config_loader_reports_runtime_frozen_id_disagreement(tmp_path: Path) -> None:
    runtime = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    runtime["model_path"] = str((ROOT / "assets/mujoco/workcell_scene.xml").resolve())
    runtime["requirements_path"] = str(FROZEN_PATH.resolve())
    del runtime["objects"]["block_small"]
    bad_path = tmp_path / "workcell.yaml"
    bad_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")

    with pytest.raises(WorkcellError, match="objects ids disagree"):
        load_workcell_config(bad_path)


def test_manual_inspector_defaults_to_open_ended_viewer() -> None:
    args = inspect_workcell.build_parser().parse_args([])

    assert args.steps == 0
    assert args.headless is False
    command = inspect_workcell._format_mjpython_command(
        CONFIG_PATH, seed=0, steps=0, viewer_sleep=inspect_workcell.DEFAULT_VIEWER_SLEEP
    )
    assert command == (
        "mjpython -m dexvision.apps.inspect_workcell "
        f"--config {CONFIG_PATH} --seed 0"
    )
    with pytest.raises(ValueError, match="positive in headless mode"):
        inspect_workcell.run_inspection(
            config_path=CONFIG_PATH,
            seed=0,
            steps=0,
            headless=True,
            viewer_sleep=0.0,
        )


def test_viewer_starts_with_unlocked_overview_camera() -> None:
    pytest.importorskip("mujoco")

    handle = SimpleNamespace(
        cam=SimpleNamespace(
            type=None,
            lookat=np.zeros(3, dtype=np.float64),
            distance=0.0,
            azimuth=0.0,
            elevation=0.0,
        ),
        opt=SimpleNamespace(label=None),
    )
    with Workcell(CONFIG_PATH) as workcell:
        inspect_workcell._configure_free_camera(handle, workcell)
        viewer_config = workcell.config.scene["viewer"]

        assert handle.cam.type == workcell.env._mujoco.mjtCamera.mjCAMERA_FREE
        assert handle.cam.lookat == pytest.approx(viewer_config["lookat_m"])
        assert handle.cam.distance == pytest.approx(viewer_config["distance_m"])
        assert handle.cam.azimuth == pytest.approx(viewer_config["azimuth_deg"])
        assert handle.cam.elevation == pytest.approx(viewer_config["elevation_deg"])
        assert handle.opt.label == workcell.env._mujoco.mjtLabel.mjLABEL_SITE


def test_viewer_projects_named_return_bins_to_matching_screen_sides() -> None:
    mujoco = pytest.importorskip("mujoco")

    with Workcell(CONFIG_PATH) as workcell:
        state = workcell.reset(seed=0)
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        handle = SimpleNamespace(cam=camera, opt=option)
        inspect_workcell._configure_free_camera(handle, workcell)
        scene = mujoco.MjvScene(workcell.env.model, maxgeom=10_000)
        mujoco.mjv_updateScene(
            workcell.env.model,
            workcell.env.data,
            option,
            mujoco.MjvPerturb(),
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )

        left = np.asarray(state.require_entity("return_bin_left").position)
        right = np.asarray(state.require_entity("return_bin_right").position)
        frozen_targets = workcell.config.requirements["workcell"]["targets"]
        screen_right = np.cross(scene.camera[0].forward, scene.camera[0].up)

    assert left == pytest.approx(frozen_targets["return_bin_left"]["center_m"])
    assert right == pytest.approx(frozen_targets["return_bin_right"]["center_m"])
    assert np.dot(right - left, screen_right) > 0.0
