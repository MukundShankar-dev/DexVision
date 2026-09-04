from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dexvision.perception.object_observations import (
    INFERRED_PERCEPTION,
    SIMULATOR_GROUND_TRUTH,
    ObjectObservation,
    ObjectObservationError,
    make_object_observation,
)
from dexvision.sim.workcell import (
    Workcell,
    create_pick_task,
    create_place_task,
    create_press_task,
    create_push_task,
    create_reach_task,
)
from dexvision.sim.world_state import (
    EntityRelation,
    FixtureObservation,
    WorldState,
    WorldStateError,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "workcell.yaml"


def test_simulator_and_inferred_sources_share_one_typed_schema() -> None:
    truth = make_object_observation(
        object_id="block_small",
        class_id="rigid_cuboid",
        position=(0.0, 0.0, 0.02),
        orientation_wxyz=(2.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        source=SIMULATOR_GROUND_TRUTH,
        confidence=1.0,
        timestamp=1.0,
        frame="mujoco_world",
    )
    inferred = replace(truth, source=INFERRED_PERCEPTION, confidence=0.8)

    assert isinstance(truth, ObjectObservation)
    assert isinstance(inferred, ObjectObservation)
    assert truth.orientation_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert inferred.object_id == truth.object_id
    assert inferred.position == truth.position


def test_object_observation_rejects_bad_frames_quaternions_and_sources() -> None:
    base = dict(
        object_id="object",
        class_id="rigid_object",
        position=(0.0, 0.0, 0.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=None,
        source=SIMULATOR_GROUND_TRUTH,
        confidence=1.0,
        timestamp=0.0,
        frame="mujoco_world",
    )
    with pytest.raises(ObjectObservationError, match="coordinate frame"):
        ObjectObservation(**{**base, "frame": ""})
    with pytest.raises(ObjectObservationError, match="unit quaternion"):
        ObjectObservation(**{**base, "orientation_wxyz": (2.0, 0.0, 0.0, 0.0)})
    with pytest.raises(ObjectObservationError, match="unsupported source"):
        ObjectObservation(**{**base, "source": "unknown"})


def test_world_state_rejects_ambiguous_ids_and_stale_state_fails_clearly() -> None:
    pytest.importorskip("mujoco")

    with Workcell(CONFIG_PATH) as workcell:
        state = workcell.reset(seed=0)

    duplicate = state.entities[0]
    with pytest.raises(WorldStateError, match="Ambiguous duplicate"):
        replace(state, entities=(*state.entities, duplicate))

    stale_entities = tuple(replace(item, timestamp=0.0) for item in state.entities)
    stale = replace(state, timestamp=1.0, entities=stale_entities)
    with pytest.raises(WorldStateError, match="stale"):
        stale.require_entity("block_small", maximum_age_s=0.1)
    with pytest.raises(WorldStateError, match="Unknown workcell entity"):
        state.require_entity("missing_object")


def test_all_five_task_factories_compute_frozen_metrics_and_dwell() -> None:
    pytest.importorskip("mujoco")

    with Workcell(CONFIG_PATH) as workcell:
        initial = workcell.reset(seed=2)

        reach_pose = (
            *initial.robot.base_position,
            *initial.robot.base_orientation_wxyz,
        )
        reach = create_reach_task(
            workcell, entity_id="block_small", approach_pose=reach_pose
        )
        reach_result = _evaluate_to_success(reach, initial)
        assert reach_result.values["approach_distance_m"] == pytest.approx(0.0)
        assert reach_result.values["approach_orientation_error_rad"] == pytest.approx(0.0)
        assert reach_result.values["maximum_scene_disturbance_m"] == pytest.approx(0.0)

        picked_object = initial.require_entity("block_small")
        picked = replace(
            picked_object,
            position=(
                picked_object.position[0],
                picked_object.position[1],
                picked_object.position[2] + 0.05,
            ),
            linear_velocity=(0.0, 0.0, 0.0),
        )
        picked_state = initial.replace_entity(picked)
        picked_state = replace(
            picked_state,
            relations=_replace_relation(
                picked_state,
                EntityRelation(
                    object_id="block_small",
                    supported_by=None,
                    held_by="rh_palm",
                    receptacle_id=None,
                ),
            ),
        )
        pick = create_pick_task(workcell, object_id="block_small")
        pick_result = _evaluate_to_success(pick, picked_state)
        assert pick_result.values["held_object_id"] == "block_small"
        assert pick_result.values["object_height_above_support_m"] == pytest.approx(0.05)

        inspection = initial.require_entity("inspection_pad")
        placed = replace(
            picked_object,
            position=inspection.position,
            linear_velocity=(0.0, 0.0, 0.0),
        )
        placed_state = initial.replace_entity(placed)
        placed_state = replace(
            placed_state,
            relations=_replace_relation(
                placed_state,
                EntityRelation(
                    object_id="block_small",
                    supported_by="workcell_table",
                    held_by=None,
                    receptacle_id="inspection_pad",
                ),
            ),
        )
        place = create_place_task(
            workcell, object_id="block_small", target_id="inspection_pad"
        )
        place_result = _evaluate_to_success(place, placed_state)
        assert place_result.values["object_to_target_distance_m"] == pytest.approx(0.0)
        assert place_result.values["object_linear_speed_mps"] == pytest.approx(0.0)
        assert place_result.values["object_inside_target"] is True
        assert place_result.values["held_object_id"] is None

        pushed_object = initial.require_entity("puck_light")
        pushed = replace(
            pushed_object,
            position=inspection.position,
            linear_velocity=(0.0, 0.0, 0.0),
        )
        pushed_state = initial.replace_entity(pushed)
        push = create_push_task(
            workcell, object_id="puck_light", target_zone="inspection_pad"
        )
        push_result = _evaluate_to_success(push, pushed_state)
        assert push_result.values["planar_object_to_target_distance_m"] == pytest.approx(
            0.0
        )
        assert push_result.values["object_linear_speed_mps"] == pytest.approx(0.0)
        assert push_result.values["object_on_board"] is True

        pressed_state = replace(
            initial,
            fixtures=(
                FixtureObservation(
                    fixture_id="start_button", press_depth_m=0.010, pressed=True
                ),
            ),
        )
        press = create_press_task(workcell, target_press_depth_m=0.008)
        press_result = _evaluate_to_success(press, pressed_state)
        assert press_result.values["button_id"] == "start_button"
        assert press_result.values["press_depth_m"] == pytest.approx(0.010)
        assert press_result.values["button_pressed"] is True
        assert press_result.values["other_button_max_depth_m"] == pytest.approx(0.0)

    assert all(
        result.success
        for result in (
            reach_result,
            pick_result,
            place_result,
            push_result,
            press_result,
        )
    )


def test_reach_metric_resets_dwell_after_nonqualifying_frame() -> None:
    pytest.importorskip("mujoco")

    with Workcell(CONFIG_PATH) as workcell:
        state = workcell.reset(seed=3)
        pose = (*state.robot.base_position, *state.robot.base_orientation_wxyz)
        task = create_reach_task(workcell, entity_id="block_small", approach_pose=pose)
        assert task.evaluate(state).dwell_steps == 1
        far_robot = replace(
            state.robot,
            base_position=(1.0, 1.0, 1.0),
        )
        failed = task.evaluate(replace(state, robot=far_robot))

    assert failed.qualifies is False
    assert failed.dwell_steps == 0
    assert failed.success is False


def _replace_relation(
    state: WorldState, replacement: EntityRelation
) -> tuple[EntityRelation, ...]:
    return tuple(
        replacement if item.object_id == replacement.object_id else item
        for item in state.relations
    )


def _evaluate_to_success(task, state: WorldState):
    result = None
    for _ in range(task.spec["success_metric"]["required_consecutive_samples"]):
        result = task.evaluate(state)
    assert result is not None
    return result
