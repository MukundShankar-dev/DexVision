"""Synthetic annotation and headless rendered calibration regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.perception.render_annotations import (
    RenderEntity, annotate_instances, camera_calibration, camera_rotation, mask_box,
    render_frame,
)


def test_visible_masks_union_geoms_and_ignore_non_geom_ids():
    import mujoco

    segmentation = np.full((8, 10, 2), -1, dtype=int)
    segmentation[2:5, 3:6] = [4, int(mujoco.mjtObj.mjOBJ_GEOM)]
    segmentation[4:6, 6:8] = [5, int(mujoco.mjtObj.mjOBJ_GEOM)]
    segmentation[0, 0] = [4, int(mujoco.mjtObj.mjOBJ_SITE)]
    entities = (RenderEntity("block", "cuboid", "cuboid", 0, (4, 5)),
                RenderEntity("hidden", "puck", "flat_puck", 1, (6,)))
    data = SimpleNamespace(xpos=np.array([[1., 2., 3.], [0., 0., 0.]]),
                           xmat=np.tile(np.eye(3).reshape(9), (2, 1)))
    masks, annotations = annotate_instances(segmentation, entities, data)
    assert masks.dtype == np.uint16
    assert annotations[0]["box_xyxy"] == [3, 2, 8, 6]
    assert annotations[0]["visible_pixels"] == 13
    assert masks[0, 0] == 0
    assert annotations[1]["visibility"] == "occluded_or_out_of_frame"
    assert annotations[1]["box_xyxy"] is None
    assert annotations[1]["pose_6d"]["rotation_matrix"] == np.eye(3).tolist()
    assert mask_box(np.zeros((2, 3), bool)) is None


def test_camera_rotation_and_optical_axis():
    rotation = camera_rotation(np.array([1., -1., 1.]), np.zeros(3))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    for lookat in ([1, -1, 1], [1, -1, 0]):
        with pytest.raises(ValueError):
            camera_rotation(np.array([1., -1., 1.]), np.array(lookat))


def test_real_render_projection_segmentation_and_partial_occlusion():
    import mujoco

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <light pos="0 0 2"/><camera name="fixed" pos="0 0 1" fovy="45"/>
      <body name="block" pos="0 0 0"><geom name="block_geom" type="box"
      size=".1 .1 .05" rgba="1 0 0 1"/></body>
      <body name="outside" pos="4 0 0"><geom name="outside_geom" type="sphere"
      size=".1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    entities = tuple(RenderEntity(name, "object", "cuboid", model.body(name).id,
                                  (model.geom(name+"_geom").id,))
                     for name in ("block", "outside"))
    try:
        renderer = mujoco.Renderer(model, height=120, width=160)
    except Exception as exc:
        pytest.skip(f"Offscreen OpenGL unavailable: {exc}")
    with renderer:
        rgb, masks, annotations, _ = render_frame(
            renderer, model, data, camera="fixed", entities=entities,
            condition="nominal", parameters={}, goal_entity="block")
        assert rgb.shape == (120, 160, 3)
        assert annotations[0]["visible"] and not annotations[1]["visible"]
        calibration = camera_calibration(model, data, "fixed", 160, 120)
        point = np.array(calibration["camera_from_world"]) @ [0, 0, 0, 1]
        pixel = np.array(calibration["intrinsics"]) @ point[:3]
        pixel = pixel[:2]/pixel[2]
        y, x = np.nonzero(masks == 1)
        np.testing.assert_allclose([x.mean(), y.mean()], pixel, atol=0.5)
        qpos = data.qpos.copy()
        _, occluded_masks, _, parameters = render_frame(
            renderer, model, data, camera="fixed", entities=entities,
            condition="partial_occlusion", goal_entity="block",
            parameters={"target_box_width_fraction": 0.12,
                        "maximum_target_mask_occlusion_fraction": 0.35})
        assert 0 < parameters["target_removed_fraction"] <= 0.35
        assert (occluded_masks == 1).sum() < (masks == 1).sum()
        np.testing.assert_array_equal(data.qpos, qpos)
