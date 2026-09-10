"""Fixed-camera RGB and simulator-ground-truth instance annotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RenderEntity:
    """One body-frame pose and its renderable geometry ids."""

    object_id: str
    class_id: str
    category: str
    body_id: int
    geom_ids: tuple[int, ...]


def camera_rotation(position: np.ndarray, lookat: np.ndarray) -> np.ndarray:
    """Return world-from-OpenGL-camera rotation; camera looks along local -z."""
    backward = np.asarray(position, dtype=float) - np.asarray(lookat, dtype=float)
    if not np.all(np.isfinite(backward)) or np.linalg.norm(backward) == 0:
        raise ValueError("Camera position and lookat must be finite and distinct.")
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("Camera look direction cannot be parallel to world up.")
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(backward, right), backward))


def camera_calibration(model, data, camera_id: str, width: int, height: int) -> dict:
    """Pinhole calibration with pixel centers at integers, SI world extrinsics."""
    camera = model.camera(camera_id)
    focal = height / (2 * np.tan(np.deg2rad(float(camera.fovy[0])) / 2))
    rotation = np.diag([1., -1., -1.]) @ data.cam_xmat[camera.id].reshape(3, 3).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ data.cam_xpos[camera.id]
    return {
        "camera_id": camera_id, "image_size": [width, height],
        "intrinsics": [[focal, 0., (width - 1) / 2],
                       [0., focal, (height - 1) / 2], [0., 0., 1.]],
        "camera_from_world": transform.tolist(), "distortion": [0., 0., 0., 0., 0.],
        "camera_frame": "x_right_y_down_z_forward", "world_frame": "mujoco_world",
        "length_units": "m", "pixel_coordinates": "integer_pixel_centers",
    }


def mask_box(mask: np.ndarray) -> list[int] | None:
    """Tight visible [xmin,ymin,xmax,ymax], with exclusive upper edges."""
    y, x = np.nonzero(mask)
    return None if not len(x) else [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]


def annotate_instances(segmentation: np.ndarray, entities: tuple[RenderEntity, ...],
                       data) -> tuple[np.ndarray, list[dict]]:
    """Convert MuJoCo [H,W,2] (id,type) pixels into visible instance masks."""
    import mujoco

    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise ValueError("Segmentation must have shape [H,W,2].")
    masks = np.zeros(segmentation.shape[:2], dtype=np.uint16)
    annotations = []
    for index, entity in enumerate(entities, 1):
        visible = ((segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) &
                   np.isin(segmentation[..., 0], entity.geom_ids))
        masks[visible] = index
        count = int(visible.sum())
        annotations.append({
            "object_id": entity.object_id, "class_id": entity.class_id,
            "category": entity.category, "mask_value": index,
            "visible": bool(count), "visible_pixels": count,
            "visibility": "visible" if count else "occluded_or_out_of_frame",
            "box_xyxy": mask_box(visible),
            "pose_6d": {"translation_m": data.xpos[entity.body_id].tolist(),
                        "rotation_matrix": data.xmat[entity.body_id].reshape(3, 3).tolist(),
                        "transform": "world_from_object"},
            "source": "simulator_ground_truth", "confidence": 1.0,
        })
    return masks, annotations


def add_box(renderer, *, position, half_size, rotation=None, color=(0.7, 0.65, 0.5, 1.)):
    """Add a render-only rigid distractor/occluder; never alters simulator state."""
    import mujoco

    scene = renderer.scene
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("Renderer has no room for visual-condition geometry.")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half_size, float),
                       np.asarray(position, float),
                       np.asarray(np.eye(3) if rotation is None else rotation).reshape(9),
                       np.asarray(color, dtype=np.float32))
    geom.objid = -1
    geom.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
    geom.segid = scene.ngeom
    scene.ngeom += 1


def render_frame(renderer, model, data, *, camera: str, entities: tuple[RenderEntity, ...],
                 condition: str, parameters: dict, goal_entity: str) -> tuple:
    """Render RGB and masks from one identical scene and frozen condition.

    Partial occlusion uses a thin 3D card parallel to the camera plane, placed
    from the unoccluded visible goal box. Its measured removal fraction is
    bounded per frame; a missing goal remains explicitly missing.
    """
    import mujoco

    option = mujoco.MjvOption()
    option.sitegroup[:] = 0  # no task cues, label anchors, or privileged markers
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera, scene_option=option)
    parameters_used = dict(parameters)
    if condition == "bounded_distractors":
        for position in parameters["positions_m"]:
            add_box(renderer, position=position, half_size=parameters["half_size_m"])
    elif condition == "partial_occlusion":
        renderer.enable_segmentation_rendering()
        original, original_annotations = annotate_instances(renderer.render(), entities, data)
        renderer.disable_segmentation_rendering()
        target = next(a for a in original_annotations if a["object_id"] == goal_entity)
        box = target["box_xyxy"]
        parameters_used["target_removed_fraction"] = 0.0
        if box:
            calibration = camera_calibration(model, data, camera, renderer.width, renderer.height)
            intrinsic = np.asarray(calibration["intrinsics"])
            x0, y0, x1, y1 = box
            depth = 0.15
            fraction = parameters["target_box_width_fraction"]
            camera_point = np.array([
                ((x0 + (x1-x0)*fraction/2) - intrinsic[0, 2])*depth/intrinsic[0, 0],
                ((y0+y1)/2 - intrinsic[1, 2])*depth/intrinsic[1, 1], depth])
            world_from_camera = np.linalg.inv(calibration["camera_from_world"])
            position = (world_from_camera @ np.r_[camera_point, 1])[:3]
            size = [max(0.5, (x1-x0)*fraction/2)*depth/intrinsic[0, 0],
                    (y1-y0)*0.5*depth/intrinsic[1, 1], 0.0001]
            # A narrow box can still cover most of an already occluded mask.
            # Bound the *measured visible-pixel loss*, without changing physics.
            base_geoms = renderer.scene.ngeom
            for attempt in range(9):
                renderer.scene.ngeom = base_geoms
                size[0] *= 0.5 if attempt else 1.0
                add_box(renderer, position=position, half_size=size,
                        rotation=world_from_camera[:3, :3])
                renderer.enable_segmentation_rendering()
                trial_masks, trial_annotations = annotate_instances(renderer.render(), entities, data)
                renderer.disable_segmentation_rendering()
                remaining = next(a["visible_pixels"] for a in trial_annotations
                                 if a["object_id"] == goal_entity)
                removed = 1 - remaining / target["visible_pixels"]
                if 0 <= removed <= parameters["maximum_target_mask_occlusion_fraction"]:
                    parameters_used["target_removed_fraction"] = removed
                    parameters_used["occluder"] = {
                        "position_m": position.tolist(), "half_size_m": list(size),
                        "rotation_matrix": world_from_camera[:3, :3].tolist(),
                        "bound_refinement_steps": attempt,
                    }
                    break
            else:
                renderer.scene.ngeom = base_geoms
                parameters_used["occluder_absence_reason"] = "insufficient_visible_goal_pixels"
        else:
            parameters_used["occluder_absence_reason"] = "goal_not_visible"
        rgb = renderer.render()
        renderer.enable_segmentation_rendering()
        masks, annotations = annotate_instances(renderer.render(), entities, data)
        return rgb, masks, annotations, parameters_used
    rgb = renderer.render()
    renderer.enable_segmentation_rendering()
    masks, annotations = annotate_instances(renderer.render(), entities, data)
    return rgb, masks, annotations, parameters_used
