"""Policy-side observation/phase/preview adapter for the 5.7 amendment."""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dexvision.evaluation.evaluate_skill import reach_features
from dexvision.perception.fiducial_pose import FiducialPoseEstimator
from dexvision.perception.instrumented_state import InstrumentedStateEstimator
from dexvision.perception.render_annotations import camera_calibration, camera_rotation
from dexvision.sim.optical_markers import OpticalMarkerRenderer
from dexvision.sim.workcell import _compute_task_metrics
from dexvision.sim.workcell_sensors import read_workcell_sensors


class PerceptionUnavailable(ValueError):
    """A terminal missing-measurement rejection, never an action fallback."""


def marker_atlas(config, recipe):
    atlas = {}
    objects = [(o.object_id, o.body, o.class_id, False) for o in config.objects]
    objects += [
        (name, f["body"], "button", True) for name, f in config.fixtures.items()
    ]
    for i, (name, body, category, fixture) in enumerate(objects):
        faces = [
            ("x", 0, [0, 0, 1]),
            ("x", 180, [0, 0, -1]),
            ("y", 90, [1, 0, 0]),
            ("y", -90, [-1, 0, 0]),
            ("x", -90, [0, 1, 0]),
            ("x", 90, [0, -1, 0]),
        ]
        for side, (axis, angle, normal) in enumerate(faces):
            offset = list(
                recipe["fixture_rig_offset_m"]
                if fixture
                else recipe["object_rig_offset_m"]
            )
            offset = (
                np.asarray(offset)
                + np.asarray(normal) * recipe["marker_cube_half_size_m"]
            ).tolist()
            rotation = Rotation.from_euler(axis, angle, degrees=True)
            atlas[str(i * 6 + side)] = {
                "object_id": name,
                "class_id": category,
                "body": body,
                "side_m": recipe["marker_side_m"],
                "offset_m": offset,
                "rotation": rotation.as_matrix().tolist(),
            }
    return atlas


class InstrumentedObserver:
    def __init__(
        self, task, inputs, row, skill, recipe, *, condition="nominal", video_path=None
    ):
        self.task, self.inputs, self.row, self.skill = (
            task,
            inputs,
            copy.deepcopy(row),
            skill,
        )
        self.recipe, self.condition = recipe, condition
        model = task.env.model
        for body_id in range(model.nbody):
            if model.body(body_id).name.startswith("pilot_"):
                model.geom_rgba[model.geom_bodyid == body_id, 3] = 0
        camera = yaml.safe_load(Path(recipe["visual_config"]).read_text())["camera"]
        cam = model.camera(camera["id"])
        cam.pos[:] = camera["position_m"]
        mujoco.mju_mat2Quat(
            cam.quat,
            camera_rotation(
                camera["position_m"], recipe.get("camera_lookat_m", camera["lookat_m"])
            ).ravel(),
        )
        cam.fovy[:] = camera["vertical_fov_deg"]
        width, height = recipe["resolution"]
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        self.scratch = mujoco.MjData(model)
        self.refresh_scratch()
        self.atlas = {
            key: tag
            for key, tag in marker_atlas(task.workcell.config, recipe).items()
            if tag["object_id"] == row["entity_id"]
        }
        calibration = camera_calibration(
            model, self.scratch, camera["id"], width, height
        )
        self.renderer = OpticalMarkerRenderer(
            model,
            self.scratch,
            self.atlas,
            camera=camera["id"],
            width=width,
            height=height,
        )
        optical = FiducialPoseEstimator(
            calibration,
            self.atlas,
            maximum_error_px=recipe["maximum_reprojection_px"],
            history_samples=recipe.get("motion_history_samples", 2),
            motion_gap_s=0.51,
        )
        self.estimator = InstrumentedStateEstimator(
            task.workcell, optical, (row["entity_id"],)
        )
        # Scripted entry controller stage is recorded at the boundary. After
        # entry only current/past inferred measurements may advance it.
        self.phase = "transport" if skill == "place" else "approach"
        self.initial = None
        self.contact_seen = self.pressed_once = False
        self.dwell = 0
        self.last_time = None
        self.last_source = None
        self.latest_rgb = None
        self.records, self.latencies = [], []
        self.policy_calls = 0
        self.video = None
        self.video_frames = 0
        self.observation_count = 0
        if video_path:
            self.video = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480)
            )
            if not self.video.isOpened():
                raise ValueError("Review video encoder unavailable")

    def refresh_scratch(self):
        mujoco.mj_copyData(self.scratch, self.task.env.model, self.task.env.data)
        mujoco.mj_forward(self.task.env.model, self.scratch)

    def pixels(self):
        rgb = self.renderer.render()
        if self.condition == "mild_illumination":
            rgb = np.rint(rgb.astype(float) * 0.9).astype(np.uint8)
        elif self.condition == "partial_occlusion":
            h, w = rgb.shape[:2]
            rgb[int(0.45 * h) : int(0.55 * h), int(0.43 * w) : int(0.57 * w)] = 100
        elif self.condition == "bounded_distractors":
            cv2.rectangle(rgb, (30, 30), (80, 80), (240, 20, 20), -1)
            cv2.rectangle(rgb, (100, 30), (150, 80), (20, 240, 20), -1)
        elif self.condition == "pixel_bias":
            rgb = cv2.warpAffine(
                rgb, np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -2.0]]), rgb.shape[1::-1]
            )
        elif self.condition == "dropout":
            rgb[:] = 0
        return rgb

    def bootstrap(self):
        self.refresh_scratch()
        self.estimator.optical.infer(self.pixels(), float(self.scratch.time))

    def observe(self):
        self.refresh_scratch()
        timestamp = float(self.scratch.time)
        if self.last_time == timestamp and self.estimator.last_world is not None:
            return self.estimator.last_world
        started = time.perf_counter()
        self.latest_start = started
        fresh = self.observation_count % self.recipe.get("camera_stride", 3) == 0
        if fresh:
            self.latest_rgb = self.pixels()
        self.observation_count += 1
        sensor_env = copy.copy(self.task.env)
        sensor_env.data = self.scratch
        disabled = ("contact_switches",) if self.condition == "sensor_dropout" else ()
        packet = read_workcell_sensors(
            sensor_env, self.task.workcell.config, disabled=disabled
        )
        delay = 0.12 if self.condition == "latency" else 0.0
        try:
            world = self.estimator.observe(
                self.latest_rgb if fresh else None,
                packet,
                now=timestamp + delay + time.perf_counter() - started,
            )
        except ValueError as exc:
            self.latencies.append(time.perf_counter() - started)
            self.record_frame(str(exc))
            raise PerceptionUnavailable(str(exc)) from exc
        latency = time.perf_counter() - started
        self.latencies.append(latency)
        self.last_time = timestamp
        if self.initial is None:
            self.initial = world
        self.update_phase(world)
        consumed = world.require_entity(self.row["entity_id"])
        self.records.append(
            {
                "timestamp": timestamp,
                "phase": self.phase,
                "measurement_timestamp": consumed.timestamp,
                "pose": [
                    *consumed.position,
                    *consumed.orientation_wxyz,
                    *consumed.linear_velocity,
                    *consumed.angular_velocity,
                ],
                "latency_s": latency,
                "action_ready_latency_s": None,
                "position_error_m": None,
                "orientation_error_rad": None,
            }
        )
        return world

    def update_phase(self, world):
        name = self.row["entity_id"]
        entity, relation = world.require_entity(name), world.relation_for(name)
        contact = any(
            name in p and any(n.startswith("rh_") for n in p) for p in world.contacts
        )
        self.contact_seen |= contact
        if self.skill == "pick":
            # Accepted run_pick_rollout advances workcell physics directly and
            # retains its entry phase; PickPolicy owns its separate causal
            # acquire/lift/hold transform/history. Preserve that exact behavior.
            return
        elif self.skill == "place":
            target = world.require_entity(self.row["typed_goal"]["target_id"])
            distance = np.linalg.norm(np.array(entity.position) - target.position)
            if self.phase == "transport" and distance <= 0.025:
                self.phase = "place"
            elif self.phase == "place" and relation.held_by != "rh_palm":
                self.phase = "release"
            elif self.phase == "release":
                self.phase = "settle"
        elif self.skill in {"push", "press"}:
            name = "push_object_to_target" if self.skill == "push" else "press_button"
            values, qualifies = _compute_task_metrics(
                name,
                self.row["typed_goal"],
                world,
                self.initial,
                self.task.workcell.config,
            )
            self.dwell = self.dwell + 1 if qualifies else 0
            required = self.task.workcell.config.requirements["skills"][name][
                "success_metric"
            ]["required_consecutive_samples"]
            if self.skill == "push":
                self.phase = (
                    "retract"
                    if self.dwell >= required
                    else "settle"
                    if qualifies
                    else "push_contact"
                    if self.contact_seen
                    else "approach"
                )
            else:
                self.pressed_once |= self.dwell >= required
                self.phase = (
                    "retract"
                    if self.pressed_once
                    else "fixture_contact"
                    if self.contact_seen
                    else "approach"
                )

    def features(
        self,
        task,
        _evaluation_truth,
        row,
        inputs,
        commanded,
        prior,
        safety,
        _evaluation_phase,
    ):
        world = self.observe()
        if self.skill == "reach" and self.last_source is None:
            position = np.array(world.require_entity(row["entity_id"]).position)
            if row["entity_id"] == "start_button":
                position += self.task.collection_config["pilot"]["scripted_button"][
                    "precontact_offset_m"
                ]
            else:
                position[0] = max(-0.16, min(0.20, position[0] - 0.035))
                position[2] = max(0.14, min(0.23, position[2] + 0.13))
            variation = self.row.get(
                "variation", self.row.get("procedural_variation", {})
            )
            for key in ("goal_position_offset_m", "controller_position_offset_m"):
                position += np.asarray(variation.get(key, [0, 0, 0]))
            self.row["typed_goal"]["approach_pose"] = [
                *position,
                *world.robot.base_orientation_wxyz,
            ]
        proxy = copy.copy(task)
        proxy.env = self.estimator.workcell.env
        proxy.workcell = self.estimator.workcell
        self.last_source = reach_features(
            proxy, world, self.row, inputs, commanded, prior, safety, self.phase
        )
        self.records[-1]["feature_sha256"] = hashlib.sha256(
            self.last_source.tobytes()
        ).hexdigest()
        self.records[-1]["features"] = self.last_source.copy()
        return self.last_source

    def record_frame(self, terminal, *, force=False):
        if (
            self.video is None
            or self.latest_rgb is None
            or (not force and self.video_frames >= self.recipe["maximum_review_frames"])
        ):
            return
        panel = cv2.cvtColor(cv2.resize(self.latest_rgb, (640, 480)), cv2.COLOR_RGB2BGR)
        calibration = self.estimator.optical.calibration
        for pose in self.estimator.last_poses:
            camera = (
                np.asarray(calibration["camera_from_world"]) @ np.r_[pose.position, 1.0]
            )
            pixel = np.asarray(calibration["intrinsics"]) @ camera[:3]
            pixel = (
                pixel[:2]
                / pixel[2]
                * np.array(
                    [
                        640 / calibration["image_size"][0],
                        480 / calibration["image_size"][1],
                    ]
                )
            )
            if np.isfinite(pixel).all():
                point = tuple(np.clip(np.rint(pixel), -10000, 10000).astype(int))
                cv2.circle(panel, point, 5, (0, 255, 0), 1)
                cv2.putText(
                    panel,
                    pose.object_id,
                    point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                )
        cv2.rectangle(panel, (0, 0), (640, 45), (0, 0, 0), -1)
        text = f"{self.skill} {self.phase} calls={self.policy_calls}: {terminal}"
        cv2.putText(
            panel,
            text[:95],
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            panel,
            "Visible optical rigs + declared simulated sensors; estimated preview",
            (5, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
        pose = self.estimator.latest.get(self.row["entity_id"])
        if pose is not None:
            age = float(self.scratch.time) - pose.timestamp
            text = f"{pose.object_id} xyz={np.array2string(np.asarray(pose.position), precision=3)} acquired={pose.timestamp:.3f}s age={age:.3f}s"
            cv2.rectangle(panel, (0, 457), (640, 480), (0, 0, 0), -1)
            cv2.putText(
                panel,
                text,
                (5, 473),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )
        self.video.write(panel)
        self.video_frames += 1

    def close(self):
        self.renderer.close()
        if self.video:
            self.video.release()


class ObservedPolicy:
    def __init__(self, policy, observer, skill):
        self.policy, self.observer, self.skill = policy, observer, skill

    def reset(self, *args):
        return self.policy.reset(*args)

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __setattr__(self, name, value):
        if name in {"policy", "observer", "skill"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.policy, name, value)

    def __call__(self, source, prior, phase=None, **_privileged_context):
        observer = self.observer
        observer.policy_calls += 1
        if self.skill in {"push", "press"}:
            # The caller's truth-derived latch/rest arguments are deliberately
            # discarded. Only observed causal state and calibrated fixture rest
            # enter the unchanged accepted contact policy.
            rest = observer.initial.require_entity(observer.row["entity_id"]).position
            if self.skill == "press":
                runtime = observer.task.workcell.config.fixtures[
                    observer.row["entity_id"]
                ]
                rest = observer.estimator.workcell.env.model.body(
                    runtime["body"]
                ).pos.copy()
            return self.policy(
                source, prior, fixture_rest=rest, pressed_once=observer.pressed_once
            )
        return self.policy(source, prior, observer.phase)
