"""Frozen recipe, verified visual inputs and RGB-only baseline composition."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.object_detector import ColorDetector, fit_palette
from dexvision.perception.object_pose import MetricPoseEstimator
from dexvision.perception.object_tracker import ObjectTracker


IMPLEMENTATION_FILES = (
    "dexvision/perception/object_detector.py", "dexvision/perception/object_tracker.py",
    "dexvision/perception/object_pose.py", "dexvision/perception/perception_baseline.py",
    "dexvision/evaluation/perception_metrics.py", "dexvision/evaluation/evaluate_perception.py",
    "dexvision/evaluation/perception_audit.py", "dexvision/learning/perception_baseline.py",
    "dexvision/perception/object_observations.py", "dexvision/sim/world_state.py",
    "dexvision/perception/body_center_model.py", "dexvision/perception/metric_geometry.py",
    "dexvision/learning/perception_repair.py", "dexvision/apps/train_perception.py",
    "dexvision/apps/evaluate_perception.py",
    "dexvision/perception/height_model.py",
)


def load_recipe(path: str | Path) -> dict:
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if digest_file(path) != expected:
        raise ValueError("Perception recipe checksum mismatch.")
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if recipe["version"] not in {"level5/perception-baseline-v1", "level5/perception-baseline-v2",
                                 "level5/perception-repair-v3", "level5/perception-repair-v4",
                                 "level5/perception-repair-v5", "level5/perception-repair-v6", "level5/perception-repair-v7", "level5/perception-repair-v8"}:
        raise ValueError("Unsupported perception recipe version.")
    if recipe["minimum_confidence"] != .8 or recipe["maximum_age_s"] != .1:
        raise ValueError("Historical delivery thresholds must remain unchanged.")
    for name, digest in recipe["input_digests"].items():
        if digest_file(Path(name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    for name, digest in recipe["implementation_digests"].items():
        if digest_file(Path(name)) != digest:
            raise ValueError(f"Frozen implementation changed: {name}")
    recipe["_recipe_sha256"] = digest_file(path)
    return recipe


def visual_rows(recipe: dict, *, allowed_splits: tuple[str, ...] | None = None) -> list[dict]:
    """Verify release membership and whole-episode/session ownership."""
    root = Path(recipe["visual_root"])
    release = json.loads(Path(recipe["source_manifest"]).read_text(encoding="utf-8"))
    for relative in ("frames.jsonl", "calibration.json"):
        path = root/relative
        key = path.as_posix()
        if release["files"][key]["sha256"] != digest_file(path):
            raise ValueError(f"Visual stream differs from immutable release: {key}")
    rows = []
    for line in (root/"frames.jsonl").read_text(encoding="utf-8").splitlines():
        # Repair work excludes test rows before decoding their annotations.
        if allowed_splits is not None:
            split = re.search(r'"split"\s*:\s*"(train|validation|test)"', line)
            if split is None or split.group(1) not in allowed_splits:
                continue
        rows.append(json.loads(line))
    released_owners = {}
    release_directory = Path(recipe["source_manifest"]).parent
    for split in ("train", "validation", "test"):
        if allowed_splits is not None and split not in allowed_splits:
            continue
        path = release_directory/"splits"/f"{split}.json"
        if digest_file(path) != recipe["input_digests"][path.as_posix()]:
            raise ValueError("Frozen source split changed.")
        for episode in json.loads(path.read_text(encoding="utf-8"))["episodes"]:
            released_owners[episode["episode_id"]] = episode
    owners = {"episode": {}, "session": {}}
    seen = set()
    for row in rows:
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown visual split.")
        key = row["source_episode_id"], row["source_frame_index"]
        if key in seen:
            raise ValueError("Duplicate episode/frame in visual stream.")
        seen.add(key)
        owner = released_owners.get(row["source_episode_id"])
        if (owner is None or owner["split"] != row["split"] or
                owner["recording_session_id"] != row["recording_session_id"]):
            raise ValueError("Visual row differs from released episode/session split ownership.")
        row["evaluation_only_source_owner"] = owner
        for field, column in (("episode", "source_episode_id"), ("session", "recording_session_id")):
            old = owners[field].setdefault(row[column], row["split"])
            if old != row["split"]:
                raise ValueError(f"Cross-split {field} ownership.")
    return rows


def read_rgb(root: Path, row: dict, *, with_training_mask: bool = False):
    """Read/verify pixels; masks may enter fitting only on training rows."""
    if with_training_mask and row["split"] != "train":
        raise ValueError("Only training labels may fit detector parameters.")
    path = root/row["rgb"]
    if digest_file(path) != row["rgb_sha256"]:
        raise ValueError(f"RGB digest mismatch: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read RGB: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not with_training_mask:
        return rgb
    path = root/row["mask"]
    if digest_file(path) != row["mask_sha256"]:
        raise ValueError(f"Mask digest mismatch: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != rgb.shape[:2]:
        raise ValueError("Invalid training mask.")
    return rgb, mask


def training_palette(recipe: dict, rows: list[dict]) -> dict:
    samples = {c: [] for c in ("rigid_cuboid", "rigid_cylinder", "rigid_flat_puck")}
    counts = np.zeros((4, 180, 32), np.int64)
    root = Path(recipe["visual_root"])
    for row in rows:
        if row["split"] != "train":
            continue
        rgb, mask = read_rgb(root, row, with_training_mask=True)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        supervised_class = np.zeros(mask.shape, np.int32)
        for annotation in row["annotations"]:
            category = annotation["class_id"]
            if category == "button" and annotation["visible"]:
                # Button has the same green appearance family; the frozen
                # calibrated ROI distinguishes fixture from loose cylinders.
                supervised_class[mask == annotation["mask_value"]] = 2
            if category in samples and annotation["visible"]:
                supervised_class[mask == annotation["mask_value"]] = list(samples).index(category)+1
                pixels = rgb[(mask == annotation["mask_value"]) &
                             (hsv[..., 1] >= recipe["minimum_saturation"])]
                # Fixed deterministic stride bounds memory; no random seed or
                # test-selected pixel sampling.
                if len(pixels):
                    samples[category].append(pixels[::recipe["palette_pixel_stride"]])
        stride = recipe["palette_pixel_stride"]
        hs = hsv.reshape(-1, 3)[::stride]
        indices = supervised_class.ravel()[::stride]*180*32 + hs[:, 0].astype(np.int32)*32 + hs[:, 1]//8
        counts += np.bincount(indices, minlength=4*180*32).reshape(4, 180, 32)
    if any(not values for values in samples.values()):
        raise ValueError("Missing training class palette.")
    return {"palette": fit_palette(samples), "color_counts": counts.tolist()}


class PerceptionBaseline:
    """Only RGB and acquisition timestamp enter the inferred branch."""

    def __init__(self, recipe: dict, model: dict):
        self.model = model
        calibration = json.loads((Path(recipe["visual_root"])/"calibration.json").read_text(encoding="utf-8"))
        self.detector = ColorDetector(model["palette"], hue_margin=model["hue_margin"],
                                      minimum_pixels=model.get("minimum_pixels", recipe["minimum_pixels"]),
                                      minimum_saturation=recipe["minimum_saturation"],
                                      minimum_value=recipe["minimum_value"],
                                      button_roi=tuple(recipe["button_roi_xyxy"]),
                                      color_counts=model.get("color_counts"),
                                      class_minimum_saturation=model.get("class_minimum_saturation"),
                                      confidence_area_pixels=model.get("confidence_area_pixels"),
                                      class_minimum_value=model.get("class_minimum_value"),
                                      class_hue_offsets=model.get("class_hue_offsets"),
                                      fragment_merge=model.get("fragment_merge"))
        self.tracker = ObjectTracker(**recipe["tracker"])
        if "body_center_models" in model:
            from dexvision.perception.body_center_model import BodyCenterEstimator
            self.pose = BodyCenterEstimator(calibration, model["body_center_models"],
                                           maximum_gap_s=recipe["tracker"]["maximum_gap_s"])
        else:
            self.pose = MetricPoseEstimator(calibration, recipe["geometry"], **recipe["pose"])

    def reset(self) -> None:
        self.tracker.reset()
        self.pose.reset()

    def infer(self, rgb: np.ndarray, timestamp: float) -> tuple:
        if hasattr(self.pose, "estimate_rgb"):
            return tuple(self.pose.estimate_rgb(rgb, track) for track in
                         self.tracker.update(self.detector.detect(rgb), timestamp))
        return tuple(self.pose.estimate(track) for track in
                     self.tracker.update(self.detector.detect(rgb), timestamp))


def save_artifact_index(directory: Path) -> None:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (directory/"SHA256SUMS").write_text("".join(
        f"{digest_file(p)}  {p.relative_to(directory).as_posix()}\n" for p in files), encoding="utf-8")


def model_digest(model: dict) -> str:
    return hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_model(recipe: dict) -> dict:
    directory = Path(recipe["model_dir"])
    model = json.loads((directory/"selected.json").read_text(encoding="utf-8"))
    selection = json.loads((directory/"selection.json").read_text(encoding="utf-8"))
    if selection["recipe_sha256"] != recipe["_recipe_sha256"]:
        raise ValueError("Selected perception recipe changed.")
    if selection["model_sha256"] != model_digest(model):
        raise ValueError("Selected perception model changed.")
    if selection["recipe_input_digests"] != recipe["input_digests"]:
        raise ValueError("Selected perception data/accepted bundle mismatch.")
    if selection["implementation_digests"] != recipe["implementation_digests"]:
        raise ValueError("Selected implementation mismatch.")
    return model


def write_report(path: Path, report: dict) -> None:
    write_json(path, report)
