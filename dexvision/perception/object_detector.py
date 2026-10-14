"""Conventional RGB colour components; no simulator labels at inference."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Visible component, exclusive XYXY box and an uncalibrated score."""

    class_id: str
    box: tuple[int, int, int, int]
    mask: np.ndarray
    confidence: float

    def __post_init__(self) -> None:
        if (not isfinite(self.confidence) or not 0 <= self.confidence <= 1 or
                len(self.box) != 4 or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]):
            raise ValueError("Invalid detection confidence/box.")


class ColorDetector:
    """Deterministic HSV segmenter fitted on training masks only.

    Colour is a simulator-specific appearance cue, not a semantic identity.
    Cylinder/button colour overlap is resolved only by the frozen fixture ROI.
    No pose, annotation, condition, task id or visibility is accepted here.
    """

    def __init__(self, palette: dict, *, hue_margin: float, minimum_pixels: int,
                 minimum_saturation: int, minimum_value: int,
                 button_roi: tuple[int, int, int, int], color_counts: list | None = None,
                 class_minimum_saturation: dict | None = None,
                 confidence_area_pixels: float | None = None,
                 class_minimum_value: dict | None = None,
                 class_hue_offsets: dict | None = None,
                 fragment_merge: dict | None = None):
        self.palette = palette
        self.hue_margin = hue_margin
        self.minimum_pixels = minimum_pixels
        self.minimum_saturation = minimum_saturation
        self.minimum_value = minimum_value
        self.button_roi = button_roi
        self.class_minimum_saturation = class_minimum_saturation or {}
        self.class_minimum_value = class_minimum_value or {}
        self.class_hue_offsets = class_hue_offsets or {}
        self.fragment_merge = fragment_merge or {}
        self.confidence_area_pixels = confidence_area_pixels or minimum_pixels*4
        self.color_likelihood = None
        if color_counts is not None:
            counts = np.asarray(color_counts, float)
            if counts.shape != (4, 180, 32) or not np.isfinite(counts).all() or np.any(counts < 0):
                raise ValueError("Invalid frozen training colour histogram.")
            # Deterministic candidate smoothing, never a test-fitted posterior.
            self.color_likelihood = np.stack([
                cv2.GaussianBlur(c, (5, 5), sigmaX=hue_margin/16, sigmaY=hue_margin/16)
                / max(c.sum(), 1.) for c in counts])

    def detect(self, rgb: np.ndarray) -> tuple[Detection, ...]:
        """Infer components from uint8 [H,W,3] RGB without mutating pixels."""
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Detector expects uint8 [H,W,3] RGB.")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        allowed = ((hsv[..., 1] >= self.minimum_saturation) &
                   (hsv[..., 2] >= self.minimum_value))
        # Core objects only. Placement targets are unsupported, not negatives
        # secretly supplied by an annotation list.
        masks = {}
        classes = ("rigid_cuboid", "rigid_cylinder", "rigid_flat_puck")
        selected_class = None
        if self.color_likelihood is not None:
            probabilities = self.color_likelihood[:, hsv[..., 0], hsv[..., 1]//8]
            selected_class = probabilities.argmax(0)
            selected_class[probabilities.max(0) <= 0] = 0
        for class_index, category in enumerate(classes, 1):
            delta = np.abs(hsv[..., 0].astype(float) - self.palette[category])
            masks[category] = (allowed & (selected_class == class_index) if selected_class is not None else
                               allowed & (np.minimum(delta, 180 - delta) <= self.hue_margin))
            if category in self.class_hue_offsets:
                lower, upper = self.class_hue_offsets[category]
                signed = (hsv[..., 0].astype(float)-self.palette[category]+90) % 180-90
                masks[category] = allowed & (signed >= lower) & (signed <= upper)
            masks[category] &= hsv[..., 1] >= self.class_minimum_saturation.get(category, self.minimum_saturation)
            masks[category] &= hsv[..., 2] >= self.class_minimum_value.get(category, self.minimum_value)
        x0, y0, x1, y1 = self.button_roi
        roi = np.zeros(rgb.shape[:2], bool)
        roi[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        if "button" in self.palette:
            delta = np.abs(hsv[..., 0].astype(float)-self.palette["button"])
            masks["button"] = allowed & roi & (np.minimum(delta, 180-delta) <= self.hue_margin)
        else:
            masks["button"] = masks["rigid_cylinder"] & roi
        masks["rigid_cylinder"] &= ~roi
        output = []
        for category, mask in masks.items():
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            for index in range(1, count):
                x, y, width, height, area = stats[index]
                if area < self.minimum_pixels:
                    continue
                component = labels == index
                # Fixed score is explicitly not a calibrated posterior.
                score = min(1., area / self.confidence_area_pixels)
                output.append(Detection(category, (int(x), int(y), int(x+width), int(y+height)),
                                        component, score))
        return merge_fragments(tuple(output), self.fragment_merge, self.confidence_area_pixels)


def merge_fragments(detections: tuple[Detection, ...], rules: dict, confidence_area_pixels: float) -> tuple[Detection, ...]:
    """Join nearby same-colour fragments using only RGB boxes/masks.

    An occluder can split one object into adjacent components. Frozen gap and
    group-extent bounds prevent an unrestricted scene-wide colour union.
    """
    groups = list(detections)
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(groups):
            rule = rules.get(left.class_id)
            if rule is None:
                continue
            for j in range(i+1, len(groups)):
                right = groups[j]
                if right.class_id != left.class_id:
                    continue
                a, b = np.asarray(left.box), np.asarray(right.box)
                separation = np.maximum(0, np.maximum(a[:2], b[:2])-np.minimum(a[2:], b[2:]))
                low, high = np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])
                if (np.linalg.norm(separation) > rule["maximum_gap_px"] or
                        max(high-low) > rule["maximum_extent_px"]):
                    continue
                mask = left.mask | right.mask
                groups[i] = Detection(left.class_id, tuple(int(v) for v in np.r_[low, high]), mask,
                                      min(1., mask.sum()/confidence_area_pixels))
                groups.pop(j)
                changed = True
                break
            if changed:
                break
    return tuple(groups)


def fit_palette(samples: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """Median hue of eligible labelled TRAIN RGB pixels; no validation fitting."""
    result = {}
    for category in ("rigid_cuboid", "rigid_cylinder", "rigid_flat_puck"):
        pixels = np.concatenate(samples[category]).astype(np.uint8).reshape(-1, 1, 3)
        hue = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)[..., 0].ravel()
        result[category] = float(np.median(hue))
    return result
