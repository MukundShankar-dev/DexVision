"""All-visible-instance denominators and temporal perception diagnostics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


CORE_CLASSES = ("rigid_cuboid", "rigid_cylinder", "rigid_flat_puck", "button")


def box_iou(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    extent = np.maximum(0., np.minimum(left[2:], right[2:])-np.maximum(left[:2], right[:2]))
    intersection = float(np.prod(extent))
    union = float(np.prod(left[2:]-left[:2])+np.prod(right[2:]-right[:2])-intersection)
    return intersection/union if union > 0 else 0.


def match_detections(annotations: list[dict], estimates: tuple) -> dict[int, int]:
    """One-to-one same-class IoU matching, evaluation only, never tracking."""
    costs = np.ones((len(annotations), len(estimates)))
    for i, annotation in enumerate(annotations):
        if annotation["visible"] and annotation["class_id"] in CORE_CLASSES:
            for j, estimate in enumerate(estimates):
                if annotation["class_id"] == estimate.track.detection.class_id:
                    costs[i, j] = 1-box_iou(annotation["box_xyxy"], estimate.track.detection.box)
    rows, columns = linear_sum_assignment(costs)
    return {int(i): int(j) for i, j in zip(rows, columns) if costs[i, j] <= .5}


def summarize_instances(rows: list[dict], *, isolation_rejection_fraction: float) -> dict:
    """Every visible core GT instance is a row, including missed/rejected poses.

    Position numbers are conditional *hypothesis* errors, with their count and
    certified availability reported separately. Zero denominators never pass.
    Identity denominators are all visible instances, not just detections.
    """
    visible = [r for r in rows if r["eligible"]]
    matched = [r for r in visible if r["matched"]]
    errors = [r["position_error_m"] for r in matched if r["position_error_m"] is not None]
    velocities = [r["velocity_error_mps"] for r in matched if r["velocity_error_mps"] is not None]
    yaw = [r["symmetry_yaw_error_rad"] for r in matched if r["symmetry_yaw_error_rad"] is not None]
    switches = sum(r["identity_switch"] for r in visible)
    recall = len(matched)/len(visible) if visible else None
    median = float(np.median(errors)) if errors else None
    p95 = float(np.percentile(errors, 95)) if errors else None
    identity_rate = 1000*switches/len(visible) if visible else None
    gates = {
        "recall": recall is not None and recall >= .95,
        "median_position": median is not None and median <= .010,
        "p95_position": p95 is not None and p95 <= .025,
        "identity": identity_rate is not None and identity_rate <= 1.,
        "stale_ambiguous_rejection": isolation_rejection_fraction == 1.,
    }
    return {
        "visible_eligible_instances": len(visible), "detections_matched_iou_0_5": len(matched),
        "missing_detections": len(visible)-len(matched), "recall": recall,
        "position_hypothesis_count": len(errors), "position_error_median_m": median,
        "position_error_p95_m": p95,
        "position_bias_xyz_m": np.mean([r["position_delta_m"] for r in matched
                                       if r["position_delta_m"] is not None], axis=0).tolist() if errors else None,
        "certified_typed_pose_count": sum(r["certified_pose"] for r in visible),
        "full_orientation_available_count": 0, "angular_velocity_available_count": 0,
        "symmetry_yaw_diagnostic_count": len(yaw),
        "symmetry_yaw_p95_rad": float(np.percentile(yaw, 95)) if yaw else None,
        "linear_velocity_diagnostic_count": len(velocities),
        "linear_velocity_error_p95_mps": float(np.percentile(velocities, 95)) if velocities else None,
        "identity_switches": switches, "identity_switches_per_1000_visible_instances": identity_rate,
        "identity_matched_coverage": len(matched)/len(visible) if visible else None,
        "reacquisitions": sum(r["reacquired"] for r in visible),
        "reject_count": sum(bool(r["rejection_reason"]) for r in visible),
        "synthetic_rejection_fraction": isolation_rejection_fraction,
        "numerical_gates": gates, "all_numerical_gates_pass": all(gates.values()),
    }


def breakdowns(rows: list[dict], rejection_fraction: float) -> dict:
    result = {}
    for field in ("class_id", "object_id", "condition", "episode", "skill_stratum", "phase_stratum"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result[field] = {key: summarize_instances(values, isolation_rejection_fraction=rejection_fraction)
                         for key, values in sorted(groups.items())}
    return result
