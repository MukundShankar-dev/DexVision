"""Recover frozen TRAIN provenance and balance roles/phases/physical sources."""

from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from dexvision.perception.height_model import decode_height


def recover_entries(rows, features, targets, calibration):
    """Attach supervision-only provenance using exact frozen harvest order.

    Require complete one-to-one visible TRAIN coverage and calibrated target
    equality. No image/inference/feature changes and no best-match reassignment.
    """
    labels = []
    for row in sorted(
        (r for r in rows if r["split"] == "train"),
        key=lambda r: (r["source_episode_id"], r["timestamp"]),
    ):
        if not row.get("phase_window_query", True):
            continue
        for label in row["annotations"]:
            if label["visible"] and label["class_id"] == "rigid_cuboid":
                labels.append((row, label))
    if len(labels) != len(features) or targets.shape != (len(features), 3):
        raise ValueError(
            "Complete ordered TRAIN coverage required; recover cannot skip or reassign labels."
        )
    values = []
    for x, y, (row, label) in zip(features, targets, labels, strict=True):
        if not np.allclose(
            decode_height(y, x, calibration),
            label["pose_6d"]["translation_m"],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Frozen TRAIN target/order differs from source label.")
        owner = row["evaluation_only_source_owner"]
        phase = next(
            (
                p["phase"]
                for p in owner["phase_intervals"]
                if p["start_frame"] <= row["source_frame_index"] < p["end_frame"]
            ),
            "unlabelled",
        )
        values.append(
            (
                x,
                y,
                owner.get("evaluation_only_parent_episode", row["source_episode_id"]),
                phase,
                label["object_id"] == owner["entity_id"],
            )
        )
    return values


def balanced_examples(entries):
    """Equal role/phase mass (primary4:other1), equal physical sources within it."""
    if not entries:
        raise ValueError("Missing TRAIN entries.")
    counts = Counter((v[2], v[3], v[4]) for v in entries)
    episodes = defaultdict(set)
    for v in entries:
        episodes[v[4], v[3]].add(v[2])
    w = np.array(
        [
            (4.0 if v[4] else 1.0)
            / (counts[v[2], v[3], v[4]] * len(episodes[v[4], v[3]]))
            for v in entries
        ]
    )
    w *= len(w) / w.sum()
    return np.stack([v[0] for v in entries]), np.stack([v[1] for v in entries]), w
