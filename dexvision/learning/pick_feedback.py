"""Explicit orientation-error direction for an independently audited pick decoder.

The MLP learns a bounded correction gain. Direction and maximum angular step are
engineered from causal object state and the entry reference, not learned vision
or an expert continuation.
"""
from __future__ import annotations

import numpy as np


def _rotation_vector(matrix):
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1) / 2, -1., 1.)
    angle = np.arccos(cosine)
    vee = np.stack((matrix[..., 2, 1] - matrix[..., 1, 2],
                    matrix[..., 0, 2] - matrix[..., 2, 0],
                    matrix[..., 1, 0] - matrix[..., 0, 1]), axis=-1) / 2
    factor = np.ones_like(angle)
    ordinary = angle > 1e-7
    np.divide(angle, np.sin(angle), out=factor, where=ordinary & (angle < np.pi - 1e-6))
    result = vee * factor[..., None]
    if np.any(angle >= np.pi - 1e-6):
        flat_result = result.reshape(-1, 3)
        for index in np.flatnonzero((angle >= np.pi - 1e-6).reshape(-1)):
            m = matrix.reshape(-1, 3, 3)[index]
            _, vectors = np.linalg.eigh((m + np.eye(3)) / 2)
            flat_result[index] = vectors[:, -1] * angle.reshape(-1)[index]
    return result


def orientation_feedback_vector(values, names, context, maximum_step_rad):
    """[...,3] bounded world-frame error; cuboid full rotation, cylinder axial, puck zero."""
    values = np.asarray(values, float)
    first = values[..., [names.index(f'entity.rotation_6d.{j}') for j in range(3)]]
    second = values[..., [names.index(f'entity.rotation_6d.{j}') for j in range(3, 6)]]
    observed = np.stack((first, second, np.cross(first, second)), axis=-1)
    context = np.asarray(context, float)
    ref = np.column_stack((context[1:4], context[4:7], np.cross(context[1:4], context[4:7])))
    full = _rotation_vector(ref @ np.swapaxes(observed, -1, -2))
    old_up, new_up = observed[..., :, 2], ref[:, 2]
    cross = np.cross(old_up, new_up)
    sine = np.linalg.norm(cross, axis=-1)
    cosine = np.clip(np.sum(old_up * new_up, axis=-1), -1., 1.)
    axial = cross * (np.arctan2(sine, cosine) / np.maximum(sine, 1e-12))[..., None]
    opposite = cosine <= -1 + 1e-9
    if np.any(opposite):
        orthogonal = np.cross(old_up, np.array([1., 0., 0.]))
        alternate = np.cross(old_up, np.array([0., 1., 0.]))
        orthogonal = np.where((np.linalg.norm(orthogonal, axis=-1) < 1e-9)[..., None], alternate, orthogonal)
        axial = np.where(opposite[..., None],
            np.pi * orthogonal / np.maximum(np.linalg.norm(orthogonal, axis=-1)[..., None], 1e-12), axial)
    family = np.argmax(values[..., [names.index(f'entity.class.{f}') for f in ('cuboid', 'cylinder', 'flat_puck')]], axis=-1)
    result = np.where((family == 0)[..., None], full, np.where((family == 1)[..., None], axial, 0.))
    length = np.linalg.norm(result, axis=-1)
    result *= np.minimum(1., maximum_step_rad / np.maximum(length, 1e-12))[..., None]
    held = values[..., names.index('entity.held.0')] > 0.5
    return np.where(held[..., None], result, 0.)


def closed_contact_counts(values, names, spec):
    """Causal consecutive full-closure/held samples, including the current observation."""
    from dexvision.learning.pick_policy import closure_fraction
    values = np.atleast_2d(values)
    fingers = values[:, [j for j, n in enumerate(names) if n.startswith('prior_applied_actions.')][7:]]
    closed = closure_fraction(fingers, spec) >= 1 - 1e-8
    held = values[:, names.index('entity.held.0')] > 0.5
    result, count = [], 0
    for active in closed & held:
        count = count + 1 if active else 0
        result.append(count)
    return np.asarray(result)
