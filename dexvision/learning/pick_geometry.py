"""Training-audited, causal compact pick geometry for the V4 repair recipe."""
from __future__ import annotations

import numpy as np

from dexvision.learning.reach_repair import calibrated_features


def hand_rotation(values, names):
    """[...,3,3] prior-command rotation; no measured/command origin mixing."""
    q = np.asarray(values)[..., [names.index(f'prior_applied_actions.base_orientation_target/q{a}') for a in 'wxyz']]
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)), axis=-1).reshape((*q.shape[:-1], 3, 3))


def family_values(values, names):
    return np.asarray(values)[..., [names.index(f'entity.class.{f}') for f in ('cuboid', 'cylinder', 'flat_puck')]]


def compact_pick_features(values, names, context, spec):
    """Relative geometry, command progress, contact and motion; no absolute posture inputs."""
    values = np.asarray(values, float)
    def get(keys):
        return values[..., [names.index(k) for k in keys]]
    rotation = hand_rotation(values, names)
    def local(v):
        return np.einsum('...ji,...j->...i', rotation, v)
    prior = get([f'prior_applied_actions.base_position_target/{a}' for a in 'xyz'])
    entity = get([f'entity.position_m.{j}' for j in range(3)])
    calibrated = calibrated_features(values, names, spec, {'calibrate_measured_frame': True})
    start = np.asarray(spec['closure']['open_targets'])
    direction = np.asarray(spec['closure']['closed_targets']) - start
    fingers = get([n for n in names if n.startswith('prior_applied_actions.')][7:])
    fraction = (fingers - start) @ direction / (direction @ direction)
    observed = get([f'entity.rotation_6d.{j}' for j in range(6)])
    observed = np.stack((observed[..., :3], observed[..., 3:], np.cross(observed[..., :3], observed[..., 3:])), axis=-1)
    ref = np.column_stack((context[1:4], context[4:7], np.cross(context[1:4], context[4:7])))
    error = np.swapaxes(rotation, -1, -2) @ ref @ np.swapaxes(observed, -1, -2) @ rotation
    identity6 = np.array([1., 0, 0, 0, 1, 0])
    error6 = np.swapaxes(error[..., :, :2], -1, -2).reshape((*values.shape[:-1], 6)) - identity6
    parts = [fraction[..., None], (prior[..., 2] - context[7])[..., None] / spec['position_scale_m'],
             (entity[..., 2] - context[0])[..., None] / spec['position_scale_m'], prior[..., 2, None] / spec['position_scale_m'],
             local(prior - calibrated[..., :3]) / spec['position_scale_m'], local(entity - prior) / spec['position_scale_m'],
             local(get([f'entity.linear_velocity_mps.{j}' for j in range(3)])) / spec['position_scale_m'],
             local(get([f'entity.angular_velocity_radps.{j}' for j in range(3)])),
             error6 / spec['position_scale_m'], get(['entity.supported.0', 'entity.held.0', 'entity.contact_with_hand.0']),
             family_values(values, names)]
    output_names = ['closure_fraction', 'command_lift', 'object_lift', 'command_height']
    for group in ('tracking_lag_hand', 'object_minus_command_hand', 'object_velocity_hand', 'object_angular_velocity_hand'):
        output_names.extend(f'{group}/{a}' for a in 'xyz')
    output_names.extend(f'object_orientation_error_hand/{j}' for j in range(6))
    output_names.extend(('supported', 'held', 'contact', 'class/cuboid', 'class/cylinder', 'class/flat_puck'))
    from dexvision.learning.phase_losses import pick_phases
    phases = pick_phases(get(['entity.held.0'])[..., 0], entity[..., 2] - context[0], spec['lift_height_m'])
    return np.concatenate(parts, axis=-1), output_names, phases


def geometry_targets(episode, targets, masks, spec, names):
    """Keep exact targets; mask closure/lift where geometry makes the gain unidentifiable."""
    from dexvision.learning.pick_policy import closure_fraction
    rotation = hand_rotation(episode.inputs, names)
    targets[:, 3:6] = np.einsum('nji,nj->ni', rotation, targets[:, 3:6])
    prior, applied = (episode.records[k] for k in ('prior_applied_actions', 'applied_actions'))
    fraction = closure_fraction(prior[:, 7:], spec)
    if np.any(targets[:, 8] < -1e-10) or np.any(targets[:, 2] < -1e-10):
        raise ValueError('V4 requires independently audited monotone closure and lift')
    if np.any((targets[:, 2] > 1e-10) & (fraction < 1 - 1e-8)):
        raise ValueError('V4 lift before full closure is unsupported')
    family = np.argmax(family_values(episode.inputs, names), axis=-1)
    puck = family == 2
    if np.any(np.abs(targets[puck, 3:6]) > 1e-10):
        raise ValueError('Puck rotation is not invariant')
    masks[puck, 3:6] = 0
    if 'orientation_feedback' in spec:
        from dexvision.learning.pick_feedback import orientation_feedback_vector
        from dexvision.learning.pick_policy import boundary_context
        from dexvision.learning.skill_datasets import relative_rotation
        settings = spec['orientation_feedback']
        context = boundary_context(episode.inputs[0], names, spec)
        direction = orientation_feedback_vector(episode.inputs, names, context, settings['maximum_step_rad'])
        actual = np.asarray([relative_rotation(a[3:7], p[3:7]) for a, p in zip(applied, prior, strict=True)])
        length = np.linalg.norm(direction, axis=-1)
        magnitude = np.linalg.norm(actual, axis=-1)
        reconstructed = direction * (magnitude / np.maximum(length, 1e-12))[:, None]
        if (np.any(np.linalg.norm(actual - reconstructed, axis=-1) > settings['reconstruction_tolerance_rad'])
                or np.any(magnitude > length + settings['reconstruction_tolerance_rad'])
                or np.any(magnitude > settings['maximum_step_rad'])):
            raise ValueError('Unassisted orientation targets violate the audited correction direction/bounds')
        targets[:, 3:6] = 0
        masks[:, 3:6] = 0
        targets[:, 3] = magnitude / settings['maximum_step_rad']
        masks[:, 3] = length > settings['unidentifiable_error_rad']
    masks[fraction >= 1 - 1e-8, 8] = 0
    maximum = prior[0, 2] + np.asarray(spec['geometry']['maximum_lift_m'])[family]
    if np.any(applied[:, 2] > maximum + 1e-8):
        raise ValueError('Source lift exceeds the family bound')
    masks[prior[:, 2] >= maximum - 1e-8, 2] = 0
    if 'binary_action_gates' in spec:
        for index in (2, 8):
            active = targets[masks[:, index] > 0, index]
            if np.any(np.minimum(np.abs(active), np.abs(active - 1)) > spec['binary_action_gates']['target_tolerance']):
                raise ValueError('Binary pick gates require independently audited zero-or-one source targets')
    return targets, masks


def geometry_action(prediction, prior, spec, plan, features, names, context):
    from dexvision.learning.pick_policy import closure_fraction
    from dexvision.learning.skill_models import decode_delta
    fraction = closure_fraction(np.asarray(prior)[7:], spec)
    family = int(np.argmax(family_values(features, names)))
    maximum = context[7] + spec['geometry']['maximum_lift_m'][family]
    if prior[2] > maximum + 1e-8:
        raise ValueError('Prior command exceeds the entry-derived lift envelope')
    if 'binary_action_gates' in spec:
        prediction = np.asarray(prediction).copy()
        prediction[[2, 8]] = prediction[[2, 8]] > spec['binary_action_gates']['threshold']
    scaled = prediction * np.asarray(spec['output_scales'])
    scaled[8:] = 0
    scaled[3:6] = hand_rotation(features, names) @ scaled[3:6]
    if family == 2:
        scaled[3:6] = 0
    if 'orientation_feedback' in spec:
        from dexvision.learning.pick_feedback import orientation_feedback_vector
        maximum_step = spec['orientation_feedback']['maximum_step_rad']
        correction = orientation_feedback_vector(features, names, context, maximum_step)
        length = np.linalg.norm(correction)
        correction *= min(length, maximum_step * max(0., prediction[3])) / max(length, 1e-12)
        scaled[3:6] = correction / plan['action']['maximum_angular_change_rad']
    scaled[2] = max(0., scaled[2]) if fraction >= 1 - 1e-8 else 0.
    action = decode_delta(scaled, prior, 'acquire', plan)
    action[2] = min(action[2], maximum)
    fraction = min(1., fraction + max(0., prediction[8]) * spec['closure']['maximum_step'])
    start = np.asarray(spec['closure']['open_targets'])
    action[7:] = start + fraction * (np.asarray(spec['closure']['closed_targets']) - start)
    return action
