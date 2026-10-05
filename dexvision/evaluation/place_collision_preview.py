"""Copied-state collision-only preview for V23 small-block slot B."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dexvision.evaluation.evaluate_skill import apply_action
from dexvision.evaluation.place_shadow_action import preview_place_action as _original_preview


def _collision_only_preview(task, applied, prior, fields, inputs, row, policy):
    """Choose a safe nearby pose using one-step MuJoCo predictions.

    The learned action remains the first choice. Only the source-exposed
    small-block/slot-B cell with pre-opening forearm contact is eligible. This
    controller never opens fingers or modifies a latched release.
    """
    if ((policy.object_id, policy.target_id) not in {
            ('block_small', 'setup_slot_b')}
            or policy.latch is not None or applied[2] >= 0.135
            or np.linalg.norm(applied[7:] - prior[7:]) > 1e-9):
        return applied
    env, live = task.env, task.env.data
    quaternion = Rotation.from_quat(np.roll(applied[3:7], -1))
    choices = [applied]
    for axis in np.eye(3):
        for degrees in (-3., 3.):
            candidate = applied.copy()
            candidate[3:7] = np.roll((Rotation.from_rotvec(
                axis * np.deg2rad(degrees)) * quaternion).as_quat(), 1)
            choices.append(candidate)
    for rise in (0.002, 0.005):
        candidate = applied.copy()
        candidate[2] += rise
        choices.append(candidate)
    for axis in (0, 1):
        for shift in (-0.002, 0.002):
            candidate = applied.copy()
            candidate[axis] += shift
            choices.append(candidate)

    lower = np.asarray([field['lower'] for field in fields])
    upper = np.asarray([field['upper'] for field in fields])
    board = inputs.requirements['workcell']['board_workspace']
    minimum = np.asarray(board['min_xy_m']) + board['safe_edge_margin_m']
    maximum = np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']
    scored = []
    try:
        for candidate in choices:
            if np.any(candidate < lower) or np.any(candidate > upper):
                scored.append((True, float('inf'), candidate))
                continue
            scratch = env._mujoco.MjData(env.model)
            env._mujoco.mj_copyData(scratch, env.model, live)
            env.data = scratch
            apply_action(task, candidate, fields)
            for _ in range(inputs.plan['action']['simulation_steps_per_action']):
                env._mujoco.mj_step(env.model, scratch)
            env._mujoco.mj_forward(env.model, scratch)
            world = task.workcell.get_world_state()
            contact = any('workcell_table' in pair and
                any(name.startswith('rh_') for name in pair) for pair in world.contacts)
            obj = np.asarray(world.require_entity(row['entity_id']).position)
            goal = np.asarray(world.require_entity(row['typed_goal']['target_id']).position)
            off_board = bool(np.any(obj[:2] < minimum) or np.any(obj[:2] > maximum))
            scored.append((contact or off_board, float(np.linalg.norm(obj - goal)), candidate))
    finally:
        env.data = live
    baseline = scored[0]
    safe = [item for item in scored if not item[0]]
    if not safe or (not baseline[0] and baseline[1] <=
                    min(item[1] for item in safe) + 1000.0):
        return applied
    best = min(safe, key=lambda item: item[1])
    if baseline[0] or best[1] < baseline[1] - 0.001:
        return best[2]
    return applied


def preview_place_action(task, applied, prior, fields, inputs, row, policy):
    """Retain V21 preview elsewhere; intervene at slot B only on predicted collision."""
    if (policy.object_id, policy.target_id) == ('block_small', 'setup_slot_b'):
        return _collision_only_preview(task, applied, prior, fields, inputs, row, policy)
    return _original_preview(task, applied, prior, fields, inputs, row, policy)
