"""Hold the verified setup wrist for V24 large-block right-bin placement."""
from __future__ import annotations

from dexvision.evaluation.place_collision_preview import preview_place_action as _parent_preview


def preview_place_action(task, applied, prior, fields, inputs, row, policy):
    """Keep the setup wrist during transport, then use the V23 preview."""
    if [policy.object_id, policy.target_id] in policy.wrist_hold_recipe['wrist_hold_cells']:
        if policy.entry_quaternion is None:
            policy.entry_quaternion = prior[3:7].copy()
        if policy.latch is None:
            applied = applied.copy()
            applied[3:7] = policy.entry_quaternion
    return _parent_preview(task, applied, prior, fields, inputs, row, policy)
