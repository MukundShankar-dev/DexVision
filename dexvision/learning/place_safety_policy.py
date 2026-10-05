"""V8 place decoder safety overlay for unchanged, fully trained V6 weights."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.place_guarded_policy import GuardedPolicy, load_guarded_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-safety-v8'


def load_safety_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V8 safety recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V8 safety recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V8 parent policy recipe changed')
    spec, inputs = load_guarded_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or overlay['command_object_lag_hold_m'] != 0.05
            or overlay['cuboid_opening_step_nonplanar'] != 0.0125
            or overlay['cuboid_opening_step_planar'] != 0.025
            or overlay['edge_release']['target_abs_y_min_m'] != 0.09
            or overlay['edge_release']['object_abs_y_min_m'] != 0.09
            or overlay['edge_release']['outward_velocity_min_mps'] != 0.02):
        raise ValueError('Place V8 overlay differs from the frozen safety controls')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V8 implementation changed: {name}')
    return overlay, spec, inputs


class SafetyPolicy(GuardedPolicy):
    """Hold on abnormal tracking lag; release an outward-moving edge cuboid."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_safety_spec(config)
        self.overlay = overlay
        self.initial_offset = None
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])

    def reset(self):
        super().reset()
        self.initial_offset = None

    def __call__(self, source, prior, phase=None):
        from dexvision.evaluation.place_repair import release_fingers

        index = self.index
        object_position = source[[index[f'entity.position_m.{j}'] for j in range(3)]]
        target_position = source[[index[f'goal.target_position.{j}'] for j in range(3)]]
        outward_velocity = source[index['entity.linear_velocity_mps.1']]
        self.spec['release_step_by_family']['cuboid'] = (
            self.overlay['cuboid_opening_step_planar']
            if source[index['goal.target_type.1']] > .5
            else self.overlay['cuboid_opening_step_nonplanar'])
        prior = np.asarray(prior, float)
        if self.initial_offset is None:
            self.initial_offset = prior[:3] - object_position
        out, prediction = super().__call__(source, prior, phase)
        held = source[index['entity.held.0']] > .5
        lag = np.linalg.norm((prior[:2] - object_position[:2]) - self.initial_offset[:2])
        if held and lag > self.overlay['command_object_lag_hold_m'] and self.latch is None:
            out = prior.copy()
        edge = self.overlay['edge_release']
        if (self.latch is None and held and source[index['entity.class.cuboid']] > .5
                and abs(target_position[1]) > edge['target_abs_y_min_m']
                and abs(object_position[1]) > edge['object_abs_y_min_m']
                and np.sign(target_position[1]) * outward_velocity > edge['outward_velocity_min_mps']
                and np.linalg.norm(target_position - object_position)
                <= self.spec['geometry']['release_distance_by_family_m']['cuboid']):
            self.latch = prior[:7].copy()
            out = prior.copy()
            out[:7] = self.latch
            out[7:] = release_fingers(prior[7:], self.spec['open_targets'],
                self.spec['closed_targets'], self.spec['release_step_by_family']['cuboid'])
        return out, prediction
