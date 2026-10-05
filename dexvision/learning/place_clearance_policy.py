"""V10 target-aware release clearance over the frozen V9 place controller."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_slow_policy import SlowSafetyPolicy, load_slow_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-clearance-v10'
TARGET_TYPES = {'return_bin_left': 'receptacle',
                'return_bin_right': 'receptacle',
                'inspection_pad': 'planar_zone',
                'setup_slot_a': 'placement_slot',
                'setup_slot_b': 'placement_slot'}
FROZEN_CLEARANCE_M = {
    'cuboid': {'receptacle': 0.0015, 'planar_zone': 0.0015,
               'placement_slot': 0.0015},
    'cylinder': {'receptacle': 0.003, 'planar_zone': 0.0,
                 'placement_slot': 0.0},
    'flat_puck': {'receptacle': 0.0, 'planar_zone': 0.003,
                  'placement_slot': 0.003},
}


def load_clearance_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V10 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V10 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V10 parent recipe changed')
    _, spec, inputs = load_slow_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or overlay['release_clearance_m'] != FROZEN_CLEARANCE_M):
        raise ValueError('Place V10 overlay differs from frozen clearance rule')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V10 implementation changed: {name}')
    return overlay, spec, inputs


class ClearancePolicy(SlowSafetyPolicy):
    """Lift the held hand by a small target-dependent amount when release starts."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_clearance_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.clearance = overlay['release_clearance_m']
        self.release_raise_m = 0.0

    def bind(self, row):
        super().bind(row)
        object_id = row['entity_id']
        family = ('cuboid' if 'block' in object_id else
                  'cylinder' if 'cylinder' in object_id else
                  'flat_puck' if 'puck' in object_id else None)
        target_type = TARGET_TYPES.get(row['typed_goal']['target_id'])
        if family is None or target_type is None:
            raise ValueError('Place V10 requires a known object family and target type')
        self.release_raise_m = self.clearance[family][target_type]

    def __call__(self, source, prior, phase=None):
        latched = self.latch is not None
        action, prediction = super().__call__(source, prior, phase)
        if self.latch is not None and not latched:
            self.latch[2] += self.release_raise_m
            action[2] = self.latch[2]
        return action, prediction
