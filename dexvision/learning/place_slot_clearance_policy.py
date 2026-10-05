"""V11 slot-aware clearance over the frozen V10 place controller."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_clearance_policy import ClearancePolicy, load_clearance_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-slot-clearance-v11'


def load_slot_clearance_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V11 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V11 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V11 parent recipe changed')
    _, spec, inputs = load_clearance_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or overlay['cuboid_placement_slot_clearance_m'] != 0.0):
        raise ValueError('Place V11 overlay differs from frozen slot clearance')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V11 implementation changed: {name}')
    return overlay, spec, inputs


class SlotClearancePolicy(ClearancePolicy):
    """Keep the measured release hand at its original height over slots."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_slot_clearance_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.slot_overlay = overlay

    def bind(self, row):
        super().bind(row)
        if ('block' in row['entity_id']
                and row['typed_goal']['target_id'] in ('setup_slot_a', 'setup_slot_b')):
            self.release_raise_m = self.slot_overlay['cuboid_placement_slot_clearance_m']
