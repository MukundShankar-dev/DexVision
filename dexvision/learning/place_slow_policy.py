"""V9 opening-rate overlay on the checksum-locked V8 place controller."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_safety_policy import SafetyPolicy, load_safety_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-slow-v9'


def load_slow_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V9 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V9 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V9 parent recipe changed')
    _, spec, inputs = load_safety_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or overlay['cuboid_opening_step_nonplanar'] != 0.00625
            or overlay['cylinder_opening_step'] != 0.00625
            or overlay['flat_puck_opening_step'] != 0.00625):
        raise ValueError('Place V9 overlay differs from frozen opening rates')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V9 implementation changed: {name}')
    return overlay, spec, inputs


class SlowSafetyPolicy(SafetyPolicy):
    """Keep V8 geometry and safety, with slower opening for nonplanar objects."""

    def __init__(self, checkpoint_path, config):
        slow, _, _ = load_slow_spec(config)
        super().__init__(checkpoint_path, slow['parent_policy_recipe'])
        self.overlay = {**self.overlay,
            'cuboid_opening_step_nonplanar': slow['cuboid_opening_step_nonplanar']}
        self.spec['release_step_by_family']['cylinder'] = slow['cylinder_opening_step']
        self.spec['release_step_by_family']['flat_puck'] = slow['flat_puck_opening_step']
