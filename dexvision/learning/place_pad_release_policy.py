"""V25 bounded geometric pad release over the exact V24 place policy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.place_wrist_hold_policy import (
    WristHoldPlacePolicy, load_wrist_hold_spec,
)
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-pad-release-v25'
PAD_RELEASE = {'object_id': 'block_large', 'target_id': 'inspection_pad', 'distance_m': 0.0245}


def load_pad_release_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V25 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V25 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V25 parent recipe changed')
    old, spec, inputs = load_wrist_hold_spec(parent)
    expected = {
        'seeds': old['seeds'], 'candidate_epochs': old['candidate_epochs'],
        'validation_seed_start': old['validation_seed_start'],
        'selection_seed_start': 1580260900, 'confirmation_seed_start': 1600260900,
        'selection_resets_per_cell': 2, 'confirmation_resets_per_cell': 2,
        'pad_release_controls': PAD_RELEASE,
    }
    if (any(overlay[key] != value for key, value in expected.items())
            or PAD_RELEASE['distance_m'] != spec['geometry']['release_distance_by_family_m']['cuboid']):
        raise ValueError('Place V25 release controls or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V25 implementation changed: {name}')
    return overlay, spec, inputs


class PadReleasePlacePolicy(WristHoldPlacePolicy):
    """Release the large block on the pad at the existing physical distance bound."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_pad_release_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.pad_release_controls = overlay['pad_release_controls']

    def __call__(self, source, prior, phase=None):
        action, prediction = super().__call__(source, prior, phase)
        controls = self.pad_release_controls
        if (self.object_id == controls['object_id'] and self.target_id == controls['target_id']
                and self.latch is None and source[self.index['entity.held.0']] > 0.5):
            obj = source[[self.index[f'entity.position_m.{j}'] for j in range(3)]]
            goal = source[[self.index[f'goal.target_position.{j}'] for j in range(3)]]
            if np.linalg.norm(obj - goal) <= controls['distance_m']:
                from dexvision.evaluation.place_repair import release_fingers

                prior = np.asarray(prior, float)
                self.latch = prior[:7].copy()
                self.latch[2] += self.release_raise_m
                action = prior.copy()
                action[:7] = self.latch
                action[7:] = release_fingers(prior[7:], self.spec['open_targets'],
                    self.spec['closed_targets'], self.spec['release_step_by_family']['cuboid'])
        return action, prediction
