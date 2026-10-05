"""V23 bounded opening over the audited V22 place policy."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_supported_descent_policy import (
    SupportedDescentPlacePolicy, load_supported_descent_spec,
)
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-bin-opening-v23'
BIN_TARGETS = ['return_bin_left', 'return_bin_right']


def load_bin_opening_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V23 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V23 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V23 parent recipe changed')
    old, spec, inputs = load_supported_descent_spec(parent)
    expected = {
        'seeds': old['seeds'], 'candidate_epochs': old['candidate_epochs'],
        'validation_seed_start': old['validation_seed_start'],
        'selection_seed_start': 1500260900,
        'confirmation_seed_start': 1520260900,
        'selection_resets_per_cell': 2, 'confirmation_resets_per_cell': 2,
        'cylinder_object_id': 'cylinder_short',
        'cylinder_target_ids': BIN_TARGETS,
        'cylinder_descent_step_m': 0.003,
        'cylinder_min_height_above_target_m': 0.008,
        'puck_object_id': 'puck_heavy',
        'puck_target_ids': BIN_TARGETS,
        'puck_descent_step_m': 0.001,
        'puck_min_height_above_target_m': 0.005,
        'slot_b_preview_mode': 'predicted_hand_table_contact_only',
    }
    if any(overlay[key] != value for key, value in expected.items()):
        raise ValueError('Place V23 control or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V23 implementation changed: {name}')
    return overlay, spec, inputs


class BinOpeningPlacePolicy(SupportedDescentPlacePolicy):
    """Retain V19 learned weights and V21 transport; lower only held bin objects."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_bin_opening_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.bin_opening_recipe = overlay

    def __call__(self, source, prior, phase=None):
        action, prediction = super().__call__(source, prior, phase)
        controls = self.bin_opening_recipe
        if self.latch is None or source[self.index['entity.held.0']] <= 0.5:
            return action, prediction
        height = (source[self.index['entity.position_m.2']]
                  - source[self.index['goal.target_position.2']])
        if (self.object_id == controls['cylinder_object_id']
                and self.target_id in controls['cylinder_target_ids']
                and height > controls['cylinder_min_height_above_target_m']):
            step = controls['cylinder_descent_step_m']
        elif (self.object_id == controls['puck_object_id']
                and self.target_id in controls['puck_target_ids']
                and height > controls['puck_min_height_above_target_m']):
            step = controls['puck_descent_step_m']
        else:
            return action, prediction
        action[2] = max(self.spec['workspace_bounds']['min'][2], float(prior[2]) - step)
        self.latch[2] = action[2]
        return action, prediction
