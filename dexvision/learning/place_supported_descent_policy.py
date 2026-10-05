"""V22 short-cylinder supported opening over the physically screened V21 policy."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_shadow_policy import ShadowPlacePolicy, load_shadow_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-supported-descent-v22'
TARGETS = ['return_bin_left', 'return_bin_right']


def load_supported_descent_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V22 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V22 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V22 parent recipe changed')
    old, spec, inputs = load_shadow_spec(parent)
    # The V20 minimum supported height remains unchanged and is inherited.
    expected = {
        'seeds': old['seeds'], 'candidate_epochs': old['candidate_epochs'],
        'validation_seed_start': old['validation_seed_start'],
        'selection_seed_start': 1460260900,
        'confirmation_seed_start': 1480260900,
        'selection_resets_per_cell': 2, 'confirmation_resets_per_cell': 2,
        'supported_descent_object_id': 'cylinder_short',
        'supported_descent_target_ids': TARGETS,
        'supported_descent_step_m': 0.002,
        'supported_descent_min_object_height_above_target_m': 0.008,
    }
    if any(overlay[key] != value for key, value in expected.items()):
        raise ValueError('Place V22 control or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V22 implementation changed: {name}')
    return overlay, spec, inputs


class SupportedDescentPlacePolicy(ShadowPlacePolicy):
    """Use the exact learned weights; lower a held short cylinder 2 mm per opening."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_supported_descent_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.supported_descent_recipe = overlay

    def __call__(self, source, prior, phase=None):
        action, prediction = super().__call__(source, prior, phase)
        controls = self.supported_descent_recipe
        if (self.object_id == controls['supported_descent_object_id']
                and self.target_id in controls['supported_descent_target_ids']
                and self.latch is not None
                and source[self.index['entity.held.0']] > 0.5):
            object_z = source[self.index['entity.position_m.2']]
            target_z = source[self.index['goal.target_position.2']]
            if object_z - target_z > controls[
                    'supported_descent_min_object_height_above_target_m']:
                action[2] = max(self.spec['workspace_bounds']['min'][2],
                    float(prior[2]) - controls['supported_descent_step_m'])
                self.latch[2] = action[2]
        return action, prediction
