"""V21 place recipe binding the V20 learned policy to a copied-state preview."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_object_calibration_policy import (
    ObjectCalibrationPlacePolicy, load_object_calibration_spec,
)
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-shadow-v21'
CELLS = [['block_large', 'return_bin_left'], ['block_small', 'setup_slot_a']]


def load_shadow_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V21 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V21 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V21 parent recipe changed')
    old, spec, inputs = load_object_calibration_spec(parent)
    if (overlay['seeds'] != old['seeds']
            or overlay['candidate_epochs'] != old['candidate_epochs']
            or overlay['validation_seed_start'] != old['validation_seed_start']
            or overlay['selection_seed_start'] != 1420260900
            or overlay['confirmation_seed_start'] != 1440260900
            or overlay['selection_resets_per_cell'] != 2
            or overlay['confirmation_resets_per_cell'] != 2
            or overlay['preview_cells'] != CELLS
            or overlay['preview_max_hand_z_m'] != 0.135
            or overlay['preview_rotation_degrees'] != [-3., 3.]
            or overlay['preview_rise_m'] != [0.002, 0.005]
            or overlay['preview_lateral_m'] != [-0.002, 0.002]
            or overlay['preview_progress_margin_m'] != 0.001):
        raise ValueError('Place V21 preview controls or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V21 implementation changed: {name}')
    return overlay, spec, inputs


class ShadowPlacePolicy(ObjectCalibrationPlacePolicy):
    """Exact V19 weights and V20 decoder, with V21 runtime preview bound by recipe."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_shadow_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.shadow_recipe = overlay
