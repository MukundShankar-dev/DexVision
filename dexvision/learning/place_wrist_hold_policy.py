"""V24 source-entry wrist hold over the audited V23 place policy."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_bin_opening_policy import (
    BinOpeningPlacePolicy, load_bin_opening_spec,
)
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-wrist-hold-v24'
WRIST_HOLD_CELLS = [['block_large', 'return_bin_right']]


def load_wrist_hold_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V24 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V24 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V24 parent recipe changed')
    old, spec, inputs = load_bin_opening_spec(parent)
    expected = {
        'seeds': old['seeds'], 'candidate_epochs': old['candidate_epochs'],
        'validation_seed_start': old['validation_seed_start'],
        'selection_seed_start': 1540260900,
        'confirmation_seed_start': 1560260900,
        'selection_resets_per_cell': 2, 'confirmation_resets_per_cell': 2,
        'wrist_hold_cells': WRIST_HOLD_CELLS,
        'wrist_hold_source': 'verified_setup_command',
    }
    if any(overlay[key] != value for key, value in expected.items()):
        raise ValueError('Place V24 wrist hold or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V24 implementation changed: {name}')
    return overlay, spec, inputs


class WristHoldPlacePolicy(BinOpeningPlacePolicy):
    """Keep exact learned weights and decoder; carry setup wrist into preview."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_wrist_hold_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.wrist_hold_recipe = overlay
        self.entry_quaternion = None

    def reset(self):
        super().reset()
        self.entry_quaternion = None
