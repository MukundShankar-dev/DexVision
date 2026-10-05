"""V14 causal wrist guard over the frozen V13 place decoder and V6 weights."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_recovery_policy import RecoveryPlacePolicy, load_recovery_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-wrist-v14'
FROZEN_WRIST_CONTROLS = {
    'large_block_left_bin_entry_quaternion_w_min': 0.82,
    'short_cylinder_slot_b_freeze_on_initial_w_decrease': True,
    'heavy_puck_slot_a_short_grasp_offset_m': 0.053,
    'heavy_puck_slot_a_freeze_on_initial_w_decrease': True,
}


def load_wrist_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V14 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V14 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V14 parent recipe changed')
    _, spec, inputs = load_recovery_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or overlay['wrist_controls'] != FROZEN_WRIST_CONTROLS):
        raise ValueError('Place V14 changed frozen wrist or confirmation boundary')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V14 implementation changed: {name}')
    return overlay, spec, inputs


class WristPlacePolicy(RecoveryPlacePolicy):
    """Hold the verified setup wrist only when its initial state warrants it."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_wrist_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.wrist_controls = overlay['wrist_controls']

    def reset(self):
        super().reset()
        self.freeze_wrist = None

    def __call__(self, source, prior, phase=None):
        action, prediction = super().__call__(source, prior, phase)
        if self.freeze_wrist is None:
            controls = self.wrist_controls
            if self.object_id == 'block_large' and self.target_id == 'return_bin_left':
                self.freeze_wrist = bool(prior[3] > controls[
                    'large_block_left_bin_entry_quaternion_w_min'])
            elif self.object_id == 'cylinder_short' and self.target_id == 'setup_slot_b':
                self.freeze_wrist = bool(action[3] < prior[3])
            elif self.object_id == 'puck_heavy' and self.target_id == 'setup_slot_a':
                self.freeze_wrist = bool(
                    self.initial_offset[2] < controls['heavy_puck_slot_a_short_grasp_offset_m']
                    or action[3] < prior[3])
            else:
                self.freeze_wrist = False
        if self.freeze_wrist:
            action[3:7] = prior[3:7]
        return action, prediction
