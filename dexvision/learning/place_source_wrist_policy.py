"""V17 source-supported wrist and pad-clearance decoder over V16 place control."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.place_contact_policy import ContactPlacePolicy, load_contact_spec
from dexvision.learning.train_skill import load_checkpoint
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-source-wrist-v17'
FROZEN_CONTROLS = {
    'hold_entry_wrist_families': ['cylinder', 'flat_puck'],
    'large_block_pad_floor_m': 0.063,
    'below_floor_lift_step_m': 0.005,
    'seed5003_orientation_feedback_targets': [
        ['block_large', 'setup_slot_a'], ['block_small', 'inspection_pad']],
}


def load_source_wrist_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V17 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V17 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V17 parent recipe changed')
    _, spec, inputs = load_contact_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['candidate_epochs'] != {'5001': 80, '5002': 20, '5003': 20}
            or overlay['selection_seed_start'] != 1260260900
            or overlay['confirmation_seed_start'] != 1280260900
            or overlay['source_wrist_controls'] != FROZEN_CONTROLS):
        raise ValueError('Place V17 controls or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V17 implementation changed: {name}')
    return overlay, spec, inputs


class SourceWristPlacePolicy(ContactPlacePolicy):
    """Keep the source setup wrist on round objects and a bounded pad approach."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_source_wrist_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.seed = load_checkpoint(checkpoint_path)['manifest']['seed']
        self.source_wrist_controls = overlay['source_wrist_controls']

    def bind(self, row):
        super().bind(row)
        if self.seed == 5003:
            targets = {tuple(item) for item in self.source_wrist_controls[
                'seed5003_orientation_feedback_targets']}
            self.orientation_feedback = (self.object_id, row['typed_goal']['target_id']) in targets

    def __call__(self, source, prior, phase=None):
        old_latch = self.latch
        action, prediction = super().__call__(source, prior, phase)
        family = ('cylinder' if self.object_id and self.object_id.startswith('cylinder')
                  else 'flat_puck' if self.object_id and self.object_id.startswith('puck')
                  else None)
        if self.latch is None and family in self.source_wrist_controls['hold_entry_wrist_families']:
            action[3:7] = self.entry_quaternion
        if (self.object_id == 'block_large' and self.target_id == 'inspection_pad'
                and old_latch is None):
            prior = np.asarray(prior, float)
            floor = self.source_wrist_controls['large_block_pad_floor_m']
            if prior[2] < floor:
                self.latch = None
                action = prior.copy()
                action[2] = min(floor, prior[2] + self.source_wrist_controls[
                    'below_floor_lift_step_m'])
            else:
                action[2] = max(action[2], floor)
                if self.latch is not None:
                    self.latch[2] = max(self.latch[2], floor)
        return action, prediction
