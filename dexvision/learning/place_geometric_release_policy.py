"""V18 bounded geometric release fallback for the V17 large-block miss."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.place_source_wrist_policy import SourceWristPlacePolicy, load_source_wrist_spec
from dexvision.learning.train_skill import load_checkpoint
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-geometric-release-v18'
FROZEN_CONTROLS = {
    'seed': 5003,
    'object_id': 'block_large',
    'target_ids': ['inspection_pad', 'setup_slot_a', 'setup_slot_b'],
    'distance_m': 0.0245,
}


def load_geometric_release_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V18 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V18 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V18 parent recipe changed')
    _, spec, inputs = load_source_wrist_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['candidate_epochs'] != {'5001': 80, '5002': 20, '5003': 20}
            or overlay['selection_seed_start'] != 1300260900
            or overlay['confirmation_seed_start'] != 1320260900
            or overlay['geometric_release_controls'] != FROZEN_CONTROLS
            or FROZEN_CONTROLS['distance_m'] != spec['geometry'][
                'release_distance_by_family_m']['cuboid']):
        raise ValueError('Place V18 controls or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V18 implementation changed: {name}')
    return overlay, spec, inputs


class GeometricReleasePlacePolicy(SourceWristPlacePolicy):
    """Open a stalled large block only inside the original release distance."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_geometric_release_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.seed = load_checkpoint(checkpoint_path)['manifest']['seed']
        self.geometric_release_controls = overlay['geometric_release_controls']

    def __call__(self, source, prior, phase=None):
        action, prediction = super().__call__(source, prior, phase)
        controls = self.geometric_release_controls
        if (self.seed == controls['seed'] and self.object_id == controls['object_id']
                and self.target_id in controls['target_ids'] and self.latch is None
                and source[self.index['entity.held.0']] > 0.5):
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
