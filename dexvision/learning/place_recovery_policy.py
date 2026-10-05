"""V13 lateral-drift release recovery over the frozen V12 place decoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.place_adaptive_policy import AdaptivePlacePolicy, load_adaptive_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-recovery-v13'


def load_recovery_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V13 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V13 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V13 parent recipe changed')
    _, spec, inputs = load_adaptive_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or overlay['recovery_controls'] != {
                'negative_lateral_gap_threshold_m': -0.003,
                'early_release_distance_m': 0.024,
                'early_release_downward_speed_mps': -0.02,
                'early_opening_step': 0.0125,
            }
            or overlay['recovery_controls']['early_release_distance_m']
            >= spec['terminal']['release_distance_m']):
        raise ValueError('Place V13 changed frozen release or confirmation boundary')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V13 implementation changed: {name}')
    return overlay, spec, inputs


class RecoveryPlacePolicy(AdaptivePlacePolicy):
    """Open inside the physical gate before a short-grasp puck hits the table."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_recovery_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.recovery_controls = overlay['recovery_controls']

    def bind(self, row):
        super().bind(row)
        # The prior episode may have switched a short-offset puck to fast opening.
        self.spec['release_step_by_family']['flat_puck'] = 0.00625

    def __call__(self, source, prior, phase=None):
        from dexvision.evaluation.place_repair import release_fingers

        controls = self.recovery_controls
        short_pad_grasp = (self.object_id == 'puck_heavy'
            and self.target_id == 'inspection_pad' and self.initial_offset is not None
            and self.initial_offset[2] < self.adaptive_controls['short_grasp_offset_threshold_m'])
        negative_drift = bool(short_pad_grasp and prior[1]
            - source[self.index['entity.position_m.1']]
            < controls['negative_lateral_gap_threshold_m'])
        if negative_drift:
            self.spec['release_step_by_family']['flat_puck'] = controls['early_opening_step']
        action, prediction = super().__call__(source, prior, phase)
        if (self.latch is None and negative_drift
                and source[self.index['entity.held.0']] > 0.5
                and source[self.index['entity.linear_velocity_mps.2']]
                < controls['early_release_downward_speed_mps']):
            obj = source[[self.index[f'entity.position_m.{j}'] for j in range(3)]]
            goal = source[[self.index[f'goal.target_position.{j}'] for j in range(3)]]
            if np.linalg.norm(goal - obj) <= controls['early_release_distance_m']:
                self.latch = np.asarray(prior[:7], float).copy()
                self.latch[2] += self.release_raise_m
                action = np.asarray(prior, float).copy()
                action[:7] = self.latch
                action[7:] = release_fingers(prior[7:], self.spec['open_targets'],
                    self.spec['closed_targets'], controls['early_opening_step'])
        return action, prediction
