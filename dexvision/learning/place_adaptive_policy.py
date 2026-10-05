"""V12 causal placement geometry over the frozen V11 controller and V6 weights."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_slot_clearance_policy import (
    SlotClearancePolicy, load_slot_clearance_spec,
)
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-adaptive-v12'


def load_adaptive_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V12 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V12 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V12 parent recipe changed')
    _, spec, inputs = load_slot_clearance_spec(parent)
    controls = overlay['adaptive_controls']
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or controls != {
                'short_grasp_offset_threshold_m': 0.053,
                'cuboid_left_bin_slow_gap_m': 0.03,
                'cuboid_left_bin_slow_velocity_mps': -0.02,
                'cuboid_left_bin_slow_opening_step': 0.00625,
                'cuboid_left_bin_fast_opening_step': 0.0125,
            }):
        raise ValueError('Place V12 changed frozen geometry or confirmation boundary')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V12 implementation changed: {name}')
    return overlay, spec, inputs


class AdaptivePlacePolicy(SlotClearancePolicy):
    """Hold a short-grasp puck descent target; adapt left-bin block opening."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_adaptive_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.adaptive_controls = overlay['adaptive_controls']
        self.target_id = None
        self.fixed_descent_z = None

    def bind(self, row):
        super().bind(row)
        self.target_id = row['typed_goal']['target_id']
        self.overlay['cuboid_opening_step_nonplanar'] = self.adaptive_controls[
            'cuboid_left_bin_slow_opening_step']

    def reset(self):
        super().reset()
        self.fixed_descent_z = None

    def _waypoint(self, object_position, target_position, prior, family):
        waypoint, step = super()._waypoint(object_position, target_position, prior, family)
        if (self.object_id == 'puck_heavy' and self.target_id == 'inspection_pad'
                and self.initial_offset is not None
                and self.initial_offset[2] < self.adaptive_controls['short_grasp_offset_threshold_m']
                and self.stage == 'lower'):
            if self.fixed_descent_z is None:
                self.fixed_descent_z = float(waypoint[2])
            waypoint[2] = self.fixed_descent_z
        return waypoint, step

    def __call__(self, source, prior, phase=None):
        if (self.latch is None and self.target_id == 'return_bin_left'
                and self.object_id in {'block_large', 'block_small'}):
            object_z = source[self.index['entity.position_m.2']]
            velocity_z = source[self.index['entity.linear_velocity_mps.2']]
            controls = self.adaptive_controls
            self.overlay['cuboid_opening_step_nonplanar'] = (
                controls['cuboid_left_bin_slow_opening_step']
                if (prior[2] - object_z >= controls['cuboid_left_bin_slow_gap_m']
                    and velocity_z <= controls['cuboid_left_bin_slow_velocity_mps'])
                else controls['cuboid_left_bin_fast_opening_step'])
        return super().__call__(source, prior, phase)
