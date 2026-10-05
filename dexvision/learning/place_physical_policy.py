"""V15 measured-state opening and collision recovery over V14 place control."""
from __future__ import annotations

from pathlib import Path

import yaml

from dexvision.learning.place_wrist_policy import WristPlacePolicy, load_wrist_spec
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-physical-v15'
FROZEN_PHYSICAL_CONTROLS = {
    'slow_opening_step': 0.00625,
    'fast_opening_step': 0.0125,
    'cylinder_left_w_max': 0.73,
    'cylinder_left_downward_speed_min_mps': -0.027,
    'cylinder_right_initial_offset_min_m': 0.040,
    'puck_initial_offset_split_m': 0.053,
    'puck_left_w_max': 0.735,
    'puck_left_downward_speed_max_mps': -0.01,
    'puck_right_downward_speed_max_mps': -0.022,
    'puck_slot_a_downward_speed_min_mps': -0.02,
    'block_left_collision_y_min_m': 0.105,
    'block_left_rising_speed_min_mps': 0.01,
    'block_left_inboard_step_m': 0.002,
    'block_left_lift_step_m': 0.003,
}


def load_physical_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V15 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay['version'] != VERSION or overlay['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place V15 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V15 parent recipe changed')
    _, spec, inputs = load_wrist_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['confirmation_seed_start'] != spec['confirmation']['seed_start']
            or overlay['physical_controls'] != FROZEN_PHYSICAL_CONTROLS):
        raise ValueError('Place V15 changed frozen physical or confirmation boundary')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V15 implementation changed: {name}')
    return overlay, spec, inputs


class PhysicalPlacePolicy(WristPlacePolicy):
    """Select bounded opening from live geometry and retreat from a bin collision."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_physical_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.physical_controls = overlay['physical_controls']

    def __call__(self, source, prior, phase=None):
        controls = self.physical_controls
        index = self.index
        vz = source[index['entity.linear_velocity_mps.2']]
        offset = (self.initial_offset[2] if self.initial_offset is not None
                  else prior[2] - source[index['entity.position_m.2']])
        if self.object_id == 'cylinder_short':
            fast = (self.target_id == 'setup_slot_a'
                    or (self.target_id == 'return_bin_right'
                        and offset >= controls['cylinder_right_initial_offset_min_m'])
                    or (self.target_id == 'return_bin_left'
                        and prior[3] < controls['cylinder_left_w_max']
                        and vz > controls['cylinder_left_downward_speed_min_mps']))
            self.spec['release_step_by_family']['cylinder'] = (
                controls['fast_opening_step'] if fast else controls['slow_opening_step'])
        elif self.object_id == 'puck_heavy':
            split = controls['puck_initial_offset_split_m']
            fast = ((self.target_id == 'inspection_pad' and offset >= split)
                    or (self.target_id == 'return_bin_left'
                        and prior[3] < controls['puck_left_w_max']
                        and vz < controls['puck_left_downward_speed_max_mps'])
                    or (self.target_id == 'return_bin_right'
                        and (offset < split or vz < controls['puck_right_downward_speed_max_mps']))
                    or (self.target_id == 'setup_slot_a' and offset >= split
                        and vz > controls['puck_slot_a_downward_speed_min_mps'])
                    or (self.target_id == 'setup_slot_b' and offset < split))
            self.spec['release_step_by_family']['flat_puck'] = (
                controls['fast_opening_step'] if fast else controls['slow_opening_step'])
        action, prediction = super().__call__(source, prior, phase)
        if (self.object_id == 'block_large' and self.target_id == 'return_bin_left'
                and self.latch is None):
            y = source[index['entity.position_m.1']]
            held = source[index['entity.held.0']]
            if (held > 0.5 and y > controls['block_left_collision_y_min_m']
                    and vz > controls['block_left_rising_speed_min_mps']):
                action[0] = prior[0]
                action[1] = prior[1] - controls['block_left_inboard_step_m']
                action[2] = prior[2] + controls['block_left_lift_step_m']
                action[3:7] = prior[3:7]
        return action, prediction
