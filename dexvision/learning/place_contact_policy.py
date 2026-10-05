"""V16 source-calibrated place wrist and table-clearance decoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from dexvision.learning.place_physical_policy import PhysicalPlacePolicy, load_physical_spec
from dexvision.learning.train_skill import load_checkpoint
from dexvision.learning.training_manifest import digest_file
from dexvision.sim.level4_expert import _orientation_preserving_hand_target

VERSION = 'level5/place-policy-contact-v16'


def load_contact_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V16 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V16 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V16 parent recipe changed')
    _, spec, inputs = load_physical_spec(parent)
    controls = overlay['contact_controls']
    if (overlay['seeds'] != spec['seeds']
            or overlay['candidate_epochs'] != {'5001': 80, '5002': 20, '5003': 20}
            or overlay['selection_seed_start'] != 1160260900
            or overlay['confirmation_seed_start'] != 1180260900
            or controls['entry_wrist_hold_targets'] != [
                ['block_large', 'inspection_pad'], ['block_large', 'setup_slot_b'],
                ['cylinder_tall', 'inspection_pad'], ['cylinder_tall', 'setup_slot_a'],
                ['cylinder_tall', 'setup_slot_b'], ['puck_light', 'inspection_pad']]
            or controls['orientation_feedback_targets'] != [
                ['block_small', '*'], ['block_large', 'setup_slot_a']]
            or controls['orientation_feedback_seeds'] != [5001, 5002]
            or controls['orientation_feedback_exclusions_by_seed'] != {
                '5001': [['block_small', 'setup_slot_a']], '5002': [], '5003': []}
            or controls['orientation_step_max_rad'] != 0.02
            or controls['large_block_floor_m'] != 0.0625
            or controls['slot_a_floor_m_by_seed'] != {
                '5001': 0.063, '5002': 0.063, '5003': 0.0625}
            or controls['light_puck_pad_floor_m'] != 0.080
            or controls['below_floor_lift_step_m'] != 0.005):
        raise ValueError('Place V16 contact control or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V16 implementation changed: {name}')
    return overlay, spec, inputs


def _quaternion_from_6d(values):
    """Convert pre-action six-dimensional orientation to wxyz."""
    first = np.asarray(values[:3], float)
    second = np.asarray(values[3:], float)
    first = first / np.linalg.norm(first)
    second = second - first * np.dot(first, second)
    second = second / np.linalg.norm(second)
    matrix = np.column_stack((first, second, np.cross(first, second)))
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def _rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    return Rotation.from_quat(np.r_[q[1:], q[0]])


class ContactPlacePolicy(PhysicalPlacePolicy):
    """Use causal object orientation and bounded hand height while still using V6 scores."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_contact_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        seed = load_checkpoint(checkpoint_path)['manifest']['seed']
        self.contact_controls = overlay['contact_controls']
        self.orientation_feedback = seed in self.contact_controls['orientation_feedback_seeds']
        self.orientation_feedback_exclusions = {
            tuple(row) for row in self.contact_controls['orientation_feedback_exclusions_by_seed'][str(seed)]}
        self.slot_a_floor = self.contact_controls['slot_a_floor_m_by_seed'][str(seed)]

    def reset(self):
        super().reset()
        self.entry_quaternion = None
        self.reference_object_orientation = None
        self.slot_a_guard = False

    def __call__(self, source, prior, phase=None):
        prior = np.asarray(prior, float)
        if self.entry_quaternion is None:
            self.entry_quaternion = prior[3:7].copy()
        if self.reference_object_orientation is None:
            self.reference_object_orientation = _quaternion_from_6d(
                source[[self.index[f'entity.rotation_6d.{j}'] for j in range(6)]])
        old_latch = self.latch
        action, prediction = super().__call__(source, prior, phase)
        key = (self.object_id, self.target_id)
        controls = self.contact_controls
        hold = {tuple(row) for row in controls['entry_wrist_hold_targets']}
        if key in hold and self.latch is None:
            action[3:7] = self.entry_quaternion
        if (key == ('block_large', 'setup_slot_a') and old_latch is None
                and action[2] < self.slot_a_floor):
            self.slot_a_guard = True
        if self.slot_a_guard and self.latch is None:
            action[3:7] = self.entry_quaternion
        feedback = (key not in self.orientation_feedback_exclusions and any(
            object_id == self.object_id and target_id in ('*', self.target_id)
            for object_id, target_id in controls['orientation_feedback_targets']))
        if self.orientation_feedback and self.latch is None and feedback:
            observed_object = _quaternion_from_6d(
                source[[self.index[f'entity.rotation_6d.{j}'] for j in range(6)]])
            forearm = _quaternion_from_6d(
                source[[self.index[f'robot.rotation6d.{j}'] for j in range(6)]])
            weld = self.spec['weld_relpose'][3:]
            measured_hand_xyzw = (_rotation(forearm) * _rotation(weld).inv()).as_quat()
            measured_hand = np.r_[measured_hand_xyzw[3], measured_hand_xyzw[:3]]
            action[3:7] = _orientation_preserving_hand_target(
                reference_object_orientation=self.reference_object_orientation,
                observed_object_orientation=observed_object,
                observed_hand_orientation=measured_hand,
                prior_requested_hand_orientation=prior[3:7],
                maximum_step_rad=controls['orientation_step_max_rad'], symmetry='none')
        floor = (controls['large_block_floor_m'] if key in {
            ('block_large', 'inspection_pad'), ('block_large', 'setup_slot_b')}
            else self.slot_a_floor if self.slot_a_guard
            else controls['light_puck_pad_floor_m']
            if key == ('puck_light', 'inspection_pad') else None)
        if floor is not None and old_latch is None:
            if prior[2] < floor:
                self.latch = None
                action = prior.copy()
                action[2] = min(floor, prior[2] + controls['below_floor_lift_step_m'])
            else:
                action[2] = max(action[2], floor)
        return action, prediction
