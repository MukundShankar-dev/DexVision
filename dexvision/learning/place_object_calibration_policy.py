"""V20 place decoder calibrated only from audited V3 training trajectories."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.place_full_source_policy import (
    FullSourcePlacePolicy, _released_payloads, load_full_source_spec,
)
from dexvision.learning.skill_datasets import feature_schema
from dexvision.learning.training_manifest import digest_file

VERSION = 'level5/place-policy-object-calibration-v20'
OBJECTS = ('block_large', 'block_small', 'cylinder_short', 'cylinder_tall',
           'puck_heavy', 'puck_light')


def training_release_errors(spec, inputs):
    """Median target-minus-object vector at first opening, V3 train only (m)."""
    manifest, payloads = _released_payloads(spec)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    index = {name: i for i, name in enumerate(names)}
    values = {name: [] for name in OBJECTS}
    for row in manifest['reports']['train']['results']:
        if not row['payload']:
            continue
        name = row['object_id']
        if name not in values:
            raise ValueError(f'Unknown V3 place training object: {name}')
        prefix = f"datasets/{manifest['plan']['release_name']}/episodes/{row['episode_id']}/"
        with np.load(io.BytesIO(payloads[prefix + 'trajectory.npz']), allow_pickle=False) as data:
            indices = np.flatnonzero(data['expert_phases'] == 'latched_release')
            if len(indices) == 0:
                raise ValueError(f'Missing V3 release label: {row["episode_id"]}')
            observed = data['observations'][indices[0]]
        object_xyz = observed[[index[f'entity.position_m.{j}'] for j in range(3)]]
        target_xyz = observed[[index[f'goal.target_position.{j}'] for j in range(3)]]
        values[name].append(target_xyz - object_xyz)
    if any(len(rows) < 30 for rows in values.values()):
        raise ValueError('Insufficient successful V3 training release coverage')
    return {name: np.median(np.asarray(rows), axis=0).tolist()
            for name, rows in values.items()}


def load_object_calibration_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V20 recipe checksum mismatch')
    overlay = yaml.safe_load(path.read_text())
    if overlay.get('version') != VERSION or overlay.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V20 recipe')
    parent = Path(overlay['parent_policy_recipe'])
    if digest_file(parent) != overlay['parent_policy_recipe_sha256']:
        raise ValueError('Place V20 parent recipe changed')
    spec, inputs = load_full_source_spec(parent)
    if (overlay['seeds'] != spec['seeds']
            or overlay['candidate_epochs'] != {'5001': 20, '5002': 20, '5003': 200}
            or overlay['validation_seed_start'] != spec['validation_seed_start']
            or overlay['selection_seed_start'] != 1380260900
            or overlay['confirmation_seed_start'] != 1400260900
            or overlay['selection_resets_per_cell'] != 2
            or overlay['confirmation_resets_per_cell'] != 2
            or overlay['cylinder_opening_descent_step_m'] != 0.001
            or overlay['cylinder_opening_min_object_height_above_target_m'] != 0.008
            or overlay['small_block_slot_b_hand_floor_m'] != 0.073
            or set(overlay['training_release_error_by_object_m']) != set(OBJECTS)):
        raise ValueError('Place V20 control or fresh reset boundary changed')
    for name, checksum in overlay['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V20 implementation changed: {name}')
    actual = training_release_errors(spec, inputs)
    for name in OBJECTS:
        expected = np.asarray(overlay['training_release_error_by_object_m'][name], float)
        if expected.shape != (3,) or not np.isfinite(expected).all() or not np.allclose(
                actual[name], expected, rtol=0, atol=1e-12):
            raise ValueError(f'Place V20 training-only calibration mismatch: {name}')
    return overlay, spec, inputs


class ObjectCalibrationPlacePolicy(FullSourcePlacePolicy):
    """Keep V19 learned scores and use V3 object release geometry and safe opening."""

    def __init__(self, checkpoint_path, config):
        overlay, _, _ = load_object_calibration_spec(config)
        super().__init__(checkpoint_path, overlay['parent_policy_recipe'])
        self.object_calibration = overlay

    def _goal_error(self, family):
        return np.asarray(self.object_calibration[
            'training_release_error_by_object_m'][self.object_id], float)

    def __call__(self, source, prior, phase=None):
        action, prediction = super().__call__(source, prior, phase)
        controls = self.object_calibration
        if self.object_id == 'block_small' and self.target_id == 'setup_slot_b':
            floor = controls['small_block_slot_b_hand_floor_m']
            if self.latch is not None and action[2] < floor:
                self.latch = None
                action = np.asarray(prior, float).copy()
                action[2] = floor
            elif self.latch is None:
                action[2] = max(action[2], floor)
        if (self.object_id.startswith('cylinder') and self.latch is not None
                and source[self.index['entity.held.0']] > 0.5):
            object_z = source[self.index['entity.position_m.2']]
            target_z = source[self.index['goal.target_position.2']]
            if object_z - target_z > controls[
                    'cylinder_opening_min_object_height_above_target_m']:
                action[2] = max(self.spec['workspace_bounds']['min'][2],
                    prior[2] - controls['cylinder_opening_descent_step_m'])
                self.latch[2] = action[2]
        return action, prediction
