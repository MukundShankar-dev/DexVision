"""Frozen V16 place contact controls and fresh qualification ownership."""
import json

import numpy as np
import pytest

from dexvision.evaluation.place_contact_policy import contact_matrix
from dexvision.evaluation.place_contact_selection import _audit_attempt, load_selection_spec, selection_matrix
from dexvision.evaluation.place_repair_audit import audit_repair_rollout
from dexvision.learning.place_contact_policy import _quaternion_from_6d, load_contact_spec
from dexvision.learning.train_skill import _checksum_index


def test_v16_selection_and_confirmation_are_disjoint_from_source():
    overlay, spec, inputs = load_contact_spec('configs/level5/place_policy_contact_v16.yaml')
    protocol, _, _ = load_selection_spec('configs/level5/place_contact_selection_v16.yaml')
    validation = contact_matrix(overlay, spec, inputs, 'validation')
    selection = selection_matrix(protocol, spec, inputs)
    confirmation = contact_matrix(overlay, spec, inputs, 'confirmation')
    assert (len(validation), len(selection), len(confirmation)) == (48, 60, 60)
    assert len({row['cell_id'] for row in selection}) == 30
    assert len({row['cell_id'] for row in confirmation}) == 30
    for first, second in ((validation, selection), (validation, confirmation),
                          (selection, confirmation)):
        assert {row['seed'] for row in first}.isdisjoint(row['seed'] for row in second)


def test_v16_orientation_decoder_uses_valid_pre_action_rotation():
    np.testing.assert_allclose(_quaternion_from_6d([1, 0, 0, 0, 1, 0]),
                               [1, 0, 0, 0], atol=1e-12)


def test_v16_selection_retains_failed_setup_without_place_audit(tmp_path):
    child = tmp_path / 'failed_setup'
    child.mkdir()
    np.savez_compressed(child / 'trajectory.npz', qpos=np.zeros((1, 2)),
                        qvel=np.zeros((1, 2)), state_timestamps=np.array([0.]))
    row = {'episode_id': 'failed_setup', 'render_condition': 'nominal'}
    result = {'terminal_result': 'rejected_before_execution',
              'terminal_reason': 'pick_setup_failed',
              'safety_counters': {'invalid_actions': 0}}
    audit = _audit_attempt(None, None, row, result, child)
    assert audit['passed'] and audit['excluded_before_place']
    with pytest.raises(ValueError, match='Only rejected setup'):
        _audit_attempt(None, None, row, {**result, 'terminal_result': 'succeeded'}, child)


def test_v16_confirmation_audit_retains_rejected_setup(tmp_path):
    child = tmp_path / 'rejected_setup'
    child.mkdir()
    np.savez_compressed(child / 'trajectory.npz', qpos=np.zeros((1, 2)),
                        qvel=np.zeros((1, 2)), state_timestamps=np.array([0.]))
    result = {'terminal_result': 'rejected_before_execution',
              'terminal_reason': 'pick_setup_failed',
              'safety_counters': {'invalid_actions': 0}}
    (child / 'result.json').write_text(json.dumps(result))
    _checksum_index(child)
    row = {'episode_id': 'rejected_setup', 'render_condition': 'nominal'}
    audit = audit_repair_rollout(None, None, row, result, child)
    assert audit['passed'] and audit['excluded_before_place']
    assert not audit['reported_success']
