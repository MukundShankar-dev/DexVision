"""Frozen V17 place controls and disjoint qualification matrices."""
import numpy as np

from dexvision.evaluation.place_source_wrist_policy import source_wrist_matrix
from dexvision.evaluation.place_source_wrist_selection import load_selection_spec, selection_matrix
from dexvision.learning.place_source_wrist_policy import SourceWristPlacePolicy, load_source_wrist_spec


POLICY = 'configs/level5/place_policy_source_wrist_v17.yaml'
SELECTION = 'configs/level5/place_source_wrist_selection_v17.yaml'


def test_v17_selection_and_confirmation_ownership():
    overlay, spec, inputs = load_source_wrist_spec(POLICY)
    protocol, _, _ = load_selection_spec(SELECTION)
    validation = source_wrist_matrix(overlay, spec, inputs, 'validation')
    selection = selection_matrix(protocol, spec, inputs)
    confirmation = source_wrist_matrix(overlay, spec, inputs, 'confirmation')
    assert (len(validation), len(selection), len(confirmation)) == (48, 60, 60)
    for matrix in (selection, confirmation):
        assert len({row['cell_id'] for row in matrix}) == 30
    for first, second in ((validation, selection), (validation, confirmation),
                          (selection, confirmation)):
        assert {row['seed'] for row in first}.isdisjoint(row['seed'] for row in second)


def test_v17_wrist_hold_and_feedback_bind():
    policy = SourceWristPlacePolicy(
        'outputs/level5/place_repair/policy_v6/seed_5003/validation_candidates/epoch_0020.pt',
        POLICY)
    protocol, spec, inputs = load_selection_spec(SELECTION)
    rows = selection_matrix(protocol, spec, inputs)
    row = next(row for row in rows
               if row['entity_id'] == 'block_small' and row['typed_goal']['target_id'] == 'inspection_pad')
    policy.bind(row)
    assert policy.orientation_feedback
    other = next(row for row in rows
                 if row['entity_id'] == 'block_small' and row['typed_goal']['target_id'] != 'inspection_pad')
    policy.bind(other)
    assert not policy.orientation_feedback
    assert np.isclose(policy.source_wrist_controls['large_block_pad_floor_m'], 0.063)
