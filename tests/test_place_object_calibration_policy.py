"""V20 training-only calibration and fresh place reset ownership."""
import json
from pathlib import Path

import numpy as np

from dexvision.evaluation.place_object_calibration_policy import object_calibration_matrix
from dexvision.evaluation.place_object_calibration_selection import (
    load_selection_spec, selection_matrix,
)
from dexvision.learning.place_object_calibration_policy import (
    load_object_calibration_spec, training_release_errors,
)


POLICY = 'configs/level5/place_policy_object_calibration_v20.yaml'
SELECTION = 'configs/level5/place_object_calibration_selection_v20.yaml'


def test_v20_offsets_come_only_from_audited_training_and_resets_are_disjoint():
    overlay, spec, inputs = load_object_calibration_spec(POLICY)
    actual = training_release_errors(spec, inputs)
    assert set(actual) == set(overlay['training_release_error_by_object_m'])
    for name, values in actual.items():
        assert np.allclose(values, overlay['training_release_error_by_object_m'][name],
                           rtol=0, atol=1e-12)
    assert actual['cylinder_short'][0] > 0 > actual['cylinder_tall'][0]
    assert actual['block_small'][0] > actual['block_large'][0]

    protocol, _, _, _ = load_selection_spec(SELECTION)
    validation = object_calibration_matrix(overlay, spec, inputs, 'validation')
    selection = selection_matrix(protocol, overlay, spec, inputs)
    confirmation = object_calibration_matrix(overlay, spec, inputs, 'confirmation')
    assert (len(validation), len(selection), len(confirmation)) == (60, 60, 60)
    assert all(len({row['cell_id'] for row in matrix}) == 30
               for matrix in (validation, selection, confirmation))
    manifest = json.loads(Path(spec['additive_data']['manifest']).read_text())
    source_seeds = {row['seed'] for split in ('train', 'validation')
                    for row in manifest['reports'][split]['results']}
    for matrix in (validation, selection, confirmation):
        assert source_seeds.isdisjoint(row['seed'] for row in matrix)
    for first, second in ((validation, selection), (validation, confirmation),
                          (selection, confirmation)):
        assert {row['seed'] for row in first}.isdisjoint(row['seed'] for row in second)
