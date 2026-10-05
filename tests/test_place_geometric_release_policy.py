"""V18 physical release bound and fresh qualification ownership."""
from dexvision.evaluation.place_geometric_release_policy import geometric_release_matrix
from dexvision.evaluation.place_geometric_release_selection import load_selection_spec, selection_matrix
from dexvision.learning.place_geometric_release_policy import load_geometric_release_spec


POLICY = 'configs/level5/place_policy_geometric_release_v18.yaml'
SELECTION = 'configs/level5/place_geometric_release_selection_v18.yaml'


def test_v18_frozen_release_bound_and_disjoint_matrices():
    overlay, spec, inputs = load_geometric_release_spec(POLICY)
    protocol, _, _ = load_selection_spec(SELECTION)
    assert overlay['geometric_release_controls']['distance_m'] == spec[
        'geometry']['release_distance_by_family_m']['cuboid']
    validation = geometric_release_matrix(overlay, spec, inputs, 'validation')
    selection = selection_matrix(protocol, spec, inputs)
    confirmation = geometric_release_matrix(overlay, spec, inputs, 'confirmation')
    assert (len(validation), len(selection), len(confirmation)) == (48, 60, 60)
    for matrix in (selection, confirmation):
        assert len({row['cell_id'] for row in matrix}) == 30
    for first, second in ((validation, selection), (validation, confirmation),
                          (selection, confirmation)):
        assert {row['seed'] for row in first}.isdisjoint(row['seed'] for row in second)
