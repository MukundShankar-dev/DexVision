"""V19 all-cell source ownership and fresh qualification boundaries."""
from dexvision.evaluation.place_full_source_policy import full_source_matrix
from dexvision.evaluation.place_full_source_selection import load_selection_spec, selection_matrix
from dexvision.learning.place_full_source_policy import full_source_datasets, load_full_source_spec


POLICY = 'configs/level5/place_policy_full_source_v19.yaml'
SELECTION = 'configs/level5/place_full_source_selection_v19.yaml'


def test_v19_audited_all_cell_source_and_disjoint_resets():
    spec, inputs = load_full_source_spec(POLICY)
    datasets = full_source_datasets(spec, inputs)
    assert (len(datasets['train'].episodes), len(datasets['validation'].episodes)) == (237, 117)
    assert {e.cell_id for e in datasets['train'].episodes} == {
        e.cell_id for e in datasets['validation'].episodes}
    assert len({e.cell_id for e in datasets['train'].episodes}) == 30
    assert {e.episode_id for e in datasets['train'].episodes}.isdisjoint(
        e.episode_id for e in datasets['validation'].episodes)
    protocol, _, _ = load_selection_spec(SELECTION)
    validation = full_source_matrix(spec, spec, inputs, 'validation')
    selection = selection_matrix(protocol, spec, inputs)
    confirmation = full_source_matrix(spec, spec, inputs, 'confirmation')
    assert (len(validation), len(selection), len(confirmation)) == (60, 60, 60)
    source_seeds = {int(episode.episode_id.rsplit('_', 1)[-1])
                    for dataset in datasets.values() for episode in dataset.episodes}
    for matrix in (validation, selection, confirmation):
        assert source_seeds.isdisjoint(row['seed'] for row in matrix)
    for first, second in ((validation, selection), (validation, confirmation),
                          (selection, confirmation)):
        assert {row['seed'] for row in first}.isdisjoint(row['seed'] for row in second)
