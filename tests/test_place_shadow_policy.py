"""V21 recipe ownership, fresh reset boundaries and preview scope."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.evaluation.place_shadow_action import preview_place_action
from dexvision.evaluation.place_shadow_selection import load_selection_spec, selection_matrix
from dexvision.learning.place_shadow_policy import load_shadow_spec


POLICY = 'configs/level5/place_policy_shadow_v21.yaml'
SELECTION = 'configs/level5/place_shadow_selection_v21.yaml'


def test_v21_recipe_and_fresh_all_cell_matrix():
    overlay, spec, inputs = load_shadow_spec(POLICY)
    protocol, _, _, _ = load_selection_spec(SELECTION)
    selection = selection_matrix(protocol, overlay, spec, inputs)
    assert overlay['preview_cells'] == [
        ['block_large', 'return_bin_left'], ['block_small', 'setup_slot_a']]
    assert len(selection) == 60
    assert len({row['cell_id'] for row in selection}) == 30
    assert not {row['seed'] for row in selection} & {
        overlay['validation_seed_start'] + 100 * index + offset
        for index in range(30) for offset in range(2)}
    assert {row['seed'] for row in selection}.isdisjoint({
        overlay['confirmation_seed_start'] + 100 * index + offset
        for index in range(30) for offset in range(2)})


@pytest.mark.parametrize(('object_id', 'target_id'), [
    ('cylinder_short', 'return_bin_left'),
    ('block_large', 'inspection_pad'),
])
def test_preview_cannot_change_unexposed_cells(object_id, target_id):
    action = np.zeros(27)
    action[2] = 0.08
    policy = SimpleNamespace(object_id=object_id, target_id=target_id, latch=None)
    assert preview_place_action(None, action, action, None, None, None, policy) is action


def test_preview_cannot_change_latched_release():
    action = np.zeros(27)
    action[2] = 0.08
    policy = SimpleNamespace(object_id='block_large', target_id='return_bin_left',
                             latch=np.ones(7))
    assert preview_place_action(None, action, action, None, None, None, policy) is action
