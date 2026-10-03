"""Causal correction direction, no overshoot, and explicit additive ownership."""
from __future__ import annotations

import numpy as np

from dexvision.learning.pick_feedback import closed_contact_counts, orientation_feedback_vector
from dexvision.learning.pick_policy import boundary_context, decode_pick, load_pick_spec
from dexvision.learning.skill_datasets import feature_schema, relative_rotation, rotation6d


def sample():
    spec, inputs = load_pick_spec('configs/level5/pick_policy_v4.yaml')
    spec['orientation_feedback'] = {'maximum_step_rad': .02001}
    spec['closed_contact_scale'] = 10
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    values = np.zeros(len(names))
    prior = np.r_[0., 0., .1, 1., 0., 0., 0., spec['closure']['closed_targets']]
    indices = [j for j, n in enumerate(names) if n.startswith('prior_applied_actions.')]
    values[indices] = prior
    values[names.index('entity.class.cuboid')] = 1.
    values[names.index('entity.held.0')] = 1.
    values[[names.index(f'entity.rotation_6d.{j}') for j in range(6)]] = rotation6d([1., 0., 0., 0.])
    context = boundary_context(values, names, spec)
    return spec, inputs, names, values, prior, context


def test_feedback_magnitude_cannot_reverse_or_overshoot_causal_error():
    spec, inputs, names, values, prior, context = sample()
    context[1:7] = rotation6d([np.cos(.002 / 2), 0., 0., np.sin(.002 / 2)])
    for gain, expected in [(-1., 0.), (0., 0.), (1., .002)]:
        output = np.zeros(26)
        output[3] = gain
        action = decode_pick(output, prior, spec, inputs.plan, features=values, names=names, context=context)
        np.testing.assert_allclose(relative_rotation(action[3:7], prior[3:7]), [0., 0., expected], atol=1e-12)


def test_cylinder_ignores_axial_yaw_and_unheld_objects_receive_no_correction():
    spec, _, names, values, _, context = sample()
    context[1:7] = rotation6d([np.cos(.1), 0., 0., np.sin(.1)])
    assert np.linalg.norm(orientation_feedback_vector(values, names, context, .02)) > 0
    values[names.index('entity.class.cuboid')] = 0.
    values[names.index('entity.class.cylinder')] = 1.
    np.testing.assert_allclose(orientation_feedback_vector(values, names, context, .02), 0., atol=1e-12)
    values[names.index('entity.class.cylinder')] = 0.
    values[names.index('entity.class.cuboid')] = 1.
    values[names.index('entity.held.0')] = 0.
    np.testing.assert_array_equal(orientation_feedback_vector(values, names, context, .02), np.zeros(3))


def test_closed_contact_history_is_causal_and_resets_when_contact_is_lost():
    spec, _, names, values, _, _ = sample()
    sequence = np.repeat(values[None], 4, axis=0)
    sequence[2, names.index('entity.held.0')] = 0
    np.testing.assert_array_equal(closed_contact_counts(sequence, names, spec), [1, 2, 0, 1])
    np.testing.assert_array_equal(closed_contact_counts(sequence[:2], names, spec), [1, 2])


def test_additive_release_ownership_preserves_original_test_and_input_documents():
    _, base = load_pick_spec('configs/level5/pick_policy_v4.yaml')
    _, amended = load_pick_spec('configs/level5/pick_policy_v5.yaml')
    assert amended.splits['test'] == base.splits['test']
    assert amended.plan == base.plan
    assert amended.release == base.release
    assert len(amended.splits['train']['episodes']) == 144
    assert len(amended.splits['validation']['episodes']) == 48
    original_ids = {r['episode_id'] for r in base.splits['train']['episodes']}
    train_ids = {r['episode_id'] for r in amended.splits['train']['episodes']}
    assert not train_ids & original_ids
    assert amended.skill['evaluation']['validation']['rollouts_per_track_per_training_seed'] == 192
    assert amended.digests['pick_additive_manifest_sha256']


def test_binary_gates_hold_at_zero_and_reproduce_exact_closure_cadence():
    spec, inputs, names, values, prior, context = sample()
    spec['binary_action_gates'] = {'threshold': .5, 'target_tolerance': 1e-10}
    prior[7:] = spec['closure']['open_targets']
    indices = [j for j, n in enumerate(names) if n.startswith('prior_applied_actions.')]
    for score in (0., .5):
        output = np.zeros(26)
        output[[2, 8]] = score
        values[indices] = prior
        np.testing.assert_allclose(decode_pick(output, prior, spec, inputs.plan,
            features=values, names=names, context=context), prior, atol=1e-12)
    output = np.zeros(26)
    output[[2, 8]] = .9
    initial_z = prior[2]
    for _ in range(20):
        values[indices] = prior
        prior = decode_pick(output, prior, spec, inputs.plan, features=values, names=names, context=context)
        assert prior[2] == initial_z, 'Lift interlock must stay closed throughout closure'
    np.testing.assert_allclose(prior[7:], spec['closure']['closed_targets'], atol=1e-12)
    values[indices] = prior
    lifted = decode_pick(output, prior, spec, inputs.plan, features=values, names=names, context=context)
    assert lifted[2] > prior[2]


def test_binary_gate_audit_rejects_continuous_source_rate():
    from types import SimpleNamespace
    import pytest
    from dexvision.learning.pick_geometry import geometry_targets
    spec, _, names, values, prior, _ = sample()
    spec.pop('orientation_feedback')
    spec['binary_action_gates'] = {'threshold': .5, 'target_tolerance': 1e-10}
    applied = prior.copy()
    applied[2] += .001
    episode = SimpleNamespace(inputs=values[None], records={'prior_applied_actions': prior[None], 'applied_actions': applied[None]})
    targets, masks = np.zeros((1,26)), np.ones((1,26))
    targets[0,2] = .5
    with pytest.raises(ValueError, match='zero-or-one'):
        geometry_targets(episode, targets, masks, spec, names)
