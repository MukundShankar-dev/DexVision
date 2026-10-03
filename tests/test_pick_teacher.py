"""Unassisted teacher physics and additive-attempt protections."""
from __future__ import annotations

import pytest

from dexvision.evaluation.pick_teacher import load_teacher_spec, teacher_attempt
from dexvision.sim.workcell import Workcell


def test_unassisted_teacher_never_overwrites_object_orientation(tmp_path, monkeypatch):
    recipe, spec, inputs = load_teacher_spec('configs/level5/pick_teacher_v1.yaml')

    def forbidden(*args, **kwargs):
        raise AssertionError('Neither scratch preflight nor live teacher may overwrite object orientation')

    monkeypatch.setattr(Workcell, 'preserve_object_orientation', forbidden)
    result = teacher_attempt(recipe, spec, inputs, recipe['cells'][0], recipe['seed_start'],
        False, tmp_path / 'attempt')
    assert result['success']
    assert not any(result['safety_counters'].values())
    assert result['entry_frame'] > 0
    assert result['selected_config']['orientation_preservation_policy'] == 'ordinary_physics_no_object_pose_writes'
    assert (tmp_path / 'attempt/trajectory.npz').exists()
    with pytest.raises(FileExistsError):
        teacher_attempt(recipe, spec, inputs, recipe['cells'][0], recipe['seed_start'],
            False, tmp_path / 'attempt')


def test_exact_preflight_rejects_dropping_grasp_before_live_use(tmp_path, monkeypatch):
    import numpy as np
    from dexvision.evaluation.pick_teacher import unassisted_pick_prefix
    from dexvision.logging.level4_collection import WorkcellPilotTask
    from dexvision.sim.level4_expert import DeterministicGraspLiftExpert
    recipe, spec, inputs = load_teacher_spec('configs/level5/pick_teacher_v2.yaml')

    def forbidden(*args, **kwargs):
        raise AssertionError('Exact preflight and live execution must use ordinary physics')

    monkeypatch.setattr(Workcell, 'preserve_object_orientation', forbidden)
    result = teacher_attempt(recipe, spec, inputs, recipe['cells'][-1], recipe['seed_start'] + 500,
        False, tmp_path / 'heavy')
    assert result['success']
    assert result['preflight_candidates'][0]['reason'] == 'slip_drop'
    selected = result['preflight_candidates'][-1]
    assert selected['success'] and selected['action_count'] == result['action_count']
    assert selected['metrics'] == result['metrics']
    assert selected['entry_frame'] == result['entry_frame']
    assert not any(result['safety_counters'].values())
    with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=result['cell'], seed=result['seed'],
            procedural_variation=result['variation']) as task:
        original_step = DeterministicGraspLiftExpert.step

        def checked_step(self, state):
            value = original_step(self, state)
            if self._task.workcell is task.workcell:
                assert value[1] == 'approach', 'Live setup must not even query an acquire command'
            return value

        monkeypatch.setattr(DeterministicGraspLiftExpert, 'step', checked_step)
        prefix, audit = unassisted_pick_prefix(task, recipe, spec, inputs)
        assert audit['live_acquire_actions_queried'] == 0
        with np.load(tmp_path / 'heavy/trajectory.npz', allow_pickle=False) as recorded:
            np.testing.assert_array_equal(prefix, recorded['actions'][:result['entry_frame']])
            np.testing.assert_array_equal(task.env.data.qpos, recorded['qpos'][result['entry_frame']])


def test_terminal_evidence_matches_saved_state_without_changing_live_data(tmp_path):
    """Regression: the old tenth cached sample had post-step speed above 0.02 m/s."""
    import numpy as np
    from dexvision.evaluation.pick_policy import physical_evidence_world
    from dexvision.evaluation.pick_repair import physical_pick_audit
    from dexvision.logging.level4_collection import WorkcellPilotTask
    from dexvision.evaluation.evaluate_skill import apply_action

    for version, expected in ((3, False), (4, True)):
        recipe, spec, inputs = load_teacher_spec(f'configs/level5/pick_teacher_v{version}.yaml')
        directory = tmp_path / f'v{version}'
        result = teacher_attempt(recipe, spec, inputs, 'pp_puck_light_inspection_pad',
                                 750262201, True, directory)
        assert result['success']
        with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
                skill_name='pick_place_sequence', goal_condition_id=result['cell'], seed=result['seed'],
                procedural_variation=result['variation']) as task, np.load(directory / 'trajectory.npz') as saved:
            start = result['entry_frame']
            for action in saved['actions'][:start]:
                apply_action(task, action, inputs.plan['action']['named_layout'])
                task.workcell.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
            live = task.env.data
            before = {k: getattr(live, k).copy() for k in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'mocap_pos', 'mocap_quat')}
            physical_evidence_world(task)
            assert task.env.data is live
            for key, value in before.items():
                np.testing.assert_array_equal(getattr(live, key), value)
            audit = physical_pick_audit(task, {'qpos': saved['qpos'][start:], 'qvel': saved['qvel'][start:],
                'applied_actions': saved['actions'][start:], 'state_timestamps': saved['timestamps'][start:]},
                {'object_instance': result['object_id'], 'terminal_result': 'succeeded'}, spec)
            assert audit['passed'] is expected
            assert audit['physical_success'] is expected
