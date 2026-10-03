"""Guarded viewer for saved, numerically qualified pick setup confirmations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexvision.evaluation.pick_setup import _load_resets, _verify_report, load_setup_spec
from dexvision.learning.train_skill import _verify_run
from dexvision.learning.training_manifest import digest_file


def review_inputs(config, reset_dir, seed):
    """Require all three confirmations and their saved-state audits before review."""
    protocol, spec, inputs = load_setup_spec(config)
    if seed not in protocol['seeds']:
        raise ValueError('Unknown pick review seed')
    frozen = _load_resets(protocol, inputs, reset_dir)
    root = Path(protocol['output_dir'])
    reports = {}
    for other in protocol['seeds']:
        directory = root / f'confirmation_{other}'
        _verify_run(directory)
        report = json.loads((directory / 'report.json').read_text())
        summary = _verify_report(report, protocol, inputs, frozen, 'confirmation', other, reset_dir)
        if not summary['state_numerical_gates_passed']:
            raise ValueError('Every pick seed must pass confirmation before manual acceptance')
        audit_dir = root / f'audit_{other}'
        _verify_run(audit_dir)
        audit = json.loads((audit_dir / 'audit.json').read_text())
        if (not audit['passed'] or not all(a['passed'] for a in audit['audits'])
                or audit['attempts'] != summary['rollout_count']
                or len(audit['audits']) != summary['rollout_count']
                or audit['report_sha256'] != digest_file(directory / 'report.json')
                or audit['checkpoint_sha256'] != report['checkpoint_sha256']):
            raise ValueError('Manual review requires matching passing physical audits for every seed')
        reports[other] = report
    return spec, inputs, frozen, reports[seed], root / f'confirmation_{seed}'


def review_setup(config, reset_dir, seed):
    """Show instance/boundary successes and each failure reason; never rerun a policy."""
    _, inputs, frozen, report, directory = review_inputs(config, reset_dir, seed)
    import time
    import mujoco.viewer
    from dexvision.evaluation.evaluate_skill import apply_action
    from dexvision.evaluation.pick_policy import create_pick_task

    records = {r['row']['episode_id']: r for r in frozen['confirmation']}
    chosen = {}
    for result in report['results']:
        key = ((result['object_instance'], result['boundary_class'], 'succeeded')
               if result['terminal_result'] == 'succeeded' else ('failure', result['terminal_reason']))
        chosen.setdefault(key, result)
    for result in chosen.values():
        record = records[result['episode_id']]
        child = directory / Path(result['result_path']).parent
        with create_pick_task(inputs, record['row'], record['metadata'], np.asarray(record['prefix_actions'])) as task:
            print(f"Seed {seed}: {result['object_instance']} ({result['object_family']}) {result['boundary_class']} "
                  f"{result['terminal_reason']} ({result['episode_id']})", flush=True)
            with np.load(child / 'trajectory.npz', allow_pickle=False) as values, mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                viewer.cam.lookat[:] = [0.04, 0., 0.08]
                viewer.cam.distance = 0.65
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                for index, (qpos, qvel) in enumerate(zip(values['qpos'], values['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'manual_acceptance': 'pending_owner_confirmation', 'cancelled': True}
                    if index:
                        apply_action(task, values['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = values['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'])
    return {'manual_acceptance': 'pending_owner_confirmation', 'reviewed': len(chosen)}
