"""Audit and replay saved repair confirmation states without rerunning the policy."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dexvision.evaluation.evaluate_skill import apply_action, create_reach_task
from dexvision.learning.reach_repair import calibrated_features, load_repair_spec
from dexvision.learning.skill_datasets import feature_schema, rotation6d
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run
from dexvision.learning.training_manifest import digest_file


def _saved_evidence(spec_path, run_dir, reset_dir):
    spec, inputs = load_repair_spec(spec_path)
    run_dir, reset_dir = Path(run_dir), Path(reset_dir)
    _verify_run(run_dir)
    _verify_run(reset_dir)
    attempt = json.loads((run_dir / 'fresh_confirmation_attempt.json').read_text())
    directory = Path(attempt['output_dir'])
    _verify_run(directory)
    report = json.loads((directory / 'report.json').read_text())
    if (report['checkpoint_sha256'] != digest_file(run_dir / 'selected.pt')
            or report['reset_manifest_sha256'] != digest_file(reset_dir / 'resets.json')):
        raise ValueError('Saved repair evidence binding mismatch')
    frozen = json.loads((reset_dir / 'resets.json').read_text())
    records = {r['row']['episode_id']: r for r in frozen['records']}
    return spec, inputs, directory, report, records


def measured_pose_evidence(qpos, address, goal, names, spec, recipe, success_metric):
    """Compare physical forearm-derived command-frame pose with the typed goal."""
    values = np.zeros((len(qpos), len(names)))
    values[:, :3] = qpos[:, address:address + 3]
    values[:, 3:9] = np.array([rotation6d(q) for q in qpos[:, address + 3:address + 7]])
    calibrated = calibrated_features(values, names, spec, {**recipe, 'calibrate_measured_frame': True})
    errors = np.linalg.norm(calibrated[:, :3] - np.asarray(goal[:3]), axis=1)
    relative = rotation6d(spec['weld_relpose'][3:])
    r_relative = np.column_stack((relative[:3], relative[3:], np.cross(relative[:3], relative[3:])))
    target = rotation6d(goal[3:])
    r_goal = np.column_stack((target[:3], target[3:], np.cross(target[:3], target[3:])))
    r_body = np.stack((values[:, 3:6], values[:, 6:9], np.cross(values[:, 3:6], values[:, 6:9])), axis=-1)
    r_difference = (r_body @ r_relative.T) @ r_goal.T
    angular = np.arccos(np.clip((np.trace(r_difference, axis1=1, axis2=2) - 1) / 2, -1, 1))
    thresholds = {c['field']: c['value'] for c in success_metric['conditions']}
    passing = ((errors <= thresholds['approach_distance_m']) &
               (angular <= thresholds['approach_orientation_error_rad']))
    dwell = 0
    for passed in passing[1:]:
        dwell = dwell + 1 if passed else 0
    return {'final_measured_distance_m': float(errors[-1]), 'final_measured_orientation_error_rad': float(angular[-1]),
            'final_measured_pose_dwell': dwell,
            'measured_pose_and_dwell_passed': dwell >= success_metric['required_consecutive_samples']}


def audit_repair(spec_path, run_dir, reset_dir, output_dir):
    spec, inputs, directory, report, records = _saved_evidence(spec_path, run_dir, reset_dir)
    first = next(iter(records.values()))
    with create_reach_task(inputs, first['row'], first['metadata']) as task:
        address = int(task.env.model.joint('rh_base_freejoint').qposadr[0])
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    results = []
    for result in report['results']:
        path = directory / Path(result['result_path']).parent
        _verify_run(path)
        with np.load(path / 'trajectory.npz', allow_pickle=False) as data:
            if len(data['qpos']) != len(data['applied_actions']) + 1:
                raise ValueError('Saved repair state/action alignment mismatch')
            physical = measured_pose_evidence(data['qpos'], address, result['typed_goal']['approach_pose'],
                names, spec, spec['models'][report['repair_arm']], inputs.skill['contract']['success_metric'])
        results.append({'episode_id': result['episode_id'], 'render_condition': result['render_condition'],
                        'commanded_success': result['terminal_result'] == 'succeeded', **physical})
    output = {'version': 'level5/reach-physical-pose-audit-v1', 'report_sha256': digest_file(directory / 'report.json'),
              'seed': report['seed'], 'count': len(results),
              'measured_pose_success_count': sum(r['measured_pose_and_dwell_passed'] for r in results),
              'commanded_success_without_measured_dwell': sum(r['commanded_success'] and not r['measured_pose_and_dwell_passed'] for r in results),
              'maximum_final_measured_distance_m': max(r['final_measured_distance_m'] for r in results),
              'results': results, 'physics_rerun': False, 'manual_acceptance': 'pending_owner_confirmation'}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _json(output_dir / 'audit.json', output)
    _checksum_index(output_dir)
    return output


def review_repair(spec_path, run_dir, reset_dir, *, playback_speed=.25):
    import mujoco.viewer

    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError('Playback speed must be finite and positive')
    _, inputs, directory, report, records = _saved_evidence(spec_path, run_dir, reset_dir)
    if not report['state_numerical_gates_passed']:
        raise ValueError('Manual review requires passing numerical confirmation gates')
    chosen = {}
    for result in report['results']:
        chosen.setdefault((result['object_family'], result['boundary_class']), result)
    for result in chosen.values():
        record = records[result['episode_id']]
        path = directory / Path(result['result_path']).parent / 'trajectory.npz'
        with np.load(path, allow_pickle=False) as data, create_reach_task(inputs, record['row'], record['metadata']) as task:
            print(f"Seed {report['seed']} | {result['target_id']} | {result['boundary_class']} | {result['terminal_result']}", flush=True)
            with mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                camera = task.workcell.config.scene['viewer']
                viewer.cam.lookat[:] = camera['lookat_m']
                viewer.cam.distance = camera['distance_m']
                viewer.cam.azimuth = camera['azimuth_deg']
                viewer.cam.elevation = camera['elevation_deg']
                for index, (qpos, qvel) in enumerate(zip(data['qpos'], data['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'reviewed': False, 'cancelled': True, 'manual_acceptance': 'pending'}
                    if index:
                        apply_action(task, data['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = data['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'] / playback_speed)
                time.sleep(3)
    return {'reviewed_scenarios': len(chosen), 'physics_rerun': False, 'manual_acceptance': 'pending_owner_confirmation'}
