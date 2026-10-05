"""Guarded replay of saved, qualified place confirmation states."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

from dexvision.evaluation.evaluate_skill import apply_action
from dexvision.evaluation.place_repair_policy import repair_matrix
from dexvision.learning.place_cadence_policy import load_cadence_spec
from dexvision.learning.place_guarded_policy import load_guarded_spec
from dexvision.learning.place_safety_policy import load_safety_spec
from dexvision.learning.place_slow_policy import load_slow_spec
from dexvision.learning.place_clearance_policy import load_clearance_spec
from dexvision.learning.place_slot_clearance_policy import load_slot_clearance_spec
from dexvision.learning.place_adaptive_policy import load_adaptive_spec
from dexvision.learning.place_recovery_policy import load_recovery_spec
from dexvision.learning.place_wrist_policy import load_wrist_spec
from dexvision.learning.place_physical_policy import load_physical_spec
from dexvision.learning.place_contact_policy import load_contact_spec
from dexvision.learning.train_skill import _verify_run
from dexvision.learning.training_manifest import digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask


def review_inputs(config: Path, seed: int, *, selection_config: Path | None = None):
    """Require matching passing confirmations and physical audits for all seeds."""
    version = yaml.safe_load(Path(config).read_text())['version']
    if version == 'level5/place-policy-pad-release-v25':
        from dexvision.learning.place_pad_release_policy import load_pad_release_spec
        overlay, spec, inputs = load_pad_release_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-wrist-hold-v24':
        from dexvision.learning.place_wrist_hold_policy import load_wrist_hold_spec
        overlay, spec, inputs = load_wrist_hold_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-bin-opening-v23':
        from dexvision.learning.place_bin_opening_policy import load_bin_opening_spec
        overlay, spec, inputs = load_bin_opening_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-supported-descent-v22':
        from dexvision.learning.place_supported_descent_policy import load_supported_descent_spec
        overlay, spec, inputs = load_supported_descent_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-shadow-v21':
        from dexvision.learning.place_shadow_policy import load_shadow_spec
        overlay, spec, inputs = load_shadow_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-object-calibration-v20':
        from dexvision.learning.place_object_calibration_policy import load_object_calibration_spec
        overlay, spec, inputs = load_object_calibration_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-full-source-v19':
        from dexvision.learning.place_full_source_policy import load_full_source_spec
        spec, inputs = load_full_source_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = spec['confirmation_seed_start']
    elif version == 'level5/place-policy-geometric-release-v18':
        from dexvision.learning.place_geometric_release_policy import load_geometric_release_spec
        overlay, spec, inputs = load_geometric_release_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-source-wrist-v17':
        from dexvision.learning.place_source_wrist_policy import load_source_wrist_spec
        overlay, spec, inputs = load_source_wrist_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-contact-v16':
        overlay, spec, inputs = load_contact_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-physical-v15':
        _, spec, inputs = load_physical_spec(config)
    elif version == 'level5/place-policy-wrist-v14':
        _, spec, inputs = load_wrist_spec(config)
    elif version == 'level5/place-policy-recovery-v13':
        _, spec, inputs = load_recovery_spec(config)
    elif version == 'level5/place-policy-adaptive-v12':
        _, spec, inputs = load_adaptive_spec(config)
    elif version == 'level5/place-policy-slot-clearance-v11':
        _, spec, inputs = load_slot_clearance_spec(config)
    elif version == 'level5/place-policy-clearance-v10':
        _, spec, inputs = load_clearance_spec(config)
    elif version == 'level5/place-policy-slow-v9':
        _, spec, inputs = load_slow_spec(config)
    elif version == 'level5/place-policy-safety-v8':
        _, spec, inputs = load_safety_spec(config)
    elif version == 'level5/place-policy-guarded-v6':
        spec, inputs = load_guarded_spec(config)
    elif version == 'level5/place-policy-cadence-v5':
        spec, inputs = load_cadence_spec(config)
    else:
        raise ValueError('Unsupported place repair review recipe')
    if seed not in spec['seeds']:
        raise ValueError('Unknown place repair seed')
    if selection_config is None:
        root = Path(spec['output_dir'])
    else:
        selection_version = yaml.safe_load(Path(selection_config).read_text())['version']
        if selection_version == 'level5/place-pad-release-selection-v25':
            from dexvision.evaluation.place_pad_release_selection import load_selection_spec
        elif selection_version == 'level5/place-wrist-hold-selection-v24':
            from dexvision.evaluation.place_wrist_hold_selection import load_selection_spec
        elif selection_version == 'level5/place-bin-opening-selection-v23':
            from dexvision.evaluation.place_bin_opening_selection import load_selection_spec
        elif selection_version == 'level5/place-supported-descent-selection-v22':
            from dexvision.evaluation.place_supported_descent_selection import load_selection_spec
        elif selection_version == 'level5/place-shadow-selection-v21':
            from dexvision.evaluation.place_shadow_selection import load_selection_spec
        elif selection_version == 'level5/place-object-calibration-selection-v20':
            from dexvision.evaluation.place_object_calibration_selection import load_selection_spec
        elif selection_version == 'level5/place-full-source-selection-v19':
            from dexvision.evaluation.place_full_source_selection import load_selection_spec
        elif selection_version == 'level5/place-geometric-release-selection-v18':
            from dexvision.evaluation.place_geometric_release_selection import load_selection_spec
        elif selection_version == 'level5/place-source-wrist-selection-v17':
            from dexvision.evaluation.place_source_wrist_selection import load_selection_spec
        elif selection_version == 'level5/place-contact-selection-v16':
            from dexvision.evaluation.place_contact_selection import load_selection_spec
        elif selection_version == 'level5/place-physical-selection-v15':
            from dexvision.evaluation.place_physical_selection import load_selection_spec
        elif selection_version == 'level5/place-wrist-selection-v14':
            from dexvision.evaluation.place_wrist_selection import load_selection_spec
        elif selection_version == 'level5/place-recovery-selection-v13':
            from dexvision.evaluation.place_recovery_selection import load_selection_spec
        elif selection_version == 'level5/place-adaptive-selection-v12':
            from dexvision.evaluation.place_adaptive_selection import load_selection_spec
        elif selection_version == 'level5/place-slot-clearance-selection-v11':
            from dexvision.evaluation.place_slot_clearance_selection import load_selection_spec
        elif selection_version == 'level5/place-clearance-selection-v10':
            from dexvision.evaluation.place_clearance_selection import load_selection_spec
        elif selection_version == 'level5/place-slow-selection-v9':
            from dexvision.evaluation.place_slow_selection import load_selection_spec
        elif selection_version == 'level5/place-safety-selection-v8':
            from dexvision.evaluation.place_safety_selection import load_selection_spec
        elif selection_version == 'level5/place-selection-v7':
            from dexvision.evaluation.place_selection import load_selection_spec
        else:
            raise ValueError('Unsupported place selection review recipe')

        protocol = load_selection_spec(selection_config)[0]
        root = Path(protocol['output_dir'])
    for other in spec['seeds']:
        selected = root / f'seed_{other}' / 'selected.pt'
        confirmation = root / f'confirmation_seed{other}'
        audit_dir = root / f'audit_seed{other}'
        _verify_run(confirmation)
        _verify_run(audit_dir)
        report = json.loads((confirmation / 'report.json').read_text())
        audit = json.loads((audit_dir / 'report.json').read_text())
        if selection_config is not None:
            bundle = root / f'seed_{other}'
            _verify_run(bundle)
            selection_evidence = json.loads((bundle / 'selection.json').read_text())
            if (selection_evidence.get('version') != protocol['version']
                    or selection_evidence['selection_recipe_sha256'] != digest_file(selection_config)
                    or not selection_evidence['all_target_gates_passed']
                    or not selection_evidence['validation_gates_passed']
                    or (selection_version in {'level5/place-contact-selection-v16',
                                              'level5/place-source-wrist-selection-v17',
                                              'level5/place-geometric-release-selection-v18',
                                              'level5/place-full-source-selection-v19',
                                              'level5/place-object-calibration-selection-v20',
                                              'level5/place-shadow-selection-v21',
                                              'level5/place-supported-descent-selection-v22',
                                              'level5/place-bin-opening-selection-v23',
                                              'level5/place-wrist-hold-selection-v24',
                                              'level5/place-pad-release-selection-v25'}
                        and not selection_evidence['all_target_physical_audit_passed'])
                    or audit['selection_evidence_sha256'] != digest_file(bundle / 'selection.json')
                    or audit['selection_recipe_sha256'] != digest_file(selection_config)):
                raise ValueError('Manual place review requires a verified selection')
        if (not report['state_numerical_gates_passed']
                or report['split'] != 'confirmation'
                or report['checkpoint_sha256'] != digest_file(selected)
                or report['recipe_sha256'] != digest_file(config)
                or not audit['passed']
                or audit['audited_attempts'] != report['rollout_count']
                or len(audit['audits']) != report['rollout_count']
                or not all(item['passed'] for item in audit['audits'])
                or audit['report_sha256'] != digest_file(confirmation / 'report.json')
                or audit['selected_checkpoint_sha256'] != digest_file(selected)
                or audit['recipe_sha256'] != digest_file(config)):
            raise ValueError('Manual place review requires all three matching passing confirmations and audits')
    report_dir = root / f'confirmation_seed{seed}'
    report = json.loads((report_dir / 'report.json').read_text())
    rows = {row['episode_id']: row for row in repair_matrix(spec, inputs, 'confirmation')}
    chosen = {}
    for result in report['results']:
        if result['terminal_result'] == 'succeeded':
            chosen.setdefault(result['target_type'], result)
    required = {'receptacle', 'planar_zone', 'placement_slot'}
    if not required.issubset(chosen):
        raise ValueError('Passing place report lacks a success for each target type')
    return inputs, report_dir, rows, [chosen[key] for key in sorted(required)]


def review_place(config: Path, seed: int, *, playback_speed: float = 1.0,
                 selection_config: Path | None = None):
    """Show one saved successful place per target type without rerunning policy."""
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError('Playback speed must be finite and positive')
    inputs, report_dir, rows, chosen = review_inputs(config, seed,
        selection_config=selection_config)
    import mujoco.viewer

    for result in chosen:
        row = rows[result['episode_id']]
        child = report_dir / Path(result['result_path']).parent
        _verify_run(child)
        with WorkcellPilotTask(
                workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
                skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=row['seed'],
                procedural_variation=row['variation']) as task:
            print(f"Seed {seed}: {result['target_type']} | {result['object_instance']} | "
                  f"{result['target_id']} | {result['episode_id']}", flush=True)
            with (np.load(child / 'trajectory.npz', allow_pickle=False) as saved,
                  mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer):
                camera = task.workcell.config.scene['viewer']
                viewer.cam.lookat[:] = camera['lookat_m']
                viewer.cam.distance = camera['distance_m']
                viewer.cam.azimuth = camera['azimuth_deg']
                viewer.cam.elevation = camera['elevation_deg']
                if len(saved['qpos']) != len(saved['applied_actions']) + 1:
                    raise ValueError('Place viewer trajectory state/action alignment mismatch')
                for index, (qpos, qvel) in enumerate(zip(saved['qpos'], saved['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'reviewed': False, 'cancelled': True,
                                'manual_acceptance': 'pending_owner_confirmation'}
                    if index:
                        apply_action(task, saved['applied_actions'][index - 1],
                                     inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = saved['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'] / playback_speed)
                time.sleep(2)
    return {'reviewed_target_types': len(chosen), 'physics_rerun': False,
            'manual_acceptance': 'pending_owner_confirmation'}
