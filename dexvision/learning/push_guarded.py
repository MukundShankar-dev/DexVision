"""Frozen physical entry/settling and workspace decoder around exact V7 weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.push_predicates import MODULES as PARENT_MODULES
from dexvision.learning.push_predicates import PushPolicy as ParentPolicy
from dexvision.learning.push_predicates import load_push_spec as load_parent_spec
from dexvision.learning.push_predicates import runtime_spec as runtime_spec
from dexvision.learning.train_skill import _verify_run
from dexvision.learning.training_manifest import content_digest, digest_file

MODULES = (*PARENT_MODULES, 'dexvision/learning/push_guarded.py', 'dexvision/evaluation/push_guarded.py')


def load_push_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Guarded push recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    parent, inputs = load_parent_spec(spec['parent_learning_recipe'])
    if (spec['version'] != 'level5/push-policy-v8' or spec['checkpoint'] != '5.5'
            or spec['parent_learning_recipe_sha256'] != digest_file(Path(spec['parent_learning_recipe']))
            or spec['source_runs'] != parent['output_dir']):
        raise ValueError('Guarded push parent/version mismatch')
    for key in ('workspace_bounds', 'weld_relpose', 'terminal', 'seeds', 'preconditions',
                'delta_scale_m', 'setup_hold_samples', 'goal_distance_cap_m'):
        if spec[key] != parent[key]:
            raise ValueError(f'Guarded push changed frozen {key}')
    if spec['entry_phase_alias'] != 'approach_to_nominal_contact_before_completion' or not spec['settle_hold']:
        raise ValueError('Guarded push execution decisions changed')
    return spec, inputs


def execution_pins(inputs):
    return {name: digest_file(inputs.root / name) for name in MODULES}


def project_workspace(prior, proposed, bounds):
    """Scale signed XY displacement uniformly to the original command workspace."""
    prior, proposed = np.asarray(prior, float), np.asarray(proposed, float)
    lower, upper = np.asarray(bounds['min']), np.asarray(bounds['max'])
    if (prior.shape != (27,) or proposed.shape != (27,) or not np.isfinite(np.r_[prior, proposed]).all()
            or np.any(prior[:3] < lower) or np.any(prior[:3] > upper)):
        raise ValueError('Workspace decoder needs a valid bounded prior/proposal')
    delta = proposed[:2] - prior[:2]
    fraction = 1.
    for axis, value in enumerate(delta):
        if value > 0:
            fraction = min(fraction, (upper[axis] - prior[axis]) / value)
        elif value < 0:
            fraction = min(fraction, (lower[axis] - prior[axis]) / value)
    fraction = float(max(0., fraction))
    if fraction == 1.:
        return proposed.copy(), fraction
    result = proposed.copy()
    result[:2] = np.clip(prior[:2] + fraction * delta, lower[:2], upper[:2])
    return result, fraction


class PushPolicy:
    """Learned signed progress with disclosed causal setup, dwell and bounds."""

    def __init__(self, checkpoint, spec, inputs):
        path = Path(checkpoint)
        run = path.parent.parent if path.parent.name == 'validation_candidates' else path.parent
        _verify_run(run)
        overlay = json.loads((run / 'entry_overlay.json').read_text())
        if (overlay['recipe'] != spec or overlay['recipe_sha256'] != content_digest(spec)
                or overlay['execution_sha256'] != execution_pins(inputs)):
            raise ValueError('Guarded push execution bundle changed')
        for split in ('validation', 'confirmation'):
            if overlay[f'{split}_resets_sha256'] != digest_file(
                    Path(spec['output_dir']) / f'{split}_resets/resets.json'):
                raise ValueError('Guarded push execution matrix changed')
        source = Path(overlay['source_run'])
        _verify_run(source)
        if digest_file(source / 'SHA256SUMS') != overlay['source_run_sha256']:
            raise ValueError('Guarded push source artifacts changed')
        parent, _ = load_parent_spec(spec['parent_learning_recipe'])
        self.parent = ParentPolicy(path, parent, inputs)
        self.names, self.spec = self.parent.names, spec
        self.events = []

    def reset(self):
        self.parent.reset()
        self.events = []

    def __call__(self, source, prior, **kwargs):
        source = np.asarray(source, float)
        nominal = source.copy()
        alias = bool(source[self.names.index('phase.approach')] > 0 and not self.parent.qualified_once)
        if alias:
            nominal[self.names.index('phase.approach')] = 0
            nominal[self.names.index('phase.push_contact')] = 1
        proposed, prediction = self.parent(nominal, prior, **kwargs)
        # Runtime phase.settle is produced only by the unchanged physical
        # qualifying predicate. Hold ordinary physics through its five-sample
        # dwell; do not let a learned transient retract invalidate that dwell.
        hold = bool(source[self.names.index('phase.settle')] > 0 and not self.parent.qualified_once)
        desired = np.asarray(prior, float).copy() if hold else proposed
        raw, fraction = project_workspace(prior, desired, self.spec['workspace_bounds'])
        self.events.append({'entry_phase_alias': alias, 'settle_hold': hold,
            'workspace_fraction': fraction, 'unbounded_learned_proposal': proposed.tolist()})
        return raw, prediction


def audit_decoder_events(events, trajectory, spec):
    """Recompute every engineered hold/bound from saved proposals and prior commands."""
    with np.load(trajectory, allow_pickle=False) as arrays:
        actions = arrays['raw_actions'] if 'raw_actions' in arrays else np.empty((0, 27))
        priors = arrays['prior_applied_actions'] if 'prior_applied_actions' in arrays else np.empty((0, 27))
        if len(events) != len(actions):
            raise ValueError('Guarded push event/trajectory length mismatch')
        for event, prior, raw in zip(events, priors, actions, strict=True):
            proposal = prior if event['settle_hold'] else event['unbounded_learned_proposal']
            expected, fraction = project_workspace(prior, proposal, spec['workspace_bounds'])
            if not np.array_equal(raw, expected) or fraction != event['workspace_fraction']:
                raise ValueError('Guarded push decoder evidence mismatch')
    return {'passed': True, 'samples': len(events),
        'entry_phase_alias_samples': sum(e['entry_phase_alias'] for e in events),
        'settle_hold_samples': sum(e['settle_hold'] for e in events),
        'workspace_bound_samples': sum(int(e['workspace_fraction'] < 1) for e in events)}
