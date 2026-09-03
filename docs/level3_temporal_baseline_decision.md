# Level 3.7 Temporal Baseline Decision

Decision version: `level3/temporal-baseline-decision-v1`

Decision: **not justified**

Level 3.7 permits a short-history GRU only when prior metrics identify measured
temporal ambiguity or measured compounding error. Neither trigger is present in
the completed Level 3.6 evidence. No sequence dataset, temporal model, temporal
training run, or GRU-versus-MLP rollout was added.

## Evidence assessment

The source of truth for this decision is the versioned Level 3.6 report at
`outputs/level3/diagnostics_v1/report.json`, together with the three preserved
full-action rollout reports named in the tracked decision artifact at
`docs/level3_temporal_baseline_decision.json`.

| Question | Recorded evidence | Assessment |
| --- | --- | --- |
| Was temporal ambiguity measured? | Level 3.6 contains per-field and per-goal single-step error, action-subset ablations, goal-input ablations, and one reach data-quality comparison. It contains no state-aliasing, conditional action-multimodality, or history-conditioned error metric. | No. |
| Was compounding error measured? | The full button and push policies both have mean completion length 1.0 control step and terminate through joint-limit failures. Reach averages 11.11 steps and terminates with 20 workspace and 15 joint-limit failures. No intervention isolates error accumulation from these safety failures. | No. |
| Was missing recovery coverage measured? | Level 3.6 explicitly lists compounding error and missing recovery coverage as possible explanations that the diagnostic does not prove. It performs no recovery-coverage comparison. | No; hypothesis only. |
| Did action representation affect outcomes? | Button base-only control changes training/held-out success from 0.000/0.000 to 0.333/0.333 and removes 84 joint-limit violations, although it introduces 56 workspace violations. Reach base-only removes the full policy's 20 workspace and 15 joint-limit violations and reduces mean final error from 0.29156 m to 0.11051 m, without producing success. | Yes; action-space coupling/safety is measured, though no variant passes. |
| Is there an offline-to-rollout mismatch? | Validation-selected MLP losses are 0.06687 for reach, 0.04679 for button, and 0.000686 for push, yet all three full policies have zero training-condition and held-out-condition rollout success. | Yes; measured mismatch, not a temporal causal attribution. |

The rollout failures therefore do not satisfy the temporal trigger. Adding a
GRU now would confound the next experiment with the already measured action-
space and safety problems and would turn an unproven explanation into an
architecture choice.

## Reassessment conditions

A temporal baseline becomes justified only after a targeted diagnostic records
at least one of the following under the existing whole-episode split discipline:

1. Current observations that are indistinguishable within a declared tolerance
   require materially different demonstrated actions, and a short history
   resolves that ambiguity.
2. A controlled rollout or replay intervention holds action layout and safety
   handling fixed and shows errors growing as a function of the policy's prior
   deviations rather than terminating immediately from invalid or unsafe
   outputs.

Collecting recovery demonstrations may be a sensible later data requirement,
but their current absence alone is not evidence that recurrence is the right
model change.
