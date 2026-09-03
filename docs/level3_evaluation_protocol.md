# Level 3 Evaluation Protocol

Version: `level3/reach-evaluation-v1`

This protocol freezes the first behavior-cloning evaluation before policy
training begins. Its machine-readable source is
`configs/level3_evaluation.yaml`.

This is a Level 3 learning-feasibility protocol for the narrow Level 2 reach
dataset. Passing it does not qualify a comprehensive `reach_object` skill or
support cross-session, cross-object, cross-camera, or open-world claims. Those
claims require the new Level 4 dataset and Level 5 qualification protocols.

## 1. Two Different Kinds of Evaluation

Offline validation and closed-loop rollout evaluation answer different
questions and must not be presented as one split.

Offline validation uses saved Level 2 demonstrations from the three configured
reach training targets. It measures action-prediction loss on whole episodes
that were not used for fitting. It does not measure whether the policy can
recover from its own mistakes.

Closed-loop rollout evaluation runs the policy in MuJoCo. The two target poses
reserved in `configs/reach_touch_dataset.yaml` are rollout-only held-out
conditions. They intentionally have no Level 2 training demonstrations and do
not need to appear as samples in the PyTorch dataset.

## 2. Offline Split

For the first reach baseline:

- Include only recomputed-successful, quality-passed episodes.
- Stratify by configured training target.
- Order whole episodes within each target by the SHA-256 digest of
  `20260903:<episode_id>`.
- Assign 80 percent of each target group to training and the remainder to
  validation, with at least one validation episode per target.
- Never split timesteps from one episode across partitions.
- Fit every normalization statistic on the training partition only.
- Do not create an offline test partition or claim that validation episodes
  constitute a held-out-target test.

The split manifest produced by Level 3.1 must save every episode assignment,
the split version, seed, goal id, and whether a recording-session id was
available.

## 3. Recording Sessions

The existing Level 2 manipulation metadata does not contain a reliable
recording-session id. A session id is not required to train the first baseline,
and old demonstrations must not be rewritten to invent one.

Consequently, the first baseline may claim episode-level generalization within
the existing collection conditions, but not cross-session, cross-operator,
cross-camera, or cross-lighting generalization.

Future demonstration collection should record an optional session id. When
session ids exist, every episode from one session must remain in one partition.
A later cross-session robustness claim requires at least one genuinely separate
session reserved from training.

For new Level 4 data, `recording_session_id` is mandatory rather than optional.
Level 2 episodes remain unchanged and may be used only as a labeled legacy
source in later releases.

## 4. Frozen Reach Rollout Matrix

Evaluate all three training targets and both held-out targets. For every target,
run the seven initial hand-base offsets declared in the YAML file: nominal and
plus/minus one centimetre along each world axis.

This produces:

```text
training-target scenarios: 3 targets x 7 offsets = 21
held-out-target scenarios: 2 targets x 7 offsets = 14
```

The Level 3 rollout evaluator must implement these offsets as evaluation-only
reset conditions. It must not modify demonstrations or fit normalization from
rollout states.

## 5. Acceptance Gates

The first reach policy passes the frozen baseline when:

```text
training-target success rate >= 0.80
held-out-target success rate >= 0.80
mean normalized action jerk <= 0.20
invalid actions = 0
workspace violations = 0
joint-limit violations = 0
every rollout ends through explicit success, failure, or timeout logic
```

Success uses the existing executable `reach_touch_target` distance, physical
contact, and dwell condition. Failed runs remain in the metrics report.

Offline validation chooses checkpoints and ordinary hyperparameters. The
held-out rollout matrix is a final evaluation, not a tuning set. If it exposes a
failure, report the result and create a new protocol version before changing
targets, perturbations, or thresholds.

## 6. Later Skills

Button press and cube push need their own frozen rollout matrices before their
policies are trained in Level 3.5A/3.5B. Their Level 2 held-out configurations
remain reserved, but this checkpoint does not invent acceptance thresholds for
models that do not yet exist.
