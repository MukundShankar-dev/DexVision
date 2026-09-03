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

The button-press and cube-push protocols are now frozen before their policies
are trained:

```text
level3/button-evaluation-v1 -> configs/level3_button_evaluation.yaml
level3/push-evaluation-v1   -> configs/level3_push_evaluation.yaml
```

They use the same deterministic whole-episode hash split as reach: clean
episodes are stratified by exact training goal, ordered by the SHA-256 digest
of `20260903:<episode_id>`, assigned 80 percent to training and the remainder
to validation, and normalized from training frames only. The legacy release
has no genuine recording-session ids, so neither protocol supports a
cross-session claim. There is no offline test partition; reserved conditions
are tested only by closed-loop rollout.

With the immutable Level 2 v1 release, the button split contains 37 training
and 18 validation episodes across nine button/depth goals. The push split
contains 80 training and 21 validation episodes across three lane goals.

## 7. Frozen Button-Press Rollout Matrix

The nine Level 2 training button/depth goals are copied exactly from
`configs/button_press_dataset.yaml`. The three reserved goals are also copied
exactly: left at 11 mm, centre at 13 mm, and right at 11 mm. They remain
rollout-only and must not be added to training or validation.

Every goal runs from nominal hand-base state and plus/minus 5 mm on each world
axis. Five millimetres is half the smallest 10 mm training press depth: large
enough to test closed-loop recovery, but it does not alter the selected button
or the frozen depth threshold. With one repetition, the matrix is:

```text
training-goal scenarios: 9 goals x 7 offsets = 63
held-out-goal scenarios: 3 goals x 7 offsets = 21
total: 84
```

Success is computed by `dexvision.sim.tasks.is_button_press_success` from the
saved/executed press depth, target depth, current and target pressed states,
and three-step consecutive dwell. Each run reports final press depth, terminal
depth shortfall `max(target - actual, 0)`, completion steps, normalized action
jerk, invalid actions, workspace and joint-limit violations, and its explicit
terminal reason. Failures remain in the report.

The frozen button gates are 80 percent success on both training and held-out
goals, at most 1 mm mean terminal depth shortfall, mean normalized action jerk
at most 0.20, and zero invalid actions or safety-limit violations. The Level 2
baseline contains 55/55 clean successful episodes across all nine training
goals; its quality gate already required action-jerk p95 at most 0.20. The
held-out depths interpolate the 10/12/14 mm training grid by exactly 1 mm, so
the shortfall gate has a direct geometric interpretation. The 80 percent
closed-loop gate allows some behavior-cloning compounding error while still
requiring broad success rather than isolated lucky runs.

## 8. Frozen Push-Cube Rollout Matrix

The three Level 2 lane goals and all three held-out start/target declarations
are copied exactly from `configs/push_cube_dataset.yaml`. In particular, the
held-out lateral lanes at -35 mm and +35 mm and the shifted centre start/target
remain unchanged and rollout-only.

Each frozen start runs nominally and with plus/minus 1 cm along world x and y.
The offset is applied to the cube start and to the hand-base x/y reset so their
relative approach remains unchanged; the target stays frozen. The cube's z
coordinate is never perturbed, keeping it on the table. One centimetre is less
than both the 35 mm success radius and the 35 mm half-spacing between adjacent
training and held-out lanes. With one repetition, the matrix is:

```text
training-goal scenarios: 3 goals x 5 offsets = 15
held-out-goal scenarios: 3 goals x 5 offsets = 15
total: 30
```

Planar distance, success, and workspace failure use the existing executable
`push_cube_distance`, `is_push_cube_success`, and
`push_cube_failure_reason` functions. Success requires the cube centre to stay
within 35 mm for five consecutive control steps. Each action advances the
saved 17-step control cadence. Reports include final planar distance,
completion steps, normalized action jerk, invalid actions, object-workspace and
joint-limit violations, and the explicit terminal-reason distribution.

The frozen push gates are 80 percent success on both training and held-out
goals, mean final planar distance at most the 35 mm target radius, mean
normalized action jerk at most 0.20, and zero invalid actions or safety-limit
violations. The Level 2 set contains 101/101 clean successful demonstrations;
their mean and p95 final distances are 20.40 mm and 28.32 mm. The independent
Level 2.10B counterfactual curl replay reached 0.871 success with a deterministic
95 percent episode-bootstrap interval of [0.802, 0.931]. Thus 0.80 is a
baseline-backed feasibility boundary rather than a post-training choice. That
counterfactual result reused curl-recorded base trajectories and is not a
held-out-policy result.

## 9. Version and Change Control

The two v1 YAML files are frozen before Level 3.5B training. Focused tests pin
their exact SHA-256 file digests as well as every training goal, held-out goal,
perturbation, terminal metric, and gate. Once any policy is evaluated against a
v1 protocol, the v1 file and report must be preserved. Changing any split,
goal, reset perturbation, metric, or threshold requires a new protocol version
and a new config file; prior results must continue to name the original version
and digest. A failed run is evidence, not permission to edit v1.
