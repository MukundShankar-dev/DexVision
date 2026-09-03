# Progress Level 3 — Learning Feasibility on Existing Data

Level 3 goal:

> Determine whether the current Level 2 demonstrations can train useful
> closed-loop policies, identify the simplest learning approach that works,
> and turn failures into concrete requirements for the Level 4 data haul.

Level 3 is a feasibility phase. Its datasets are intentionally narrow:

```text
reach_touch_target: 55 clean successes
button_press: 55 clean successes
push_cube_to_target: 101 clean successes
```

A successful result proves the learning and evaluation pipeline works on these
task distributions. It does not prove cross-session, cross-object,
cross-camera, cross-operator, open-world, or long-horizon generalization. The
comprehensive dataset and qualified skill library belong to Level 4.

Start with state-based behavior cloning and the simplest model that answers the
feasibility question. Do not add image policies, compact VLMs, skill
orchestration, or a large new collection campaign in this level.

The first reach-policy split and rollout gates are frozen in
`configs/level3_evaluation.yaml` and `docs/level3_evaluation_protocol.md`.
Offline action prediction and closed-loop rollout are separate evaluations.

All policies consume demonstrations recorded with the full Level 1.13 action:

```text
base position target
base orientation target
finger actuator targets
```

An experiment may expose a declared action subset, but it must not change the
stored demonstrations or infer undocumented field offsets.

## Level boundaries

```text
Level 3:
saved Level 2 state + typed goal -> learned policy -> bounded low-level action

Level 4:
multi-session comprehensive data -> qualified reusable skill library

Level 6:
language request -> typed plan -> deterministic supervisor -> qualified skills
```

Long-horizon requests such as sandwich assembly must not be learned end to end
in Level 3. Visual object identity, broader object/goal coverage, grasp/lift/
place data, recovery data, and diverse composed pilots are Level 4 work.

---

## Level 3.0 — Roadmap Rebaseline

### Goal

Align every project document around the feasibility-first Level 3, the new
comprehensive Level 4, the renumbered polish Level 5, and future orchestration
Level 6.

### Pass criteria

```text
[x] Level 3 is explicitly limited to learning feasibility on Level 2 data
[x] Level 4 owns comprehensive dataset collection and skill qualification
[x] The previous portfolio Level 4 is preserved as Level 5
[x] Language-guided orchestration is consistently Level 6
[x] At least three diverse final pilot tasks are defined
[x] Real tomato cutting is explicitly outside the core acceptance scope
[x] Current implementation target remains Level 3.1
```

No learning, collection, perception, policy runtime, or orchestration code is
implemented by this checkpoint.

---

## Level 3.1 — Goal-Conditioned Per-Skill Dataset Loader

### Goal

Load the immutable Level 2 release into PyTorch using deterministic whole-
episode splits and executable schema metadata.

### Files

```text
dexvision/learning/datasets.py
dexvision/learning/splits.py
tests/test_learning_dataset.py
```

### Input fields

```text
task/skill name
typed task parameters and goal
sampled initial state
robot qpos/qvel
object/task state when present
full Level 1.13 actions
timestamps and tracking quality
quality/recomputed-success labels
executable observation and action layouts
```

### First observation

Build state observations from saved arrays, not live camera frames:

```text
robot qpos and qvel
hand base position and stable orientation representation
finger state
task-relevant object/fixture state
typed goal parameters
previous action only if explicitly configured
```

Never infer field offsets from expected dimensions. Resolve fields through the
saved executable layout and fail on missing/incompatible schemas.

### Split rules

```text
group by complete episode
group by recording_session_id when it genuinely exists
use the frozen reach split version and seed
fit normalization on training only
save assignments, data digests, schema versions, and normalization statistics
do not invent session ids for legacy episodes
```

### Pass criteria

```text
[ ] Dataset loads directly from the verified Level 2 release/extraction
[ ] Samples include obs, goal, action, episode id, and timestep
[ ] Episode leakage is impossible
[ ] Reach assignments match the frozen protocol
[ ] Training-only normalization is reproducible
[ ] Missing fields and schema mismatches fail clearly
[ ] Synthetic tests need no camera, GUI, or GPU
```

### Codex prompt

```text
Implement Level 3.1 only. Build a goal-conditioned, per-skill PyTorch dataset
from saved Level 2 episodes, use executable layouts, create deterministic
whole-episode splits, and fit normalization on training only. Implement the
frozen reach split exactly. Do not add a model or training loop.
```

---

## Level 3.2 — Goal-Conditioned MLP Policy

### Goal

Implement the smallest useful state-based behavior-cloning policy.

### Files

```text
dexvision/learning/models.py
tests/test_learning_models.py
```

### Model

```text
normalized state observation + typed goal
  -> small configurable MLP
  -> normalized full action or declared action subset
```

The output contract must identify each base, orientation, and finger field. A
policy may not silently output a different action layout from its dataset.

### Pass criteria

```text
[ ] Forward pass accepts batched observations and goals
[ ] Output dimension and field layout match the configured action schema
[ ] Outputs are finite and deterministic in evaluation mode
[ ] Save/load round trip preserves predictions
[ ] CPU-only tests pass
```

### Codex prompt

```text
Implement Level 3.2 only. Add a small configurable goal-conditioned MLP with an
explicit observation/action schema contract and CPU-only shape/save-load tests.
Do not add training or rollout.
```

---

## Level 3.3 — Behavior-Cloning Training Loop

### Goal

Train, validate, checkpoint, and reproduce the MLP baseline.

### Files

```text
dexvision/learning/train_bc.py
dexvision/apps/train_policy.py
configs/level3_bc.yaml
tests/test_train_tiny.py
```

### Outputs

```text
checkpoint and SHA-256 digest
model/config/schema versions
dataset and split manifest digests
training-only normalization statistics
train/validation loss history
seed and environment metadata
```

### Pass criteria

```text
[ ] Tiny synthetic or five-demo subset can be overfit
[ ] Validation never updates normalization or model parameters
[ ] Checkpoint resumes deterministically where practical
[ ] Loss history and compatibility metadata are saved
[ ] CLI has clear missing-data/dependency errors
[ ] Automated test requires no GPU
```

### Codex prompt

```text
Implement Level 3.3 only. Add the behavior-cloning training loop and CLI using
the existing dataset and MLP. Save reproducibility metadata and add a tiny
CPU-only overfit test. Do not add MuJoCo rollout.
```

---

## Level 3.4 — Frozen Reach Closed-Loop Rollout

### Goal

Measure whether action-prediction learning produces useful autonomous behavior
under the frozen reach protocol.

### Files

```text
dexvision/learning/policies.py
dexvision/evaluation/evaluate_policy.py
dexvision/apps/evaluate_policy.py
tests/test_policy_rollout.py
```

### Required behavior

```text
validate checkpoint, schemas, normalization, and target pose
run all frozen training-target and held-out-target scenarios
apply base/wrist and finger outputs through bounded controls
stop on explicit success, failure, or timeout
save every run, including failures
report invalid actions, workspace violations, and joint-limit violations
```

### Metrics

```text
training-target and held-out-target success rates
final distance and completion steps
normalized action jerk
invalid-action and safety-limit counts
terminal-reason distribution
```

### Pass criteria

```text
[ ] Headless smoke rollout works with a deterministic test policy
[ ] Frozen scenario matrix is implemented without tuning on held-out runs
[ ] Full Level 1.13 action fields are applied or an ablation is named explicitly
[ ] Results include failures and exact protocol/config digests
[ ] Numerical gates in docs/level3_evaluation_protocol.md are reported honestly
```

If a visual manual check is required, the agent must provide the exact viewer
command and observable pass/fail criteria. It must not mark that manual check
complete on the user's behalf.

---

## Level 3.5A — Button and Push Evaluation Freeze

### Goal

Freeze task-specific splits, rollout matrices, and numerical gates before
training the other two Level 2 manipulation datasets.

### Files

```text
configs/level3_button_evaluation.yaml
configs/level3_push_evaluation.yaml
docs/level3_evaluation_protocol.md
tests/test_level3_evaluation_configs.py
```

### Pass criteria

```text
[ ] Whole-episode offline splits are deterministic
[ ] Existing held-out task conditions remain untouched
[ ] Rollout perturbations and terminal metrics are executable
[ ] Thresholds are justified from task geometry and Level 2 baselines
[ ] Protocol versions cannot be changed silently after evaluation
```

---

## Level 3.5B — Cross-Task Feasibility Baselines

### Goal

Train and evaluate the same simple pipeline on `button_press` and
`push_cube_to_target`.

### Pass criteria

```text
[ ] Both tasks train through the same dataset/model/training interfaces
[ ] Both use their frozen evaluation protocols
[ ] Results compare offline loss with closed-loop success
[ ] Task-specific failures are recorded, not hidden or retuned away
[ ] No new model family is introduced in this checkpoint
```

The result may be negative. A task that fails is useful evidence for Level 4
data, observation, or policy requirements.

---

## Level 3.6 — Data and Action-Space Diagnostics

### Goal

Measure which parts of the current data and action representation matter.

### Comparisons

```text
quality-passed versus broader eligible demonstrations, where both sets exist
base-only versus finger-only versus full action
goal-conditioned versus fixed-goal control, where meaningful
per-task error by action field and goal condition
```

Do not manufacture a comparison when the released dataset contains no valid
unfiltered counterpart. Record that limitation instead.

### Pass criteria

```text
[ ] Compared runs share data split, seed, model size, and evaluation protocol
[ ] Action-subset layouts are explicit and reversible
[ ] Tables include success, final task error, jerk, and safety violations
[ ] Conclusions distinguish measured effects from hypotheses
```

---

## Level 3.7 — Conditional Temporal Baseline

### Goal

Try a short-history GRU only if single-step state BC shows a measured temporal
ambiguity or compounding-error failure.

This checkpoint may be closed with a documented "not justified" decision. Do
not add a Transformer, CNN, VLM, diffusion policy, or ACT implementation merely
to expand the technology list.

### Files when justified

```text
dexvision/learning/sequence_datasets.py
dexvision/learning/temporal_models.py
tests/test_sequence_dataset.py
```

### Pass criteria

```text
[ ] Triggering failure mode is identified in prior metrics
[ ] Sequence windows never cross episode boundaries
[ ] Hidden state resets between episodes
[ ] MLP and GRU compare on identical splits and rollout conditions
[ ] Added complexity is retained only if results justify it
```

---

## Level 3.8 — Feasibility Report and Level 4 Data Requirements

### Goal

Close the learning experiment with an evidence-backed go/no-go decision and a
specific Level 4 collection plan.

### Files

```text
docs/level3_results.md
outputs/level3/<versioned metrics and plots>
docs/progress_level_4.md, only for evidence-driven scope corrections
```

### Required report

```text
what trained and what failed
offline versus closed-loop results
generalization claims that are and are not supported
data gaps by task, session, object, goal, and failure mode
recommended model carried into Level 4
required new observations, perception labels, and collection metadata
estimated Level 4 collection cells and episode counts
```

### Pass criteria

```text
[ ] Every result traces to a config, split, dataset, and checkpoint digest
[ ] Negative results and protocol changes are visible
[ ] No cross-session or open-world claim is made from Level 2 data
[ ] Level 4 requirements follow from measured failures and target pilot tasks
[ ] Current status advances only after the report is complete
```

---

# Level 3 Completion Checklist

```text
[x] Roadmap rebaselined around learning feasibility and later data scale-up
[ ] Deterministic per-skill dataset loader works
[ ] Training-only normalization and split manifests are saved
[ ] Goal-conditioned MLP and tiny overfit test work
[ ] Behavior-cloning training is reproducible
[ ] Frozen reach rollout is evaluated honestly
[ ] Button and push protocols are frozen before their training runs
[ ] Cross-task feasibility baselines are reported
[ ] Data/action diagnostics are reported
[ ] Temporal policy is justified by evidence or explicitly skipped
[ ] Feasibility report defines the Level 4 data haul
```
