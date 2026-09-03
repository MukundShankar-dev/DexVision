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
comprehensive dataset belongs to Level 4; the qualified skill library belongs
to Level 5.

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
multi-session comprehensive data -> immutable dataset release

Level 5:
Level 4 release -> full-scale learning -> qualified reusable skill library

Level 7:
language request -> typed plan -> deterministic supervisor -> qualified skills
```

Long-horizon requests such as sandwich assembly must not be learned end to end
in Level 3. Broader object/goal coverage and visual/grasp/lift/place/recovery
data are Level 4 work. Full-scale perception/policy learning, qualification,
and diverse composed pilots are Level 5 work.

---

## Level 3.0 — Roadmap Rebaseline

### Goal

Establish Level 3 as a feasibility-first phase and reserve a later phase for a
comprehensive data and skill campaign. Level 3.0B below supersedes its initial
combined data/learning numbering.

### Pass criteria

```text
[x] Level 3 is explicitly limited to learning feasibility on Level 2 data
[x] A later comprehensive data/skill campaign is defined
[x] Current implementation target remains Level 3.1
```

No learning, collection, perception, policy runtime, or orchestration code is
implemented by this checkpoint.

---

## Level 3.0B — Dataset/Learning Phase Separation

### Goal

Separate the comprehensive data haul from full-scale skill learning and shift
all later roadmap phases consistently.

### Pass criteria

```text
[x] Level 4 owns comprehensive dataset collection and immutable release
[x] Level 5 owns full-scale skill learning, qualification, and runtime
[x] The former portfolio phase is preserved as Level 6
[x] Language-guided orchestration is consistently Level 7
[x] Diverse pilots belong to Level 5 and include more than sandwich assembly
[x] Real tomato cutting remains outside the core acceptance scope
[x] Current implementation target remains Level 3.1
```

No learning, collection, perception, policy runtime, or orchestration code is
implemented by this docs-only checkpoint.

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
[x] Dataset loads directly from the verified Level 2 release/extraction
[x] Samples include obs, goal, action, episode id, and timestep
[x] Episode leakage is impossible
[x] Reach assignments match the frozen protocol
[x] Training-only normalization is reproducible
[x] Missing fields and schema mismatches fail clearly
[x] Synthetic tests need no camera, GUI, or GPU
```

Automated checks passed on September 3, 2026 using `conda run -n dexvision
pytest -q tests/test_learning_dataset.py tests/test_dataset_schema.py
tests/test_replay_loader.py tests/test_level3_evaluation_protocol.py` with 37
passed and `conda run -n dexvision ruff check dexvision/learning/datasets.py
dexvision/learning/splits.py tests/test_learning_dataset.py`. The verified
Level 2 extraction loaded 55 clean reach, 55 clean button, and 101 clean
push-cube episodes. The frozen reach split contains 43 training and 12
validation episodes, uses training-only normalization, and records that the
legacy release has no recording-session ids. No manual verification was
required.

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
[x] Forward pass accepts batched observations and goals
[x] Output dimension and field layout match the configured action schema
[x] Outputs are finite and deterministic in evaluation mode
[x] Save/load round trip preserves predictions
[x] CPU-only tests pass
```

Automated checks passed on September 3, 2026 using `conda run -n dexvision
pytest -q tests/test_learning_models.py tests/test_learning_dataset.py
tests/test_dataset_schema.py tests/test_replay_loader.py
tests/test_level3_evaluation_protocol.py tests/test_imports.py` with 47 passed,
`conda run -n dexvision ruff check dexvision/learning/models.py
tests/test_learning_models.py`, and `conda run -n dexvision pytest -q` with 411
passed. The model checkpoint preserves its exact observation, goal, dataset-
action, and output-action layouts. Named action subsets are supported only as
explicit order-preserving ablations. No training or rollout was implemented,
and no manual verification was required.

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
[x] Tiny synthetic or five-demo subset can be overfit
[x] Validation never updates normalization or model parameters
[x] Checkpoint resumes deterministically where practical
[x] Loss history and compatibility metadata are saved
[x] CLI has clear missing-data/dependency errors
[x] Automated test requires no GPU
```

Automated checks passed on September 3, 2026 using `conda run -n dexvision
pytest -q tests/test_train_tiny.py tests/test_learning_models.py
tests/test_learning_dataset.py` with 18 passed, `conda run -n dexvision ruff
check dexvision tests`, and `conda run -n dexvision pytest -q` with 415 passed.
The real frozen reach dataset completed the configured 100-epoch CPU training
command and produced a checkpoint whose SHA-256 sidecar verified successfully.
The final smoke-run training and validation losses were 0.01456721 and
0.09558076. Checkpoints preserve the exact policy schema, optimizer state,
loss history, config and environment metadata, dataset/split/normalization
digests, and training-only normalization statistics. No manual verification
was required, and no rollout was implemented.

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
[x] Headless smoke rollout works with a deterministic test policy
[x] Frozen scenario matrix is implemented without tuning on held-out runs
[x] Full Level 1.13 action fields are applied or an ablation is named explicitly
[x] Results include failures and exact protocol/config digests
[x] Numerical gates in docs/level3_evaluation_protocol.md are reported honestly
```

Automated checks passed on September 3, 2026 using `conda run -n dexvision
pytest -q tests/test_policy_rollout.py tests/test_level3_evaluation_protocol.py`
with 7 passed, `conda run -n dexvision ruff check
dexvision/learning/policies.py dexvision/evaluation/evaluate_policy.py
dexvision/apps/evaluate_policy.py tests/test_policy_rollout.py`, and `conda run
-n dexvision pytest -q` with 420 passed. The real 100-epoch reach checkpoint
was reproduced with SHA-256
`8ae19960d877a46010191c68743ee7337dbc90fbe094b4286955688f50488bbc`, then
all 35 frozen headless scenarios were saved and evaluated. The policy did not
pass the frozen baseline: training-target success was 0.381, held-out-target
success was 0.643, mean normalized action jerk was 0.040596, invalid actions
were 0, workspace violations were 7, and joint-limit violations were 14.
Terminal reasons were 17 successes, 11 joint-limit violations, and 7 workspace
violations; all 18 failures remain in the report. No held-out target, offset,
threshold, or model setting was changed after evaluation. No manual
verification was required.

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
[x] Whole-episode offline splits are deterministic
[x] Existing held-out task conditions remain untouched
[x] Rollout perturbations and terminal metrics are executable
[x] Thresholds are justified from task geometry and Level 2 baselines
[x] Protocol versions cannot be changed silently after evaluation
```

Automated checks passed on September 3, 2026 using `conda run -n dexvision
pytest -q tests/test_level3_evaluation_configs.py` with 8 passed, the focused
evaluation/dataset/task suite with 59 passed, `conda run -n dexvision ruff
check dexvision tests`, and `conda run -n dexvision pytest -q` with 431 passed.
The frozen button protocol contains 63 training-goal and 21 held-out-goal
rollout scenarios; the frozen push protocol contains 15 of each. Both preserve
the exact Level 2 held-out declarations, use deterministic whole-episode
training/validation splits, bind terminal metrics to existing task callables,
and pin their v1 YAML digests in tests. Gates are fixed from task geometry,
the Level 2 quality threshold, dataset-summary counts, and the Level 2.10B
push-cube baseline. No policy was trained or evaluated, and no manual
verification was required.

---

## Level 3.5B — Checkpoint-Selection Repair and Cross-Task Baselines

### Goal

Correct the validation-checkpoint selection flaw exposed by Level 3.4, rerun
reach without changing its data or frozen gates, then train and evaluate the
same corrected simple pipeline on `button_press` and `push_cube_to_target`.

Level 3.4 evaluated the epoch-100 checkpoint even though validation loss was
best at epoch 15. This is a training/checkpoint-selection defect, not permission
to tune on held-out rollouts. The original checkpoint, report, and negative
result remain immutable evidence.

### Files

```text
dexvision/learning/train_bc.py
dexvision/apps/train_policy.py
configs/level3_bc.yaml
task-specific button/push training configs
tests/test_train_tiny.py
tests/test_checkpoint_selection.py
task-specific button/push training and rollout tests
```

### Required sequence

```text
1. Save separate last and best-validation checkpoints during training.
2. Select best by lowest offline validation loss; break an exact tie by the
   earliest epoch.
3. Retrain reach with the same dataset digest, split, seed, architecture,
   optimizer settings, epoch budget, action layout, and normalization.
4. Evaluate the best-validation reach checkpoint against the unchanged 35-run
   matrix and numerical gates under new versioned v2 output paths.
5. Preserve and compare the Level 3.4 v1 checkpoint/report; never overwrite it.
6. Train and evaluate button and push with the same corrected selection rule
   and their Level 3.5A-frozen protocols.
```

Held-out rollout outcomes may be reported but must not choose an epoch,
hyperparameter, or model. A v2 reach improvement is evidence for checkpoint
selection; it does not erase the original v1 result.

### Required outputs

```text
best and last checkpoint paths plus SHA-256 digests
selected epoch and validation metric
complete loss history
v1-versus-v2 reach comparison using identical frozen scenarios
button and push offline/closed-loop reports
dataset, split, config, schema, checkpoint, and protocol digests
```

### Pass criteria

```text
[x] Best and last checkpoints are distinct, versioned, and reproducible
[x] Selection uses offline validation only and follows the frozen tie-break rule
[x] Corrected reach uses identical data/split/seed/model/training settings and gates
[x] Original Level 3.4 artifacts remain unchanged and v2 writes new paths
[x] Both tasks train through the same dataset/model/training interfaces
[x] Both use their frozen evaluation protocols
[x] Results compare offline loss with closed-loop success
[x] Task-specific failures are recorded, not hidden or retuned away
[x] No new model family is introduced in this checkpoint
```

Any result may remain negative. A failed corrected reach, button, or push task
is useful evidence for Level 3.6 diagnostics and Level 4 data, observation, or
policy requirements. Do not add an unplanned hyperparameter sweep.

Automated checks passed on September 3, 2026 using `conda run -n dexvision
pytest -q tests/test_checkpoint_selection.py tests/test_cross_task_policy.py
tests/test_train_tiny.py tests/test_policy_rollout.py
tests/test_level3_evaluation_configs.py` with 23 passed, `conda run -n
dexvision ruff check dexvision tests`, and `conda run -n dexvision pytest -q`
with 437 passed. Repeated fixed-seed training reproduced all six best/last
checkpoint SHA-256 digests exactly.

Offline validation selected reach epoch 15 (loss 0.06687358), button epoch 17
(loss 0.04679401), and push epoch 97 (loss 0.00068553). The corrected reach
policy used the unchanged 35-scenario protocol but achieved 0.000 training and
0.000 held-out success, with 20 workspace and 15 joint-limit terminal
failures; this is worse than the preserved Level 3.4 v1 result and is recorded
under `outputs/level3/reach_rollout_v2/` without overwriting v1. Button ran all
84 frozen scenarios and push ran all 30. Both achieved 0.000 training and
held-out success: button recorded 84 joint-limit terminal failures, while push
recorded 30 joint-limit terminal failures with 60 violating action values.
No held-out outcome selected a checkpoint or changed a model, training setting,
scenario, or gate. No manual verification was required.

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

### Files

```text
configs/level3_diagnostics.yaml
dexvision/evaluation/level3_diagnostics.py
dexvision/apps/run_level3_diagnostics.py
tests/test_level3_diagnostics.py
```

### Pass criteria

```text
[x] Compared runs share data split, seed, model size, and evaluation protocol
[x] Action-subset layouts are explicit and reversible
[x] Tables include success, final task error, jerk, and safety violations
[x] Conclusions distinguish measured effects from hypotheses
```

Automated checks passed on September 3, 2026 using `conda run -n dexvision
python -m dexvision.apps.run_level3_diagnostics --help`, `conda run -n
dexvision ruff check dexvision tests`, and `conda run -n dexvision pytest -q`
with 442 passed. The real CPU/headless command `conda run -n dexvision python
-m dexvision.apps.run_level3_diagnostics --config
configs/level3_diagnostics.yaml` completed 13 report rows: the three preserved
full-action baselines and ten new controlled experiments. Every task compared
full, base-only, finger-only, and fixed-training-mean goal input using the same
seed, hidden layers, dataset split, and frozen scenario matrix. Omitted action
fields held their prior applied values under an explicit reversible layout.

Reach also compared its 55 quality-passed successes with all 69 recomputed
successes; all 55 shared episodes retained their baseline split assignments.
Button and push contained no recomputed-success episodes outside their 55 and
101 quality-passed sets, so the report records those comparisons as unavailable.
All variants remained negative on reach and push. Button base-only measured
0.333 training and held-out success, and fixed-goal measured 0.190/0.190,
versus 0.000/0.000 for full control, but both retained safety violations and
did not pass the frozen gates. The versioned JSON/CSV report includes success,
final task error, jerk, safety counts, per-field/per-goal offline error, exact
layouts, digests, measured conclusions, and separately labeled hypotheses.
No manual verification was required.

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

### Pass criteria when justified

```text
[ ] Triggering failure mode is identified in prior metrics
[ ] Sequence windows never cross episode boundaries
[ ] Hidden state resets between episodes
[ ] MLP and GRU compare on identical splits and rollout conditions
[ ] Added complexity is retained only if results justify it
```

### Closure criteria when not justified

```text
[x] Prior metrics were assessed against the documented temporal trigger
[x] Temporal ambiguity and compounding error remain unmeasured
[x] Recovery-coverage absence remains an explicitly unproven hypothesis
[x] The decision preserves measured action-space, safety, and rollout evidence
[x] No sequence dataset, GRU, or temporal training run was added
```

Decision: **not justified**. The versioned decision report and its readable
rationale are recorded in `docs/level3_temporal_baseline_decision.json` and
`docs/level3_temporal_baseline_decision.md`. Full button and push policies
terminated after a mean of 1.0 control step with joint-limit violations, while
reach terminated through 20 workspace and 15 joint-limit failures. These runs
measure safety failure, not temporal error accumulation. Level 3.6 also shows
that action-subset changes alter button success and safety outcomes, but it has
no state-aliasing, history-conditioned-error, or controlled compounding-error
metric. Its compounding-error and missing-recovery explanations remain labeled
as hypotheses. A GRU is therefore not implemented. No manual verification was
required. Automated checks passed on September 3, 2026 using `conda run -n
dexvision pytest -q tests/test_level3_temporal_decision.py
tests/test_roadmap_docs.py` with 12 passed, `conda run -n dexvision ruff check
dexvision tests`, and `conda run -n dexvision pytest -q` with 445 passed.

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
recommended model for Level 5 and the Level 4 data fields it requires
required new observations, perception labels, and collection metadata
estimated Level 4 collection cells and episode counts
```

### Pass criteria

```text
[ ] Every result traces to a config, split, dataset, and checkpoint digest
[ ] Negative results and protocol changes are visible
[ ] No cross-session or open-world claim is made from Level 2 data
[ ] Level 4 requirements follow from measured failures and Level 5 target pilots
[ ] Current status advances only after the report is complete
```

---

# Level 3 Completion Checklist

```text
[x] Roadmap separates feasibility, comprehensive data, and full-scale learning
[x] Deterministic per-skill dataset loader works
[x] Training-only normalization and split manifests are saved
[x] Goal-conditioned MLP and tiny overfit test work
[x] Behavior-cloning training is reproducible
[x] Frozen reach rollout is evaluated honestly
[x] Button and push protocols are frozen before their training runs
[x] Cross-task feasibility baselines are reported
[x] Data/action diagnostics are reported
[ ] Temporal policy is justified by evidence or explicitly skipped
[ ] Feasibility report defines the Level 4 data haul
```
