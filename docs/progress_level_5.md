# Progress Level 5 — Workcell Skill Learning and Qualification

Level 5 goal:

> Train, evaluate, and package a compact library of vision-grounded skills for
> the bounded tabletop workcell released by Level 4, then prove that the same
> skills support workspace clearing, inspection-station operation, and
> workspace setup.

Level 5 begins only after the immutable Level 4 release and split manifests
verify successfully. It consumes the final Level 3 findings; it does not assume
that the Level 3 MLP was successful. The simplest justified model remains the
baseline, but a model change must follow a measured failure and a written
hypothesis.

Level 5 may not recollect or edit data to rescue a run. Missing coverage
requires a new versioned Level 4 amendment/release. Language-guided planning is
Level 7; Level 5 uses deterministic scripted plans only.

Every numbered section is one checkpoint. Complete its checks and manual gate,
update `docs/CURRENT_STATUS.md`, and stop before the next checkpoint.

---

## Level 5 System Boundary

```text
immutable Level 4 release + frozen splits + typed goals
  -> reproducible loaders and training manifests
  -> learned state-grounded policies
  -> detector/tracker and metric object observations
  -> perception-grounded policy rollouts
  -> evidence-gated corrections/model changes
  -> cross-skill qualification
  -> typed registry and deterministic executor
  -> three workcell pilots and one combined work order
```

The policy owns bounded closed-loop action prediction. The deterministic
executor owns schema validation, preconditions, safety, timeouts, terminal
checks, retry bookkeeping, and structured results.

### Required planner-visible skills

| Skill | Preconditions | Parameters | Terminal success |
|---|---|---|---|
| `reach_object` | Safe hand state; fresh unambiguous entity observation | `entity_id`, `approach_pose` | Approach pose and dwell pass |
| `pick_object` | Correct object supported, unheld, and inside approach envelope | `object_id` | Object held above lift height for stability dwell |
| `place_held_object` | Exactly one compatible object held | `target_id` | Object settled in target and no longer held |
| `push_object_to_target` | Object supported; hand inside push approach envelope | `object_id`, `target_zone` | Object settled inside target zone |
| `press_button` | Fresh button observation; hand inside fixture approach envelope | `button_id` | Correct button depth/state dwell passes |

`rotate_dial` is optional and exists only when the Level 4 manifest marks it
promoted. Grasp/lift/hold and transport/place/release remain internal phases,
not independently required policies.

### Structured result contract

Every execution returns one terminal result:

```text
succeeded
failed
timed_out
cancelled
rejected_before_execution
```

The result includes request/execution ids, skill/checkpoint versions, start and
end timestamps, precondition evaluation, terminal reason, final task metrics,
retry count, safety counters, and trajectory/report paths. A policy action is
never treated as proof of task success; success is recomputed from world state.

### Qualification tracks

Every required skill has two distinct tracks:

```text
state-grounded: simulator world state supplies ids and metric poses
perception-grounded: rendered RGB -> detector/tracker/pose -> same world-state API
```

State-grounded qualification is necessary but insufficient for the final
vision-based claim. Default-registry skills must pass both tracks. A compact
VLM is optional for semantic disambiguation and is never required for metric
pose, safety, or actions.

### Minimum evaluation protocol

Level 5.0 freezes exact matrices before training. Unless it records a stricter
protocol, every skill uses:

```text
3 independent training seeds
checkpoint selection from validation only
at least 30 held-out state-grounded rollouts per selected checkpoint
at least 30 held-out perception-grounded rollouts per selected checkpoint
at least 3 rollouts in every required held-out condition cell
the same reset seeds for compared models/ablations
all runs saved, including failures and rejected preconditions
```

### Default numerical qualification gates

Level 5.0 may tighten these gates. Relaxing one requires explicit user approval
before any test rollout and a new protocol version.

| Skill/track | Aggregate success | Worst required object/goal family | Additional gate |
|---|---:|---:|---|
| `reach_object`, state | ≥ 0.80 | ≥ 0.65 | Final pose/dwell must recompute |
| `pick_object`, state | ≥ 0.75 | ≥ 0.60 | Drop rate ≤ 0.10 |
| `place_held_object`, state | ≥ 0.75 | ≥ 0.60 | Premature release ≤ 0.10; settled placement ≥ 0.90 among nominally placed objects |
| `push_object_to_target`, state | ≥ 0.80 | ≥ 0.65 | Object never leaves board |
| `press_button`, state | ≥ 0.90 | ≥ 0.75 | Wrong-button activation = 0 |
| `rotate_dial`, state, if promoted | ≥ 0.70 | ≥ 0.60 | No overshoot beyond fixture safety range |
| Any perception-grounded skill | ≥ 0.65 | ≥ 0.50 | No more than 0.15 below its state-grounded success rate |

All qualification tracks also require:

```text
invalid/non-finite actions = 0
workspace violations = 0
joint-limit safety violations = 0
explicit terminal results = 100 percent
mean action jerk <= the frozen per-skill gate
no test-data normalization, tuning, or checkpoint selection
```

The per-skill jerk gate is frozen in 5.0 from Level 4 expert statistics and may
not exceed 1.25 times the expert 95th percentile without explicit approval.

### Status definitions

```text
qualified: both required tracks and every safety/compatibility gate pass
experimental: runnable, but one or more generalization/perception gates fail
failed: cannot execute reliably, violates safety, or has incompatible artifacts
```

A state-only success may be reported as an intermediate result, but it remains
`experimental` in the default registry until the required perception track
passes. Failed and experimental entries remain visible in reports and disabled
by default.

### Non-goals

```text
LLM planning or an HTTP service
raw VLM-to-action control
general arbitrary-object manipulation
learned regrasp/drop recovery or learned retry selection
hinged lids, tools, cutting, pouring, liquids, or deformables
complex RL, diffusion policies, or large foundation policies
cloud training or required multi-GPU infrastructure
real robot deployment
```

---

## Standard Checkpoint Procedure

Every Level 5 checkpoint must preserve dataset/split/config/schema/checkpoint
digests and provide exact focused tests, lint commands, generated artifacts,
and manual pass/fail criteria. Automated tests use synthetic data, tiny CPU
runs, and headless MuJoCo. They must not require a webcam, GUI, GPU, or model
download.

Run these before every checkpoint commit after focused checks pass:

```bash
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

Training/evaluation outputs belong under `outputs/level5/` and are not silently
committed. Published checkpoints and reports must follow the release policy
frozen in Level 5.0.

---

## Level 5.0 — Learning, Evaluation, and Artifact Freeze

### Goal

Convert the final Level 3 diagnosis and immutable Level 4 release into one
machine-readable training/qualification plan before full-scale training.

### Inputs

```text
final Level 3.8 results and recommended model/observation/action decisions
datasets/level4-v1 manifest, checksums, schemas, and split manifests
docs/level4_dataset_report.md
Level 4 world-state and skill-goal contracts
```

### Files

```text
docs/level5_learning_plan.md
configs/level5/learning.yaml
configs/level5/evaluation_common.yaml
configs/level5/skills/reach_object.yaml
configs/level5/skills/pick_object.yaml
configs/level5/skills/place_held_object.yaml
configs/level5/skills/push_object_to_target.yaml
configs/level5/skills/press_button.yaml
configs/level5/skills/rotate_dial.yaml, only if promoted
tests/test_level5_learning_plan.py
tests/test_level5_evaluation_configs.py
```

### Required decisions

```text
baseline model and evidence for carrying it forward
allowed escalation models and the failure each would address
per-skill observation, goal, action, precondition, and terminal schemas
immutable train/validation/test manifest digests
normalization ownership and statistics format
training epochs/steps, batch size, seeds, optimizer, and early-stop rule
checkpoint-selection metric and tie-break rule
state/perception rollout matrices and exact reset seeds
all numerical gates, including expert-derived jerk limits
artifact paths, naming, checksums, retention, and publication rules
qualified/experimental/failed logic
```

The Level 3.4 result must be explicitly handled. Low reach success and 21
workspace/joint-limit terminations cannot be dismissed because jerk and action
validity passed. The plan must cite the final Level 3 diagnostic conclusion and
state whether it changes data weighting, observation/action fields, model
history, safety shaping, or merely the expected baseline.

### Commands

```bash
conda run -n dexvision pytest -q tests/test_level5_learning_plan.py tests/test_level5_evaluation_configs.py
conda run -n dexvision ruff check tests/test_level5_learning_plan.py tests/test_level5_evaluation_configs.py
```

### Pass criteria

```text
[ ] Every required skill and evaluation track has an executable frozen protocol
[ ] Validation selects checkpoints; held-out tests cannot tune anything
[ ] Three seeds, rollout counts, reset matrices, gates, and artifact paths are explicit
[ ] Model escalation follows measured Level 3/4 evidence
[ ] Dataset, split, config, and schema digests are immutable inputs
[ ] No full-scale training begins in this checkpoint
```

Manual verification: none.

---

## Level 5.1 — Reproducible Skill Dataset and Training Infrastructure

### Goal

Load Level 4 skill segments and train any configured policy through one
reproducible interface without task-specific hidden offsets.

### Files

```text
dexvision/learning/skill_datasets.py
dexvision/learning/training_manifest.py
dexvision/learning/skill_models.py
dexvision/learning/train_skill.py
dexvision/apps/train_skill.py
tests/test_skill_datasets.py
tests/test_training_manifest.py
tests/test_train_skill_tiny.py
```

### Required behavior

```text
resolve named observation/action/goal layouts from saved schemas
whole-episode and session/condition-grouped splits only
training-only normalization
select expert, failure, correction, visual, and legacy streams explicitly
deterministic sampling and weighting from config
record seed, environment, device, dataset/split/config/schema digests
save optimizer, scheduler, normalization, and exact checkpoint-selection state
resume without changing sample assignment or random sequence
CPU smoke path and optional CUDA acceleration
```

### Commands

```bash
conda run -n dexvision pytest -q tests/test_skill_datasets.py tests/test_training_manifest.py tests/test_train_skill_tiny.py
conda run -n dexvision ruff check dexvision/learning/skill_datasets.py dexvision/learning/training_manifest.py dexvision/learning/skill_models.py dexvision/learning/train_skill.py dexvision/apps/train_skill.py tests/test_skill_datasets.py tests/test_training_manifest.py tests/test_train_skill_tiny.py
python -m dexvision.apps.train_skill --config configs/level5/skills/reach_object.yaml --dry-run
```

### Pass criteria

```text
[ ] No session, episode, object, goal, or image split leakage
[ ] Every tensor shape and field name follows an executable schema
[ ] Failure/correction/legacy streams are opt-in and separately countable
[ ] Interrupted training resumes reproducibly
[ ] A tiny CPU dataset overfits and reloads its selected checkpoint
[ ] Dry-run prints inputs, split counts, model size, device, outputs, and digests
```

Manual verification: none.

---

## Level 5.2 — State-Grounded Reach Policy

### Goal

Train and qualify `reach_object` against the frozen Level 4 state-grounded
matrix, retaining the Level 3 policy as an explicit comparison.

### Files

```text
configs/level5/skills/reach_object.yaml
dexvision/evaluation/evaluate_skill.py
dexvision/apps/evaluate_skill.py
tests/test_reach_skill_training.py
tests/test_reach_skill_rollout.py
```

### Required comparisons

```text
Level 3 selected reach checkpoint on compatible Level 4 scenarios
Level 5 baseline with all other choices fixed
per-session, per-object, per-goal, and near-boundary breakdowns
training/validation curves versus held-out closed-loop success
workspace/joint-limit patterns versus the Level 3.4 report
```

### Commands

```bash
python -m dexvision.apps.train_skill --config configs/level5/skills/reach_object.yaml
python -m dexvision.apps.evaluate_skill --config configs/level5/skills/reach_object.yaml --checkpoint <checkpoint> --track state --output-dir outputs/level5/reach_object/state
conda run -n dexvision pytest -q tests/test_reach_skill_training.py tests/test_reach_skill_rollout.py
```

### Pass criteria

```text
[ ] Three seeded runs and the frozen checkpoint-selection rule complete
[ ] At least 30 held-out state rollouts are saved with explicit terminal results
[ ] Aggregate and worst-family success gates pass or the skill is marked experimental/failed
[ ] Invalid, workspace, and joint-limit violations are zero for qualification
[ ] Level 3.4 failure categories are compared directly, not hidden by aggregate success
```

Manual verification is required only if the state policy meets numerical
qualification gates. Run:

```bash
python -m dexvision.apps.evaluate_skill --config configs/level5/skills/reach_object.yaml --checkpoint <checkpoint> --track state --viewer --scenario-set manual_stratified
```

Pass when all displayed targets match their ids, trajectories remain inside
the workspace, and visible contact/dwell agrees with the terminal result. Stop
for user confirmation before marking the policy state-qualified.

---

## Level 5.3 — State-Grounded Pick Policy

### Goal

Train and qualify `pick_object` from a valid approach envelope through acquire,
lift, and stability phases for the three supported rigid-object families.

### Files

```text
configs/level5/skills/pick_object.yaml
dexvision/learning/phase_losses.py
tests/test_pick_skill_training.py
tests/test_pick_skill_rollout.py
```

### Required metrics

```text
acquisition rate, lift-height success, stable-held dwell
slip/drop rate and wrong-object contact
success by session, family, instance, size, mass, friction, and approach
phase-conditioned offline error and closed-loop terminal result
unwanted contact, jerk, workspace, and joint-limit counters
```

### Commands

```bash
python -m dexvision.apps.train_skill --config configs/level5/skills/pick_object.yaml
python -m dexvision.apps.evaluate_skill --config configs/level5/skills/pick_object.yaml --checkpoint <checkpoint> --track state --output-dir outputs/level5/pick_object/state
conda run -n dexvision pytest -q tests/test_pick_skill_training.py tests/test_pick_skill_rollout.py
```

### Pass criteria

```text
[ ] Held-object success is recomputed from contact/object motion, not predicted action
[ ] Internal phase outcomes remain visible in every report
[ ] Failed acquisitions, slips, drops, and worst-family results are retained
[ ] Required success/drop/safety gates pass or status is experimental/failed
[ ] Held-out object instances never influence checkpoint selection
```

Manual verification is required only after numerical gates pass. Use the same
evaluation command with `--viewer --scenario-set manual_stratified`. Pass when
one rollout per object family visibly acquires the correct object, clears the
lift height, and holds without table penetration or hidden reset. Stop for user
confirmation.

---

## Level 5.4 — State-Grounded Place-Held-Object Policy

### Goal

Train and qualify `place_held_object` from a verified held-object precondition
through transport, placement, controlled release, and settling.

### Files

```text
configs/level5/skills/place_held_object.yaml
tests/test_place_skill_training.py
tests/test_place_skill_rollout.py
```

### Required metrics

```text
transport retention and drop rate
target/receptacle placement success
release timing and premature-release rate
post-release stability, bounce, and final pose error
success by source, target, object family/instance, and session
safety and unwanted-contact counters
```

### Commands

```bash
python -m dexvision.apps.train_skill --config configs/level5/skills/place_held_object.yaml
python -m dexvision.apps.evaluate_skill --config configs/level5/skills/place_held_object.yaml --checkpoint <checkpoint> --track state --output-dir outputs/level5/place_held_object/state
conda run -n dexvision pytest -q tests/test_place_skill_training.py tests/test_place_skill_rollout.py
```

### Pass criteria

```text
[ ] Execution rejects missing, multiple, wrong, or stale held-object state
[ ] Transport, place, release, and settle phases are independently measurable
[ ] Every required target and held-out goal region is evaluated
[ ] Misplacements and unstable post-release states remain failures
[ ] Required success/release/safety gates pass or status is experimental/failed
```

Manual verification is required after numerical gates pass. Run the viewer on
at least one target of each type. Pass when the same held object is transported,
released inside tolerance, remains stable, and the reported result matches the
scene. Stop for user confirmation.

---

## Level 5.5 — State-Grounded Push, Press, and Optional Dial Policies

### Goal

Train and qualify the remaining contact skills through the common training and
evaluation interfaces.

### Files

```text
configs/level5/skills/push_object_to_target.yaml
configs/level5/skills/press_button.yaml
configs/level5/skills/rotate_dial.yaml, only if promoted
tests/test_push_skill_rollout.py
tests/test_press_skill_rollout.py
tests/test_dial_skill_rollout.py, only if promoted
```

### Required coverage

```text
push: object family, size, mass, friction, start, direction, and target zone
press: button id/state, approach direction, depth, dwell, and wrong-contact cases
dial: start angle, target angle, direction, overshoot, and contact loss, if promoted
all: Level 2 seed comparison reported separately from Level 4 generalization
```

### Commands

```bash
python -m dexvision.apps.train_skill --config configs/level5/skills/push_object_to_target.yaml
python -m dexvision.apps.train_skill --config configs/level5/skills/press_button.yaml
python -m dexvision.apps.evaluate_skill --config configs/level5/skills/push_object_to_target.yaml --checkpoint <checkpoint> --track state --output-dir outputs/level5/push_object_to_target/state
python -m dexvision.apps.evaluate_skill --config configs/level5/skills/press_button.yaml --checkpoint <checkpoint> --track state --output-dir outputs/level5/press_button/state
conda run -n dexvision pytest -q tests/test_push_skill_rollout.py tests/test_press_skill_rollout.py
```

Run equivalent dial commands/tests only when the Level 4 manifest promotes it.

### Pass criteria

```text
[ ] Push and press each complete three seeded training runs and frozen rollouts
[ ] Physical task state recomputes every success and failure
[ ] Wrong contact, contact loss, overshoot, and object-off-board failures remain visible
[ ] Required per-skill success and safety gates pass or status is experimental/failed
[ ] Optional dial is qualified, experimental, failed, or explicitly absent
```

Manual verification is required after numerical gates pass. View one
stratified push and one press rollout; add one dial rollout only if promoted.
Pass when contact and terminal state visibly match each report. Stop for user
confirmation.

---

## Level 5.6 — Detector, Tracker, and Metric Pose Baseline

### Goal

Train or integrate a lightweight visual pipeline that converts the Level 4
single-camera images into the same object-observation schema used by the state
policies.

### Files

```text
dexvision/perception/object_detector.py
dexvision/perception/object_tracker.py
dexvision/perception/object_pose.py
dexvision/evaluation/evaluate_perception.py
dexvision/apps/train_perception.py
dexvision/apps/evaluate_perception.py
configs/level5/perception.yaml
tests/test_object_perception.py
tests/test_perception_metrics.py
```

### Build order

```text
1. deterministic simulator-truth observation adapter
2. conventional detector or segmenter baseline
3. stable temporal ids and stale-observation rejection
4. metric pose/center estimate with confidence and calibration
5. optional compact VLM semantic selector behind a separate interface
```

Default visual gates, frozen before training:

```text
core-class detection recall at IoU 0.5 >= 0.95
median metric position error <= 0.010 m
95th-percentile metric position error <= 0.025 m
identity switches <= 1 per 1000 tracked frames
stale or ambiguous observation rejection = 100 percent in synthetic tests
optional semantic selector accuracy >= 0.90 if it is claimed
```

### Commands

```bash
python -m dexvision.apps.train_perception --config configs/level5/perception.yaml
python -m dexvision.apps.evaluate_perception --config configs/level5/perception.yaml --output-dir outputs/level5/perception
conda run -n dexvision pytest -q tests/test_object_perception.py tests/test_perception_metrics.py
```

### Pass criteria

```text
[ ] Train/validation/test visual splits match Level 4 episode ownership
[ ] Detection, identity, pose, and optional semantic metrics are separate
[ ] Ground-truth and inferred observations use the same typed world-state schema
[ ] Ambiguous, low-confidence, and stale results fail closed
[ ] Model size, license, latency, device, and RTX 5070 Ti memory use are documented
[ ] No visual model emits actuator commands
```

Manual verification is required on a held-out contact sheet/video. Pass when
ids remain attached to the correct objects, boxes/masks and pose markers align,
and ambiguity is visibly rejected. Stop for user confirmation.

---

## Level 5.7 — Perception-Grounded Skill Rollouts

### Goal

Run the selected policies unchanged while the detector/tracker supplies ids and
poses through the world-state API.

### Files

```text
dexvision/learning/perception_policy.py
dexvision/evaluation/evaluate_skill.py
configs/level5/evaluation_perception.yaml
tests/test_perception_policy_integration.py
tests/test_perception_grounded_rollout.py
```

### Required comparisons

```text
identical reset seeds and selected policy checkpoints for state/perception tracks
perception availability, confidence, staleness, and rejection reason
success-rate delta from state-grounded execution
failure attribution: perception, policy, supervisor, physics, or timeout
latency and control-frequency impact
```

### Commands

```bash
python -m dexvision.apps.evaluate_skill --config configs/level5/evaluation_perception.yaml --all-required-skills --track perception --output-dir outputs/level5/perception_rollouts
conda run -n dexvision pytest -q tests/test_perception_policy_integration.py tests/test_perception_grounded_rollout.py
```

### Pass criteria

```text
[ ] At least 30 frozen perception-grounded rollouts exist per required skill
[ ] The policy checkpoint and reset seeds match the state comparison
[ ] Perception errors cannot silently fall back to simulator truth
[ ] Success, worst-family, degradation, latency, and safety gates are reported
[ ] Each skill passes or receives an explicit experimental/failed reason
```

Manual verification: view one held-out rollout per required skill with
perception overlays. Pass when the overlay, selected id/pose, action, and
terminal result remain consistent. Stop for user confirmation.

---

## Level 5.8 — Corrective Learning and Evidence-Gated Model Escalation

### Goal

Use Level 4 corrections or a more expressive model only where frozen metrics
identify a failure the proposed change can plausibly address.

### Files

```text
docs/level5_model_decisions.md
dexvision/learning/corrective_sampling.py
configs/level5/ablations/
tests/test_corrective_sampling.py
tests/test_model_escalation.py
```

### Allowed comparisons

```text
expert-only versus expert-plus-correction data
current-frame MLP versus short-history GRU/temporal model
base/wrist-only versus justified full-action or weighted-action ablation
state-grounded versus perception-grounded observations
```

ACT-style chunking is allowed only if the Level 3/5 evidence identifies
temporal ambiguity or compounding error. Diffusion policies, foundation
policies, and RL require a separate user-approved future checkpoint.

### Commands

```bash
python -m dexvision.apps.train_skill --config configs/level5/ablations/<ablation>.yaml
python -m dexvision.apps.evaluate_skill --config configs/level5/ablations/<ablation>.yaml --checkpoint <checkpoint> --frozen-comparison
conda run -n dexvision pytest -q tests/test_corrective_sampling.py tests/test_model_escalation.py
```

### Pass criteria

```text
[ ] Triggering failure, hypothesis, and acceptance threshold are written first
[ ] Compared runs use identical frozen data, splits, reset seeds, and metrics
[ ] Test results do not select the model
[ ] Added complexity is retained only if the predeclared metric improves
[ ] Compute, memory, latency, and reproducibility costs are reported
[ ] Learned regrasp/drop recovery is not introduced
```

Manual verification: none unless the retained model changes visible temporal
behavior; then replay identical old/new scenarios side by side and stop for
user confirmation.

---

## Level 5.9 — Cross-Skill Qualification and Release Candidate Registry

### Goal

Assign every skill an evidence-backed status and freeze exactly which
checkpoints may enter the executor and pilots.

### Files

```text
dexvision/learning/skill_cards.py
dexvision/learning/skill_registry.py
dexvision/apps/qualify_skills.py
configs/level5/qualification.yaml
outputs/level5/qualification/registry.json
docs/level5_qualification_report.md
tests/test_skill_cards.py
tests/test_skill_qualification.py
```

### Required skill-card fields

```text
name and semantic version
dataset/split/config/schema/checkpoint digests
observation, goal, and action schemas
preconditions and supported object/target ids
timeout, retryability, safety limits, and terminal conditions
state/perception metrics and status
known limitations and disabled conditions
```

### Commands

```bash
python -m dexvision.apps.qualify_skills --config configs/level5/qualification.yaml --output-dir outputs/level5/qualification
conda run -n dexvision pytest -q tests/test_skill_cards.py tests/test_skill_qualification.py
```

### Pass criteria

```text
[ ] Every required skill is qualified, experimental, or failed
[ ] Only policies passing both required tracks are qualified by default
[ ] Every status traces to immutable reports and digests
[ ] Worst-condition and negative results remain visible
[ ] Optional dial status is explicit
[ ] Registry generation is deterministic
```

Manual verification: none.

Do not continue to pilots if any skill required by all three pilots is not
qualified. Report the blocker or create a versioned Level 4/5 amendment.

---

## Level 5.10 — Deterministic Skill Executor

### Goal

Expose qualified checkpoints through one policy-independent, typed runtime
without adding a language planner or web service.

### Files

```text
dexvision/learning/skill_executor.py
dexvision/learning/skill_results.py
dexvision/apps/run_skill.py
configs/level5/executor.yaml
tests/test_skill_executor.py
tests/test_skill_result_contract.py
```

### Required behavior

```text
load only qualified, digest-compatible registry entries by default
validate parameters, ids, frames, units, and observation freshness
check preconditions before actions
enforce action bounds, workspace, joint limits, timeout, and cancellation
recompute success/failure from world state
move to safe pose after failure when safe to do so
retry once only for explicitly retryable non-safety failures
abort immediately on drop, workspace, joint-limit, ambiguity, or stale perception
idempotent request ids: duplicate request never repeats physical action
save complete SkillResult and trajectory
```

### Commands

```bash
python -m dexvision.apps.run_skill --config configs/level5/executor.yaml --skill reach_object --parameters <parameters-json> --dry-run
conda run -n dexvision pytest -q tests/test_skill_executor.py tests/test_skill_result_contract.py
```

### Pass criteria

```text
[ ] Invalid/incompatible skills and parameters are rejected before execution
[ ] Preconditions, safety, timeout, cancellation, and terminal checks are executable
[ ] Retryable and abort-only failures follow the frozen table
[ ] Duplicate request ids cannot duplicate an action
[ ] Mock and learned policies share the same request/result interface
[ ] No LLM, HTTP API, or multi-skill planner exists
```

Manual verification: run one qualified skill with `--viewer` only after
headless tests pass. Pass when the printed request/result, visible behavior,
and saved trajectory agree. Stop for user confirmation.

---

## Level 5.11 — Deterministic Workcell Pilots

### Goal

Prove that the qualified skill library composes into three coherent workcell
tasks before any language planner is introduced.

### Files

```text
dexvision/evaluation/workcell_pilots.py
dexvision/apps/run_workcell_pilot.py
configs/level5/pilots/workspace_clearing.yaml
configs/level5/pilots/inspection_station.yaml
configs/level5/pilots/workspace_setup.yaml
configs/level5/pilots/combined_inspection_job.yaml
tests/test_workcell_pilots.py
tests/test_pilot_result_contract.py
```

### Frozen pilot definitions

1. **Workspace clearing:** start with three loose rigid parts. Pick/place two
   into their configured return bins and push the flat puck into its zone.
2. **Inspection-station operation:** reach/pick the requested part, place it on
   the inspection pad, reach the Start button, and press it. Set the dial first
   only when the dial skill is promoted and qualified.
3. **Workspace setup:** place two requested components into `setup_slot_a` and
   `setup_slot_b` from randomized safe source poses.
4. **Combined acceptance work order:** place the blue cylinder on the
   inspection pad, place the red block in the left tray/setup slot, optionally
   set the dial to 45 degrees, then press Start.

The sequence is deterministic YAML/Python, not generated by an LLM. A human may
reset between trials but may not repair state between skills.

### Evaluation matrix and gates

```text
20 frozen reset seeds per core pilot
10 frozen reset seeds for the combined work order
each core pilot success rate >= 0.70
combined work-order success rate >= 0.50
workspace/joint-limit/invalid-action violations = 0
hidden human interventions = 0
every skill request/result and world-state transition logged
failure attribution and retry count reported
```

### Commands

```bash
python -m dexvision.apps.run_workcell_pilot --config configs/level5/pilots/workspace_clearing.yaml --headless --all-seeds
python -m dexvision.apps.run_workcell_pilot --config configs/level5/pilots/inspection_station.yaml --headless --all-seeds
python -m dexvision.apps.run_workcell_pilot --config configs/level5/pilots/workspace_setup.yaml --headless --all-seeds
python -m dexvision.apps.run_workcell_pilot --config configs/level5/pilots/combined_inspection_job.yaml --headless --all-seeds
conda run -n dexvision pytest -q tests/test_workcell_pilots.py tests/test_pilot_result_contract.py
```

### Pass criteria

```text
[ ] All three core pilots meet their frozen success and safety gates
[ ] The combined work order meets its separate acceptance gate
[ ] Every trial starts from reset and ends through an explicit terminal result
[ ] No experimental/failed skill or simulator-truth fallback is used
[ ] Human repair between skills is impossible or detected
[ ] Failures and deterministic retry/abort behavior remain visible
```

Manual verification is required after headless gates pass. Run each pilot once
with `--viewer --seed <documented-seed>`. Pass when the requested objects and
targets are correct, no state is repaired between skills, and visible final
state matches the saved pilot result. Stop for user confirmation.

---

## Level 5.12 — Qualified Skill Release and Level 6 Handoff

### Goal

Publish reproducible learning results, checkpoints, skill cards, registry, and
pilot evidence for Level 6 portfolio work.

### Files

```text
releases/level5-v1/README.md
releases/level5-v1/manifest.json
releases/level5-v1/SHA256SUMS
releases/level5-v1/skill_cards/
releases/level5-v1/registry.json
docs/level5_results.md
dexvision/apps/verify_skill_release.py
tests/test_level5_release.py
```

### Required results

```text
per-skill training and checkpoint-selection manifests
state/perception held-out metrics for every seed and condition
qualified/experimental/failed registry with negative results
perception model metrics, license, latency, and hardware use
failure, correction, retry, and abort analysis
three pilot and combined-work-order results
known limitations and unsupported claims
artifact/schema/config/dataset/split/checkpoint digests
```

### Commands

```bash
python -m dexvision.apps.verify_skill_release --release-dir releases/level5-v1
conda run -n dexvision pytest -q tests/test_level5_release.py tests/test_skill_qualification.py tests/test_workcell_pilots.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

### Pass criteria

```text
[ ] Every claim traces to immutable data, splits, config, code, and checkpoint digests
[ ] Release verification works from a clean directory
[ ] Negative results and disabled skills remain documented
[ ] All three workcell pilots and combined acceptance work order have reproducible reports
[ ] Level 6 receives commands/assets for results tables and demo capture
[ ] Level 7 prerequisites are exported without implementing orchestration
```

Manual verification: execute the documented clean-directory release verifier.
Pass only when all expected files restore and every checksum matches. Stop for
user confirmation before completing Level 5.

---

# Level 5 Completion Checklist

```text
[ ] 5.0 Level 3 evidence, model choices, protocols, gates, and artifacts are frozen
[ ] 5.1 loader/training infrastructure is deterministic and resumable
[ ] 5.2 reach is evaluated against Level 3 failures and Level 5 gates
[ ] 5.3 pick is evaluated across supported object families
[ ] 5.4 place-held-object is evaluated across targets and held-out goals
[ ] 5.5 push and press are evaluated; optional dial status is explicit
[ ] 5.6 detector/tracker/pose pipeline meets frozen visual gates
[ ] 5.7 perception-grounded rollouts have no simulator-truth fallback
[ ] 5.8 corrections/model escalation are evidence-gated
[ ] 5.9 every skill is qualified, experimental, or failed
[ ] 5.10 only qualified compatible skills execute by default
[ ] 5.11 all three workcell pilots and the combined work order pass
[ ] 5.12 immutable skill release and Level 6 handoff verify cleanly
[ ] No LLM, open-world manipulation, learned recovery, or deferred task is claimed
```
