# Progress Level 5 — Workcell Skill Learning and Qualification

Current status: Last Completed **5.6**; Next Target **5.7 (unstarted)**.
V29 and fresh confirmation V8 pass every automated numerical and artifact check.
The owner confirmed the 16-panel/36-clip manual review passed on October 16,
2026; [level5_perception_acceptance.json](level5_perception_acceptance.json)
binds this acceptance to the frozen artifacts. All consumed test sets, datasets,
source, failed attempts and results remain preserved. All five inferred-control
tracks remain unsupported. Stop before 5.7; it requires a new user request.
Previous completed stages retain their original results and owner acceptance.
See [CURRENT_STATUS.md](CURRENT_STATUS.md) for checkpoint selection and
[level5_perception_verification_v29.json](level5_perception_verification_v29.json)
for the final automated evidence.

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

The original live hand-control prototype is outside the Level 5 system
boundary. Level 5 must not collect human demonstrations or corrections, require
a webcam/hand tracker, or use a human-control source to rescue training. It
consumes only the frozen scripted expert, scripted failure/correction, visual,
and explicitly qualified policy streams from Level 4.

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

Qualification also separates a standalone learned controller from a learned
residual around the deterministic expert. A residual-assisted executor can be
useful and may ship, but a zero residual already reproduces the expert. It may
be described as a learned contribution only when a frozen ablation shows a
material improvement over the unmodified expert on validation and then on the
single predeclared test evaluation. Otherwise the honest label is
`scripted_expert` with an experimental learned component, not a qualified
standalone learned skill.

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

### Training amendment after 5.2A

For all unstarted checkpoints, follow
[`level5_training_guidance.md`](level5_training_guidance.md). It supersedes
automatic reuse of the historical 5.0 full-action/scaling/early-stop recipe.
Completed 5.0–5.2 sections and their checksum-locked configs remain historical
evidence. Each future skill must freeze its own justified recipe before runs;
the reach architecture and 1,000-epoch budget are not universal defaults.

Future command blocks below are implementation targets, not currently supported
CLIs or authorization to train. `<versioned-skill-recipe>` must become a real,
independently locked recipe and have loader/test support within that checkpoint.
Do not substitute the old skill YAML without auditing its inherited settings.
Add recipe and lock paths to that checkpoint's files when the recipe is frozen.

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

**Complete — September 14, 2026.**

### Goal

Convert the final Level 3 diagnosis and immutable Level 4 release into one
machine-readable training/qualification plan before full-scale training.

### Inputs

```text
final Level 3.8 results and recommended model/observation/action decisions
datasets/level4-v1 manifest, checksums, schemas, and split manifests
docs/level4_dataset_report.md
Level 4.5B nested-subset scaling report and readiness decision
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
standalone-policy versus expert-residual claim and ablation rules
```

The Level 3.4 result must be explicitly handled. Low reach success and 21
workspace/joint-limit terminations cannot be dismissed because jerk and action
validity passed. The plan must cite the final Level 3 diagnostic conclusion and
state whether it changes data weighting, observation/action fields, model
history, safety shaping, or merely the expected baseline.

The Level 4 release must also be challenged rather than accepted from its raw
episode count. If the 4/8/16-per-cell validation curve was still materially
improving, any required readiness gate failed, or the final audit found
near-duplicate procedural samples, stop Level 5.0 and request a versioned Level
4 data amendment. Do not use test rollouts to decide whether more data are
needed.

### Commands

```bash
conda run -n dexvision pytest -q tests/test_level5_learning_plan.py tests/test_level5_evaluation_configs.py
conda run -n dexvision ruff check tests/test_level5_learning_plan.py tests/test_level5_evaluation_configs.py
```

### Pass criteria

```text
[x] Every required skill and evaluation track has an executable frozen protocol
[x] Validation selects checkpoints; held-out tests cannot tune anything
[x] Three seeds, rollout counts, reset matrices, gates, and artifact paths are explicit
[x] Model escalation follows measured Level 3/4 evidence
[x] Level 4 scaling evidence supports release readiness rather than relying on episode count alone
[x] Expert-only, standalone-policy, and expert-plus-residual claims have frozen ablations
[x] Dataset, split, config, and schema digests are immutable inputs
[x] No full-scale training begins in this checkpoint
```

Manual verification: none.

Completion evidence (September 14, 2026): `docs/level5_learning_plan.md` and
`configs/level5/` freeze the five required skill contracts, causal named
183-value observation layout, per-skill numeric goals, bounded 26-value policy
heads with full 27-field action records, three training seeds, validation-only
selection, explicit paired reset matrices, training-expert-derived jerk gates,
claim ablations, input digests and artifact retention/publication rules.
`configs/level5/SHA256SUMS` locks the plan and all seven YAML files.

Release verification with `--require-ready` passed all 33,350 files and frozen
split metadata. The 4/8/16 probe passes its original readiness gates, and the
final audit passes its similarity checks. The probe's expert-assisted nature
and its pre-replacement membership are documented without claiming standalone
learning success. Tests independently recompute jerk and validate exact reset
seeds from archive bytes; no ignored working episode supplies frozen statistics.

The listed 14 focused tests pass. Including release and documentation
regressions gives 65 passing tests. Repository-wide Ruff and whitespace checks
pass; the full suite passed **696 tests, 1 skipped in 522.71 seconds**. The skip
is the existing platform-dependent offscreen OpenGL test. The owner requested
Level 5 from a clean `main` working tree. Existing datasets, release metadata,
and completed Level 4 checkpoint sections remain unchanged. No policy training,
normalization fitting, held-out policy evaluation, or 5.1 implementation occurred.
Level 5.0 is complete; stop here. Level 5.1 is planned but has not started.

---

## Level 5.1 — Reproducible Skill Dataset and Training Infrastructure

**Complete — September 14, 2026.**

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
select expert, scripted-failure, scripted-correction, visual, and legacy-release streams explicitly
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
[x] No session, episode, object, goal, or image split leakage
[x] Every tensor shape and field name follows an executable schema
[x] Scripted failure/correction and legacy-release streams are opt-in and separately countable
[x] Interrupted training resumes reproducibly
[x] A tiny CPU dataset overfits and reloads its selected checkpoint
[x] Dry-run prints inputs, split counts, model size, device, outputs, and digests
```

Manual verification: none.

Implementation notes:

- `load_training_inputs` checks the 5.0 checksum lock, release readiness,
  executable layouts, model assets, whole-session/cell/lineage ownership,
  reserved object/goal ids, images, and normalization ownership. The baseline
  resolves exactly the frozen skill intervals. It never enumerates working
  directories to decide membership.
- `ReleaseSource` reads the immutable archive by default. An explicit
  `--release-root` reads restored files only after their release hashes match.
  Failure/correction records and frozen RGB/mask bytes are explicit separate
  selections; legacy archive reads preserve absent session ids. These streams
  never silently become baseline targets. Correction training remains disabled
  under the unchanged 5.0 plan until its later opt-in checkpoint.
- `load_skill_datasets` loads train/validation expert actions only. It rebuilds
  reset state and uses named state row t-1 for action t. Forward kinematics
  supplies typed object state and contacts without integrating actions. The
  existing physical workcell task state machine reconstructs causal phases;
  saved expert/controller and audited labels are retained only for disagreement
  reporting. All 183 common feature names and skill-specific numeric goal names
  are explicit. Continuous normalization uses float64 population statistics
  over train-manifest interval intersections; binary masks and one-hot columns
  remain unscaled.
- Both frozen MLP families are supported. The standalone model predicts bounded
  translation, rotation-vector, wrist and finger groups. Rotation is capped in
  vector norm, not merely componentwise. Phase/safety masks govern the group
  loss; full requested/commanded/applied/history records remain available.
- `train_skill` samples cell, episode and frame deterministically and saves
  each completed optimizer step, including partial-epoch sums, optimizer,
  scheduler (`null` in this protocol), all RNG states, sampler digest, offline
  selection history, normalization, environment and input/code digests. Resume
  uses the same output directory and rejects changed inputs or ownership.
  Existing run directories cannot be used for a new run. CPU is tested; the
  optional CUDA path requires an available compatible device and is untested
  on this Mac.
- Real runs retain `best_offline.pt` and validation candidates. `selected.pt`
  is created only by `select_checkpoint` with complete validation evidence bound
  to candidate, manifest, seed, matrix digest and rollout count, using the frozen
  ranking. Test evidence is rejected. Unavailable terminal error is serialized
  as `null` with a reason and ranks as positive infinity. No rollout evaluator,
  qualified skill, full-scale training or Level 5.2 implementation is supplied
  by this checkpoint. The selected-checkpoint acceptance test uses an explicitly
  synthetic validation stub, not fabricated robot rollout evidence.

API entry points are `load_training_inputs`, `select_streams`, `ReleaseSource`,
`load_skill_datasets`, `train_skill`, `load_checkpoint`, and `select_checkpoint`.
For an interrupted authorized training run, use its original config, seed,
model, device and `--output-dir` with `--resume`; `--stop-after-steps` provides
a controlled interruption without altering the frozen training budget.

Completion evidence (September 14, 2026): the 36 focused tests pass, as do
58 focused/documentation regressions and repository-wide Ruff. The full suite
passed **732 tests, 1 skipped in 558.51 seconds**. The skip is the existing
platform-dependent offscreen OpenGL check. The last training/selection fixes
also passed their 12 focused CPU tests. The listed dry-run was executed after
activating the `dexvision` Conda environment and passed without writing a run
or fitting normalization. It reports 80/32/48 reach episode assignments for
train/validation/test and the named 192-value reach input.

A separate read-only full-loader smoke loaded every nominal training and
validation interval for all five skills from the immutable archive. Test action
arrays were not loaded. Training statistics were fitted in memory only:

| Skill | Train episodes / frames | Validation episodes / frames | Input width |
|---|---:|---:|---:|
| Reach | 80 / 1,839 | 32 / 777 | 192 |
| Pick | 144 / 18,401 | 48 / 6,138 | 194 |
| Place held object | 144 / 16,845 | 48 / 5,285 | 195 |
| Push | 96 / 10,545 | 32 / 2,963 | 189 |
| Press | 96 / 1,744 | 32 / 592 | 185 |

Pick/place intervals share parent episodes; the union is 416 training and
144 validation episodes, not 752 separate recordings. The total is 65,129
eligible skill frames. An independent checksum-verified action read across
those 560 unique episodes measured maximum angular change 0.050001002642 rad,
below the frozen 0.139626-rad limit.

Causal physical phases differ from stored controller labels on train/validation
frames as follows: reach 16/11, pick 6,897/2,293, place 10,956/3,976, push 741/296,
press 1,104/368. These are disclosed preprocessing disagreements, not repaired
labels or skill qualification evidence. They remain available per episode in
`SkillEpisode.records`; downstream training/evaluation must preserve that
distinction. No full-scale policy training or policy rollout evaluation ran.
Existing datasets, release archives, and the completed 5.0 section are unchanged.
No manual verification is required. Level 5.1 is complete; stop here. Level 5.2
is the next planned checkpoint and has not started.

---

## Level 5.2 — State-Grounded Reach Policy

**Complete — September 14, 2026; policy qualification failed.**

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
[x] Three seeded runs and the frozen checkpoint-selection rule complete
[x] At least 30 held-out state rollouts are saved with explicit terminal results
[x] Aggregate and worst-family success gates pass or the skill is marked experimental/failed
[x] Zero invalid/workspace/joint-limit violations is enforced for qualification; unsafe policies are failed
[x] Level 3.4 failure categories are compared directly, not hidden by aggregate success
```

Manual verification is required only if the state policy meets numerical
qualification gates. Run:

```bash
python -m dexvision.apps.evaluate_skill --config configs/level5/skills/reach_object.yaml --checkpoint <checkpoint> --track state --viewer --scenario-set manual_stratified
```

Pass when all displayed targets match their ids, trajectories remain inside
the workspace, and visible contact/dwell agrees with the terminal result. Stop
for user confirmation before marking the policy state-qualified.

Completion evidence: all three baseline and three frozen reference runs finish
under the unchanged protocol. The 29 candidates receive 3,480 validation
attempts; every seed selects from validation only. All six selected checkpoints
receive 120 saved held-out attempts each (30 distinct resets crossed with four
conditions), for 720 total. All six obtain zero success and are explicitly
**failed**, with workspace/actuator-limit failures preserved. No policy is
state-qualified. Manual verification is not required because every numerical
qualification gate set fails; no owner acceptance is claimed.

The 19 focused tests pass, including two final added guards. Focused reach plus
documentation checks pass 31 tests. The full suite passed **749 tests, 1 skipped
in 606.78 seconds** before those two test-only additions, with no subsequent
implementation changes. Repository-wide Ruff and whitespace checks pass.
Independent artifact readback verifies all 720 trajectories, file checksums,
selection rankings, matrix coverage, action/state history and recomputed metrics
without rerunning physics. Exact commands, per-seed results, historical Level
3.4 comparisons, artifact paths and limitations are in
`docs/level5_reach_results.md`.

The pinned Level 3 selected checkpoint has incompatible observation/goal
semantics, so zero direct Level 4 legacy rollouts are claimed. The frozen
absolute-action reference is trained separately with otherwise matching
choices. Existing WorldState reach pose metrics use commanded mocap base pose;
learner proprioception uses measured free-joint pose. The report preserves this
limitation. Frozen plan/config hashes, datasets, releases and completed 5.0/5.1
sections remain unchanged. No perception, correction training, later skill,
registry or executor was implemented. Stop here; 5.3 is not started.

---

## Level 5.2A — Reach Training Diagnosis and Repair

**Complete — September 17, 2026; owner accepted the reach repair.**

Final implementation: state-based behavior cloning with five calibrated/scaled
inputs, two 128-unit ReLU hidden layers and two active outputs in the retained
26-output scaffold. Learned planar progress and vertical movement pass through
an explicit goal-direction/bounded-step decoder; orientation and joints hold.
The complete data, masked loss, optimizer, 1,000-epoch budget and runtime
description is in **Final V5 learning method** in `docs/level5_reach_repair.md`.
This is a reach-specific method, with no learned RGB perception claim.

### Goal

Explain the failed reach training method with measured training/validation
comparisons, repair the demonstrated causes, and document the method and its
limits for later skills. Preserve the completed 5.2 experiment and frozen v1
protocol. No other skill training is part of this checkpoint.

### Acceptance

- [x] Reproducible diagnosis distinguishes feature scaling, action identifiability,
      accumulated control error, and checkpoint stopping/selection.
- [x] A versioned, preregistered validation comparison isolates the proposed fix.
- [x] All three seeds meet original state success/worst-family/jerk/safety gates
      on the full validation matrix with complete retained trajectories.
- [x] Generalization evidence is evaluated only after method selection; previously
      inspected test cases are disclosed, never relabeled as untouched evidence.
- [x] Focused tests, full regression checks, artifact integrity and exact commands pass.
- [x] Future training documentation adopts measured decisions with per-skill caveats.
- [x] Required manual review passes before state qualification is claimed.

Manual review is required only after the numerical repair gates pass. Preserve
pending status until owner confirmation. Document exact replay commands and
criteria with the resulting artifacts.

Automated commands (use the `dexvision` environment):

```bash
conda run -n dexvision pytest -q tests/test_reach_repair.py tests/test_level5_learning_plan.py
conda run -n dexvision pytest -q
conda run -n dexvision ruff check .
```

Training, screening, validation, confirmation and saved-state review commands
are recorded with versioned paths in `docs/level5_reach_repair.md`. V5's three
validation-selected policies pass 120/120 attempts each with zero safety
violations; selection epochs are 300, 207 and 300. Each also passes 240/240
fresh confirmation attempts; all 720 pass independent measured pose/dwell
audit. The full regression suite passes 763 tests with one platform-dependent
skip; Ruff, hashes and retained-artifact checks pass. The owner accepted the
review and authorized completion, commit and push on September 17, 2026.
`docs/level5_reach_acceptance.json` binds that acceptance to all three selected
V5 model digests without rewriting historical evaluation receipts. The bounded
state reach gate is complete; perception and registry qualification remain
future work. At that completion handoff, 5.3 was the unstarted next target.

---

## Level 5.3 — State-Grounded Pick Policy

**Complete — September 17, 2026, with negative state qualification.**
Three seeds completed 200 epochs and four full candidate validation matrices
each. Validation selected epochs 200, 80 and 20. One-time held-out tests achieved
136/216, 120/216 and 140/216 successes; all fail success/drop/safety gates and
remain **failed**. No manual verification is required after this negative result.
The full suite passes 775 tests with one skip; Ruff, recipe/config locks, release
integrity and artifact checks pass. V1's workspace metadata conflict and all
negative attempts remain preserved; v2 binds the original recorded runtime
workspace. See [`level5_pick_results.md`](level5_pick_results.md) and
[`level5_pick_verification.json`](level5_pick_verification.json). No future skill,
perception qualification, registry admission or owner acceptance is claimed.

Before training, apply `docs/level5_training_guidance.md` and retain the reach
diagnosis as evidence. Audit object-relative geometry, tracking error, finger
closure, lift height and causal acquire/lift/hold information. Keep required
closure/orientation outputs active; reach's fixed posture cannot grasp. Freeze
the pick recipe, scales, active channels and validation candidate schedule.
Verify actual object acquisition and retention, not hand motion alone.

### Goal

Train and qualify `pick_object` from a valid approach envelope through acquire,
lift, and stability phases for the three supported rigid-object families.

### Files

```text
configs/level5/pick_policy_v1.yaml and .sha256 (preserved diagnostic)
configs/level5/pick_policy_v2.yaml and .sha256
dexvision/learning/pick_policy.py
dexvision/evaluation/pick_policy.py
docs/level5_pick_results.md
docs/level5_pick_verification.json
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
python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v2.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/pick_policy_v2.yaml --checkpoint outputs/level5/pick_53_v2/seed_5001/selected.pt --track state --release-root . --output-dir outputs/level5/pick_53_v2/test_5001
conda run -n dexvision pytest -q tests/test_pick_skill_training.py tests/test_pick_skill_rollout.py
```

Run each training/selection/test command for seeds 5001, 5002 and 5003.
Full candidate-selection commands and exact viewer criteria are in the result
report; test requires a validation-selected checkpoint and runs only once.

### Pass criteria

```text
[x] Held-object success is recomputed from contact/object motion, not predicted action
[x] Internal phase outcomes remain visible in every report
[x] Failed acquisitions, slips, drops, and worst-family results are retained
[x] Required success/drop/safety gates pass or status is experimental/failed
[x] Held-out object instances never influence checkpoint selection
```

Manual verification is required only after numerical gates pass. Use the same
evaluation command with `--viewer --scenario-set manual_stratified`. Pass when
one rollout per object family visibly acquires the correct object, clears the
lift height, and holds without table penetration or hidden reset. Stop for user
confirmation.

---

## Level 5.3A — Pick Training Diagnosis and Repair

**Complete — October 3, 2026: automated gates and owner manual acceptance pass;
the owner authorized the completion commit and push.**
The completed 5.3 negative result and all failed repairs through V6 remain
preserved. The owner approved the versioned unassisted expert/data amendment.
Source V1's 191/192 physical audit failure is retained; corrected V2 repeats all
192 frozen cases and passes independent physical verification without replacing
resets or overwriting original data. Teacher V4 uses ordinary physics and
recomputed post-integration evidence on a separate state copy.

V7's state MLP emits binary close/lift gate scores and continuous orientation
magnitude. Rates, posture synergy, geometric direction and bounds are engineered;
closure targets are constant one. Its causal 29-feature transform, decoder,
scales, selected weights and cadence remain a single bound bundle. No expert
continuation or object pose writes assist the learned rollout. All instances
are exposed and scripted family-aware approach/preflight remains required.

### Acceptance

- [x] Training/validation diagnosis and separate checksum-locked recipe recorded.
- [x] Causal transform, target/decoder parity, zero-action and safety tests pass.
- [x] V7 completes all three training budgets and all full validation selections.
- [x] One-time fresh confirmation passes all unchanged gates for all three seeds.
- [x] All 648 saved physical outcomes pass independent reconstruction audit.
- [x] Owner manual review accepted October 3, 2026 (“Manual review passed”);
  selected artifact hashes are recorded in `docs/level5_pick_acceptance.json`.

All three runs complete 200 epochs / 14,400 updates. Full candidate validation
selects epochs 138/20/80, each at 192/192. Each fresh confirmation passes 212/216,
with zero safety violations, worst-family success 94.4444%, drop rate 1.85185%
and passing jerk. One small-block reset drops under all four conditions for all
seeds; those failures remain visible and are included in owner review. These are
54 unique fresh starts per seed, not unseen-instance or perception evidence.

Full regression: 808 passed, one skipped; focused checks: 46 passed. Ruff,
original locks, archive hashes, LFS attributes and model/report/code integrity
pass. Exact commands, methods, failed attempts, selected digests and manual
pass/fail criteria are in `docs/level5_pick_repair.md` and its verification JSON.
Owner manual acceptance is recorded; no further manual check is needed.
Last Completed is 5.3A; Next Target is 5.4. No 5.4, registry, deployment or later
checkpoint work is started. Saved training artifacts remain local under
`outputs/level5/pick_repair/v7/` (Git-ignored); this commit publishes source,
recipes, evidence documentation and dataset archives, not the trained weights.

---

## Level 5.4 — State-Grounded Place-Held-Object Policy

**Complete — October 3, 2026, with negative state qualification.** All three
seeds completed the locked 200-epoch budget and full candidate validation.
Validation selected epochs 20, 56 and 20. The one-time held-out state tests
achieved 4/216, 0/216 and 12/216 successes, with premature release and safety
failures across all seeds. Every target type was evaluated. All 16 reported
successes pass independent saved-state physical reconstruction; this does not
change the failed numerical gates. The full suite passes 816 tests with one
skip, focused checks pass 18, and Ruff/artifact/source/recipe integrity pass.
See [`level5_place_results.md`](level5_place_results.md) and
[`level5_place_verification.json`](level5_place_verification.json). No manual
viewer gate is triggered because no selected policy passes numerical gates.
No perception, registry, composition, deployment or later checkpoint work is
claimed.

Apply `docs/level5_training_guidance.md`. Freeze a place-specific recipe with
held-object-to-target geometry, grasp offset, clearance, retention and causal
transport/lower/release/settle information. Keep finger release outputs active;
do not inherit reach's fixed joints. Validate object pose and post-release
settling independently of the commanded hand endpoint.

### Goal

Train and qualify `place_held_object` from a verified held-object precondition
through transport, placement, controlled release, and settling.

### Files

```text
configs/level5/skills/place_held_object.yaml (frozen historical parent)
configs/level5/place_policy_v1.yaml and .sha256
dexvision/learning/place_policy.py
dexvision/evaluation/place_policy.py
dexvision/apps/train_skill.py and evaluate_skill.py (version dispatch)
dexvision/learning/train_skill.py (place causal-phase sampler support)
docs/level5_place_results.md and level5_place_verification.json
tests/test_place_skill_training.py
tests/test_place_skill_rollout.py
tests/test_level5_learning_plan.py (independent recipe lock inventory)
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
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/place_policy_v1.yaml --seed 5001 --dry-run
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/place_policy_v1.yaml --seed 5001
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/place_policy_v1.yaml --select-candidates outputs/level5/place_54_v1/locked/seed_5001 --split validation --track state --output-dir outputs/level5/place_54_v1/locked/validation_5001
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/place_policy_v1.yaml --checkpoint outputs/level5/place_54_v1/locked/seed_5001/selected.pt --split test --track state --output-dir outputs/level5/place_54_v1/locked/test_5001
conda run --no-capture-output -n dexvision pytest -q tests/test_place_skill_training.py tests/test_place_skill_rollout.py
```

Repeat train, full candidate selection and the single selected test for seeds
5002 and 5003, with new output directories. The complete loop, evidence paths
and integrity commands are in `docs/level5_place_results.md`. Test runs are
single-use and must not be repeated under new paths.

### Pass criteria

```text
[x] Execution rejects missing, multiple, wrong, or stale held-object state
[x] Transport, place, release, and settle phases are independently measurable
[x] Every required target and held-out goal region is evaluated
[x] Misplacements and unstable post-release states remain failures
[x] Required success/release/safety gates pass or status is experimental/failed;
    status is **failed** for all three selected policies
```

Manual verification is required only after numerical gates pass. None of the
three selected policies passes, so no viewer review is required or accepted in
this checkpoint. For a later numerically passing version, run the guarded
viewer on at least one target of each type and stop for owner confirmation.
Pass when the same held object is transported, released inside tolerance,
remains stable, and the reported result matches the scene.

---

## Level 5.4A — Place-Held-Object Policy Repair

**Complete (October 5, 2026).** The owner confirmed the prescribed V25 manual
viewer review with "Passed". The acceptance receipt is
[`level5_place_acceptance.json`](level5_place_acceptance.json). Keep
the completed negative V1 result above intact. This checkpoint repairs only
the place source, policy and qualification path. Method evidence and current
limits are in [`level5_place_repair.md`](level5_place_repair.md).

Earlier failed versions and their then-pending gates are preserved below as
history. The final V25 evidence and acceptance conclude this checkpoint section.

### Required work

```text
[x] Qualify a versioned ordinary-physics place teacher on exposed development resets
[x] Publish any needed additive, immutable train/validation source with split and physical audits
[x] Freeze a place-specific learned feature/action/decoder recipe and fresh confirmation resets
[x] Train all three seeds through the frozen budget and select by full closed-loop validation
[x] Preserve the V15 one-time numerical confirmation as historical evidence;
    its later independent physical audit failed
[x] Run each physically screened final repair bundle once on fresh confirmation;
    pass every frozen 5.4 state gate
[x] Independently audit saved physical states and artifact/recipe integrity
[x] Obtain owner manual viewer acceptance after numerical gates pass
```

Do not use the V1 held-out outcomes for repair selection, weaken its place
success/safety limits, rewrite Level 4 or failed V1 artifacts, or begin 5.5.
Validation and development resets already exposed by prior experiments are
development evidence only. A fresh reset does not make a previously exposed
family or target type unseen. An unassisted scripted pick may establish the
verified held-object precondition; it is disclosed and excluded from the place
policy's actions and metrics. Failed setup and policy attempts stay counted.

The manual gate remains conditional. Once all numerical and independent
physical checks pass, replay at least one saved attempt for each of receptacle,
planar-zone and placement-slot targets. Pass only when the correct held object
is transported, released within the frozen tolerance, remains stable, and the
scene matches the saved report. Stop for owner confirmation before checking
completion or advancing the status file to 5.5.

### Current diagnostic and software commands

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher --config configs/level5/place_teacher_v2.yaml --output-dir outputs/level5/place_repair/teacher_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher --config configs/level5/place_teacher_coverage_v1.yaml --output-dir outputs/level5/place_repair/teacher_coverage_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher --config configs/level5/place_teacher_coverage_v2.yaml --output-dir outputs/level5/place_repair/teacher_coverage_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher --config configs/level5/place_teacher_coverage_v3.yaml --output-dir outputs/level5/place_repair/teacher_coverage_v3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher --config configs/level5/place_teacher_search_coverage_v1.yaml --output-dir outputs/level5/place_repair/teacher_search_coverage_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher --config configs/level5/place_teacher_search_coverage_v2.yaml --output-dir outputs/level5/place_repair/teacher_search_coverage_v2
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment freeze --config configs/level5/place_data_v1.yaml --output-dir outputs/level5/place_repair/data_v1
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment freeze --config configs/level5/place_data_v2.yaml --output-dir outputs/level5/place_repair/data_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment collect --config configs/level5/place_data_v2.yaml --output-dir outputs/level5/place_repair/data_v2 --split train
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment collect --config configs/level5/place_data_v2.yaml --output-dir outputs/level5/place_repair/data_v2 --split validation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment verify --config configs/level5/place_data_v2.yaml --output-dir outputs/level5/place_repair/data_v2
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment release --config configs/level5/place_data_v2.yaml --output-dir outputs/level5/place_repair/data_v2 --release-dir datasets
shasum -a 256 -c datasets/level5-place-unassisted-v2.tar.gz.sha256
conda run --no-capture-output -n dexvision pytest -q tests/test_place_repair_teacher.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check dexvision tests
```

Those output directories are immutable evidence already created in this
workspace. Reproduction must use new directories. The first teacher screen
passes; the three frozen static all-cell screens fail (15/30, 22/30 and 26/30).
The frozen copied-task search screen passes **29/30** new `850260900` resets
with 9/10 worst-family success and zero safety violations; the one premature
release remains a failed attempt. This clears the scripted-teacher feasibility
item only. It is not learned-policy qualification, and the 5.4A checkpoint
remains open.

The additive data V1 collection stopped after a board-margin failure, so it
has no release. Search V2 passed 59/60 fresh nominal/boundary resets with
zero selected live safety violations. Data V2 has new frozen assignments;
train passed 140/144 and validation 47/48 with zero live safety violations.
Independent saved-state reconstruction passed all 187 successful payloads;
all five failed resets remain excluded and counted. The new immutable archive,
checksum and manifest are under `datasets/level5-place-unassisted-v2*`.
Versioned learned V2/V3/V4/V5/V6 recipes and separate fresh confirmation resets
are frozen. V2 direct-delta control drifted and released prematurely in a partial
validation diagnostic; V3's fixed hand waypoint stopped too high in two
single-reset diagnostics. All three seeds finished the 200-epoch budget for
each recipe, and those negative weights remain local. V4 uses live object
feedback and training-only release geometry, but every candidate failed a
worst-family gate on completed family attempts; its resolved source transport
step was mapped too slowly. V5 corrected that step to 0.005 m and selected all
three seeds after full validation, but a new all-target development screen
failed the zero-safety and settling gates. V6 freezes a bounded held-object
waypoint, cuboid release at 0.0245 m within the original 0.025 m limit, and
slower cylinder opening. Its separate nominal and boundary all-target
development screens each passed 27/30 with zero safety and all numerical
gates. All three V6 seeds completed 200 epochs / 12,200 updates. Full V6
validation selected all three, but seed 5002's selected epoch missed the
settling gate and failed a separate all-target safety screen. The frozen V7
all-target selector rejected both eligible seed 5002 candidates for safety;
no three-seed bundle or confirmation resulted. V8 reuses the exact V6 learned
weights and freezes a measured command/object lag hold, target-specific
cuboid opening and a conditional geometric release for an outward-moving
cuboid near a receptacle edge. Its exact decoder passed 53/60 on exposed
all-target development resets with zero safety and all original gates.
Complete V8 validation and its distinct fresh all-target selection screen
selected seeds 5001 and 5002, but neither eligible seed-5003 candidate passed
all gates; no three-seed bundle resulted. V9 freezes slower opening for
nonplanar cuboids, cylinders and pucks on the exact V6 weights. Predeclared
epochs 80, 20 and 20 passed all gates on the now-exposed V8 selection resets
with zero safety and then passed full V9 validation. The distinct fresh
all-target screen passed seed 5001 but rejected seeds 5002 and 5003 on the
unchanged settling gate. No V9 three-seed bundle exists; the one-time
confirmation resets remain unused.

V10 added a target-aware 0–3 mm palm rise at first release on the same
learned weights. It passed two complete exposed all-target development
matrices for all three seeds with zero safety, but full validation rejected
seed 5002 on the original worst-family gate (36/64 cuboid successes). No V10
selection or confirmation was run. V11 freezes zero extra rise for cuboid
placement slots after that diagnosed bounce regression, with all other V10
controls and the frozen gates intact. All three V11 candidates passed complete
validation. Its fresh 1060 all-target selection selected seed 5001 but
rejected seeds 5002 and 5003 on the unchanged settling gate (52/58 and 49/58
settled nominal placements). The original 930 one-time confirmation remains
unused. V12 froze a targeted initial descent waypoint for short-grasp heavy
pucks on the inspection pad and a measured-state opening choice for cuboids
over the left receptacle. Its targeted exposed numeric prediction was rejected
by an independent saved-state audit: one nominal success had 15 pre-release
hand-table-contact frames. Partial V12 validation was stopped without fresh
selection. V13 gates an earlier physical release on the measured negative
lateral-drift state; six exposed low-offset pad attempts passed independent
development audits. Complete V13 validation passed all three seeds, but its
fresh 1100 selection rejected seeds 5001 and 5002 for one object-workspace
violation each, and seed 5003 for 51/59 settled nominal placements. No V13
selected bundle or 930 confirmation exists. A V14 wrist guard now passes
targeted exposed 1100 development checks for all three seeds, with all changed
attempts passing saved-state audit. Complete V14 validation passed all seeds,
but the separate fresh 1120 selection passed only seed 5001: seed 5002 had
one object-workspace violation, and seed 5003 settled 51/59 nominal placements.
No three-seed V14 bundle or reserved 930 confirmation exists. The 5.4A
checkpoint stays open. A V15 measured-state opening and collision decoder
passes targeted exposed 1100/1120 development checks for all three seeds.
Complete V15 validation, a distinct unused 1140 all-target selection, the
original 930 confirmation, independent audit and owner viewer acceptance
remained required before 5.4A could be completed. Complete V15 validation,
fresh 1140 selection and the one-time 930 confirmation passed all frozen
numerical state gates for all three seeds with zero recorded safety violations.
The independent saved-state audit then rejected all three selected bundles:
8, 2 and 7 distinct reported successful resets had pre-opening hand-table
contact. The 930 matrix is consumed and exposed, and V15 is not qualified.
The failed reports are retained. A further source-grounded repair needs a
new frozen selection and confirmation matrix and passing physical audit
before owner viewer review. The 5.4A checkpoint stays open.
V2 source coverage was audited: training has only 18 left-bin/pad/slot-A
cells, validation only six slot-B cells, and neither contains any right-bin
cell. A checksum-locked V3 data plan freezes 240 training and 120 validation
resets, covering all 30 cells in both splits with disjoint resets and
simulator sessions. Collection passed 237/240 training and 117/120
validation attempts with zero live safety; all six failures remain counted
without payloads. Independent saved-state audit passed all 354 successful
trajectories. The immutable V3 archive, checksum and manifest are under
`datasets/level5-place-unassisted-v3*`, with LFS and checksum verification.
V2 and every failed policy report remain unchanged. The V3 source alone
does not qualify a learned policy.

The frozen V3 source commands are:

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment freeze --config configs/level5/place_data_v3.yaml --output-dir outputs/level5/place_repair/data_v3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment collect --config configs/level5/place_data_v3.yaml --output-dir outputs/level5/place_repair/data_v3 --split train
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment collect --config configs/level5/place_data_v3.yaml --output-dir outputs/level5/place_repair/data_v3 --split validation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment verify --config configs/level5/place_data_v3.yaml --output-dir outputs/level5/place_repair/data_v3
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment release --config configs/level5/place_data_v3.yaml --output-dir outputs/level5/place_repair/data_v3 --release-dir datasets
shasum -a 256 -c datasets/level5-place-unassisted-v3.tar.gz.sha256
```

Verification and release passed after both full split reports were complete.
The publication command requires the same source safety and success gates.

V16 separately freezes a source-calibrated measured-orientation and
hand-clearance overlay on the exact V6 epochs 80, 20 and 20. The 930, 1100
and 1140 exposed nominal development matrices pass the original numerical
gates and independent saved-state audits after narrow per-seed contact fixes.
The checksum-locked policy and selection recipes reserved fresh 1160
all-target selection and 1180 one-time confirmation resets. Full V16
validation passed all state gates for all three seeds (180/192, 172/192,
172/192; zero recorded safety). The fresh 1160 screen rejected all three:
seed 5001 failed the 90% settling gate (51/59) and one physical audit;
seeds 5002 and 5003 passed numerical gates but had one and three
reported-success pre-opening hand-table contacts on independent saved-state
audit. No V16 bundle was selected. The 1180 confirmation remains unused and
the manual viewer gate is not triggered.

V17 uses successful V3 source geometry to freeze a narrow engineered wrist
and pad-clearance overlay around the exact V2-trained V6 weights. Four
packaged trajectories matched exposed 1160 diagnostic actions exactly; those
replays are development evidence. Full V17 validation passed all state gates
for three seeds (172/192, 164/192, 168/192, zero recorded safety). The fresh
1260 all-cell selection passed all numerical and independent saved-state
physical gates for each seed: 52/60, 51/60 and 52/60, with zero physical
audit failures among 180 attempts. Epochs 80, 20 and 20 are selected. The
1260 screen is exposed. One-time 1280 confirmation passed the numerical
gates for seeds 5001 and 5002 at 212/240 each, but seed 5003 failed the
90% settling gate at 192/220 settled nominal placements (87.3%). Independent
saved-state audits passed all 720 attempts as physically consistent; that
does not satisfy the failed numerical gate. V17 is not qualified, the 1280
matrix is exposed and owner viewer review is not triggered. V17 learned
weights still come from V2, which lacks right-bin source; V3 informed
engineered controls only.

V18 freezes a seed-5003 large-block geometric opening fallback within the
unchanged 0.0245 m cuboid release limit after V17's consumed 1280 numerical
failure. Exposed 1280 targeted replays converted four of five distinct
unreleased pad/slot attempts to settled placements; the fifth remained a
failure outside the bound. All changed 1160/1260/1280 diagnostics passed
independent saved-state physical audit, and one packaged V18 trajectory
matched the diagnostic actions exactly. These are development evidence.
Fresh 1300 all-cell selection and 1320 one-time confirmation are frozen;
complete V18 validation passed all numerical gates (172/192, 164/192 and
172/192, zero recorded safety). Fresh 1300 selection rejected all seeds on
the unchanged settling gate (51/59, 47/56 and 53/59). Seed 5001 also had
one saved-state physical audit failure. No V18 bundle was selected; 1300 is
exposed, 1320 confirmation is unused, and owner viewer review is not
triggered. V18 is not qualified.

V19 now freezes a separate three-seed learned recipe on the audited V3
all-cell source (237 training and 117 validation episodes). It keeps the
V18 bounded decoder, original physical and numerical gates, causal features,
fixed scales, phase-balanced sampling and a full 200-epoch budget; V6 weights
initialize decoder controls only and are replaced before V19 inference.
The old V2-specific validation helper rejected V3's 120 source-validation
resets before training began, so V19 freezes new disjoint 1330 all-cell
policy-validation resets (60), 1340 selection (60) and 1360 one-time
confirmation (60). Source dry run and focused ownership tests pass. All
three seeds completed 200 epochs and 20,400 updates; offline best epochs
were 199, 195 and 173, while full closed-loop validation compares frozen
candidates 20, 80 and 200. All nine 1330 validation reports are complete;
the frozen ranking chose epochs 20, 20 and 200, passing every numerical
gate at 220/240, 220/240 and 216/240 with zero recorded safety. Seed-5003
epochs 20/80 and seed-5001 epoch 80 remain failed evidence. Fresh 1340
all-cell selection rejected all three seeds on the unchanged settling gate:
50/59, 51/58 and 50/58 nominal placements settled, below 90%. Aggregate
successes were 50/60, 51/60 and 50/60, with zero recorded safety. Independent
saved-state audit found one reported-success pre-opening hand-table contact
for seeds 5001 and 5003; seed 5002 had none. No V19 bundle was selected.
The 1340 resets are exposed, 1360 confirmation is unused, and owner viewer
review is not triggered. The 5.4A checklist remains open.

V20 freezes a separate object-calibrated decoder on the exact V19 learned
weights. Six release waypoint vectors come from V3 successful **training**
episodes only; the recipe recomputes and verifies them from the archive.
Bounded supported descent applies only while a cylinder remains held above
target during opening, and a 0.073 m hand-height guard applies only to a
small block over slot B. The exposed 1340 development matrix passed all
frozen numerical gates for epochs 20, 20 and 200 (54/60 successes each;
54/59, 54/58 and 54/58 settled nominal placements) with zero saved-state
physical audit failures. Complete 1330 V20 validation passed all original
numerical gates at 220/240, 212/240 and 216/240 successes, zero recorded
safety. Fresh 1380 all-cell selection passed seed 5001 (54/60 successes,
zero audit failures), but rejected seed 5002 for two reported-success
pre-opening hand-table contacts and seed 5003 for one object-workspace
violation. The 1380 matrix is exposed. No three-seed V20 bundle was
selected; one-time 1400 confirmation, independent confirmation audit and
owner viewer acceptance remain unstarted. No 5.4A completion checkbox
changes from these results.

V21 was a copied-state transport preview layered on exact V19
weights and the V20 decoder. It is restricted to two source-exposed risk
cells and does not alter finger opening or release. All twelve affected-cell
attempts on exposed 1380 resets passed a disposable development replay and
independent saved-state audit; mixing untouched cases predicts passing
numerical gates for all three seeds. The checksum-locked implementation adds
action-bound filtering and passed an integrated contact-case replay. Complete
1330 validation passed all original numerical gates at 212/240, 208/240 and
212/240 successes with zero recorded safety. Fresh 1420 selection passed at
55/60, 57/60 and 54/60 successes with all original numerical gates, zero
recorded safety and 180/180 independent physical audits. Exact epochs 20,
20 and 200 were selected. One-time 1440 confirmation rejected V21: seeds
5001 and 5002 each settled 204/228 nominal placements (89.47%), below the
unchanged 90% gate; seed 5003 passed numerically at 208/228 (91.23%). All
three had zero recorded safety. Independent 1440 saved-state audit passed
720/720 attempts.
No V21 bundle qualifies, and the 5.4A checklist remains open.

V22 freezes a narrow 2 mm held short-cylinder opening descent over the exact
V21 policy for the two bin targets. On exposed 1440 resets it repaired nine
bounced attempts, with all twelve cylinder-bin attempts physically audited;
exposed 1420 selection totals were preserved and all twelve 1330 cylinder-bin
attempts passed audit. Fresh 1460 selection and 1480 one-time confirmation
resets were frozen. Complete 1330 V22 validation passed all original gates at
216/240, 212/240 and 216/240 successes with zero recorded safety. Fresh
1460 selection then rejected V22: seeds 5001 and 5003 missed the original
settling gate at 52/58 and 50/58, while seed 5002 passed numerical gates at
53/58. Seeds 5001 and 5002 each had one reported-success pre-opening
hand-table contact on the same small-block slot-B reset. No three-seed bundle
was selected; 1460 is exposed and 1480 confirmation remains unused. The 5.4A
checklist stays open.

V23 freezes a bounded follow-up over exact V19 learned weights, V20 object
calibration and V21 copied-state transport preview. The small-block slot-B
preview intervenes only when its baseline predicts hand-table contact; while
held above bin targets, short-cylinder opening descends 3 mm per action above
8 mm object clearance and heavy-puck opening descends 1 mm per action above
5 mm clearance. Every changed attempt on complete exposed 1330, 1420, 1440
and 1460 matrices passed saved-state audit; merged development estimates pass
all original numerical gates for all three seeds on each matrix. Two frozen
integrated replays passed physical audit. Complete 1330 V23 validation passed
all original gates at 204/240, 200/240 and 204/240 successes, with 92.7%,
90.9% and 92.7% settling and zero recorded safety. Fresh 1500 selection
passed all numerical gates at 54/60, 54/60 and 53/60 successes, zero
recorded safety and settling above 96%, but seed 5002 failed one independent
saved-state audit: a reported-success large-block/right-bin placement had
11 pre-opening hand-table contact samples. V23 is rejected, 1500 is exposed,
1520 confirmation remains unused, and the 5.4A checklist stays open.

V24 freezes a verified setup wrist hold only during large-block/right-bin
transport, retaining exact V19 learned weights and V23 controls. The V23
failed seed rotated the wrist and moved the block upward relative to the
hand; the hold lets it qualify for release before forearm contact. All six
exposed 1500 right-bin attempts and six 1330 right-bin validation probes
succeed with passing physical audits. Six 1460 probes pass audit with one
counted bounce per seed. The integrated V24 replay matches diagnostic actions
and states exactly. Focused tests pass 16/16; Ruff passes. Distinct 1540
selection and 1560 one-time confirmation resets are frozen and unused.
Full V24 regression passed 888 tests with one skip. Complete validation
passed every original numerical gate at 204/240, 200/240 and 204/240
successes, zero recorded safety and 92.7%, 90.9% and 92.7% settling.
Fresh 1540 selection passed every numerical gate at 53/60, 53/60 and
51/60 successes, zero recorded safety and 96.4%, 94.6% and 92.7%
settling. Its physical audit passed 179/180 attempts: seed 5002's
large-block/inspection-pad reported success had one pre-opening hand-table
contact sample. V24 is rejected, 1540 is exposed and 1560 confirmation
remains unused. No qualification or manual acceptance is claimed.

V25 freezes a large-block/inspection-pad geometric release at the first held
measured distance inside the unchanged 0.0245 m cuboid bound. Existing
opening rate, release rise, V24 wrist hold and exact V19 weights remain.
All six exposed 1540 pad attempts settle with passing physical audits; six
1330 probes preserve a counted timeout and an audited success per seed.
Integrated actions and states match the passing diagnostic exactly. Focused
tests pass 18/18; Ruff passes. Distinct 1580 selection and 1600 one-time
confirmation resets were frozen before use. Full regression passed 890 tests
with one skip. Complete validation passed every numerical gate at 204/240,
200/240 and 204/240 successes, 92.7%, 90.9% and 92.7% settling, and
zero recorded safety. Fresh 1580 selection passed every numerical and
physical gate at 53/60, 53/60 and 52/60 successes, zero recorded safety
and 180/180 passing saved-state audits. Exact epochs 20, 20 and 200 are
packaged; their model tensors and manifests match the frozen candidates.
One-time 1600 confirmation passes every frozen numerical gate for all three
seeds at 196/240, 204/240 and 192/240 successes; 94.2%, 94.4% and
92.3% settling; worst-family success 65%, 75% and 65%; premature release
8.33% each; zero recorded safety violations. All 720 independent physical
audits pass, including counted failures. Source/config pins, artifact indexes,
exact selected candidate tensors/manifests, confirmation markers and all
three guarded viewer inputs verify. Evidence is recorded in
`docs/level5_place_repair_verification_v25.json`. Owner manual acceptance was
confirmed on October 5, 2026 with "Passed" and is separately bound to the
recipes, selected bundles, confirmation reports and audits in
`docs/level5_place_acceptance.json`. Historical automated receipts retain
their then-pending manual status. Checkpoint 5.4A is complete with bounded
state qualification; 5.5 has not started.

---

## Level 5.5 — State-Grounded Push, Press, and Optional Dial Policies

**Current outcome:** Checkpoint 5.5 is complete after owner push viewer
acceptance ("Passed", October 13, 2026). V8 push passes all three state gates;
accepted V1 press is unchanged. `docs/level5_push_acceptance.json` binds the
owner statement to the exact bundles and four prescribed review cells.
Last Completed: 5.5. Next Target: 5.6, unstarted. Dial is absent, not promoted.

**V1 automated work finished; press owner accepted; push repair requested
(October 13, 2026).**
Separate push/press V1 recipes implement contact-phase standalone behavior
cloning on the unchanged immutable release. All six 200-epoch budgets and full
candidate validation/selection completed. One-time held-out press evaluations
pass every state gate at 120/120 per seed; push fails at 40/128, 56/128 and
44/128, with workspace/wrong-contact failures and zero flat-puck successes.
Every saved-state physical consistency audit passes, including failures.
Dial is explicitly absent, not promoted. Full regression: 900 passed, one
skipped; focused: 20 passed; Ruff and artifact/selected-weight checks pass.
The owner confirmed "Press passes"; the separate acceptance receipt is
`docs/level5_press_acceptance.json`. Failed push does not trigger review. The
owner explicitly required push to pass too, leaving this checkpoint and its
checklist box open at that stage. Repair must preserve V1, freeze a separate recipe
and fresh confirmation, and retain the original state/safety gates. The accepted
press runtime stays unchanged. V2 completes three full training budgets but fails parked-puck disturbance;
V3 preserves those weights and fixes the ordinary-physics entry setup.
Every V3 validation candidate passes, but its one-time confirmation fails
success/worst-family gates (92/128, 96/128, 68/128; zero safety, all physical
audits passing). Those failures and consumed resets remain preserved.
V4 completes all training/validation but fails the three-seed requirement:
seed 5003 has zero cuboid success, and V4 confirmation remains unopened.
V5 separately freezes entry-relative geometry, removing constant family/height
cues, and one learned signed along rate after whole-interval lateral invariance
and 4/4 long-cuboid ordinary-physics feasibility evidence. New disjoint
selection/confirmation starts are frozen; all three 200-epoch budgets finish.
V5 focused tests pass 29 tests, feature parity is exact, and full regression
passes 909 tests with one skip. Every V5 candidate matrix finishes, but all
selected seeds fail worst-family gates (140/152, 25% cuboid success each);
confirmation is unopened and all failures remain preserved. V6 adds
a separately frozen causal physical-goal-success latch after a source-only
retraction audit, retains every original label/gate and freezes new disjoint
selection/confirmation before training. V6 focused tests pass 30 tests; full
training finishes. Every epoch-20 complete selection matrix passes at 148/152,
75% cuboid success, zero safety and all physical audits passing. All
candidates finish; epochs 80/20/20 are selected. Final regression passes
910 tests with one skip. One-time fresh V6 confirmation fails at
100/128, 100/128 and 96/128 successes, with cuboid rates below 65%; safety
is zero and all 384 physical audits pass. Consumed failures are preserved.
V7 separately freezes physical predicates in place of continuous tracking
and velocity proxies. It preserves signed learned action, causal latch and
all physical gates, with 46-reset/184-attempt candidate selection (original
30 plus 16 fresh all-cell resets) and disjoint 32-reset confirmation.
All attempts, including one failed scripted confirmation setup, remain counted.
V7 focused tests pass 31 tests; all three 200-epoch / 8,400-update budgets
finish. Full regression passes 911 tests with one skip; Ruff passes. All
three epoch-20 selection matrices pass at 184/184 with zero safety and
passing physical audits. All 12 complete candidate matrices pass (2,208/2,208 successes), and
validation selects epochs 200/61/80. One-time V7 confirmation finishes at 124/128, 112/128 and 116/128.
All success thresholds pass, but seeds 5002/5003 record 12/8 workspace
violations; all 384 physical audits pass. The failed three-seed result is
preserved in `docs/level5_push_repair_v7_failed.json`, and those resets are
consumed. V8 preserves exact full-budget V7 candidates with a separately
frozen causal entry/settling and original-workspace decoder overlay.
Every alias, hold and bound is recorded and independently audited. Source
advancing-before-contact and whole-interval workspace identity audits pass.
V8 focused tests pass 34 tests; Ruff passes. New 22555000-series all-cell
selection (32 fresh plus original 30; 62 unique/248 attempts) and disjoint
22755000-series confirmation are frozen before overlay selection.
All 62 selection and 32 confirmation setups freeze ready. An original
validation smoke passes with zero safety; all four engineered dwell holds
independently qualify. Final V8 regression passes 914 tests with one skip; focused tests pass 34
and Ruff passes after the reporting fix. All 12 complete candidate matrices pass across 2,976 attempts, with every
physical/decoder/phase audit passing and zero executed safety. Validation
selects epochs 200/61/20 at 244/248, 244/248, 248/248 successes. The timeouts
and 36/176/0 selected workspace-bound samples remain disclosed. One-time fresh V8 confirmation passes every original state gate at
116/128, 116/128, 120/128 successes, with worst-family rates 81.25%, 81.25%,
87.5%, zero executed safety/wrong contacts and all 384 physical/decoder/phase
audits passing. The 12/12/8 goal timeouts stay counted. Confirmation records
348/532/876 engineered workspace-bound samples and 468/476/480 physical
settling holds; these are not learned safety/terminal control. Verification
in `docs/level5_push_repair_verification_v8.json` passes exact tensor,
source/code/config/reset and checksum bindings, observed physical contact
for every success, and unchanged accepted press bundles. Final regression:
914 passed, one skipped; focused 34; Ruff passes. Owner push viewer
acceptance is recorded separately in `docs/level5_push_acceptance.json`;
original automated verification retains its historical pending status.
Last Completed is 5.5; Next Target is 5.6, unstarted.
All original gates and immutable data stay unchanged. See `docs/level5_push_repair.md` and the failed-result receipts.
No 5.6 work is started. The latest owner acceptance authorizes commit and push.
See [`level5_contact_results.md`](level5_contact_results.md) for the audited
method, previous pick/place adjustments, exact commands and manual criteria.
No completed checkpoint or future feature is changed by this work.

Apply `docs/level5_training_guidance.md` and freeze separate recipes for push
and press. Push needs object-to-zone and hand-to-object geometry plus contact
and object-motion feedback; hand arrival alone is not success. Press needs
fixture alignment, measured depth and causal press/dwell/retract state, with
both pressing and release directions. Do not copy reach's positive planar
progress restriction into either skill without evidence. The earlier
expert-assisted push probe does not qualify a standalone learned controller.

### Goal

Train and qualify the remaining contact skills through the common training and
evaluation interfaces.

### Files

```text
configs/level5/skills/push_object_to_target.yaml
configs/level5/skills/press_button.yaml
configs/level5/skills/rotate_dial.yaml, only if promoted
configs/level5/push_policy_v1.yaml and push_policy_v1.sha256
configs/level5/press_policy_v1.yaml and press_policy_v1.sha256
configs/level5/push_policy_v2.yaml and push_policy_v2.sha256
configs/level5/push_policy_v3.yaml and push_policy_v3.sha256
configs/level5/push_setup_v3.yaml and push_setup_v3.sha256
dexvision/learning/push_repair.py
dexvision/evaluation/push_repair.py and push_settled.py
tests/test_push_repair.py
docs/level5_push_repair.md and versioned verification/failure receipts
dexvision/learning/contact_policy.py
dexvision/evaluation/contact_policy.py
dexvision/apps/train_skill.py and evaluate_skill.py (recipe dispatch only)
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
python -m dexvision.apps.train_skill --config configs/level5/push_policy_v1.yaml --seed 5001 --release-root .
python -m dexvision.apps.train_skill --config configs/level5/press_policy_v1.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v1.yaml --select-candidates outputs/level5/push_55_v1/seed_5001 --split validation --track state --release-root . --output-dir outputs/level5/push_55_v1/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/press_policy_v1.yaml --select-candidates outputs/level5/press_55_v1/seed_5001 --split validation --track state --release-root . --output-dir outputs/level5/press_55_v1/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v1.yaml --checkpoint outputs/level5/push_55_v1/seed_5001/selected.pt --track state --release-root . --output-dir outputs/level5/push_55_v1/test_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/press_policy_v1.yaml --checkpoint outputs/level5/press_55_v1/seed_5001/selected.pt --track state --release-root . --output-dir outputs/level5/press_55_v1/test_5001
conda run -n dexvision pytest -q tests/test_push_skill_rollout.py tests/test_press_skill_rollout.py
```

Run equivalent dial commands/tests only when the Level 4 manifest promotes it.
Repeat train/selection/test for seeds 5001, 5002 and 5003. Existing output
directories must not be overwritten; use new paths when reproducing.

### Pass criteria

```text
[x] Push and press each complete three seeded training runs and frozen rollouts
[x] Physical task state recomputes every success and failure
[x] Wrong contact, contact loss, overshoot, and object-off-board failures remain visible
[x] Push and press both pass required state success and safety gates (owner-required; V8 push, accepted V1 press)
[x] Optional dial is qualified, experimental, failed, or explicitly absent
```

Manual verification is required after numerical gates pass. View one
stratified push and one press rollout; add one dial rollout only if promoted.
Pass when contact and terminal state visibly match each report. Stop for user
confirmation.

Owner acceptance is complete as of October 13, 2026: "Press passes" and then
"Passed" for the prescribed four-cell V8 push review. Separate receipts are
`docs/level5_press_acceptance.json` and `docs/level5_push_acceptance.json`.
The original automated receipts and all failed attempts remain unchanged.
Checkpoint 5.5 is complete; 5.6 remains unstarted.

---

## Level 5.6 — Detector, Tracker, and Metric Pose Baseline

Owner-authorized publication: code, compact V25/V29 model bundles, manifests,
checksums, protocols, acceptance receipt and status docs. The large dataset
archives/parts remain local to respect the reported 10 GiB LFS allowance.
Their complete contents are preserved; a clone alone lacks these new payloads.
See `datasets/README.md` for local restoration. Original scientific bindings
and acceptance remain valid. This publication does not start 5.7.

**Current outcome: 5.6 complete; V29/fresh V8 automated gates and owner manual acceptance pass.**
The owner confirmed the held-out review passed on October 16, 2026.
Last Completed is **5.6**; Next Target is **5.7 (unstarted)**, on branch `main`.
`docs/level5_perception_acceptance.json` binds this owner-reported acceptance to
the exact frozen artifacts; no agent visual observation is claimed. Historical
automated receipts retain their then-pending manual status. No 5.7 work is
authorized by this completion request. The earlier V8 validation diagnostic
review remains separately accepted.

V29 retains exact V25 selected seed 11/source epoch 3,000 current-RGB MLP weights,
TRAIN transforms, fixed 8 px / 8 px / 10 mm output scales and calibrated ray decoder,
with zero new optimizer updates. All three original current candidates and all
six source fits remain preserved. Cuboid association uses the calibrated
projection of the current learned body center, with a cuboid-only uncertain
replacement-ID guard and history rollback. Other classes keep original birth/
visible-box association. Published masks, boxes and positions are unchanged.
Original tracker limits and all original visual gates remain unchanged.
Added identity coverage requires 99% unambiguous primary matches overall and 95%
in every supported primary class/phase; V29 passes both.

Validation: 5,923/5,923 primary matches and positions, zero identity switches, median
0.904 mm/p95 3.414 mm;
all 545 six-phase queries pass (worst phase p95 14.082 mm).
Fresh V8: 512 ordinary-physics source resets/26,687 RGB frames,
primary recall 100.0000%, median 0.954 mm,
p95 3.675 mm, 14,244 positions and
0 primary ID switches. All original/global/core/primary, supported phase,
position coverage, added identity coverage, and mandatory release/settle gates
pass independently. Release: 46 visible from
13 physical sources;
settle: 24 visible from
13 physical sources.

Full software: 1,177 passed/1 skipped; focused 64, Ruff and
whitespace pass. Actual 64 validation truth-isolation queries and 48 cached/uncached runtime
paths, exact accepted-transform oracles, portable RGB relocation, archive/member
readbacks, source/body-origin/schema audit, all 512 video decodes, typed/raw
parity and actual consumed-protocol retry rejection pass. The review has 16
panels and 36 fixed-plus-worst temporal clips.
Evidence: `docs/level5_perception_verification_v29.json`.

Model semantic SHA: `3a25ea7b49cc05ce7d67e89226a7e8c231a7feaea945d4863e7ed56a5cf5cadd`.
Active current-weight SHA: `0ba58825982fbfd714f028c3b8a9c4195e9f4ec5befe678e5273025863bbdb85`.
The baseline V2 and fresh V2–V7 consumed tests stay permanently closed, with all
archives, data, failed/positive candidates, reports and reservations preserved.
V7 failed one primary cuboid/release ID switch; V26/V27 failed independent identity
coverage. V28 positive validation and the independently frozen V29 guard plan
are preserved. Fresh V8 is now consumed too and cannot be retried or used for
training/selection. All source outcomes were retained; no test tuning or reuse.

The availability-aware schema is shared by truth and inference. Physical
linear/angular motion, full orientation, semantic identity, robot/fixture
telemetry and contacts remain explicitly unavailable and fail closed. All five
accepted inferred-control tracks remain unsupported. Qualification scope is
calibrated simulator pixel-center images on exposed task families/cells;
no real-camera, unseen-family, hardware, full-feature or closed-loop control
qualification is claimed. Python traced peak: 32,454,158 bytes for
model construction and 64 queries after imports; native/full-media peak remains
unmeasured and RTX 5070 Ti is untested.

Manual command (dexvision Conda environment, repository root):
`python -m dexvision.apps.repair_perception review --evaluation-dir outputs/level5/perception_confirmation_v8/evaluation`.
The owner confirmed all prescribed review criteria passed on October 16, 2026;
`docs/level5_perception_acceptance.json` records acceptance of the 16 panels and
36 scheduled clips. The command above is retained for saved-media reproduction;
no manual check remains for 5.6. Stop before 5.7.

### Goal

Train or integrate a lightweight visual pipeline that converts the Level 4
single-camera images into shared typed estimates, with explicit
field provenance and compatibility with the actual accepted state bundles.
Preserve the selected policies, transforms, scales, decoders, cadence and
historical receipts. Closed-loop perception qualification belongs to 5.7.
The additive availability-aware `WorldEstimate` is shared by truth/inference;
conversion to unchanged accepted `WorldState` requires all consumer fields to
pass the full boundary. Unobservable fields may be explicitly unsupported,
but must never be padded into a valid control state.

### Accepted-bundle input and observability audit

Start from the exact selected artifacts in each owner acceptance receipt, not
the original 5.0 model/input defaults. Inspect runtime consumers and engineered
controls as well as the neural feature transform.

| Accepted bundle | Dependencies to audit |
| --- | --- |
| Reach V5, `level5_reach_acceptance.json` | Requested approach geometry, calibrated measured/commanded robot origins and tracking lag; provenance of any entity-derived goal |
| Pick V7, `level5_pick_acceptance.json` | Object pose/orientation, motion, supported/held/contact state, entry reference, closure/contact history and family-aware geometry |
| Place V25 with V19 weights, `level5_place_acceptance.json` | Object/target geometry, orientation/motion, retention, release/settling phases, wrist/entry context and collision-preview state |
| Push V8 with V7 weights, `level5_push_acceptance.json` | Initial object/goal basis, current position/speed, contact/support, tracking predicates, causal phases/completion latch and engineered settling holds |
| Press V1, `level5_press_acceptance.json` | Fixture rest pose and +X slide axis, measured displacement/depth, orientation/motion, contact and pressed/release dwell history |

Produce a field-source inventory for every consumed input, derived goal,
phase, latch and preview dependency. Classify its producer as inferred vision,
declared robot/fixture sensor telemetry, frozen calibration/geometry,
evaluation-only simulator truth, or unsupported. Record units, coordinate/body
origin, timestamps, validity/confidence, reset/history rules and consumer.
An available simulator field does not establish a deployment sensor source.
Missing values, fabricated identity quaternions, zero velocities or empty
contact lists cannot masquerade as valid measurements. Unknown relation or
fixture state must block the affected consumer explicitly.

The accepted place runtime's `place_wrist_hold_preview.py` calls
`place_collision_preview.py`, which copies live MuJoCo state and uses predicted
contacts/object positions to choose an action. Record this as a privileged
compatibility gap; object-pose replacement alone cannot remove it. Do not
silently disable, rewrite or requalify that accepted decoder in 5.6. Any later
estimated-state preview requires an explicitly versioned integration amendment
and paired evaluation; otherwise the affected skill remains unsupported for the
perception track. Evaluation truth must never feed policy-visible history,
terminal-derived latches or action selection.

### Files

```text
dexvision/perception/object_detector.py
dexvision/perception/object_tracker.py
dexvision/perception/object_pose.py
dexvision/evaluation/evaluate_perception.py
dexvision/apps/train_perception.py
dexvision/apps/evaluate_perception.py
configs/level5/perception.yaml
configs/level5/perception.sha256
tests/test_object_perception.py
tests/test_perception_metrics.py
docs/level5_perception_results.md (recipe, field-source inventory and compatibility report)
```

### Build order

```text
1. accepted-bundle field-source inventory and frozen perception recipe
2. deterministic truth adapter for labels, evaluation and parity tests only
3. conventional detector or segmenter baseline with stable temporal ids
4. calibrated world-frame pose/orientation and timestamp-aware motion estimates
5. explicit validity, confidence, ambiguity and stale-observation rejection
6. optional compact VLM semantic selector only with a separately justified recipe
```

### Recipe, data and temporal evaluation

Freeze a separately versioned, checksum-locked perception recipe before fitting
or selection. Name the immutable visual/source manifests, camera calibration,
implementation/schema digests, accepted policy references, method/model,
feature/preprocessing scales, supervision, training budget, candidate schedule,
selection ranking, evaluation denominators and artifact paths. Declare three
training seeds for a trained baseline, or explain why training/seeds are not
applicable to a deterministic method. Do not inherit policy-specific MLP heads,
normalization or offline-loss early stopping. Select candidates and confidence/
staleness handling on training/validation only; freeze the final configuration
before one held-out evaluation. Retain failed candidates and missing cells.

Keep the 5.0 locks and `evaluation_common.yaml` unchanged. Explicitly bind the
parent confidence 0.8 and maximum age 0.1 s, or document any separate versioned
validation-selected handling; never choose thresholds from held-out outcomes.
Preserve every original numerical qualification/safety gate.

Use the released `level4/visual-camera-v1` intrinsics/extrinsics, metre units,
camera right/down/forward axes and world-frame wxyz pose convention. Declare
the object body origin, height/dimensions, symmetry and geometric assumptions
needed to recover metric pose from one RGB view. A visible-mask centroid or
board-plane intersection is not automatically the object's 3D body center.
Report unobservable orientations/occluded objects as ambiguous or unsupported;
retained truth annotations for invisible entities are not inferred detections.
Audit measured forearm versus command-body calibration separately.

Audit visual coverage against all accepted skill entries and phases, including
the later pick/place source releases and scripted setups. The original compact
Level 4 visual subset does not automatically cover their changed geometry.
Keep session/episode/cell ownership across all derived render variants and
augmentations. Geometric augmentation must transform calibration/metric labels
consistently. Missing coverage requires an explicit versioned data amendment,
not edits to immutable releases or use of test scenes for training.

The existing visual export samples every fifth source frame and excludes some
duplicates. Build episode-local timestamped sequences with declared gaps;
reset tracking at episode boundaries. Estimate velocity using real elapsed time,
never array position or an assumed policy sample interval. Evaluate identity,
motion, occlusion, reacquisition and stale rejection on sequences with a frozen
gap policy, not only shuffled frames/contact sheets. Report inference latency,
observation age and delivery cadence against each bundle's control cadence.

### Metrics and compatibility

Original visual gates remain unchanged:

```text
core-class detection recall at IoU 0.5 >= 0.95
median metric position error <= 0.010 m
95th-percentile metric position error <= 0.025 m
identity switches <= 1 per 1000 tracked frames
stale or ambiguous observation rejection = 100 percent in synthetic tests
optional semantic selector accuracy >= 0.90 if it is claimed
```

Define visible/eligible instance counts, missing detections/poses, rejected
observations and conditional pose-error denominators before evaluation; report
coverage/rejection alongside accuracy so omissions cannot inflate results.
Break down detection, identity, position, orientation, linear/angular velocity,
fixture displacement, availability and latency by split, class/instance,
condition and relevant skill/phase. Declare symmetry-aware orientation scoring
and any supplemental validity/error limits from training/validation before test.

Add offline feature-parity and task-sensitivity diagnostics using the exact
accepted transforms. Press release depth is 2 mm; push tracking envelope is
10 mm and its original speed predicate is 20 mm/s. The generic 10/25 mm pose
gates alone do not prove those predicates reliable. Report bias, velocity/depth
error, phase/predicate disagreements and uncertainty near decision boundaries.
Do not relax gates or claim closed-loop success from these diagnostics.

Every required field must have a tested source or an explicit unsupported
disposition. An honestly documented incompatible skill need not prevent 5.6's
baseline acceptance when its own gates and manual review pass, but it blocks
that skill's perception qualification/default admission. Resolve and evaluate
integration changes separately in the authorized later checkpoint.

Synthetic isolation tests must vary withheld truth labels/state while holding
RGB, allowed telemetry and calibration fixed: inferred observations and offline
policy-visible features/history must stay unchanged. Separately test missing
orientation/velocity/contact/fixture fields, explicit invalidity, stale ages,
ambiguous identities and reset/gap handling without a webcam, GPU or visible GUI.

### Commands

The following are the completed V29/freshV8 runs. Do not rerun a consumed
confirmation or overwrite immutable outputs. New reproduction needs a separate
frozen recipe, fresh case allocation and new output paths. Manual command and
explicit pass/fail criteria are in docs/level5_perception_repair.md.

```bash
conda activate dexvision
python -m dexvision.apps.guarded_body_perception select --config configs/level5/perception_repair_v29.yaml
python -m dexvision.apps.guarded_body_perception freeze --config configs/level5/perception_confirmation_v8.yaml --output-dir outputs/level5/perception_confirmation_v8/protocol
python -m dexvision.apps.render_perception_amendment collect --protocol outputs/level5/perception_confirmation_v8/protocol/protocol.json --output-dir outputs/level5/perception_confirmation_v8/dataset
python -m dexvision.apps.repair_perception package --protocol outputs/level5/perception_confirmation_v8/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v8/dataset --release datasets/level5-perception-confirmation-v8.tar.gz
python -m dexvision.apps.guarded_body_confirmation_perception --protocol outputs/level5/perception_confirmation_v8/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v8/dataset --output-dir outputs/level5/perception_confirmation_v8/evaluation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n dexvision pytest -q
conda run --no-capture-output -n dexvision pytest -q tests/test_perception_guarded_body.py tests/test_perception_body_tracking.py tests/test_perception_continuity.py tests/test_perception_identity.py tests/test_perception_dual.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check .
git diff --check
```

### Pass criteria

```text
[x] Train/validation/test visual splits match Level 4 episode ownership
[x] Versioned recipe, budget, selection, thresholds and data/code/artifact digests are frozen
[x] All five accepted bundles have a field-source inventory and explicit compatibility disposition
[x] Truth labels/evaluation are isolated from inferred observations, derived phases and action selection
[x] Detection, identity, position/orientation, motion, fixture depth and availability metrics are separate
[x] Timestamped sequence tests cover gaps, occlusion, reacquisition and identity continuity
[x] Offline field-evidence sidecar tests shared types, per-field timing/provenance and unsupported-field rejection
[x] Ground-truth and inferred observations use the same availability-aware typed estimate schema; accepted full control schema remains guarded
[x] Frame/body-origin calibration and exact-transform oracles are checked; unavailable full inferred-feature parity explicitly rejects
[x] Original and supplemental primary visual gates pass on validation; sensitivity/unsupported fields disclosed
[x] Fresh independently frozen V8 confirmation passes; all earlier consumed sets remain closed
[x] Ambiguous, low-confidence, and stale results fail closed
[x] Model size, license, CPU latency, cadence and Python memory documented; native peak unmeasured and RTX 5070 Ti explicitly untested
[x] No visual model emits actuator commands
[x] Existing V8 validation diagnostic review is owner confirmed complete (separate from held-out acceptance)
[x] Manual held-out contact-sheet and temporal-video review is owner accepted (October 16, 2026; level5_perception_acceptance.json)
```

Fresh V8 confirmation was performed once and passes all automated gates and
audits. Its held-out manual review is now owner accepted, completing 5.6.
All consumed sets remain closed. Do not repeat inference or proceed to 5.7.

### Historical review feedback and prescribed acceptance criteria

The following feedback records the earlier, then-incomplete handoffs. The
October 16 held-out acceptance above closes the remaining manual gate.

The owner reported "I think mostly good" and authorized commit/push. This
positive preliminary feedback is bound in `docs/level5_perception_review_feedback.json`.
The later "Okay done. I thought this was already done" confirms the validation
diagnostic review complete, separately bound in
`docs/level5_perception_validation_acceptance.json`. Held-out manual acceptance
is not claimed. Reproduction command/pass/fail criteria are
in `docs/level5_perception_repair.md`. It does not fulfill the remaining
independent confirmation or full typed-schema requirements.

Manual verification for full qualification requires both a fresh held-out
contact sheet and temporal video,
with source episode/frame/timestamp and inferred id, confidence/age, mask/box,
pose axes and rejection reason. Include supported skill/fixture strata and
occlusion/reacquisition cases, and label unsupported strata explicitly.
Pass when every scheduled panel/sequence is present, ids stay with the correct
objects, pose/box/mask overlays align with visible geometry, timestamps and
units agree, and missing/stale/ambiguous observations visibly reject. Fail for
identity swaps, unexplained pose jumps, misaligned overlays, hidden truth-filled
estimates, missing strata, silent stale acceptance or review errors. Stop for
owner confirmation after automated checks; that confirmation is now recorded
above for 5.6. Do not start 5.7.

---

## Level 5.7 — Perception-Grounded Skill Rollouts

The paired baseline must load the whole selected state-policy bundle: feature
transform, scales, phase rules, active outputs, decoder and cadence as well as
weights. Audit derived goals/features/phases for hidden simulator truth.
Attribute noise, dropout, bias and latency failures on validation before any
separate robustness-training amendment; retain the unchanged comparison.
See `docs/level5_training_guidance.md` for the amendment and isolation rules.

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

First rule out frame, scale, observation aliasing and action-representation
errors using `docs/level5_training_guidance.md`. A failed MLP or low offline
loss alone does not justify a larger model. History must resolve measured
ambiguity; chunking must address feedback/contact changes and interruption.
Scripted correction arms retain lineage and use declared sampling/update
budgets; abort-only failures cannot become expert action targets.

### Goal

Use Level 4 deterministic scripted corrections or a more expressive model only
where frozen metrics identify a failure the proposed change can plausibly
address.

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
expert-only versus expert-plus-scripted-correction data
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
feature transform, calibration, scales, active action mask, and decoder digests
causal phase/history rules, control interval, and selected-epoch evidence
learned versus engineered behaviour and any expert assistance
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
load the selected weights together with their exact transform/decoder bundle
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

Before pilot test runs, validate composition on actual predecessor terminal
states. A scripted standalone start is not evidence that the preceding policy
produces a supported entry state. Preserve object identity, retention and prior
applied commands; initialize phase/history from physical state. Reject invalid
transitions without teleporting, hidden resets or expert rescue. Follow
`docs/level5_training_guidance.md` for diagnosis and amendment boundaries.

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
complete selected policy bundles, including transforms/scales/decoders and cadence
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
[x] 5.0 Level 3 evidence, model choices, protocols, gates, and artifacts are frozen
[x] 5.1 loader/training infrastructure is deterministic and resumable
[x] 5.2 reach is evaluated against Level 3 failures and Level 5 gates
[x] 5.2A reach repair receives manual acceptance after passing automated gates
[x] 5.3 pick is evaluated across supported object families
[x] 5.3A pick repair passes automated gates and owner manual acceptance
[x] 5.4 place-held-object is evaluated across targets and held-out goals;
    state qualification is negative and no manual review is triggered
[x] 5.4A place repair passes frozen state gates, physical audits and owner manual acceptance
[x] 5.5 push and press are evaluated; optional dial status is explicit
[ ] 5.6 detector/tracker/pose pipeline meets frozen visual gates
[ ] 5.7 perception-grounded rollouts have no simulator-truth fallback
[ ] 5.8 corrections/model escalation are evidence-gated
[ ] 5.9 every skill is qualified, experimental, or failed
[ ] 5.10 only qualified compatible skills execute by default
[ ] 5.11 all three workcell pilots and the combined work order pass
[ ] 5.12 immutable skill release and Level 6 handoff verify cleanly
[ ] No LLM, open-world manipulation, learned recovery, or deferred task is claimed
```
