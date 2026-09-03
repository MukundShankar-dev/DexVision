# Progress Level 3 — Skill Policy Learning and Imitation Learning

Level 3 goal:

> Train reusable low-level skill policies from Level 2 demonstrations so the
> simulated robot hand can perform resettable tasks without live human
> teleoperation.

Start with state-based behavior cloning. Do not start with raw images. Vision policies are a later checkpoint.

The first reach-policy split and rollout acceptance criteria are frozen in
`configs/level3_evaluation.yaml` and `docs/level3_evaluation_protocol.md`.
Offline demonstration validation and held-out closed-loop rollout evaluation
are separate evaluations.

Warning:
Do not train serious policies until the final Level 1.13 action space is stable.
Changing from finger-only actions to base-plus-finger actions after recording
demos can invalidate saved datasets, trained checkpoints, rollout metrics, and
policy comparisons.

Level 3 learns from Level 2 skill demonstrations recorded with the full Level
1.13 teleoperation action space:

```text
base position target
base orientation target
finger actuator targets
```

Early behavior-cloning baselines may train on simplified action subsets, such
as finger-only targets or task-specific reduced controls, when that makes a
checkpoint easier to validate. The saved Level 2 dataset should still preserve
the full action at every timestep so later policies can use the complete
hand/base/wrist command without recollecting demonstrations.

---

## Level 3 Skill Policy Framing

Each Level 2 task can produce one learned skill policy. A skill policy is a
learned controller with:

```text
skill name
skill version
task/environment id
typed parameters and goal schema
observation schema
action schema
policy checkpoint
preconditions
termination condition
success condition
failure condition
evaluation metrics
known limitations
```

Reusable skills should be goal-conditioned whenever the task has a variable
target. A policy trained only for one fixed target is a useful task baseline,
but it must not be presented as a parameterized orchestration skill.

Examples:

```text
reach_touch_target(target_pose)
button_press(button_id, target_press_depth)
push_object_to_target(object_id, target_pose)
rotate_dial(dial_id, target_angle)
```

`free_space_gesture` is primarily a calibration, representation, and
data-pipeline dataset. It is not required to become a Level 5 orchestration
skill unless a later checkpoint defines a concrete goal-conditioned use for it.

Primitive skills to learn over Level 3:

```text
reach_touch_target
reach_object
button_press
push_cube_to_target
grasp_object
pinch_lift_object
place_object
release_object
rotate_dial
```

Current concrete task ids use `button_press` and `push_cube_to_target`. Future
skill-card APIs may expose more general aliases such as `press_button` and
`push_object_to_target` once the task metadata can parameterize objects and
targets cleanly.

The Level 3 boundary is:

```text
observation + typed goal parameters -> closed-loop skill policy -> low-level action
```

The future Level 5 boundary is:

```text
skill name + typed parameters -> supervised skill executor -> terminal SkillResult
```

Level 3 should not learn "make a sandwich" end to end. Long-horizon symbolic
planning, object perception, world-state tracking, and skill composition are
future Level 5 concerns. Level 3's job is to train and evaluate primitive
policies that a future planner could call.

Only a skill whose Level 2 task, pilot, relabeling, quality report, scaled
dataset, and held-out evaluation conditions are complete may enter Level 3.
The first Level 3 scope is `reach_touch_target`, followed by `button_press` and
`push_cube_to_target` when their dataset summaries mark them ready. Later
skills such as reach-object, grasp, lift, place, release, and dial rotation
must first return through the same Level 2 task-and-pilot gates; their presence
in the candidate list is not permission to skip those prerequisites.

Recommended learned skill curriculum:

```text
1. reach_touch_target
2. button_press
3. push_cube_to_target
4. reach_object
5. grasp_object
6. pinch_lift_object
7. place_object
8. release_object
9. rotate_dial
```

`reach_touch_target` is the first learned skill because it tests base/wrist
motion and target reaching without object dynamics. `button_press` is second
because it adds simple contact against a constrained fixture. `push_cube_to_target`
is third because it adds object dynamics and planar manipulation.
`reach_object` becomes the object-relative approach primitive before grasping.
Grasp, lift, place, release, and dial rotation remain later skills until
reaching, pressing, and pushing are reliable.

---

## Level 3.1 — Goal-Conditioned Per-Skill Dataset Loader

### Goal

Load saved Level 2 skill demos into PyTorch with deterministic per-skill splits.

### Files

```text
dexvision/learning/datasets.py
tests/test_learning_dataset.py
```

### Inputs

From Level 2 demos:

```text
skill_name
task_id
typed task/skill parameters
sampled initial state
robot_states
object_states
actions, preserving the full Level 1.13 command
timestamps
quality report
success labels
executable observation layout version
```

### Learning Observation Schema

The first state-based dataset should build observations from saved Level 2
state, not live camera frames:

```text
hand/base pose
hand/base velocity, if available
finger joint positions
finger joint velocities
object pose
target pose
goal/skill parameters
```

Object and target fields may use masks or neutral placeholders for tasks that
do not include movable objects, but the schema should make those choices
explicit.

Every field must be extracted through the versioned executable observation
layout created in Level 2.4C. The loader must not infer column offsets from
array widths or duplicate hardcoded MuJoCo ordering.

### Learning Action Schema

The dataset action should preserve the full Level 1.13 command:

```text
base position target
base orientation target
finger actuator targets
```

Quaternions may be stored in demos for replay because MuJoCo uses wxyz
quaternions. For learning, the dataset may convert orientation to a 6D rotation
representation or another stable representation to avoid quaternion sign
discontinuities.

Early BC baselines may return simplified subsets, such as finger-only or
base-only actions, but the saved demo format should retain the complete action
so later experiments do not need recollected data.

### Output

PyTorch samples:

```python
{
    "obs": Tensor,
    "goal": Tensor,
    "obs_mask": Tensor,
    "action": Tensor,
    "skill_name": str,
    "task_id": str,
    "demo_id": str,
    "timestep": int
}
```

### Run

```bash
pytest tests/test_learning_dataset.py
```

### Pass Criteria

```text
[ ] Loads synthetic demos
[ ] Full base-plus-finger action schema documented
[ ] Filters failed demos optionally
[ ] Filters low-quality demos optionally
[ ] Filters/selects by skill_name/task_id
[ ] Returns correct tensor shapes
[ ] Observation and action normalization use training-split statistics only
[ ] Train/validation split is deterministic
[ ] Splits are grouped by whole episode and initial/goal condition
[ ] No timesteps from one episode appear in multiple splits
[ ] Existing data without session ids are labeled as not testing cross-session generalization
[ ] Future session ids, when present, are kept wholly within one split
[ ] Held-out rollout goals are excluded from demonstration splits
[ ] Split manifest records version, seed, goal, and optional session provenance
```

### Codex Prompt

```text
Implement a PyTorch dataset for saved demos.
Use the executable observation layout, goal parameters, target state, and full Level 1.13 actions to create samples.
Support filtering by success and quality score.
Support filtering/selecting by skill_name/task_id.
Split by whole episode and initial-goal condition, never by random timestep.
Keep a whole session in one split when real session ids are available, but do
not invent session ids for the existing Level 2 data or claim cross-session
generalization.
Compute normalization statistics from the training split only.
Treat the reserved held-out goals as closed-loop rollout scenarios, not as
required demonstration test samples.
Add tests with synthetic demos.
Do not implement a model yet.
```

---

## Level 3.2 — MLP Behavior Cloning Skill Baseline

### Goal

Create the simplest goal-conditioned per-skill imitation policy.

### Files

```text
dexvision/learning/models.py
tests/test_models.py
```

### Model

```text
input: normalized observation vector + normalized goal vector
output: full action vector
architecture: MLP
loss: MSE
```

The first MLP should predict:

```text
base translation target or delta
base orientation target or delta
finger actuator targets
```

Absolute targets are the first baseline because they match replay and are
easier to validate against saved demonstrations. Delta actions can be tested
later if absolute action prediction is too brittle or task-dependent.

Start with `reach_touch_target(target_pose)`. The same trained checkpoint
should accept multiple configured target poses; do not train one checkpoint
per target position.

### Run

```bash
pytest tests/test_models.py
```

### Pass Criteria

```text
[ ] Forward pass works
[ ] Output shape matches full action dimension
[ ] Model supports configurable hidden sizes
[ ] Goal tensor changes the model input
[ ] Model supports batched observation masks
[ ] No GPU required
```

### Codex Prompt

```text
Implement an MLP behavior cloning skill policy in dexvision/learning/models.py.
Condition the first reach_touch_target policy on target_pose.
Predict the full action vector: base translation target or delta, base orientation target or delta, and finger actuator targets.
Add tests for forward pass, output shape, and configurable hidden sizes.
Do not add training loop yet.
```

---

## Level 3.3 — Training Loop and Tiny Overfit Test

### Goal

Train an MLP behavior-cloning skill policy on saved demonstrations.

### Files

```text
dexvision/learning/train_bc.py
dexvision/apps/train_policy.py
configs/level3_bc.yaml
tests/test_train_tiny.py
```

### Run

```bash
python -m dexvision.apps.train_policy --config configs/level3_bc.yaml --skill reach_touch_target
pytest tests/test_train_tiny.py
```

### Pass Criteria

```text
[ ] Training starts
[ ] Loss decreases on tiny synthetic dataset
[ ] Checkpoint saved
[ ] Train/val losses saved
[ ] Skill name and task id are saved with the checkpoint metadata
[ ] Goal schema and executable observation-layout version are saved
[ ] Normalization statistics and split manifest are saved
[ ] CPU works
```

### Codex Prompt

```text
Implement a behavior cloning training loop.
Use the existing dataset and MLP model.
Save checkpoints and train/val loss history.
Train one goal-conditioned skill at a time, starting with reach_touch_target(target_pose).
Save the goal schema, observation/action layout versions, normalization statistics, and episode-level split manifest.
Add a tiny overfit test where loss decreases on synthetic data.
Do not implement MuJoCo rollout evaluation yet.
```

---

## Level 3.4 — Rollout Evaluation Per Skill

### Goal

Run a trained goal-conditioned skill policy in closed loop and report
per-skill metrics on held-out goals and initial states.

### Files

```text
dexvision/learning/policies.py
dexvision/apps/evaluate_policy.py
dexvision/evaluation/metrics.py
tests/test_policy_shapes.py
```

### Run

```bash
python -m dexvision.apps.evaluate_policy --policy data/policies/mlp_bc.pt --task reach_touch_target --episodes 10
pytest tests/test_policy_shapes.py
```

Use `reach_touch_target` for the first behavior-cloning rollout/evaluation
checkpoint. `button_press` should follow after reaching works, and
`push_cube_to_target` is a later rollout task once simple contact is reliable.

The first reach evaluation must consume the frozen scenarios and acceptance
gates in `configs/level3_evaluation.yaml`. Do not tune on the held-out rollout
matrix or change its thresholds after seeing a policy result without creating a
new protocol version.

Evaluation must include target poses not used for training. A memorized fixed
trajectory to one target does not pass this checkpoint.

### Pass Criteria

```text
[ ] Policy checkpoint loads
[ ] Skill metadata matches task environment
[ ] Typed goal parameters validate against the task schema
[ ] Policy outputs valid actions
[ ] Rollout applies hand base/wrist target commands
[ ] Rollout applies finger actuator targets
[ ] Workspace limits are checked
[ ] Joint limits are checked
[ ] Success and failure termination conditions stop the rollout
[ ] Held-out target positions are evaluated
[ ] All frozen training-target and held-out-target reset perturbations are evaluated
[ ] Rollout runs without crashing
[ ] Metrics saved
[ ] Frozen success, jerk, invalid-action, workspace, and joint-limit gates are reported
[ ] Videos optional but supported
```

### Action Application

Policy rollout must apply both halves of the Level 1.13 command:

```text
hand base/wrist target
finger actuator targets
```

The rollout path should reject or clip invalid commands before stepping MuJoCo,
with explicit checks for workspace limits and robot joint or actuator limits.

### Metrics

```text
success rate
success rate on held-out goals
final distance to target
episode length
mean action jerk
workspace-limit violations
joint-limit violations
mean contact count, optional
failure reason counts
```

### Codex Prompt

```text
Implement policy rollout evaluation in MuJoCo.
Load a trained goal-conditioned MLP checkpoint, validate target_pose, apply both base/wrist targets and finger actuator targets, run it on held-out reach_touch_target goals, and save metrics.
Check workspace limits and joint limits during rollout.
Stop on explicit success, failure, or timeout and report the terminal reason.
Treat button_press and push_cube_to_target as later rollout tasks.
Do not add vision policy yet.
```

---

## Level 3.4B — Skill Card Export

### Goal

Export a machine-readable skill card for each trained policy so future Level 5
orchestration can discover what the policy can do.

Level 5 is not implemented in this repo. Skill cards are metadata artifacts for
future composition; they do not cause an LLM or planner to output raw actions.

### Files

```text
dexvision/learning/skill_cards.py
dexvision/apps/export_skill_card.py
tests/test_skill_cards.py
```

### Example Card

```json
{
  "skill_name": "reach_touch_target",
  "skill_version": "1.0.0",
  "task_id": "reach_touch_target",
  "policy_checkpoint": "data/policies/reach_touch_target_mlp_bc.pt",
  "policy_checkpoint_sha256": "sha256:<checkpoint-digest>",
  "observation_schema_version": "level2/observation-layout-v2",
  "action_schema_version": "level1.13/full-action-v1",
  "parameter_schema": {
    "type": "object",
    "required": ["target_pose"],
    "properties": {
      "target_pose": {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": {"type": "number"},
        "x-units": "meters",
        "x-frame": "world"
      }
    }
  },
  "preconditions": [
    "target is inside workspace",
    "robot hand calibrated",
    "policy action schema matches environment schema"
  ],
  "success_condition": "selected fingertip or palm site reaches target threshold for dwell time",
  "timeout_steps": 240,
  "failure_conditions": [
    "timeout",
    "workspace violation",
    "joint limit violation",
    "invalid or unavailable task observation"
  ],
  "metrics": {
    "success_rate": null,
    "mean_completion_time": null,
    "mean_final_distance": null
  },
  "known_limitations": []
}
```

### Run

```bash
python -m dexvision.apps.export_skill_card --policy data/policies/reach_touch_target_mlp_bc.pt --out data/skill_cards/reach_touch_target.yaml
pytest tests/test_skill_cards.py
```

### Pass Criteria

```text
[ ] Skill card includes skill name and task id
[ ] Skill version and checkpoint digest are included
[ ] Observation/action schema versions are declared
[ ] Parameters use a machine-validatable typed schema
[ ] Parameter units and coordinate frames are declared
[ ] Preconditions are declared
[ ] Success/failure conditions are declared
[ ] Timeout and terminal result fields are declared
[ ] Metrics fields are exported
[ ] No Level 5 planner is implemented
```

### Codex Prompt

```text
Implement skill card export for trained Level 3 policies.
Export metadata only: typed parameter schema, schema versions, checkpoint path/digest, preconditions, success/failure/timeout conditions, metrics, and limitations.
Do not implement Level 5 orchestration or an LLM planner.
```

---

## Level 3.4C — Supervised Skill Runtime

### Goal

Provide one policy-independent runtime that loads a validated skill card and
checkpoint, executes the learned policy in closed loop, enforces safety and
termination rules, and returns a structured result.

This is not an LLM planner, HTTP server, or multi-skill orchestrator.

### Files

```text
dexvision/learning/skill_executor.py
dexvision/learning/skill_registry.py
dexvision/apps/run_skill.py
tests/test_skill_executor.py
```

### Contract

```python
executor.list_skills() -> tuple[SkillSummary, ...]
executor.start(skill_name, parameters, request_id=None) -> ExecutionHandle
executor.status(execution_id) -> SkillResult
executor.cancel(execution_id) -> SkillResult
```

`SkillResult` should include:

```text
execution_id
caller request_id, when provided
skill name/version
status: running, succeeded, failed, timed_out, cancelled, rejected
terminal reason
steps executed
initial and final task state summaries
metrics
retryable flag
```

While an execution is active, `status()` may return `running`; every terminal
state must be immutable once reported.

The runtime must validate:

```text
skill exists
parameter schema
policy/checkpoint digest
observation/action schema compatibility
task/environment compatibility
preconditions
workspace and actuator limits
success/failure/timeout termination
```

### Run

```bash
python -m dexvision.apps.run_skill --skill reach_touch_target --target-pose 0.1 0.0 0.12 --headless
pytest tests/test_skill_executor.py
```

### Pass Criteria

```text
[ ] Registry discovers compatible skill cards
[ ] Invalid parameters are rejected before simulation steps
[ ] Schema/checkpoint mismatches are rejected clearly
[ ] Repeated caller request ids do not start duplicate executions
[ ] Reach skill runs through one reusable executor
[ ] Success, failure, timeout, cancellation, and rejection are distinct
[ ] Workspace and actuator safety checks cannot be bypassed by the policy
[ ] Structured SkillResult is saved or returned
[ ] No LLM, planner, web API, or multi-skill chaining is implemented
```

### Codex Prompt

```text
Implement only the supervised skill registry and executor.
Run reach_touch_target through the same runtime future callers will use.
Validate parameters, schemas, checkpoint compatibility, safety, and terminal conditions.
Return a structured SkillResult.
Do not implement an LLM, web API, or multi-skill orchestration.
```

---

## Level 3.5 — Clean vs Unfiltered Data Comparison

### Goal

Show that data quality matters.

### Experiment

Train two policies:

```text
Policy A: all demos
Policy B: only demos passing quality filters
```

Run this comparison per skill, starting with `reach_touch_target`.

### Run

```bash
python -m dexvision.apps.train_policy --config configs/level3_bc_all.yaml
python -m dexvision.apps.train_policy --config configs/level3_bc_filtered.yaml
python -m dexvision.apps.compare_policies --policies data/policies/all.pt data/policies/filtered.pt
```

### Pass Criteria

```text
[ ] Both policies train
[ ] Both policies evaluate
[ ] Comparison table saved
[ ] Results are grouped by skill
[ ] README explains result
```

### Codex Prompt

```text
Add an experiment script that compares behavior cloning trained on all demos versus quality-filtered demos.
Save a comparison table with success rate, final distance, and action smoothness.
Run per-skill comparisons.
Do not add new model architectures.
```

---

## Level 3.5A — Action-Space Ablation Experiment

### Goal

Measure whether learned task performance comes from finger control, base/wrist
control, or the combined full action space.

### Experiment

Train and compare:

```text
finger-only BC
base-only BC
base + finger BC
```

Use the same cleaned Level 2 demos and task splits where possible. Dataset
loading may expose action subsets for each ablation, but the underlying demos
must preserve the full Level 1.13 action.

### Metrics

```text
success rate
final distance to target
action smoothness
workspace-limit violations
joint-limit violations
```

### Pass Criteria

```text
[ ] Finger-only BC trains and evaluates
[ ] Base-only BC trains and evaluates
[ ] Base + finger BC trains and evaluates
[ ] Comparison table includes all required metrics
[ ] Results explain which action components matter for each skill/task
```

### Codex Prompt

```text
Add an action-space ablation experiment for finger-only, base-only, and base-plus-finger behavior cloning.
Use the same saved Level 2 demos and report success rate, final distance, smoothness, workspace-limit violations, and joint-limit violations.
Do not add new policy architectures.
```

---

## Level 3.5B — Corrective Demonstrations and Recovery Evaluation

### Goal

Measure and reduce behavior-cloning compounding error before treating a policy
as a reusable orchestration skill.

Start with `reach_touch_target`. Collect a small, separately labeled set of
operator corrections from policy rollouts that drift, stall, or approach a
limit. Keep the original expert dataset immutable.

### Experiment

Compare:

```text
baseline behavior cloning
behavior cloning plus corrective demonstrations
```

Evaluate both on the same held-out goals and initial states.

### Pass Criteria

```text
[ ] Corrective episodes are stored separately from original expert demos
[ ] Failure reason and intervention point are recorded
[ ] Training can include or exclude corrective data deterministically
[ ] Both policies evaluate on identical held-out conditions
[ ] Success, timeout, action jerk, and limit violations are compared
[ ] Result documents whether corrections improved robustness
```

### Codex Prompt

```text
Add a small corrective-demonstration experiment for reach_touch_target.
Preserve the original expert dataset and label correction/intervention episodes separately.
Compare baseline BC with BC plus corrections on identical held-out goals.
Do not add a new model architecture or Level 5 orchestration.
```

---

## Level 3.6 — Temporal Behavior Cloning

### Goal

Improve policy with short history.

### Files

```text
dexvision/learning/sequence_datasets.py
dexvision/learning/temporal_models.py
tests/test_sequence_dataset.py
```

### Model Options

Start with:

```text
GRU policy
```

Later:

```text
small Transformer
Temporal CNN
```

### Run

```bash
python -m dexvision.apps.train_policy --config configs/level3_gru_bc.yaml
pytest tests/test_sequence_dataset.py
```

### Pass Criteria

```text
[ ] Sequence dataset works
[ ] GRU forward pass works
[ ] Hidden state resets between episodes
[ ] Evaluation runs
[ ] Compare with MLP BC
```

### Codex Prompt

```text
Implement sequence-based behavior cloning using a short history window and a GRU policy.
Add tests for sequence batching and model output shape.
Compare to the MLP baseline.
Do not add image input yet.
```

---

## Level 3.7 — Vision Dataset Support

### Goal

Add image observations without yet training a complex vision model.

### Files

```text
dexvision/learning/vision_dataset.py
tests/test_vision_dataset.py
```

### Inputs

```text
camera frames or rendered MuJoCo frames
robot state
target state
action
```

### Run

```bash
pytest tests/test_vision_dataset.py
```

### Pass Criteria

```text
[ ] Image frames load
[ ] Image tensor shape correct
[ ] Image/action alignment correct
[ ] Optional transforms work
```

### Codex Prompt

```text
Implement vision dataset loading for recorded frames.
Return image tensor, proprioception vector, and action.
Add tests with synthetic images.
Do not implement CNN policy yet.
```

---

## Level 3.8 — CNN + Proprioception Policy

### Goal

Train a basic vision-conditioned imitation policy.

### Files

```text
dexvision/learning/vision_models.py
tests/test_vision_model.py
```

### Model

```text
small CNN encoder
proprioception MLP
fusion MLP
action head
```

### Run

```bash
python -m dexvision.apps.train_policy --config configs/level3_vision_bc.yaml
pytest tests/test_vision_model.py
```

### Pass Criteria

```text
[ ] Model forward pass works
[ ] Tiny image dataset can be overfit
[ ] Policy can run in simulation
[ ] Results compared against state BC
```

### Codex Prompt

```text
Implement a small CNN + proprioception behavior cloning policy.
Train it on image observations plus robot state.
Add tests and a tiny overfit check.
Do not add large pretrained models yet.
```

---

## Level 3.9 — ACT-Style Action Chunking Stretch

### Goal

Predict chunks of future actions for smoother manipulation.

### Files

```text
dexvision/learning/action_chunking.py
tests/test_action_chunking.py
```

### Concept

Instead of:

```text
obs_t -> action_t
```

Predict:

```text
obs_t -> actions_t:t+K
```

### Run

```bash
python -m dexvision.apps.train_policy --config configs/level3_action_chunking.yaml
pytest tests/test_action_chunking.py
```

### Pass Criteria

```text
[ ] Dataset returns action chunks
[ ] Model outputs [K, action_dim]
[ ] Temporal aggregation works
[ ] Compare against single-step policy
```

### Codex Prompt

```text
Implement an action-chunking behavior cloning variant.
The dataset should return K future actions and the policy should predict action chunks.
Add temporal aggregation during rollout.
Compare against single-step BC.
```

---

# Level 3 Completion Checklist

```text
[ ] Full base+finger action schema documented
[ ] Executable observation layout is consumed without inferred offsets
[ ] Per-skill PyTorch demo dataset loader works
[ ] Dataset splits are grouped by episode/session/initial-goal condition
[ ] Training-only normalization statistics are saved
[ ] Goal-conditioned MLP BC skill baseline works
[ ] Training loop and tiny overfit test work
[ ] Tiny overfit test passes
[ ] Goal-conditioned policy rollout in MuJoCo works on held-out goals
[ ] Rollout applies base and finger commands
[ ] Explicit success/failure/timeout termination works
[ ] Metrics saved
[ ] Filtered-vs-unfiltered comparison complete
[ ] Action-space ablation documented
[ ] Typed, versioned skill card export works
[ ] Supervised skill registry/executor returns structured SkillResult
[ ] Corrective-demonstration comparison is documented
[ ] Temporal BC implemented
[ ] Vision dataset implemented
[ ] Vision BC policy implemented
[ ] Optional action chunking implemented
[ ] Results documented
```
