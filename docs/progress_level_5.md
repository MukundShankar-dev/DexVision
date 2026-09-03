# Progress Level 5 — Full-Scale Skill Learning and Qualification

Level 5 goal:

> Train, evaluate, and package a comprehensive set of reusable manipulation
> skills from the frozen Level 4 dataset.

Level 3 establishes whether learning works at all on the narrow Level 2 data.
Level 4 creates the larger, multi-session, visually grounded dataset. Level 5
is where those data become qualified policies and a supervised skill library.

Level 5 must not collect an unplanned replacement dataset to rescue a failed
experiment. If the frozen Level 4 data are insufficient, report the failure,
define the missing coverage, and return through a new versioned Level 4 data
checkpoint before retraining.

Language-guided task planning is Level 7. Level 5 may run deterministic
scripted multi-skill plans, but it must not add an LLM planner or allow a model
to emit raw high-rate actuator commands.

---

## Level 5 Skill-Learning Boundary

```text
Level 4 release + typed goal + world state
  -> trained Level 5 perception/controller components
  -> supervised skill executor
  -> structured terminal SkillResult
```

The skill controller owns bounded closed-loop action prediction. Visual
perception produces stable object ids and poses. A compact VLM may help map a
phrase to an object id, but metric localization and safety checks remain with
the grounded perception and supervisor layers.

Core skill families:

```text
approach: reach_object
acquire: grasp_object
retain: hold_object, lift_object
move: transport_object
deposit: place_object, release_object
planar interaction: push_object_to_target, slide_object_to_target
fixture interaction: press_button, rotate_dial
recovery: retry_approach, regrasp_object, recover_dropped_object
```

Candidate skills such as alignment, stacking, hinged-lid operation, and simple
guided tool use remain optional until every required core skill has a measured
qualification result.

---

## Level 5.0 — Learning Scope and Evaluation Freeze

### Goal

Turn the Level 3 results and Level 4 release into a frozen training and
qualification matrix before full-scale training starts.

### Files

```text
docs/level5_learning_plan.md
configs/level5_learning.yaml
configs/level5_evaluation/*.yaml
tests/test_level5_learning_plan.py
```

### Required decisions

```text
model carried forward from Level 3
per-skill observation, goal, and action schemas
session/object/goal/scene split manifests from Level 4
ground-truth-state versus perception-grounded evaluation tracks
numerical offline, rollout, safety, and recovery gates
training budget, seeds, checkpoint-selection rule, and stop conditions
qualified, experimental, and failed status definitions
```

### Pass criteria

```text
[ ] Every core skill has a frozen protocol before training
[ ] Validation selects checkpoints; held-out test conditions do not tune them
[ ] Model escalation rules are evidence-based
[ ] Dataset and split digests are immutable inputs
[ ] No full-scale training starts before this checkpoint passes
```

---

## Level 5.1 — Full-Scale Dataset and Training Infrastructure

### Goal

Load the comprehensive release efficiently and reproducibly across every skill
family.

### Files

```text
dexvision/learning/skill_datasets.py
dexvision/learning/training_manifest.py
dexvision/apps/train_skill.py
tests/test_skill_datasets.py
```

### Requirements

```text
session-grouped and condition-grouped splits
training-only normalization
state, visual, expert, failure, and corrective streams
deterministic sampling and weighting
schema/layout validation without inferred offsets
checkpoint/config/dataset/split digests
CPU smoke tests and optional CUDA acceleration
```

### Pass criteria

```text
[ ] No split leakage across reserved sessions or conditions
[ ] Expert, failure, and corrective data can be selected independently
[ ] Image/state/action alignment is verified at load time
[ ] Interrupted training can resume without changing sample assignments
[ ] Small CPU-only end-to-end test passes
```

---

## Level 5.2 — Reach and Approach Policies

### Goal

Train and qualify goal-conditioned reaching across the Level 4 object and pose
coverage.

### Evaluation

```text
held-out recording sessions
held-out object instances and goal regions
workspace-boundary starts
ground-truth object state
perception-grounded object state, separately reported
```

### Pass criteria

```text
[ ] Offline and closed-loop metrics are saved
[ ] Cross-session and condition-held-out results meet frozen gates or fail visibly
[ ] Invalid actions and workspace/joint-limit violations are zero or within gate
[ ] `reach_object` has a typed valid-initial-state and terminal-state envelope
```

---

## Level 5.3 — Grasp, Hold, and Lift Policies

### Goal

Train acquisition and retention policies that generalize across the frozen
rigid-object families.

### Required metrics

```text
grasp acquisition rate
stable hold duration
lift-height success
slip and drop rate
unwanted contact and safety violations
performance by object shape, mass, friction, and session
```

### Pass criteria

```text
[ ] Each policy has independently measurable success and failure
[ ] Held-object state is physically recomputed, not inferred from the action
[ ] Failed acquisitions, slips, and drops remain in reports
[ ] Qualified policies meet every frozen per-object and aggregate gate
```

---

## Level 5.4 — Transport, Place, and Release Policies

### Goal

Train object-in-hand transitions needed by useful composed tasks.

### Required metrics

```text
transport retention rate
target-pose/receptacle placement success
post-release stability
premature-release and misplacement rates
performance by source, destination, object, and session
```

### Pass criteria

```text
[ ] Execution rejects episodes where the object is not actually held
[ ] Place and release policies have explicit, compatible terminal envelopes
[ ] Held-out receptacles and goal regions are evaluated
[ ] Failed transfers and unstable post-release states remain visible
```

---

## Level 5.5 — Push, Slide, Press, and Rotate Policies

### Goal

Generalize the Level 2 contact-task baselines using the larger Level 4 object,
fixture, and session coverage.

### Pass criteria

```text
[ ] Push/slide results cover varied size, mass, friction, start, and target
[ ] Press results cover held-out button ids/depths and wrong-contact failures
[ ] Rotate results cover held-out starts/angles and contact-loss failures
[ ] Level 2 and Level 5 results are compared without merging their claims
```

---

## Level 5.6 — Visual Grounding and Perception Integration

### Goal

Train or integrate the visual system that supplies stable object identities and
poses to the learned skills.

### Build order

```text
1. Train/evaluate a conventional detector or segmenter on Level 4 renders.
2. Add temporal identity tracking and pose estimation.
3. Evaluate held-out scenes, object instances, occlusion, and lighting.
4. Integrate perception-grounded skill rollouts behind the same world-state API.
5. Optionally test a compact local VLM for semantic disambiguation.
```

The VLM does not replace the detector/tracker for metric control and does not
output actuator commands. Model size, GPU memory, license, latency, and
failure behavior must be documented; local execution on the intended RTX 5070
Ti is preferred for the optional semantic component.

### Pass criteria

```text
[ ] Detection, pose, tracking, and semantic-selection metrics are separate
[ ] Ground-truth-state and perception-grounded policy results are not conflated
[ ] Ambiguous/stale observations are rejected safely
[ ] CPU tests use synthetic data; GPU/model downloads are not test requirements
```

---

## Level 5.7 — Corrective and Recovery Learning

### Goal

Use the separately labeled Level 4 corrective data to reduce compounding error
and qualify explicit recovery skills.

### Comparisons

```text
expert-only baseline
expert plus corrective demonstrations
recovery-enabled versus abort-only execution
```

### Pass criteria

```text
[ ] Compared policies share frozen splits and evaluation conditions
[ ] Policy/checkpoint and intervention provenance are preserved
[ ] Recovery success, retry count, jerk, timeout, and safety metrics are reported
[ ] A failed recovery cannot be mislabeled as ordinary task success
```

---

## Level 5.8 — Evidence-Gated Model Escalation

### Goal

Add temporal history, action chunking, or a stronger visual encoder only where
the simpler policy has a measured failure that the change could address.

Candidate models:

```text
short-history GRU
small temporal CNN or Transformer
ACT-style action chunking
compact pretrained visual encoder
```

Diffusion policies, large foundation models, or complex RL require explicit
user approval and a separate checkpoint.

### Pass criteria

```text
[ ] Baseline failure and model hypothesis are written before implementation
[ ] New and old models use identical data and held-out evaluation
[ ] Added complexity is retained only if metrics justify it
[ ] Compute, latency, memory, and reproducibility costs are reported
```

---

## Level 5.9 — Cross-Skill Qualification

### Goal

Assign each learned policy an evidence-backed status before it can enter the
runtime.

Statuses:

```text
qualified: passes all frozen required gates
experimental: runs but misses at least one required gate
failed: cannot execute reliably or violates a safety/compatibility gate
```

### Pass criteria

```text
[ ] Every core skill receives one explicit status
[ ] Metrics include session-, object-, goal-, and failure-mode breakdowns
[ ] Failed and experimental skills remain visible but are disabled by default
[ ] Qualification is reproducible from immutable data and checkpoint digests
```

---

## Level 5.10 — Skill Cards, Registry, and Supervised Executor

### Goal

Expose qualified policies through one typed, policy-independent runtime without
adding an LLM or long-horizon planner.

### Files

```text
dexvision/learning/skill_cards.py
dexvision/learning/skill_registry.py
dexvision/learning/skill_executor.py
dexvision/apps/run_skill.py
tests/test_skill_executor.py
```

### Pass criteria

```text
[ ] Cards include versioned schemas, checkpoint digest, parameters, and metrics
[ ] Registry excludes incompatible, experimental, or failed policies by default
[ ] Preconditions, timeouts, safety limits, and terminal states are enforced
[ ] Repeated request ids do not duplicate physical actions
[ ] No LLM, web API, or multi-skill planner is implemented
```

---

## Level 5.11 — Diverse Scripted Pilot Tasks

### Goal

Prove that the qualified skill library supports several materially different
tasks before Level 7 adds language planning.

### Core pilots

```text
1. Sort and pack varied objects into matching bins/receptacles.
2. Clear the workspace into a tray and recover from at least one miss.
3. Press specified buttons and rotate a dial to target states in order.
4. Assemble a sandwich proxy by stacking rigid bread/filling props on a plate.
5. Set a snack place using a cup, plate, and utensil proxy.
```

At least three materially different pilots must pass. Every pilot uses a
deterministic scripted plan and the typed calls intended for future Level 7.
It fails if a human silently repairs state between skills.

Real tomato cutting is not a core acceptance task. A guided knife through a
pre-scored or pre-segmented rigid proxy may be an explicitly labeled later
experiment, but it is not evidence of deformable-food cutting or tool safety.

### Pass criteria

```text
[ ] At least three pilots run from reset through explicit termination
[ ] Every skill request, result, and world-state transition is recorded
[ ] Recovery and abort behavior are deterministic and visible
[ ] Per-pilot success rates and failure reasons are reported
[ ] Sandwich assembly is one pilot, not the project's sole final claim
```

---

## Level 5.12 — Skill-Library Results and Level 6 Handoff

### Goal

Publish the full learning results and freeze the artifacts that Level 6 will
polish and Level 7 may later orchestrate.

### Required artifacts

```text
per-skill training and evaluation manifests
qualified/experimental/failed registry
checkpoint and schema digests
ground-truth versus perception-grounded comparisons
failure and recovery analysis
diverse pilot results
known limitations and unsupported claims
```

### Pass criteria

```text
[ ] Every claim traces to immutable data, config, split, and checkpoint digests
[ ] Negative results and disabled skills are documented
[ ] At least three diverse scripted pilots pass
[ ] Level 6 receives reproducible results rather than presentation-only claims
[ ] Level 7 prerequisites are listed without implementing orchestration
```

---

# Level 5 Completion Checklist

```text
[ ] Learning scope and per-skill protocols are frozen
[ ] Comprehensive dataset loader and training manifests are reproducible
[ ] Reach/approach skills are evaluated
[ ] Grasp/hold/lift skills are evaluated
[ ] Transport/place/release skills are evaluated
[ ] Push/slide/press/rotate skills are evaluated
[ ] Visual grounding is measured and safely integrated
[ ] Corrective/recovery learning is evaluated
[ ] Model escalation is evidence-gated
[ ] Every core skill is qualified, experimental, or failed
[ ] Qualified skills run through one supervised executor
[ ] At least three diverse scripted pilots pass
[ ] Full-scale results and Level 6 handoff are documented
```
