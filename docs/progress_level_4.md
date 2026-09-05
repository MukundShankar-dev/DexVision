# Progress Level 4 — Comprehensive Workcell Skill Dataset

Level 4 goal:

> Convert the Level 3 feasibility findings into a versioned, multi-session
> dataset for a bounded tabletop workcell. The release must support the five
> required operational skills that Level 5 will train and qualify.

Level 4 is a data-engineering and task-definition phase. It does not train
full-scale policies, qualify skills, run long-horizon pilots, or add an LLM.
Negative Level 3 results are inputs: Level 4 must collect the coverage and
state needed to address measured failures rather than silently changing a
frozen Level 3 test.

Every numbered section is one checkpoint. Complete it, run its listed checks,
perform any listed manual verification, and update `docs/CURRENT_STATUS.md`
before selecting the next checkpoint.

---

## Level 4 System Boundary

```text
Level 3 results and failure analysis
  -> frozen workcell/data specification
  -> task scenes and typed world state
  -> session-aware recording and phase labels
  -> pilot collection and final count freeze
  -> multi-session data haul
  -> visual annotations and split audit
  -> immutable Level 4 release
  -> Level 5 training and qualification
```

The future language model selects typed skills; it does not control the hand:

```text
language request (Level 7)
  -> symbolic plan
  -> deterministic supervisor
  -> qualified Level 5 skill
  -> bounded base/wrist/finger actions
```

### Required operational skills

| Skill | Required initial state | Goal | Successful terminal state |
|---|---|---|---|
| `reach_object(entity_id, approach_pose)` | Hand in safe neutral/workspace state | Reach an object or fixture approach pose | Approach distance/orientation and dwell gates pass without disturbing the scene |
| `pick_object(object_id)` | Hand in that object's valid approach envelope; object supported and not held | Acquire and lift one supported rigid object | Correct object is held above the configured height for the stability dwell |
| `place_held_object(target_id)` | Exactly one known object is held | Transport, place, and release at a target | Object settles inside the target tolerance and the hand is no longer holding it |
| `push_object_to_target(object_id, target_zone)` | Hand in the valid push approach envelope; object supported | Move an object along the surface | Object settles inside the target zone without leaving the board |
| `press_button(button_id)` | Hand in the valid fixture approach envelope | Depress the named button | Correct button reaches the depth/state threshold for the dwell window |

`rotate_dial(dial_id, target_angle)` is optional. Level 4.3 may promote it only
if the pilot passes without reducing required-skill coverage. If promoted, its
data budget is additional to the required 250–350 new episodes.

### Internal phase labels

The dataset may use the following labels for segmentation and diagnosis:

```text
approach
acquire
lift
stabilize
transport
place
release
settle
push_contact
fixture_contact
retract
```

They are not separate planner-visible policies. A complete pick/place
demonstration counts as one recorded episode even when it yields multiple
training segments. Reports must publish both episode counts and segment counts
so data are never double-counted.

Level 4.0 must also freeze an online phase state machine. Its current phase and
transitions must be computable causally from the current and prior typed world
state, task request, and terminal metrics; future frames and retrospective
labels are forbidden at execution time. Store both the online phase and any
audited annotation so disagreement can be measured. Phase-specific action
relevance masks must be versioned with the transition rules.

### Recorded action and safety contract

Every Level 4 control sample must distinguish:

```text
requested action and request source (operator, script, or policy)
commanded action before safety handling
applied action after clipping, rate limiting, rejection, or safe fallback
per-field clipping/rejection mask and stable reason code
the prior commanded and prior applied action
the resulting state timestamp and control interval
```

Commanded and applied arrays retain the complete named action layout. Level
4.0 must freeze units, coordinate frames, absolute-versus-delta semantics,
per-field bounds, rate limits, and phase-specific relevance for base,
orientation, wrist, and finger fields. Quaternions must be normalized and
sign-continuous with the previous applied quaternion; the first sample uses a
frozen canonical-sign rule. These records must make bounded residual targets
derivable without rewriting an accepted episode. Unsafe requested/commanded
motion may remain as failure evidence, but it must never become an expert
applied-action target.

### Default workcell vocabulary

Level 4.0 must freeze exact ids, but the minimum intended scope is:

```text
object families: block/cuboid, cylinder, flat puck
instances: at least two geometrically distinct instances per family
fixtures: start_button; mode_dial only if promoted
targets: return_bin_left, return_bin_right, inspection_pad,
         setup_slot_a, setup_slot_b
camera: one fixed rendered/workcell camera; no multi-camera calibration
```

The primary claims are limited to these rigid objects, fixtures, and bounded
tabletop poses. Arbitrary-object grasping is not implied.

### Default new-data budget

Level 4.3 freezes final counts after pilot measurements. Unless that checkpoint
documents a justified revision, use this plan:

| Data group | Minimum new accepted episodes | Planning target |
|---|---:|---:|
| Reach/object-or-fixture approach | 30 | 30–40 |
| Complete pick/place sequences | 120 | 120–150 |
| Push-to-zone | 40 | 40–55 |
| Button press | 30 | 30–40 |
| Ordinary failures and safe corrections | 30 | 30–50 |
| Required total | 250 | 250–350 |
| Optional dial | 0 unless promoted | Additional 30–40 |

Global totals are not sufficient coverage. Level 4.0 must assign every frozen
coverage cell to train, validation, test, or intentionally unsupported status
and declare `minimum_accepted_by_split` for that cell. An intentionally
held-out object/goal cell must have zero training ownership and positive
validation or test ownership. Level 4.3 freezes the exact per-cell minima after
the pilot; a global total may not compensate for a missing required cell.

The immutable Level 2 release remains legacy seed data: 55 reach, 55 button,
and 101 push successes. It does not count toward the new-session minimum, and
it must not be rewritten to invent session metadata.

### Session and split policy

New data requires a genuine `recording_session_id`. A session means a separate
recording block with a fresh process start and reset/calibration record; renaming
episodes from one sitting does not create multiple sessions.

Minimum split policy:

```text
at least 4 genuine sessions
sessions A/B: training pool
session C: validation only
session D: untouched cross-session test
additional sessions: assigned wholly to exactly one split before collection use
held-out object instances and goal regions: test only
normalization/statistics: training split only
```

No session may cross splits. No test episode, held-out instance, or held-out
goal may influence collection tuning, normalization, checkpoint choice, or
threshold selection. `operator_id` is always recorded, but the default project
makes no cross-operator claim; any later cross-operator claim requires multiple
operators and a separately frozen operator-held-out split.

### Fixed-camera visual claim boundary

Level 4 uses one calibrated camera pose and does not claim cross-camera or
real-world visual transfer. Level 4.0 must freeze a compact visual-condition
matrix covering nominal rendering plus mild illumination variation, partial
occlusion, and bounded workcell distractors. It must assign each condition cell
to splits and define object/fixture coverage. Camera pose and intrinsics remain
fixed. Level 4.7 must report missing cells rather than silently broadening the
claim or duplicating frames.

### Non-goals and deferred work

```text
learned regrasp or dropped-object recovery
learned retry selection
general grasping of arbitrary objects
hinged lids or articulated household objects
tools, cutting, pouring, liquids, or deformables
arbitrary household/open-world scenes
multiple calibrated cameras
end-to-end VLM control or actuator prediction
full-scale policy training
```

Initial failure handling remains deterministic: detect failure, move to a safe
pose, retry once only when the failure class is explicitly retryable, then
abort and report the reason.

---

## Standard Checkpoint Procedure

Every Level 4 checkpoint must state and preserve:

```text
inputs and prerequisite checkpoint
files added or changed
schema/config versions
exact automated test and lint commands
exact manual command and observable pass/fail criteria, when required
generated artifacts and their storage location
known limitations and explicitly excluded work
```

Automated tests must use synthetic data and headless MuJoCo where possible.
They must not require a webcam, GUI, downloaded model, or GPU. Run focused
checks first, then the full suite before committing:

```bash
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

---

## Level 4.0 — Workcell and Dataset Requirements Freeze

### Goal

Translate the completed Level 3 report into a machine-readable Level 4
requirements document before new task or collection code is written.

### Inputs

```text
docs/level3_results.md or the final Level 3.8 results artifact
docs/level3_evaluation_protocol.md
datasets/level2-v1 manifest, checksum, and quality summaries
docs/task_environment.md
```

Level 3.4 already shows low reach success and workspace/joint-limit
terminations. The final Level 3 diagnostics must decide whether the relevant
Level 4 response is coverage, observation/state, action representation, model
history, or a combination. Do not assume “more data” without that analysis.

### Files

```text
docs/level4_dataset_plan.md
configs/level4_dataset.yaml
tests/test_level4_dataset_plan.py
```

### Required specification

`configs/level4_dataset.yaml` must contain:

```text
dataset name and schema versions
five required skills and optional-dial flag
object families, instance ids, geometry metadata, and train/held-out roles
fixture and target ids
valid reset ranges and safe workspace margin
typed goal fields, units, and coordinate frames
phase-label vocabulary and transition rules
success, failure, timeout, and retryability rules
provisional and maximum episode counts by data group
per-cell train/validation/test minima and split ownership
minimum four sessions and whole-session split policy
RGB/state/action stream requirements
requested, commanded, and applied action semantics plus safety reason codes
causal online phase state machine and phase-specific action relevance masks
fixed-camera visual-condition matrix and explicit claim boundary
quality thresholds and acceptance workflow
Level 3 failure -> Level 4 requirement traceability table
```

### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_dataset_plan.py
conda run -n dexvision ruff check tests/test_level4_dataset_plan.py
```

### Pass criteria

```text
[x] Every Level 3 failure category is accepted, deferred, or mapped to a concrete requirement
[x] Every core skill has typed preconditions, goals, terminal states, and executable metrics
[x] Object, fixture, target, phase, session, and split vocabularies are machine-readable
[x] The 250–350 required-episode envelope and optional-dial treatment are explicit
[x] Every required coverage cell has split ownership and a per-split minimum
[x] Four whole sessions provide separate training, validation, and test ownership
[x] Requested, commanded, and applied actions are reconstructable with safety reasons
[x] Online phases are causal and the fixed-camera visual claim is bounded
[x] No collection has started under an unfrozen schema
```

Manual verification: none.

The frozen human-readable plan is in `docs/level4_dataset_plan.md`; its
machine-readable authority is `configs/level4_dataset.yaml`. The specification
enumerates the five required skill contracts, six rigid-object instances,
fixture/target/reset vocabularies, 79 exclusively split-owned coverage cells,
four whole-session slots, the exact 250-new-episode minimum and 350 planning
maximum, complete 27-field requested/commanded/applied action records, causal
online phases, quality/acceptance gates, and the bounded fixed-camera visual
matrix. Every Level 3 gap is accepted, deferred with an evidence trigger, or
explicitly unsupported. No workcell, recording, collection, policy, or dataset
payload was added or changed. Automated checks passed on September 3, 2026
using `conda run -n dexvision pytest -q tests/test_level4_dataset_plan.py` with
9 passed, both the checkpoint and repository Ruff commands, and `conda run -n
dexvision pytest -q` with 461 passed.

Stop after this checkpoint. Do not build the workcell in the same change.

---

## Level 4.1 — Workcell Scene, World State, and Task Contracts

### Goal

Create one resettable MuJoCo workcell and a typed state contract shared by
tasks, perception, recording, evaluation, and the future executor.

### Files

```text
assets/mujoco/workcell_scene.xml
dexvision/sim/workcell.py
dexvision/sim/world_state.py
dexvision/perception/object_observations.py
configs/workcell.yaml
dexvision/apps/inspect_workcell.py
tests/test_workcell_scene.py
tests/test_world_state_contract.py
```

### Required behavior

```text
deterministic reset from a named seed
bounded collision-free object/fixture placement
stable object_id and class_id for an episode
pose, velocity, units, and coordinate frame for each entity
held/support/receptacle relationships when known
observation source, confidence, and timestamp
identical schema for simulator truth and inferred perception
task factories for reach, pick, place-held, push, and press
optional dial factory disabled by default
```

The scene must support all three final pilots without swapping XML files. Scene
randomization may move objects only inside the ranges frozen in Level 4.0.

### Commands

```bash
conda run -n dexvision pytest -q tests/test_workcell_scene.py tests/test_world_state_contract.py
conda run -n dexvision ruff check dexvision/sim/workcell.py dexvision/sim/world_state.py dexvision/perception/object_observations.py dexvision/apps/inspect_workcell.py tests/test_workcell_scene.py tests/test_world_state_contract.py
python -m dexvision.apps.inspect_workcell --config configs/workcell.yaml --seed 0
```

### Pass criteria

```text
[x] Headless tests reset every required entity and compute every task metric
[x] Stable ids survive an episode and reset predictably
[x] Frames, units, valid ranges, and stale-state behavior are explicit
[x] Unsupported or ambiguous ids fail with actionable errors
[x] One scene contains the clearing, inspection, and setup regions
```

Manual verification is required. The viewer passes when every required object,
both return bins, the inspection pad, both setup slots, and the Start button
are visible and reachable, nothing begins interpenetrating, and repeated reset
with the same seed reproduces the layout. It fails on clipping, unreachable
targets, unstable objects, mislabeled fixtures, or a nondeterministic reset.

Automated implementation checks passed on September 3, 2026 using the two
checkpoint test files with 14 passed, the checkpoint and repository-wide Ruff
commands, the 1,200-step headless inspector, and the full suite with 475
passed. The first interactive attempt exposed a locked fixed-camera view and a
hard 1,200-step viewer exit. The inspector now starts with a movable overview
camera, stays open until the viewer is closed, and provides keyboard controls
for reset, seed changes, pause, and labels. Follow-up reviews found the
unlabeled colors too ambiguous and then exposed that MuJoCo's default viewer
hides site group 4. Workcell-only label anchors now use visible site group 0
and are enabled by default without labeling every internal robot body. A later
manual review found that the camera projection visually reversed the
left/right return bins and that randomized object-to-anchor assignment could
obscure the setup-slot markers. Camera-only adjustments failed re-verification
because the frozen target coordinates themselves encoded left and right
opposite to the operator-facing view, while the first camera-space regression
also used a reversed horizontal-vector sign. The target centers and MuJoCo
bodies are now corrected together so ids, labels, world state, and task metrics
all use operator-visible left/right. The corrected regression checks both
frozen-to-runtime coordinate equality and screen-space ordering. Objects use
deterministic per-object spawn lanes inside the frozen reset bounds, the setup
markers are smaller, and an automated clearance assertion prevents objects
from starting over either slot.
The user confirmed on September 3, 2026 that final interactive verification
passed: operator-facing left/right labels were correct, the setup-slot overlap
was gone, and the viewer satisfied the checkpoint criteria. Level 4.1 is
complete. Level 4.2 was not started.

---

## Level 4.2 — Session-Aware Recording and Phase-Label Schema

### Goal

Extend the immutable episode format so new workcell data has genuine session
provenance, task parameters, reconstructable phase boundaries, and optionally
synchronized RGB.

### Files

```text
dexvision/logging/dataset_schema.py
dexvision/logging/demo_logger.py
dexvision/logging/session_manifest.py
dexvision/logging/phase_labels.py
dexvision/apps/record_demo.py
dexvision/apps/validate_level4_episode.py
tests/test_level4_episode_schema.py
tests/test_phase_labels.py
```

### Required episode metadata

```text
episode_id and recording_session_id
anonymous stable operator_id
skill/sequence name and typed goal
object, fixture, and target ids
sampled reset state and random seed
camera/render/calibration version when RGB is enabled
teleoperation, policy, or correction source
requested, commanded, and applied actions plus safety masks/reasons
online phase and optional audited phase annotation
phase intervals with monotonic non-overlapping frame indices
intervention interval and failure reason when applicable
schema, config, code, observation, and action versions
synchronized timestamps for state, action, task metrics, and images
```

The writer is append-only. It must resume without collisions and must never
overwrite a Level 2 episode or an accepted Level 4 episode.

### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_episode_schema.py tests/test_phase_labels.py tests/test_demo_logger.py
conda run -n dexvision ruff check dexvision/logging/dataset_schema.py dexvision/logging/demo_logger.py dexvision/logging/session_manifest.py dexvision/logging/phase_labels.py dexvision/apps/record_demo.py dexvision/apps/validate_level4_episode.py tests/test_level4_episode_schema.py tests/test_phase_labels.py
python -m dexvision.apps.validate_level4_episode --episode <episode-directory>
```

### Pass criteria

```text
[x] Missing or duplicate session ids fail clearly
[x] Phase transitions are valid and reconstructable from saved indices/state
[x] Online phases use no future frames and disagreements with annotations are reportable
[x] A complete pick/place episode yields compatible reach, pick, and place segments
[x] Requested, commanded, and applied actions reproduce safety handling exactly
[x] RGB is optional, but enabled frames align to timestamps and state
[x] Legacy Level 2 episodes still load without invented Level 4 fields
[x] Resume behavior never overwrites an existing episode
```

Manual verification: none; use synthetic episodes.

The Level 4 episode extension stores genuine session/operator provenance,
typed goals and reset state, five reconstructable action stages, per-field
safety evidence, causal online/audited phases, optional aligned RGB, and
separate state/action/task/image timestamps while leaving legacy Level 2
episodes unchanged. Session manifests reject duplicate ids, resumed episode
allocation skips existing directories, and Level 4 episode writes are atomic
and append-only even when the legacy overwrite option is supplied. Complete
pick/place phase intervals derive compatible reach, pick, and place segments
without changing episode counts. Automated checks passed on September 4, 2026
using the three checkpoint test files with 31 passed, the checkpoint and
repository-wide Ruff commands, the module validator against a synthetic saved
episode, and the full suite with 489 passed. Level 4.3 collection was not
started.

---

## Level 4.3 — Collection Pilot and Final Coverage Freeze

### Goal

Prove that every required task is recordable and measurable before the bulk
data haul, then replace provisional counts with a frozen coverage matrix.

The September 4 teleoperation pilot changed how this checkpoint must be
completed. Free-space reach is usable, but monocular webcam control did not
produce viable contact demonstrations for press, push, or pick/place. Do not
solve that by collecting more failures, increasing model size, or adding an
LLM/VLM. Level 4.3 now establishes one reliable deterministic expert for every
promoted skill, then validates one shared small learnable interface on button
and push before any bulk collection. Level 5 remains responsible for learning
and qualifying the complete skill set.

### Level 4.3 execution order

Level 4.3 is divided into the lettered checkpoints below. Each letter is one
checkpoint under the repository's one-checkpoint rule: implement it, run its
checks, update status, and stop. Do not begin Level 4.4 until 4.3I passes.

The fixed order is:

```text
4.3A common expert interface plus scripted reach
4.3B scripted button press
4.3C scripted constrained push
4.3D scripted grasp and lift
4.3E scripted place and complete pick/place
4.3F expert/replay qualification audit
4.3G small state-only button learnability probe
4.3H small state-only push learnability probe
4.3I source-mix, count-matrix, and storage freeze
```

The learning probes are formulation tests, not Level 5 full-scale training.
They exist to prove that the expert data and low-dimensional control interface
are learnable before hundreds of episodes are collected.

#### Level 4.3A — Common Scripted Expert and Safe-Waypoint Reach

Add a deterministic, simulator-state-only expert boundary:

```python
expert.reset(task, world_state)
requested_action, phase, done, reason = expert.step(world_state)
```

The expert must emit the existing complete named requested-action layout before
logging. For `source=scripted`, the stored `requested_action` is the nominal
scripted expert output, so `applied_action - requested_action` remains a
derivable residual without mutating the episode schema. Preserve commanded and
applied actions, prior actions, safety masks/reasons, causal phases, and all
existing provenance. Do not add a parallel recorder or episode format.

Use the palm/grasp site, not the forearm root, as the planning point. Implement
only a simple task-relative waypoint generator: rise to a collision-free transit
height, move horizontally, enter a protected pre-contact corridor, then descend.
Validate candidate segments in a scratch copy of the MuJoCo state for workspace
bounds, joint limits, table/fixture contacts, and disturbance of non-target
objects. Do not add OMPL, RRT, or a general-purpose planner unless this bounded
method is measured to fail.

The reference scripted reach must pass at least five randomized valid resets,
recompute success from saved state, replay deterministically, and produce zero
safety violations or non-target disturbance failures.

#### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_expert.py tests/test_level4_scripted_reach.py
```

#### Pass criteria

```text
[x] Expert reset/step behavior and failure reasons are deterministic and tested
[x] Full requested/commanded/applied action records remain schema compatible
[x] Candidate waypoints are validated on copied state before live execution
[x] Five randomized scripted reaches recompute and replay successfully
[x] Workspace, joint-limit, fixture, table, and neighbor-disturbance failures are zero
```

Manual verification: visibly replay one accepted scripted reach and confirm the
same target, path, terminal result, and absence of unintended contact.

Implementation status (September 4, 2026): the common named-action expert,
configuration-owned safe-waypoint reach, copied-state MuJoCo validator, existing
Level 4 recorder integration, and deterministic replay cue restoration are
implemented. The focused checkpoint suite passes with 8 tests across seeds
0--4, repository-wide Ruff passes, the related collection/schema/replay suite
passes with 58 tests, and the full suite passes with 517 tests. The five pass
criteria above passed. The user confirmed the corrected visible replay on
September 4, 2026: the scripted hand reached and dwelled at the cyan pre-grasp
cross, the recording stopped as intended without cube interaction, and the
target cage was corrected to fully enclose the selected block. Level 4.3A is
complete.

The first visible replay review did not pass: the intentionally requested
`--speed 0.1` reduced a 30 Hz episode to roughly three visible pose updates per
second, and the target cage's bottom plane was anchored at the object's center
instead of below it. The cage is now symmetric about the selected entity and
provably encloses `block_small`; the corrected result was accepted by the user.

#### Level 4.3B — Deterministic Button-Press Expert

Implement button press before the other contact skills. Keep the hand posture
and orientation fixed, approach along the button normal, enter
`fixture_contact`, satisfy the existing depth/dwell metric, and retract. Test
multiple valid resets and reject any wrong-button contact, unrelated fixture or
table collision, limit event, or safety intervention.

#### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_button_expert.py
```

#### Pass criteria

```text
[x] Five randomized button resets recompute and replay successfully
[x] Causal approach, fixture_contact, and retract phases are reconstructable
[x] Wrong-button contacts, unintended collisions, and safety violations are zero
```

Implementation status (September 4, 2026):
`DeterministicButtonPressExpert` keeps the neutral hand posture and base
orientation fixed, reaches a configuration-owned pre-contact pose, advances
only along the button's positive joint normal, holds the existing depth metric
for three qualifying samples, and retracts until the button is released. The
start button is now mounted inside the frozen safe workspace at
`[0.11, -0.11, 0.20]` m; its passive slide has recovery travel beyond the
largest accepted depth while the task goal range remains `0.008--0.014` m.
Every trajectory is qualified in copied MuJoCo state before recording. Five
randomized resets recorded and replayed deterministically with reconstructable
`approach`, `fixture_contact`, and `retract` phases, no non-target hand
contacts, no object disturbance above 0.005 m, and zero logged safety masks,
reasons, or interventions. The listed checkpoint suite passes with 2 tests.
Repository-wide Ruff and the full 519-test suite also pass. No manual
verification is required for this headless deterministic checkpoint.

#### Level 4.3C — Deterministic Constrained-Push Expert

Define a task-local frame from object start to target. Approach behind the
object through the safe transit path, descend with fixed posture/orientation,
move straight along the task-local forward axis until the existing target/dwell
metric passes, then retract. During contact, do not introduce lateral, height,
orientation, or finger motion. Cover more than one object family and push
direction without disturbing neighboring objects or leaving the board.

#### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_push_expert.py
```

#### Pass criteria

```text
[x] Five varied push resets recompute and replay successfully
[x] Contact motion is constrained to the frozen task-local forward axis
[x] Board exits, neighbor disturbance, and safety violations are zero
[x] User confirms the remediated cuboid push slides without tipping in visible replay
```

Remediation status (September 4, 2026; complete):
`DeterministicPushExpert` derives a normalized start-to-target axis from the
saved reset state and uses a configuration-owned, family-specific index
posture. Standalone push trials retain the selected object in its seeded pose
and park non-target objects on the lower floor. The expert rises through the
safe transit plane, rotates there at a bounded rate, approaches behind the
object, descends, and then holds height,
orientation, and every finger target fixed while translating only along the
task-local forward axis. It aims for 0.030 m from the target center, waits for
the frozen 0.035 m distance/speed/dwell metric, and retracts axially. Push
qualification now additionally requires table support and no more than 10
degrees of object tilt for every contact/settle/retract sample and for five
terminal samples after release. Scripted recording success uses the final push
metric instead of latching a transient success. Copied-state
qualification rejects workspace or board exits, joint-limit excursions beyond
the configured solver tolerance, contact with any non-target body, and planar
neighbor displacement above 0.005 m. Five resets across cuboid and flat-puck
families and independently recomputed task axes record and replay
deterministically with zero safety masks/reasons/interventions; their final
metrics remain qualified after retraction and object tilt stays within the
10-degree bound. The listed 2-test checkpoint suite, the 33-test combined
push/button/grasp/collection suite, repository-wide Ruff, and the full 524-test
suite pass. The user confirmed that the remediated push, button press, and
standalone grasp-and-lift replays looked good. Level 4.3C is complete again.

#### Level 4.3D — Deterministic Grasp-and-Lift Expert

Treat grasp-and-lift as a standalone skill before placement. Define separate
templates for cuboids, cylinders, and flat pucks. Each template uses an
object-relative grasp transform, a fixed wrist orientation, deterministic open
and closed hand poses, and one scalar grasp-synergy value. Execute approach,
close, lift, and hold, with success determined from object support, lift height,
retention, and stability—not from phase completion alone.

#### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_grasp_lift_expert.py
```

#### Pass criteria

```text
[x] Three valid resets per object family recompute and replay successfully
[x] Object-relative templates and scalar grasp synergy are configuration owned
[x] Lift/hold success uses measured object physics and zero safety violations
```

Implementation status (September 4, 2026):
`DeterministicGraspLiftExpert` resolves configuration-owned cuboid, cylinder,
and flat-puck templates from each seeded object pose. Each template fixes the
grasp-site offset, wrist quaternion, full-flexion endpoint, scalar grasp
synergy, and lift distance. The expert records causal approach, acquire, lift,
and stabilize phases while copied-state qualification checks the complete
trajectory before live execution. Standalone grasp trials keep the selected
object in its seeded workcell pose and park non-target objects on the lower
floor; this isolates grasp physics from the clutter failure already established
by the 4.3 pilot without removing any named object or changing the full object
state schema. The manipulation workspace now includes low tabletop contact
poses; support contact is permitted only before the object clears the 0.040 m
lift threshold. Qualification then requires the selected object to be held by
at least two hand bodies, unsupported by the table, lifted at least 0.040 m,
and moving no faster than 0.020 m/s for ten consecutive samples. Three seeds
for each of the three families record and replay deterministically with planar
neighbor disturbance below 0.005 m and zero safety masks, reasons, or
interventions. The listed 2-test checkpoint suite, 52 related regression tests,
repository-wide Ruff, and the full 524-test suite pass. No manual verification
is required for this deterministic headless checkpoint.
Later visible review accepted the standalone grasp-and-lift behavior while
noting that the cuboid rotates substantially in the grasp. Preserving the
object's initial orientation is not part of this checkpoint's success metric;
the place/complete-pick-place checkpoint must treat it as an explicit design
decision rather than silently changing this qualified grasp controller.

#### Level 4.3E — Deterministic Place and Complete Pick/Place Expert

Only after grasp-and-lift qualifies, add transport, descend to a valid target,
release, allow the object to settle, and retract. Compose this with the qualified
grasp expert into complete pick/place without weakening the standalone pick and
place metrics or rewriting the causal phase labels.

#### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_place_expert.py tests/test_pick_place_segments.py
```

#### Pass criteria

```text
[x] Place succeeds from a genuinely held object and replays deterministically
[x] Ten complete successes span cuboid, cylinder, and flat-puck families
[x] Final target, settled-state, source-object, and neighbor checks all pass
[x] Complete episodes still yield compatible reach, pick, and place segments
```

Manual verification: visibly replay one complete scripted pick/place and confirm
that the selected object is acquired, transported to the highlighted target,
released into a stable supported state, and left undisturbed while the hand
retracts. No non-target object may visibly move. Do not complete 4.3E until the
user confirms this replay.

Implementation status (September 4, 2026): automated implementation is
complete and manual verification passed. `DeterministicPlaceExpert` starts
only from a genuinely held object, transports through copied-state-validated
waypoints, applies deterministic family-specific release offsets, opens the
grasp, clears the released object, waits for support plus linear/angular
settling, and retracts. Visible review rejected the original top-down grasp
because the cuboid yawed about 24 degrees while held. The shared grasp stage now
uses a shape-aware 80--90 degree side-on (hammer-curl) wrist pose and an explicit
rotation-only world-orientation hold during lift, stabilize, transport, and
place. Translation and release remain physics-driven. Cuboid wrist yaw follows
the object's seeded yaw; axial objects retain their symmetry. Replay reapplies
the hold at a bounded internal cadence and reconstructs task-local contact
dynamics and source/destination cues from immutable metadata. Round-object
grasping and settling use six-dimensional table contact with explicit rolling
resistance only for pick and pick/place, preserving the qualified push physics.
Ten complete cuboid, cylinder, and flat-puck episodes record, recompute, and
replay successfully, remain within five degrees of the seeded held orientation,
produce zero safety intervention, and disturb neighbors by less than 0.005 m. A
real saved complete episode also validates and derives compatible reach, pick,
and place action segments. The listed 3-test checkpoint suite and standalone
grasp/push regressions pass, the related suite passes with 41 tests,
repository-wide Ruff passes, and the full suite passes with 527 tests. The saved
viewer uses an oblique source/goal-readable camera. The user confirmed on
September 4, 2026 that the corrected cuboid pick/place replay worked. Level
4.3E is complete.

#### Level 4.3F — Expert Architecture and Replay Qualification

Audit scripted reach, button, push, grasp-and-lift, and complete pick/place as
one architecture. All successes must be regenerated from immutable metadata,
recomputed from saved state, and replayed with the same result. Ordinary failures
remain retained separately. Do not declare the architecture qualified from one
hand-picked reset.

#### Commands

```bash
conda run -n dexvision python -m dexvision.apps.summarize_level4_coverage --config configs/level4_dataset.yaml --dataset-dir data/demos/level4_pilot
conda run -n dexvision pytest -q tests/test_level4_expert.py tests/test_level4_collection.py tests/test_level4_coverage.py
```

#### Pass criteria

```text
[x] Every required scripted skill has repeated recomputed successes
[x] Every accepted trajectory replays and retains complete provenance
[x] Accepted trajectories have zero safety violations and zero neighbor disturbance
[x] Failures remain auditable and never count as expert data
```

Manual verification: visibly replay one accepted trajectory for reach, button,
push, grasp-and-lift, and complete pick/place. Stop until the user confirms all
five.

Implementation status: the versioned expert audit regenerates the frozen task
from each episode's seed, coverage cell, typed goal, and reset metadata, then
headlessly replays the saved requested/commanded/applied actions and recomputes
the task metric. The audit requires two accepted resets for each of the five
source skills, checks exact action provenance, causal phases, aligned
timestamps, reset equivalence, zero logged safety intervention, and less than
the configured 0.005 m planar neighbor disturbance. Complete pick/place also
supplies derived reach, pick, and place evidence. Ordinary failed recordings
remain in the report but cannot contribute to accepted counts. A fresh
10-episode, two-seed audit accepted all 10 trajectories; the focused checkpoint
suite passes with 25 tests. The operator-owned pilot directory remains below
its later collection minima, as expected at this architecture checkpoint.
The user confirmed on September 5, 2026 that the visible reach, button, push,
grasp-and-lift, and complete pick/place replays looked good. Level 4.3F is
complete; no 4.3G work has started.

#### Level 4.3G — Small State-Only Button Learnability Probe

Collect 20 scripted button successes, freeze session-owned train/validation/test
splits, then train one small MLP. Increase to at most 50 successes only when the
20-episode result identifies data volume—not action semantics, normalization,
control rate, phase handling, or rollout integration—as the limiting factor.

Use simulator state only: end-effector-to-target pose, button state, relevant
robot/base velocity, causal phase one-hot, and previous applied action or delta.
The initial learned output is `dx, dy, dz` in the task-local frame; fixed deterministic
posture/orientation logic expands it into the existing full requested-action
layout. Do not add image input, action chunking, a larger network, or a new log
schema. Freeze the pilot metric, seeds, and held-out resets before training.

#### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_lowdim_policy.py tests/test_level4_button_learning_pilot.py
```

#### Pass criteria

```text
[x] Exactly one frozen small-MLP recipe is evaluated before any data increase
[x] Held-out closed-loop button success is at least 0.80 over 20 or more resets
[x] Workspace, joint-limit, wrong-button, and unintended-contact violations are zero
[x] A failure is diagnosed before changing data volume or model class
```

Implementation status: `configs/level4_button_learning_pilot.yaml` freezes
exactly 20 scripted successes in whole-session 14/3/3 train/validation/test
splits, one 64-by-64 tanh MLP, one phase-balanced MSE recipe, and 20 separately
seeded held-out test-cell rollouts. Deterministic approach-class offsets plus
seed jitter make the held-out hand starts physically distinct rather than
moving only irrelevant objects. The 22-value causal observation contains only
end-effector-to-button pose, button state, hand-base velocity, causal phase,
previous applied XYZ delta, and requested depth. The model emits only fixture-
frame `dx, dy, dz`; a deterministic adapter supplies fixed orientation and
finger posture and bounded per-phase motion in the existing full action layout.

The first varied-reset run failed 10 of 20 rollouts when open-finger contact
pushed `rh_MFJ1` past its limit. This was diagnosed before any data or model
change. A minimal fixed-posture interior margin removed that mechanism; three
remaining right-offset failures then identified the same endpoint issue on
`rh_FFJ1`. Applying the same one-percent index margin produced 20/20 successful
held-out press-and-release rollouts with zero workspace, joint-limit,
wrong-button, unintended-contact, or invalid-action events. The dataset remains
20 episodes and the original MLP recipe remains the only model evaluated. The
listed checkpoint suite passes with 5 tests and 37 related workcell/learning
regressions pass. Repository-wide Ruff passes and the full suite passes with
533 tests. No manual verification is required. Level 4.3G is complete; no
4.3H work has started.

#### Level 4.3H — Small State-Only Push Learnability Probe

Run only if 4.3G passes. Reuse the same observation conventions, phase input,
normalization, control rate, low-dimensional action adapter, small-model class,
and frozen evaluation discipline for push. Keep the scripted contact constraint
as the nominal controller; learn only the bounded task-local delta or residual.
Action chunking is allowed later only if single-step control works offline but
measured rollout error shows temporal ambiguity or compounding error. If that
evidence appears, test one small ACT-style horizon of 8 or 16 as a separately
approved checkpoint; do not silently add it here.

#### Commands

```bash
conda run -n dexvision pytest -q tests/test_level4_push_learning_pilot.py
```

#### Pass criteria

```text
[x] Push reuses the qualified low-dimensional interface without schema drift
[x] Held-out closed-loop push success is at least 0.70 over 20 or more resets
[x] Board exits, neighbor disturbance, and safety violations are zero
[x] Any case for action chunking is supported by measured temporal evidence
```

Implementation status (September 5, 2026): the push probe reuses the 4.3G
training-only normalization, causal phase and previous-delta inputs, 17-step
control cadence, shared task-frame XYZ adapter, and single 64-by-64 tanh MLP.
Exactly 20 successful scripted trajectories are isolated in whole-session
14/3/3 splits over disjoint qualified push conditions. The first learned-only
delta rollout scored 0/20 because one-step approach error compounded into the
workspace boundary; it had zero contact, tilt, neighbor, or numerical failures.
Following the required diagnosis, the frozen scripted controller remains the
nominal collision/contact/orientation controller and the MLP learns only the
measured task-frame tracking residual, bounded to 1 mm (0.5 mm on the contact
axis). The corrected single-step formulation completed 20/20 held-out cuboid
and flat-puck resets with zero board exits, neighbor disturbance, workspace,
joint-limit, unintended-contact, tip, or invalid-action events. No action
chunking, image input, larger model, recipe change, or data increase was used.

The provisional right-bin test cells were not used to manufacture a score:
copied-state qualification showed that their current task-axis routes intersect
the fixture/button region, and the flat-puck pre-contact waypoint also exits the
safe workspace. The probe therefore freezes its own disjoint split from already
qualified cells. Level 4.3I must revise the provisional full collection matrix
using this measured infeasibility; it must not claim those right-bin cells are
collectable under the current nominal controller. The listed checkpoint suite
passes with 3 tests, 11 related learning/expert regressions pass, repository-wide
Ruff passes, and the full suite passes with 536 tests. No manual verification is
required. Level 4.3H is complete; no 4.3I work has started.

#### Level 4.3I — Final Source Mix and Coverage Freeze

Use measured expert success rates, collection cost, replay evidence, pilot
learning results, and storage size to replace the provisional matrix. Scripted
expert data may supply nominal successes; working teleoperation may supply reach
or explicitly labeled corrective interventions. Keep `scripted`,
`teleoperation`, `policy_rollout`, and `corrective_intervention` provenance
separate. Do not bulk collect merely to meet the current 250–350 estimate; revise
that envelope if the qualified interfaces justify a different defensible count.

#### Commands

```bash
conda run -n dexvision python -m dexvision.apps.summarize_level4_coverage --config configs/level4_dataset.yaml --dataset-dir data/demos/level4_pilot
conda run -n dexvision pytest -q tests/test_level4_collection.py tests/test_level4_coverage.py tests/test_roadmap_docs.py
```

#### Pass criteria

```text
[ ] Final matrix states accepted minima by skill, source, object/goal cell, and split
[ ] Source mix follows measured expert and learning evidence
[ ] Session isolation, held-out cells, storage handling, and release rules are frozen
[ ] Level 4.4 is still not started until the user accepts the revised matrix
```

### Level 4.3 guardrails

```text
Use privileged simulator state before RGB or learned perception.
Use deterministic experts before imitation learning.
Use small low-dimensional policies before action chunking or larger models.
Use nominal successful trajectories before collecting recovery corrections.
Do not add an LLM, VLM, general planner, RL loop, or bulk data haul in Level 4.3.
Do not change the Level 4.2 episode schema or create a second recording path.
Do not advance to Level 4.4 while any lettered checkpoint is incomplete.
```

### Files

```text
dexvision/logging/level4_collection.py
dexvision/evaluation/dataset_coverage.py
dexvision/apps/summarize_level4_coverage.py
dexvision/sim/workcell_rate_control.py
docs/level4_pilot_report.md
configs/level4_dataset.yaml
tests/test_level4_collection.py
tests/test_level4_coverage.py
```

### Pilot protocol

Record in at least two genuine sessions:

```text
5 accepted reach episodes
10 accepted complete pick/place episodes spanning all three object families
5 accepted push episodes
5 accepted press episodes
ordinary failures retained separately; never relabel them as expert successes
5 dial episodes only when considering optional promotion
```

The pilot report must include collection minutes per accepted episode,
rejection reasons, phase-label agreement, replay/metric recomputation, object
and target coverage, storage per episode, and observed safety problems.

### Commands

```bash
python -m dexvision.apps.record_demo --config configs/level4_dataset.yaml --session-id <session-id> --skill <skill-name>
python -m dexvision.apps.summarize_level4_coverage --config configs/level4_dataset.yaml --dataset-dir data/demos/level4_pilot
conda run -n dexvision pytest -q tests/test_level4_collection.py tests/test_level4_coverage.py
```

### Pass criteria

```text
[ ] Every required skill has at least one replayed, recomputed success
[ ] All object families and target types appear in the pilot
[ ] Episode and segment counts are reported separately
[ ] Final per-cell split minima fit the 250–350 envelope or a revision is justified
[ ] Storage projection determines Git LFS versus external payload handling
[ ] Dial is explicitly promoted or deferred
```

Manual verification is required for one replay of each required skill. Each
replay passes when saved state/action timing reproduces the intended behavior
and the recomputed terminal result matches the visible result. Any wrong
object, wrong target, phase misalignment, unexplained collision, or label
disagreement fails the checkpoint.

Stop until the user confirms the manual replays.

Automated Level 4.3 support was implemented on September 4, 2026. It adds a
live workcell recorder for reach, complete pick/place, push, and press,
append-only per-attempt pilot review evidence, read-only dataset discovery,
session/cell split checks, separate episode and derived-segment counts,
phase-agreement and safety/rejection summaries, collection-time and storage
projection, randomized-object replay restoration, and a coverage-summary CLI.
The optional dial is explicitly deferred. The first manual reach attempt found
that the forearm root was incorrectly occupying the board as the logical palm
control point; that rejected episode was removed at the user's request. The
workcell now keeps the forearm outside the board, initializes the free joint at
the weld pose, calibrates recorder motion around the workcell neutral palm, and
prompts for the Level 4 operator label. A second genuine reach attempt was
retained as an ordinary failure after showing that robot-orientation imitation
was still ergonomically infeasible. Reach control now treats an upright,
webcam-facing human palm as a translation clutch, locks robot orientation, and
shows a cyan target cue. The neutral and translation gains were also adjusted
after the recorded trajectory showed that the first target required excessive
camera-frame travel. A third attempt entered the distance/orientation gates for
36 frames but moved two neighboring objects by about 7 cm because the approach
marker was too low. The marker is now a 0.148 m collision-free pre-grasp cue,
the disturbance metric is retained in dense task state, and an automated test
proves five qualifying frames trigger stop. A fourth attempt exposed a timing
defect: one 2 ms physics step per 30 FPS camera frame made visible motion about
17 times slower than real time. It reached only 0.0745 m and disturbed the scene
by 0.0218 m. The recorder now enforces 17 physics steps per frame and uses more
responsive target filtering and motion limits. Automated checks passed with 18
checkpoint tests, 86 controller/workcell regression tests, repository-wide
Ruff, recorder-help and pilot-summary smoke commands, and 508 full-suite tests.
The real
pilot directory contains five ordinary failures and no accepted episodes. The
fifth retained attempt entered the distance gate for 32 frames but required
near-edge camera travel and accumulated 0.0223 m scene disturbance. A further
attempt then demonstrated the structural limit of absolute monocular mapping:
it came no closer than 0.0400 m while displacing `block_large` by 0.1728 m.
Absolute gain tuning is therefore retired for the reach pilot. The pending
manual trial uses centered nonlinear Cartesian rate control, a high safe-transit
plane, and a target-only descent corridor. The selected object is enclosed in a
bright emissive magenta wireframe cage, while the floating cyan cross separately
marks the desired palm position. A `--workcell-dry-run` mode discards all
temporary frames and does not touch pilot/session evidence. At the user's
request, all five retained failures and their manifest/report were removed from
the active pilot directory and parked recoverably under `/private/tmp`. Four
clean retained trials then tested every required action once. Reach visibly and
recomputably succeeded at 0.0162 m terminal error, five-frame dwell, and about
0.0021 m scene disturbance. Pick/place displaced the intended block and a
neighbor without grasping; push never moved its selected block; button press
never entered the wall fixture's reachable approach range and recorded zero
press depth. The user judged only reach usable under the current interaction.
These are pivot measurements, not expert acceptances; no acceptance sidecars
were created. All
Level 4.3 pass-criteria boxes remain unchecked, coverage counts remain
provisional, and manual verification is still required.

The team-facing interim findings and architecture-decision options are
summarized in `docs/level4_pilot_report.md` under **Interim Mini-Report —
Teleoperation Feasibility**.

---

## Level 4.4 — Reach, Push, and Press Multi-Session Haul

### Goal

Extend the three Level 2 task families into the workcell vocabulary with
genuine sessions, multiple objects/fixtures, and balanced goal coverage.

### Files

```text
configs/level4_dataset.yaml
dexvision/logging/level4_collection.py
dexvision/apps/record_demo.py
dexvision/apps/summarize_level4_coverage.py
tests/test_level4_core_collection.py
```

### Required coverage

```text
reach: object and fixture approaches, interior and near-boundary safe poses
push: blocks and pucks, varied start/target direction, mass, and friction
press: multiple button ids or states when present, varied approach poses
all: nominal and perturbed safe starts, successful and ordinary failed attempts
all: at least four genuine sessions by the end of the checkpoint
```

Any Level 3.4 workspace/joint-limit pattern must have an explicit coverage cell
or an explicit non-data mitigation documented in `docs/level4_dataset_plan.md`.
Do not collect demonstrations that cross safety limits merely to populate a
cell.

### Commands

```bash
python -m dexvision.apps.summarize_level4_coverage --config configs/level4_dataset.yaml --dataset-dir data/demos/level4
conda run -n dexvision pytest -q tests/test_level4_core_collection.py tests/test_level4_coverage.py
```

### Pass criteria

```text
[ ] Frozen reach, push, and press success counts and coverage cells are met
[ ] Sessions A/B, C, and D remain exclusively train, validation, and test owned
[ ] No one session or target dominates beyond the frozen balance tolerance
[ ] Test-session, held-out-object, and held-out-goal data remain untouched
[ ] Every accepted episode passes schema, replay, quality, and recomputed labels
[ ] Rejected and failed attempts remain auditable outside the expert set
```

Manual verification: none beyond the 4.3 replay gate unless a new failure mode
cannot be understood from saved state and headless replay.

---

## Level 4.5 — Complete Pick/Place Multi-Session Haul

### Goal

Collect complete rigid-object pick/place sequences and derive consistent
`reach_object`, `pick_object`, and `place_held_object` segments.

### Files

```text
dexvision/logging/phase_labels.py
dexvision/logging/level4_collection.py
dexvision/apps/record_demo.py
dexvision/apps/summarize_level4_coverage.py
tests/test_pick_place_segments.py
tests/test_level4_pick_place_coverage.py
```

### Required variation

```text
three object families and at least two instances per family
multiple source poses and all required target types
object size, mass, friction, and approach direction
valid grasp variation inside the supported envelope
clean placements plus separately labeled slips, drops, premature releases,
and stable misplacements
```

### Executable boundaries

```text
reach ends: approach distance/orientation dwell passes
pick starts: reach terminal state; object supported and not held
pick ends: correct object exceeds lift height and held-state dwell passes
place starts: exactly one correct object is held
place ends: object settles in tolerance and held state becomes false
```

### Commands

```bash
python -m dexvision.apps.summarize_level4_coverage --config configs/level4_dataset.yaml --dataset-dir data/demos/level4 --skill pick_place_sequence
conda run -n dexvision pytest -q tests/test_pick_place_segments.py tests/test_level4_pick_place_coverage.py
```

### Pass criteria

```text
[ ] At least 120 accepted complete sequences exist unless 4.3 froze a higher count
[ ] Episode, segment, object-family, instance, source, target, and session counts match manifests
[ ] Held-object state and every phase boundary recompute from saved data
[ ] Placement tolerance and post-release stability are executable
[ ] A failed phase cannot contribute a later expert-success segment
[ ] Held-out instances and targets remain isolated
```

Manual verification is required for a stratified sample of at least six
replays: one per object family and at least one per target type. Pass when the
visible phase transitions and final placement agree with the saved labels.
Stop for user confirmation before marking 4.5 complete.

---

## Level 4.6 — Failures and Corrective Demonstrations

### Goal

Preserve representative natural failures and record safe operator corrections
without contaminating expert-only training data or requiring learned recovery.

### Files

```text
dexvision/logging/corrective_demos.py
dexvision/apps/record_correction.py
dexvision/evaluation/correction_summary.py
tests/test_corrective_demos.py
tests/test_correction_summary.py
```

### Required behavior

```text
link every correction to its triggering episode and source category
require source policy/checkpoint only when source category is policy_rollout;
store null plus a stable not-applicable reason for teleoperation/scripted sources
record failure class, retryability, intervention start/end, and outcome
preserve pre-intervention frames and original terminal result
separate expert, ordinary failure, policy rollout, and correction streams
never promote unsafe motion into a correction target
```

Required failure classes are approach miss, wrong contact, failed acquisition,
slip/drop, placement miss, premature release, timeout, workspace violation, and
joint-limit violation. Unsafe failures are abort-only and must not be replayed
as targets.

### Commands

```bash
python -m dexvision.apps.record_correction --config configs/level4_dataset.yaml --source-rollout <episode-directory>
conda run -n dexvision pytest -q tests/test_corrective_demos.py tests/test_correction_summary.py
```

### Pass criteria

```text
[ ] At least 30 failure/correction episodes meet the frozen category coverage
[ ] Corrections can be included or excluded deterministically
[ ] Correction provenance validates conditionally by source category
[ ] Failure, retry, correction, and final outcome are not conflated
[ ] Workspace/joint-limit failures are abort-only
[ ] Baseline and later correction-trained comparisons can use identical splits
```

Manual verification: inspect one corrected pick/place replay. Pass when the
failure and intervention boundary are visibly correct and the original failure
remains recoverable from the saved metadata. Stop for user confirmation.

---

## Level 4.7 — Rendered Visual Grounding Dataset

### Goal

Add aligned single-camera visual supervision for object detection, identity,
and metric pose estimation while retaining simulator truth as the reference.

### Files

```text
dexvision/perception/render_annotations.py
dexvision/logging/visual_stream.py
dexvision/apps/export_visual_dataset.py
configs/level4_visual_dataset.yaml
tests/test_render_annotations.py
tests/test_visual_alignment.py
```

### Required streams and annotations

```text
RGB from one named fixed camera at a frozen sampling stride
frame timestamp and source episode/frame index
object id, class id, visibility, 2D box, segmentation mask, and 6D pose
camera intrinsics/extrinsics and image size
train/validation/test ownership inherited from the source episode
asset source/license metadata
frozen visual-condition id: nominal, mild illumination, partial occlusion,
or bounded workcell distractor
```

The export must not duplicate images across splits. A compact VLM may later
select an object semantically, but Level 4 does not download or train a VLM.
Camera pose and intrinsics remain fixed, and no cross-camera or real-world
transfer claim is made.

### Commands

```bash
python -m dexvision.apps.export_visual_dataset --config configs/level4_visual_dataset.yaml --dataset-dir data/demos/level4 --output-dir data/visual/level4
conda run -n dexvision pytest -q tests/test_render_annotations.py tests/test_visual_alignment.py
```

### Pass criteria

```text
[ ] Every exported frame maps to one source episode and state timestamp
[ ] Boxes, masks, poses, ids, camera calibration, and visibility are complete
[ ] Occluded/out-of-frame objects follow the frozen annotation rule
[ ] Visual splits inherit episode/session/condition isolation
[ ] Every frozen visual-condition cell meets its split-owned coverage minimum
[ ] Export size and storage plan are reported before release
```

Manual verification is required for a contact sheet sampled across all object
families and splits. Pass when masks/boxes align visually, ids are correct, and
no test scene appears in training. Stop for user confirmation.

---

## Level 4.8 — Dataset Audit and Frozen Split Manifests

### Goal

Prove that the complete dataset satisfies its schema, coverage, quality,
provenance, and leakage requirements before packaging.

### Files

```text
dexvision/evaluation/dataset_audit.py
dexvision/evaluation/split_audit.py
dexvision/apps/audit_level4_dataset.py
configs/level4_splits.yaml
docs/level4_dataset_report.md
tests/test_level4_dataset_audit.py
tests/test_level4_split_audit.py
```

### Required reports

```text
episodes and segments by skill, phase, session, object, target, and outcome
coverage cells and minima separately for train, validation, and test
quality/rejection counts and reasons
failure/correction coverage
visual annotation completeness
train/validation/test manifests and per-file SHA-256
dataset and config digests
known imbalance, missing cells, and unsupported claims
```

### Commands

```bash
python -m dexvision.apps.audit_level4_dataset --config configs/level4_dataset.yaml --splits configs/level4_splits.yaml --dataset-dir data/demos/level4 --output-dir outputs/level4/audit
conda run -n dexvision pytest -q tests/test_level4_dataset_audit.py tests/test_level4_split_audit.py
```

### Pass criteria

```text
[ ] Frozen minimum counts and required coverage cells pass
[ ] No episode, session, held-out condition, object instance, goal, or image leaks
[ ] Every accepted episode passes schema, quality, and recomputed-task checks
[ ] Training-only normalization inputs are explicitly identified
[ ] All shortages and biases are visible; none are repaired by silent duplication
```

Manual verification: none. A failing audit creates a versioned collection
amendment; it does not permit editing accepted episodes in place.

---

## Level 4.9 — Immutable Dataset Release and Level 5 Handoff

### Goal

Publish a reproducible Level 4 release while preserving the Level 2 archive and
editable operator workspace.

### Files

```text
datasets/level4-v1/README.md
datasets/level4-v1/manifest.json
datasets/level4-v1/SHA256SUMS
datasets/level4-v1/splits/
docs/level4_dataset_report.md
dexvision/apps/verify_dataset_release.py
tests/test_level4_release.py
```

### Storage rule

Use Git LFS only when the bounded archive fits the documented host quota. If
RGB makes the payload too large, Git stores manifests, splits, checksums,
licenses, and retrieval instructions while immutable external object storage
holds the payload. Never commit mutable `data/demos/` contents or silently omit
a required stream.

### Commands

```bash
python -m dexvision.apps.verify_dataset_release --release-dir datasets/level4-v1
conda run -n dexvision pytest -q tests/test_level4_release.py tests/test_level4_split_audit.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

### Pass criteria

```text
[ ] Clean-clone retrieval and SHA-256 verification are documented and tested
[ ] Release, schema, config, split, and source-code versions are immutable
[ ] Level 2 and Level 4 releases remain independently retrievable
[ ] Licenses/provenance and storage quota/recovery procedure are documented
[ ] Level 5 receives exact observation/action/goal layouts and frozen splits
[ ] Limitations and failed coverage claims are published honestly
```

Manual verification: perform the documented clean-directory retrieval and
verification command. Pass only when every expected artifact is restored and
all checksums match. Stop for user confirmation before completing Level 4.

---

# Level 4 Completion Checklist

```text
[x] 4.0 requirements and Level 3 failure traceability are frozen
[x] 4.1 one resettable workcell and typed world state pass manual inspection
[x] 4.2 session-aware append-only schema and phase labels pass
[ ] 4.3 pilot collection freezes final counts; dial is promoted or deferred
[ ] 4.4 reach, push, and press coverage passes across genuine sessions
[ ] 4.5 complete pick/place coverage and phase replays pass
[ ] 4.6 failures and corrections remain separate and auditable
[ ] 4.7 single-camera visual annotations and alignment pass
[ ] 4.8 coverage, quality, provenance, and leakage audits pass
[ ] 4.9 immutable release restores and verifies from a clean directory
[ ] Four sessions remain split-owned; per-cell minima and visual conditions are audited
[ ] Requested, commanded, and applied actions plus causal phases are reconstructable
[ ] No full-scale training, policy qualification, LLM, or future extension is claimed
```
