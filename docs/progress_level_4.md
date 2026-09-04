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
[ ] Missing or duplicate session ids fail clearly
[ ] Phase transitions are valid and reconstructable from saved indices/state
[ ] Online phases use no future frames and disagreements with annotations are reportable
[ ] A complete pick/place episode yields compatible reach, pick, and place segments
[ ] Requested, commanded, and applied actions reproduce safety handling exactly
[ ] RGB is optional, but enabled frames align to timestamps and state
[ ] Legacy Level 2 episodes still load without invented Level 4 fields
[ ] Resume behavior never overwrites an existing episode
```

Manual verification: none; use synthetic episodes.

---

## Level 4.3 — Collection Pilot and Final Coverage Freeze

### Goal

Prove that every required task is recordable and measurable before the bulk
data haul, then replace provisional counts with a frozen coverage matrix.

### Files

```text
dexvision/logging/level4_collection.py
dexvision/evaluation/dataset_coverage.py
dexvision/apps/summarize_level4_coverage.py
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
[ ] 4.2 session-aware append-only schema and phase labels pass
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
