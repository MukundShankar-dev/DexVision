# Progress Level 4 — Comprehensive Skill Dataset

Level 4 goal:

> Turn the Level 3 learning proof into a broad, versioned, multi-session
> dataset that can support full-scale reusable-skill learning in Level 5.

Level 3 answers whether the current Level 2 data and learning pipeline can
produce useful policies. It does not establish a comprehensive skill dataset.
Level 4 is the deliberate data haul: expand task coverage, collect independent
sessions, record failures and operator corrections, add visual grounding data,
audit coverage, and publish an immutable release. Full-scale policy training, skill
qualification, skill cards/runtime, and composed pilots belong to Level 5.

Level 4 must remain incremental. Every numbered section below is a separate
checkpoint. Do not begin bulk collection until its task, metadata, coverage,
and acceptance rules are frozen.

---

## Level 4 Scope and System Boundary

The intended stack separates semantic decisions from continuous control:

```text
language request
  -> symbolic task plan (future Level 7)
  -> object ids, target ids, and typed skill calls
  -> deterministic supervisor and world-state checks
  -> learned Level 5 skill policy
  -> bounded base/wrist/finger actions
```

For visual operation, semantic selection and metric localization are also
separate concerns:

```text
camera/rendered image
  -> object detection/segmentation or simulator ground truth
  -> object pose and stable object id
  -> optional compact VLM for semantic disambiguation
  -> typed skill parameters
```

A VLM may help choose which visible object matches a phrase. It must not emit
high-rate actuator commands. In simulation, ground-truth object state remains
the control and evaluation reference while visual perception is developed and
measured independently.

### Core operational skills

```text
reach_object(object_id, approach_pose)
pick_object(object_id)
place_held_object(target_pose_or_receptacle)
push_object_to_target(object_id, target_zone)
press_button(button_id)
```

`rotate_dial(dial_id, target_angle)` is an optional sixth skill. Promote it only
if Level 4.0 pilots show that it fits the same hand/workspace limits without
displacing the five required skills.

Grasp, lift, hold, transport, release, and slide are useful phase labels and
success measurements inside the operational skills above; they are not
separate policies that must each be collected, trained, and orchestrated.
This keeps the project small enough to finish while preserving the state
transitions needed for diagnosis.

`reach_touch_target`, `button_press`, and `push_cube_to_target` from Level 2 are
seed tasks and regression baselines. Level 4 generalizes them across objects,
poses, targets, and genuinely separate recording sessions.

### Explicitly deferred extensions

The first complete project does not require:

```text
general grasping of arbitrary objects
regrasp_object or recover_dropped_object learned policies
learned retry behavior
open_hinged_lid
tool use, cutting, pouring, liquids, or deformable objects
arbitrary household or open-world scenes
end-to-end VLM control
```

Initial failure handling is deterministic supervisor logic: detect failure,
move to a safe pose, retry once when the failure is retryable, then abort and
report the reason. Corrective demonstrations may improve a core policy, but
learned recovery is not a Level 4 dataset requirement.

---

## Level 4.0 — Comprehensive Dataset Scope Freeze

### Goal

Freeze the first comprehensive release plan before creating tasks or recording
hundreds of episodes. This checkpoint defines the skills the data must support;
it does not train those skills.

### Files

```text
docs/level4_dataset_plan.md
configs/level4_dataset.yaml
tests/test_level4_dataset_plan.py
```

### Required decisions

```text
core and optional skill list
object families and object/target identifiers
train/validation/test condition matrix
minimum independent recording sessions
minimum accepted episodes per operational skill or complete sequence
failure and correction sampling targets
required RGB/state/action streams
data-quality gates and planned Level 5 policy-qualification handoff
dataset release name and schema versions
```

Level 4.0 freezes exact counts only after a small pilot estimates task
variability and collection cost. The planning envelope is roughly 250–350 new
accepted episodes in total, collected across at least three genuine sessions,
three simple rigid-object families, and three or four target
zones/receptacles. Reuse compatible Level 2 reach, press, and push episodes as
clearly labeled legacy seed data. About 100–150 complete pick/place sequences
may supply phase labels for acquisition, lift, transport, placement, and
release; do not demand a separate 100 episodes for every internal phase.
Coverage and genuinely independent sessions matter more than inflating the
episode count. Splits must reserve complete sessions, object instances, and
goal regions where the intended claim requires them.

### Pass criteria

```text
[ ] Every core skill has a typed goal and executable success metric
[ ] Coverage cells and collection counts are machine-readable
[ ] Session-held-out and condition-held-out evaluation are defined
[ ] Correction/failure data remain distinguishable from expert successes
[ ] No dataset collection has started under an unfrozen schema
```

---

## Level 4.1 — Object Catalog, World State, and Perception Contract

### Goal

Define how tasks, policies, perception, and a future planner refer to the same
objects and targets.

### Files

```text
dexvision/sim/world_state.py
dexvision/perception/object_observations.py
docs/world_state_contract.md
tests/test_world_state_contract.py
```

### Contract

Each visible or simulated entity should expose:

```text
stable object_id and class_id
pose, velocity, units, and coordinate frame
geometry/size metadata needed by the skill
grasped/held/support/receptacle relationships when known
source: simulator_ground_truth, detector, tracker, or operator label
confidence and timestamp for inferred state
```

Start with MuJoCo ground truth. Add a detector/segmenter baseline before using
a VLM for object selection. A compact local VLM is optional for phrases such
as "the red block beside the cup"; metric pose and safety checks still come
from the grounded object-state layer.

### Pass criteria

```text
[ ] Stable ids survive one episode and reset predictably
[ ] Frames and units are explicit
[ ] Stale or ambiguous observations fail clearly
[ ] Ground-truth and inferred observations share one typed schema
[ ] Tests do not require a camera, GPU, or downloaded model
```

---

## Level 4.2 — Collection and Provenance Upgrade

### Goal

Make new data capable of supporting real generalization claims.

### Required metadata

```text
recording_session_id, mandatory for new Level 4 episodes
operator_id or anonymous stable operator label
task/skill and goal parameters
object instances and sampled initial state
camera/render configuration and calibration version
teleoperation, policy, or correction source
intervention/failure reason when applicable
schema/config/code revision and random seed
timestamps synchronized across images, state, and actions
```

Do not rewrite Level 2 episodes to invent session ids. They remain valid
Level 3 feasibility data and may be incorporated as a clearly labeled legacy
source.

### Pass criteria

```text
[ ] New episode schema records genuine session provenance
[ ] Recording resumes safely without duplicate episode ids
[ ] RGB capture is optional but frame alignment is validated when enabled
[ ] Dataset summaries report coverage by session, object, goal, and source
[ ] Raw episodes remain immutable after acceptance
```

---

## Level 4.3 — Reach and Approach Dataset Haul

### Goal

Collect a goal-conditioned `reach_object` dataset across object classes,
positions, orientations, and approach directions.

### Coverage

```text
multiple object sizes and simple geometries
interior and boundary workspace positions
nominal and perturbed initial hand poses
clear and partially occluded visual views when RGB is recorded
successful reaches plus separately labeled near-miss corrections
```

### Pass criteria

```text
[ ] Pilot task and quality gates pass before scale-up
[ ] Frozen coverage/count target is met
[ ] Entire sessions and held-out object/goal conditions remain untouched
[ ] Replay and recomputed labels pass for every accepted episode
[ ] Dataset release candidate report is reproducible
```

---

## Level 4.4 — Pick-Object Dataset Haul

### Goal

Collect `pick_object` sequences for simple rigid objects. A sequence includes
the acquisition, lift, and brief stability phases required to finish with a
verified held object.

### Required variation

```text
object size, shape, mass, and friction families
approach direction and pre-grasp pose
target lift height and short stability duration
slip, failed acquisition, and drop outcomes
```

Internal grasp, lift, and hold boundaries must be labeled for diagnosis, but
they do not create three separately exposed policies. The frozen plan may pair
a successful pick with a later placement in one complete demonstration when
the phase boundary and held-object state are reconstructable.

### Pass criteria

```text
[ ] Physical contact and object motion define success, not operator labels alone
[ ] Drop/slip outcomes are retained with explicit labels
[ ] Object-held state and internal phase boundaries can be recomputed from saved data
[ ] Frozen coverage/count target is met across independent sessions
```

---

## Level 4.5 — Place-Held-Object Dataset Haul

### Goal

Collect `place_held_object` sequences from a verified held-object precondition
through transport, placement, release, and post-release stability.

### Coverage

```text
source and destination regions
object/receptacle combinations
marked target zones and simple receptacles
different initial grasp states within the valid precondition envelope
clean release, premature release, and misplacement outcomes
```

### Pass criteria

```text
[ ] Preconditions require an actually held object
[ ] Target pose/receptacle ids are typed and saved
[ ] Transport retention, placement tolerance, and post-release stability are executable metrics
[ ] Frozen coverage/count target is met across independent sessions
```

---

## Level 4.6 — Push, Press, and Optional Dial Dataset Haul

### Goal

Generalize the Level 2 contact tasks beyond one cube, fixed fixtures, and one
collection condition.

### Coverage

```text
push objects with varied size, mass, friction, start, and target
multiple button ids and press depths
multiple dial starts and target angles only if the optional skill was promoted
contact-loss and wrong-contact outcomes
```

The existing Level 2 `button_press` and `push_cube_to_target` episodes remain
legacy training data. New Level 4 episodes must add genuine session and object
variation rather than duplicating the original distribution.

### Pass criteria

```text
[ ] Every skill has held-out sessions and conditions
[ ] Success labels recompute from saved physical state
[ ] Contact and safety metrics are reported
[ ] Frozen coverage/count targets are met
```

---

## Level 4.7 — Failures and Corrective Demonstrations

### Goal

Record representative policy drift, failed approaches, unstable picks,
misplacements, and safe operator corrections without contaminating the
expert-success set. This checkpoint improves diagnosis and may improve the
core policies; it does not create learned recovery skills.

### Rules

```text
preserve the original expert datasets
record policy/checkpoint and failure trigger
record intervention start and end
label whether the correction is safe and whether it succeeds
keep evaluation failures out of the training set until a new protocol version
```

### Pass criteria

```text
[ ] Corrections can be included/excluded deterministically
[ ] Common failure modes have measurable correction coverage
[ ] Baseline and correction-trained policies use identical evaluation conditions
[ ] Retryability is part of the terminal result
```

---

## Level 4.8 — Visual Grounding Dataset and Integrity Checks

### Goal

Record the aligned visual supervision that Level 5 will use to connect object
language and pixels to object ids and poses.

### Build order

```text
1. Export rendered RGB plus simulator boxes, masks, poses, and stable ids.
2. Validate image/state/action alignment and annotation completeness.
3. Reserve scene, object-instance, and session-held-out visual splits.
4. Run non-learning integrity checks and simple deterministic smoke baselines.
```

Level 5 may train a detector/tracker plus an optional compact VLM for semantic
selection. Level 4 must not use a model result to hide incomplete or misaligned
annotations.

### Pass criteria

```text
[ ] Visual labels derive from known simulator state or reviewed annotations
[ ] Object ids remain aligned across frames
[ ] Held-out visual splits are frozen before Level 5 training
[ ] Annotation completeness and alignment metrics are reported
[ ] External asset sources and licenses are documented
```

---

## Level 4.9 — Versioned Comprehensive Dataset Release

### Goal

Publish an immutable, documented Level 4 dataset release without replacing the
Level 2 snapshot.

### Required artifacts

```text
Git LFS archive or documented external artifact storage when too large
SHA-256 checksum
machine-readable manifest
episode/session/condition split manifests
schema and migration notes
quality, coverage, and known-bias report
license/provenance statement for every external asset
```

Git LFS is appropriate for a bounded release that fits repository hosting
quotas. If synchronized images make the release too large, keep manifests and
checksums in Git and use versioned external object storage for the immutable
payload. Do not silently omit large streams or commit mutable working data.

### Pass criteria

```text
[ ] Clean-clone download and checksum verification are documented
[ ] Split leakage tests pass
[ ] Every release artifact is immutable and versioned
[ ] Storage cost/quota and recovery procedure are documented
```

---

# Level 4 Completion Checklist

```text
[ ] Comprehensive dataset scope and coverage are frozen
[ ] New data has genuine session provenance
[ ] Core reach, pick, place-held, push, and press data is collected
[ ] Optional dial data exists only if Level 4.0 promoted it
[ ] Failures and corrective data are separately labeled
[ ] Visual grounding data and integrity checks exist
[ ] Immutable Level 4 dataset release is reproducible
[ ] Session/object/goal/visual split manifests are frozen for Level 5
[ ] Dataset limitations, biases, and missing coverage are reported honestly
[ ] No full-scale policy training or skill qualification is claimed
```
