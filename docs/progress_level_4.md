# Progress Level 4 — Comprehensive Skill Dataset and Skill Library

Level 4 goal:

> Turn the Level 3 learning proof into a broad, versioned, multi-session
> dataset and a qualified library of reusable manipulation skills.

Level 3 answers whether the current Level 2 data and learning pipeline can
produce useful policies. It does not establish a comprehensive skill library.
Level 4 is the deliberate data haul: expand task coverage, collect independent
sessions, record recovery behavior, add visual grounding data, retrain the
successful Level 3 approach, and qualify each skill behind one typed runtime.

Level 4 must remain incremental. Every numbered section below is a separate
checkpoint. Do not begin bulk collection until its task, metadata, coverage,
and acceptance rules are frozen.

---

## Level 4 Scope and System Boundary

The intended stack separates semantic decisions from continuous control:

```text
language request
  -> symbolic task plan (future Level 6)
  -> object ids, target ids, and typed skill calls
  -> deterministic supervisor and world-state checks
  -> learned Level 4 skill policy
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

### Core skill families

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

`reach_touch_target`, `button_press`, and `push_cube_to_target` from Level 2 are
seed tasks and regression baselines. Level 4 generalizes them across objects,
poses, targets, and genuinely separate recording sessions.

### Candidate extensions

Only promote these after the core families qualify:

```text
align_object
stack_object
open_hinged_lid
close_hinged_lid
tool_use_simple, such as a guided scoop or slide
```

Real tomato cutting is not a core pilot. Cutting deformable food introduces
fracture/deformation modeling, blade contact, force control, tool safety, and
often arm-level motion that the present hand-only setup does not validate. A
later experiment may move a guided knife through a pre-scored or pre-segmented
rigid proxy, but it must be described as a cutting proxy, not autonomous food
preparation.

---

## Level 4.0 — Dataset and Skill Scope Freeze

### Goal

Freeze the first comprehensive release plan before creating tasks or recording
hundreds of episodes.

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
minimum accepted episodes per skill family
failure and recovery sampling targets
required RGB/state/action streams
quality and policy-qualification gates
dataset release name and schema versions
```

The default planning floor is 100 accepted episodes per core skill family,
collected across at least three genuine sessions. Change that floor only after
a documented pilot estimates task variability and collection cost. Coverage is
more important than inflating episode count: splits must reserve complete
sessions, object instances, and goal regions where the intended claim requires
them.

### Pass criteria

```text
[ ] Every core skill has a typed goal and executable success metric
[ ] Coverage cells and collection counts are machine-readable
[ ] Session-held-out and condition-held-out evaluation are defined
[ ] Recovery/failure data remain distinguishable from expert successes
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

## Level 4.4 — Grasp, Hold, and Lift Dataset Haul

### Goal

Collect object-conditioned acquisition and retention data for simple rigid
objects.

### Required variation

```text
object size, shape, mass, and friction families
approach direction and pre-grasp pose
grasp family when explicitly supported
target lift height and hold duration
slip, failed acquisition, and drop outcomes
```

Grasp, hold, and lift may share episodes, but labels and terminal-state
boundaries must allow each skill to be trained and evaluated independently.

### Pass criteria

```text
[ ] Physical contact and object motion define success, not operator labels alone
[ ] Drop/slip outcomes are retained with explicit labels
[ ] Object-held state can be recomputed from saved data
[ ] Frozen coverage/count target is met across independent sessions
```

---

## Level 4.5 — Transport, Place, and Release Dataset Haul

### Goal

Collect the object-in-hand transitions needed to complete useful tasks.

### Coverage

```text
source and destination regions
object/receptacle combinations
free placement, constrained placement, and stacking-alignment subsets
different initial grasp states within the valid precondition envelope
clean release, premature release, and recoverable misplacement
```

### Pass criteria

```text
[ ] Preconditions require an actually held object
[ ] Target pose/receptacle ids are typed and saved
[ ] Placement tolerance and post-release stability are executable metrics
[ ] Frozen coverage/count target is met across independent sessions
```

---

## Level 4.6 — Push, Slide, Press, and Rotate Dataset Haul

### Goal

Generalize the Level 2 contact tasks beyond one cube, fixed fixtures, and one
collection condition.

### Coverage

```text
push/slide objects with varied size, mass, friction, start, and target
multiple button ids and press depths
multiple dial starts and target angles
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

## Level 4.7 — Corrective and Recovery Demonstrations

### Goal

Record how to recover from policy drift, failed approaches, unstable grasps,
drops, and near-limit states without contaminating the expert-success set.

### Rules

```text
preserve the original expert datasets
record policy/checkpoint and failure trigger
record intervention start and end
label whether recovery is safe and whether it succeeds
keep evaluation failures out of the training set until a new protocol version
```

### Pass criteria

```text
[ ] Corrections can be included/excluded deterministically
[ ] Common failure modes have measurable recovery coverage
[ ] Baseline and recovery-trained policies use identical evaluation conditions
[ ] Retryability is part of the terminal result
```

---

## Level 4.8 — Visual Grounding Dataset and Baseline

### Goal

Connect object language and pixels to the object ids and poses used by skills.

### Build order

```text
1. Export rendered RGB plus simulator boxes/masks/poses.
2. Validate image/state/action alignment.
3. Train or integrate a conventional detector/segmenter baseline.
4. Measure localization and identity tracking separately from motor policy.
5. Optionally test a compact local VLM for semantic selection/disambiguation.
```

The default architecture is detector/tracker plus a normal language model or
small VLM for high-level selection. Do not make a VLM responsible for precise
poses or direct hand control unless an explicit experiment demonstrates that
contract safely and reproducibly.

### Pass criteria

```text
[ ] Visual labels derive from known simulator state or reviewed annotations
[ ] Object ids remain aligned across frames
[ ] Detection/selection metrics are reported on held-out scenes
[ ] Motor-policy results identify whether they use ground truth or perception
[ ] GPU/model requirements and licenses are documented
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

## Level 4.10 — Retrain and Qualify the Skill Library

### Goal

Apply the Level 3 learning result to each core skill family and decide whether
it is qualified, experimental, or failed.

### Required evaluation

```text
offline session-held-out action prediction
closed-loop task success
held-out object and goal conditions
action smoothness and safety limits
failure reason distribution
perception-grounded rollout where applicable
comparison with the Level 3 baseline
```

Use the simplest Level 3 model that worked. Temporal or action-chunked models
are justified only when the measured Level 3/4 failure mode calls for them.

### Pass criteria

```text
[ ] Every core skill has a frozen evaluation protocol
[ ] Results distinguish ground-truth-state and visual-perception inputs
[ ] Failed skills are reported rather than hidden
[ ] Qualified skills meet explicit numerical gates on held-out conditions
```

---

## Level 4.11 — Skill Cards, Registry, and Supervised Executor

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
[ ] Registry excludes incompatible or unqualified policies by default
[ ] Preconditions, timeouts, safety limits, and terminal states are enforced
[ ] Repeated request ids do not duplicate physical actions
[ ] No LLM, web API, or multi-skill planner is implemented
```

---

## Level 4.12 — Diverse Scripted Pilot Tasks

### Goal

Prove that the skill library supports more than one hand-authored showcase
before Level 6 adds language planning.

### Core pilots

```text
1. Sort and pack: move varied objects into matching bins/receptacles.
2. Clear the workspace: push reachable objects together, then pick/place them
   into a tray while handling at least one recoverable miss.
3. Operate a control panel: press specified buttons and rotate a dial to target
   states in a given order.
4. Assemble a simple sandwich proxy: stack rigid bread/filling props on a plate
   in a specified order; no deformable food, spreading, or cutting claim.
```

Candidate fifth pilot:

```text
Set a snack place: position a cup, plate, and utensil proxy at named target
poses. This emphasizes precise placement and object identity rather than the
same stacking behavior as sandwich assembly.
```

At least three materially different pilots must pass. Each pilot uses a
deterministic scripted plan and the same typed skill calls intended for future
Level 6 orchestration. A pilot fails if a human silently repairs state between
skills.

### Pass criteria

```text
[ ] At least three diverse pilots run from reset through explicit termination
[ ] Every step records the skill request, result, and world-state transition
[ ] Recovery and abort behavior are visible and deterministic
[ ] Per-pilot success rates and failure reasons are reported
[ ] Sandwich assembly is one pilot, not the project's sole final claim
```

---

# Level 4 Completion Checklist

```text
[ ] Comprehensive dataset scope and coverage are frozen
[ ] New data has genuine session provenance
[ ] Core reach/grasp/lift/transport/place/release data is collected
[ ] Push/slide/press/rotate data is generalized
[ ] Corrective and recovery data is separately labeled
[ ] Visual grounding data and a measured baseline exist
[ ] Immutable Level 4 dataset release is reproducible
[ ] Core policies are evaluated on held-out sessions and conditions
[ ] Qualified skills run through one supervised executor
[ ] At least three diverse scripted pilots pass
[ ] Limitations and failed skills are reported honestly
```
