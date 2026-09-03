# Progress Level 2 — Demonstration Recording, Replay, Data Quality, and Retargeting Benchmarks

Level 2 goal:

> Turn the Level 1 teleoperation demo into a reproducible skill-demonstration,
> replay, validation, and benchmarking system.

This level is about engineering maturity: defining resettable MuJoCo task
environments, recording skill demonstrations, replaying them, validating them,
filtering bad data, summarizing per-skill datasets, and comparing retargeting
methods.

Level 2 does not train policies. Its output is a set of reliable skill-demo
datasets that Level 3 can use to train reusable low-level robot skill policies.
Future Level 5 orchestration may consume those policies, but Level 5 is not
implemented in this repo.

---

## Level 2 Prerequisite

Serious Level 2 demo recording is blocked until Level 1.8B — Final Hand Model
Decision and Separation is complete, and structured manipulation demos should
use the completed Level 1.13D full teleoperation action space.

The final robot hand defines the action space: joint names, actuator names,
action dimensions, joint limits, neutral pose, and the meaning of recorded
controls. Changing the hand model after collecting demonstrations can invalidate
saved demos, replay results, benchmark comparisons, and trained Level 3 policies.

Use the debug hand only for smoke tests until `docs/robot_hand_model.md`
documents the final hand choice and the scripted gestures pass on that final
model.

Level 1.13D adds the remaining teleoperation controls needed for task
demonstrations: base x/y translation, depth/in-out control, relative wrist/palm
orientation, and finger bend control. Level 2 recordings should preserve that
full action space instead of falling back to finger-only actions.

Warning:
Do not record serious skill demos until the task environment, action schema,
robot model, and base/finger teleop interface are stable. Changing the action
space, task state schema, or robot model after collection can invalidate demos,
replay results, relabeling, Level 3 policies, and future skill cards.

---

## Level 2.0 — Task Board Environment and Task Set Design

### Goal

Define the staged MuJoCo skill-learning task board environment and initial
resettable task set before recording demonstrations.

This is a planning checkpoint only. It documents what Level 2 will record as
skill-demo data and keeps Level 3 learning and future skill orchestration out
of scope.

### Files

```text
docs/progress_level_2.md
docs/task_environment.md
docs/level5_future.md
docs/CURRENT_STATUS.md
```

### Build

Create a task environment design document that defines:

```text
tabletop workspace
Shadow Hand with base pose control
simple objects
resettable tasks
fixed success metrics
initial skill/task set and per-task schemas
demonstration requirements
future orchestration relevance
```

Initial task set:

```text
free_space_gesture
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

Each task should document:

```text
task purpose
required objects
initial state
action space
observation/state fields
demonstration requirements
success condition
failure condition
max episode length
why it matters for future orchestration
whether it is early/core or later/stretch
```

### Run

```bash
git diff --check
rg -n "Level 2.0|free_space_gesture|reach_touch_target|reach_object|button_press|push_cube_to_target|grasp_object|pinch_lift_object|place_object|release_object|rotate_dial" docs/progress_level_2.md docs/task_environment.md docs/CURRENT_STATUS.md
```

### Pass Criteria

```text
[x] Level 2.0 is listed before Level 2.1
[x] Skill-learning task board environment is documented
[x] Initial skill/task set is documented with success/failure criteria
[x] Demonstration requirements and future orchestration relevance are documented
[x] Level 2.1 schema notes require the full Level 1.13 action space
[x] Future Level 5 skill orchestration is documented as out of scope for this repo
[x] No Level 3 learning or Level 5 orchestration code is implemented
```

### Codex Prompt

```text
Add the Level 2.0 planning checkpoint.
Create docs/task_environment.md with the staged tabletop skill-learning task board and initial task set.
Document future Level 5 skill orchestration as a separate repo concern.
Update CURRENT_STATUS so Level 2.0 is the next target.
Do not implement demo recording, learning, or skill orchestration.
```

---

## Level 2.1 — Demo Episode Schema

### Goal

Define the exact format for one recorded demonstration.

Each Level 2 demo should be tied to a `task_id` and `skill_name` so Level 3 can
train per-skill policies from the same data format.

### Files

```text
dexvision/logging/dataset_schema.py
tests/test_dataset_schema.py
docs/module_contracts.md
```

### Build

Create:

```python
@dataclass
class DemoEpisode:
    metadata: dict
    landmarks: np.ndarray | None
    features: np.ndarray
    actions: np.ndarray
    robot_states: np.ndarray
    object_states: np.ndarray | None
    task_states: np.ndarray | None
    tracking_quality: np.ndarray
    timestamps: np.ndarray
    success: bool | None
```

Schema notes:

```text
actions must record the full Level 1.13 teleoperation command, not only finger
targets.

At minimum, each timestep must be recoverable as:
- base_position_target, shape [3]
- base_orientation_target, MuJoCo wxyz quaternion, shape [4]
- finger_actuator_targets, shape [N_finger_actuators]

robot_states must include:
- robot qpos
- robot qvel
- commanded mocap/base target when used by the scene

object_states and task_states must include:
- object pose/velocity for object tasks
- task target state, such as target position, button state, dial angle, or goal zone
- success metric inputs needed to recompute the episode result

tracking_quality must include:
- hand detected flag
- anatomical handedness, when available
- hand tracking confidence
- feature/control confidence
- optional dropped-frame or reacquire markers

metadata must include:
- skill name
- task name
- task id/environment id
- episode id
- action schema version
- observation schema version
- robot model path or model id
- retargeter/config name
- control rate
- Level 1 teleop config snapshot
- task config snapshot
- dependency/version notes when available
```

Validation checks:

```text
matching time dimension
timestamps monotonic
no NaNs in required arrays
actions have correct dimension
metadata has required fields
tracking_quality has matching time dimension
action schema can reconstruct base position, base orientation, and finger targets
observation/state schema can reconstruct robot, hand/base, object, target, and task state
success metric inputs are present for task-specific relabeling
```

### Run

```bash
pytest tests/test_dataset_schema.py
```

### Pass Criteria

```text
[x] Valid demo passes validation
[x] Invalid time dimensions are caught
[x] NaNs are caught
[x] Missing metadata is caught
[x] Full Level 1.13 action schema is required
[x] Tracking quality is validated
[x] Skill/task identifiers are required
[x] Success metric inputs are validated when required by the task
[x] Test uses synthetic arrays
```

### Codex Prompt

```text
Implement dexvision/logging/dataset_schema.py.
Create a DemoEpisode dataclass plus validate_demo, save_demo, and load_demo stubs if appropriate.
Add tests with synthetic arrays.
Require skill/task metadata and the full Level 1.13 action schema.
Do not add live recording yet.
```

---

## Level 2.2 — Demo Logger

### Goal

Record Level 1 teleop runs as skill-demo episodes on disk.

### Files

```text
dexvision/logging/demo_logger.py
dexvision/apps/record_demo.py
tests/test_demo_logger.py
```

### Record

Each demo directory should contain:

```text
metadata.json
features.npy
actions.npy
robot_states.npy
timestamps.npy
landmarks.npy, optional
object_states.npy, optional
task_states.npy, optional
tracking_quality.npy
camera.mp4, optional
```

Recorded episodes must preserve:

```text
base_position_target
base_orientation_target
finger_actuator_targets
robot qpos/qvel
hand/base pose
object pose/velocity
target pose
task state
tracking quality
success metric inputs
metadata/config snapshot
```

### Run

```bash
mjpython -m dexvision.apps.record_demo --task free_space_gesture --retargeter curl --output data/demos/free_space_gesture_attempt_001 --level1-13-full
pytest tests/test_demo_logger.py
```

### Pass Criteria

```text
[x] Demo directory created
[x] Metadata saved
[x] Arrays saved
[x] Arrays have same T
[x] Recording can start/stop cleanly
[x] No giant video required by default
```

### Codex Prompt

```text
Implement demo recording only.
Add dexvision/logging/demo_logger.py and dexvision/apps/record_demo.py.
The app should wrap the existing Level 1 teleop loop and save features, actions, robot states, timestamps, metadata, and optional landmarks.
Actions must include base_position_target, base_orientation_target, and finger_actuator_targets.
Record skill_name/task_id and task state when available.
Do not add replay, filtering, or learning yet.
```

---

## Level 2.2B — Dataset Collection Runbook and Tracker

### Goal

Create the practical Level 2 dataset collection runbook and tracker that
operators can follow before recording serious demo datasets, and define when
Level 2 data is complete enough to move into Level 3 behavior cloning.

This is a docs-only checkpoint. It wraps the Level 2 collection workflow
without implementing replay, filtering, learning, or Level 5 orchestration.

### Files

```text
docs/progress_level_2.md
docs/level2_dataset_runbook.md
docs/CURRENT_STATUS.md
docs/codex_prompts.md
```

### Build

Create a runbook that documents:

```text
Level 2 purpose
recorded episode fields
full Level 1.13 action schema
dataset folder naming convention
current and future task recording commands
per-task demo targets
per-demo recording checklist
manual quality checklist
replay/validation/filter command placeholders
Level 3 readiness criteria
Level 2 completion tracker table
explicit do-not-do-yet guidance
```

Recommended dataset layout:

```text
data/demos/
  raw/<task_id>/<YYYY-MM-DD>_<attempt_number>/
  processed/<task_id>/
  reports/quality/
  reports/summaries/
```

The runbook must preserve the full Level 1.13 action guidance:

```text
base_position_target
base_orientation_target
finger_actuator_targets
```

### Run

```bash
git diff --check
rg -n "level2_dataset_runbook|Dataset Collection Runbook|base_position_target|base_orientation_target|finger_actuator_targets|free_space_gesture_attempt_001" docs/
```

### Pass Criteria

```text
[x] Runbook exists
[x] Task recording commands exist
[x] Naming convention exists
[x] Per-task demo targets exist
[x] Replay/filter commands are documented and marked TODO until implemented
[x] Dataset readiness criteria are documented
[x] No replay, filtering, learning, or Level 5 orchestration code is implemented
```

### Codex Prompt

```text
Update the Level 2 dataset collection runbook and tracker only.
Read CURRENT_STATUS first and do not override the active next checkpoint unless
the user explicitly asks to change it.
Keep this docs-only: do not implement replay, filtering, learning, or Level 5
orchestration.
Make sure the runbook preserves the full Level 1.13 action schema:
base_position_target, base_orientation_target, and finger_actuator_targets.
```

---

## Level 2.3 — Demo Replay

### Goal

Replay saved skill demonstrations in MuJoCo.

### Files

```text
dexvision/logging/replay_demo.py
dexvision/apps/replay_demo.py
tests/test_replay_loader.py
```

### Run

```bash
mjpython -m dexvision.apps.replay_demo --demo data/demos/raw/free_space_gesture/2026-06-14_001
pytest tests/test_replay_loader.py
```

### Pass Criteria

```text
[x] Saved demo loads
[x] Actions replay in MuJoCo
[x] Full base/finger action schema replays
[x] Replay can run headless or with viewer
[x] Replay speed can be normal/slow
[x] Missing files produce clear errors
```

### Codex Prompt

```text
Implement demo replay.
The replay app should load a saved demo, validate it, and apply recorded actions to the MuJoCo hand scene.
It must apply base_position_target, base_orientation_target, and finger_actuator_targets.
Support --headless and --speed.
Do not add quality filtering or learning yet.
```

---

## Level 2.4 — Free-Space Gesture Skill Dataset

### Goal

Record a small skill-demo dataset of non-object gestures.

### Task

Gestures:

```text
open palm
fist
point
pinch
peace sign
wave
```

Suggested minimum 10-demo mix:

```text
open_palm x2
fist x2
pinch x2
wave x2
point x1
peace_sign x1
```

Preferred balanced collection if recording time is available:

```text
open_palm x10
fist x10
point x10
pinch x10
peace_sign x10
wave x10
```

The smaller mix is only the checkpoint minimum. A balanced 10-per-gesture set
is better training data and easier to inspect later.

Recording recipe for each clip:

```text
1. Start from a neutral upright palm pose in frame.
2. Press c to calibrate/center the hand; saved recording begins only after this succeeds.
3. Hold the requested static gesture still for about 3-5 seconds. Wave is the
   only moving gesture in this checkpoint.
4. Press q to stop/save the clip.
5. Replay the clip and delete/re-record it manually if the motion is bad.
```

Recorder shutdown verification:

```text
After q saves the clip, the camera preview and MuJoCo viewer should close and
the terminal prompt should return within 3 seconds. A "Saved demo" message
without the prompt returning is a failed shutdown check.
```

Pinch-specific note:

```text
Pinch demos should show the robot thumb and index moving toward an approximate
pinch. True fingertip contact is not required, but it should read visually as
a thumb-index pinch. If the thumb stays open while the index curls, delete that
clip and re-record after restarting the recorder with the current pinch overlay
retargeting patch.
The live camera overlay shows a Pinch close bar; use that bar, not Thumb curl
alone, to confirm whether the app sees a pinch. Pinch close should rise while
middle/ring/pinky bend stay low.
```

### Build

Add metadata field:

```text
gesture_label, optional
```

### Run

```bash
mjpython -m dexvision.apps.record_demo --task free_space_gesture --retargeter curl --output data/demos/raw/free_space_gesture/2026-06-14_001 --gesture-label open_palm --level1-13-full
mjpython -m dexvision.apps.replay_demo --demo data/demos/raw/free_space_gesture/2026-06-14_001
```

### Pass Criteria

```text
[x] At least 10 demos recorded
[x] Demos replay correctly
[x] Gesture labels optional but supported
[x] Recording waits until c calibrates/centers the hand before saving frames
[x] Pinch overlay retargeting supports approximate thumb-index pinch actions
[x] Recorder exits cleanly after q on macOS
[x] Bad demos can be manually deleted
```

The July 18, 2026 collection audit found 60 schema-valid episodes with 10
episodes for each gesture label, and all 60 completed headless MuJoCo replay.
The original `2026-07-14_001` pinch contains extra open-hand transition
footage, but its uninterrupted 3.97-second valid pinch satisfies the recording
recipe, so it remains usable raw data with that caveat documented.

### Codex Prompt

```text
Add support for a free_space_gesture task label and optional gesture metadata during recording.
Do not implement object tasks yet.
```

---

## Level 2.4B — Repository Reproducibility Baseline

### Goal

Make the repository cloneable and testable before adding task environments or
collecting larger manipulation datasets.

This checkpoint happens after the active free-space gesture collection
checkpoint. It must not change the completed Level 1 behavior, the saved Level
2 action format, or the Level 2.4 gesture collection requirements.

### Files

```text
.gitignore
pyproject.toml
environment.yml, or an equivalent committed environment specification
README.md
docs/
```

### Build

Establish a reproducible baseline:

```text
track the roadmap, task-environment, runbook, and orchestration docs in Git
ignore __pycache__, *.pyc, .DS_Store, and local raw datasets
document how to create the dexvision environment from a clean machine
declare runtime and development dependencies
include pytest and ruff in the development environment
preserve macOS and Windows setup instructions
```

Do not delete or rewrite the operator's local raw datasets. Repository cleanup
must remove generated files from version control without removing local source
data unless the user explicitly requests that separate destructive action.

### Run

```bash
git ls-files docs/progress_level_3.md docs/task_environment.md docs/skill_orchestration_future.md
ruff check dexvision tests
pytest
python -m dexvision.apps.health_check
```

### Pass Criteria

```text
[x] Future roadmap and task-design docs are tracked by Git
[x] Generated Python caches and OS metadata are ignored
[x] A clean-environment creation command is documented
[x] Runtime and development dependencies are declared
[x] Ruff and pytest run in the dexvision environment
[x] Existing automated tests still pass
[x] No local raw dataset is deleted
```

### Codex Prompt

```text
Implement only the Level 2.4B repository reproducibility baseline.
Do not change teleoperation, recording, replay, task, retargeting, or learning behavior.
Preserve local raw datasets.
Stop after the repository and environment checks pass.
```

---

## Level 2.4C — Executable Observation Layout Contract

### Goal

Make every saved observation field reconstructable by code before object-task
pilots or Level 3 learning begin.

The current full Level 1.13 action schema remains unchanged. Existing Level 2.4
gesture demos must remain replayable; if the observation layout version
changes, provide an explicit compatibility adapter or migration path instead
of silently reinterpreting old arrays.

### Files

```text
dexvision/logging/dataset_schema.py
dexvision/logging/demo_logger.py
tests/test_dataset_schema.py
tests/test_demo_logger.py
docs/module_contracts.md
```

### Build

Define an executable layout for every dense state array. Each field must
declare:

```text
field name
source array
column range or named MuJoCo joints/actuators
shape
dtype
units
coordinate frame
optional/mask behavior
normalization guidance for future learning
```

At minimum, code must reconstruct and validate:

```text
robot qpos
robot qvel
actuator controls
base position
base orientation
finger joint positions
finger joint velocities
tracking quality
object state, when present
target/task state, when present
```

### Run

```bash
pytest tests/test_dataset_schema.py tests/test_demo_logger.py tests/test_replay_loader.py
```

### Pass Criteria

```text
[x] Observation fields have explicit executable mappings
[x] Dense array widths are validated against their layouts
[x] Named robot fields preserve MuJoCo joint/actuator order
[x] Units and coordinate frames are declared
[x] Optional task/object fields have explicit masks or absence rules
[x] Existing Level 2.4 demos remain replayable
[x] Synthetic extraction tests verify every declared field
[x] No task environment or learning model is implemented
```

### Codex Prompt

```text
Implement only the executable observation layout contract.
Keep the full Level 1.13 action schema unchanged.
Make existing free-space demos replay-compatible through an explicit versioned adapter if needed.
Add synthetic schema and extraction tests.
Do not implement task environments, policy learning, or orchestration.
```

---

## Level 2.5 — Task Board and Reach-Touch Task

### Goal

Add the shared task-board scene plus one resettable, goal-parameterized task:
`reach_touch_target`.

Do not add `button_press` or `push_cube_to_target` in this checkpoint.

### Files

```text
dexvision/sim/tasks.py
assets/mujoco/task_board_scene.xml
tests/test_task_specs.py
```

### Initial Task

```text
reach_touch_target
```

The task spec should define:

```text
task_id
skill_name
required objects
initial state
typed skill parameters, including target_pose or a named target site
observation/state fields
success metric inputs
success condition
failure condition
max episode length
deterministic seed/reset behavior
```

### Run

```bash
python -m dexvision.apps.check_task --task reach_touch_target
pytest tests/test_task_specs.py
```

### Pass Criteria

```text
[x] Task board scene loads
[x] reach_touch_target state and success metric work
[x] Reset supports deterministic starts
[x] Target pose can be selected from a configured set
[x] Sampled target and initial state are saved in task state
[x] Success functions work on synthetic states
[x] No button, cube-push, demonstration-collection, or learning work is added
```

### Codex Prompt

```text
Implement only the shared task-board scene and reach_touch_target task schema.
Add typed target parameters, deterministic reset, state extraction, and success metrics.
Use synthetic tests for success/failure functions.
Do not add button_press, push_cube_to_target, demonstration collection, or learning.
```

---

## Level 2.6 — Reach-Touch Pilot Demonstrations

### Goal

Record a small pilot dataset for `reach_touch_target` before implementing
generic relabeling and quality filters.

This is a pipeline-validation pilot, not the final training dataset.

### Run

```bash
mjpython -m dexvision.apps.record_demo --task reach_touch_target --retargeter curl --output data/demos/raw/reach_touch_target/2026-06-14_001 --level1-13-full
```

### Pilot Dataset

```text
5 manually reviewed demos
at least 3 configured target positions
one task attempt per episode
```

### Pass Criteria

```text
[x] Demos include task state
[x] Success/failure label saved
[x] Replay shows the intended skill behavior
[x] Full Level 1.13 action schema is saved
[x] Target pose and initial state are saved
[x] No large-scale collection starts before relabeling and quality filters exist
```

### Codex Prompt

```text
Update record_demo only for the reach_touch_target pilot.
Log target/task state and the operator success/failure label.
Record only five manually reviewed pilot episodes across at least three targets.
Do not train any policy.
```

---

## Level 2.6B — Reach-Touch Success Relabeling

### Goal

Recompute `reach_touch_target` success/failure from saved state instead of
trusting only the operator's end-of-recording label.

### Files

```text
dexvision/logging/relabel_success.py
dexvision/apps/relabel_demos.py
tests/test_success_relabeling.py
```

### Run

```bash
python -m dexvision.apps.relabel_demos --dataset data/demos/raw/reach_touch_target
pytest tests/test_success_relabeling.py
```

### Pass Criteria

```text
[x] reach_touch_target success can be recomputed
[x] Relabel report is saved
[x] Missing metric inputs produce clear errors
[x] Operator and recomputed labels are both preserved for audit
[x] No other task relabeler is added
```

### Codex Prompt

```text
Implement success relabeling only for reach_touch_target pilot demos.
Use saved target/task state and success metric inputs.
Preserve both the operator label and recomputed label.
Do not train policies.
```

---

## Level 2.7 — Pilot Quality Filters

### Goal

Flag bad demonstrations automatically, using the reach-touch pilot as the
first real dataset.

### Files

```text
dexvision/logging/quality_filters.py
dexvision/apps/filter_demos.py
tests/test_quality_filters.py
```

### Filters

```text
low tracking confidence
too many missing frames
high feature jitter
high action jerk
too many joint-limit hits
recomputed task failure
workspace-limit hits
```

### Run

```bash
python -m dexvision.apps.filter_demos --dataset data/demos/raw/reach_touch_target
pytest tests/test_quality_filters.py
```

### Pass Criteria

```text
[x] Good synthetic demo passes
[x] Low-confidence synthetic demo fails
[x] High-jitter synthetic demo fails
[x] Failed task demo is flagged
[x] Report saved as JSON/CSV
[x] Report groups results by skill_name/task_id
[x] Thresholds are versioned and configurable
[x] Raw episodes remain immutable
```

### Codex Prompt

```text
Implement quality filtering for saved demos.
Add filters for confidence, missing frames, action jerk, workspace limits, and recomputed task success.
Create a filter_demos app that writes a report.
Group quality reports by skill/task.
Do not delete or rewrite raw episodes.
Use tests with synthetic demos.
Do not add learning.
```

---

## Level 2.7B — Reach-Touch Dataset Summary

### Goal

Summarize the reach-touch pilot and confirm it is safe to scale.

### Files

```text
dexvision/logging/dataset_summary.py
dexvision/apps/summarize_demos.py
tests/test_dataset_summary.py
```

### Summary Fields

```text
skill_name
task_id
num_episodes
num_success
success_rate
mean_episode_length
mean_tracking_confidence
quality pass/fail counts
action schema version
observation schema version
```

### Run

```bash
python -m dexvision.apps.summarize_demos --dataset data/demos
pytest tests/test_dataset_summary.py
```

### Pass Criteria

```text
[x] Summary is produced per skill
[x] Empty or missing datasets produce clear warnings
[x] Schema versions are reported
[x] JSON/CSV output is saved
[x] Pilot quality failures and relabel disagreements are reported
```

### Codex Prompt

```text
Implement the dataset summary path and run it first on reach_touch_target.
Report counts, success rate, tracking quality, and schema versions.
Do not train policies.
```

---

## Level 2.7C — Scale Reach-Touch Dataset

### Goal

Collect the first Level 3-ready dataset only after replay, relabeling, quality
filtering, and summary tooling work on the pilot.

### Target Dataset

```text
minimum: 50 clean reach_touch_target demos
preferred: 100 clean reach_touch_target demos
multiple configured target positions
separate held-out target positions reserved for Level 3 evaluation
```

### Collection Helper

Use the quality-gated balanced selector for each new attempt:

```bash
python -m dexvision.apps.select_reach_target --run
```

The selector chooses randomly among the configured targets with the fewest
quality-passed successful episodes. A live attempt records into
`data/demos/staging/reach_touch_target/`; only an attempt that passes the
current Level 2.7 quality filters is moved into
`data/demos/raw/reach_touch_target/`. Failed attempts are preserved under
`data/demos/rejected/reach_touch_target/` for audit and tuning, do not count
toward clean target balance, and do not rewrite or delete existing raw data.

### Pass Criteria

```text
[x] Raw demos remain immutable
[x] Every episode validates and replays
[x] Every episode has a recomputed success label
[x] Quality report covers every episode
[x] At least 50 clean successful demos remain
[x] Target-position distribution is summarized
[x] Held-out evaluation targets are identified before training
[x] Dataset summary marks reach_touch_target ready for Level 3
```

The completed collection contains 76 immutable raw episodes and 55 clean
successful episodes: 18 left, 18 center, and 19 right. All 76 episodes
validated and completed headless replay for 9,540 recorded action steps.
Relabeling covers every episode with 69 recomputed successes and zero
operator/recomputed disagreements; quality filtering covers every episode
with 55 passes and 21 retained audit failures. The versioned split in
`configs/reach_touch_dataset.yaml` reserves two interpolated, unrecorded target
positions for Level 3 evaluation. The v2 dataset summary reports
`level3_ready: true`.

### Codex Prompt

```text
Scale only the reach_touch_target dataset after the pilot gates pass.
Collect at least 50 clean successful demos over multiple target positions.
Reserve target positions for held-out Level 3 evaluation.
Do not implement another task or train a policy.
```

---

## Level 2.7D — Button-Press Task

### Goal

Implement one goal-parameterized `button_press` task.

Do not record button demonstrations in this checkpoint.

### Files

```text
dexvision/sim/tasks.py
assets/mujoco/task_board_scene.xml
tests/test_task_specs.py
tests/test_success_relabeling.py
```

### Skill Parameters

```text
button_id
target_press_depth or pressed-state target
optional approach pose
```

### Pass Criteria

```text
[x] Button task reset is deterministic
[x] Button state and press-depth metric are saved
[x] Success can be recomputed
[x] Parameter and terminal-state schemas are declared
[x] Synthetic success/failure tests pass
[x] No cube-push or learning work is added
```

The task board now contains three passive spring-return buttons with bounded
slide joints. `ButtonPressTask` supports seeded button selection, explicit
button ids, press-depth or pressed-state goals, and an optional approach pose.
Its dense task state preserves the selected button, current/target press
metrics, dwell and terminal state, approach goal, and deterministic initial
button/robot state. Automated checks passed on July 18, 2026 using
`conda run -n dexvision pytest tests/test_task_specs.py
tests/test_success_relabeling.py` with 27 passed, `conda run -n dexvision ruff
check dexvision tests`, the existing reach-touch headless smoke check, and
`conda run -n dexvision pytest` with 313 passed.

### Codex Prompt

```text
Implement only the button_press task schema, reset, state extraction, and success metric.
Do not record demonstrations or implement push_cube_to_target or learning.
```

---

## Level 2.7E — Button-Press Pilot

### Goal

Record five button-press demonstrations and pass them through the existing
replay, relabel, quality, and summary pipeline.

### Pass Criteria

```text
[x] Five pilot demos cover the configured button parameters
[x] Every pilot validates and replays
[x] Recomputed success labels are available
[x] Pilot quality report and summary are produced
[x] No dataset scale-up, cube-push task, or learning work is added
```

The completed pilot contains five operator-verified, schema-valid episodes
covering left, center, and right buttons plus target press depths from 0.010 to
0.014 metres. All five completed headless replay, recomputed as successful,
passed the Level 2.7 quality filters, and had zero operator/recomputed label
disagreements. The selected button is always bright green while both
non-targets are dark gray; reach-touch fixtures are hidden and non-colliding
only during button recording/replay. A target-isolation audit confirmed that
the selected button was the primary press in every retained episode. No dataset
scale-up, cube-push task, or learning work was added.

### Codex Prompt

```text
Record and validate only the five-demo button_press pilot.
Reuse the existing replay, relabel, quality, and summary paths.
Do not scale the dataset or implement push_cube_to_target or learning.
```

---

## Level 2.7F — Push-Cube Task

### Goal

Implement one goal-parameterized `push_cube_to_target` task.

Do not record cube-push demonstrations in this checkpoint.

### Files

```text
dexvision/sim/tasks.py
assets/mujoco/task_board_scene.xml
tests/test_task_specs.py
tests/test_success_relabeling.py
```

### Skill Parameters

```text
object_id
target_pose or target_zone_id
optional approach side
```

### Pass Criteria

```text
[x] Cube and target resets are deterministic
[x] Object pose/velocity and target state are saved
[x] Success can be recomputed
[x] Parameter and terminal-state schemas are declared
[x] Synthetic success/failure tests pass
[x] No demonstration collection or learning starts yet
```

The shared task-board scene now contains a free-joint cube, three deterministic
start sites, three named target zones, and a movable non-colliding target
marker. `PushCubeTask` supports typed object, target-pose/zone, and optional
approach-side parameters; preserves current and initial object pose/velocity,
target state, robot state, planar target distance, dwell, and terminal state;
and exposes a pure saved-state success predicate. Existing reach and button
tasks hide and isolate the cube fixture. Automated checks passed on July 19,
2026 using `conda run -n dexvision pytest tests/test_task_specs.py
tests/test_success_relabeling.py` with 37 passed, `conda run -n dexvision ruff
check dexvision tests`, and `conda run -n dexvision pytest` with 328 passed.
No cube demonstrations, dataset relabel dispatch, or learning code was added.

### Codex Prompt

```text
Implement only the push_cube_to_target task schema, reset, state extraction, and success metric.
Do not record demonstrations or add learning in this checkpoint.
```

---

## Level 2.7G — Push-Cube Pilot

### Goal

Record five cube-push demonstrations and pass them through the existing replay,
relabel, quality, and summary pipeline.

This checkpoint includes only the minimum recorder/replay/audit integration
needed to produce and validate the five-demo pilot. Do not add a scaled
collection planner, training code, or unrelated object tasks.

### Files

```text
dexvision/apps/record_demo.py
dexvision/logging/replay_demo.py
dexvision/logging/relabel_success.py
dexvision/logging/quality_filters.py
dexvision/logging/dataset_summary.py
tests/test_push_cube_recording.py
tests/test_replay_loader.py
tests/test_success_relabeling.py
tests/test_quality_filters.py
tests/test_dataset_summary.py
docs/level2_dataset_runbook.md
```

### Build

Extend the existing task-demo pipeline only far enough to:

```text
accept object_id, target_pose or target_zone_id, and optional approach_side
record the full Level 1.13 action plus PushCubeState object/task arrays
save resolved cube start state, target state, target radius, and dwell requirement
restore the recorded cube start and target cue during semantic replay
recompute planar distance-and-dwell success from saved task state
dispatch push_cube_to_target through existing quality and summary paths
preserve raw episodes without rewriting them
```

Before live collection, automated tests must use synthetic/headless episodes
and prove that malformed or inconsistent cube metric inputs fail clearly.

### Run

```bash
conda run -n dexvision pytest tests/test_push_cube_recording.py tests/test_replay_loader.py tests/test_success_relabeling.py tests/test_quality_filters.py tests/test_dataset_summary.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest
```

After those checks pass, record exactly five pilot episodes spanning all three
configured target zones. Use the actual recording date and a unique attempt
number for each output directory:

```bash
mjpython -m dexvision.apps.record_demo \
  --task push_cube_to_target \
  --retargeter curl \
  --object-id push_cube \
  --target-zone-id push_target_left \
  --approach-side left \
  --output data/demos/raw/push_cube_to_target/YYYY-MM-DD_001 \
  --level1-13-full
```

Then run replay, relabeling, filtering, and summary commands against the
retained pilot:

```bash
mjpython -m dexvision.apps.replay_demo \
  --demo data/demos/raw/push_cube_to_target/YYYY-MM-DD_001
conda run -n dexvision python -m dexvision.apps.relabel_demos \
  --dataset data/demos/raw/push_cube_to_target
conda run -n dexvision python -m dexvision.apps.filter_demos \
  --dataset data/demos/raw/push_cube_to_target
conda run -n dexvision python -m dexvision.apps.summarize_demos \
  --dataset data/demos
```

### Pass Criteria

```text
[x] Five pilot demos cover configured object/target parameters
[x] Every pilot validates and replays
[x] Recomputed success labels are available
[x] Pilot quality report and summary are produced
[x] Raw episodes preserve full actions plus cube pose/velocity and target state
[x] All three configured target zones are represented
[x] Operator and recomputed labels are retained for audit
[x] User confirms viewer replay shows the intended cube push into its saved target
[x] No dataset scale-up or learning work is added
```

The minimum recorder, semantic replay, relabel, quality-filter, and summary
integration is implemented. Automated checks passed on July 19, 2026 using
`conda run -n dexvision pytest tests/test_push_cube_recording.py
tests/test_task_specs.py
tests/test_replay_loader.py tests/test_success_relabeling.py
tests/test_quality_filters.py tests/test_dataset_summary.py` with 74 passed,
`conda run -n dexvision ruff check dexvision tests`, and
`conda run -n dexvision pytest` with 342 passed after the cube-scene usability
correction. The vertical wall and unrelated fixtures are now hidden, the hand
free joint and mocap target reset together immediately behind the cube, and
live recording ignores the nominal task timeout. The
operator-authorized unusable `2026-07-19_001` trials were inspected and
deleted. The replacement trial had 100% detected frames and 0.956 mean tracking
confidence, but only 5.6 cm forward/depth command range over 31.7 seconds and
pushed the cube away from the target. Cube-specific translation/depth gains,
smoothing response, deadband, and the bounded per-frame step were tuned from
that evidence. A later `2026-07-19_002` attempt still felt delayed even though
its command trajectory was responsive. Its physical hand base trailed the
commanded mocap target by 3.3 cm on average and 7.4 cm at the 95th percentile
because the generic 2-step loop advanced only 0.004 seconds of simulation per
roughly 0.033-second camera frame. Cube recording now uses at least 17 MuJoCo
steps per frame at 30 Hz, records the effective step count, and a headless
regression check reduces final follow error from about 2.5 cm to 2 mm. The
inspected failed attempt was deleted under the operator's authorization.
A post-cadence `2026-07-19_002` attempt confirmed responsive 17-step control,
but the horizontal simulated palm conflicted with the operator's upright real
palm and made height versus push direction difficult to interpret. The cube
moved 27 cm and rose more than 10 cm, indicating an abrupt scoop rather than
residual input latency. The hand now starts palm-up vertically, the neutral
position is moved slightly farther behind the cube to preserve a stable reset,
and position/depth/orientation smoothing and the per-frame step cap are
moderately damped. A headless reset-and-push regression keeps reset cube drift
below 2 mm and moves the cube more than 5 cm with less than 2 cm of vertical
lift. The inspected failed attempt was deleted under the operator's standing
authorization.
A subsequent 410-frame `2026-07-19_002` trial had 100% hand detection, 0.949
mean tracking confidence, and only 2.3 mm mean physical follow error, ruling
out sluggish simulation as the remaining cause. The operator clarified that
the real palm faces the webcam. Although the simulated palm was vertical, its
fingers—not its palm normal—pointed along the push axis. The cube moved 13.3 cm
backward and never became closer to the target. Cube reset now places a
camera-facing, fingers-up palm behind the cube, webcam-relative depth motion
maps along the palm normal, and the live viewer starts from a clear
three-quarter angle. Forearm/wrist collisions below the tabletop are disabled
for this pose while palm/finger collisions remain active. A headless regression
keeps reset drift below 2 mm and moves the cube about 18 cm along the table with
less than 1 mm vertical disturbance. The focused suite now passes 76 tests.
Ruff passes and the full suite passes 344 tests. The inspected failed trial was
deleted under the operator's standing authorization.
A further 418-frame `2026-07-19_002` trial exposed two concrete orientation
bugs. The reset quaternion presented the back of the robot hand toward the
cube, and the Level 1.13 full-control preset had re-enabled live base
orientation after task setup. The commanded quaternion then traversed nearly
its full component ranges: rendered frames showed the hand rotate out of view
after about 1.2 seconds and later reappear sideways across the table. Cube mode
now uses the model's actual palm-facing identity reset quaternion and forcibly
disables live base orientation after applying the full-control preset.
Translation, webcam-relative depth, and finger articulation remain active.
The saved effective teleop config now records the resolved base/orientation/
depth enable flags. A headless fixed-orientation push moves the cube about
21 cm along the table with less than 1 mm vertical displacement, and the
focused suite passes 76 tests.
A subsequent 756-frame `2026-07-19_002` trial kept the commanded quaternion
fixed, but exposed task-geometry clipping. The visible forearm/wrist crossed
the tabletop at reset, and image up/down spanned 15 cm of commanded height.
At the lower/forward workspace limits, table contact deflected the physical
hand despite the fixed mocap quaternion, producing 3.6 cm mean and 11.6 cm p95
follow error, sideways poses, and cube lift. Cube mode now hides and disables
the below-table forearm/wrist support geometry, fixes robot height at the
stable push plane, reduces depth gain and forward/lateral bounds, and ignores
real-hand up/down. Explicit named target zones select the matching lateral
cube-start lane so the pilot is a straight planar push. A headless bounded push
ends 8.6 mm from target centre with 0.02 mm vertical drift, fixed orientation,
and 2.2 mm base-follow error. The focused suite passes 76 tests.
A September 2, 2026 audit of `2026-07-19_004` found a clean 127-frame planar
push: 100% hand detection, 0.958 mean tracking confidence, 1.4 mm mean base
follow error, fixed orientation, 1.1 mm maximum cube lift, and no visual
clipping in rendered start/contact/final frames. The cube finished 2.72 cm from
target centre and satisfied the required five-frame dwell. Headless replay,
relabeling, and all quality gates pass. The episode is retained as immutable
diagnostic/reference data, but its operator label is null because push-cube was
accidentally omitted from the shared operator-label prompt, so it does not
count toward the five accepted pilots. The prompt now includes push-cube, and
the quality filter correctly excludes intentionally fixed workspace axes from
limit-hit accounting. Focused checks pass with 79 tests, Ruff passes, and the
full suite passes with 347 tests.
A second September 2 attempt, `2026-09-02_001`, is a labeled and recomputed
success with no label disagreement. It has 100% detection, 0.964 mean tracking
confidence, 1.15 mm mean base-follow error, fixed orientation, 0.72 mm cube
lift, a 2.58 cm final target distance, and clean rendered start/contact/final
frames. Headless replay and relabeling pass. It nevertheless fails the quality
gate because its lateral command remained at the left workspace bound for 48
of 147 frames (32.7%, versus the 10% maximum). The episode is retained as raw
diagnostic data but does not count toward the five accepted pilots. The lateral
workspace is now centered on the selected target/start lane with 7 cm of
steering margin on either side; fixed height, fixed orientation, and bounded
forward motion are unchanged. Focused checks continue to pass with 79 tests.
A third retained attempt, `2026-09-02_002`, is again visually clean and has an
operator/recomputed success agreement, headless replay, 100% detection, 0.960
mean tracking confidence, 1.09 mm mean follow error, fixed orientation, 0.60 mm
cube lift, and a 2.49 cm final target distance. It narrowly fails only the
workspace gate because 12.8% of its 242 frames saturated the lateral boundary,
above the 10% maximum. Because each named start and target are already
lane-aligned, lateral input now remains fixed at that lane along with height
and orientation; only palm-normal depth motion drives the pilot. The episode
is retained as labeled diagnostic data but does not count among the five
accepted pilots. Focused checks pass with 79 tests.
`2026-09-02_003` is the first accepted pilot and covers `push_target_left`.
It contains 97 frames with an affirmative operator label, recomputed success,
zero disagreement, headless replay, and a quality pass. Tracking was present
for 100% of frames at 0.956 mean confidence; mean base-follow error was 1.44
mm, orientation/lateral position/height remained fixed, maximum cube lift was
0.59 mm, and final planar target distance was 2.76 cm with the required
five-frame dwell. Rendered reset/contact/final frames showed no clipping or
uncontrolled motion. After this acceptance, the operator-authorized diagnostic
episodes `2026-07-19_004`, `2026-09-02_001`, and `2026-09-02_002` were deleted
and reports regenerated. The dataset summary now dispatches target
distribution and clean-success aggregation for push-cube, reporting exactly
one episode and one clean success for the left lane. Four more accepted pilots
covering centre and right as well as left are still required. Focused checks
pass with 79 tests.
Four additional September 2 recordings (`2026-09-02_001`,
`2026-09-02_002`, `2026-09-02_004`, and `2026-09-02_005`) also pass headless
replay, operator/recomputed label agreement, every automated quality gate, and
rendered-frame inspection. Across the five current episodes, hand detection is
100%, mean confidence is 0.956-0.977, mean base-follow error is 1.44-2.33 mm,
maximum cube lift is 0.35-0.74 mm, and final planar target distance is
2.09-2.76 cm. The regenerated reports contain five successes, zero label
disagreements, five quality passes, and five clean successes. All five target
`push_target_left`, though, so the three-zone coverage requirement remains
open. Keep `2026-09-02_002`, `2026-09-02_003`, and `2026-09-02_005`; record a
centre and a right replacement; then remove redundant left-lane episodes
`2026-09-02_001` and `2026-09-02_004` so the final pilot contains exactly five
episodes. The operator subsequently authorized deletion of those two redundant
episodes. Reports were regenerated and now contain three retained successes,
zero disagreements, three quality passes, and three clean successes, all for
the left lane.
Replacement episodes `2026-09-02_006` and `2026-09-02_007` cover the centre
and right lanes. Both pass schema validation, headless semantic replay,
operator/recomputed label agreement, every quality gate, and rendered
start/middle/final inspection. Their mean tracking confidence is 0.954 and
0.933, mean base-follow error is 1.32 and 2.22 mm, maximum cube lift is 0.63
and 0.48 mm, and final target distance is 2.56 and 2.70 cm. The retained pilot
is now exactly five episodes with left=3, centre=1, right=1; reports contain
five successes, zero disagreements, five quality passes, and five clean
successes. Verification also exposed that the replay CLI's one-step default
did not reproduce recordings made at 17 MuJoCo steps per frame. Replay now
uses saved `recording.sim_steps_per_frame` by default, supports an explicit
override, and falls back to one step for legacy data. Focused tests pass with
57 tests, Ruff passes, and the full suite passes with 350 tests. On September
2, 2026, the user manually replayed all five retained episodes and confirmed
that each showed the correct saved cube start and target marker, a clean planar
push, and the intended final target behavior. Every Level 2.7G pass criterion
is satisfied and the checkpoint is complete.

### Codex Prompt

```text
Implement and validate only the minimum recorder/replay/relabel/filter/summary
integration required for the five-demo push_cube_to_target pilot.
Use synthetic/headless tests before live recording, preserve raw episodes, and
require manual viewer confirmation for all five retained replays.
Do not scale the dataset or add learning.
```

---

## Level 2.7H — Scale Button-Press Dataset

### Goal

Scale the button-press dataset only after its pilot passes replay, relabeling,
quality, and summary gates.

### Files

```text
configs/button_press_dataset.yaml
dexvision/logging/collection_planner.py
dexvision/apps/select_button_goal.py
dexvision/logging/dataset_summary.py
dexvision/apps/summarize_demos.py
tests/test_button_collection_planner.py
tests/test_dataset_summary.py
docs/module_contracts.md
docs/level2_dataset_runbook.md
```

### Target Dataset

```text
button_press: 50-100 clean successful demos
held-out button poses/press targets identified before Level 3
```

### Collection

Use the balanced quality-gated selector once per attempt:

```bash
python -m dexvision.apps.select_button_goal --run
```

The selector chooses randomly among the configured button/depth training goals
with the fewest clean successful episodes. It records into staging and moves
only quality-passed successes into `data/demos/raw/button_press/`. Reserved
held-out button/depth states are declared in
`configs/button_press_dataset.yaml` and are never selected for training.

After collection, regenerate all dataset-level audit artifacts:

```bash
python -m dexvision.apps.relabel_demos --dataset data/demos/raw/button_press
python -m dexvision.apps.filter_demos --dataset data/demos/raw/button_press
python -m dexvision.apps.summarize_demos --dataset data/demos
```

### Run

```bash
conda run -n dexvision python -m dexvision.apps.select_button_goal --help
conda run -n dexvision pytest tests/test_button_collection_planner.py tests/test_dataset_summary.py tests/test_button_press_pilot.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest
```

### Pass Criteria

```text
[x] Button pilot passed every gate
[x] Every episode has replay, relabel, and quality results
[x] Initial-state and goal distributions are summarized
[x] Held-out evaluation states are declared before training
[x] Dataset summary marks button_press Level 3-ready
```

The completed collection contains 55 immutable raw episodes: the original five
pilots plus 50 quality-gated scaled recordings. All 55 episodes validated and
completed semantic headless replay for 5,475 action steps, recomputed as
successful, passed every quality filter, and had zero operator/recomputed label
disagreements. Clean training coverage is balanced across all nine configured
button/depth goals, with seven `button_left_depth_010` episodes and six for each
other goal. The versioned split reserves three interpolated 0.011 m/0.013 m
button/depth states exclusively for Level 3 evaluation. The v3 JSON/CSV summary
reports the goal and initial-state distributions and marks `button_press`
`level3_ready: true`. The user completed the 50-recording live collection on
September 2, 2026 through the quality-gated selector. Level 2.7H is complete.

### Codex Prompt

```text
Scale only button_press after its pilot gate passes.
Preserve raw episodes and run validation, relabeling, filtering, and summaries for every button episode.
Do not train policies.
```

---

## Level 2.7I — Scale Push-Cube Dataset

### Goal

Scale the cube-push dataset only after its pilot passes replay, relabeling,
quality, and summary gates.

### Files

```text
configs/push_cube_dataset.yaml
dexvision/logging/collection_planner.py
dexvision/apps/select_push_cube_goal.py
dexvision/logging/dataset_summary.py
dexvision/apps/summarize_demos.py
tests/test_push_cube_collection_planner.py
tests/test_dataset_summary.py
docs/module_contracts.md
docs/level2_dataset_runbook.md
```

### Target Dataset

```text
push_cube_to_target: 100+ clean successful demos
held-out cube starts and target poses identified before Level 3
```

### Collection

Use the balanced quality-gated selector once per attempt:

```bash
python -m dexvision.apps.select_push_cube_goal --run
```

The selector chooses randomly among the configured cube start/target/approach
training goals with the fewest clean successful episodes. It records into
staging and moves only quality-passed successes into
`data/demos/raw/push_cube_to_target/`. Reserved held-out starts and target poses
are declared in `configs/push_cube_dataset.yaml` and are never selected for
training.

After collection, regenerate all dataset-level audit artifacts:

```bash
python -m dexvision.apps.relabel_demos --dataset data/demos/raw/push_cube_to_target
python -m dexvision.apps.filter_demos --dataset data/demos/raw/push_cube_to_target
python -m dexvision.apps.summarize_demos --dataset data/demos
```

### Run

```bash
conda run -n dexvision python -m dexvision.apps.select_push_cube_goal --help
conda run -n dexvision pytest tests/test_push_cube_collection_planner.py tests/test_dataset_summary.py tests/test_push_cube_recording.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest
```

### Pass Criteria

```text
[x] Cube-push pilot passed every gate
[x] Every episode has replay, relabel, and quality results
[x] Initial-state and goal distributions are summarized
[x] Held-out evaluation states are declared before training
[x] Dataset summary marks push_cube_to_target Level 3-ready
```

The completed collection contains 101 immutable raw episodes. All 101 episodes
validated and completed semantic headless replay for 7,176 action frames using
their saved 17-step simulation cadence. All 101 recomputed as successful,
passed every quality filter, and had zero operator/recomputed label
disagreements. Clean training coverage is balanced across the three configured
lane-aligned goals: left=33, centre=34, and right=34. Three interpolated cube
start/target-pose states remain held out for Level 3 evaluation. Dataset-summary
v4 reports exact cube goal and initial-state distributions and marks
`push_cube_to_target` `level3_ready: true`. The user completed the live
quality-gated collection on September 2, 2026. Level 2.7I is complete.

### Codex Prompt

```text
Scale only push_cube_to_target after its pilot gate passes.
Preserve raw episodes and run validation, relabeling, filtering, and summaries for every cube-push episode.
Do not train policies.
```

---

## Level 2.7J — Optional Skill Card Export Metadata

### Goal

Export metadata stubs that Level 3 can fill in after training a policy.

### Files

```text
dexvision/logging/skill_card_metadata.py
dexvision/apps/export_skill_metadata.py
tests/test_skill_card_metadata.py
```

### Metadata

```text
skill_name
skill version
task_id
observation schema version
action schema version
typed parameter schema, including units and coordinate frames
preconditions
success_condition
failure_conditions
timeout
terminal-state fields
dataset summary path
known limitations
```

### Pass Criteria

```text
[x] Metadata stub can be exported per skill
[x] Full Level 1.13 action schema is declared
[x] Typed parameters include units and coordinate frames
[x] Success/failure conditions are included
[x] Timeout and terminal-state fields are included
[x] No policy checkpoint is required yet
```

The policy-free metadata exporter builds validated JSON stubs for the
implemented `reach_touch_target`, `button_press`, and `push_cube_to_target`
skills from their executable task specs and matching dataset-summary groups.
Each stub records the full Level 1.13 action layout, typed goal parameters,
transition conditions, timeout, terminal-state contract, dataset readiness,
and known limitations while leaving `policy_checkpoint` unset for Level 3.
Automated checks passed on September 2, 2026 using the focused suite with 7
tests, CLI exports against the repository's v4 dataset summary for all three
skills, Ruff, and the full suite with 369 tests. Level 2.7J is complete.

### Codex Prompt

```text
Export optional skill-card metadata stubs from Level 2 task specs and dataset summaries.
Do not train or evaluate policies.
Do not implement Level 5 orchestration.
```

---

## Level 2.8 — Retargeter B: Fingertip Target Baseline

### Goal

Implement a second retargeting method for comparison.

### Files

```text
dexvision/retargeting/fingertip_ik_retargeter.py
tests/test_fingertip_retargeter.py
```

### Build

Use fingertip positions normalized in a palm coordinate frame.

Output robot joint targets using either:

```text
simple approximate mapping
MuJoCo IK if feasible
optimization-lite solve
```

### Run

```bash
pytest tests/test_fingertip_retargeter.py
```

### Pass Criteria

```text
[x] Fingertip targets are computed
[x] Outputs obey joint limits
[x] Works on synthetic open/fist landmarks
[x] Fallback exists if solve fails
```

The fingertip baseline transforms MediaPipe-compatible landmarks into a
palm-local frame normalized by palm width, then uses an inexpensive geometric
solve to map fingertip extension to the existing bounded Shadow Hand targets.
It accepts landmark arrays or tracking results and falls back to the last valid
targets, or a safe open pose before the first valid solve. Automated checks
passed on September 2, 2026 using the focused suite with 7 tests, Ruff, and the
full suite with 376 tests. Level 2.8 is complete and did not require manual
verification.

### Codex Prompt

```text
Implement a fingertip-target retargeter as a second baseline.
It should convert hand landmarks to normalized fingertip targets and output robot joint targets.
Keep it simple and robust.
Do not replace the curl retargeter.
```

---

## Level 2.9 — Retargeter C: Optimization Retargeter

### Goal

Implement a higher-quality retargeter.

### Files

```text
dexvision/retargeting/optimization_retargeter.py
tests/test_optimization_retargeter.py
```

### Objective

Minimize:

```text
fingertip tracking error
+ joint limit penalty
+ smoothness penalty
```

### Run

```bash
pytest tests/test_optimization_retargeter.py
```

### Pass Criteria

```text
[x] Returns valid joint targets
[x] Respects limits
[x] Smoothness penalty reduces jumps
[x] Fallback if optimization fails
[x] Solve time is logged
```

The optimization retargeter solves for five bounded normalized finger controls
against palm-local 3D fingertip targets. Its objective includes configurable
fingertip-error, actuator-limit, and previous-solution smoothness terms. SciPy
L-BFGS-B is used when available, with a deterministic projected-gradient solver
when SciPy is absent; failed solves fall back to the last valid clipped targets
or the safe open pose. Solve time, objective, success, and backend diagnostics
are retained, and successful solve time is logged. Automated checks passed on
September 2, 2026 using `conda run -n dexvision pytest
tests/test_optimization_retargeter.py` with 7 passed, the three-retargeter suite
with 30 passed, `conda run -n dexvision ruff check dexvision tests`, and
`conda run -n dexvision pytest` with 383 passed. Level 2.9 is complete and did
not require manual verification.

### Codex Prompt

```text
Implement an optimization-based retargeter with joint-limit clipping and smoothness penalty.
Use scipy if available, but provide a safe fallback.
Add tests with synthetic targets.
Do not add learning.
```

---

## Level 2.10 — Retargeting Benchmark

### Goal

Compare retargeters systematically.

### Files

```text
dexvision/evaluation/benchmark_retargeters.py
dexvision/apps/benchmark_retargeters.py
tests/test_retargeting_metrics.py
```

### Metrics

```text
mean latency
mean action jerk
joint-limit violation rate
fingertip error, if available
task success rate
operator notes, optional
```

### Run

```bash
python -m dexvision.apps.benchmark_retargeters --task push_cube_to_target --episodes 10
pytest tests/test_retargeting_metrics.py
```

### Pass Criteria

```text
[ ] Runs at least curl vs fingertip retargeter
[ ] Saves metrics JSON/CSV
[ ] Produces a plot
[ ] README summary table updated
```

### Codex Prompt

```text
Implement retargeter benchmarking.
Compare available retargeters on the same task and save metrics.
Add simple plots for latency, jerk, joint-limit rate, and success.
Do not add policy learning.
```

---

# Level 2 Completion Checklist

```text
[x] Skill-learning task board environment and initial skill set documented
[x] DemoEpisode schema implemented
[x] Demo logger records full-action skill-demo data
[x] Dataset collection runbook and tracker documented
[x] Demo replay works for base/wrist/finger actions
[x] Free-space gesture demos recorded
[x] Repository and environment are reproducible from a clean clone
[x] Observation layouts are executable and versioned
[x] reach_touch_target task and deterministic reset work
[x] Reach-touch pilot demos pass replay
[x] Reach-touch success relabeling works
[x] Pilot quality filters work without mutating raw data
[x] Reach-touch summary and scaled dataset are Level 3-ready
[x] button_press task and pilot pass all data-quality gates
[x] push_cube_to_target task and pilot pass all data-quality gates
[x] Button and push datasets are scaled only after their pilot gates pass
[x] Optional skill-card metadata export documented
[x] Fingertip retargeter implemented
[ ] Optimization retargeter implemented
[ ] Retargeter benchmark produces results
[ ] Level 2 README/results updated
```
