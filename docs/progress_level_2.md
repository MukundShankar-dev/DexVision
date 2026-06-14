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

### Build

Add metadata field:

```text
gesture_label, optional
```

### Run

```bash
mjpython -m dexvision.apps.record_demo --task free_space_gesture --retargeter curl --output data/demos/raw/free_space_gesture/2026-06-14_001 --level1-13-full
mjpython -m dexvision.apps.replay_demo --demo data/demos/raw/free_space_gesture/2026-06-14_001
```

### Pass Criteria

```text
[ ] At least 10 demos recorded
[ ] Demos replay correctly
[ ] Gesture labels optional but supported
[ ] Bad demos can be manually deleted
```

### Codex Prompt

```text
Add support for a free_space_gesture task label and optional gesture metadata during recording.
Do not implement object tasks yet.
```

---

## Level 2.5 — Task Schemas and Core Skill Tasks

### Goal

Add task specs, reset logic, state extraction, and success metrics for the
first core skill tasks.

### Files

```text
dexvision/sim/tasks.py
assets/mujoco/task_board_scene.xml
tests/test_task_specs.py
```

### Initial Tasks

```text
reach_touch_target
button_press
push_cube_to_target
```

These task specs should define:

```text
task_id
skill_name
required objects
initial state
observation/state fields
success metric inputs
success condition
failure condition
max episode length
```

### Run

```bash
python -m dexvision.apps.check_task --task reach_touch_target
pytest tests/test_task_specs.py
```

### Pass Criteria

```text
[ ] Task board scene loads
[ ] reach_touch_target state and success metric work
[ ] button_press state and success metric work
[ ] push_cube_to_target state and success metric work
[ ] Reset supports deterministic starts
[ ] Success functions work on synthetic states
```

### Codex Prompt

```text
Implement task schemas for reach_touch_target, button_press, and push_cube_to_target.
Add reset, state extraction, and success metrics.
Use synthetic tests for success/failure functions.
Do not add learning yet.
```

---

## Level 2.6 — Record Core Skill Demonstrations

### Goal

Record real teleop demonstrations for the first core skill tasks.

### Run

```bash
mjpython -m dexvision.apps.record_demo --task reach_touch_target --retargeter curl --output data/demos/raw/reach_touch_target/2026-06-14_001 --level1-13-full
mjpython -m dexvision.apps.record_demo --task button_press --retargeter curl --output data/demos/raw/button_press/2026-06-14_001 --level1-13-full
mjpython -m dexvision.apps.record_demo --task push_cube_to_target --retargeter curl --output data/demos/raw/push_cube_to_target/2026-06-14_001 --level1-13-full
```

### Target Dataset

```text
minimum per skill: 20 demos
better per skill: 50 demos
ideal for Level 3 per skill: 100-200 demos
```

### Pass Criteria

```text
[ ] Demos include task state
[ ] Demos include object state when the skill has objects
[ ] Success/failure label saved
[ ] Replay shows the intended skill behavior
[ ] Full Level 1.13 action schema is saved
```

### Codex Prompt

```text
Update record_demo to support reach_touch_target, button_press, and push_cube_to_target.
Log task/object state and success/failure at the end of each episode.
Do not train any policy.
```

---

## Level 2.6B — Task-Specific Success Relabeling

### Goal

Recompute success/failure labels from saved state instead of trusting only the
operator's end-of-recording label.

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
[ ] reach_touch_target success can be recomputed
[ ] button_press success can be recomputed
[ ] push_cube_to_target success can be recomputed
[ ] Relabel report is saved
[ ] Missing metric inputs produce clear errors
```

### Codex Prompt

```text
Implement task-specific success relabeling for saved demos.
Use saved task/object state and success metric inputs.
Do not train policies.
```

---

## Level 2.7 — Quality Filters

### Goal

Flag bad demos automatically.

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
task failure
object not moved
workspace-limit hits
```

### Run

```bash
python -m dexvision.apps.filter_demos --dataset data/demos/raw/push_cube_to_target
pytest tests/test_quality_filters.py
```

### Pass Criteria

```text
[ ] Good synthetic demo passes
[ ] Low-confidence synthetic demo fails
[ ] High-jitter synthetic demo fails
[ ] Failed task demo is flagged
[ ] Report saved as JSON/CSV
[ ] Report groups results by skill_name/task_id
```

### Codex Prompt

```text
Implement quality filtering for saved demos.
Add filters for confidence, missing frames, action jerk, and task success.
Create a filter_demos app that writes a report.
Group quality reports by skill/task.
Use tests with synthetic demos.
Do not add learning.
```

---

## Level 2.7B — Per-Skill Dataset Summary

### Goal

Summarize each skill dataset before Level 3 training.

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
[ ] Summary is produced per skill
[ ] Empty or missing datasets produce clear warnings
[ ] Schema versions are reported
[ ] JSON/CSV output is saved
```

### Codex Prompt

```text
Implement per-skill dataset summaries for saved demos.
Report counts, success rate, tracking quality, and schema versions.
Do not train policies.
```

---

## Level 2.7C — Optional Skill Card Export Metadata

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
task_id
observation_schema
action_schema
inputs/parameters, such as target_pose or object_id
preconditions
success_condition
failure_conditions
dataset summary path
known limitations
```

### Pass Criteria

```text
[ ] Metadata stub can be exported per skill
[ ] Full Level 1.13 action schema is declared
[ ] Success/failure conditions are included
[ ] No policy checkpoint is required yet
```

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
[ ] Fingertip targets are computed
[ ] Outputs obey joint limits
[ ] Works on synthetic open/fist landmarks
[ ] Fallback exists if solve fails
```

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
[ ] Returns valid joint targets
[ ] Respects limits
[ ] Smoothness penalty reduces jumps
[ ] Fallback if optimization fails
[ ] Solve time is logged
```

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
[ ] Free-space gesture demos recorded
[ ] Core task schemas implemented for reach_touch_target, button_press, and push_cube_to_target
[ ] Core skill demos recorded
[ ] Task-specific success relabeling works
[ ] Quality filters work
[ ] Per-skill dataset summary works
[ ] Optional skill-card metadata export documented
[ ] Fingertip retargeter implemented
[ ] Optimization retargeter implemented
[ ] Retargeter benchmark produces results
[ ] Level 2 README/results updated
```
