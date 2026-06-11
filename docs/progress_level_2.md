# Progress Level 2 — Demonstration Recording, Replay, Data Quality, and Retargeting Benchmarks

Level 2 goal:

> Turn the Level 1 teleoperation demo into a reproducible data-generation and benchmarking system.

This level is about engineering maturity: recording demonstrations, replaying them, validating them, filtering bad data, and comparing retargeting methods.

---

## Level 2 Prerequisite

Serious Level 2 demo recording is blocked until Level 1.8B — Final Hand Model
Decision and Separation is complete.

The final robot hand defines the action space: joint names, actuator names,
action dimensions, joint limits, neutral pose, and the meaning of recorded
controls. Changing the hand model after collecting demonstrations can invalidate
saved demos, replay results, benchmark comparisons, and trained Level 3 policies.

Use the debug hand only for smoke tests until `docs/robot_hand_model.md`
documents the final hand choice and the scripted gestures pass on that final
model.

---

## Level 2.1 — Demo Episode Schema

### Goal

Define the exact format for one recorded demonstration.

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
    timestamps: np.ndarray
    success: bool | None
```

Validation checks:

```text
matching time dimension
timestamps monotonic
no NaNs in required arrays
actions have correct dimension
metadata has required fields
```

### Run

```bash
pytest tests/test_dataset_schema.py
```

### Pass Criteria

```text
[ ] Valid demo passes validation
[ ] Invalid time dimensions are caught
[ ] NaNs are caught
[ ] Missing metadata is caught
[ ] Test uses synthetic arrays
```

### Codex Prompt

```text
Implement dexvision/logging/dataset_schema.py.
Create a DemoEpisode dataclass plus validate_demo, save_demo, and load_demo stubs if appropriate.
Add tests with synthetic arrays.
Do not add live recording yet.
```

---

## Level 2.2 — Demo Logger

### Goal

Record Level 1 teleop runs to disk.

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
camera.mp4, optional
```

### Run

```bash
python -m dexvision.apps.record_demo --task free_gesture --retargeter curl --output data/demos/free_gesture
pytest tests/test_demo_logger.py
```

### Pass Criteria

```text
[ ] Demo directory created
[ ] Metadata saved
[ ] Arrays saved
[ ] Arrays have same T
[ ] Recording can start/stop cleanly
[ ] No giant video required by default
```

### Codex Prompt

```text
Implement demo recording only.
Add dexvision/logging/demo_logger.py and dexvision/apps/record_demo.py.
The app should wrap the existing Level 1 teleop loop and save features, actions, robot states, timestamps, metadata, and optional landmarks.
Do not add replay, filtering, or learning yet.
```

---

## Level 2.3 — Demo Replay

### Goal

Replay saved demonstrations in MuJoCo.

### Files

```text
dexvision/logging/replay_demo.py
dexvision/apps/replay_demo.py
tests/test_replay_loader.py
```

### Run

```bash
python -m dexvision.apps.replay_demo --demo data/demos/free_gesture/demo_001
pytest tests/test_replay_loader.py
```

### Pass Criteria

```text
[ ] Saved demo loads
[ ] Actions replay in MuJoCo
[ ] Replay can run headless or with viewer
[ ] Replay speed can be normal/slow
[ ] Missing files produce clear errors
```

### Codex Prompt

```text
Implement demo replay.
The replay app should load a saved demo, validate it, and apply recorded actions to the MuJoCo hand scene.
Support --headless and --speed.
Do not add quality filtering or learning yet.
```

---

## Level 2.4 — Free-Space Gesture Dataset

### Goal

Record a small dataset of non-object gestures.

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
python -m dexvision.apps.record_demo --task free_gesture --output data/demos/free_gesture
python -m dexvision.apps.replay_demo --demo data/demos/free_gesture/demo_001
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
Add support for a free_gesture task label and optional gesture metadata during recording.
Do not implement object tasks yet.
```

---

## Level 2.5 — Object Task: Push Cube

### Goal

Add the first manipulation task.

### Files

```text
dexvision/sim/tasks.py
assets/mujoco/push_cube_scene.xml
tests/test_push_cube_task.py
```

### Scene

```text
table or ground plane
robot hand
cube
target zone
```

### Success

```text
final cube distance to target < threshold
```

### Run

```bash
python -m dexvision.apps.check_task --task push_cube
pytest tests/test_push_cube_task.py
```

### Pass Criteria

```text
[ ] Task scene loads
[ ] Cube state can be read
[ ] Target position exists
[ ] Success function works on synthetic states
[ ] Reset randomizes or supports fixed starts
```

### Codex Prompt

```text
Implement a simple push_cube MuJoCo task.
Add task state extraction, reset, and success metric.
Use a cube and target zone.
Do not add learning yet.
```

---

## Level 2.6 — Record Push Cube Demonstrations

### Goal

Record real teleop demonstrations for push_cube.

### Run

```bash
python -m dexvision.apps.record_demo --task push_cube --output data/demos/push_cube
```

### Target Dataset

```text
minimum: 20 demos
better: 50 demos
ideal for Level 3: 100-200 demos
```

### Pass Criteria

```text
[ ] Demos include object state
[ ] Success/failure label saved
[ ] Cube movement visible
[ ] Replay shows object interaction
```

### Codex Prompt

```text
Update record_demo to support the push_cube task.
Log object state and success/failure at the end of the episode.
Do not train any policy.
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
```

### Run

```bash
python -m dexvision.apps.filter_demos --dataset data/demos/push_cube
pytest tests/test_quality_filters.py
```

### Pass Criteria

```text
[ ] Good synthetic demo passes
[ ] Low-confidence synthetic demo fails
[ ] High-jitter synthetic demo fails
[ ] Failed task demo is flagged
[ ] Report saved as JSON/CSV
```

### Codex Prompt

```text
Implement quality filtering for saved demos.
Add filters for confidence, missing frames, action jerk, and task success.
Create a filter_demos app that writes a report.
Use tests with synthetic demos.
Do not add learning.
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
python -m dexvision.apps.benchmark_retargeters --task push_cube --episodes 10
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
[ ] DemoEpisode schema implemented
[ ] Demo logger records teleop data
[ ] Demo replay works
[ ] Free gesture demos recorded
[ ] Push cube task implemented
[ ] Push cube demos recorded
[ ] Quality filters work
[ ] Fingertip retargeter implemented
[ ] Optimization retargeter implemented
[ ] Retargeter benchmark produces results
[ ] Level 2 README/results updated
```
