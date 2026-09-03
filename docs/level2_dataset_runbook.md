# Level 2 Dataset Collection Runbook

This runbook is the practical operator guide for collecting Level 2
demonstration datasets. It tracks what to record, where to put it, how much is
enough for each stage, and what still must be validated before Level 3 behavior
cloning begins.

## A. Purpose

Level 2 turns live Level 1.13 teleoperation into replayable, validated,
skill-specific demonstration datasets for Level 3 skill learning.

The completed Level 2 release is now the bounded input to a Level 3 learning-
feasibility study. It is not the comprehensive dataset envisioned for the
final project. New multi-session, cross-object, visual-grounding, and recovery
data belongs to Level 4 and must be published as a separate immutable release.

The goal is not to train policies yet. The goal is to collect demonstrations
that can be loaded, replayed, validated, filtered, summarized, and trusted as
supervised training data later.

## B. What Gets Recorded

Each demo episode should save:

- `metadata.json`
- `timestamps.npy`
- `actions.npy`
- `robot_states.npy`
- `features.npy`
- `tracking_quality.npy`
- optional `landmarks.npy`
- optional `object_states.npy`
- optional `task_states.npy`
- optional `camera.mp4`

The full Level 1.13 action at each timestep is:

- `base_position_target`
- `base_orientation_target`
- `finger_actuator_targets`

`robot_states` should preserve:

- `qpos`
- `qvel`
- the commanded mocap/base target when applicable

Hand and tracking fields should preserve:

- finger bend/extension/curl diagnostics
- palm/wrist orientation diagnostics when available
- hand tracking confidence
- feature/control confidence
- hand detected, dropped-frame, and reacquire markers when available

Task and object state should be saved when a task has objects or success
metrics:

- object pose and velocity
- target pose or goal region
- button/dial/contact state where applicable
- success metric inputs needed for later relabeling

Why these fields matter:

- Actions are the Level 3 supervised target.
- Robot/object/task state is the Level 3 observation and evaluation source.
- Features and tracking quality support debugging and quality filtering.
- Metadata makes the dataset reproducible across robot models, retargeters,
  configs, action schemas, and observation schemas.

## C. Naming Convention

Use a scalable folder structure for serious datasets:

```text
data/demos/
  raw/
    free_space_gesture/
      2026-06-14_001/
      2026-06-14_002/
    reach_touch_target/
      2026-06-14_001/
  staging/
    reach_touch_target/
  rejected/
    reach_touch_target/
  processed/
    free_space_gesture/
    reach_touch_target/
  reports/
    quality/
    summaries/
```

Naming rules:

- Include the task name.
- Include the recording date in `YYYY-MM-DD` format.
- Include a zero-padded attempt number, such as `001`.
- Optionally include an operator or session id, such as
  `2026-06-14_mukund_001`.
- Never overwrite or delete an existing raw episode.
- For quality-gated reach-touch collection, let the selector choose the next
  unused raw name. Failed attempts are retained outside `raw/` under
  `rejected/`.

## C.1 Immutable Git LFS Release

The editable collection tree under `data/demos/` remains ignored by Git. After
a level or intentional dataset revision is complete, publish one immutable
compressed snapshot under `datasets/` rather than force-adding thousands of
individual NumPy files.

The completed Level 2 snapshot is:

```text
datasets/dexvision_level2_v1.tar.gz
datasets/dexvision_level2_v1.tar.gz.sha256
datasets/dexvision_level2_v1_manifest.json
```

The archive is tracked with Git LFS and contains `raw/`, `rejected/`, and
generated reports. Temporary staging data and one-off smoke recordings are not
included. Verify and restore it from the repository root with:

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/dexvision_level2_v1.tar.gz.sha256
tar -xzf datasets/dexvision_level2_v1.tar.gz
```

Never overwrite a published archive. A future intentional dataset change must
use a new release version and update its checksum and manifest.

The current single-command style is still acceptable for quick checks:

```text
data/demos/free_space_gesture_attempt_001
```

For real collection, prefer:

```text
data/demos/raw/<task_id>/<YYYY-MM-DD>_<attempt_number>
```

## D. Initial Recording Commands

Current implemented free-space full Level 1.13 recording:

```bash
mjpython -m dexvision.apps.record_demo \
  --task free_space_gesture \
  --retargeter curl \
  --output data/demos/free_space_gesture_attempt_001 \
  --level1-13-full
```

Recommended scalable form for the same current task:

```bash
mjpython -m dexvision.apps.record_demo \
  --task free_space_gesture \
  --retargeter curl \
  --output data/demos/raw/free_space_gesture/2026-06-14_001 \
  --level1-13-full
```

Optional end labels, when known at recording time:

```bash
--success
--failure
```

Current quality-gated reach-touch collection:

```bash
python -m dexvision.apps.select_reach_target --run
```

This command balances the clean-success distribution, launches the recorder,
and admits only quality-passed successful episodes into `raw/`.

Current quality-gated button-press collection:

```bash
python -m dexvision.apps.select_button_goal --run
```

This command uses `configs/button_press_dataset.yaml`, balances exact
button/depth training goals by clean-success count, and never selects the
reserved held-out evaluation states. Run it once per attempted episode until
the dataset summary reports at least 50 clean successes and every training goal
has at least five.

Current quality-gated push-cube collection:

```bash
python -m dexvision.apps.select_push_cube_goal --run
```

This command uses `configs/push_cube_dataset.yaml`, balances the three proven
lane-aligned cube start/target/approach goals by clean-success count, and never
selects the reserved held-out cube starts or target poses. Run it once per
attempted episode until the dataset summary reports at least 100 clean
successes and every training goal has at least 30.

Button-press pilot recording is implemented for Level 2.7E:

```bash
mjpython -m dexvision.apps.record_demo \
  --task button_press \
  --retargeter curl \
  --button-id button_left \
  --target-press-depth 0.010 \
  --output data/demos/raw/button_press/2026-07-18_001 \
  --level1-13-full
```

Collect exactly five pilot attempts across all three configured button ids,
then run the task-specific relabel, quality, and summary commands below. In
every recording and replay, press the single bright green target button; the
two dark gray buttons are non-targets.

The Level 2.7G push-cube recorder, semantic replay restoration, dataset
relabeling, quality filtering, and summary integration are implemented. After
the listed synthetic/headless checks pass, collect exactly five pilot episodes
covering all three configured target zones. Use the actual date and unique
attempt numbers, preserve every raw episode, and manually inspect all five
viewer replays before marking the checkpoint complete.

In cube mode, the vertical reach/button wall and unrelated fixtures are hidden.
The Shadow Hand starts behind the orange cube with its palm facing the cube and
its fingers up. Face the real palm toward the webcam with fingers up when
pressing `c`. Moving the real palm toward the webcam then moves the simulated
palm along its normal toward the cube. Image left/right and up/down are
intentionally ignored, preventing sideways saturation and table penetration.
Named targets start the cube in the same lateral lane, so a straight
palm-normal push is sufficient. The MuJoCo viewer starts from a
three-quarter angle showing the palm, cube, and target together. Press `q` when
the attempt is finished. Do not grasp or lift the cube. Cube recording has no
task timeout; only success, a workspace safety failure, or `q` ends the
attempt.
After the attempt, answer the terminal's operator success prompt with `y` or
`n`. An episode without this operator label may remain as immutable diagnostic
data but does not count toward the five accepted pilots.
The simulated wrist orientation is intentionally fixed in cube mode. Rotating
the real wrist must not rotate or flip the robot hand; only translation, depth,
and finger articulation are controlled live.
Cube-mode controls use a responsive task-specific profile with moderate
planar-position/depth smoothing plus a bounded position step and workspace.
This keeps deliberate real-hand motion visible while preventing table
penetration and abrupt motion that can scoop or launch the cube.
Cube recording also overrides the generic 2-step simulation cadence with
real-time stepping. At the current 30 Hz control rate and 0.002-second MuJoCo
timestep, it runs 17 simulation steps per camera frame so the physical hand
tracks the commanded mocap pose without a delayed trailing motion.

Push-cube pilot command:

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

Future recording commands remain TODO until their checkpoints are implemented:

```bash
# TODO later/stretch Level 2 task support
mjpython -m dexvision.apps.record_demo \
  --task rotate_dial \
  --retargeter curl \
  --output data/demos/raw/rotate_dial/2026-06-14_001 \
  --level1-13-full

# TODO later/stretch Level 2 task support
mjpython -m dexvision.apps.record_demo \
  --task pinch_lift_object \
  --retargeter curl \
  --output data/demos/raw/pinch_lift_object/2026-06-14_001 \
  --level1-13-full
```

## E. Per-Task Demo Targets

These targets are rough engineering targets, not scientific guarantees.

For every new manipulation task:

- implement one resettable, parameterized task
- record 5 manually reviewed pilot demos
- verify replay
- recompute success from saved state
- run quality filters and a dataset summary
- fix the task/data path before collecting a larger dataset

Do not collect 20-200 episodes for a new task before its five-demo pilot passes
all of those gates.

For usable Level 2 validation after the pilot passes:

- 20 demos per task

For the first Level 3 behavior-cloning baseline:

- `reach_touch_target`: complete with 55 clean demos
- `button_press`: 50-100 clean demos
- `push_cube_to_target`: 100+ clean demos
- `grasp_object` and `pinch_lift_object`: stretch until the task scene,
  success metrics, and replay/filtering path are stable

## F. Task Recording Checklist

For each manually named demo:

- Pick a unique output directory before starting. The reach-touch selector
  does this automatically.
- Open the camera and MuJoCo viewer with `--level1-13-full`.
- Calibrate the Level 1.13 base pose, or confirm auto-calibration captured the
  intended neutral pose.
- Confirm hand tracking confidence is good.
- Confirm base position/depth/orientation control responds.
- Confirm finger control responds.
- Start recording by running the command for this attempt.
- Perform exactly one intended task attempt.
- Stop recording after the task outcome is clear.
- Add `--success` or `--failure` if the outcome is known at recording time.
- Note obvious failure causes outside the saved arrays, such as bad lighting or
  accidental occlusion.

Current recorder behavior: stopping with `q`, closing the viewer, or pressing
Ctrl-C after frames have been recorded saves the collected frames. For
reach-touch collection through `select_reach_target --run`, the attempt is
first saved under `staging/`; a clean success moves into `raw/`, while a failed
or invalid take moves into `rejected/`. Existing raw episodes remain immutable.
For tasks without this gate, record a replacement and retain or clearly flag
the original raw take.

## G. Manual Quality Checklist

A demo is bad if:

- hand tracking is lost for too long
- the base pose jumps unexpectedly
- the robot leaves the intended workspace
- severe jitter dominates the action
- the wrong task was performed
- the object starts in the wrong pose
- the recording stopped too early
- the operator accidentally blocks the camera
- the success/failure label is wrong

Keep historical questionable raw demos for audit and flag them in quality
reports. New quality-gated reach-touch failures belong in `rejected/`, not
`raw/`. Do not move any failed take into `processed/` as clean training data.

## H. Replay And Validation Commands

Replay is implemented. Use headless mode for automated coverage or omit
`--headless` for viewer inspection:

```bash
python -m dexvision.apps.replay_demo --demo <episode_path> --headless
mjpython -m dexvision.apps.replay_demo --demo <episode_path>
```

Replay loading validates the saved schema and required arrays before applying
actions. Push-cube replay also restores the recorded cube start pose/velocity
and target marker. Success relabeling is implemented for reach-touch,
button-press, and push-cube:

```bash
python -m dexvision.apps.relabel_demos \
  --dataset data/demos/raw/reach_touch_target

python -m dexvision.apps.relabel_demos \
  --dataset data/demos/raw/button_press

python -m dexvision.apps.relabel_demos \
  --dataset data/demos/raw/push_cube_to_target
```

Quality filtering is implemented for reach-touch, button-press, and push-cube:

```bash
python -m dexvision.apps.filter_demos \
  --dataset data/demos/raw/reach_touch_target

python -m dexvision.apps.filter_demos \
  --dataset data/demos/raw/button_press

python -m dexvision.apps.filter_demos \
  --dataset data/demos/raw/push_cube_to_target
```

Dataset summaries and Level 3 readiness evaluation are implemented:

```bash
python -m dexvision.apps.summarize_demos --dataset data/demos
```

The summary uses `configs/reach_touch_dataset.yaml`,
`configs/button_press_dataset.yaml`, and `configs/push_cube_dataset.yaml` to
validate the versioned training-goal and held-out-evaluation splits.

## I. Dataset Readiness Criteria For Level 3

A task dataset is ready for the first Level 3 behavior-cloning baseline when:

- the schema validates
- observation fields are reconstructable through an executable versioned layout
- demos replay successfully
- actions reconstruct the full Level 1.13 command:
  `base_position_target`, `base_orientation_target`, and
  `finger_actuator_targets`
- success labels exist or can be recomputed from saved state
- bad demos are filtered or clearly flagged
- enough clean demos exist for the task target
- typed goal parameters, units, and coordinate frames are saved
- held-out initial/goal conditions are declared before training
- observation and action dimensions are stable
- metadata includes the action schema version, observation schema version,
  robot model/config, retargeter/config, task config, and Level 1 teleop config
  snapshot

Do not treat raw demos as Level 3-ready just because they were saved.

Current `reach_touch_target` readiness:

- 76 immutable raw episodes
- 69 recomputed successes with zero operator/recomputed disagreements
- 55 clean successful episodes
- clean distribution: left 18, center 18, right 19
- all 76 episodes validate and complete headless replay
- held-out Level 3 targets:
  `reach_eval_left_center = [0.14, -0.05, 0.47]` metres and
  `reach_eval_center_right = [0.14, 0.03, 0.50]` metres
- dataset summary result: `level3_ready: true`

Current `button_press` scaled-dataset status:

- task-board scene and task-specific recorder are implemented
- 55 immutable raw episodes cover all nine configured button/depth goals
- all 55 episodes validate and complete headless replay for 5,475 action steps
- all 55 recompute as successful with zero label disagreements
- all 55 pass the Level 2.7 quality filters
- the selected target is bright green and non-target buttons are dark gray
- target-isolation review confirmed the selected button was the primary press
  in every retained episode
- the versioned scale-up config reserves interpolated 0.011 m/0.013 m
  button/depth combinations for Level 3 evaluation
- the quality-gated balanced selector is available through
  `python -m dexvision.apps.select_button_goal --run`
- clean training coverage is six or seven episodes per configured goal
- dataset summary result: `level3_ready: true`

Current `push_cube_to_target` scaled-dataset status:

- the Level 2.7I quality-gated balanced selector is available through
  `python -m dexvision.apps.select_push_cube_goal --run`
- the versioned scale-up config declares three lane-aligned training goals and
  three interpolated held-out cube start/target-pose states
- the completed immutable dataset contains 101 clean successful episodes with
  balanced goal coverage left=33, centre=34, and right=34
- all 101 episodes validate, complete semantic headless replay, recompute as
  successful, pass every quality filter, and have zero label disagreements
- dataset-summary v4 reports `level3_ready: true`

- task-board scene and task-specific recorder are implemented
- replay restores the saved cube start pose/velocity and target marker
- planar distance-and-dwell relabeling, quality filtering, and summary dispatch
  have automated coverage
- cube mode hides the reach/button wall and starts a camera-facing, fingers-up
  palm directly behind the cube
- webcam-relative depth motion maps to the simulated palm normal, and a
  three-quarter viewer angle keeps the palm, cube, and target visible together
- recording has no nominal task timeout; the operator may take as long as
  needed and press `q` when finished
- the confusing `2026-07-19_001` trial was inspected, confirmed to contain no
  cube motion, and deleted at the operator's request
- a replacement tuning trial had excellent tracking and contacted the cube,
  but recorded base motion was too weak and pushed the cube away from the goal;
  it was inspected, deleted under the same operator authorization, and used to
  tune a more responsive cube-control profile
- the next inspected trial showed the remaining delay came from advancing only
  0.004 seconds of simulation per roughly 0.033-second camera frame; cube mode
  now advances a full nominal control period and the failed trial was deleted
- a later trial confirmed low physical follow error but exposed that the palm
  was edge-on to the push direction; the cube moved backward, so the reset pose
  and depth workspace were changed to a palm-normal push with fingers up
- the next trial exposed a reversed palm-facing quaternion plus the full-control
  preset re-enabling live wrist rotation; rendered frames showed the back of the
  hand at reset, the robot leaving view, and then lying sideways on the table
- cube mode now uses the actual palm-facing reset and forcibly locks robot wrist
  orientation while preserving translation, depth, and finger control
- the following trial showed visible forearm/wrist clipping through the table
  and table contact rotating the physical hand when image up/down drove the
  target below the surface
- cube mode now hides the below-table support bodies, fixes hand height, limits
  forward/lateral travel, and aligns named cube starts with their target lane
- `2026-07-19_004` is retained as a successful diagnostic/reference episode:
  it replays headlessly, recomputes success with five in-target frames, and
  passes every quality filter, but its legacy null operator label means it does
  not count toward the five accepted pilots
- push-cube recording now requires the same explicit operator label as the
  earlier manipulation pilots, and fixed-height axes are supported by quality
  filtering
- `2026-09-02_001` is retained as labeled diagnostic data: its clean rendered
  motion, operator/recomputed success agreement, and headless replay pass, but
  48 of 147 frames sat on the left workspace boundary, so it fails the quality
  gate and does not count toward the five accepted pilots
- lateral bounds are now centered on each named target lane with 7 cm of
  steering margin per side; fixed height/orientation safeguards are unchanged
- `2026-09-02_002` is also retained as labeled diagnostic data: motion, replay,
  labels, tracking, and target dwell pass, but 12.8% of frames still saturated
  the lateral boundary and therefore exceeded the 10% quality limit
- because every named start is aligned with its target, lateral control is now
  fixed to the selected lane; the pilot is a single palm-normal forward push
- `2026-09-02_003` is accepted as pilot 1/5 for `push_target_left`: operator and
  recomputed labels agree, headless replay and all quality filters pass, and
  rendered start/contact/final frames show a clean clipping-free planar push
- after that acceptance, diagnostic episodes `2026-07-19_004`,
  `2026-09-02_001`, and `2026-09-02_002` were deleted at the operator's request
- the summary now reports push-cube target distribution and one clean success
- four more September 2 recordings pass replay, label agreement, quality, and
  visual inspection; the current five-episode set has five clean successes
- all five current recordings target `push_target_left`, so the required
  centre/right coverage is still missing; retain `2026-09-02_002`,
  `2026-09-02_003`, and `2026-09-02_005`, record one centre and one right
  replacement
- redundant left episodes `2026-09-02_001` and `2026-09-02_004` were deleted
  at the operator's request; regenerated reports contain three retained clean
  left-lane successes
- the three-zone five-episode pilot has now received manual replay confirmation
- replacement episodes `2026-09-02_006` (centre) and `2026-09-02_007` (right)
  pass headless replay, labels, quality gates, and rendered-frame inspection
- the retained pilot is exactly five clean successes with target coverage
  left=3, centre=1, right=1 and zero label disagreements
- replay now uses the saved recording simulation cadence by default; this is
  required to reproduce the push contact dynamics, while legacy recordings
  without saved cadence retain the one-step fallback
- on September 2, 2026, the user confirmed all five viewer replays showed the
  correct saved cube start, target marker, clean planar push, and intended
  final target behavior; Level 2.7G is complete
- dataset scale-up was not started as part of Level 2.7G

Level 2 skill-card metadata stubs can be exported after regenerating the
dataset summary:

```bash
python -m dexvision.apps.export_skill_metadata --task reach_touch_target
python -m dexvision.apps.export_skill_metadata --task button_press
python -m dexvision.apps.export_skill_metadata --task push_cube_to_target
```

By default, each command reads
`data/demos/reports/summaries/dataset_summary.json` and writes a policy-free
stub under `data/skill_metadata/`. The exporter checks that the task spec and
dataset-summary action/observation schema versions agree. Each stub includes
the full Level 1.13 action layout, typed parameters, success/failure and
terminal-state contracts, timeout, dataset readiness, preconditions, and known
limitations. `policy_checkpoint` remains null until Level 3 training.

## J. Level 2 Completion Tracker

| Task | Scene implemented | Record command | Replay works | Quality filter works | Clean demos target | Ready for Level 3 |
|---|---|---|---|---|---:|---|
| free_space_gesture | yes, no object scene required | yes | yes | not yet applied | 60 raw | no |
| reach_touch_target | yes | yes, quality-gated | yes, 76/76 | yes, 55 pass | 55/50 | yes |
| button_press | yes | yes, quality-gated | yes, 55/55 | yes, 55 pass | 55/50 | yes |
| push_cube_to_target | yes | yes, quality-gated | yes, 101/101 | yes, 101 pass | 101/100 | yes |
| rotate_dial | no | TODO | no | no | 100 | stretch |
| pinch_lift_object | no | TODO | no | no | 100+ | stretch |

Update this table as Level 2 checkpoints land. A row should only move to
`Ready for Level 3 = yes` after replay, validation, filtering, and clean-demo
counts are all satisfied.

Manipulation-task order and current completion:

```text
[x] 1. reach_touch_target task and five-demo pilot
[x] 2. relabel/filter/summary gates proven on the reach pilot
[x] 3. scale reach_touch_target and reserve held-out target positions
[x] 4. button_press task and five-demo pilot through the same gates
[x] 5a. push_cube_to_target task schema/reset/state/success metric
[x] 5b. push_cube_to_target five-demo pilot through replay/relabel/quality/summary gates
[x] 6. scale only the task datasets whose pilots pass
[x] 7. export optional policy-free skill metadata stubs
```

Train/validation/test planning must happen before dataset scale-up. Split by
whole episode, recording session, and initial/goal condition; never place
timesteps from one episode into multiple splits.

## K. Do Not Do Yet

- Do not train a task-specific Level 3 policy until that task has replay,
  validation, relabeling, filtering, a clean scaled dataset, and held-out
  evaluation conditions. `reach_touch_target`, `button_press`, and
  `push_cube_to_target` now satisfy these data gates; later tasks do not.
- Do not collect large datasets before task schemas, success metrics, action
  schemas, and observation schemas stabilize.
- Do not implement Level 6 orchestration in this repo.
- Do not use finger-only action logs for serious skill learning now that the
  full Level 1.13 action space exists.
- Do not assume free-space gesture demos are enough for object manipulation
  policies.
