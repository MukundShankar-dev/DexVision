# Level 2 Dataset Collection Runbook

This runbook is the practical operator guide for collecting Level 2
demonstration datasets. It tracks what to record, where to put it, how much is
enough for each stage, and what still must be validated before Level 3 behavior
cloning begins.

## A. Purpose

Level 2 turns live Level 1.13 teleoperation into replayable, validated,
skill-specific demonstration datasets for Level 3 skill learning.

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
- Avoid overwriting existing runs. Prefer a new attempt directory; use
  `--overwrite` only when intentionally replacing a known bad run.

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

Future task commands, TODO until task scenes/state extraction are implemented:

```bash
# TODO Level 2.5/2.6
mjpython -m dexvision.apps.record_demo \
  --task reach_touch_target \
  --retargeter curl \
  --output data/demos/raw/reach_touch_target/2026-06-14_001 \
  --level1-13-full

# TODO Level 2.7E
mjpython -m dexvision.apps.record_demo \
  --task button_press \
  --retargeter curl \
  --output data/demos/raw/button_press/2026-06-14_001 \
  --level1-13-full

# TODO Level 2.7G
mjpython -m dexvision.apps.record_demo \
  --task push_cube_to_target \
  --retargeter curl \
  --output data/demos/raw/push_cube_to_target/2026-06-14_001 \
  --level1-13-full

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

- `reach_touch_target`: 50-100 clean demos
- `button_press`: 50-100 clean demos
- `push_cube_to_target`: 100+ clean demos
- `grasp_object` and `pinch_lift_object`: stretch until the task scene,
  success metrics, and replay/filtering path are stable

## F. Task Recording Checklist

For each demo:

- Pick a unique output directory before starting.
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
Ctrl-C after frames have been recorded saves the collected frames. There is not
yet a dedicated discard key. For a bad take, delete the attempt directory or
rerun the same path with `--overwrite`.

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

Keep questionable demos in `raw/` only if they can be flagged later. Do not move
them into `processed/` as clean training data.

## H. Replay And Validation Commands

Replay is the next Level 2 code checkpoint and is not implemented yet:

```bash
# TODO Level 2.3
mjpython -m dexvision.apps.replay_demo --demo <path>
```

Validation CLI is not implemented yet. The schema validator exists as library
logic; a CLI can be added later if needed:

```bash
# TODO future Level 2 validation helper
python -m dexvision.apps.validate_demo --demo <path>
```

Quality filtering is planned for Level 2.7:

```bash
# TODO Level 2.7
python -m dexvision.apps.filter_demos --dataset <task_dataset>
```

Dataset summaries are planned for Level 2.7B:

```bash
# TODO Level 2.7B
python -m dexvision.apps.summarize_demos --dataset data/demos
```

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

## J. Level 2 Completion Tracker

| Task | Scene implemented | Record command | Replay works | Quality filter works | Clean demos target | Ready for Level 3 |
|---|---|---|---|---|---:|---|
| free_space_gesture | yes, no object scene required | yes | no | no | 20 | no |
| reach_touch_target | no | TODO | no | no | 50 | no |
| button_press | no | TODO | no | no | 50 | no |
| push_cube_to_target | no | TODO | no | no | 100 | no |
| rotate_dial | no | TODO | no | no | 100 | stretch |
| pinch_lift_object | no | TODO | no | no | 100+ | stretch |

Update this table as Level 2 checkpoints land. A row should only move to
`Ready for Level 3 = yes` after replay, validation, filtering, and clean-demo
counts are all satisfied.

Future manipulation-task order:

```text
1. reach_touch_target task and five-demo pilot
2. generic relabel/filter/summary gates proven on the reach pilot
3. scale reach_touch_target and reserve held-out target positions
4. button_press task and five-demo pilot through the same gates
5. push_cube_to_target task and five-demo pilot through the same gates
6. scale only the task datasets whose pilots pass
```

Train/validation/test planning must happen before dataset scale-up. Split by
whole episode, recording session, and initial/goal condition; never place
timesteps from one episode into multiple splits.

## K. Do Not Do Yet

- Do not train Level 3 policies until replay, validation, and filtering exist.
- Do not collect large datasets before task schemas, success metrics, action
  schemas, and observation schemas stabilize.
- Do not implement Level 5 orchestration in this repo.
- Do not use finger-only action logs for serious skill learning now that the
  full Level 1.13 action space exists.
- Do not assume free-space gesture demos are enough for object manipulation
  policies.
