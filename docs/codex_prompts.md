# Codex Prompt Library

Use these prompts one at a time. Do not paste the whole file into Codex at once.

---

## Recommended Default Prompt

This is the preferred prompt for current work. It delegates checkpoint choice to
`docs/CURRENT_STATUS.md` and the active progress file so copied historical
prompts do not accidentally revive old project assumptions.

```text
You are working in the DexVision repo.

Read:
- AGENTS.md
- docs/CURRENT_STATUS.md
- docs/module_contracts.md
- the active progress file named in docs/CURRENT_STATUS.md

Task:
Implement the Next Target Checkpoint from docs/CURRENT_STATUS.md only.

Constraints:
- Do not implement future checkpoints.
- Do not add unrelated dependencies.
- Keep logic in the correct module.
- Add tests or a smoke-test script.
- If automated-only, update the checkpoint checkbox when tests pass.
- If manual verification is required, do not update the checkbox until the user confirms it passed.
- Update docs/CURRENT_STATUS.md after a checkpoint is truly complete.

After implementation, tell me:
- what files changed
- how to run it
- what tests to run
- known limitations
```

---

## Prompt Safety Notes

```text
docs/CURRENT_STATUS.md is the source of truth.
The active progress file named in docs/CURRENT_STATUS.md is the only progress file to follow.
Copied old prompts must not override CURRENT_STATUS, the active progress file, or module contracts.
If a prompt conflicts with CURRENT_STATUS or docs/module_contracts.md, stop and update the prompt before using it.
```

The checkpoint-specific prompts below are historical examples unless their
checkpoint is also the current target in `docs/CURRENT_STATUS.md`.

Current roadmap names:

```text
Level 3: learning feasibility on the existing Level 2 data
Level 4: comprehensive dataset and qualified skill library
Level 5: robustness, reproducibility, and portfolio polish
Level 6: future language-guided orchestration
```

Do not use a historical prompt's old scope to reintroduce work that the active
progress file now assigns to another level.

---

## Historical Level 1 Camera Prompt

```text
Implement Level 1.1 Camera Smoke Test only.

Create:
- dexvision/camera/opencv_camera.py
- dexvision/apps/check_camera.py
- tests/test_camera.py

Requirements:
- Open camera with OpenCV.
- Support --camera-id, --width, --height.
- Display FPS.
- Quit cleanly on q.
- Tests should not require a real webcam.

Do not add MediaPipe or MuJoCo.
```

---

## Historical Level 1 Hand Tracking Prompt

```text
Implement Level 1.2 Hand Landmark Tracking only.

Create/update:
- dexvision/perception/hand_tracker.py
- dexvision/perception/visualization.py
- dexvision/apps/check_hand_tracking.py
- tests/test_hand_tracker_schema.py

Use the existing OpenCVCamera wrapper.
Return a HandTrackingResult dataclass.
Return anatomical Left/Right handedness for normal unmirrored OpenCV input.
Draw landmarks and skeleton.
Handle no-hand frames.

Do not add MuJoCo, feature extraction, retargeting, or learning.
```

---

## Historical Level 1 Feature Prompt

```text
Implement Level 1.3 Finger Feature Extraction only.

Create/update:
- dexvision/features/hand_features.py
- dexvision/apps/check_hand_features.py
- tests/test_hand_features.py

Compute:
- thumb/index/middle/ring/pinky curl
- thumb-index pinch distance
- confidence

Add visual bars for feature values.
Use synthetic landmark tests.

Do not add MuJoCo.
```

---

## Historical Level 1 MuJoCo Prompt

```text
Implement Level 1.5 MuJoCo Import and Simple Scene only.

Create:
- assets/mujoco/simple_scene.xml
- dexvision/sim/mujoco_env.py
- dexvision/apps/check_mujoco.py
- tests/test_mujoco_load.py

The scene should have a ground plane, light, and cube.
The app should load and step the simulation.
The test should run headless.

Do not add camera or hand tracking.
```

---

## Level 2.0 Task Board Planning Prompt

```text
Implement Level 2.0 Task Board Environment and Task Set Design only.

Read:
- AGENTS.md
- docs/CURRENT_STATUS.md
- docs/module_contracts.md
- docs/progress_level_2.md

Create or update docs only:
- docs/progress_level_2.md
- docs/task_environment.md
- docs/level6_future.md or docs/skill_orchestration_future.md
- docs/CURRENT_STATUS.md, only if the checkpoint status truly needs it

Define the staged tabletop MuJoCo task board, resettable tasks, fixed success
metrics, and initial task set.

Do not implement demo recording, replay, learning, or skill orchestration.
```

---

## Level 2.1 Demo Episode Schema Prompt

```text
Implement Level 2.1 Demo Episode Schema only.

Create/update:
- dexvision/logging/dataset_schema.py
- tests/test_dataset_schema.py
- docs/module_contracts.md, if the schema contract changes

The demo schema must preserve the full Level 1.13 action at every timestep:
- base_position_target
- base_orientation_target
- finger_actuator_targets

Also include:
- robot qpos/qvel
- object/task state when present
- tracking quality
- timestamps
- metadata/config snapshot

Use synthetic arrays for tests.
Do not add live recording, replay, filtering, benchmarking, learning, or Level 6 orchestration.
```

---

## Level 2.2 Demo Logger Prompt

```text
Implement Level 2.2 Demo Logger only.

Create/update:
- dexvision/logging/demo_logger.py
- dexvision/apps/record_demo.py
- tests/test_demo_logger.py

Record:
- metadata.json
- features.npy
- actions.npy
- robot_states.npy
- timestamps.npy
- optional landmarks.npy
- optional object_states.npy
- optional task_states.npy
- tracking_quality.npy

The `actions.npy` data must preserve the full Level 1.13 teleoperation command:
- base_position_target
- base_orientation_target
- finger_actuator_targets

Metadata must include the action schema version, task name, robot model/config,
retargeter/config, control rate, and Level 1 teleop config snapshot.

Wrap the existing Level 1 teleop loop and record its full base/wrist/finger command.
Do not add replay, filtering, benchmarking, or learning.
```

---

## Level 2 Dataset Collection Runbook Update Prompt

```text
Update the Level 2 dataset collection runbook/checklist only.

Read:
- AGENTS.md
- docs/CURRENT_STATUS.md
- docs/module_contracts.md
- docs/progress_level_2.md
- docs/level2_dataset_runbook.md

Keep CURRENT_STATUS as the source of truth. Do not change the active next
checkpoint unless the user explicitly asks for that status change.

Update docs only:
- docs/level2_dataset_runbook.md
- docs/progress_level_2.md, only if tracker/pass criteria text changes
- docs/CURRENT_STATUS.md, only if checkpoint status truly changes
- docs/codex_prompts.md, only if this reusable prompt needs a correction

Make sure the runbook preserves the full Level 1.13 action schema:
- base_position_target
- base_orientation_target
- finger_actuator_targets

Do not implement replay, filtering, learning, Level 6 orchestration, or any
code changes.
```

---

## Historical Level 3 Training Prompt

```text
Implement Level 3.3 Training Loop only.

Create/update:
- dexvision/learning/train_bc.py
- dexvision/apps/train_policy.py
- configs/level3_bc.yaml
- tests/test_train_tiny.py

Use the existing learning dataset and MLP model.
Save checkpoints and train/val loss history.
Add a tiny overfit test.

Do not add policy rollout or vision models yet.
```
