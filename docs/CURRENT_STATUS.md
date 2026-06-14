# Current Project Status

This file is the source of truth for what Codex or another coding agent should work on next.

Agents should read this file before selecting any checkpoint.

---

## Current Level

Level 2 — Demonstration Recording, Replay, Data Quality, and Retargeting Benchmarks

---

## Current Progress File

`docs/progress_level_2.md`

---

## Last Completed Checkpoint

Level 2.1 — Demo Episode Schema

Note: the previous Level 1.3B index-only decoupling patch is superseded by the
completed Level 1.3B local per-finger replacement and bend-control decision.

---

## Next Target Checkpoint

Level 2.2 — Demo Logger

---

## Current Branch

`main`

Suggested first feature branch:

`feature/level1-health-check`

---

## Manual Verification Status

Level 1.0 did not require manual verification.

Level 1.1 camera smoke test manually passed on June 10, 2026 with camera ID 0.
Live feed appeared, FPS was visible at about 30 FPS, and q quit cleanly.

Level 1.2 hand landmark tracking manually passed on June 10, 2026 with camera ID 0.
Landmarks, skeleton lines, anatomical left/right handedness, and confidence displayed, no-hand frames did not crash, and q quit cleanly.
Handedness labels were re-verified after correcting MediaPipe's mirrored-label convention; user's left hand now displays Left and right hand displays Right.

Level 1.3 finger feature extraction manually passed on June 10, 2026 with camera ID 0.
Feature bars updated live, open hand/fist/pointing/pinch behavior looked correct after thumb curl calibration, no-hand frames did not crash, and q quit cleanly.

Level 1.3B index curl decoupling was completed after Level 1.10 exposed feature coupling, but the user later reported that the local tuning still was not good enough because all fingers could affect each other in practice.
That index-only patch is superseded by the now-completed Level 1.3B local per-finger feature extractor replacement.
Automated replacement checks passed on June 10, 2026 using `pytest tests/test_hand_features.py`, `pytest tests/test_smoothing.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py`, and `pytest`.
Manual screenshot review on June 10, 2026 found the replacement much better: long-finger `extension` values are close to the intended live poses and are currently the most intuitive signal.
Manual review also found raw `curl` values confusing/not well-correlated enough to trust as the robot-control signal, and thumb behavior still needs separate work.
The long-finger control-signal decision has now been implemented and manually accepted: for index/middle/ring/pinky, robot control uses `bend = clamp(1.0 - extension, 0.0, 1.0)`.
Raw `curl` remains available as a diagnostic/local geometric signal, not the Level 1.10 robot-control signal.
Thumb control remains conservative and acceptable for the current through-Level-1.10 scope; fuller thumb behavior can be revisited during full-hand teleop tuning.
Automated bend-control checks passed on June 11, 2026 using `pytest tests/test_hand_features.py tests/test_smoothing.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py tests/test_index_curl_calibration.py`.
Full automated checks passed on June 11, 2026 using `pytest` with 109 passed and 19 skipped.
Manual verification passed on June 11, 2026 after the user reported the live feature viewer and one-finger MuJoCo teleop behavior were clean.
Level 1.3B is now complete.

Level 1.4 feature smoothing manually passed on June 10, 2026 with camera ID 0.
Raw and smoothed feature bars displayed live, smoothed values jittered less than raw values while remaining responsive, lost tracking did not produce NaNs, confidence drops held or decayed controls safely, and q quit cleanly.

Level 1.5 MuJoCo simple scene manually passed on June 10, 2026 using `mjpython -m dexvision.apps.check_mujoco --model assets/mujoco/simple_scene.xml --viewer --steps 600`.
The MuJoCo viewer opened, the ground grid and orange cube were visible, no camera or MediaPipe was involved, and the simulation stepped to 6.000 seconds.

Level 1.6 robot hand model load did not require manual verification.
Headless checks passed on June 10, 2026 using `conda run -n dexvision python -m dexvision.apps.check_hand_model --model assets/mujoco/hand_scene.xml`; 10 named limited joints and 10 named position actuators were discoverable, and rest stability passed.
That simple model was later preserved as `assets/mujoco/debug_hand_scene.xml` when Level 1.8B selected the Shadow Hand as the final candidate.

Level 1.7 move one robot joint manually passed on June 10, 2026 using `mjpython -m dexvision.apps.check_one_joint --model assets/mujoco/debug_hand_scene.xml --joint index_mcp`.
The MuJoCo viewer opened, the simple hand model was visible, the index MCP joint moved up while the rest of the hand stayed fixed, printed targets stayed within `[-0.150, 1.450]`, and the simulation completed 600 steps to 1.200 seconds without instability.

Level 1.8 move all robot fingers manually passed on June 10, 2026 using `mjpython -m dexvision.apps.check_hand_actuation --model assets/mujoco/debug_hand_scene.xml --gestures configs/debug_hand_gestures.yaml`.
The MuJoCo viewer opened, scripted open hand, fist, point, approximate pinch, peace sign, and relax gestures moved the fingers, printed targets respected actuator limits, and the simulation remained stable. Gesture shapes are approximate in the simple two-joint-per-finger hand model, but acceptable for the "can move all fingers" checkpoint.

Level 1.8B final hand model decision and separation manually passed on June 10, 2026.
Automated final-hand checks passed on June 10, 2026 using `conda run -n dexvision python -m dexvision.apps.check_hand_model --model assets/mujoco/hand_scene.xml --steps 20` and `conda run -n dexvision python -m dexvision.apps.check_hand_actuation --model assets/mujoco/hand_scene.xml --headless --steps-per-gesture 3 --print-interval 3`.
The final candidate is the right Shadow Hand E3M5 model from Google DeepMind MuJoCo Menagerie, vendored under `assets/mujoco/menagerie/shadow_hand/` and loaded by `assets/mujoco/hand_scene.xml`.
The original simple hand remains at `assets/mujoco/debug_hand_scene.xml` with debug gestures in `configs/debug_hand_gestures.yaml`.
Manual verification confirmed by the user after visual inspection: the Shadow Hand viewer opened, scripted open/fist/point/pinch/peace/relax gestures moved clearly enough for this checkpoint, and no instability/errors were reported. Pinch and peace remain approximate until future visual tuning.

Level 1.9 curl retargeter did not require manual verification.
Automated checks passed on June 10, 2026 using `conda run -n dexvision pytest tests/test_curl_retargeter.py`.
The retargeter maps configured `HandFeatures` control fields to final Shadow Hand actuator targets from `configs/level1_teleop.yaml`, handles missing or low-confidence features as an open hand, clips outputs to configured limits, and supports per-finger scaling.
After the Level 1.3B bend decision, long-finger mappings use `index_bend`, `middle_bend`, `ring_bend`, and `pinky_bend`; thumb remains conservative on `thumb_curl`.

Level 1.10 one-finger teleop manually passed on June 11, 2026 using `mjpython -m dexvision.apps.check_one_finger_teleop --camera-id 0 --show-camera-window --print-interval 10`.
The live camera overlay and MuJoCo viewer ran together, the robot index finger followed the derived `index_bend` signal, non-index robot fingers stayed neutral/open, tracking loss remained safe, and jitter was acceptable.
Printed diagnostics showed raw index curl, raw index extension, smoothed index extension, and index bend side by side; raw `index_curl` is diagnostic only.

Live one-finger teleoperation manual verification is complete through Level 1.10.

Level 1.11 full-hand teleoperation manually passed on June 11, 2026 using `mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 --print-interval 10`.
Automated checks passed on June 11, 2026 using `conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`, `conda run -n dexvision pytest tests/test_run_level1_teleop.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py`, and `conda run -n dexvision pytest` with 138 passed.
Manual verification was confirmed after the user reported the live full-hand teleop behavior looked good.

Level 1.12 polished demo manually passed on June 11, 2026 after the user
confirmed the demo looked good.
Automated checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision pytest tests/test_run_level1_teleop.py tests/test_level1_demo_docs.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py`,
and `conda run -n dexvision pytest` with 140 passed.
Level 1.12 was previously treated as the end of Level 1, but Level 1.13 has
been added before serious Level 2 manipulation demo work. No Level 2
implementation has been started yet.

Level 1.13 hand base/wrist pose control has been split into staged
sub-checkpoints. Level 1.13D final pose-control polish/safety is complete
after manual camera plus MuJoCo viewer verification.
Automated implementation checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision python -m dexvision.apps.check_hand_base_control --help`,
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`,
MuJoCo-adjacent hand model/actuation checks, and `conda run -n dexvision pytest`
with 149 passed.
Manual testing on June 11, 2026 found that base-control runs could stop after a
while on the instability guard (`max_abs_qvel` exceeded while qpos stayed
bounded) and that palm rotation did not visually align well for upright hand
poses. A stabilization/orientation patch now rate-limits mocap translation and
rotation, uses a lower base smoothing alpha, and changes enabled base-control
rotation to `palm_absolute` with a configurable offset.
Follow-up manual testing on June 11, 2026 found that right-hand behavior looked
better than left-hand behavior, and that wrist rotation/fist poses could still
drive the base down/sideways into the workspace clamp. A follow-up patch now
uses the wrist as the default base translation anchor, keeps palm-center
translation available only through `base_control.position_source: palm_center`,
canonicalizes anatomical left-hand orientation vectors for the right Shadow
Hand target, and raises the qvel guard only while base control is active so
bounded mocap-weld free-joint transients do not stop the app. Updated automated
checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 31 passed, a headless 720-frame MuJoCo base-control stress check
(`max_abs_qpos=1.071107`, `max_abs_qvel=60.115224`), and
`conda run -n dexvision pytest` with 160 passed.
Additional manual testing on June 11, 2026 still showed a bounded qvel guard
stop (`max_abs_qpos=0.897984`, `max_abs_qvel=152.293542`) when the left hand
entered the frame, and the user suspected stale relative mapping after tracking
loss/re-entry. A follow-up patch changes the default base translation behavior
to `base_control.position_mode: absolute`, resets base smoothing/source neutral
on tracking loss and large reacquire jumps, keeps first-valid-frame neutral
delta behavior available only through `base_control.position_mode: relative`,
and raises the base-control-only qvel guard further while preserving the
stricter finger-only teleop guard.
Updated automated checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 34 passed, a headless 900-frame tracking-loss/reacquire MuJoCo stress
check (`max_abs_qpos=1.083558`, `max_abs_qvel=58.271888`), both Level 1.13 CLI
help commands, and `conda run -n dexvision pytest` with 163 passed.

Additional manual testing on June 11, 2026 showed the mapping was still not
usable enough for the checkpoint: base x stayed nearly zero, base y changed too
little, base z was clamped near the floor, movement did not intuitively follow
the real hand, and base orientation was premature/noisy. Level 1.13 is being
simplified into staged verification. The default base-control path should now
be calibrated `image_2d` relative pose mapping: press `c` to capture neutral
palm center and neutral human palm orientation, map normalized image palm-center
dx/dy to clamped base translation, and optionally map palm-orientation deltas to
robot base orientation with `--enable-base-orientation`.
Updated automated checks for the simplified path passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 46 passed, both Level 1.13 CLI help commands, and
`conda run -n dexvision pytest` with 175 passed.
The latest update keeps orientation disabled by default but makes
`--enable-base-orientation` use calibrated relative palm orientation, adds
orientation axis signs/remap config, and maps image up/down to robot height by
default. Depth/in-out control is now handled separately by Level 1.13B using
monocular hand scale.
Updated automated checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 53 passed, both Level 1.13 CLI help commands, and
`conda run -n dexvision pytest` with 182 passed.

Level 1.13 has been split into staged checkpoints:

```text
1.13A Base x/y translation
1.13B Depth / In-Out Base Control
1.13C Relative palm rotation
1.13D Final pose-control polish/safety
```

Level 1.13A base x/y translation was manually verified by the user on
June 11, 2026: real hand left/right/up/down moved the simulated hand base as
intended, and finger teleop still worked.

Level 1.13B depth/in-out control manually passed on June 11, 2026 after the
user confirmed the depth behavior worked and the upright palm-forward neutral
start pose looked good. It uses monocular hand scale as the depth proxy, keeps
x/y translation and finger teleop working, starts from the user's intended
upright palm-forward neutral pose, and does not apply palm/wrist rotation.
Automated checks passed using `conda run -n dexvision pytest` with 193 passed.

Level 1.13C relative palm/wrist rotation implementation has automated checks
passing, and manual verification passed after the user confirmed roll, pitch,
yaw, and all three axes together worked in the live MuJoCo plus camera loop.
The user pushed this state as commit `5fe920d initial roll/pitch/yaw`.
The implementation keeps orientation opt-in behind
`--enable-base-orientation`, uses `orientation_mode: relative_palm`, lets
`--orientation-dofs roll` stage the first roll-only test, and keeps
roll/pitch/yaw signs, remap, clamps, smoothing, and deadband configurable.
Automated checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision python -m dexvision.apps.check_hand_base_control --help`,
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 73 passed, and `conda run -n dexvision pytest` with 202 passed. Ruff was
not available in the `dexvision` environment.

Level 1.13D final pose-control polish/safety manually passed after the user
confirmed the widened roll/pitch/yaw range is enough for the current scope. The
implementation improves the initial limited orientation range enough for
object-approach and table-pickup style wrist poses while keeping bounded
orientation clamps, smoothing, rate limits, tracking-loss holds, translation,
depth, and finger teleop intact. Final Level 1 full teleop with base/depth/
orientation/finger control was manually smoke-tested by the user for the
current scope with no reported crashes. Level 1 is complete.

Level 2.0 — Task Board Environment and Task Set Design did not require manual
verification. The task board environment, initial skill/task set, demonstration
requirements, and future Level 5 orchestration boundary are documented. No
demo recording, replay, learning, two-hand control, or future skill
orchestration code was implemented.

Level 2.1 — Demo Episode Schema did not require manual verification.
Automated schema checks passed on June 14, 2026 using
`pytest tests/test_dataset_schema.py`. The implementation defines the
in-memory demo episode schema, full Level 1.13 action schema validation,
observation schema validation, synthetic validation tests, and no live
recording, replay, learning, or two-hand control.

Level 2.2 — Demo Logger is the next target. Implement only demo recording; do
not add replay, filtering, learning, or two-hand control.

For checkpoints involving camera, GUI, MuJoCo viewer, or live teleoperation, the agent should not mark the checkpoint complete until the user confirms the manual verification passed.

---

## Agent Selection Rules

When selecting work:

1. Read `AGENTS.md`.
2. Read `docs/CURRENT_STATUS.md`.
3. Read `docs/module_contracts.md`.
4. Read the progress file listed under **Current Progress File**.
5. Select only the **Next Target Checkpoint** unless the user explicitly says otherwise.
6. Do not work on another level unless this file is updated.
7. Do not continue to the next checkpoint after finishing the current one.

---

## Updating This File

After a checkpoint is fully complete:

1. Move the finished checkpoint into **Last Completed Checkpoint**.
2. Set **Next Target Checkpoint** to the next unchecked checkpoint in the active progress file.
3. Update **Current Branch** if needed.
4. Add a note under **Manual Verification Status** if the checkpoint required user verification.

Example:

```text
Last Completed Checkpoint:
Level 1.0 — Repo and Health Check

Next Target Checkpoint:
Level 1.1 — Camera Smoke Test
```

---

## Completion Policy

A checkpoint may be marked complete only when:

```text
automated tests pass
listed smoke-test command runs, when applicable
manual verification is confirmed by the user, when required
progress checkbox is updated
commit is created with an appropriate message
```

Manual checkpoints include, but are not limited to:

```text
camera feed display
hand landmark visual overlay
MuJoCo viewer display
robot joint movement in GUI
live teleoperation
demo video capture
```
