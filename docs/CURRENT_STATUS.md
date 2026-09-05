# Current Project Status

This file is the source of truth for what Codex or another coding agent should work on next.

Agents should read this file before selecting any checkpoint.

---

## Current Level

Level 4 — Comprehensive Multi-Session Dataset Collection and Versioned Release

---

## Current Progress File

`docs/progress_level_4.md`

---

## Last Completed Checkpoint

Level 4.3F — Expert Architecture and Replay Qualification

The versioned cross-skill audit regenerates tasks from immutable metadata,
headlessly replays saved actions, recomputes task success, and separates
ordinary failures from accepted expert data. Two independently seeded episodes
for each of reach, button, push, standalone grasp-and-lift, and complete
pick/place qualified automatically. The user accepted one visible replay for
all five skills on September 5, 2026.

Note: the previous Level 1.3B index-only decoupling patch is superseded by the
completed Level 1.3B local per-finger replacement and bend-control decision.

---

## Next Target Checkpoint

Level 4.3G — Small State-Only Button Learnability Probe

---

## Current Branch

`main`

Suggested next feature branch:

`codex/level43g-button-learning`

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
requirements, and future Level 7 orchestration boundary are documented. No
demo recording, replay, learning, two-hand control, or future skill
orchestration code was implemented.

Level 2.1 — Demo Episode Schema did not require manual verification.
Automated schema checks passed on June 14, 2026 using
`pytest tests/test_dataset_schema.py`. The implementation defines the
in-memory demo episode schema, full Level 1.13 action schema validation,
observation schema validation, synthetic validation tests, and no live
recording, replay, learning, or two-hand control.

Level 2.2 — Demo Logger manually passed on June 14, 2026 after the user
confirmed the full Level 1.13-style recording path worked. The verified manual
path used `mjpython -m dexvision.apps.record_demo --task free_space_gesture
--retargeter curl --output data/demos/free_space_gesture_level113_check
--level1-13-full --overwrite`; it opened the MuJoCo viewer and separate camera
overlay, recorded full base/depth/orientation/finger teleoperation, saved the
demo arrays and metadata, and did not require a video file by default.
Automated checks passed using
`pytest tests/test_dataset_schema.py tests/test_demo_logger.py` with 20 passed,
plus the synthetic recorder smoke command. No replay, filtering, learning, or
two-hand control was implemented.

Level 2.2B — Dataset Collection Runbook and Tracker did not require manual
verification. It was completed as a docs-only checkpoint before continuing to
replay and serious dataset recording on June 14, 2026. The runbook documents
the full Level 1.13 action schema, recording naming convention, current
free-space recording command, future task command placeholders, per-task demo
targets, manual quality checklist, TODO replay/filter/summary commands, and
Level 3 dataset readiness criteria. No code, replay, filtering, learning, or
Level 7 orchestration was implemented. At that point, the next implementation
checkpoint was Level 2.3 — Demo Replay.

Level 2.3 — Demo Replay manually passed on June 14, 2026 after the user
confirmed the MuJoCo replay worked. The implementation loads saved Level 2.2
demo directories, reconstructs and validates the saved action/observation
schemas, applies full Level 1.13 `base_position_target`,
`base_orientation_target`, and `finger_actuator_targets` to the hand scene, and
supports headless or viewer replay plus replay speed control. Automated checks
passed using `pytest tests/test_replay_loader.py`, and a synthetic headless
MuJoCo replay smoke test passed. No quality filtering, dataset collection,
learning, or future checkpoint work was implemented.

Level 2.4 — Free-Space Gesture Skill Dataset manually passed on July 18, 2026.
The final raw collection contains 60 schema-valid episodes balanced at 10 each
for `open_palm`, `fist`, `point`, `pinch`, `peace_sign`, and `wave`. All 60
episodes completed headless MuJoCo replay, gesture-specific audits found
sufficiently long valid poses or wave motion in every clip, and the user
confirmed that pressing q saved the episode, closed the recorder windows, and
returned the macOS terminal prompt normally. The original
`2026-07-14_001` pinch contains extra open-hand transition footage but retains
an uninterrupted 3.97-second valid pinch and remains documented as usable raw
data. No quality filtering, object-task implementation, or learning was
started.

Level 2.4B — Repository Reproducibility Baseline did not require manual
verification. The repository now declares its runtime and development
dependencies in `pyproject.toml` and `environment.yml`, documents clean macOS
and Windows setup, tracks the roadmap/task/runbook/orchestration docs, and
ignores generated Python caches, OS metadata, operator demos, and outputs.
Generated files were removed from Git tracking without deleting the local
datasets. Automated checks passed on July 18, 2026 using
`conda run -n dexvision ruff check dexvision tests`,
`conda run -n dexvision pytest` with 240 passed, and
`conda run -n dexvision python -m dexvision.apps.health_check`.

Level 2.4C — Executable Observation Layout Contract did not require manual
verification. The versioned Level 2 observation schema now provides executable
dense-array mappings, ordered MuJoCo joint and actuator names, semantic units
and coordinate frames, explicit optional-state absence rules, and a legacy
Level 2.4 replay compatibility adapter. Automated checks passed on July 18,
2026 using `conda run -n dexvision pytest tests/test_dataset_schema.py
tests/test_demo_logger.py tests/test_replay_loader.py` with 39 passed.

Level 2.5 — Task Board and Reach-Touch Task did not require manual
verification. The shared MuJoCo task-board scene now provides three reachable
named target sites, and `reach_touch_target` provides typed goal parameters,
seeded deterministic reset, reconstructable task/initial state, fixed
distance-and-dwell success metrics, workspace/timeout failures, and a headless
CLI smoke check. Automated checks passed on July 18, 2026 using
`conda run -n dexvision python -m dexvision.apps.check_task --task
reach_touch_target`, `conda run -n dexvision pytest tests/test_task_specs.py`
with 9 passed, `conda run -n dexvision ruff check dexvision tests`, and
`conda run -n dexvision pytest` with 256 passed.

Level 2.6 — Reach-Touch Pilot Demonstrations manually passed on July 18, 2026.
The final pilot contains five schema-valid, operator-reviewed episodes across
the right, center, and left targets. Saved task state confirms five consecutive
physical palm-contact frames and computed success in every episode, all five
episodes completed semantic headless replay against their recorded target, and
the user confirmed that all five viewer replays showed the intended skill
behavior. Automated checks passed using `conda run -n dexvision pytest` with
272 passed and `conda run -n dexvision ruff check dexvision tests`.

Level 2.6B — Reach-Touch Success Relabeling did not require manual
verification. The task-specific relabeler recomputes target distance and
consecutive qualifying palm-contact frames from saved task state, preserves
operator and recomputed labels together in a dataset-level JSON report, and
does not rewrite raw episode files. The five pilot episodes all recomputed as
successful and agreed with their operator labels. Automated checks passed on
July 18, 2026 using `conda run -n dexvision python -m
dexvision.apps.relabel_demos --dataset data/demos/raw/reach_touch_target`,
`conda run -n dexvision pytest tests/test_success_relabeling.py` with 8 passed,
`conda run -n dexvision ruff check dexvision tests`, and `conda run -n
dexvision pytest` with 280 passed.

Level 2.7 — Pilot Quality Filters did not require manual verification. The
read-only filter evaluates tracking confidence, missing frames, feature
jitter, action jerk, actuator-limit hits, recomputed task failure, and
workspace-limit hits using versioned configurable thresholds. It saves a
dataset-level JSON report grouped by skill and task without rewriting raw
episodes. The five reach-touch pilots produced four passes and one intentional
quality flag: `2026-07-18_003` exceeded the default action-jerk threshold
(`0.260570` versus `0.200000`). Automated checks passed on July 18, 2026 using
`conda run -n dexvision python -m dexvision.apps.filter_demos --dataset
data/demos/raw/reach_touch_target`, `conda run -n dexvision pytest
tests/test_quality_filters.py` with 8 passed, `conda run -n dexvision ruff
check dexvision tests`, and `conda run -n dexvision pytest` with 288 passed.

Level 2.7B — Reach-Touch Dataset Summary did not require manual verification.
The read-only summary scans the documented `data/demos/raw/` layout, groups
episodes by skill and task, reports success, episode length, tracking
confidence, action/observation schema versions, quality pass/fail counts, and
relabel disagreements, and saves JSON plus CSV outputs under
`data/demos/reports/summaries/`. The reach-touch pilot summary reports five
recomputed successes, four quality passes, one quality failure
(`2026-07-18_003` for high action jerk), and zero relabel disagreements.
Automated checks passed on July 18, 2026 using `conda run -n dexvision python
-m dexvision.apps.summarize_demos --dataset data/demos`, `conda run -n
dexvision pytest tests/test_dataset_summary.py` with 7 passed, `conda run -n
dexvision ruff check dexvision tests`, and `conda run -n dexvision pytest`
with 295 passed.

Level 2.7C — Scale Reach-Touch Dataset manually passed on July 18, 2026 after
the user completed the live collection and asked for the final dataset gates
to be applied. The immutable raw collection contains 76 episodes, of which 69
recompute as successful and 55 pass every Level 2.7 quality filter. Clean
training coverage is balanced across `reach_target_left` (18),
`reach_target_center` (18), and `reach_target_right` (19), with zero
operator/recomputed label disagreements. All 76 episodes validated and
completed headless replay for 9,540 recorded action steps. The versioned
`configs/reach_touch_dataset.yaml` split reserves
`reach_eval_left_center` at `[0.14, -0.05, 0.47]` metres and
`reach_eval_center_right` at `[0.14, 0.03, 0.50]` metres exclusively for
Level 3 evaluation. The v2 JSON/CSV summary reports the target distribution,
55 clean successes, full relabel/quality coverage, and `level3_ready: true`.
Existing failed raw attempts remain immutable for audit; the quality-gated
collection helper now balances clean successes and keeps new rejected attempts
outside `raw/`.
Automated checks passed on July 18, 2026 using the full 76-episode validation
and headless replay audit, `conda run -n dexvision python -m
dexvision.apps.relabel_demos --dataset data/demos/raw/reach_touch_target`,
`conda run -n dexvision python -m dexvision.apps.filter_demos --dataset
data/demos/raw/reach_touch_target`, `conda run -n dexvision python -m
dexvision.apps.summarize_demos --dataset data/demos`, `conda run -n dexvision
ruff check dexvision tests`, and `conda run -n dexvision pytest` with 303
passed.

Level 2.7D — Button-Press Task did not require manual verification. The shared
task-board scene now contains three passive, bounded spring-return buttons, and
`button_press` provides typed button/depth/pressed-state/approach parameters,
seeded deterministic reset, reconstructable task and initial state, explicit
terminal-state fields, and recomputable press-depth-and-dwell success metrics.
No button demonstrations, cube-push task, or learning code was added.
Automated checks passed on July 18, 2026 using `conda run -n dexvision pytest
tests/test_task_specs.py tests/test_success_relabeling.py` with 27 passed,
`conda run -n dexvision ruff check dexvision tests`, the existing reach-touch
headless smoke check, and `conda run -n dexvision pytest` with 313 passed.

Level 2.7E — Button-Press Pilot manually passed on July 18, 2026 after the user
recorded five bright-green-target attempts and confirmed the replacement center
take was clean. The retained pilot covers all three configured buttons and
target press depths from 0.010 to 0.014 metres. All five episodes validated,
completed headless replay, recomputed as successful, passed quality filtering,
and had zero operator/recomputed label disagreements. A target-isolation audit
confirmed the selected button was the primary press in every retained episode;
the replacement center episode pressed only its target button. Reach-touch
fixtures remain hidden and non-colliding only for button recording/replay.
Automated checks passed with 318 tests. No dataset scale-up, cube-push task, or
learning work was added.

Level 2.7F — Push-Cube Task did not require manual verification. The shared
task-board scene now provides a free-joint cube, deterministic start and target
selection, typed object/target/approach parameters, reconstructable current and
initial object pose/velocity plus target and robot state, explicit terminal
fields, and a recomputable planar distance-and-dwell success metric. Existing
reach and button tasks isolate the cube fixture. No cube demonstrations,
dataset relabel dispatch, or learning code was added. Automated checks passed
on July 19, 2026 using `conda run -n dexvision pytest
tests/test_task_specs.py tests/test_success_relabeling.py` with 37 passed,
`conda run -n dexvision ruff check dexvision tests`, and `conda run -n
dexvision pytest` with 328 passed.

Level 2.7G — Push-Cube Pilot is complete.
Recorder, saved cube/object state, semantic replay restoration, planar
distance-and-dwell relabeling, quality filtering, and summary integration have
automated coverage. Checks passed on July 19, 2026 using the checkpoint-focused
suite, Ruff, and the full suite. A follow-up usability correction on July 19,
2026 hides the vertical wall/unrelated fixtures, aligns the hand free joint and
mocap neutral immediately behind the cube, and disables nominal
task-timeout enforcement during live recording. The unusable
`2026-07-19_001` trial had no cube motion and was deleted at the operator's
request. Updated focused checks passed with 73 tests and the full suite passed
with 341 tests. A replacement `2026-07-19_001` tuning attempt had 100% hand
detection and contacted the cube, but its commanded base range was too weak
and the cube moved away from the target. That inspected attempt was also
deleted under the operator's authorization. The cube profile now uses stronger
translation/depth gains, faster smoothing response, a smaller depth deadband,
and a bounded larger position step. A subsequent `2026-07-19_002` attempt
showed that the remaining perceived delay was MuJoCo cadence: the generic
2-step loop advanced 0.004 seconds of simulation per roughly 0.033-second
camera frame, leaving 3.3 cm mean and 7.4 cm p95 physical-hand follow error.
Cube recording now advances at least one nominal control period per frame
(17 steps at 30 Hz with the current model), reducing a headless final follow
error from about 2.5 cm to 2 mm. That inspected failed attempt was deleted.
Another post-cadence `2026-07-19_002` attempt showed responsive tracking but a
control-frame mismatch: the simulated palm was horizontal while the operator's
real palm was upright. The cube traveled 27 cm and lifted more than 10 cm,
consistent with an abrupt scoop rather than remaining latency. Cube recording
now starts the robot palm upright to match calibration, moves its neutral
slightly farther behind the cube for a stable reset, and uses moderate
position/depth/orientation damping with a smaller per-frame step cap. The
inspected failed attempt was deleted under the operator's authorization.
Updated focused checks pass with 74 tests and the full suite with 342 tests.
A subsequent 410-frame `2026-07-19_002` trial had 100% hand detection, 0.949
mean tracking confidence, and only 2.3 mm mean physical follow error. The
remaining difficulty was geometric: the operator's palm faced the webcam, but
the simulated fingers rather than the palm normal pointed along the push axis,
and the cube moved 13.3 cm backward. Cube reset now uses a camera-facing,
fingers-up palm, maps webcam-relative depth along the palm normal, and starts
the viewer at a clear three-quarter angle. Palm/finger contacts remain active;
only forearm/wrist collisions below the tabletop are disabled. A headless
regression moves the cube about 18 cm along the table with less than 1 mm
vertical disturbance. Updated focused checks pass with 76 tests, Ruff passes,
and the full suite passes with 344 tests. The inspected failed trial was
deleted under the operator's standing authorization.
A further 418-frame `2026-07-19_002` trial exposed a reversed palm-facing reset
plus the Level 1.13 full-control preset re-enabling live base orientation.
Rendered frames showed the back of the hand at reset, the robot rotate out of
view after about 1.2 seconds, and later reappear sideways on the table. Cube
mode now uses the actual palm-facing identity quaternion and forcibly locks
base orientation while retaining translation, webcam-relative depth, and
finger control. The saved effective teleop snapshot records the resolved
control enable flags. A fixed-orientation headless push moves the cube about
21 cm along the table with less than 1 mm vertical displacement, and the
focused suite passes 76 tests.
A subsequent 756-frame `2026-07-19_002` trial kept the command orientation
fixed but exposed geometry clipping: the visible forearm/wrist crossed the
table at reset, and 15 cm of enabled image-height motion drove the palm into
the table. Contact deflected the physical hand sideways, with 3.6 cm mean and
11.6 cm p95 follow error plus cube lift. Cube mode now hides/disables the
below-table support bodies, fixes hand height, tightens forward/lateral bounds,
reduces depth gain, and ignores real-hand up/down. Named targets use the
matching cube-start lane for a straight planar push. A headless bounded push
ends 8.6 mm from target centre with 0.02 mm vertical drift, fixed orientation,
and 2.2 mm base-follow error. The focused suite passes 76 tests.
On September 2, 2026, `2026-07-19_004` was audited as a clean successful
127-frame planar push: 100% detection, 0.958 mean tracking confidence, 1.4 mm
mean follow error, fixed orientation, 1.1 mm maximum cube lift, successful
five-frame target dwell, clean rendered frames, headless replay, and a quality
pass. It is retained as immutable diagnostic/reference data but does not count
toward the five accepted pilots because the old recorder saved a null operator
label. Push-cube now uses the shared explicit operator-label prompt, and the
quality filter accepts intentionally fixed workspace axes without counting
them as limit hits. Focused checks pass with 79 tests, Ruff passes, and the full
suite passes with 347 tests.
The labeled `2026-09-02_001` retry is visually clean and agrees with recomputed
success: 100% detection, 0.964 confidence, 1.15 mm mean follow error, fixed
orientation, 0.72 mm cube lift, and 2.58 cm final target distance. It remains
raw diagnostic data rather than an accepted pilot because 48 of 147 frames
(32.7%) saturated the left workspace boundary, exceeding the 10% quality
limit. The target-aware lateral workspace now provides 7 cm of steering margin
on each side of the selected start/target lane while preserving fixed height,
orientation, and forward safety limits. Focused checks pass with 79 tests.
The labeled `2026-09-02_002` retry is also visually clean and agrees with
recomputed success: headless replay, 100% detection, 0.960 confidence, 1.09 mm
mean follow error, fixed orientation, 0.60 mm cube lift, and 2.49 cm final
distance all pass. It narrowly fails only the workspace gate because 12.8% of
frames saturated the lateral boundary, above the 10% maximum. It is retained
as diagnostic data. Since named starts and targets are lane-aligned, lateral
position is now fixed to the selected lane together with height/orientation;
only palm-normal forward motion controls the pilot. Focused checks pass with
79 tests.
`2026-09-02_003` is accepted as pilot 1/5 for `push_target_left`: its operator
label and recomputed success agree, headless replay and all quality gates pass,
tracking was 100% at 0.956 mean confidence, mean follow error was 1.44 mm,
maximum cube lift was 0.59 mm, and it ended 2.76 cm from target centre after
the required five-frame dwell. Rendered frames show a clean clipping-free
planar push. The diagnostic `2026-07-19_004`, `2026-09-02_001`, and
`2026-09-02_002` directories were deleted at the operator's request, and the
reports were regenerated. Push-cube summary aggregation now reports target
distribution and one clean success. Four more accepted pilots covering all
three target lanes remain required. Focused checks pass with 79 tests.
Four additional recordings (`2026-09-02_001`, `2026-09-02_002`,
`2026-09-02_004`, and `2026-09-02_005`) were audited on September 2. Together
with `2026-09-02_003`, all five have affirmative operator labels, recomputed
success, zero label disagreement, successful headless replay, and quality
passes. Tracking was present for 100% of every episode, mean confidence ranged
from 0.956 to 0.977, mean base-follow error ranged from 1.44 to 2.33 mm,
maximum cube lift ranged from 0.35 to 0.74 mm, and final planar target distance
ranged from 2.09 to 2.76 cm. Rendered final frames are clean. However, all five
recordings use `push_target_left`, so they do not satisfy the required coverage
of all three target zones. Retain the strongest three left-lane episodes
(`2026-09-02_002`, `2026-09-02_003`, and `2026-09-02_005`) and record one
centre and one right episode. At the operator's request, the redundant
left-lane `2026-09-02_001` and `2026-09-02_004` recordings were deleted and
the reports regenerated; the retained set now contains three clean left-lane
episodes awaiting those two replacements.
The replacement `2026-09-02_006` centre and `2026-09-02_007` right episodes
both pass schema validation, headless replay, relabeling, every quality gate,
and rendered start/middle/final inspection. They have 100% tracking, mean
confidence of 0.954 and 0.933, mean base-follow error of 1.32 and 2.22 mm,
maximum cube lift of 0.63 and 0.48 mm, and final planar target distances of
2.56 and 2.70 cm, respectively. The retained pilot now contains exactly five
clean successes with target distribution left=3, centre=1, right=1, zero label
disagreements, and five quality passes. During verification, replay was found
to default to one MuJoCo step per action even though these recordings saved 17
steps per frame; that default did not reproduce the contact dynamics. Replay
now automatically uses the saved cadence, retains an explicit override, and
falls back to one step for legacy demos. The focused checkpoint suite passes
with 57 tests, Ruff passes, and the full suite passes with 350 tests.
The exactly-five pilot, three-zone coverage, and pipeline-report requirements
are satisfied. On September 2, 2026, the user manually replayed all five
retained episodes and confirmed that their saved cube starts, target markers,
planar pushes, and final target behavior looked correct. Level 2.7G is complete.

Level 2.7H — Scale Button-Press Dataset manually passed on September 2, 2026
after the user completed 50 quality-gated live recordings. Together with the
five retained pilots, the immutable raw dataset contains 55 episodes. All 55
validated and completed semantic headless replay for 5,475 action steps,
recomputed as successful, passed every quality filter, and had zero label
disagreements. The nine training button/depth goals contain six or seven clean
successes each, three interpolated button/depth states remain held out, and the
v3 dataset summary reports `level3_ready: true`. Level 2.7H is complete.
Level 2.7I — Scale Push-Cube Dataset manually passed on September 2, 2026 after
the user completed the live quality-gated collection. The immutable raw dataset
contains 101 episodes and 7,176 action frames. All 101 episodes validated and
completed semantic headless replay using their saved 17-step cadence; all 101
recomputed as successful, passed every quality filter, and had zero label
disagreements. Clean goal coverage is left=33, centre=34, and right=34. Three
interpolated cube start/target-pose states remain held out, and dataset-summary
v4 reports `level3_ready: true`. Focused checks pass with 27 tests, Ruff passes,
and the full suite passes with 362 tests. Level 2.7I is complete.

Level 2.7J — Optional Skill Card Export Metadata did not require manual
verification. Policy-free JSON metadata stubs can be exported for the three
implemented task skills from their task specs and matching dataset-summary
groups. The stubs declare the full Level 1.13 action layout, typed parameters,
transition and terminal-state contracts, dataset provenance/readiness, and
known limitations while leaving the policy checkpoint unset for Level 3.
Automated checks passed on September 2, 2026 using `conda run -n dexvision
pytest tests/test_skill_card_metadata.py` with 7 passed, real CLI exports for
all three skills against the v4 dataset summary, `conda run -n dexvision ruff
check dexvision tests`, and `conda run -n dexvision pytest` with 369 passed.
Level 2.7J is complete.

Level 2.8 — Retargeter B: Fingertip Target Baseline did not require manual
verification. The baseline computes palm-local, palm-width-normalized fingertip
targets from MediaPipe-compatible landmarks, maps them to bounded Shadow Hand
targets with a simple geometric solve, and falls back to the last valid or safe
open targets when solving fails. Automated checks passed on September 2, 2026
using `conda run -n dexvision pytest tests/test_fingertip_retargeter.py` with 7
passed, `conda run -n dexvision ruff check dexvision tests`, and `conda run -n
dexvision pytest` with 376 passed. Level 2.8 is complete.

Level 2.9 — Retargeter C: Optimization Retargeter did not require manual
verification. The bounded solver minimizes palm-local 3D fingertip error,
configured actuator-limit violations, and change from the previous finger
controls. It uses SciPy L-BFGS-B when available, has a deterministic projected-
gradient path when SciPy is absent, clips every output to configured limits,
falls back to last-valid or safe-open targets, and records solve diagnostics.
Automated checks passed on September 2, 2026 using `conda run -n dexvision
pytest tests/test_optimization_retargeter.py` with 7 passed, the three-
retargeter suite with 30 passed, `conda run -n dexvision ruff check dexvision
tests`, and `conda run -n dexvision pytest` with 383 passed. Level 2.9 is
complete.

Level 2.10 — Retargeting Benchmark did not require manual verification. The
benchmark compares curl, fingertip, and optimization retargeters on identical
saved episode streams, saves JSON/CSV metrics and a dependency-free SVG plot,
and recomputes push-cube task success through counterfactual headless MuJoCo
replay. The September 2, 2026 10-episode baseline completed successfully;
the focused metric suite passed with 5 tests, Ruff passed, and the full suite
passed with 388 tests. Level 2.10 is complete. No Level 3 policy learning was
implemented.

Level 2.10B — Full-Dataset Retargeting Benchmark Validation did not require
manual verification. All 101 immutable push-cube episodes and 7,176 frames
were evaluated for curl, fingertip, and optimization retargeters. Versioned
JSON/CSV output now includes deterministic 95% episode-bootstrap intervals;
the SVG contains six metric panels; and headless MuJoCo replay measures actual
distal-fingertip-to-cube distance and contact-frame rate alongside task
success. The README reports the full results and explicitly discloses that the
shared base trajectories were originally collected with curl retargeting, so
independent live per-retargeter trajectories are not claimed. Focused checks
passed with 6 tests, Ruff passed, and the full suite passed with 389 tests.
Level 2 is complete. No Level 3 policy learning was implemented.

Level 2.11 — Learning Readiness Freeze did not require manual verification.
The completed Level 2 raw datasets, rejected attempts, and reports are packaged
as one immutable Git LFS archive with a SHA-256 checksum and release manifest;
the editable `data/demos/` working tree remains ignored. The first reach-policy
offline split, rollout-only held-out targets, initial-state perturbations, and
numerical acceptance gates are frozen in
`configs/level3_evaluation.yaml`. Existing episodes were not rewritten to
invent session ids, and the protocol explicitly forbids a cross-session claim.
No policy, training loop, rollout evaluator, VLM, or planner was implemented.

Level 3.0 — Roadmap Rebaseline did not require manual verification. It
established Level 3 as a bounded learning-feasibility phase and introduced a
later comprehensive data campaign. Level 3.0B corrects the later separation:
Level 4 owns the comprehensive multi-session dataset and immutable release;
Level 5 owns full-scale skill learning, qualification, supervised runtime, and
diverse scripted pilots; Level 6 owns portfolio polish, robustness, and
reproducibility; and Level 7 is future language-guided orchestration. Neither
docs-only checkpoint modifies completed Level 1/2 work or dataset contents.

Level 3.1 — Goal-Conditioned Per-Skill Dataset Loader did not require manual
verification. Automated checks passed on September 3, 2026 with 37 focused
tests and Ruff. The verified Level 2 release checksum passed, all three clean
skill datasets loaded successfully, and the frozen reach split produced 43
training and 12 validation episodes. A full-suite run excluding the unrelated
in-progress roadmap-doc test passed with 399 tests.

Level 3.3 — Behavior-Cloning Training Loop did not require manual verification.
Automated checks passed on September 3, 2026 with 18 focused tests, Ruff, and
the full 415-test suite. The configured 100-epoch CPU training command also
completed on the extracted immutable reach dataset, and its checkpoint digest
verified. No camera, GUI, live teleoperation, or MuJoCo rollout was involved.

Level 3.4 — Frozen Reach Closed-Loop Rollout did not require manual
verification. Automated checks passed on September 3, 2026 with 7 focused
tests, Ruff, and the full 420-test suite. The real checkpoint completed all 35
frozen headless scenarios and saved all 17 successes and 18 failures. It failed
the frozen success and safety gates without any held-out retuning; the exact
results are recorded in `outputs/level3/reach_rollout_v1/report.json`.

Level 3.5A — Button and Push Evaluation Freeze did not require manual
verification. Automated checks passed on September 3, 2026 with 8 focused
tests, 59 evaluation/dataset/task tests, Ruff, and the full 431-test suite. The
two v1 protocols preserve all reserved Level 2 conditions and are frozen before
any button or push policy training.

Level 3.5B — Checkpoint-Selection Repair and Cross-Task Baselines did not
require manual verification. Automated checks passed on September 3, 2026 with
23 focused tests, Ruff, and the full 437-test suite. Fixed-seed reruns reproduced
all best/last checkpoint digests. Corrected reach, button, and push completed
their unchanged frozen matrices and all failed their numerical gates; all
rollout failures and the reach v1-versus-v2 comparison remain in versioned
reports under `outputs/level3/`.

Level 3.6 — Data and Action-Space Diagnostics did not require manual
verification. Automated checks passed on September 3, 2026 with Ruff and the
full 442-test suite. The real CPU/headless matrix completed all 13 report rows
under the frozen task protocols. All common episodes retained their baseline
split in the broader reach comparison, and button/push correctly record that no
broader eligible counterpart exists. The versioned report is under
`outputs/level3/diagnostics_v1/`.

Level 3.7 — Conditional Temporal Baseline did not require manual verification.
The evidence trigger was assessed and not met: prior metrics do not measure
temporal ambiguity or isolate compounding error, and Level 3.6 keeps
compounding error and missing recovery coverage explicitly categorized as
hypotheses. The versioned decision records the measured action-space, safety,
and offline-to-rollout evidence. No sequence dataset, GRU, or temporal training
run was added. Automated checks passed on September 3, 2026 using `conda run -n
dexvision pytest -q tests/test_level3_temporal_decision.py
tests/test_roadmap_docs.py` with 12 passed, `conda run -n dexvision ruff check
dexvision tests`, and `conda run -n dexvision pytest -q` with 445 passed.

Level 3.8 — Feasibility Report and Level 4 Data Requirements did not require
manual verification. The report traces the negative full-action baselines,
checkpoint-selection change, controlled diagnostics, and temporal decision to
their configs, splits, datasets, schemas, checkpoints, protocols, and report
digests. It closes Level 3 with a no-go on policy qualification and a go on the
existing Level 4 workcell-data plan. Automated checks passed on September 3,
2026 using the focused Level 3.8/temporal/roadmap suite with 18 passed, Ruff,
and the full suite with 451 passed.

Level 4.0 — Workcell and Dataset Requirements Freeze did not require manual
verification. The frozen plan and YAML specification define the workcell/data
scope, typed skill and terminal-metric contracts, 79 split-owned coverage
cells, four whole-session split slots, reconstructable action-safety records,
causal online phases, fixed-camera visual conditions, quality gates, and Level
3 failure traceability. Automated checks passed on September 3, 2026 with 9
focused tests, Ruff, and the full 461-test suite. No Level 4 collection or
workcell implementation was started.

Level 4.1 — Workcell Scene, World State, and Task Contracts manually passed on
September 3, 2026. Automated checks passed with 14 focused tests, the
1,200-step headless inspector, repository-wide Ruff, and the full 475-test
suite. The first manual
attempt failed because the fixed camera was locked on an obstructed view and
the viewer exited normally after its hard 1,200-step limit. The corrected
inspector uses a movable overview camera, stays open until explicitly closed,
and exposes keyboard reset, seed, pause, and label controls. The checkpoint is
now also displaying workcell-only entity labels by default after follow-up
review found the unlabeled colors ambiguous and showed that MuJoCo hid the
original label anchors in disabled site group 4. The corrected anchors use
visible site group 0. A later review found that the overview visually reversed
the two return-bin sides and that objects could obscure the setup-slot markers. The
user confirmed the object/setup-slot issue was fixed, but camera-only changes
failed left/right re-verification because the frozen target coordinates encoded
the names opposite to the operator-facing view and the first camera regression
used a reversed horizontal-vector sign. The frozen target centers and MuJoCo
bodies are now corrected together so ids, labels, world state, and metrics
agree. The regression checks frozen-to-runtime coordinate equality and actual
screen-space ordering. The user confirmed that the final viewer passed, with
correct operator-facing return-bin sides and no object/setup-slot overlap.
Level 4.1 is complete.

Level 4.2 — Session-Aware Recording and Phase-Label Schema did not require
manual verification. New Level 4 episodes preserve genuine session/operator
provenance, typed goal/reset metadata, requested/commanded/applied and prior
actions, per-field safety masks/reasons, causal online phases, audited labels,
phase intervals, synchronized state/action/task timestamps, and optional
aligned RGB. Session manifests reject duplicate ids; resumed allocation and
atomic Level 4 writes never overwrite existing episodes. Legacy Level 2
episodes still load without invented Level 4 fields. Automated checks passed
on September 4, 2026 with 31 focused tests, repository-wide Ruff, a validator
module smoke run against a synthetic episode, and the full 489-test suite.
Level 4.2 is complete; Level 4.3A is the next target within the revised Level
4.3 sequence.

Level 4.3 — Collection Pilot and Final Coverage Freeze has automated pilot
recording, review, replay, and coverage-report tooling implemented, but remains incomplete. The
tooling keeps expert acceptance evidence append-only, retains ordinary failures
separately, reports episodes and derived pick/place segments separately, audits
session/cell split ownership, measures phase agreement and collection/storage
statistics, and leaves the count matrix provisional. The workcell recorder now
resolves frozen cells for reach, complete pick/place, push, and press, creates a
genuine append-only session per process, stores all six object states and task
metrics, emits online phases, and restores randomized objects for replay. The
optional dial is explicitly deferred. Automated checks passed on September 4,
2026 with 18 checkpoint tests, 86 controller/workcell regression tests,
repository-wide Ruff, recorder-help and coverage CLI smoke commands, and 508
full-suite tests.
The first manual reach attempt exposed a workcell control-frame defect: the
long forearm root was being positioned as if it were the palm, leaving the hand
cramped against the fixtures. That rejected episode was removed at the user's
request. The weld now offsets the forearm outside the board, reset aligns the
free joint without a transient, the recorder calibrates at the workcell neutral
palm pose, and Level 4 attempts request an explicit operator label. A second
genuine reach attempt was retained as an ordinary failure after it showed that
copying robot palm orientation was still ergonomically infeasible. Reach now
uses an upright, webcam-facing human neutral as a translation clutch, locks the
robot's palm-down orientation, displays a cyan target marker, and uses a closer
neutral plus higher translation gain so the first target does not require
near-edge camera travel. The third reach attempt held the palm within the
distance and orientation gates for 36 frames but displaced two neighboring
objects by roughly 7 cm because the marker was too low in the crowded scene.
The reach marker is now a collision-free pre-grasp target at 0.148 m, live task
state preserves the scene-disturbance metric, and an automated regression
proves five qualifying frames trigger recorder stop. The fourth attempt then
exposed a timing defect: at 30 camera FPS the recorder advanced only one 2 ms
physics step per frame, making visible hand motion about 17 times slower than
real time. It reached only 0.0745 m from the marker and disturbed the scene by
0.0218 m. Workcell recording now enforces 17 physics steps per frame, uses more
responsive filtering, and permits larger safe per-frame target motion. The
fifth retained attempt showed the timing fix was visibly better and entered the
distance gate for 32 frames, but low translation gain forced near-edge camera
travel and the wandering path accumulated 0.0223 m scene disturbance. The next
attempt came no closer than 0.0400 m and displaced `block_large` by 0.1728 m,
confirming that absolute monocular position mapping is not usable in this
cluttered scene. The pending manual trial replaces it with a centered, nonlinear
Cartesian velocity joystick: returning the hand to center stops the robot, a
high transit plane prevents lateral travel through clutter, and descent is
permitted only in a corridor over the target. The actual target object now has
a bright emissive magenta wireframe outline; the cyan cross separately marks
the desired palm position. `--workcell-dry-run` tests this without retaining an
episode or modifying the session manifest. At the user's request, the five
retained failed attempts and their manifest/report were removed from the active
pilot directory and parked recoverably at
`/private/tmp/dexvision_level4_pilot_pre_reset_20260904_001`; the active counter
was reset for `pilot_train_001/episode_000001`. Four clean retained trials now
provide the pivot evidence. Reach recomputed and visibly passed at 0.0162 m
terminal error with the required five-frame dwell and about 0.0021 m maximum
scene disturbance. Pick/place never qualified, moved `block_small` about
0.0486 m and a neighboring cylinder about 0.0450 m, and confirmed that contact
control is not usable through the current mapping. Push never moved the selected
block toward its 0.2884 m-distant target. Button press remained at zero depth;
the commanded base stayed near x=-0.18..-0.16 m while the wall button is near
x=+0.13 m. The user manually judged reach usable, pick/place and push unusable,
and button potentially viable only with a reachable planned approach. No expert
acceptance sidecars have been created, so the collection protocol, final count
freeze, and required visible replay checks remain pending.

The architecture decision is now made: webcam teleoperation remains a usable
reach/correction interface, while deterministic simulator-state experts become
the nominal source for contact skills. Level 4.3 is split into 4.3A–4.3I:
common expert plus scripted reach; button; constrained push; grasp-and-lift;
place/complete pick-place; replay qualification; small state-only button and
push learnability probes; and final source/count freeze. The first learned
interface is a low-dimensional task-local delta expanded by deterministic code
into the unchanged full action schema. Do not add RGB, an LLM/VLM, a larger
model, a general planner, action chunking without evidence, or a bulk haul.
Do not mark Level 4.3 complete or advance to Level 4.4 until all lettered
checkpoints and visible replay gates pass.

Level 4.3A is complete. `dexvision/sim/level4_expert.py` provides the common
deterministic reset/step boundary and a configuration-owned four-segment reach
through rise, horizontal transit, protected corridor entry, and descent. Every
candidate action is first exercised in copied MuJoCo state and checked for
workspace, joint-limit, table/fixture-contact, and non-target-disturbance
failures. Scripted reach uses the existing `record_demo` logger and unchanged
27-field Level 4 action schema; requested, commanded, and applied rows are
preserved with zero safety interventions, and replay restores the same fully
centered object cage and goal cross. Automated checks passed on September 4, 2026 with 8
focused tests over five randomized seeds, 58 related regression tests,
repository-wide Ruff, and the full 517-test suite. The user confirmed the
corrected visible pre-grasp replay, which advanced work to Level 4.3B.

The first visible replay review initially remained pending because `--speed 0.1`
displayed only about three action updates per second and exposed a real target
cue defect: the cage began at the block center and did not enclose its lower
half. The cage geometry is now centered vertically around the selected entity,
with an automated enclosure regression. The user then accepted the corrected
pre-grasp reach behavior and confirmed that stopping at the cyan cross without
cube interaction was the intended Level 4.3A result.

Level 4.3B is complete. `DeterministicButtonPressExpert` uses a fixed neutral
hand pose and orientation, a task-relative safe approach, button-normal contact
motion, the frozen press-depth/dwell metric, and a complete retract/release
phase. The button was moved into the declared safe workspace after the pilot
proved its prior height unreachable. Copied-state validation and five
randomized record/replay checks reject unintended contacts and meaningful joint
limit excursions; all retained scripted samples have zero safety masks,
reasons, and interventions. The listed 2-test checkpoint suite and 38 related
regression tests passed on September 4, 2026. No manual verification was
required. This advanced work to Level 4.3C.

Level 4.3C visible replay remediation is complete. The first
visible cuboid replay failed review because the block tipped about 45 degrees,
briefly crossed the distance threshold, and rolled back while the recorder
retained the transient success. The remediated controller uses a lower,
partially flexed cuboid index posture, parks non-target objects for standalone
skill trials, and requires table support, at most 10 degrees of tilt, and a
still-qualified metric for five samples after hand release. Five cuboid and
flat-puck resets pass copied-state validation and replay; the 2-test checkpoint
suite, the 33-test combined expert/collection suite, repository-wide Ruff, and
the full 524-test suite pass. The user confirmed on September 4, 2026 that the
new cuboid push slides cleanly without tipping; the accompanying button and
standalone grasp replays also passed visual review.

Level 4.3D is complete. `DeterministicGraspLiftExpert` uses separate
configuration-owned cuboid, cylinder, and flat-puck object-relative templates,
fixed wrist orientations, deterministic retargeter open/full-flexion endpoints,
and one scalar closure synergy per family. Standalone trials retain the selected
object's seeded workcell pose while parking non-target objects on the lower
floor, isolating grasp physics from the pilot-confirmed clutter limitation
without removing objects from the schema. Copied-state and live qualification
require initial table support, a measured two-contact held relation, at least
0.040 m of lift, loss of table support, retention, and ten stable samples below
0.020 m/s. Three seeds per family record and replay deterministically with
planar neighbor disturbance below 0.005 m and zero logged safety interventions.
The listed 2-test checkpoint suite, 52 related regression tests,
repository-wide Ruff, and the full 524-test suite passed on September 4, 2026.
No manual verification was originally required. Visible review noted that the
cuboid rotates substantially during the otherwise stable grasp.

Level 4.3E is complete. The complete controller uses copied-state-validated transport,
descent, release, settle, and retract behavior. After visible review exposed
about 24 degrees of cuboid yaw in the prior top-down grasp, the shared grasp
stage was revised to a shape-aware side-on (hammer-curl) wrist pose. A declared
rotation-only hold keeps the seeded object orientation through lift, stabilize,
transport, and place without constraining translation or release. Cuboid wrist
yaw follows the seeded object yaw; axial objects use their rotational symmetry.
Family release offsets remain configuration-owned, and success requires the
released object to stay within 0.025 m of its target while supported, unheld,
and below linear and angular speed limits. Task-local six-dimensional table
contact and rolling resistance apply only to standalone pick and composed
pick/place, preserving push behavior; the exact overrides and source/target
cues are restored from episode metadata. Ten complete successes span cuboid,
cylinder, and flat-puck families; all recompute and replay within five degrees
of the seeded held orientation, with compatible reach/pick/place segments, zero
safety intervention, and less than 0.005 m neighbor disturbance. Focused place,
segment, standalone grasp, and push suites pass after this remediation; the
related suite passes with 41 tests, repository-wide Ruff passes, and the full
suite passes with 527 tests. The saved viewer now uses an oblique camera that
keeps the hand, object, and named goal sides readable. The user confirmed on
September 4, 2026 that the corrected cuboid pick/place replay worked.

Level 4.3F is complete. The versioned cross-skill audit regenerated, headlessly
replayed, and recomputed two independently seeded accepted episodes for each of
reach, button, push, standalone grasp-and-lift, and complete pick/place. All 10
had complete action/provenance/timestamp evidence, zero safety interventions,
and neighbor disturbance below the configured limit. An ordinary failed episode
is covered by the automated audit and remains rejected without contributing to
expert counts. The required 25-test checkpoint suite passes. The existing
operator-owned pilot directory remains intentionally unchanged and below its
later collection minima. The user confirmed on September 5, 2026 that one
visible replay for all five skills looked good. Last Completed is now 4.3F and
Next Target is 4.3G; no 4.3G work has started.

The team-facing interim findings, four-action measurement table, adopted
scripted-expert pivot, and revised plan are documented in
`docs/level4_pilot_report.md` under **Interim Mini-Report — Teleoperation
Feasibility**.

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
