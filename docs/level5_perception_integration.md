# Level 5.7 instrumented simulator integration

Checkpoint 5.7 remains incomplete until owner manual acceptance. This is a new
instrumented simulator experiment, not an expansion of V29's accepted natural
RGB qualification. The earlier rejection-only preflight and all failed development
attempts remain preserved. No 5.8 work or policy training is authorized here.

The owner delegated the observation-source choice after being told V29 cannot
supply the accepted policies' semantic identity, orientation, motion, contact or
fixture inputs, and asked to avoid large additional storage. The chosen amendment
uses visible rigid optical rigs and explicit ideal simulated sensor ports. It
requires no image dataset, new neural fit, model selection or bulk data release.
All accepted model weights, transforms, engineered decoders and source pins stay
unchanged. `dexvision/evaluation/evaluate_skill.py` remains byte-identical to its
accepted version; a private function-global mapping changes only the observation
boundary of the original rollout functions.

## Sources and limits

- Requested movable objects and fixtures carry six calibrated ArUco faces on a
  virtual cube. Only the requested entity's rig is rendered. Object rigs have a
  0.35 m body-local vertical offset; the fixture offset is (-0.12,-0.12,0.10) m.
  Tags are 0.05 m squares on a cube of half-size 0.035 m. These are render-only
  rigid decorations, with no inertia/collision and no real-camera claim. Their
  large offsets and lack of physical support are explicit simulation limitations.
- The calibrated 1280x960 camera looks at (0,0,0.23) m. Pilot goal decorations are
  hidden. RGB pixels alone enter ArUco detection and geometric pose estimation.
  At least two distinct tag faces and <=1.5 px inlier reprojection error are
  required. Deterministic pair hypotheses reject corrupt faces. Known marker
  geometry maps pose back to the object's body center and world-frame wxyz pose.
  IDs/classes come from the predeclared tag atlas, not hidden simulator labels.
- Two distinct-time poses provide backward-difference average linear/angular motion
  ending at the newest acquisition, rather than an exact instantaneous velocity. No zero motion
  is fabricated at birth. Camera acquisition occurs every three original control
  ticks (about 9.8 Hz versus 29.4 Hz control). Holds retain the original timestamp,
  without extrapolation or restamping. The 0.1 s delivery/action-age limit is
  unchanged and includes processing time. Confidence 0.99 is a deterministic
  geometry-validity heuristic, not calibrated statistical uncertainty.
- Ideal simulated ports expose robot joint/base position and velocity, hand
  mocap command feedback, actuator commands, named fixture slide encoders and
  named body-pair contact switches. The free-base port is ideal pose telemetry,
  not a claimed physical joint encoder. These ports are explicitly privileged
  simulated measurements; this is not camera-only control or hardware validation.
  No movable object's position, quaternion, motion or task outcome is exported
  by a sensor port. Missing/stale ports reject without fallback. Fixture X/depth
  comes from its encoder; cross-axis pose and identity still require vision.
- Fixed target poses and zero motion are declared calibration invariants. Mounted
  fixture/target manipulation relations are explicitly inapplicable. Object
  support/retention combines observed pose and the declared contact switches.
- The calibrated model is the declared trial configuration: camera intrinsics/
  extrinsics, robot kinematics, static target/fixture/table poses, object dimensions,
  masses and friction (including predeclared procedural variations). These model
  parameters are privileged simulator calibration, not visual estimates. Dynamic
  object qpos/qvel are never calibration inputs. This limits deployment claims.
- Place's original preview logic runs only on a fresh estimated model/data copy,
  populated with these optical and sensor measurements. It never copies live
  MjData. Unobserved movable bodies are excluded from this partial preview;
  scripted entry parks non-targets away from the interaction, and an unexpected
  measured robot/requested-object contact with an unobserved body rejects.
  This limited preview is an explicit source amendment and does not establish
  clutter avoidance. The paired state baseline keeps its original live preview.
- Post-entry phases and press latches are causal functions of the measured world.
  Truth-derived runner phases/latches/rest positions are discarded at policy
  calls. Reach approach geometry uses the inferred entry pose and the frozen
  geometric decoder conventions. No expert runs after the entry boundary.
  Independent simulator truth is used only for terminal/safety scoring and
  post-proposal error measurement. Evaluation termination is not a deployed
  observation-based supervisor qualification.

## Frozen comparison and compact evidence

Both entry matrices are frozen before optical/control evaluation, using new
scripted seeds on previously exposed task families. Original required cells,
three model seeds (5001/5002/5003), >=30 unique test resets per skill and >=3
resets per required cell are retained. Each reset crosses all four original
visual conditions. One independently executed state reference per reset/model
is reused for the four comparisons; those copies do not inflate independent
state counts. Identical three prior-command bootstrap holds occur in both arms;
entry qpos/qvel/time hashes must match exactly. Every failed setup remains counted.
Pick uses its accepted unassisted approach prefix; place uses its accepted
scripted pick/post-pick setup. Contact experts stop before the first contact-stage
request. Fixed contact-family finger posture comes from the original validation
prefix, and push retains its accepted settling holds. No held-out learned rollout
is searched for a better start, and no post-entry expert correction is applied.
The held-out matrix has reach 30 resets/3 cells, pick 54/18, place 90/30,
push 32/4 and press 30/2: 236 independent skill resets, 708 independent state
executions and 2,832 perception attempts across three models/four conditions.
Validation has 42 resets and 188 perception attempts including 20 diagnostic
interventions, plus 42 independent state executions.

Validation uses seed 5001 and one fresh reset per cell. Its first case per skill
also probes pixel bias, image dropout, 120 ms latency and contact-port dropout.
Validation is diagnostic, not training or model selection. All three exact
receipt-bound selected bundles are then evaluated once on a disjoint frozen test
matrix. Reservations prevent a new output path from retrying a consumed split.
Implementation/config hashes and a compact source snapshot bind the experiment.

No per-frame RGB dataset is saved. Float64 state/action/observation traces are
compressed; only the first initialized observation-boundary attempt per skill/condition for seed 5001
gets a 640x480 saved review clip. This predeclared scheduling rule skips only
pre-entry setup failures that never create an observer; it does not inspect or
select policy success, terminal status or optical accuracy. The initial v1
freeze-only protocol is preserved; its review scheduling was corrected before
any optical/control evaluation, then v2 froze the identical deterministic entry
assignments again. Neither v1 matrix was consumed by a learned policy. Results preserve negative outcomes, acquisition
ages, source provenance, physical audits, selected-weight hashes, latency,
success/degradation/jerk/safety and per-condition/cell/family breakdowns. A hard
640 MiB limit covers the final experiment, frozen prefixes and source snapshot.
Final payload bytes are 426.2 MiB (about 469 MiB allocated on this filesystem).
Earlier development evidence is separately retained and not counted as test data.
Validation projected about 430 MiB before held-out clips, so a storage-only
amendment raised the original 320 MiB ceiling to 640 MiB before any test run.
The original protocol, validation reports, frozen entries and source snapshot
remain immutable. Protocol loading verifies all observation/policy pins unchanged
and compares the original integration AST: all original functions except the
protocol loader remain identical, with only inherited-proof references normalized
in the evaluator. No seed, model, control, metric threshold or visual condition
changed. `storage_amendment.json` binds the exact parent/active hashes.

An independent event-analysis recipe is also frozen before test. It fills missing
pick/drop and place/release/settle event metrics for interrupted attempts by
forwarding saved qpos/qvel in a separate simulator, without stepping a recorded
trajectory, rendering, or invoking a policy. Original results are untouched.
Use this supplement for complete event gates; the engine's missing-as-zero event
aggregates are only lower bounds when early interruption lacks terminal fields.
Zero-action attempts have no policy-caused event and stay in the denominator.
No missing event may certify a passing coverage gate.

## Commands

From the repository root, using the dexvision Conda environment:

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/evaluation_perception.yaml --all-required-skills --track perception --freeze-only
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/evaluation_perception.yaml --all-required-skills --track perception --split validation --output-dir outputs/level5/perception_instrumented_v2/validation
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/evaluation_perception.yaml --all-required-skills --track perception --output-dir outputs/level5/perception_instrumented_v2/test
conda run --no-capture-output -n dexvision pytest -q tests/test_instrumented_recovery.py tests/test_perception_storage_amendment.py tests/test_instrumented_events.py tests/test_instrumented_groups.py tests/test_instrumented_review_overlay.py tests/test_instrumented_perception.py tests/test_perception_grounded_rollout.py tests/test_perception_policy_integration.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check .
conda run --no-capture-output -n dexvision pytest -q
git diff --check
python -m dexvision.apps.audit_instrumented_events --config configs/level5/perception_event_audit_v2.yaml --evaluation-dir outputs/level5/perception_instrumented_v2/test --output-dir outputs/level5/perception_instrumented_checks/test_events
```

Do not rerun a consumed freeze/evaluation command. Use the saved report and clips.
Final software checks pass: **1,235 passed, one skipped in 765.05 s** in the
full suite; **54 focused tests passed**. Ruff and whitespace checks pass. Exact
artifact, source, audit and test bindings are in
`level5_perception_integration_verification_v2.json`. Failed/interrupted software
logs remain preserved alongside the final passing logs.
Original scorer fields such as `world_states_path`, `render_condition` and
`render_condition_effect` retain their legacy staging/state-scorer meanings;
verbose staging world JSON is intentionally not retained. Use `trajectory.npz`
and `observations.npz`, plus the new `condition` and `observation_source` fields,
for the actual saved trace and optical condition. The final integrity audit
checks those compact artifacts, not nonexistent temporary staging paths.
Engine latency summaries include rejected observer calls and the final QA image;
saved action-ready latencies identify the actual admitted control queries.
No separate detector IoU/recall or hardware calibration qualification is claimed
for this fiducial producer; the unchanged V29 receipt applies only to V29. Failed
numerical gates remain failed; a successful software run is not skill qualification.

## Frozen test outcome (owner review pending)

All 3,540 scheduled requests are accounted for: 708 independent state references
and 2,832 perception attempts on 236 unique skill/reset assignments. The 12 RGB
requests paired with a failed place setup explicitly reuse the measured failed
setup and never invoke observation/control. There are 20 fixed held-out review
clips. All 218 frames also pass the read-only corrected-header presentation audit,
with source video/result hashes and pixels below the header unchanged. All 13,017 indexed artifacts, 161,301 consumed feature rows, paired entry
hashes and field-age/action alignment read back correctly. The original 4,917
pre-interruption files remain byte-identical. Applied policy actions: 67,715
state and 161,184 perception; setup/expert actions are excluded.

Rates below are for seeds 5001 / 5002 / 5003, with all scheduled failures included.
These are exposed-family instrumented simulator results, not unmarked RGB or
hardware qualification.

| Skill | State success % | Perception success % | Canonical worst-family % | Disposition/reason |
|---|---|---|---|---|
| Reach | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | Experimental: cadence gate fails |
| Pick | 100 / 100 / 100 | 41.7 / 45.8 / 43.5 | 30.6 / 33.3 / 31.9 | Marker loss, degradation, jerk/cadence; seed 5003 has two object-workspace violations |
| Place | 85.6 / 85.6 / 82.2 | 45.0 / 44.2 / 41.9 | 35.8 / 33.3 / 27.5 | Marker loss, degradation, settling/cadence failures |
| Push | 50 / 50 / 50 | 0 / 0 / 0 | 0 / 0 / 0 | Failed: half the reset setups tip the object; remaining perception requests lack fresh two-marker pose/motion; zero applied perception actions |
| Press | 100 / 100 / 100 | 100 / 100 / 99.2 | 100 / 100 / 99.2 | Experimental: cadence, plus jerk for seeds 5002/5003 |

Pick's reconstructed drop rate is 0% for all three seeds. Place's reconstructed
settled fraction among nominal placements is 86.2% / 86.4% / 81.6%, below the
90% gate. Its premature-release rate is 1.11% per perception seed. No unavailable
event field is treated as evidence of a passed gate. The two pick violations
occur in seed-5003 mild-illumination cases 002 and 004; they remain failures.


Read-only admitted-control timing also fails cadence, independently of the
engine's final-QA-image timings: reach has 35 overrun samples, pick 2,076,
place 5,546 and press 241 (34 ms limit). Their per-seed p95 action-ready
latencies are respectively 30.55–31.31, 31.84–36.77, 33.96–34.49 and
34.18–34.63 ms. Push has no admitted actions. All admitted action ages,
including processing, remain <=100 ms; this safety-age pass does not waive
the stricter cadence gate. Details are in the hashed `control_latency.json`.

Every perception seed misses the original zero-miss 34 ms cadence gate. No skill
passes all qualification gates, and no default registry admission is claimed.
Push's 384 scheduled perception attempts are explicitly not executed closed-loop
push rollouts. All 30/54/90/32/30 reset minima and required cells are retained,
including failed setups/rejections as required by the original denominator rule.
The final verification JSON binds corrected canonical group/event metrics while
preserving original engine summaries and every negative outcome.

The one-time test is consumed and closed: no training, model selection, tuning
or control rerun may use it. Saved-state analysis and hashed media playback are
allowed. Checkpoint 5.7 remains incomplete pending the owner's manual review;
manual acceptance cannot turn failed numerical skill gates into qualification.

## Owner manual verification

```bash
conda activate dexvision
python -m dexvision.apps.review_instrumented_results --evaluation-dir outputs/level5/perception_instrumented_v2/test
```

Review every scheduled clip (at least one nominal held-out rollout per required
skill), using `n` to advance and Escape to cancel. Pass when marker identity,
body-center pose, acquisition age, action/phase and the saved terminal result
agree throughout. A declared missing/stale observation must stop execution;
failures/rejections are acceptable negative evidence only when accurately shown.
Fail on unreadable/missing media, swapped identities, inconsistent overlay/body
pose, an action admitted with stale/missing required fields, or a terminal result
that disagrees with physical behavior. This review cannot turn failed numerical
skill gates into a pass. Report the result explicitly; completion stays pending.

## Interrupted request accounting

The original held-out command stopped at place/5001/006/nominal after its scripted
post-pick setup failed. The observer was never initialized, then a precondition
callback and error recorder dereferenced the absent context. No learned policy
control ran for this interrupted request. The original log, 1,291 completed
requests and all 4,917 existing files are preserved and hash-indexed before
continuation. The failed setup's paired state result is the measured evidence.

A separately locked recovery runner preserves every completed request, records
the empty interrupted request as a failed pre-entry disposition, and executes
only unstarted requests with the original frozen action/observation functions.
A failed paired place setup rejects RGB requests before observation/policy work;
its physical setup evidence is explicitly reused, not counted as independent
perception execution. Safety counters and all rejected requests stay counted.
The regression test verifies this path and rejects relabelling a started/valid
attempt. No model, pose, decoder, threshold, seed or nuisance condition changes.
The original command is not rerun, and consumed control attempts are not retried.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.recover_instrumented_rollouts --config configs/level5/perception_rollout_recovery_v1.yaml
```

A separate canonical family audit uses the frozen request's object and goal
mappings for every outcome, including early failures missing legacy target
labels. Missing legacy labels must not invent an extra all-failure goal family.
The original engine summaries remain unchanged; the final verification reports
both the engine value and the canonical object/goal-family minimum under the
original thresholds. Group boundaries come from frozen metadata, not outcomes.
Successes require an independent physical audit; unknown physical success cannot
pass. This read-only reporting correction does not rerun or tune controllers.

The frozen raw recorder's `calls=0` HUD counter was not forwarded through its
observer proxy. This is a display-only diagnostic; action traces and feature
hashes are independent. The manual command above uses a read-only player that
replaces only that header in memory with actual recorded applied-action counts,
XYZ and measured phase. Original clips, pose overlays, acquisition-time footer,
terminal results and all file hashes remain unchanged. No extra video dataset
is generated, and the player never records owner acceptance.
