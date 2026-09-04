# Level 4.3 Collection Pilot Report

Status: **pilot evidence collected; scripted-expert pivot selected; implementation pending**

This report is intentionally not a claim that Level 4.3 has passed. The
provisional 250–350 episode coverage matrix remains unchanged until the adopted
Level 4.3 sequence supplies the evidence required to revise it at 4.3I.

## Interim Mini-Report — Teleoperation Feasibility

Date: September 4, 2026

### Question evaluated

Can monocular webcam hand tracking provide usable demonstrations for reach,
pick/place, push, and button press in the crowded Level 4 workcell?

### Clean four-action trial

| Skill | Frames / duration | Measured result | Operator finding |
|---|---:|---|---|
| Reach | 724 / 24.31 s | Recomputed success; 0.0162 m terminal error; required 5-frame dwell; about 0.0021 m maximum scene disturbance | Usable with centered nonlinear rate control and explicit target cue |
| Pick/place | 834 / 27.98 s | No qualifying grasp or placement; phases reached approach/acquire/lift; `block_small` moved 0.0486 m and neighboring `cylinder_tall` moved 0.0450 m | Current teleoperation is not viable for grasp/contact control |
| Push | 297 / 9.92 s | Selected block did not move; target distance remained 0.2884 m | Current approach cannot establish and maintain the required contact path |
| Button press | 161 / 5.42 s | Zero press depth; commanded base x remained about -0.18 to -0.16 m while the wall button is near +0.13 m | Potentially viable only with a reachable planned approach |

The bright magenta source outline and separate cyan goal cross resolved the
previous target ambiguity. The centered nonlinear rate controller also resolved
the travel-versus-precision problem for free-space reach. These improvements did
not make contact-rich tasks usable.

### Finding

The pilot supports **free-space reach teleoperation**, but it rejects the
assumption that pick/place, push, and press are directly collectable as simple
extensions of the same monocular control mapping. Their failures are dominated
by collision-free approach, reachable orientation, grasp/contact establishment,
and constrained contact motion—not by missing reach examples.

One trial is visibly and recomputably successful, but the coverage tool reports
zero expert-accepted episodes because no append-only acceptance sidecar or
manual replay evidence has been created. Level 4.3 therefore remains incomplete.

### Recommended pivot

Use a hybrid classical pipeline:

1. Plan collision-free hand-base motion through a clearance plane and protected
   approach corridor.
2. Use fixed task-specific pre-contact poses and orientations for grasp, push,
   and wall press.
3. Execute short verified contact controllers for close/lift, straight-line
   push, and press/retract.
4. Retain the working reach teleoperation as a correction/debug interface.
5. Learn only residual contact behavior or policies from successful
   planner-generated trajectories; record their provenance as planned/scripted,
   not human teleoperation.
6. Defer VLA control to later high-level target selection and qualified-skill
   orchestration. It is not the next solution for low-level collision avoidance.

### Decision adopted before collection resumes

Level 4.0 allowed scripted provenance in the schema, but Level 4.3 began with
the operational assumption that human teleoperation could supply all four pilot
groups. Continuing under that assumption is not supported by the evidence. The
team selected the hybrid direction. Level 4.3 is now split into explicit
lettered checkpoints that build deterministic, simulator-state experts in this
order: reach, button, constrained push, grasp-and-lift, then place and complete
pick/place. Each expert must recompute and replay successfully with no safety or
neighbor-disturbance failures before the next data decision.

After the expert architecture qualifies, two bounded learnability probes use
20–50 scripted successes, a small MLP, simulator state, causal phase input, and
a low-dimensional task-local Cartesian action. Deterministic control expands
that output into the unchanged full requested-action schema. Button is tested
before push. Vision, action chunking, larger models, recovery learning, and
language orchestration remain evidence-gated future work.

Do not begin the Level 4.4 data haul or treat the current reach run as accepted
until Level 4.3A–4.3I and the required acceptance/replay evidence are complete.

## Frozen Pilot Protocol

Collect at least 25 accepted episodes across at least two genuine recording
sessions:

| Data group | Accepted minimum |
|---|---:|
| Standalone reach | 5 |
| Complete pick/place | 10 |
| Push | 5 |
| Button press | 5 |

The ten complete pick/place episodes must span cuboid, cylinder, and flat-puck
families. Across the pilot, every configured target type must appear. Ordinary
failures and rejected attempts remain separate from expert successes.

The optional dial is **deferred**. It is disabled in the workcell, and required
skill coverage takes priority. No dial episodes are included in the required
250–350 episode budget.

## Acceptance and Measurement

Each expert-accepted episode needs an append-only `pilot_review.json` beside
the immutable episode arrays. Acceptance requires schema validation, timestamp
alignment, headless replay, terminal-metric recomputation, operator/recomputed
label agreement, quality thresholds, valid coverage assignment, and session
split audit. Later visible replay confirmations append separately to
`manual_replay_manifest.json`, so neither the episode nor its automated review
is overwritten. The coverage summary reports:

- collection minutes per accepted episode;
- explicit rejection reasons;
- audited/online phase-label agreement;
- replay and metric-recomputation evidence;
- object-family and target-type coverage;
- episode and derived-segment counts separately;
- bytes per episode and projected release payload handling.

The Git LFS planning threshold is 2 GiB for the projected episode payload. A
projection above that threshold selects external immutable object storage for
payloads while Git retains manifests, checksums, splits, licenses, and retrieval
instructions. The current four-trial projection selects Git LFS, but that result
is preliminary because no representative accepted-task mix exists yet.

## Commands

Generate the read-only coverage report:

```bash
conda run -n dexvision python -m dexvision.apps.summarize_level4_coverage \
  --config configs/level4_dataset.yaml \
  --dataset-dir data/demos/level4_pilot
```

Record commands use a genuine session, a stable pseudonymous operator id, and
the exact frozen coverage-cell id selected for the attempt. One invocation is
one fresh process/calibration session. The first training reach pilot is:

```bash
mjpython -m dexvision.apps.record_demo \
  --task level4_workcell \
  --skill reach_object \
  --session-id pilot_train_001 \
  --operator-id operator_local_01 \
  --session-split train \
  --goal-condition-id reach_block_small_interior \
  --level4-dataset-config configs/level4_dataset.yaml \
  --workcell-config configs/workcell.yaml \
  --task-seed 0 \
  --require-hand-detected \
  --min-hand-detected-frames 10
```

The recorder selects the workcell model, enables full Level 1.13 camera/viewer
control, creates the session manifest entry, and allocates the episode path.
Use a new session id for every new invocation; reusing an id from another
process is rejected as false session provenance.

Before recording more evidence, test the replacement reach controller without
retaining frames or changing the session manifest:

```bash
mjpython -m dexvision.apps.record_demo \
  --task level4_workcell \
  --skill reach_object \
  --goal-condition-id reach_block_small_interior \
  --workcell-dry-run \
  --level4-dataset-config configs/level4_dataset.yaml \
  --workcell-config configs/workcell.yaml \
  --task-seed 0 \
  --duration-seconds 45
```

## Manual Replay Gate

After automated pilot requirements pass, replay one accepted standalone reach,
one accepted complete pick/place sequence (covering pick and place), one push,
and one press episode with the MuJoCo viewer:

```bash
mjpython -m dexvision.apps.replay_demo --demo <accepted-episode-directory>
```

A replay passes only when saved state/action timing reproduces the intended
motion and the recomputed terminal result matches the visible result. A wrong
object, wrong target, phase misalignment, unexplained collision, or label
disagreement fails. Level 4.3 stays incomplete until the user confirms these
visible replays.

## Results

The first manual reach attempt exposed a control-frame defect: the long forearm
root occupied the board because it was positioned as the logical palm point.
The user rejected the attempt and requested deletion, so its episode directory
was removed while its append-only session provenance remains. The workcell weld
and recorder neutral mapping were corrected. A second genuine attempt improved
approach distance but was retained as an ordinary failure because imitating the
robot's palm-down orientation was physically awkward and produced a 1.19-radian
orientation error. Reach now fixes robot orientation, uses a comfortable
webcam-facing upright palm as a translation clutch, and renders a cyan approach
marker. The neutral pose and translation gain were adjusted because the failed
trajectory showed that the original mapping required near-edge camera travel.
The next attempt reached the 2.5 cm distance gate for 36 frames with zero
orientation error, but moved `block_large` and `cylinder_short` by roughly 7 cm;
the low marker had routed the palm through the staged objects. The reach cue is
now a 0.148 m collision-free pre-grasp target, dense state exposes disturbance,
and a five-frame success/automatic-stop regression passes. Accepted pilot
collection remains pending. A subsequent attempt exposed that the 30 FPS
recorder advanced only one 2 ms physics step per frame, so visible motion lagged
real time by about 17x; its minimum distance was 0.0745 m and scene disturbance
reached 0.0218 m. Workcell recording now enforces 17 physics steps per frame and
uses more responsive target filtering and step limits. Run the summary command
after each session and copy metrics here only after acceptance sidecars and the
session manifest validate. The following attempt entered the distance gate for
32 frames but needed near-edge camera travel and accumulated 0.0223 m scene
disturbance. A further attempt came no closer than 0.0400 m and displaced
`block_large` by 0.1728 m, showing that increasing absolute-mapping gain cannot
provide both travel and precision in this cluttered scene. The replacement
manual trial uses centered nonlinear rate control, high transit, and target-only
descent. A bright magenta wireframe cage identifies the actual target object,
while the cyan cross marks the desired robot-palm goal. The user found the new
marker materially clearer and then requested a clean retained test of each
required action. The five historical failures plus their manifest/report were
removed from the active pilot directory and parked recoverably at
`/private/tmp/dexvision_level4_pilot_pre_reset_20260904_001`. The active pilot
counter is reset; the next recording is `pilot_train_001/episode_000001`.

A clean four-action trial followed. Reach visibly and recomputably succeeded:
terminal error was 0.0162 m, the five-frame dwell completed, and maximum scene
disturbance was about 0.0021 m. Pick/place advanced through approach, acquire,
and lift labels but never qualified; `block_small` moved about 0.0486 m and
`cylinder_tall` about 0.0450 m through unwanted contact. Push left its selected
block stationary and remained 0.2884 m from the destination. Button press stayed
at zero depth because the commanded base remained around x=-0.18..-0.16 m while
the wall fixture is near x=+0.13 m. The user judged reach usable, pick/place and
push unusable through this teleoperation mapping, and button potentially viable
with a different reachable approach. No trial has an expert-acceptance sidecar;
the evidence supports a planning/controlled-contact pivot rather than bulk
teleoperation collection.
