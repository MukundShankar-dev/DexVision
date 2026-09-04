# DexVision Project and Teleoperation Reassessment

Date: September 4, 2026

Purpose: give the team enough project context and pilot evidence to decide how
Level 4 demonstrations should be produced. This is a discussion brief, not a
completed architecture decision.

## What DexVision is trying to build

DexVision is a simulated dexterous-hand project. A camera observes a person's
hand, MediaPipe estimates hand motion, and that motion controls a Shadow Hand
inside MuJoCo. The system records demonstrations, trains reusable manipulation
skills in PyTorch, and evaluates those skills under fixed safety and success
rules.

The eventual language model is a planner, not a motor controller. A request
such as "clear the workspace" or "prepare this part for inspection" should be
turned into typed calls to already-qualified skills: reach an object, pick it
up, place it, push it to a zone, or press a button. A deterministic supervisor
checks the world state, enforces limits, runs the skill, and verifies whether
the requested result actually happened.

The first complete target is a bounded tabletop workcell, not a general home
robot. It contains six rigid objects, return bins, an inspection pad, setup
slots, and a Start button. The final demonstrations are workspace clearing,
inspection-station operation, and workspace setup.

## Work completed so far

- **Level 1 - Teleoperation:** OpenCV and MediaPipe tracking control the
  simulated hand's base, wrist orientation, and fingers. The team iteratively
  fixed finger coupling, depth control, calibration, smoothing, tracking loss,
  and orientation mapping. Free-space hand control works.
- **Level 2 - Data infrastructure:** Demonstrations can be recorded, replayed,
  validated, quality-filtered, summarized, and packaged immutably. The first
  release contains 55 reach, 55 button, and 101 push successes, plus three
  evaluated retargeting methods.
- **Level 3 - Learning feasibility:** A reproducible behavior-cloning pipeline,
  validation-only checkpoint selection, and frozen MuJoCo evaluation all work.
  However, none of the learned policies passed the closed-loop gates. Offline
  error did not predict safe rollout success, and action coupling produced
  immediate workspace and joint-limit failures. This justified collecting a
  better, session-aware dataset rather than merely training a larger model.
- **Level 4.0-4.2 - New workcell foundation:** The team froze a 250-350 episode,
  four-session plan; built the resettable workcell and typed world state; and
  implemented append-only recording with requested, commanded, and applied
  actions, safety reasons, causal phases, optional RGB, and replayable metadata.

Level 4.3 is the current checkpoint. Its purpose is to prove that every required
skill can actually be demonstrated and replayed before collecting hundreds of
episodes.

## Why the team is reassessing now

The pilot first exposed genuine implementation defects: the wrong hand anchor,
an unsuitable orientation mapping, a 17-times-too-slow physics loop, unclear
target cues, and absolute position control that wandered through clutter. Those
issues were corrected or replaced. A centered velocity-style controller, safe
transit height, target corridor, and clear visual cues made reach usable.

The same corrections did not make contact-rich tasks usable:

| Trial | Measured result | Practical reading |
|---|---|---|
| Reach | Success at 1.62 cm terminal error and about 0.21 cm maximum scene disturbance | Free-space positioning is viable. |
| Pick/place | No grasp; target and neighboring objects moved about 4.9 cm and 4.5 cm | Approach is possible, but controlled grasp/contact is not. |
| Push | The selected block did not move and remained 28.8 cm from its target | The mapping cannot maintain the required contact path. |
| Button | Zero press depth; the hand never reached a viable pre-contact pose | Collision-free approach must be solved before pressing can be judged. |

These are one clean trial per action, so they are directional evidence rather
than final statistics. They are still enough to stop bulk collection: three of
four required action groups cannot currently produce expert demonstrations.

## Decision now required

There are two coherent directions.

**Keep monocular human imitation as the central research question.** Pause
Level 4 and redesign the input stack around a clearer 6-DoF device, VR
controllers, depth tracking, or waypoint controls. Grasp closure and contact
feedback would still require special treatment. This preserves a pure human-
demonstration story, but adds hardware, calibration, and schedule risk.

**Keep the reliable learned-workcell goal.** Use a hybrid expert pipeline:
classical planning or inverse kinematics for collision-free approaches, then
short deterministic controllers for grasp/lift, straight push, and
press/retract. Keep webcam teleoperation for target selection, free-space
corrections, and debugging. Learned policies imitate successful planned or
shared-autonomy trajectories and may later learn bounded residual behavior.

The second direction is the better fit for the stated project. It changes how
expert data are generated, but it does not abandon imitation learning, the
typed skill API, safety supervision, visual grounding, or future language-model
orchestration. Every trajectory must honestly record whether it came from a
human, planner, script, policy, or shared-autonomy combination.

## Recommended next move

Do not start the Level 4.4 data haul. Create a versioned Level 4 plan revision
that permits planned, scripted, and shared-autonomy demonstration sources.
Then run a small architecture pilot in increasing difficulty:

1. Collision-free planned reach to a pre-contact pose.
2. Planned reach plus deterministic button press and retract.
3. Planned reach plus a constrained straight-line push.
4. Planned reach plus scripted grasp closure and stable lift.
5. Complete pick, transport, place, release, and settle.

Each stage must visibly and recomputably succeed, replay cleanly, produce zero
workspace or joint-limit violations, and avoid unexplained movement of other
objects. Stop at the first failing layer. Freeze the 250-350 episode matrix only
after every required skill is demonstrably recordable.

## Questions for the team

1. Is the core contribution webcam-based human imitation, or the reliable
   learned skill library and its later orchestration?
2. Are planner-generated and shared-autonomy trajectories acceptable expert
   data when provenance is explicit?
3. Is there appetite for a better 6-DoF input device, or should the project
   remain hardware-light?

Recommended answer: keep the workcell goal, adopt the hybrid expert pipeline,
retain webcam teleoperation for reach and corrections, and revise Level 4.3
before collecting at scale.
