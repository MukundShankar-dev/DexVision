# Status

**Active checkpoint:** R4 — Pick data generation (see `docs/PLAN.md`)
**State:** in progress.
**Last completed:** R3 — Pick scripted expert and gate (owner approved 2026-10-20)
**Legacy pipeline:** git tag `legacy-v1`

## Pick task (`configs/tasks/pick.yaml`)

- **Scene.** One object per episode, chosen by the seed from all six, inside an
  8 × 12 cm region with any yaw. The hand starts within ±2 cm of neutral.
- **Success**, all measured:
  - the object is at least 5 cm up;
  - the thumb plus at least one other finger touch it;
  - nothing but the hand touches it;
  - this holds for 0.5 s within 8 s.
- **Action limits.** Hand speed is capped at 0.2 m/s and 1 rad/s; actuator
  targets change by at most 3 rad/s per joint.

## Pick expert (`dexvision/policies/pick_expert.py`, `configs/experts/pick.yaml`)

- **Inputs.** It sees exactly the policy's declared observation, which a test
  enforces, and nothing else: no contacts, no simulator access, no lookahead.
- **Parameters.** One set for all six objects.
- **Grasp.** Designed from measured fingertip kinematics: the thumb tucks
  behind the object, and the index, middle and ring fingers curl down in
  front.
- **Tuning history.** All tuning used train seeds.
  - v1, a fist-like closing posture: 8/60.
  - v2, the opposition pinch: 60/60 on train seeds 0–59, then 300/300 on
    train seeds 0–299.
  - No val-based changes.

## Baselines and gates

| Skill | Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|---|
| pick | val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| pick | val | random | 0 / 100 | [0.0%, 3.7%] | 99 timeouts, 1 knocked off the table |
| pick | val | expert | 199 / 200 | [97.2%, 99.9%] | Every object ≥ 86% lower bound. The 1 failure is a light puck that slipped after 3.5 cm (seed 100183) |

**Pick gate (owner approved 2026-10-20, written before any training).**
A learned pick policy passes only if its single test run of 200 episodes
meets both conditions:
1. the Wilson 95% lower bound on overall success is at least 80%, which
   needs at least 170/200;
2. every object's success rate is at least 60%.

Checkpoint selection uses val success only. The MLP-BC baseline is reported
against the same gate.

## Test ledger

`results/test_ledger.jsonl`. It does not exist yet, so no test-split runs
have happened.

## Open caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` observations are simulator ground truth: privileged, state
  track only.
- Pick v1 is easy for a privileged expert: one isolated object and a small
  start region. Harder variants (clutter, a wider region) can come after a
  learned policy passes.
