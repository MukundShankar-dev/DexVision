# Status

**Active checkpoint:** R2 — Pick task spec, evaluator and baselines (see `docs/PLAN.md`)
**State:** implemented and tests pass. Awaiting owner sign-off before R3.
**Last completed:** R1 — Honest simulator core (owner approved 2026-10-20)
**Legacy pipeline:** git tag `legacy-v1`

## Pick task (`configs/tasks/pick.yaml`)

- **Scene.** One object per episode, chosen by the seed from all six, inside an
  8 × 12 cm region with any yaw. The hand starts within ±2 cm of neutral.
- **Horizon.** 8 s (160 steps).
- **Success**, all measured:
  - the object is at least 5 cm above its start height;
  - the thumb plus at least one other finger touch it;
  - nothing but the hand touches it;
  - this holds for 0.5 s.
- **Seeds.** train [0, 100k), val [100k, 110k), test [200k, 210k), with a
  budget of 3 test runs per policy.

## Baselines and gates

| Skill | Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|---|
| pick | val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| pick | val | random | 0 / 100 | [0.0%, 3.7%] | 95 timeouts; 5 objects knocked off the table |

**Pick gate:** not set yet. It is written here in R3, after the expert is
measured and before any training.

## Test ledger

`results/test_ledger.jsonl`. It does not exist yet, so no test-split runs
have happened.

## Open questions for the owner

- **Finger target rate.** Nothing limits how fast finger targets change. The
  random baseline's fingers flick objects at up to about 1.3 m/s, which is
  probably faster than a real Shadow Hand. Should a per-step target rate
  limit go into the action interface before R3?

## Open caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` observations are simulator ground truth: privileged, state
  track only.
- The pick scene has one object and no clutter.
