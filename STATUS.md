# Status

**Active checkpoint:** R7 — push (R2–R6 repeated; see `docs/PLAN.md`)
**State:** task spec, expert and val baselines done. Awaiting owner sign-off
on the push gate before data generation and training.
**Last completed:** R7 reach — both methods pass (owner approved 2026-10-21)
**Legacy pipeline:** git tag `legacy-v1`

## Shared protocol (every skill)

- **Method:** `configs/learning/method.yaml`, unchanged across skills.
  Diffusion and MLP-BC, seed 0, 60k steps, checkpoints every 10k, trained on
  the desktop RTX 5070 Ti.
- **Selection:** the most val successes on the first 200 val seeds, ties to
  the later step. Then one ledgered 200-episode test run per method.
- **Platform:** all results are evaluated on the desktop (Linux x86-64),
  because MuJoCo is not bit-identical across arm64 and x86-64. Provenance
  records the platform.

## Pick (done, R6)

- Six objects; lift ≥ 5 cm with a thumb-opposition hold for 0.5 s. Expert
  v3. Data: 985/1000 kept. Val: zero 0/100, random 0/100, expert 200/200.
- **Test:** MLP-BC **200/200**, Diffusion **200/200** (gate passed). Hiding
  the object pose drops val success to 0 and 12 of 200.
- Ledger: the pick MLP entry says `dirty` only because of uncommitted
  `results/` files; that flag now ignores `results/`.

## Reach (done)

- Bring the measured grasp point within 1.5 cm of a goal for 0.5 s; any hand
  collision fails. Val: zero 0/100, random 0/100, expert 200/200.
- **Test:** MLP-BC **200/200**, Diffusion **200/200** (gate passed). With a
  fake goal (the training mean), success drops to 1 and 2 of 200.

## Push (`configs/tasks/push.yaml`, `dexvision/policies/push_expert.py`)

- **Task:** slide one of four objects (both blocks, both cylinders; no
  pucks) 5–8 cm toward a goal within ±45° of +x. Success (measured on the
  object): within 2 cm of the goal and still upright on the table for 0.5 s,
  within 10 s. Hand contact with the trays, wall fixture or button fails the
  episode.
- **Fan narrowed during design** from ±90° to ±60° to ±45°. The forearm
  trails about 30 cm behind the hand and hit the wall fixture and button past
  about 105° of hand yaw. Pushes also need steering room.
- **Expert:** fingers bent 90° at the knuckle form a wall. It approaches high
  behind the object, descends, and pushes along object → goal, re-aimed every
  step. It slides back before rising to re-approach (rising while touching
  toppled tall cylinders) and never turns the hand past 90°. Train seeds
  0–299: 300/300 clean, 293/300 with DART noise.

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| val | random | 6 / 100 | [2.8%, 12.5%] | genuine: objects moved 5–7 cm (see note) |
| val | expert | 200 / 200 | [98.1%, 100%] | every object 100% |
| val | expert + DART noise | 196 / 200 | [95.0%, 99.2%] | robustness, not a policy |

The random policy's successes are real pushes (checked). Chance is higher
here because the hand starts behind the object and every goal lies away from
it.

**Push gate (proposed, written before any push training; needs owner
sign-off).** A learned push policy passes only if its single 200-episode test
run has a Wilson 95% lower bound ≥ 80% (≥ 170/200) and every object ≥ 60%.
Checkpoints are selected on val only.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- Pick's `object_*` observations are privileged simulator state. Reach has
  none.
- Reach is easy: free space and a smooth single-strategy expert. It checks
  that the pipeline transfers unchanged to a second skill.
