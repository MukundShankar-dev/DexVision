# State-grounded place-held-object policy (5.4)

Status: **checkpoint 5.4 complete with a negative state qualification result**.
The frozen three-seed experiment and one-time held-out evaluations are complete.
All selected policies fail numerical gates and remain **failed**. No manual
viewer acceptance is required or claimed; no place policy is qualified for
deployment.

## Input audit and method

The independent recipe is `configs/level5/place_policy_v1.yaml`, protected by
its adjacent SHA-256 sidecar. It consumes the immutable Level 4 place segments:
144 training episodes / 16,845 frames and 48 validation episodes / 5,285 frames.
The frozen evaluation matrices contain 30 validation starts (120 state attempts)
and 54 separate test starts (216 state attempts) per candidate/selected seed.
The four historical condition labels repeat a reset under identical state and
physics; they do not test RGB or perception robustness. Validation contains
placement-slot targets; held-out test contains all three target types, including
goal regions withheld from place training. Prior project exposure to these
families is not an unseen-family claim.

The source-action audit found bidirectional xyz transport/lowering, wrist
rotation, and thirteen varying finger-release channels. Nine other action
channels remain exactly invariant across the full training interval and are
held only after cache construction rejects any source mismatch. A zero policy
output holds the complete prior command. Every nonzero decoded delta uses the
original action rate cap, and the original actuator/workspace safety checks run
before application. The recipe explicitly binds the Level 4 recorder's runtime
workspace (z minimum 0.01 m). This follows the 5.3 correction for the stale
generic action metadata without changing collection-time bounds.

The causal feature transform includes measured held-object-to-target geometry,
object-to-command grasp offset, target-to-command error, prior command height,
measured object linear/angular motion, hand joint/target state, target type,
retention/support/contact flags and the workcell's causal place phases. Measured
forearm pose is converted into the command frame with the verified model weld
transform, following the 5.2A/5.3 frame repair. Position and linear velocity
use fixed 0.1 m and 0.1 m/s divisors; no validation statistics are fitted.
On the first frozen validation entry, the live pre-action feature vector matches
the saved training transform exactly (maximum absolute difference 0).

Phase counts on train are 13,009 transport, 188 place, 144 release, 1,691 settle
and 1,813 retract frames. Uniform cell/episode sampling alone would rarely
sample release, so the locked place sampler selects a present causal phase
uniformly within each sampled cell/episode. Training uses a 128x128 ReLU MLP,
Adam 0.001, batch 256, 200 full epochs and seeds 5001/5002/5003. Offline-loss
patience is disabled as a stopping rule. The early/late/final and best-offline
candidates all receive the complete validation matrix. Selection is safety
first, then closed-loop success/worst cell/error, never last epoch by default.

The Level 4 source recorder used a declared rotation-only object-orientation
hold during scripted pick/place collection. The evaluator replays this hold
only in the exact source-action prefix before the held-object boundary. Once
the policy takes over, it uses ordinary physics: no object pose writes, expert
actions, saved continuation or hidden reset. The setup prefix is excluded from
policy metrics. Entry rejects missing, multiple, wrong, supported or stale held
objects and remains in the denominator. Current simulator state, not predicted
hand arrival, defines release, target distance, support, speed and stable dwell.
Post-integration physical evidence is recomputed on a copied MuJoCo state, as
learned from the 5.3A terminal-evidence correction. Premature release and bounce
remain failures even if the object later settles. All attempts preserve actions,
states, safety reasons and outcomes for physical review.

Interrupted pre-lock draft runs and their cache remain in ignored
`outputs/level5/place_54_v1/` paths. They are excluded from model selection.
The locked recipe uses `outputs/level5/place_54_v1/locked/` and a new cache;
historical draft files are not overwritten.

## Commands

Run from the repository root. Use the `dexvision` Conda environment; existing
run, validation and test directories are immutable evidence and must not be
reused for a second attempt. On macOS, single-thread BLAS/OpenMP bounds CPU
contention without changing the model recipe.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/place_policy_v1.yaml --seed 5001 --dry-run
for seed in 5001 5002 5003; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/place_policy_v1.yaml --seed "$seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/place_policy_v1.yaml --select-candidates "outputs/level5/place_54_v1/locked/seed_$seed" --split validation --track state --output-dir "outputs/level5/place_54_v1/locked/validation_$seed"
done
for seed in 5001 5002 5003; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/place_policy_v1.yaml --checkpoint "outputs/level5/place_54_v1/locked/seed_$seed/selected.pt" --split test --track state --output-dir "outputs/level5/place_54_v1/locked/test_$seed"
done
conda run --no-capture-output -n dexvision pytest -q tests/test_place_skill_training.py tests/test_place_skill_rollout.py
conda run --no-capture-output -n dexvision ruff check dexvision tests
conda run --no-capture-output -n dexvision pytest -q
cd configs/level5
shasum -a 256 -c place_policy_v1.sha256
```

The held-out test loop is allowed only after all three complete validation
selections are frozen. Each selected checkpoint has one exclusive test attempt;
failed attempts stay visible. Manual viewer review is required only when the
complete numerical report passes every state gate.

## Full validation and held-out results

All three seeds completed 200 epochs / 13,200 updates. Each evaluated four
candidates on the complete 120-attempt validation matrix. Full validation
selected epochs 20, 56 and 20; none passed validation numerical gates. The
single held-out test for each selected checkpoint then covered 54 unique starts
in four independent executions, including receptacle, planar-zone and
placement-slot targets. All 648 attempts and failures are saved; no reset was
filtered, retried or replaced.

| Seed | Selected epoch | Validation successes | Test successes | Worst family | Premature release | Settled among nominal placements | Safety violations | Mean jerk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5001 | 20 | 20/120 | 4/216 | 0% | 55.56% | 50% | 92 | 0.001782 |
| 5002 | 56 | 0/120 | 0/216 | 0% | 48.15% | 0% | 116 | 0.014014 |
| 5003 | 20 | 12/120 | 12/216 | 0% | 42.59% | 60% | 104 | 0.002008 |

The state gate requires at least 75% aggregate and 60% worst-family success,
no more than 10% premature release, at least 90% settling among nominally
placed objects, zero safety violations and mean jerk no greater than 0.010525.
Every seed fails success, release, settling and safety. Seed 5002 also fails
jerk. All three target types are present in the held-out breakdowns; seed 5001
passes only 4 placement-slot attempts, seed 5002 none, and seed 5003 4
placement-slot, 4 planar-zone and 4 receptacle attempts. Family breakdowns,
sessions, instances, cells, individual terminal reasons and exact digests are
in `docs/level5_place_verification.json` and the ignored evaluation reports.

The first closed-loop divergence is premature opening during transport. For
seed 5001's epoch-20 candidate, small negative finger predictions accumulate
before the object nears the goal; 92/120 validation attempts release early.
A validation-only diagnostic held finger commands while measured object-target
distance exceeded 0.025 m. On 30 unique validation resets it improved the
candidate from 5/30 to 19/30 successes, but still recorded two premature
releases, eight bounces and one workspace violation. It remains a **failed
diagnostic**, not a selected bundle, retrained policy or held-out result. A
future separately frozen repair would need to address release gating, the
remaining bounce/safety failures and the source's assisted orientation hold.
The Level 5.4 test outcomes were not used to tune this recipe or diagnostic.

Artifact checks pass for all three seeds: run, validation and test checksum
indexes; exact selected-to-candidate model weights; recipe/source and current
execution-code binding; one reserved test attempt; complete matrices. An
independent saved-state reconstruction using qpos/qvel and MuJoCo forward
confirms that all 16 reported test successes satisfy target distance,
unsupported release, table support and both speed bounds for their final ten
physical states. This audit does not turn the many failed attempts into
successes. The source archive checksum and LFS attribute pass. Focused tests:
18 passed. Full regression: **816 passed, one skipped**. Ruff, the recipe
checksum and `git diff --check` pass. The first full regression run found that
the historical 5.0 lock-inventory test did not recognize a new independent
place recipe; the test was extended without changing the frozen 5.0 lock, and
the second full run passed.

The numerical failures mean the conditional manual viewer gate is **not
triggered**. The guarded viewer refuses failed reports; no owner observation or
manual acceptance is claimed. The policy remains state-only and disabled for
any registry or workcell deployment. All original data/release archives and
the separate failed draft runs are preserved. Level 5.5 has not started.
