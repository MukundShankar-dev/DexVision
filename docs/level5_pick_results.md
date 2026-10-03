# State-grounded pick policy (5.3)

Status: **checkpoint complete September 17, 2026, with a negative qualification
result**. All three selected policies fail held-out numerical gates and remain
**failed**. Automated checks pass. No viewer acceptance is required or claimed,
and no pick policy is qualified for deployment. Checkpoint 5.4 was not started.

## Results

Each seed has four complete candidate validation matrices (480 attempts), then
one 216-attempt held-out test of its validation-selected weights. Across the
three seeds this is 1,440 validation and 648 test attempts. The four render
labels repeat each reset independently without changing state or physics.

| Seed | Selected epoch | Validation successes | Test successes | Worst test family | Test drop rate | Test safety violations |
|---|---:|---:|---:|---:|---:|---:|
| 5001 | 200 | 120/120 | 136/216 (62.96%) | 38.89% | 20.37% | 36 |
| 5002 | 80 | 116/120 | 120/216 (55.56%) | 38.89% | 24.07% | 48 |
| 5003 | 20 | 116/120 | 140/216 (64.81%) | 44.44% | 16.67% | 40 |

Seeds 5001 and 5002 pass their selected validation gates. Seed 5003's safest
best-performing candidate still has four validation joint-limit violations;
it is retained as failed, not removed or replaced by a better seed. All test
seeds fail aggregate success (required 75%), worst-family success (60%), drops
(at most 10%) and zero safety violations. Seeds 5001 and 5002 also fail jerk:
0.00321713 and 0.00354662 exceed the frozen 0.00312751 threshold. Seed 5003 passes
jerk at 0.00296724. Test safety counts are 24/24/32 actuator target-limit
violations and 12/24/8 object-board escapes. There are no invalid/non-finite
predictions or hand-command workspace violations. Drop counts can overlap
another terminal failure; they are not restricted to the terminal reason label.

The held-out tests were not repeated and did not tune the method. These results
show an executable learned pick controller with insufficient contact robustness
and held-out generalization. Strong validation and low offline loss are not
qualification. Any future repair needs a separately frozen validation-only
hypothesis and must disclose that this held-out family/test evidence is now
exposed. No new repair checkpoint or later skill was started here.

[`level5_pick_verification.json`](level5_pick_verification.json) records selected
weight, training-manifest, selection, test-report, recipe and source digests,
per-seed/family metrics, gates and automated verification. Full trajectories,
models and all failures remain under `outputs/level5/pick_53_v1/` and
`outputs/level5/pick_53_v2/`; these working outputs remain Git-ignored.

Automated verification: **12 focused pick tests**, **22 pick/protocol-lock
tests**, and **775 passed / 1 skipped in 581.27 seconds** for the full suite.
Ruff, original frozen config locks, both new recipe locks, release integrity/
publication readiness, selected-weight binding, matrix completeness and artifact
checksums pass. The first regression run exposed a lock-inventory test that did
not yet recognize independent pick recipes; it was extended without rewriting
the historical 5.0 lock, then the full suite passed.

## Scope and frozen method

The independent recipe is `configs/level5/pick_policy_v2.yaml`, with its own
`.sha256` lock. The historical 5.0 plan/configs, completed checkpoint sections,
release archives and raw demonstrations are unchanged. Only checkpoint 5.3 is
implemented. There is no place policy, perception, registry or executor work.

The transform has 121 input values. The source audit uses checksum-verified
training and validation bytes only:
144 training episodes / 18,401 frames and 48 validation episodes / 6,138 frames.
All are the exact frozen pick intervals from scripted pick/place recordings.
No corrective data or held-out action targets enter training or selection.
The recorded sequence's causal phase can advance to `transport` before its
expert finishes lifting/holding. Pick therefore derives acquire/lift/stabilize
from current measured height and contact-derived retention. Finger state and
prior applied targets distinguish partially closed from fully closed states.
Closure supervision stays active before the first contact; sequence phase
annotations never suppress it. Retrospective success and expert stage are not
policy inputs.

The feature transform uses object-minus-command geometry, calibrated measured
tracking error, command height, lift above entry height, measured rotation,
velocities and joint positions/velocities, previous applied orientation/finger
targets, object pose/motion/contact/class, and causal entry object orientation.
It converts the measured forearm origin through the model's checksum-verified
weld transform before comparing it with the command origin. Position features
use a fixed 0.1 m divisor; other retained fields use fixed scale 1. No tiny
empirical standard deviation is used, and validation never fits statistics.

Training independently establishes zero planar command and zero wrist/selected
abduction deltas throughout the pick slice. The decoder holds those nine
channels only. Seventeen outputs remain active: vertical motion, three rotation
axes and thirteen finger channels. The retained 26-output scaffold has nine
masked outputs. Vertical and rotation outputs have fixed scale 0.04 and 0.15
relative to the original per-sample limits; fingers retain their original
bounded delta scale. Active channels allow both signs. Zero network output
holds the previous action; inference queries no expert and reads no saved
continuation. These are approach-conditioned pick invariants, not evidence for
fixed-posture reach reuse or unconstrained whole-workspace grasping.

Each seed (5001, 5002, 5003) trains a two-layer 128-unit ReLU MLP for 200 epochs,
72 batches per epoch and 14,400 updates. Adam, learning rate 0.001, batch 256,
uniform cell/episode/frame sampling and group-normalized masked MSE remain
explicitly inherited from the frozen parent. Offline-loss stopping is disabled.
The candidate schedule is epochs 20, 80, 200 plus the best offline epoch.
Phase coverage and per-phase offline errors are reported without additional
phase balancing in this first version. Full validation, not loss or the final
epoch, selects each seed independently using the unchanged safety-first ranking.

## Preserved v1 interface failure and v2 correction

V1 completed all three 200-epoch training runs. Validation stopped after the
first two complete candidate matrices per seed and partial third candidates
exposed a source-interface conflict. Every saved attempt failed the workspace
check on its first action; no held-out evaluation or selection occurred. There are 1,042 complete attempt results and three partial attempts. Counts
and interruption reasons are in `outputs/level5/pick_53_v1/interrupted_validation.json`.
The v1 recipe, weights and all complete/partial evaluation evidence remain intact.

The generic action-layout metadata declares z >= 0.08 m. However, the immutable
Level 4 `workcell.safe_workspace`, the recorder's `_workcell_pilot_base_config`,
and the task's runtime `_workspace_failure` use z >= 0.01 m. Released pick
commands legitimately descend below 0.08 m: training prior z spans
0.0338259222–0.1480760929 m, and validation spans 0.0338230595–0.1480759685 m.
This is an inherited metadata/runtime disagreement, not evidence about learned
acquisition. The first validation start is z=0.038 m.

The independently locked v2 recipe explicitly binds the **existing recorded
runtime** workspace: min `[-0.18, -0.18, 0.01]`, max `[0.22, 0.18, 0.24]` m.
No collection-time safety limit, actuator limit, rate cap, qualification threshold,
raw-action rejection rule or dataset is relaxed. The original generic layout and
v1 configs are preserved. Jerk still uses the original frozen normalization
ranges so the original expert-derived threshold stays directly comparable.
A new preflight rejects expert/prior targets or entries outside the recipe's
bounds before training/evaluation. V2 repeats the identical three-seed budget,
features, targets and candidate schedule with new provenance-bound bundles.

## Standalone evaluation and physical evidence

Every validation candidate receives 30 unique resets x four condition labels
(120 independent attempts). Each selected seed receives the original single
held-out matrix of 54 unique resets x four labels (216 attempts). The labels
change neither state observations nor physics; they make no RGB/perception
robustness claim. Test-family identities are excluded from model selection.
These families have appeared in earlier project datasets/reach work; this is
skill-specific held-out ownership, not a claim of globally unseen objects.

The evaluator replays only the checksum-verified action prefix strictly before
the frozen start frame. This is disclosed scripted setup and is excluded from
policy metrics/jerk. At the boundary, supported, unheld, fresh selected-object
state and a measured grasp-site distance at most 0.06 m are required. Terminal
dwell is reset. A rejected entry remains in the denominator; no replacement
frame, hidden reset, expert rescue or retry is permitted. The policy never
receives the prefix or later source actions. This does not establish composition
with the learned reach policy.

Success requires the original frozen physical pick height/dwell metric **and**
an independent stricter stability check: at least two distinct hand contact
bodies, the correct held object at least 0.04 m above its original support
height, no support contact, speed at most 0.02 m/s, and ten consecutive samples.
The learned feature's height reference is causal entry height; terminal metrics
retain the original source support-height reference. Loss of retention for more
than five samples after acquisition is a drop failure. Contact-derived held
state is a simulator heuristic, not a grasp-force or real-robot guarantee.

Every attempt saves qpos/qvel, physical world states, timestamps, raw/predicted,
commanded and applied actions, prior actions, safety masks/reasons, physical
phase outcomes and result. Reports include acquisition, lift, stable dwell,
drops, wrong contact, jerk, condition/cell/family/instance/session and exact
size/mass/friction/approach breakdowns. Raw workspace/actuator-limit violations
remain failures after clipping. Joint-limit counters mean actuator target-limit
violations, as in the parent evaluator; contact penetration and hand/table
contacts remain visible in physical metrics. No zero soft-joint overshoot claim
is made. Background objects are intentionally parked outside the board by the
released trial setup; the selected object is checked against the board limits.

Selected weights must match the validation-selected candidate. An exclusive
run-local test reservation prevents repeating a failed or interrupted test
under a different output directory. Recipe, source, transform/scales, decoder,
control cadence, schema, execution-code and weight digests accompany the bundle.

## Files changed

```text
configs/level5/pick_policy_v1.yaml
configs/level5/pick_policy_v1.sha256
configs/level5/pick_policy_v2.yaml
configs/level5/pick_policy_v2.sha256
dexvision/learning/pick_policy.py
dexvision/learning/phase_losses.py
dexvision/evaluation/pick_policy.py
dexvision/apps/train_skill.py
dexvision/apps/evaluate_skill.py
tests/test_pick_skill_training.py
tests/test_pick_skill_rollout.py
tests/test_level5_learning_plan.py
docs/level5_pick_results.md
docs/level5_pick_verification.json
docs/CURRENT_STATUS.md
docs/progress_level_5.md
docs/module_contracts.md
```

## Commands

Run from the repository root. Use new output/cache directories for reproduction;
existing results and failed attempts must not be overwritten. The session's
verified cache is `outputs/level5/pick_53_v2/cache`. All data reads verify release
hashes even when `--release-root .` uses locally restored payload files.

```bash
conda activate dexvision
python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v2.yaml --seed 5001 --dry-run
python -m dexvision.apps.verify_dataset_release --release-dir datasets/level4-v1 --require-ready
shasum -a 256 -c configs/level5/SHA256SUMS
```

Run each of these for seeds 5001, 5002 and 5003, substituting the seed in all
paths. On macOS/Linux the session used `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
prefixes to bound CPU threads. On Windows set the equivalent environment
variables in the shell; they do not change the recipe.

```bash
python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v2.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/pick_policy_v2.yaml --select-candidates outputs/level5/pick_53_v2/seed_5001 --split validation --track state --release-root . --output-dir outputs/level5/pick_53_v2/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/pick_policy_v2.yaml --checkpoint outputs/level5/pick_53_v2/seed_5001/selected.pt --split test --track state --release-root . --output-dir outputs/level5/pick_53_v2/test_5001
conda run -n dexvision pytest -q tests/test_pick_skill_training.py tests/test_pick_skill_rollout.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

For reproduction, pass a new `--cache-dir` and `--output-dir` to training, and
use the resulting run path in evaluation. `--resume` is only for an interrupted
run with matching source/config/environment/tensor digests. Each new independent
training run gets its own one-time test reservation; do not use a new run to
retune against previously observed test outcomes.

## Manual acceptance

**None is required for these recorded results:** all three selected policies
fail numerical gates. No viewer was opened and no owner acceptance is inferred.
The guarded command/criteria below document the acceptance path; the viewer
correctly refuses these failed reports.

Required only if the numerical gates pass. On macOS, for each numerically
passing seed use the following command (substitute its seed); elsewhere use
`python` instead of `mjpython`:

```bash
conda activate dexvision
mjpython -m dexvision.apps.evaluate_skill --config configs/level5/pick_policy_v2.yaml --checkpoint outputs/level5/pick_53_v2/seed_5001/selected.pt --track state --viewer --scenario-set manual_stratified
```

The viewer replays saved policy states and does not repeat held-out evaluation.
It shows each object-family/boundary stratum and leaves acceptance pending.
Pass only when each family acquires the correct object, visibly clears lift
height, and holds stably without table penetration, hidden reset, unintended
collision, or disagreement with the saved result. Fail on any mismatch, drop,
erratic movement, crash, missing family, or incomplete replay. Closing early is
incomplete. Only the owner's explicit confirmation can close a required manual
gate; numerical success alone cannot mark the checkpoint complete.
