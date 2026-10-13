# State-grounded push and press (5.5)

Checkpoint 5.5 only. Automated implementation and frozen evaluations finished
October 13, 2026. **Press passes every numerical state gate; push fails state
qualification.** The owner subsequently confirmed "Press passes" and requested
push repair while explicitly keeping 5.5 incomplete. Acceptance is separately
bound in `level5_press_acceptance.json`; the historical automated receipt is
unchanged. The V1 results below remain preserved negative/positive evidence.
No perception, registry admission or subsequent checkpoint is claimed.
Dial is explicitly absent: the frozen Level 4 manifest does not promote it.

## Method and prior-stage adjustments

The pick and place results and their repair reports were reviewed before this
work, together with `level5_training_guidance.md` and the 5.2A/bundle contracts.
Their frame calibration, stale workspace metadata, accumulating unsupported
action channels and post-integration physical-audit findings inform the checks
here. Their task-specific geometric and collision-preview decoders are not
reused. Immutable releases, original 5.0 configs and completed results remain
unchanged. No human control, additional demonstrations or corrective data enter
these runs.

Separate SHA-256-locked recipes are `configs/level5/push_policy_v1.yaml` and
`configs/level5/press_policy_v1.yaml`. Both consume checksum-verified Level 4
training/validation contact-through-retract intervals. Every source channel is
audited across the whole interval, with historical phase masks disabled for
that audit. Source safety preflight binds the recorded runtime workspace
`[-0.18,-0.18,0.01]` to `[0.22,0.18,0.24]` m, original actuator limits and
action cadence. No frozen qualification or jerk gate is relaxed.

Push has 96 training episodes / 10,545 frames and 32 validation episodes /
2,963 frames. Both signed planar channels vary: training X spans approximately
-4.081 to +2.265 mm/sample and Y -3.288 to +4.622 mm/sample. Height, rotation
and all joint commands are exactly invariant in this interval. The MLP learns
signed X/Y deltas at fixed 5 mm/component scale; it does not query an expert or
receive engineered planar directions. Invariance is contact-phase evidence,
not qualification of an approach controller, reapproach or changing posture.

Press has 96 training episodes / 1,744 frames and 32 validation episodes /
592 frames. The fixture's XML slide axis is **world +X**. Its contact-through-
retract command varies only along X, from -5 to +21 mm/sample. The model learns
that signed delta within the original 25 mm cap; Y/Z, orientation and joints
hold only after the independent interval audit. Fixture depth is measured
displacement from its known entry rest pose along the pinned slide axis. Tests
verify equality with the physical fixture joint. A causal latch records the
three-sample depth/state dwell and remains set through release; no hidden
expert phase or future success is an input. Release at at most 2 mm depth
without contact for three samples is an additional terminal requirement.
Maximum fixture depth is bounded by the 22 mm XML travel plus 0.5 mm soft-limit
tolerance; overshoot and wrong contacts remain failures.

Features use entity/target-relative geometry, calibrated command/measured
tracking lag, command height, measured entity motion, rotation, contact/support
and family state. Position and linear velocity divisors are fixed 0.1 m and
0.1 m/s; fixture depth uses 0.01 m. Validation fits no statistics. The measured
forearm origin is converted through the model's pinned weld transform. The
first validation entry's offline/live transformed features match exactly for
each skill (maximum absolute difference 0).

The draft source preflight found that a 0.12 m named-hand-site entry envelope
rejected legitimate fingertip-contact starts (measured distances 0.143889 m
push and 0.133055 m press). Before any training, the final recipe froze a
0.18 m envelope for that named site. The draft recipe/cache receipts remain in
ignored outputs. This changes an entry-domain definition, not workspace,
actuator, rate, board-margin or qualification safety limits. Push requires the
selected supported object and compatible family; its multi-contact retention
heuristic is not used to reject a supported push as a grasp.

Each seed 5001/5002/5003 executes all 200 epochs with the existing two-layer
128-unit ReLU MLP, Adam 0.001 and batch 256. Offline-loss early stopping is
disabled. Sampling balances cell, episode, present causal phase and frame.
Every epoch-20/80/200 and best-offline candidate receives the complete original
closed-loop validation matrix; safety-first selection chooses each seed
independently. Last epoch and selected epoch are distinct artifacts.

Scripted source actions only establish the exact pre-contact entry. Their
checksum-verified prefix is disclosed and excluded from policy metrics and
jerk. No source continuation or expert query is allowed after entry. Policies
run ordinary physics without object pose/orientation writes, hidden reset or
retry. Zero network output holds all prior commands. All raw action violations,
precondition failures and timeouts stay in the denominator.

Success/failure metrics are recomputed from copied post-integration MuJoCo
states, preserving live dynamics. Every attempt's saved qpos/qvel is then
reconstructed independently on a separate simulator without integration,
including failures. Push needs actual object-zone placement, support, tilt,
board containment, speed and five-sample settling. Press needs physical pressed
dwell followed by release. Wrong contact/displacement/activation, contact loss,
overshoot, safety reasons and full trajectories remain visible. Reports retain
size, mass, friction, start, direction, goal, cell, session and boundary data.
`object_size_m` retains MuJoCo `geom.size` parameters (box half extents or
cylinder radius/half height); friction lists retain sliding/torsional/rolling
coefficients. They are reported diagnostics, not fitted policy inputs.

Validation uses 30 unique starts / 120 attempts per candidate and skill.
Selected push test uses 32 unique starts / 128 attempts; press uses 30 / 120.
Each reset is independently repeated under four historical render labels that
have identical state observations and physics; this is not RGB robustness.
These families were exposed in earlier project stages. An exclusive run-local
test marker prevents repeating a held-out test under another output directory.
Weights, schema, recipe, calibration, scales, decoder, cadence, source/code pins
and selection evidence travel together in each bundle.

## Separate Level 2 comparison

The historical Level 3 policies trained on Level 2 data use seed **20260903**;
they are not the new three-seed Level 4 experiment. Evidence is preserved in
`outputs/level3/feasibility_v1/summary.json`, with source release manifest
`datasets/dexvision_level2_v1_manifest.json`. Historical best-validation button
epoch 17 achieved 0/84 frozen scenarios with 84 joint-target limit violations;
push epoch 97 achieved 0/30 with 60 joint-target limit violations. Held-out
success was zero for each. These are historical, unpaired matrices and schemas,
not rerun Level 4 transfer tests. Level 2 has no recorded session ids, so no
cross-session provenance is invented. No legacy data enter these new policies.

## Validation-only push diagnosis

At the first action of validation episode `level45b_000663`, object-to-zone
error is approximately `[-0.0281694, +0.0596794]` m. The source expert delta
is `[-0.000426854, +0.000904321]` m. The epoch-20 policies for seeds
5001/5002/5003 instead predict Y deltas of `-0.000603741`, `-0.000788345`
and `-0.000349263` m. Each already chooses the wrong Y direction before
feedback accumulation. Exact offline/live entry parity rules out the earlier
body-origin interface mismatch at this entry; longer training alone did not
qualify this unconstrained world-XY representation. This is a validation-only
diagnostic, saved in
`outputs/level5/contact_55_v1/push_validation_diagnostic.json`, not a new
decoder, dataset amendment, selection criterion or test-driven repair.

## Final automated results

All six runs completed 200 epochs: 8,400 optimizer updates per push seed and
1,400 per press seed. Every retained candidate received full validation:
1,440 push attempts and 1,320 press attempts. Selection used validation only,
followed by one held-out matrix per selected seed. No held-out matrix was
repeated or used to repair/tune V1. All 2,760 validation and 744 held-out
attempts, including failures, pass independent saved-state metric reconstruction.

| Skill | Seed | Selected epoch | Selected validation successes | Held-out successes | Worst held-out family | Workspace violations | Wrong-contact events | Mean jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Push | 5001 | 20 | 28/120 | 40/128 (31.25%) | 0% | 68 | 16 | 0.000527148 |
| Push | 5002 | 200 | 4/120 | 56/128 (43.75%) | 0% | 12 | 16 | 0.000533581 |
| Push | 5003 | 200 | 0/120 | 44/128 (34.38%) | 0% | 56 | 20 | 0.000703499 |
| Press | 5001 | 200 | 120/120 | 120/120 (100%) | 100% | 0 | 0 | 0.003134890 |
| Press | 5002 | 189 | 120/120 | 120/120 (100%) | 100% | 0 | 0 | 0.003181001 |
| Press | 5003 | 199 | 120/120 | 120/120 (100%) | 100% | 0 | 0 | 0.003208788 |

Every push seed fails the unchanged 80% aggregate / 65% worst-family and zero
workspace/wrong-contact gates. Flat-puck test success is zero for every seed;
the counted successes are cuboids. Raw command violations remain failures
despite bounded application. Invalid actions, joint-target limit violations,
object-board exits and wrong-button activations are zero in the held-out
reports. Contact losses, stalls/timeouts and failures remain saved. Low jerk
and passing physical consistency audits do not qualify these failed controllers.
Push is explicitly **failed**, and its conditional viewer is not triggered.

All press seeds pass 90% aggregate / 75% worst-family success, the unchanged
0.0041666666666666675 mean-jerk cap, and every safety/physical check. Epoch-20
press candidates achieved 100% validation success but failed jerk; they remain
negative evidence. The selected later models include actual pressing and
release. Their final released depth is near zero: terminal success means the
earlier physical pressed dwell followed by release, not a claim that the spring
button stays pressed at the endpoint. The reported parent depth-shortfall error
at that released endpoint remains visible and is not a failed-dwell count.
Press has a **numerical state pass with subsequent owner acceptance** and stays
experimental for default registry purposes; no vision/deployment admission.

Read-back verification checks all artifact indexes, exact selected candidate
tensors/manifests, six full budgets, source/config/code pins, complete matrices,
one-time markers and every independent physical audit. Evidence and hashes are
in [`level5_contact_verification.json`](level5_contact_verification.json).
Both skills' selected weights and full trajectories remain in local ignored
`outputs/level5/{push,press}_55_v1/`; they are not a published model release.

Focused tests: **20 passed** (including ten new push/press tests). Ruff passes.
Final full regression: **900 passed, one platform-dependent offscreen skip** in
793.08 seconds. Original 5.0 checksums, Level 4 archive hash, LFS attribute and
release integrity/publication-readiness checks pass. The interview PDF alone
was added to `.gitignore` at the owner's request; its bytes were not modified.
Datasets and completed checkpoint sections are preserved. Last Completed
remains 5.4A, Next Target remains 5.5, and no 5.6 work was started.

## Commands

Changed files (completed checkpoint sections and original dataset/config locks
are preserved):

```text
.gitignore
configs/level5/push_policy_v1.yaml
configs/level5/push_policy_v1.sha256
configs/level5/press_policy_v1.yaml
configs/level5/press_policy_v1.sha256
dexvision/learning/contact_policy.py
dexvision/evaluation/contact_policy.py
dexvision/apps/train_skill.py
dexvision/apps/evaluate_skill.py
tests/test_push_skill_rollout.py
tests/test_press_skill_rollout.py
tests/test_level5_learning_plan.py
docs/CURRENT_STATUS.md
docs/progress_level_5.md
docs/module_contracts.md
docs/level5_contact_results.md
docs/level5_contact_verification.json
```

From the repository root, use the dexvision environment. Existing run and
evaluation directories are write-once. For reproduction supply new training
`--output-dir` and cache `--cache-dir` paths, then use those run paths below.
Repeat each training/selection/test command for seeds 5001, 5002 and 5003.
The session bounded CPU threads with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`;
on Windows set the equivalent environment variables in the shell.

```bash
conda activate dexvision
python -m dexvision.apps.verify_dataset_release --release-dir datasets/level4-v1 --require-ready
shasum -a 256 -c configs/level5/SHA256SUMS
python -m dexvision.apps.train_skill --config configs/level5/push_policy_v1.yaml --seed 5001 --release-root . --dry-run
python -m dexvision.apps.train_skill --config configs/level5/press_policy_v1.yaml --seed 5001 --release-root . --dry-run
python -m dexvision.apps.train_skill --config configs/level5/push_policy_v1.yaml --seed 5001 --release-root .
python -m dexvision.apps.train_skill --config configs/level5/press_policy_v1.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v1.yaml --select-candidates outputs/level5/push_55_v1/seed_5001 --split validation --track state --release-root . --output-dir outputs/level5/push_55_v1/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/press_policy_v1.yaml --select-candidates outputs/level5/press_55_v1/seed_5001 --split validation --track state --release-root . --output-dir outputs/level5/press_55_v1/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v1.yaml --checkpoint outputs/level5/push_55_v1/seed_5001/selected.pt --split test --track state --release-root . --output-dir outputs/level5/push_55_v1/test_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/press_policy_v1.yaml --checkpoint outputs/level5/press_55_v1/seed_5001/selected.pt --split test --track state --release-root . --output-dir outputs/level5/press_55_v1/test_5001
conda run -n dexvision pytest -q tests/test_push_skill_rollout.py tests/test_press_skill_rollout.py tests/test_level5_learning_plan.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

## Conditional owner viewer verification

**Historical prescribed press review, now owner accepted.** All press seeds
pass numerical and physical gates. Push V1 fails numerical gates, so its viewer
correctly refuses qualification review.
On macOS run the press saved-state review command for passing seed 5001;
use `python` instead of `mjpython` elsewhere. Review includes a saved success
from every held-out cell. It does not rerun or consume another held-out test.

```bash
conda activate dexvision
mjpython -m dexvision.apps.evaluate_skill --config configs/level5/press_policy_v1.yaml --checkpoint outputs/level5/press_55_v1/seed_5001/selected.pt --track state --viewer --scenario-set manual_stratified
```

Pass: every selected push visibly moves the correct supported object into its
target zone and settles on the board; every press contacts the correct fixture,
achieves the recorded physical depth/dwell, retracts and releases; visible
contacts and terminal states agree with reports. Fail: wrong contact, stalled
object despite hand arrival, leaving the board, unstable endpoint, excessive
depth, failure to release, unexplained pose jump/penetration, report mismatch,
crash or missing stratum. Closing early is incomplete. Owner confirmation is
required; numerical success or opening a viewer cannot complete checkpoint 5.5.
