# Checkpoint 5.5 push repair

Development started October 13, 2026 after the owner accepted press and required
push to pass. At development start, Last Completed was 5.4A and push manual
acceptance was required; only local commits were authorized. The later owner
acceptance below completes 5.5 and authorizes commit and push. No 5.6 work starts.

## Preserved evidence and diagnosis

Commit `61de6af` preserves the failed push V1, passing press V1, original
verification, and separate owner press acceptance. The accepted press execution
modules, immutable Level 4 source and original Level 5.0 config lock remain
unchanged. V1 test is consumed; it cannot select or tune this repair. Test
families and goals are already exposed, so fresh resets do not constitute
unseen-family evaluation.

The previous pick/place adjustments and training guidance are documented in
`level5_contact_results.md`, `level5_pick_repair.md` and
`level5_place_repair.md`: measured/command calibration, physically recomputed
outcomes, full-budget training, closed-loop selection, preserved failures and
separate fresh confirmation apply here. Their skill-specific engineered controls
are not adopted without push evidence.

Validation-only V1 diagnosis found wrong world-Y direction at the first action
of `level45b_000663`, despite exact offline/live feature parity. All three seeds
predicted negative Y for a positive-Y target error. The repair changes the
coordinate representation before changing model size, optimizer or data volume.

## Independently frozen V2 hypothesis

`configs/level5/push_policy_v2.yaml` and its adjacent SHA freeze the experiment.
The fixed task basis comes from the initial object pose and current static goal,
known entry context also available to execution. It is never inferred from
future actions. Inputs include task-relative object-to-goal, command-to-object,
calibrated command/measured lag, object velocity, height, angular velocity,
contact/support/family and causal physical phases. Fixed physical scales remain
0.1 m and 0.1 m/s. The recipe removes absolute world-direction and target-type
features.

Learned outputs are signed along/lateral displacements, scaled by 0.005 m.
The engineered decoder rotates them into world XY and preserves their direction
when enforcing the unchanged per-world-component limit. Zero output holds.
Height, orientation and joints hold only after the independent whole-interval
source audit. No expert controls or object pose writes occur after entry.
The same 128-by-128 ReLU MLP, Adam 0.001, batch 256, phase/cell/episode-balanced
sampling and full 200-epoch budget remain. All early/late/best-offline/final
candidates receive the complete original validation matrix before safety-first
selection. Every saved physical state is independently reconstructed.

The exact initial-pose basis source audit finds train signed along targets
[-0.005, 0.002] m and lateral numerical roundoff only (less than 2e-17 m).
Negative retraction and lateral output remain representable; no positive-only
progress rule is imposed. The 96 training / 32 validation episodes and
10,545 / 2,963 frames are unchanged. Cached tensors originate from the prior
checksum-verified vectorization, then are checked against original metadata,
source ownership, frame intervals and action bounds. No test tensors supervise
training.

Before training/selection, 32 fresh procedural confirmation resets are frozen:
eight per original test cell, seeds `20555000 + cell_index*100 + repetition`.
Boundary cells retain boundary perturbations. The original sampler and resolved
scripted push settings establish only the exact pre-contact prefix; the first
contact-phase expert command is excluded. Copied-state source feasibility is
setup planning, never policy selection. No reset is filtered and setup failures
remain counted. Four render labels repeat identical state/physics, giving 128
attempts per seed but only 32 unique resets. Confirmation is reserved once per
selected seed, with no retries. All original success, safety, contact, board,
jerk and physical-audit gates remain unchanged.

## Commands

All commands use the `dexvision` Conda environment from the repository root.

```bash
conda activate dexvision
python -m dexvision.apps.train_skill --config configs/level5/push_policy_v2.yaml --dry-run --release-root .
python -m dexvision.apps.train_skill --config configs/level5/push_policy_v2.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v2.yaml --select-candidates outputs/level5/push_55_v2/seed_5001 --split validation --track state --release-root . --output-dir outputs/level5/push_55_v2/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v2.yaml --checkpoint outputs/level5/push_55_v2/seed_5001/selected.pt --track state --split test --output-dir outputs/level5/push_55_v2/confirmation_5001
pytest -q tests/test_push_repair.py tests/test_push_skill_rollout.py tests/test_press_skill_rollout.py tests/test_level5_learning_plan.py
ruff check dexvision tests
pytest -q
```

Repeat training, complete selection and single confirmation for seeds 5002 and
5003. Existing outputs are immutable attempts; reproductions use new paths.
Confirmation freezing uses `freeze_push_confirmation` in
`dexvision/evaluation/push_repair.py`, before any selected weights exist.
The preserved freeze logs include two implementation errors before training:
using a nonexistent row reset field, then passing prefixed actuator names to
an expert that itself prefixes them. Neither queried learned policies or ran
confirmation. Corrected freezing retains every predefined reset.

## Status

Historical initial status: focused tests passed 24 tests before the V3 setup
addition. This section records the V2 experiment; the V3 status and final
qualification/verification receipt below govern the active push repair.
No owner push viewer acceptance is claimed.

## V2 negative result and V3 setup repair

All three V2 training budgets finished (200 epochs, 8,400 updates per seed).
Every candidate received the complete 120-attempt validation matrix. All three
seeds at epochs 20 and 80 succeeded on 104/120 attempts, but four distinct
nominal resets failed wrong-contact/non-target disturbance (16 repeated-label
failures). Later candidate results remain preserved in
`docs/level5_push_repair_v2_failed.json`. No V2 confirmation was executed.

Saved-state reconstruction identified a parked non-target heavy puck with a
transient 0.011772 m vertical fall; target hand contacts, all other parked
objects and fixture depth did not account for the violation. The identical
failure occurred at sample 46 with a zero-output policy. A development-only
120-sample hold of the final source-prefix command allowed ordinary physics to
settle before learned entry. The four affected validation resets then passed
for all three epoch-20 candidates: 12/12 successes and physical audits. Two
training cuboid resets also passed for each seed (6/6). These diagnostics are
not qualification. All attempts, including failed scripts and negative results,
remain local under the versioned output trees.

The separate checksum-locked `push_policy_v3.yaml` overlay retains exact V2
learned transforms, targets, decoder, cadence and full-budget candidate tensors.
`push_setup_v3.yaml` freezes independent source-prefix generation and 32 fresh
confirmation seeds beginning at 20755000. The overlay appends exactly 120 holds
(4.08 simulated seconds) to every validation and fresh confirmation prefix,
without search, live policy consultation, active pushing, object writes or
altered success/safety thresholds. It changes the declared contact-entry setup;
free-space approach and that settling interval remain engineered prerequisites,
not learned skill behaviour. All post-entry controls are the learned V2 policy.

Each V3 selection bundle separately copies the immutable V2 trained artifacts
and binds `entry_overlay.json` to the original source run, overlay/setup
recipes, fresh starts and exact execution-code digests. It runs every candidate
on the complete original validation reset matrix with this fixed setup.
Reports separately hash the original parent reset matrix and the actual
execution prefixes, so the changed entry time is not hidden in the historical
training manifest. Newly selected tensors must equal their original candidate.
An exclusive reservation permits one fresh V3 confirmation per selected seed.
All failures and original gates remain intact. V1 and V2 confirmation families
are exposed; fresh V3 resets make no unseen-family claim.

V3 uses the already completed V2 training budgets rather than retraining
identical targets. Its commands are:

```bash
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v3.yaml --select-candidates outputs/level5/push_55_v2/seed_5001 --split validation --track state --release-root . --output-dir outputs/level5/push_55_v3/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v3.yaml --checkpoint outputs/level5/push_55_v3/seed_5001/selected.pt --track state --split test --output-dir outputs/level5/push_55_v3/confirmation_5001
```

Repeat for 5002 and 5003. `freeze_settled_confirmation` creates the distinct
fresh starts before any V3 selection. Focused suite: 25 passed. Ruff passes.
The pre-overlay full suite passed 904 tests with one skip; final overlay
regression and complete V3 validation are in progress. Push remains unqualified
until all numerical/physical gates and owner manual acceptance pass.

## Historical V3 viewer command (not eligible after failed confirmation)

On macOS, from the repository root:

```bash
conda activate dexvision
mjpython -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v3.yaml --checkpoint outputs/level5/push_55_v3/seed_5001/selected.pt --track state --viewer --scenario-set manual_stratified
```

The guarded command replays one saved success in each of the four fresh
confirmation cells: cuboid/slot B and flat-puck/inspection-pad, each interior
and boundary. It refuses review unless all three seeds' fresh numerical and
physical confirmation reports pass.

Pass only if all four clips show the intended object physically pushed toward
its named goal, remaining upright/supported on the board and visibly settling
in the reported goal region. The hand must contact the intended object, with
no wrong-object/fixture activation or neighbor disturbance. Fail if an object
stalls, tips, leaves the board, disturbs a non-target, or the visible final state
does not match the success report. Missing clips, cancellation or a command
error do not count as a pass. Confirm pass/fail explicitly afterward; neither
this command nor automated metrics mark owner acceptance or 5.5 complete.

The settled-entry zero-output ablation timed out after all 300 samples at
0.065323 m object/goal error, zero safety and a passing physical audit. The
engineered setup/basis alone therefore did not achieve that validation task.

## V3 consumed negative confirmation; separate V4 distance repair

Complete V3 selection picked epochs 80/80/146 after all 11 candidates passed
120/120 validation attempts each. Final regression passed 905 tests with one
skip. Its exclusive fresh confirmation **failed**: 92/128, 96/128 and 68/128
successes, with cuboid rates 43.75%, 50% and 6.25%, versus 100% flat-puck
success for all three seeds. All safety counters remain zero and all 384
independent physical audits pass. The saved timeouts remain counted; V3 is not
qualified, its 20755000-series resets are consumed, and no viewer acceptance
is triggered. Exact bindings are in `level5_push_repair_v3_failed.json`.

A first-action range audit identified only the along-goal distance outside the
cuboid training range for one stalled seed-5002 reset: 0.156822 m versus the
training range 0.026844–0.050935 m. Its learned progress rapidly shrinks toward
zero or becomes negative, while flat-puck confirmation succeeds. This diagnoses
unsupported long-range cuboid extrapolation, rather than a safety-gate failure.
No V1 or V3 confirmation is rerun for development or qualification.

The independent source-only target audit finds 13 training cuboid and 6, 771
training flat-puck frames with positive along distance above 0.05 m. Every
cuboid target is +0.001 m; flat-puck targets are +0.001 m or the first-boundary
+0.002 m, with no negative far-distance targets. Validation's 2, 052 such frames
have +0.001 m targets and no negative targets. Lateral targets are numerical
roundoff only. These are training/validation observations and labels; no failed
confirmation actions supervise the repair.

V4's separately locked recipe keeps the complete V2 state/action transform and
changes only positive along-goal distance to `min(distance, 0.05 m)`. Exact
near-goal distances, negative distances and lateral errors remain available.
Signed learned outputs, geometric rotation, fixed posture, model, optimizer,
sampling, 200-epoch budgets and 17-step cadence stay intact. Physical metrics
use uncapped measured distance and all original thresholds. Saturating one
observation does not force a positive action or turn a distant object into a
success. It removes a distance regime with invariant source instructions.

Before any V4 training, freeze the original 30 validation starts plus eight
fresh all-cell selection starts (two per test cell, 20955000-series), giving
38 unique resets and 152 repeated-condition attempts per candidate. These
families/goals are exposed by V3; fresh selection is development evidence, not
unseen-family qualification. Separately freeze 32 distinct 21155000-series
confirmation starts, reserved once after method/weight selection. Both matrices
retain the fixed 120-sample setup hold. No reset is filtered or replaced;
failed setup remains counted. Original supervised split ownership, source
archives and V1–V3 weights/recipes/results remain unchanged.

V4 freezes the expanded validation membership/digest in each training manifest,
separately retaining the original parent validation digest. Fresh selection
prefixes have no supervised policy targets. Cache supervision still contains
only the original 96 training and 32 source-validation episodes.

```bash
python -m dexvision.apps.train_skill --config configs/level5/push_policy_v4.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v4.yaml --select-candidates outputs/level5/push_55_v4/seed_5001 --split validation --track state --output-dir outputs/level5/push_55_v4/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v4.yaml --checkpoint outputs/level5/push_55_v4/seed_5001/selected.pt --track state --split test --output-dir outputs/level5/push_55_v4/confirmation_5001
```

Repeat for 5002 and 5003. `freeze_entry_matrices` in
`dexvision/evaluation/push_distance.py` must run before training. V4 automated
checks and training are pending; checkpoint 5.5 remains incomplete and owner
push verification is not yet triggered.

## V4 negative selection and independently frozen V5 entry geometry

All V4 candidates completed the frozen 152-attempt validation/selection matrix.
Selected epochs are 200/80/200. Seeds 5001/5002 pass (148/152, 75% cuboid
success), but seed 5003 fails (136/152, zero cuboid success). All safety counts
and physical-audit failures are zero. Its confirmation remains unopened.
Regression: 907 passed, one skipped. The full bindings and preserved failures
are in `level5_push_repair_v4_failed.json`.

A separate deterministic development controller using the source nominal 1 mm
along step and 20 mm stop distance passed all four fresh 20955000-series cuboid
selection resets, with zero safety and passing physical audits. This establishes
ordinary-physics long-path feasibility with the same posture. It is a scripted
source diagnostic, not learned qualification. V4 models still regress toward
zero progress after short travel. Family/absolute-height/static contact-offset
cues allow a model to encode the short training path; capping distance alone
is insufficient for the three-seed requirement.

V5 freezes geometry relative to the first pre-action observation: measured
hand/object geometry change and calibrated command-tracking change, task-plane
goal errors, object velocity/angular speed/height change, contact/support and
causal physical phase. It excludes family, absolute height and static contact-
offset cues. A per-policy reference resets at every independent rollout;
it is causal spatial calibration context, not a recurrent architecture, future
label, expert query or instance ordinal. Offline/live first-feature parity is
exact (maximum absolute error zero). Source normalization remains identity after
fixed 0.1 m / 0.1 m/s / 5 rad/s / 0.01 m scales.

The complete source audit proves task-lateral displacement is numerical roundoff
through contact, settling and signed retraction for all train/validation frames.
Together with the 4/4 long-path source feasibility screen, this independently
licenses V5 holding that channel rather than copying reach's restriction blindly.
V5 learns only signed along progress in the existing bounded scaffold. Decoder
geometry sets task-lateral exactly zero and holds height/orientation/joints;
negative/retract actions and zero-output hold remain valid. There is no positive
minimum step, engineered nominal progress or online expert control. Learned
rate, engineered direction/posture/caps and safety/terminal checks are distinct.
No generic manipulation, learned vision or unconstrained full-hand claim follows.

V5 keeps the original 96/32 supervised episodes and full three-seed 200-epoch,
8,400-update budgets, 128-by-128 ReLU MLP and optimizer. Its scalar active mask
makes the base loss average over that one learned component; inactive outputs
are ignored by loss and decoder. The unchanged sampler balances source
cell/episode/present causal phase/frame. All candidates receive the original
30 validation resets plus eight separately frozen 21355000-series all-cell
selection starts. The distinct 32-reset 21555000-series confirmation is reserved
once only after selection. No V1/V3 test or V4 selection outcome is relabeled as
new qualification. Original gates and uncapped physical metrics stay intact.

```bash
python -m dexvision.apps.train_skill --config configs/level5/push_policy_v5.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v5.yaml --select-candidates outputs/level5/push_55_v5/seed_5001 --split validation --track state --output-dir outputs/level5/push_55_v5/validation_5001
python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v5.yaml --checkpoint outputs/level5/push_55_v5/seed_5001/selected.pt --track state --split test --output-dir outputs/level5/push_55_v5/confirmation_5001
```

Repeat for 5002 and 5003. V5 focused tests pass 29 tests and Ruff passes;
complete candidate validation and final regression are pending. Owner push
review is not yet triggered; 5.5 remains incomplete.

## V6 causal completion-state hypothesis

V5's entry-relative representation moves long cuboids substantially, but early
candidates stop around 43–49 mm from the goal. That is outside the unchanged
35 mm acceptance limit. Current-state geometry alone does not distinguish
nominal progress from scripted post-success retraction once physical phase
returns to push/settle.

The producer in `WorkcellPilotTask._evaluate_single` sets push `phase.retract`
only when the current `TaskMetricResult.success` is true. A source-only audit
finds 767/768 training and 255/256 validation retract-sized actions occur after
that currently/past observed physical phase has appeared. The remaining one
per split stays unaltered; no action target creates or backdates a latch.
The new independently locked V6 observation adds a causal physical-goal-
succeeded-once bit, the maximum of current/past physical `phase.retract` flags.
It does not use the saved expert controller stage, retrospective episode
success, future state or target action. Reset it alongside the entry reference
for every independent rollout.

V6 preserves all V5 inputs, source labels, signed along mask, model, optimizer,
budget and decoder. Post-success negative retraction remains represented and
learned rather than deleting those data. Standalone push execution ends at its
first passing physical dwell; it never queries the post-completion controller
to create nominal progress. Three full budgets, all-candidate 38-reset
validation and one separate 32-reset confirmation still apply. New disjoint
21755000-series selection and 21955000-series confirmation are frozen before
training; prior consumed tests and unsuccessful versions remain preserved.
Focused suite: 30 passed; Ruff passes. V6 training/validation and final
regression are pending. No owner push acceptance or 5.5 completion is claimed.

V5 completed all candidate matrices and selected epochs 192/199/80. Every
selected seed succeeds on 140/152 attempts but only 25% of cuboid attempts,
so every seed fails the unchanged worst-family gate. Safety counts and all
physical-audit failures are zero. Confirmation was not opened. Complete
bindings and failures are in `level5_push_repair_v5_failed.json`; final
regression passed 909 tests with one skip. V6's three full budgets now finish,
and complete candidate validation/final regression are running. No manual
acceptance or positive qualification is claimed at this stage.

Complete V6 epoch-20 matrices pass for every seed: 148/152 successes, 75%
cuboid success, zero safety and all 152 physical audits passing per seed.
These are selection/development results, not confirmation or owner acceptance.
Remaining candidates still receive the full frozen matrix before selection.

Final V6 regression passes **910 tests, one skipped**, in 777.54 seconds.
Complete candidate validation is still running; confirmation is unopened.

Complete V6 selection finishes all 12 candidate matrices (1,824 counted
attempts). The selected epochs are **80/20/20**, each at 148/152 successes,
75% cuboid success, zero safety and all physical audits passing. The later
best-offline/final candidates can fail; they remain counted and preserved.
The exclusive 21955000-series confirmation is now running for these exact
weights. Numerical passes do not establish owner acceptance.

The owner command, eligible only after all three V6 fresh confirmations pass,
is:

```bash
conda activate dexvision
mjpython -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v6.yaml --checkpoint outputs/level5/push_55_v6/seed_5001/selected.pt --track state --viewer --scenario-set manual_stratified
```

Pass/fail criteria are the four-cell criteria above: intended physical contact,
upright/supported object remaining on the board, no wrong-object/fixture or
neighbor disturbance, and a visible settled endpoint matching the report in
cuboid/slot B and flat-puck/inspection-pad, each interior and boundary.
Missing clips, cancellation, stalling, tipping, board exit or mismatch are
failures. Explicit owner confirmation is required; keep 5.5 incomplete until
it arrives. Press already has separately recorded owner acceptance.


## V6 confirmation failure and V7 physical predicates

V6 one-time fresh confirmation completes at **100/128, 100/128 and 96/128**
successes. Cuboid rates are **56.25%, 56.25% and 50%**, below the unchanged
65% worst-family threshold. Pucks pass every attempt; safety is zero and
all 384 independent physical audits pass. The complete failed binding is
`level5_push_repair_v6_failed.json`. The preceding V6 viewer command is
historical and ineligible; no manual acceptance was triggered. The 21955000
resets are consumed and will not be retried.

V7 tests a simpler 12-input observation with capped signed-along/radial
goal errors, physical alignment/speed/tracking predicates, contact/support,
current physical phases and the causal completion latch. The source speed
limit is 0.02 m/s; alignment and tracking envelopes are frozen at 0.035 m
and 0.01 m. Continuous tracking/velocity and constant family/height cues are
excluded. The learned signed along action remains active, with no minimum
progress or online expert. Near-goal/signed errors and all physical gates
remain exact. Source train/validation counts and labels are unchanged.

The original 30 selection resets plus 16 fresh 22155000-series resets, four
per original test cell, give **46 unique resets / 184 condition attempts**
per candidate. A disjoint 32-reset 22355000-series confirmation is frozen
before training. One confirmation setup tips during the scripted approach;
its empty prefix fails the contact-entry distance check and stays counted
as four rejected attempts. No filtering, replacement or policy execution
occurs for that setup. The three full 200-epoch budgets and full candidate
selection are required before the separate confirmation is opened.

The inherited primary recipe initially retained an obsolete seed-start
metadata field (21155000) while its actual locked setup recipe and frozen
records used 22355000. Before any training, that declaration was corrected
and cache/matrix recipe bindings reindexed without changing records or arrays.
The pre-correction recipe is preserved under the ignored run root.
Focused tests pass **31**; full regression and training are running.


### V7 exact automated commands

All commands use the `dexvision` environment. Matrices and cache were frozen
before training using `freeze_entry_matrices` and `prepare_push_cache`; the
cache reuses checksum-verified original train/validation tensors, with no
confirmation supervision. Run each listed seed, without replacing prior
attempt directories or retrying a reserved confirmation.

```bash
conda activate dexvision
for seed in 5001 5002 5003; do
  python -m dexvision.apps.train_skill --config configs/level5/push_policy_v7.yaml --seed "$seed" --release-root .
done
for seed in 5001 5002 5003; do
  python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v7.yaml --select-candidates "outputs/level5/push_55_v7/seed_$seed" --split validation --track state --output-dir "outputs/level5/push_55_v7/validation_$seed"
done
# Only after all three selected validation candidates pass:
for seed in 5001 5002 5003; do
  python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v7.yaml --checkpoint "outputs/level5/push_55_v7/seed_$seed/selected.pt" --track state --split test --output-dir "outputs/level5/push_55_v7/confirmation_$seed"
done
pytest -q tests/test_push_repair.py tests/test_push_skill_rollout.py tests/test_press_skill_rollout.py tests/test_level5_learning_plan.py
pytest -q
ruff check dexvision tests
shasum -a 256 -c datasets/level4-v1.tar.gz.sha256
git check-attr filter -- datasets/level4-v1.tar.gz
```

Runs already present under these paths are immutable evidence, not rerun
targets. `conda run --no-capture-output -n dexvision` was used to execute the
Python/pytest/Ruff commands without relying on an activated parent shell.
Original `configs/level5/SHA256SUMS` and all independently locked push recipes
are checked alongside bundle/source pins.


Complete V7 epoch-20 selection passes all three seeds at **184/184**,
100% worst-family success, zero safety and all 552 physical audits passing.
Remaining candidates still require their complete matrices. This is selection
evidence, not fresh confirmation or owner acceptance. First validation-entry
offline/live transformed inputs match exactly (maximum absolute error 0).
The preserved parity log is `outputs/level5/push_55_v7/logs/feature_parity.log`.


Final V7 full regression passes **911 tests, one skipped**, in 776.56 seconds.
Focused suite passes **31**; repository Ruff passes. All three training seeds
complete **200 epochs / 8,400 updates**. All remaining selection candidates
are still evaluated before opening confirmation.


Complete V7 selection evaluates all **12** candidate matrices at
**184/184 each (2,208/2,208 total)**, with zero safety and every physical
audit passing. Validation selects epochs **200/61/80** for seeds
5001/5002/5003. Exact weights, recipe, source/code/schema and checksum
bindings pass. Only now are the separately reserved 22355000-series
confirmations opened, once per selected seed.


## V7 failure and V8 bounded physical execution

V7 confirmation completes at **124/128, 112/128, 116/128**, with worst-family
rates **93.75%, 81.25%, 87.5%**. Seed5001 passes; seeds 5002/5003 fail the
zero-workspace gate with **12/8 violations**. Every physical audit passes.
The four failed setup attempts per seed remain counted. The full failed
three-seed binding is `level5_push_repair_v7_failed.json`; no manual review
is triggered and all 22355000-series resets are consumed.

One seed 5003 failure switches from nominal contact to approach after its
first action; `phase.approach` is constant zero throughout source training.
Its inferred action then turns negative and walks the hand toward the
workspace edge. A seed 5002 failure instead predicts negative movement
while a currently qualifying state is still collecting its five-sample
dwell, repeatedly breaking the dwell and eventually leaving the workspace.

V8 retains exact V7 full-budget candidates and introduces a separately pinned
execution overlay, following the prior pick/place separation of learned
weights from explicitly qualified geometric/terminal controls. It aliases
physical pre-contact approach to nominal contact progress only before
completion. A source-only audit finds **48 training and nine validation**
positive movement samples before actual contact with nominal contact phase;
those causal starts support this encoding without inventing new action labels.

The overlay holds a currently qualifying settle state through the unchanged
physical dwell. It uses no future state; separate saved-state reconstruction
checks every hold against the original physical qualifying predicate. Signed
XY progress is uniformly scaled to the original command workspace; zero
holds and negative motion remain valid, with no minimum progress. The whole
source-command audit confirms all **10,545 training /2,963 validation**
commands remain unchanged by this bound. No original safety/success
threshold changes. Every unbounded learned proposal, phase alias, physical
hold and workspace fraction is saved and independently recomputed. Decoder
bound events are disclosed rather than described as learned safety. Goal
failures still count even if the decoder prevents unsafe movement.

`entry_overlay.json` binds exact parent weights/source artifacts, execution
code, recipe and both matrices. The copied training manifest retains V7's
original training/selection evidence; the separately frozen V8 execution
matrix is bound by the overlay, not rewritten into that historical manifest.
New22555000-series all-cell selection adds32 fresh starts (eight per cell)
to the original 30, for **62 unique/248 attempts** per candidate. Separate
22755000-series confirmation contains 32 fresh starts. All setup failures
remain counted. V8 has no new training or source-label change.
Focused tests pass **34**; Ruff passes; final regression and freeze run.
The accepted press runtime remains unchanged and 5.5 remains incomplete.


All V8 selection/confirmation setups freeze ready. An original-validation
smoke succeeds with zero safety, all saved-state physical checks passing and
four independently qualifying engineered dwell holds. Full V8 candidate
selection is running. It applies the unchanged safety-first ranking to the
new 248-attempt execution evidence, with its own strict overlay/matrix/count
checks; the immutable parent training manifest is not altered to impersonate
the new selection. Source aliases operate only within the frozen contact-entry
envelope, not as a newly qualified free-space approach controller.


V8's first selection launch exposed a JSON logging defect: a NumPy int64
workspace-bound counter was not serializable. Seeds5001/5002 raised;
seed 5003 and the in-progress regression were stopped. No confirmation
reservation existed. All three partial bundles, validation trajectories,
logs and exact execution code are preserved in
`outputs/level5/push_55_v8/reporting_failure/`. Frozen matrices were untouched.
The repair uses native float/int counters and adds the failing NumPy case to
the audit unit test. All **16,940 actions /116 archived attempts** recompute
exactly unchanged; counters now serialize. Partial goal timeouts remain
preserved and are not filtered from the restarted complete matrix. Focused
tests pass **34** after the fix; full regression and complete selection
restart with the same frozen method, weights and matrices. No consumed confirmation is retried.


### V8 exact commands

All learned source runs already completed200 epochs/8400 updates under the
locked V7 recipe above. V8 reuses those exact candidates; its train CLI rejects
a new fit. Matrix freeze was executed before candidate selection:

```bash
conda activate dexvision
python -c "from dexvision.evaluation.push_guarded import freeze_entry_matrices; print(freeze_entry_matrices('configs/level5/push_policy_v8.yaml', '.'))"
for seed in 5001 5002 5003; do
  python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v8.yaml --select-candidates "outputs/level5/push_55_v8/seed_$seed" --split validation --track state --output-dir "outputs/level5/push_55_v8/validation_$seed"
done
# Only after all three selected validation candidates pass:
for seed in 5001 5002 5003; do
  python -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v8.yaml --checkpoint "outputs/level5/push_55_v8/seed_$seed/selected.pt" --track state --split test --output-dir "outputs/level5/push_55_v8/confirmation_$seed"
done
pytest -q tests/test_push_repair.py tests/test_push_skill_rollout.py tests/test_press_skill_rollout.py tests/test_level5_learning_plan.py
pytest -q
ruff check dexvision tests
shasum -a 256 -c datasets/level4-v1.tar.gz.sha256
git check-attr filter -- datasets/level4-v1.tar.gz
```

These directories preserve this run and are not overwrite/retry targets.
Conditional owner review, eligible only after all three confirmations pass:

```bash
conda activate dexvision
mjpython -m dexvision.apps.evaluate_skill --config configs/level5/push_policy_v8.yaml --checkpoint outputs/level5/push_55_v8/seed_5001/selected.pt --track state --viewer --scenario-set manual_stratified
```

Review all four fresh-confirmation cells: cuboid/slot B and puck/inspection pad,
each interior and boundary. Pass only when the intended object is visibly
pushed, remains upright/supported on the board, no wrong object/fixture or
neighbor disturbance occurs, and the settled goal endpoint matches the report.
The physical limits remain35mm planar error, 20mm/s object speed, at most10
degrees tilt and five qualifying samples. Stalling, tipping, board exit,
wrong contact, a mismatched terminal state, missing clips, cancellation or
viewer errors fail. Viewer playback never records owner acceptance by itself.
Press already has separate owner acceptance; keep 5.5 incomplete until the
owner explicitly confirms this push review passed. No 5.6 work or push.


Final V8 regression after the serialization repair passes **914 tests, one
skipped**, in 773.48 seconds. Focused tests pass **34**; repository Ruff
passes. Complete candidate selection continues; fresh confirmation is unopened.


All **12 V8 complete candidate matrices** pass the unchanged state gates
across **2,976 attempts**. Every physical/decoder/phase audit passes and
executed safety is zero. Validation selects epochs **200/61/20**, with
**244/248, 244/248, 248/248** successes. Failed goals remain counted; selected
workspace-bound samples are **36/176/0**, and qualifying settling holds
**1,000/996/1,004**. These are disclosed engineered operations, not learned
safety or terminal control. Exact selected/candidate tensor and source/code
bindings pass. Only now is the reserved 22755000-series confirmation opened,
once per selected seed. Numerical success still requires owner review.


## Final V8 automated result and owner acceptance

One-time fresh confirmation passes every original state gate for all three
selected seeds, with **116/128, 116/128, 120/128** successes and worst-family
rates **81.25%, 81.25%, 87.5%**. All executed safety/wrong-contact counts are
zero. The **12/12/8 goal timeouts** stay counted, not filtered. All **384**
independent physical, decoder and causal/settling audits pass. Every successful
attempt has observed physical hand/object contact in the saved metrics.
Confirmation uses **32 unique resets shared across the three models**, with
four labels repeating identical state physics. No perception/unseen-family
qualification or independent-condition inflation is claimed.

Confirmation engineering counters are **8/8/8 entry-phase alias samples**,
**468/476/480 qualifying settling holds**, and **348/532/876 workspace-bound
samples**. The model learns signed along movement; initial direction, fixed
posture, entry encoding, settling hold and workspace bounds are engineered.
The safety result applies to this complete frozen execution bundle, not to
unconstrained neural proposals or learned vision.

`level5_push_repair_verification_v8.json` verifies every selected/candidate
tensor, parent artifacts and complete candidate evidence; source/schema/code/
config/cache/overlay/reset checksums; disjoint confirmation; exclusive test
reservations; all physical/decoder/phase audits; successful physical contact;
and unchanged accepted press bundles. All previous failed receipts and the
serialization-failure snapshot remain preserved. Original datasets, release
archive and 5.0 locks are unchanged. Full regression passes **914 tests,
one skipped**; focused **34**; Ruff passes.

The owner confirmed **"Passed"** on October 13, 2026 after receiving the exact
V8 manual command and four-cell pass/fail criteria above. This is owner-reported
acceptance; no agent visual observation or local viewer-log completion is
asserted. `level5_push_acceptance.json` binds the exact three selected bundles,
overlays, confirmation reports, selection evidence and four prescribed review
attempts. It also binds the unchanged prior press acceptance. Original automated
verification retains its then-pending status, and failures remain preserved.

**Checkpoint 5.5 is complete.** Last Completed is 5.5; Next Target is 5.6,
unstarted. Dial remains absent, not promoted. The latest owner instruction
("Feel free to commit and push") authorizes publishing the checkpoint commits.


## Changed files

Combined checkpoint source/documentation inventory, including local commit
`61de6af` and the subsequent push repair. Generated weights, caches, logs
and trajectories stay under ignored `outputs/level5/`; existing datasets
and the ignored interview PDF are preserved.

```text
.gitignore
configs/level5/press_policy_v1.sha256
configs/level5/press_policy_v1.yaml
configs/level5/push_policy_v1.sha256
configs/level5/push_policy_v1.yaml
configs/level5/push_policy_v2.sha256
configs/level5/push_policy_v2.yaml
configs/level5/push_policy_v3.sha256
configs/level5/push_policy_v3.yaml
configs/level5/push_policy_v4.sha256
configs/level5/push_policy_v4.yaml
configs/level5/push_policy_v5.sha256
configs/level5/push_policy_v5.yaml
configs/level5/push_policy_v6.sha256
configs/level5/push_policy_v6.yaml
configs/level5/push_policy_v7.sha256
configs/level5/push_policy_v7.yaml
configs/level5/push_policy_v8.sha256
configs/level5/push_policy_v8.yaml
configs/level5/push_selection_v4.sha256
configs/level5/push_selection_v4.yaml
configs/level5/push_selection_v5.sha256
configs/level5/push_selection_v5.yaml
configs/level5/push_selection_v6.sha256
configs/level5/push_selection_v6.yaml
configs/level5/push_selection_v7.sha256
configs/level5/push_selection_v7.yaml
configs/level5/push_selection_v8.sha256
configs/level5/push_selection_v8.yaml
configs/level5/push_setup_v3.sha256
configs/level5/push_setup_v3.yaml
configs/level5/push_setup_v4.sha256
configs/level5/push_setup_v4.yaml
configs/level5/push_setup_v5.sha256
configs/level5/push_setup_v5.yaml
configs/level5/push_setup_v6.sha256
configs/level5/push_setup_v6.yaml
configs/level5/push_setup_v7.sha256
configs/level5/push_setup_v7.yaml
configs/level5/push_setup_v8.sha256
configs/level5/push_setup_v8.yaml
dexvision/apps/evaluate_skill.py
dexvision/apps/train_skill.py
dexvision/evaluation/contact_policy.py
dexvision/evaluation/push_canonical.py
dexvision/evaluation/push_distance.py
dexvision/evaluation/push_guarded.py
dexvision/evaluation/push_latched.py
dexvision/evaluation/push_predicates.py
dexvision/evaluation/push_repair.py
dexvision/evaluation/push_settled.py
dexvision/learning/contact_policy.py
dexvision/learning/push_canonical.py
dexvision/learning/push_distance.py
dexvision/learning/push_guarded.py
dexvision/learning/push_latched.py
dexvision/learning/push_predicates.py
dexvision/learning/push_repair.py
docs/CURRENT_STATUS.md
docs/level5_contact_results.md
docs/level5_contact_verification.json
docs/level5_press_acceptance.json
docs/level5_push_repair.md
docs/level5_push_repair_v2_failed.json
docs/level5_push_repair_v3_failed.json
docs/level5_push_repair_v4_failed.json
docs/level5_push_repair_v5_failed.json
docs/level5_push_repair_v6_failed.json
docs/level5_push_repair_v7_failed.json
docs/level5_push_repair_verification_v8.json
docs/module_contracts.md
docs/progress_level_5.md
tests/test_level5_learning_plan.py
tests/test_press_skill_rollout.py
tests/test_push_repair.py
tests/test_push_skill_rollout.py
```
