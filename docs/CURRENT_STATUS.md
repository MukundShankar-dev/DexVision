# Current Project Status

This file is the source of truth for what Codex or another coding agent should work on next.

Agents should read this file before selecting any checkpoint.

---

## Current Level

Level 5 — Workcell Skill Learning and Qualification

---

## Current Progress File

`docs/progress_level_5.md`

---

## Last Completed Checkpoint

Level 5.6 — Detector, Tracker, and Metric Pose Baseline

Completed October 16, 2026 after the owner confirmed the V29/fresh V8 manual
review passed. All automated gates pass (full regression: 1,177 passed,
one skipped). Owner-reported acceptance of the 16 panels and 36 temporal clips
is bound to exact artifacts in `docs/level5_perception_acceptance.json`.
All five accepted inferred-control tracks remain unsupported; this completion
qualifies the documented visual baseline only. No default registry admission
or closed-loop control qualification is claimed.

---

## Next Target Checkpoint

Level 5.7 — Perception-Grounded Skill Rollouts

**Implementation and automated evidence finished; checkpoint incomplete pending
owner manual review.** The owner authorized a compact instrumented observation
amendment: visible fiducials plus declared ideal simulated robot/base, fixture
and contact ports feed the exact accepted policies. V29 natural RGB control
remains unsupported; no new model training, dataset archive or default registry
admission occurs.

The frozen test accounts for all 3,540 requests (708 state references, 2,832
perception attempts), all three selected seeds and every required cell/condition.
All 13,017 indexed artifacts, 161,301 consumed feature rows, paired entries and
admitted observation ages pass readback. Twenty held-out review clips contain
218 decoded/annotated frames. The interrupted zero-action place request was
recorded after preserving 1,291 completed outcomes/4,917 files; continuation
executed only unstarted requests, with zero consumed-control retries.

Software checks: **1,235 passed/one skipped** full regression (765.05 s), **54
focused passed**, Ruff and whitespace pass. These software passes do not imply
skill qualification. Perception success: reach 100%; pick 41.7–45.8%; place
41.9–45.0%; press 99.2–100%; push 0% with zero applied perception actions.
All tracks fail at least one original numerical gate; pick seed 5003 also has
two object-workspace violations. Independent event/group audits retain all
negative outcomes and original thresholds. No test tuning or control rerun.

A before-test storage-only amendment raised the initial 320 MiB ceiling to
640 MiB, with unchanged controls/seeds/gates and AST-bound validation inheritance.
The experiment occupies about 469 MiB on disk; frozen releases and existing
operator data are unchanged. The original protocols, validation and failed
attempts remain preserved. See `docs/level5_perception_integration.md` and
`docs/level5_perception_integration_verification_v2.json` for exact bindings,
commands, limits and review criteria. Last Completed stays **5.6**; Next Target
stays **5.7**. Do not mark it complete before owner confirmation or start 5.8.

---

### Historical 5.5 completion

Level 5.5 — State-Grounded Push, Press, and Optional Dial Policies

Completed October 13, 2026 after the owner accepted press ("Press passes") and
push ("Passed"). Push V8 selected epochs 200/61/20 pass one-time confirmation
at 116/128, 116/128 and 120/128 successes, zero executed safety/wrong contact,
and 384/384 physical/decoder/phase audits. Press V1 selected epochs 200/189/199
pass at 120/120 each with 360/360 physical audits. Dial is explicitly absent.
Full regression: 914 passed, one skipped; focused: 34 passed; Ruff passes.

`docs/level5_push_acceptance.json` and `docs/level5_press_acceptance.json` bind
owner-reported reviews to exact artifacts. Historical automated receipts,
failed attempts and datasets remain unchanged. Qualification applies to frozen
state bundles on fresh resets of exposed families, with disclosed engineered
geometry, entry/settling control and bounds; no learned vision, unseen-family,
default registry admission or deployment qualification is claimed.

---

## Completed 5.6 outcome and qualification history

Owner-authorized publication: push code, compact V25/V29 model bundles,
manifests, checksums, protocols, acceptance and status docs. Keep the large
5.6 dataset payloads and multipart data local because the full upload exceeds
the owner's reported 10 GiB LFS allowance. A clone alone does not include
these new datasets. See `datasets/README.md` and
`docs/level5_perception_publication.json` for exact scope and local restoration.
Acceptance and scientific bindings are unchanged; 5.7 remains unstarted.

**Current outcome: 5.6 complete; V29/fresh V8 automated gates and owner manual acceptance pass.**
The owner confirmed the held-out review passed on October 16, 2026.
Last Completed is **5.6**; Next Target is **5.7 (unstarted)**, on branch `main`.
`docs/level5_perception_acceptance.json` binds this owner-reported acceptance to
the exact frozen artifacts; no agent visual observation is claimed. Historical
automated receipts retain their then-pending manual status. No 5.7 work is
authorized by this completion request. The earlier V8 validation diagnostic
review remains separately accepted.

V29 retains exact V25 selected seed 11/source epoch 3,000 current-RGB MLP weights,
TRAIN transforms, fixed 8 px / 8 px / 10 mm output scales and calibrated ray decoder,
with zero new optimizer updates. All three original current candidates and all
six source fits remain preserved. Cuboid association uses the calibrated
projection of the current learned body center, with a cuboid-only uncertain
replacement-ID guard and history rollback. Other classes keep original birth/
visible-box association. Published masks, boxes and positions are unchanged.
Original tracker limits and all original visual gates remain unchanged.
Added identity coverage requires 99% unambiguous primary matches overall and 95%
in every supported primary class/phase; V29 passes both.

Validation: 5,923/5,923 primary matches and positions, zero identity switches, median
0.904 mm/p95 3.414 mm;
all 545 six-phase queries pass (worst phase p95 14.082 mm).
Fresh V8: 512 ordinary-physics source resets/26,687 RGB frames,
primary recall 100.0000%, median 0.954 mm,
p95 3.675 mm, 14,244 positions and
0 primary ID switches. All original/global/core/primary, supported phase,
position coverage, added identity coverage, and mandatory release/settle gates
pass independently. Release: 46 visible from
13 physical sources;
settle: 24 visible from
13 physical sources.

Full software: 1,177 passed/1 skipped; focused 64, Ruff and
whitespace pass. Actual 64 validation truth-isolation queries and 48 cached/uncached runtime
paths, exact accepted-transform oracles, portable RGB relocation, archive/member
readbacks, source/body-origin/schema audit, all 512 video decodes, typed/raw
parity and actual consumed-protocol retry rejection pass. The review has 16
panels and 36 fixed-plus-worst temporal clips.
Evidence: `docs/level5_perception_verification_v29.json`.

Model semantic SHA: `3a25ea7b49cc05ce7d67e89226a7e8c231a7feaea945d4863e7ed56a5cf5cadd`.
Active current-weight SHA: `0ba58825982fbfd714f028c3b8a9c4195e9f4ec5befe678e5273025863bbdb85`.
The baseline V2 and fresh V2–V7 consumed tests stay permanently closed, with all
archives, data, failed/positive candidates, reports and reservations preserved.
V7 failed one primary cuboid/release ID switch; V26/V27 failed independent identity
coverage. V28 positive validation and the independently frozen V29 guard plan
are preserved. Fresh V8 is now consumed too and cannot be retried or used for
training/selection. All source outcomes were retained; no test tuning or reuse.

The availability-aware schema is shared by truth and inference. Physical
linear/angular motion, full orientation, semantic identity, robot/fixture
telemetry and contacts remain explicitly unavailable and fail closed. All five
accepted inferred-control tracks remain unsupported. Qualification scope is
calibrated simulator pixel-center images on exposed task families/cells;
no real-camera, unseen-family, hardware, full-feature or closed-loop control
qualification is claimed. Python traced peak: 32,454,158 bytes for
model construction and 64 queries after imports; native/full-media peak remains
unmeasured and RTX 5070 Ti is untested.

Manual command (dexvision Conda environment, repository root):
`python -m dexvision.apps.repair_perception review --evaluation-dir outputs/level5/perception_confirmation_v8/evaluation`.
The owner confirmed all prescribed review criteria passed on October 16, 2026;
`docs/level5_perception_acceptance.json` records acceptance of the 16 panels and
36 scheduled clips. The command above is retained for saved-media reproduction;
no manual check remains for 5.6. Stop before 5.7.

### Earlier V8 validation and typed-boundary work

Previously V8 passed the original aggregate and supplemental primary-target
visual gates on TRAIN/validation-only repair. On 531 validation frames, recall
is 99.61%, median 0.051 mm and p95 0.83 mm. Primary-target recall is 97.60%,
median 0.44 mm and p95 7.10 mm, with zero identity switches and 100% synthetic
stale/ambiguity rejection. All four primary-class recall gates pass.

V1 through V7 failed/insufficient attempts, the consumed V2 held-out result,
source snapshots and all candidate matrices remain preserved. Repair recipes
reject test evaluation before loading models/images; no held-out rerun or tuning
occurred. V8 freezes RGB component grouping, causal feature/scaling, all 21 pose
candidates and the selected three-neighbour TRAIN regressor with its exact
calibrated ray decoder. The 940 false positives and uncertified/unavailable
fields remain disclosed; no typed-schema, skill or deployment qualification
is claimed. Accepted policies and datasets remain unchanged.

Final V8 full regression: 963 passed, one skipped; focused: 59 passed. Ruff,
whitespace and complete artifact readback pass.

Owner validation diagnostic review is confirmed complete in
`docs/level5_perception_validation_acceptance.json`. Fresh independent
confirmation and full schema availability were then pending. Details/commands are in
`docs/level5_perception_repair.md`; artifact bindings and software verification
are in `docs/level5_perception_repair_verification_v8.json`. Historical V2
reports/receipt remain unchanged. Last Completed remains 5.5; Next Target
remains 5.6. No 5.7 work has started.

October 14 continuation: `dexvision/perception/observation_boundary.py` adds
an offline field-evidence sidecar and guarded conversion to the unchanged
`ObjectObservation`/`WorldState` types. Each required field has its own producer,
units, frame/body origin, confidence, validity and acquisition time. Synthetic
schema parity is tested; V8 RGB output stays uncertified and rejects because
semantic identity/full orientation are unavailable. Frozen V8 implementations,
recipes, artifacts and datasets are unchanged. Fresh independently frozen
confirmation data/protocol was not then present locally; full inferred parity and
held-out owner acceptance remained pending. Continuation checks: 1,040 passed, one skipped
in the full suite (733.00 s), 136 focused, original focused 59; Ruff, whitespace,
V8/accepted artifact pins and saved-estimate boundary readback pass.
Checkpoint 5.6 is not complete.

---

## Historical Level 5.4A Owner Acceptance

Level 5.4A — Place-Held-Object Policy Repair

Completed October 5, 2026 with V25 bounded state qualification and owner
manual viewer acceptance ("Passed"). The exact V19 learned weights at epochs
20, 20 and 200 are packaged with the frozen V25 runtime, source/config pins,
engineered geometric decoder, wrist hold, collision preview and bounded
pad opening. Full validation and fresh all-cell selection pass every state
gate; all 180 selection physical audits pass. One-time 1600 confirmation
passes all three seeds at 196/240, 204/240 and 192/240 successes, with
94.2%, 94.4% and 92.3% settling and zero recorded safety violations.
All 720 independent confirmation audits and artifact integrity checks pass.
Full regression: 890 passed, one skipped; focused tests: 18 passed; Ruff passes.

`docs/level5_place_acceptance.json` binds the owner's prescribed seed-5001
three-target-type review to the exact three selected bundles, recipes,
confirmation reports and physical audits. The original automated verification
in `docs/level5_place_repair_verification_v25.json` retains its historical
pending manual status. Failed policy versions and counted attempts remain
preserved; original datasets were not overwritten. This is state-based
qualification on fresh resets of exposed families, without learned vision,
unseen-family, default registry admission or deployment qualification.
Commands and limits are documented in `docs/level5_place_repair.md`.

---

## Level 5.5 Qualification History

Earlier version and checkpoint narratives below retain their status at the time
of those results. The current selection and manual acceptance are defined by
Last Completed, Next Target and Manual Verification Status.

Level 5.5 — State-Grounded Push, Press, and Optional Dial Policies

**Current outcome:** Checkpoint 5.5 is complete after owner push viewer
acceptance ("Passed", October 13, 2026). V8 push passes all three state gates;
accepted V1 press is unchanged. `docs/level5_push_acceptance.json` binds the
owner statement to the exact bundles and four prescribed review cells.
Last Completed: 5.5. Next Target: 5.6, unstarted. Dial is absent, not promoted.

V1 automated work finished October 13, 2026; owner press viewer review passed.
Separate checksum-locked push/press recipes and standalone contact-phase
interfaces are implemented. All six runs completed 200 epochs and every
candidate received full closed-loop validation, then each selected seed ran
its held-out matrix once. Press epochs 200/189/199 pass every state gate at
120/120 successes each, zero safety, and 360/360 passing physical audits.
Push epochs 20/200/200 fail success/worst-family and safety gates at 40/128,
56/128 and 44/128 successes, with zero flat-puck test successes; all failures
remain counted and 384/384 physical consistency audits pass. Dial is absent,
not promoted. Full regression: 900 passed, one skipped; focused: 20 passed;
Ruff, source/release integrity and selected-weight binding pass.

The owner confirmed "Press passes" and explicitly requested a push repair,
keeping 5.5 incomplete. Press acceptance is separately bound in
`docs/level5_press_acceptance.json`; the original automated verification keeps
its historical pending-manual status. V2 full-budget task-frame training and complete validation are preserved as
a negative result in `docs/level5_push_repair_v2_failed.json`: parked non-target
puck vertical motion also triggers wrong-contact with zero policy output.
V3 reuses exact V2 weights with a fixed ordinary-physics entry hold.
All 1320 V3 validation attempts pass, but its one-time 20755000-series
confirmation fails success/worst-family gates at 92/128, 96/128 and 68/128,
with zero safety and all 384 physical audits passing. All failures are
preserved in `docs/level5_push_repair_v3_failed.json`; those resets are consumed
and not eligible for a retry. Source-only action audits support the separate
V4 positive-goal-distance observation cap: cuboid training distance was only
27-51 mm, while a stalled fresh first state was about 157 mm away. Distances
beyond 50 mm share the same nominal forward target. V4 preserves signed outputs
and uncapped physical gates, then adds fresh 20955000-series all-cell selection
and disjoint 21155000-series confirmation, frozen before training.
V4 completes three 200-epoch budgets and every 152-attempt candidate matrix,
but seed 5003 has zero cuboid selection success. Seeds 5001/5002 have passing
candidates; the required three-seed method remains failed and its confirmation
is unopened. All negative artifacts are retained in
`docs/level5_push_repair_v4_failed.json`. A separate V5 entry-relative geometry
representation removes constant family/height/contact-offset cues and learns
one signed task-along rate. Whole-interval source lateral invariance and a
4/4 ordinary-physics long-cuboid development screen independently license the
held lateral/posture channels. V5 freezes disjoint 21355000-series selection
and 21555000-series confirmation before its full three-seed training budgets.
Focused tests: 29 passed; offline/live entry feature parity is exact.
V5 completes every candidate matrix and fails worst-family gates for all three
selected epochs 192/199/80 (140/152 attempts, 25% cuboid success each), with
zero safety and passing physical audits. Its confirmation is unopened; full
regression passes 909 tests with one skip. The failed receipt is preserved.
A source-only audit identifies 767/768 training and 255/256 validation retract-
sized actions after a current/past physical-success phase. V6 adds a causal
physical-goal-succeeded-once bit, derived from that existing physical phase,
to distinguish post-success retraction from nominal progress. It preserves all
source labels, signed action/decoder, uncapped gates and prior failures.
Disjoint 21755000-series selection and 21955000-series confirmation are frozen
before V6 training; focused tests pass 30 tests. All three full training
budgets finish. Complete epoch-20 selection matrices pass for all seeds at
148/152 successes, 75% cuboid success, zero safety and passing physical audits.
Complete V6 selection finishes all 12 candidate matrices and selects epochs
80/20/20, each at 148/152 with 75% cuboid success and zero safety. Its one-time
fresh confirmation fails at 100/128, 100/128 and 96/128 successes,
with cuboid rates 56.25%, 56.25% and 50%. Safety is zero and all 384
physical audits pass. The negative receipt is
`docs/level5_push_repair_v6_failed.json`; those resets are consumed. Final
V6 regression passes 910 tests with one skip. V7 freezes simpler physical
predicates, retains signed learned progress and causal completion history,
and expands fresh all-cell selection to 16 resets plus the original 30.
The disjoint 22155000-series selection and 22355000-series confirmation
are frozen before three full training budgets. All 46 validation and 32
confirmation resets remain counted, including one failed confirmation setup.
V7 focused tests pass 31 tests; all three 200-epoch / 8,400-update budgets
finish. Full regression passes 911 tests with one skip; Ruff passes. All
three complete epoch-20 selection matrices pass at 184/184 with zero safety
and passing physical audits. All 12 complete candidate matrices pass (2,208/2,208 successes), and
validation selects epochs 200/61/80. One-time V7 confirmation finishes at 124/128, 112/128 and 116/128.
All success thresholds pass, but seeds 5002/5003 record 12/8 workspace
violations; all 384 physical audits pass. The failed three-seed result is
preserved in `docs/level5_push_repair_v7_failed.json`, and those resets are
consumed. V8 preserves exact full-budget V7 candidates with a separately
frozen causal entry/settling and original-workspace decoder overlay.
Every alias, hold and bound is recorded and independently audited. Source
advancing-before-contact and whole-interval workspace identity audits pass.
V8 focused tests pass 34 tests; Ruff passes. New 22555000-series all-cell
selection (32 fresh plus original 30; 62 unique/248 attempts) and disjoint
22755000-series confirmation are frozen before overlay selection.
All 62 selection and 32 confirmation setups freeze ready. An original
validation smoke passes with zero safety; all four engineered dwell holds
independently qualify. Final V8 regression passes 914 tests with one skip; focused tests pass 34
and Ruff passes after the reporting fix. All 12 complete candidate matrices pass across 2,976 attempts, with every
physical/decoder/phase audit passing and zero executed safety. Validation
selects epochs 200/61/20 at 244/248, 244/248, 248/248 successes. The timeouts
and 36/176/0 selected workspace-bound samples remain disclosed. One-time fresh V8 confirmation passes every original state gate at
116/128, 116/128, 120/128 successes, with worst-family rates 81.25%, 81.25%,
87.5%, zero executed safety/wrong contacts and all 384 physical/decoder/phase
audits passing. The 12/12/8 goal timeouts stay counted. Confirmation records
348/532/876 engineered workspace-bound samples and 468/476/480 physical
settling holds; these are not learned safety/terminal control. Verification
in `docs/level5_push_repair_verification_v8.json` passes exact tensor,
source/code/config/reset and checksum bindings, observed physical contact
for every success, and unchanged accepted press bundles. Final regression:
914 passed, one skipped; focused 34; Ruff passes. Owner push viewer
acceptance is recorded separately in `docs/level5_push_acceptance.json`;
original automated verification retains its historical pending status.
Last Completed is 5.5; Next Target is 5.6, unstarted.
No consumed V1/V3/V6 matrix is retried.
See `docs/level5_push_repair.md`.
Do not rerun or tune against V1's
consumed held-out matrix. No perception/default registry or deployment
qualification is claimed. Checkpoint 5.5 is complete; 5.6 remains unstarted.
The owner's latest acceptance authorizes committing and pushing this update.

---

## Historical Level 5.4 Result

Level 5.4 — State-Grounded Place-Held-Object Policy

Completed October 3, 2026 with automated verification and a **negative state
qualification** result. The independently locked place V1 recipe trained all
three seeds for 200 epochs / 13,200 updates. Full closed-loop validation selected
epochs 20, 56 and 20. One-time held-out evaluations achieved 4/216, 0/216 and
12/216 successes; every seed fails success, premature-release and safety gates.
All three target types and held-out goal regions were evaluated. No viewer check
is required after these numerical failures, and no owner visual acceptance or
qualified place skill is claimed.

The source is the unchanged immutable Level 4 pick/place release. The new
policy uses causal state features, calibrated measured-to-command geometry,
active finger release and a post-integration physical outcome check. The exact
selected weights are bound to the recipe and validation candidates. All 16
reported held-out successes pass independent saved-state reconstruction of the
final ten physical states. Full regression: 816 passed, one skipped; focused
tests: 18 passed. Ruff, config/source checksums, LFS attribute, artifact indexes
and selected-weight integrity pass. Exact commands, per-seed metrics, diagnostic
limits and hashes are in `docs/level5_place_results.md` and
`docs/level5_place_verification.json`.

Failed place attempts, draft runs and the validation-only release-gating
diagnostic remain in Git-ignored `outputs/level5/place_54_v1/`. Selected V1
weights, rollouts and audit evidence also remain local and are not a published
model release. Original Level 4 and Level 5 pick data, historical configs and
completed checkpoint results are unchanged. Level 5.5 has not started.

---

## Level 5.4A Repair History

Level 5.4A — Place-Held-Object Policy Repair

Completed October 5, 2026 after the owner confirmed V25 manual review with
"Passed". The history below preserves failed repair versions and development
evidence; acceptance is bound in `docs/level5_place_acceptance.json`.
The completed negative 5.4 result remains historical evidence. The active
repair work and its validation-only diagnosis are in
`docs/level5_place_repair.md`; its separate checkpoint section is in
`docs/progress_level_5.md`. The V2 ordinary-physics teacher passes its exposed
slot-B screen (29/30, zero safety), but its all-cell screen fails (15/30). A
frozen V3 teacher passed 30/30 on the exposed development matrix, then **failed
a separate fresh all-cell screen at 22/30**. Frozen V4 passed 29/30 on that
exposed matrix and failed another fresh all-cell screen at **26/30**. A bounded
exact-reset candidate search passed a further fresh development screen at
**29/30**, with 9/10 worst-family success and zero safety violations. This
cleared only the scripted-source feasibility item. All negative attempts are
retained. At that stage no additive release or learned policy existed. The
earlier regression passed 823 tests with one skip before the search changes.
Frozen additive data V1 collection stopped after a new
`860260906` object-workspace safety failure; no archive was published.
Search V2 preserves that failure, uses a safe no-success fallback and is
passed a fresh 60-reset nominal/boundary screen at **59/60**, worst family
19/20, zero selected live safety violations. Additive data V2 is frozen with
new assignments. Collection passed 140/144 training and 47/48 validation
resets with zero live safety violations. Independent saved-state reconstruction
passed all 187 successful payloads; the five failures remain counted and
excluded from supervision. The immutable V2 archive, checksum and manifest
are under `datasets/level5-place-unassisted-v2*` with the LFS attribute
verified. Versioned learned V2/V3/V4/V5/V6 recipes and fresh confirmation
resets are frozen. V2 direct action deltas and V3 fixed hand waypoints failed
development diagnostics after all three 200-epoch training budgets; their
weights and attempts remain local negative evidence. V4 object-feedback
partial validation proves every candidate fails a worst-family gate; its
task-local transport cadence was incorrectly mapped. V5 corrected that
0.005 m step and selected all three seeds after full validation, but a
separate all-target development screen exposed one object-workspace violation
and a failed settling gate. Its one-time confirmation was not used. V6 keeps
the same learned targets and adds a bounded held-object waypoint, a 0.0245 m
cuboid release trigger within the frozen 0.025 m limit, and slower coupled
cylinder opening. It passed separate 27/30 nominal and 27/30 boundary
all-target development screens with zero safety, then completed all three
200-epoch training runs. Complete V6 validation selected epochs 20, 200 and
80, but seed 5002 epoch 200 failed the settling gate and one separate
all-target development safety check. A frozen V7 all-target selection screen
rejected both eligible seed 5002 candidates for object-workspace violations;
no V7 three-seed bundle was completed. V8 retains the exact three full-budget
V6 weight sets and freezes a measured command/object lag hold, target-specific
cuboid opening, and a geometric release fallback for an outward-moving
cuboid near a receptacle edge. Its exact decoder passed 53/60 on an exposed
all-target development matrix, with zero safety and all original gates.
Full V8 validation completed. Its fresh all-target selector chose seed 5001
epoch 80 and seed 5002 epoch 20, but rejected both eligible seed 5003
candidates: epoch 20 missed the original settling gate (50/58 settled
nominal placements), and epoch 80 had two object-workspace violations. No
V8 three-seed bundle or confirmation resulted. A V9 decoder overlay freezes
slower opening for nonplanar cuboids, cylinders and pucks on the exact V6
weights. Predeclared epochs 80, 20 and 20 passed all gates on the now-exposed
V8 selection starts, with zero safety, and then passed full V9 validation.
The distinct fresh 1020-series selection screen passed seed 5001 but rejected
seeds 5002 and 5003 on the unchanged 90% settling gate (50/56 and 53/60
settled nominal placements). No V9 three-seed bundle exists. V10 froze a
target-aware 0–3 mm release clearance on the same exact V6 weights. It
passed both complete exposed 1000/1020 all-target development screens for
all three seeds with zero safety, but full validation rejected seed 5002:
36/64 cuboid successes missed the unchanged worst-family gate. Its new 1040
selection was not run. V11 freezes zero added rise for cuboids over placement
slots after diagnosing those bounces. All three predeclared candidates passed
complete V11 validation. The fresh 1060 selection passed seed 5001 (49/60,
49/54 settled, zero safety) but
rejected seed 5002 (52/60, 52/58 settled) and seed 5003 (49/60, 49/58
settled) on the unchanged 90% settling gate. Seed 5001 is selected, but no
three-seed V11 bundle exists. The original one-time 930 confirmation remains
unused; no V11 qualification or manual acceptance is claimed. V12 added a
short-grasp heavy-puck inspection-pad descent waypoint and a measured-state
left-bin cuboid opening choice on the unchanged V6 weights. Its targeted
exposed-reset numeric diagnostics predicted all 1060 gates pass, but an
independent saved-state audit found 15 hand-table-contact frames before
opening on a nominal success. Partial V12 validation was stopped without a
fresh 1080 selection or confirmation. V13 adds an opening fallback for that
measured negative lateral drift, only within the original physical release
limit. Six exposed low-offset pad attempts passed its development physical
audit. Complete V13 validation passed all three seeds, but the fresh 1100
selection rejected all three: seeds 5001 and 5002 each had one object-
workspace violation, and seed 5003 settled 51/59 nominal placements,
below 90%. No V13 selected bundle exists. The 1100 resets are now exposed
development evidence; the original 930 confirmation is still unused.
V14 now freezes a narrowly gated post-setup wrist hold for the measured
large-block left-bin and short-cylinder/heavy-puck slot failure states.
Targeted exposed 1100 development replays predict all three numerical gates
pass, with every changed attempt passing independent saved-state audit.
Complete V14 validation passed every state gate, but its fresh 1120 selection
passed only seed 5001. Seed 5002 had one object-workspace violation, and seed
5003 settled 51/59 nominal placements, below 90%. No three-seed V14 bundle
exists. The 1120 resets are now exposed development evidence; the 930
confirmation remains untouched. Level 5.5 has not started.
V15 now freezes bounded measured-state opening rates and an inboard lift for
the observed large-block collision. Its targeted live changes plus unchanged
saved outcomes pass every frozen gate on the exposed 1100 and 1120 matrices
for all three seeds; all changed attempts pass saved-state audit. Complete
V15 complete validation and the distinct fresh 1140 selection passed all
three seeds with zero recorded safety violations. The original one-time 930
confirmation also passed every numerical state gate (204/240, 208/240 and
216/240), but independent saved-state audit rejected all three bundles:
8, 2 and 7 distinct reported successful resets had hand-table contact before
opening. The 930 confirmation is consumed and exposed. V15 is not qualified;
a source-grounded wrist/clearance diagnostic was tested within 5.4A. Its
full exposed 930 nominal matrix passed the original numerical gates but
still had one and two audited-success contact failures for seeds 5001 and
5002. No V16 recipe or fresh qualification run was frozen.
Any subsequent candidate needs a newly frozen selection and confirmation
matrix plus independent physical audit. Level 5.4A is still open.
The V2 source has no right-bin cell in either split and keeps slot B out of
training. A checksum-locked V3 additive data plan now assigns all 30 cells to
both train and validation using 240 and 120 disjoint new resets and distinct
simulator sessions. The V3 release was published after 237/240 training and 117/120
validation successes, zero live safety, and 354/354 passing saved-state
source audits. The six failures remain counted without payloads. The new
archive, checksum and manifest are under `datasets/level5-place-unassisted-v3*`
with Git LFS and checksum verification. Historical V2 data, V15 failed
confirmation and the 5.4A completion gates remain unchanged.
V16 also freezes measured-object wrist feedback and bounded commanded-hand
clearance around the exact V6 weights. Its exposed 930/1100/1140 nominal
diagnostics pass numerical and saved-state physical checks for all three
seeds. Full V16 validation passed every state gate on 192 rollouts per seed:
180/192, 172/192 and 172/192 successes, with zero recorded safety.
The fresh 1160 all-cell selection rejected all three seeds: seed 5001
settled 51/59 nominal placements (below 90%) and had one reported-success
pre-opening hand-table contact; seeds 5002 and 5003 passed numerical gates
but had one and three such physical audit failures. No V16 bundle was
selected. The 1180 one-time confirmation remains unused. V3 all-cell
source is available for a separately frozen learning repair; owner viewer
acceptance remains conditional. No place qualification exists yet.
V17 freezes source-grounded wrist holds, a bounded pad clearance change and
two targeted seed-5003 orientation-feedback cases on the exact V6 weights.
The V3 all-cell release informed these engineered controls but did not train
those weights. Full V17 validation passed all numerical gates (172/192,
164/192, 168/192; zero recorded safety). The fresh 1260 all-cell selection
passed 52/60, 51/60 and 52/60, with all numerical gates, zero recorded
safety and zero physical-audit failures among 180 saved attempts. Exactly
epochs 80, 20 and 20 were selected. The 1260 matrix is now exposed; the
one-time 1280 confirmation passed numerical gates for seeds 5001 and 5002
at 212/240 each, but seed 5003 failed the 90% settling gate at 192/220
settled nominal placements (87.3%). Independent saved-state audits found
all 720 attempts physically consistent, but cannot override the numerical
failure. V17 is not qualified; 1280 is now exposed, no owner viewer review
is triggered, and a later recipe needs distinct unused selection and
confirmation resets. Level 5.4A remains open and 5.5 has not started.
V18 now freezes a bounded geometric release fallback for seed 5003's large
block within the unchanged 0.0245 m cuboid release distance. Exposed 1280
targeted diagnostics converted four of five unreleased pad/slot attempts to
settled placements, with all changed 1160/1260/1280 attempts passing
independent saved-state audit. Fresh 1300 selection and 1320 confirmation
resets are frozen but unused. V18 has no qualification claim yet.
Complete V18 validation passed all numerical gates for the three seeds
(172/192, 164/192 and 172/192; zero recorded safety). Fresh 1300 selection
rejected all seeds on the 90% settling gate (51/59, 47/56 and 53/59);
seed 5001 additionally had one saved-state physical audit failure. No V18
bundle was selected. The 1300 matrix is exposed, 1320 confirmation remains
unused, and owner viewer review is not triggered. Level 5.4A remains open.
V19 is a separate learned recipe on the published V3 all-cell source: 237
successful training episodes and 117 successful source-validation episodes,
with 30 cells in each split. It keeps the V18 bounded decoder and unchanged
physical and numerical gates, but trains new learned scores for three seeds
over the full frozen 200-epoch budget. New disjoint 1330 policy-validation,
1340 selection and 1360 one-time confirmation matrices are frozen. Source
dry run and focused ownership tests pass. All three 200-epoch, 20,400-update
training runs have finished. All nine closed-loop validation reports are
complete; the frozen ranking selected epochs 20, 20 and 200 at 220/240,
220/240 and 216/240 successes, all numerical gates and zero recorded safety.
The distinct 1340 all-cell selection rejected all three seeds on the
unchanged settling gate: 50/59, 51/58 and 50/58 nominal placements settled
(84.7%, 87.9%, 86.2%). Aggregate successes were 50/60, 51/60 and 50/60,
with zero recorded safety. Independent saved-state audit found one
reported-success pre-opening hand-table contact for seeds 5001 and 5003;
seed 5002 had none. No V19 bundle was selected. The 1340 matrix is exposed,
1360 one-time confirmation remains unused, and no owner viewer review is
triggered. Targeted development probes of slower opening and post-detachment
hand retreat did not fix the shared bounces. Bounded supported descent fixed
five targeted failures without audit contact, but its complete exposed 1340
screen regressed previously successful cells: 47/60, 48/60 and 48/60
successes, with all three settling gates failing and one physical audit
failure per seed. It is rejected as a general repair and does not change
the frozen V19 result. Level 5.4A
remains open and 5.5 has not started.

V20 is now a separately frozen 5.4A decoder repair on the exact V19 learned
weights. It replaces the V2 family release waypoint offsets with six
training-only V3 object medians, adds bounded supported descent during
cylinder opening, and guards small-block slot-B hand height at 0.073 m.
Its exposed 1340 development screen passed every frozen numerical gate
for epochs 20, 20 and 200: 54/60 successes each, 54/59, 54/58 and 54/58
settled nominal placements, zero recorded safety and zero independent
saved-state audit failures across 180 attempts. These exposed results do
not qualify V20. Complete 1330 V20 policy validation passed all original
numerical gates: 220/240, 212/240 and 216/240 successes, zero recorded
safety. Fresh 1380 selection then passed seed 5001 (54/60 successes, zero
physical audit failures), but rejected seed 5002 for two reported-success
pre-opening hand-table contacts and seed 5003 for one object-workspace
violation. The 1380 matrix is exposed, no three-seed V20 bundle was
selected, and 1400 one-time confirmation remains untouched. The 1360 V19
confirmation also remains unused. Level 5.4A is open; Level 5.5 has not started.

An exposed V20 follow-up with source-height floors for large-block left-bin
and small-block slot-A placements was rejected: it left pre-opening table
contact, caused long stalls and introduced premature releases. No V21
recipe or fresh qualification matrix was frozen.
Software regression after V20: 876 passed, one skipped; Ruff, V3 archive
checksum, LFS filter, and Git diff whitespace checks passed. V20 remains
unqualified because its fresh selection failed physical/safety gates.

V21 was a checksum-locked copied-state transport preview
using exact V19 learned weights and the V20 decoder. Twelve attempts in the
two exposed 1380 risk cells passed a disposable development replay and
saved-state audit; one integrated replay of the frozen implementation also
passed. Complete 1330 validation passed all original numerical gates for
three seeds (212/240, 208/240, 212/240), with zero recorded safety. Fresh
1420 selection passed all original numerical gates at 55/60, 57/60 and
54/60 successes, zero recorded safety and 180/180 independent physical
audits. Exact epochs 20, 20 and 200 were selected. One-time 1440 confirmation
rejected seeds 5001 and 5002 on the unchanged settling gate (204/228 each,
89.47%); seed 5003 passed numerically (208/228, 91.23%). All three had zero
recorded safety. Independent saved-state confirmation audit passed all 720
attempts; 1440 is exposed. This is still Level 5.4A; no place
qualification or owner manual acceptance is claimed.

V22 is now a separately frozen 5.4A repair over exact V21 weights and
decoder. It uses a bounded 2 mm supported opening descent only for held
short-cylinder placements over bins. Exposed 1330/1420/1440 diagnostics and
an integrated saved-state audit pass the targeted screen; this is development
evidence. Complete 1330 V22 validation passed every original numerical gate
at 216/240, 212/240 and 216/240 successes with zero recorded safety. Fresh
1460 selection rejected V22: seeds 5001 and 5003 settled 52/58 and 50/58
nominal placements, below the original 90% gate; seed 5002 passed numerical
gates at 53/58. Seeds 5001 and 5002 each had one reported-success
pre-opening hand-table contact on a small-block slot-B reset. The 1460
matrix is exposed, no three-seed V22 bundle exists, and 1480 one-time
confirmation is unused. Level 5.4A remains open.

V23 is a separately checksum-locked repair on exact V19 learned weights and
the V21 copied-state transport preview. It changes small-block slot-B preview
only when the baseline action predicts hand-table contact; over bins it uses
3 mm short-cylinder and 1 mm heavy-puck held opening descent above frozen
height thresholds. Disposable changed-attempt probes on complete exposed
1330/1420/1440/1460 matrices pass all original numerical gates for all seeds,
with passing saved-state audits for every changed attempt. These are
development estimates, not V23 qualification. Distinct 1500 all-cell
selection and 1520 one-time confirmation resets are frozen. Two integrated
replays, including the prior slot-B contact and puck-bin bounce cells, pass
saved-state audit. Complete 1330 V23 validation passed every original
numerical gate at 204/240, 200/240 and 204/240 successes, with 92.7%,
90.9% and 92.7% settling and zero recorded safety. Fresh 1500 selection
passed all original numerical gates at 54/60, 54/60 and 53/60 successes,
with zero recorded safety, but seed 5002 failed one independent saved-state
audit: a reported-success large-block/right-bin placement had 11 pre-opening
hand-table contact samples. V23 is rejected; 1500 is exposed, 1520 one-time
confirmation is unused, and no owner viewer review is triggered. A narrower
follow-up repair is under development inside 5.4A.

V24 now freezes the verified setup wrist during large-block/right-bin
transport on exact V19 weights and the V23 decoder. The failed V23 seed's
learned wrist rotation caused the held block to rise relative to the hand,
requiring unsafe descent before release. The source-entry hold converted
that exposed failure to a settled placement with zero pre-opening table
contact. All six exposed 1500 right-bin attempts and six 1330 right-bin
validation attempts succeed and pass physical audit; all six 1460 probes
pass audit with one counted bounce per seed. The integrated frozen V24 replay
matches its passing diagnostic actions and states exactly. New disjoint 1540
selection and 1560 one-time confirmation resets are frozen and unused.
Full V24 software regression passed 888 tests with one skip; Ruff and recipe
checksums pass. Complete V24 validation passed every numerical gate at
204/240, 200/240 and 204/240 successes, 92.7%, 90.9% and 92.7% settling,
and zero recorded safety. Fresh 1540 selection passed every numerical gate
at 53/60, 53/60 and 51/60 successes, with zero recorded safety. Its physical
audit passed 179/180 attempts: seed 5002's large-block/inspection-pad reported
success had one pre-opening hand-table contact sample. V24 is rejected,
1540 is exposed and 1560 one-time confirmation remains unused. Level 5.4A
remains open; a release-timing follow-up is under development.

V25 now freezes a geometric large-block/inspection-pad opening trigger at
the first held measured distance inside the unchanged 0.0245 m cuboid bound.
It retains exact V19 weights, V24 wrist hold, opening rate and release rise.
All six exposed 1540 pad attempts settle with passing physical audits; six
1330 pad probes preserve one counted timeout outside the bound and one
audited success per seed. The integrated V25 replay matches diagnostic
actions and states exactly. Focused tests pass 18/18 and Ruff passes. New
1580 selection and 1600 one-time confirmation resets were frozen before use;
full regression passed 890 tests with one skip. Complete V25 validation
passed every numerical gate at 204/240, 200/240 and 204/240 successes,
92.7%, 90.9% and 92.7% settling, and zero recorded safety. Fresh 1580
selection passed every numerical and physical gate at 53/60, 53/60 and
52/60 successes, zero recorded safety and 180/180 passing saved-state
audits. Exact epochs 20, 20 and 200 are packaged; selected model tensors
and manifests match the frozen V19 candidates. One-time 1600 confirmation
passes every frozen numerical gate for all three seeds: 196/240, 204/240
and 192/240 successes; 94.2%, 94.4% and 92.3% settling; worst-family
success 65%, 75% and 65%; premature release 8.33% each; zero recorded
safety violations. All 720 independent saved-state physical audits pass,
including counted failures. Artifact indexes, source/config pins, exact
candidate tensors/manifests, confirmation markers and viewer prerequisites
verify. Evidence hashes and metrics are recorded in
`docs/level5_place_repair_verification_v25.json`. Owner viewer acceptance
was confirmed on October 5, 2026 with "Passed" and is separately recorded in
`docs/level5_place_acceptance.json`. Historical automated receipts remain
unchanged. Checkpoint 5.4A is complete; 5.5 has not started.

---

## Current Branch

`main`

---

## Manual Verification Status

Level 5.7 owner manual review is pending. From the repository root in dexvision:
`python -m dexvision.apps.review_instrumented_results --evaluation-dir outputs/level5/perception_instrumented_v2/test`.
Review all 20 prescribed clips (including one nominal attempt per required skill).
Pass when selected identity/body XYZ and orientation, acquisition age, applied
action/phase and terminal result agree; unavailable/stale observations must stop
control. Fail on unreadable/missing media, swapped or inconsistent poses, stale
admitted actions, or a physical/terminal mismatch. No owner acceptance is recorded.
A manual evidence pass cannot qualify failed numerical skill gates. Last Completed
remains 5.6; Next Target remains 5.7. Do not advance to 5.8.

Level 5.6 manual verification is complete. On October 16, 2026, the owner
confirmed the prescribed V29/fresh V8 held-out review passed: 16 panels and
36 fixed-plus-worst temporal clips. All automated gates also pass.
`docs/level5_perception_acceptance.json` binds the owner statement to the exact
model, protocol, report, media manifest and automated verification receipt.
This is owner-reported acceptance; no agent visual observation or local
viewer-log completion is claimed. No manual check remains for 5.6.
Last Completed is 5.6; Next Target is 5.7, unstarted.

### Historical manual-verification records

The following paragraphs preserve each earlier handoff's then-current status;
they do not override the 5.6 completion and acceptance above.


Level 5.6 is not qualified. The owner confirmed the existing V8 validation
diagnostic review complete with "Okay done. I thought this was already done"
on October 14, 2026. `docs/level5_perception_validation_acceptance.json` binds
that statement to the model/report and 16-video/contact-sheet review manifest.
No repeat of that validation review is required. The exact reproduction
command and pass/fail criteria remain in `docs/level5_perception_repair.md`.
This diagnostic review cannot substitute for fresh independent confirmation or
full typed-schema qualification. The consumed held-out set remains blocked.
The owner said "I think mostly good" and authorized commit/push on October 14,
2026. `docs/level5_perception_review_feedback.json` binds this positive preliminary
feedback to the exact V8 media/model and remains historical evidence. Fresh
held-out manual acceptance and checkpoint completion are not claimed.

Level 5.5 manual verification is complete. The owner confirmed "Press passes"
and then "Passed" for the prescribed V8 push seed-5001 saved-state viewer
review of all four cuboid/slot-B and puck/inspection-pad interior/boundary cells.
`docs/level5_press_acceptance.json` and `docs/level5_push_acceptance.json` bind
these owner-reported statements to the exact passing bundles and evidence.
No agent visual observation or local viewer-log completion is claimed.
Historical automated receipts retain their then-pending manual status.
No manual gate remains for 5.5. Last Completed is 5.5; Next Target is 5.6,
unstarted. The latest owner instruction authorizes commit and push.

Level 5.4A V25 owner manual viewer acceptance was confirmed on October 5,
2026 with "Passed" after the prescribed seed-5001 saved-state review of one
successful attempt per planar-zone, placement-slot and receptacle target.
`docs/level5_place_acceptance.json` binds that owner statement to the exact
recipes, selected weights, confirmation reports and physical audits. All
three one-time numerical confirmations and 720/720 independent physical
audits pass. This is owner-reported acceptance; no agent visual observation
or local viewer-log completion is claimed. Historical automated receipts
retain their then-pending manual status. No manual gate remains for 5.4A.
At that October 5 acceptance, Last Completed was 5.4A and Next Target was 5.5.
Earlier failed versions and all failed attempts remain preserved.

Level 5.4 numerical gates fail for all three selected state policies, so the
conditional manual viewer review is not triggered. No manual check is
outstanding, and no owner observation or acceptance is claimed. The guarded
viewer refuses these failed reports. The later owner-requested 5.4A repair has since completed with bounded
state qualification and owner acceptance; 5.5 has not started.

Level 5.3A V7 owner manual acceptance was confirmed on October 3, 2026 with
“Manual review passed.” `docs/level5_pick_acceptance.json` binds that statement
to the three selected models, recipe, confirmation reports and physical audits.
All three numerical confirmations and 648 saved-state physical audits pass; the
known small-block drop remains a failed attempt. No manual check remains.
The owner authorized the 5.3A completion commit and push. At that handoff,
Last Completed was 5.3A and Next Target was 5.4; 5.4 has since completed with
negative state qualification. Original automated
receipts retain their historical pending status; owner acceptance is separate.

Level 5.3 is complete with automated verification and a negative qualification
result. All three selected policies fail the numerical gates, so no manual
viewer check is required or outstanding. No owner acceptance, human observation,
state qualification or registry admission is claimed. The guarded viewer path
and exact pass/fail criteria remain documented in `docs/level5_pick_results.md`;
it refuses these failed reports. All failed policies and attempts are preserved.

Level 5.2A owner manual acceptance was confirmed on September 17, 2026, following
the owner's earlier "Look okay to me" review and subsequent completion request.
The exact statements and model digests are in `docs/level5_reach_acceptance.json`.
V5 trains three seeds for 1,000 epochs and selects epochs 300, 207 and 300 using
full validation. All three pass 120/120 validation attempts and 240/240 fresh
confirmation attempts with zero safety violations and passing jerk gates.
Saved-state audit independently verifies physical pose and dwell in all 720
confirmation attempts. The full suite passes 763 tests with one skip; Ruff,
artifact integrity, original config locks and release checksum checks pass.
Exact viewer commands, pass/fail criteria, method evidence and limitations are
in `docs/level5_reach_repair.md`. The selected V5 policies pass the bounded
state-grounded reach gate, including owner acceptance. No perception or default
registry qualification is claimed. No manual check remains for 5.2A; 5.3 has
since completed with failed numerical gates and no manual acceptance. Historical
automated receipts retain their original pending manual status; the later owner
acceptance is a separate record.

Level 5.2 is complete with automated verification and an explicit negative
qualification result. Every selected policy fails the state numerical gates,
so no manual viewer acceptance is required or outstanding. No human visual
observation, owner acceptance or state qualification is claimed.

Level 5.1 is complete with automated verification only; no manual check is
outstanding. Tiny CPU training, selected-checkpoint reload, exact mid-epoch
resume, frozen-source loading, dry-run, focused tests, Ruff and full regression
checks passed. No human visual observation or learned-policy qualification is
claimed. The required manual gate for a numerically qualified reach policy
belongs to checkpoint 5.2; its policies failed the numerical gate and did not
trigger that manual check.

Level 5.0 is complete with automated verification only; no manual check is
outstanding. Level 4 release integrity and publication readiness were reverified
before starting, including all 33,350 payload files and frozen split metadata.

Level 4.9 clean-directory verification passed. The owner delegated execution,
received the passing result, and authorized completion on September 14, 2026.
The documented command and an additional normal clean-clone retrieval both
restored the expected files with matching hashes. Acceptance is recorded in
`datasets/level4-v1/completion_receipt.json`; no human visual observation is
invented. All Level 4 gates were complete at that handoff; Level 5 had not yet started.

Level 4.8 previously passed all automated audit gates without requiring manual
verification. Existing data, session entries, visual artifacts and earlier
audit reports remain preserved.

Level 4.5A manual verification passed on September 8, 2026. The user accepted
six stratified pick/place replays covering cuboid, cylinder, and flat-puck
objects plus receptacle, planar-zone, and placement-slot targets. Visible
acquisition, lift, transport, release, settling, and retraction agreed with the
saved labels. Level 4.5A is complete.

Level 4.5B manual verification passed on September 9, 2026. The user accepted
eight v19 procedural replays: one nominal and one boundary case for each of
reach, complete pick/place, push, and button press. The visible resets and goal
variation, acquisition/retention where applicable, trajectories, terminal
results, settling, and retraction matched the saved metadata. The approvals are
recorded append-only in `data/demos/level4/manual_replay_manifest.json`. Level
4.5B is complete.

Level 4.6 manual verification passed on September 9, 2026. The user confirmed
that corrected pick/place episode `level46_2091` looked good: the visible
approach miss, deterministic intervention boundary, recovery, placement,
settling, and saved metadata agreed. The approval is recorded append-only in
`data/demos/level4/correction_manual_replay_manifest.json`. Level 4.6 is
complete.

Level 4.7 manual verification passed on September 10, 2026. The user accepted
the four condition contact sheets with “Looks good.” The append-only receipt
at `data/visual/level4/manual_review_approval.json` records the confirmation and
reviewed artifact hashes. Original export-time reports remain unchanged and
retain their historical pending status. Automated gates passed with 15 focused
tests, 597 full-suite tests, Ruff, and independent file-integrity checks.
Level 4.7 is complete; its visual approval remains preserved. Level 4.8
completed its independent dataset audit on September 14, 2026.

Level 1.0 did not require manual verification.

Level 1.1 camera smoke test manually passed on June 10, 2026 with camera ID 0.
Live feed appeared, FPS was visible at about 30 FPS, and q quit cleanly.

Level 1.2 hand landmark tracking manually passed on June 10, 2026 with camera ID 0.
Landmarks, skeleton lines, anatomical left/right handedness, and confidence displayed, no-hand frames did not crash, and q quit cleanly.
Handedness labels were re-verified after correcting MediaPipe's mirrored-label convention; user's left hand now displays Left and right hand displays Right.

Level 1.3 finger feature extraction manually passed on June 10, 2026 with camera ID 0.
Feature bars updated live, open hand/fist/pointing/pinch behavior looked correct after thumb curl calibration, no-hand frames did not crash, and q quit cleanly.

Level 1.3B index curl decoupling was completed after Level 1.10 exposed feature coupling, but the user later reported that the local tuning still was not good enough because all fingers could affect each other in practice.
That index-only patch is superseded by the now-completed Level 1.3B local per-finger feature extractor replacement.
Automated replacement checks passed on June 10, 2026 using `pytest tests/test_hand_features.py`, `pytest tests/test_smoothing.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py`, and `pytest`.
Manual screenshot review on June 10, 2026 found the replacement much better: long-finger `extension` values are close to the intended live poses and are currently the most intuitive signal.
Manual review also found raw `curl` values confusing/not well-correlated enough to trust as the robot-control signal, and thumb behavior still needs separate work.
The long-finger control-signal decision has now been implemented and manually accepted: for index/middle/ring/pinky, robot control uses `bend = clamp(1.0 - extension, 0.0, 1.0)`.
Raw `curl` remains available as a diagnostic/local geometric signal, not the Level 1.10 robot-control signal.
Thumb control remains conservative and acceptable for the current through-Level-1.10 scope; fuller thumb behavior can be revisited during full-hand teleop tuning.
Automated bend-control checks passed on June 11, 2026 using `pytest tests/test_hand_features.py tests/test_smoothing.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py tests/test_index_curl_calibration.py`.
Full automated checks passed on June 11, 2026 using `pytest` with 109 passed and 19 skipped.
Manual verification passed on June 11, 2026 after the user reported the live feature viewer and one-finger MuJoCo teleop behavior were clean.
Level 1.3B is now complete.

Level 1.4 feature smoothing manually passed on June 10, 2026 with camera ID 0.
Raw and smoothed feature bars displayed live, smoothed values jittered less than raw values while remaining responsive, lost tracking did not produce NaNs, confidence drops held or decayed controls safely, and q quit cleanly.

Level 1.5 MuJoCo simple scene manually passed on June 10, 2026 using `mjpython -m dexvision.apps.check_mujoco --model assets/mujoco/simple_scene.xml --viewer --steps 600`.
The MuJoCo viewer opened, the ground grid and orange cube were visible, no camera or MediaPipe was involved, and the simulation stepped to 6.000 seconds.

Level 1.6 robot hand model load did not require manual verification.
Headless checks passed on June 10, 2026 using `conda run -n dexvision python -m dexvision.apps.check_hand_model --model assets/mujoco/hand_scene.xml`; 10 named limited joints and 10 named position actuators were discoverable, and rest stability passed.
That simple model was later preserved as `assets/mujoco/debug_hand_scene.xml` when Level 1.8B selected the Shadow Hand as the final candidate.

Level 1.7 move one robot joint manually passed on June 10, 2026 using `mjpython -m dexvision.apps.check_one_joint --model assets/mujoco/debug_hand_scene.xml --joint index_mcp`.
The MuJoCo viewer opened, the simple hand model was visible, the index MCP joint moved up while the rest of the hand stayed fixed, printed targets stayed within `[-0.150, 1.450]`, and the simulation completed 600 steps to 1.200 seconds without instability.

Level 1.8 move all robot fingers manually passed on June 10, 2026 using `mjpython -m dexvision.apps.check_hand_actuation --model assets/mujoco/debug_hand_scene.xml --gestures configs/debug_hand_gestures.yaml`.
The MuJoCo viewer opened, scripted open hand, fist, point, approximate pinch, peace sign, and relax gestures moved the fingers, printed targets respected actuator limits, and the simulation remained stable. Gesture shapes are approximate in the simple two-joint-per-finger hand model, but acceptable for the "can move all fingers" checkpoint.

Level 1.8B final hand model decision and separation manually passed on June 10, 2026.
Automated final-hand checks passed on June 10, 2026 using `conda run -n dexvision python -m dexvision.apps.check_hand_model --model assets/mujoco/hand_scene.xml --steps 20` and `conda run -n dexvision python -m dexvision.apps.check_hand_actuation --model assets/mujoco/hand_scene.xml --headless --steps-per-gesture 3 --print-interval 3`.
The final candidate is the right Shadow Hand E3M5 model from Google DeepMind MuJoCo Menagerie, vendored under `assets/mujoco/menagerie/shadow_hand/` and loaded by `assets/mujoco/hand_scene.xml`.
The original simple hand remains at `assets/mujoco/debug_hand_scene.xml` with debug gestures in `configs/debug_hand_gestures.yaml`.
Manual verification confirmed by the user after visual inspection: the Shadow Hand viewer opened, scripted open/fist/point/pinch/peace/relax gestures moved clearly enough for this checkpoint, and no instability/errors were reported. Pinch and peace remain approximate until future visual tuning.

Level 1.9 curl retargeter did not require manual verification.
Automated checks passed on June 10, 2026 using `conda run -n dexvision pytest tests/test_curl_retargeter.py`.
The retargeter maps configured `HandFeatures` control fields to final Shadow Hand actuator targets from `configs/level1_teleop.yaml`, handles missing or low-confidence features as an open hand, clips outputs to configured limits, and supports per-finger scaling.
After the Level 1.3B bend decision, long-finger mappings use `index_bend`, `middle_bend`, `ring_bend`, and `pinky_bend`; thumb remains conservative on `thumb_curl`.

Level 1.10 one-finger teleop manually passed on June 11, 2026 using `mjpython -m dexvision.apps.check_one_finger_teleop --camera-id 0 --show-camera-window --print-interval 10`.
The live camera overlay and MuJoCo viewer ran together, the robot index finger followed the derived `index_bend` signal, non-index robot fingers stayed neutral/open, tracking loss remained safe, and jitter was acceptable.
Printed diagnostics showed raw index curl, raw index extension, smoothed index extension, and index bend side by side; raw `index_curl` is diagnostic only.

Live one-finger teleoperation manual verification is complete through Level 1.10.

Level 1.11 full-hand teleoperation manually passed on June 11, 2026 using `mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 --print-interval 10`.
Automated checks passed on June 11, 2026 using `conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`, `conda run -n dexvision pytest tests/test_run_level1_teleop.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py`, and `conda run -n dexvision pytest` with 138 passed.
Manual verification was confirmed after the user reported the live full-hand teleop behavior looked good.

Level 1.12 polished demo manually passed on June 11, 2026 after the user
confirmed the demo looked good.
Automated checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision pytest tests/test_run_level1_teleop.py tests/test_level1_demo_docs.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py`,
and `conda run -n dexvision pytest` with 140 passed.
Level 1.12 was previously treated as the end of Level 1, but Level 1.13 has
been added before serious Level 2 manipulation demo work. No Level 2
implementation has been started yet.

Level 1.13 hand base/wrist pose control has been split into staged
sub-checkpoints. Level 1.13D final pose-control polish/safety is complete
after manual camera plus MuJoCo viewer verification.
Automated implementation checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision python -m dexvision.apps.check_hand_base_control --help`,
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`,
MuJoCo-adjacent hand model/actuation checks, and `conda run -n dexvision pytest`
with 149 passed.
Manual testing on June 11, 2026 found that base-control runs could stop after a
while on the instability guard (`max_abs_qvel` exceeded while qpos stayed
bounded) and that palm rotation did not visually align well for upright hand
poses. A stabilization/orientation patch now rate-limits mocap translation and
rotation, uses a lower base smoothing alpha, and changes enabled base-control
rotation to `palm_absolute` with a configurable offset.
Follow-up manual testing on June 11, 2026 found that right-hand behavior looked
better than left-hand behavior, and that wrist rotation/fist poses could still
drive the base down/sideways into the workspace clamp. A follow-up patch now
uses the wrist as the default base translation anchor, keeps palm-center
translation available only through `base_control.position_source: palm_center`,
canonicalizes anatomical left-hand orientation vectors for the right Shadow
Hand target, and raises the qvel guard only while base control is active so
bounded mocap-weld free-joint transients do not stop the app. Updated automated
checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 31 passed, a headless 720-frame MuJoCo base-control stress check
(`max_abs_qpos=1.071107`, `max_abs_qvel=60.115224`), and
`conda run -n dexvision pytest` with 160 passed.
Additional manual testing on June 11, 2026 still showed a bounded qvel guard
stop (`max_abs_qpos=0.897984`, `max_abs_qvel=152.293542`) when the left hand
entered the frame, and the user suspected stale relative mapping after tracking
loss/re-entry. A follow-up patch changes the default base translation behavior
to `base_control.position_mode: absolute`, resets base smoothing/source neutral
on tracking loss and large reacquire jumps, keeps first-valid-frame neutral
delta behavior available only through `base_control.position_mode: relative`,
and raises the base-control-only qvel guard further while preserving the
stricter finger-only teleop guard.
Updated automated checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 34 passed, a headless 900-frame tracking-loss/reacquire MuJoCo stress
check (`max_abs_qpos=1.083558`, `max_abs_qvel=58.271888`), both Level 1.13 CLI
help commands, and `conda run -n dexvision pytest` with 163 passed.

Additional manual testing on June 11, 2026 showed the mapping was still not
usable enough for the checkpoint: base x stayed nearly zero, base y changed too
little, base z was clamped near the floor, movement did not intuitively follow
the real hand, and base orientation was premature/noisy. Level 1.13 is being
simplified into staged verification. The default base-control path should now
be calibrated `image_2d` relative pose mapping: press `c` to capture neutral
palm center and neutral human palm orientation, map normalized image palm-center
dx/dy to clamped base translation, and optionally map palm-orientation deltas to
robot base orientation with `--enable-base-orientation`.
Updated automated checks for the simplified path passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 46 passed, both Level 1.13 CLI help commands, and
`conda run -n dexvision pytest` with 175 passed.
The latest update keeps orientation disabled by default but makes
`--enable-base-orientation` use calibrated relative palm orientation, adds
orientation axis signs/remap config, and maps image up/down to robot height by
default. Depth/in-out control is now handled separately by Level 1.13B using
monocular hand scale.
Updated automated checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 53 passed, both Level 1.13 CLI help commands, and
`conda run -n dexvision pytest` with 182 passed.

Level 1.13 has been split into staged checkpoints:

```text
1.13A Base x/y translation
1.13B Depth / In-Out Base Control
1.13C Relative palm rotation
1.13D Final pose-control polish/safety
```

Level 1.13A base x/y translation was manually verified by the user on
June 11, 2026: real hand left/right/up/down moved the simulated hand base as
intended, and finger teleop still worked.

Level 1.13B depth/in-out control manually passed on June 11, 2026 after the
user confirmed the depth behavior worked and the upright palm-forward neutral
start pose looked good. It uses monocular hand scale as the depth proxy, keeps
x/y translation and finger teleop working, starts from the user's intended
upright palm-forward neutral pose, and does not apply palm/wrist rotation.
Automated checks passed using `conda run -n dexvision pytest` with 193 passed.

Level 1.13C relative palm/wrist rotation implementation has automated checks
passing, and manual verification passed after the user confirmed roll, pitch,
yaw, and all three axes together worked in the live MuJoCo plus camera loop.
The user pushed this state as commit `5fe920d initial roll/pitch/yaw`.
The implementation keeps orientation opt-in behind
`--enable-base-orientation`, uses `orientation_mode: relative_palm`, lets
`--orientation-dofs roll` stage the first roll-only test, and keeps
roll/pitch/yaw signs, remap, clamps, smoothing, and deadband configurable.
Automated checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision python -m dexvision.apps.check_hand_base_control --help`,
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 73 passed, and `conda run -n dexvision pytest` with 202 passed. Ruff was
not available in the `dexvision` environment.

Level 1.13D final pose-control polish/safety manually passed after the user
confirmed the widened roll/pitch/yaw range is enough for the current scope. The
implementation improves the initial limited orientation range enough for
object-approach and table-pickup style wrist poses while keeping bounded
orientation clamps, smoothing, rate limits, tracking-loss holds, translation,
depth, and finger teleop intact. Final Level 1 full teleop with base/depth/
orientation/finger control was manually smoke-tested by the user for the
current scope with no reported crashes. Level 1 is complete.

Level 2.0 — Task Board Environment and Task Set Design did not require manual
verification. The task board environment, initial skill/task set, demonstration
requirements, and future Level 7 orchestration boundary are documented. No
demo recording, replay, learning, two-hand control, or future skill
orchestration code was implemented.

Level 2.1 — Demo Episode Schema did not require manual verification.
Automated schema checks passed on June 14, 2026 using
`pytest tests/test_dataset_schema.py`. The implementation defines the
in-memory demo episode schema, full Level 1.13 action schema validation,
observation schema validation, synthetic validation tests, and no live
recording, replay, learning, or two-hand control.

Level 2.2 — Demo Logger manually passed on June 14, 2026 after the user
confirmed the full Level 1.13-style recording path worked. The verified manual
path used `mjpython -m dexvision.apps.record_demo --task free_space_gesture
--retargeter curl --output data/demos/free_space_gesture_level113_check
--level1-13-full --overwrite`; it opened the MuJoCo viewer and separate camera
overlay, recorded full base/depth/orientation/finger teleoperation, saved the
demo arrays and metadata, and did not require a video file by default.
Automated checks passed using
`pytest tests/test_dataset_schema.py tests/test_demo_logger.py` with 20 passed,
plus the synthetic recorder smoke command. No replay, filtering, learning, or
two-hand control was implemented.

Level 2.2B — Dataset Collection Runbook and Tracker did not require manual
verification. It was completed as a docs-only checkpoint before continuing to
replay and serious dataset recording on June 14, 2026. The runbook documents
the full Level 1.13 action schema, recording naming convention, current
free-space recording command, future task command placeholders, per-task demo
targets, manual quality checklist, TODO replay/filter/summary commands, and
Level 3 dataset readiness criteria. No code, replay, filtering, learning, or
Level 7 orchestration was implemented. At that point, the next implementation
checkpoint was Level 2.3 — Demo Replay.

Level 2.3 — Demo Replay manually passed on June 14, 2026 after the user
confirmed the MuJoCo replay worked. The implementation loads saved Level 2.2
demo directories, reconstructs and validates the saved action/observation
schemas, applies full Level 1.13 `base_position_target`,
`base_orientation_target`, and `finger_actuator_targets` to the hand scene, and
supports headless or viewer replay plus replay speed control. Automated checks
passed using `pytest tests/test_replay_loader.py`, and a synthetic headless
MuJoCo replay smoke test passed. No quality filtering, dataset collection,
learning, or future checkpoint work was implemented.

Level 2.4 — Free-Space Gesture Skill Dataset manually passed on July 18, 2026.
The final raw collection contains 60 schema-valid episodes balanced at 10 each
for `open_palm`, `fist`, `point`, `pinch`, `peace_sign`, and `wave`. All 60
episodes completed headless MuJoCo replay, gesture-specific audits found
sufficiently long valid poses or wave motion in every clip, and the user
confirmed that pressing q saved the episode, closed the recorder windows, and
returned the macOS terminal prompt normally. The original
`2026-07-14_001` pinch contains extra open-hand transition footage but retains
an uninterrupted 3.97-second valid pinch and remains documented as usable raw
data. No quality filtering, object-task implementation, or learning was
started.

Level 2.4B — Repository Reproducibility Baseline did not require manual
verification. The repository now declares its runtime and development
dependencies in `pyproject.toml` and `environment.yml`, documents clean macOS
and Windows setup, tracks the roadmap/task/runbook/orchestration docs, and
ignores generated Python caches, OS metadata, operator demos, and outputs.
Generated files were removed from Git tracking without deleting the local
datasets. Automated checks passed on July 18, 2026 using
`conda run -n dexvision ruff check dexvision tests`,
`conda run -n dexvision pytest` with 240 passed, and
`conda run -n dexvision python -m dexvision.apps.health_check`.

Level 2.4C — Executable Observation Layout Contract did not require manual
verification. The versioned Level 2 observation schema now provides executable
dense-array mappings, ordered MuJoCo joint and actuator names, semantic units
and coordinate frames, explicit optional-state absence rules, and a legacy
Level 2.4 replay compatibility adapter. Automated checks passed on July 18,
2026 using `conda run -n dexvision pytest tests/test_dataset_schema.py
tests/test_demo_logger.py tests/test_replay_loader.py` with 39 passed.

Level 2.5 — Task Board and Reach-Touch Task did not require manual
verification. The shared MuJoCo task-board scene now provides three reachable
named target sites, and `reach_touch_target` provides typed goal parameters,
seeded deterministic reset, reconstructable task/initial state, fixed
distance-and-dwell success metrics, workspace/timeout failures, and a headless
CLI smoke check. Automated checks passed on July 18, 2026 using
`conda run -n dexvision python -m dexvision.apps.check_task --task
reach_touch_target`, `conda run -n dexvision pytest tests/test_task_specs.py`
with 9 passed, `conda run -n dexvision ruff check dexvision tests`, and
`conda run -n dexvision pytest` with 256 passed.

Level 2.6 — Reach-Touch Pilot Demonstrations manually passed on July 18, 2026.
The final pilot contains five schema-valid, operator-reviewed episodes across
the right, center, and left targets. Saved task state confirms five consecutive
physical palm-contact frames and computed success in every episode, all five
episodes completed semantic headless replay against their recorded target, and
the user confirmed that all five viewer replays showed the intended skill
behavior. Automated checks passed using `conda run -n dexvision pytest` with
272 passed and `conda run -n dexvision ruff check dexvision tests`.

Level 2.6B — Reach-Touch Success Relabeling did not require manual
verification. The task-specific relabeler recomputes target distance and
consecutive qualifying palm-contact frames from saved task state, preserves
operator and recomputed labels together in a dataset-level JSON report, and
does not rewrite raw episode files. The five pilot episodes all recomputed as
successful and agreed with their operator labels. Automated checks passed on
July 18, 2026 using `conda run -n dexvision python -m
dexvision.apps.relabel_demos --dataset data/demos/raw/reach_touch_target`,
`conda run -n dexvision pytest tests/test_success_relabeling.py` with 8 passed,
`conda run -n dexvision ruff check dexvision tests`, and `conda run -n
dexvision pytest` with 280 passed.

Level 2.7 — Pilot Quality Filters did not require manual verification. The
read-only filter evaluates tracking confidence, missing frames, feature
jitter, action jerk, actuator-limit hits, recomputed task failure, and
workspace-limit hits using versioned configurable thresholds. It saves a
dataset-level JSON report grouped by skill and task without rewriting raw
episodes. The five reach-touch pilots produced four passes and one intentional
quality flag: `2026-07-18_003` exceeded the default action-jerk threshold
(`0.260570` versus `0.200000`). Automated checks passed on July 18, 2026 using
`conda run -n dexvision python -m dexvision.apps.filter_demos --dataset
data/demos/raw/reach_touch_target`, `conda run -n dexvision pytest
tests/test_quality_filters.py` with 8 passed, `conda run -n dexvision ruff
check dexvision tests`, and `conda run -n dexvision pytest` with 288 passed.

Level 2.7B — Reach-Touch Dataset Summary did not require manual verification.
The read-only summary scans the documented `data/demos/raw/` layout, groups
episodes by skill and task, reports success, episode length, tracking
confidence, action/observation schema versions, quality pass/fail counts, and
relabel disagreements, and saves JSON plus CSV outputs under
`data/demos/reports/summaries/`. The reach-touch pilot summary reports five
recomputed successes, four quality passes, one quality failure
(`2026-07-18_003` for high action jerk), and zero relabel disagreements.
Automated checks passed on July 18, 2026 using `conda run -n dexvision python
-m dexvision.apps.summarize_demos --dataset data/demos`, `conda run -n
dexvision pytest tests/test_dataset_summary.py` with 7 passed, `conda run -n
dexvision ruff check dexvision tests`, and `conda run -n dexvision pytest`
with 295 passed.

Level 2.7C — Scale Reach-Touch Dataset manually passed on July 18, 2026 after
the user completed the live collection and asked for the final dataset gates
to be applied. The immutable raw collection contains 76 episodes, of which 69
recompute as successful and 55 pass every Level 2.7 quality filter. Clean
training coverage is balanced across `reach_target_left` (18),
`reach_target_center` (18), and `reach_target_right` (19), with zero
operator/recomputed label disagreements. All 76 episodes validated and
completed headless replay for 9,540 recorded action steps. The versioned
`configs/reach_touch_dataset.yaml` split reserves
`reach_eval_left_center` at `[0.14, -0.05, 0.47]` metres and
`reach_eval_center_right` at `[0.14, 0.03, 0.50]` metres exclusively for
Level 3 evaluation. The v2 JSON/CSV summary reports the target distribution,
55 clean successes, full relabel/quality coverage, and `level3_ready: true`.
Existing failed raw attempts remain immutable for audit; the quality-gated
collection helper now balances clean successes and keeps new rejected attempts
outside `raw/`.
Automated checks passed on July 18, 2026 using the full 76-episode validation
and headless replay audit, `conda run -n dexvision python -m
dexvision.apps.relabel_demos --dataset data/demos/raw/reach_touch_target`,
`conda run -n dexvision python -m dexvision.apps.filter_demos --dataset
data/demos/raw/reach_touch_target`, `conda run -n dexvision python -m
dexvision.apps.summarize_demos --dataset data/demos`, `conda run -n dexvision
ruff check dexvision tests`, and `conda run -n dexvision pytest` with 303
passed.

Level 2.7D — Button-Press Task did not require manual verification. The shared
task-board scene now contains three passive, bounded spring-return buttons, and
`button_press` provides typed button/depth/pressed-state/approach parameters,
seeded deterministic reset, reconstructable task and initial state, explicit
terminal-state fields, and recomputable press-depth-and-dwell success metrics.
No button demonstrations, cube-push task, or learning code was added.
Automated checks passed on July 18, 2026 using `conda run -n dexvision pytest
tests/test_task_specs.py tests/test_success_relabeling.py` with 27 passed,
`conda run -n dexvision ruff check dexvision tests`, the existing reach-touch
headless smoke check, and `conda run -n dexvision pytest` with 313 passed.

Level 2.7E — Button-Press Pilot manually passed on July 18, 2026 after the user
recorded five bright-green-target attempts and confirmed the replacement center
take was clean. The retained pilot covers all three configured buttons and
target press depths from 0.010 to 0.014 metres. All five episodes validated,
completed headless replay, recomputed as successful, passed quality filtering,
and had zero operator/recomputed label disagreements. A target-isolation audit
confirmed the selected button was the primary press in every retained episode;
the replacement center episode pressed only its target button. Reach-touch
fixtures remain hidden and non-colliding only for button recording/replay.
Automated checks passed with 318 tests. No dataset scale-up, cube-push task, or
learning work was added.

Level 2.7F — Push-Cube Task did not require manual verification. The shared
task-board scene now provides a free-joint cube, deterministic start and target
selection, typed object/target/approach parameters, reconstructable current and
initial object pose/velocity plus target and robot state, explicit terminal
fields, and a recomputable planar distance-and-dwell success metric. Existing
reach and button tasks isolate the cube fixture. No cube demonstrations,
dataset relabel dispatch, or learning code was added. Automated checks passed
on July 19, 2026 using `conda run -n dexvision pytest
tests/test_task_specs.py tests/test_success_relabeling.py` with 37 passed,
`conda run -n dexvision ruff check dexvision tests`, and `conda run -n
dexvision pytest` with 328 passed.

Level 2.7G — Push-Cube Pilot is complete.
Recorder, saved cube/object state, semantic replay restoration, planar
distance-and-dwell relabeling, quality filtering, and summary integration have
automated coverage. Checks passed on July 19, 2026 using the checkpoint-focused
suite, Ruff, and the full suite. A follow-up usability correction on July 19,
2026 hides the vertical wall/unrelated fixtures, aligns the hand free joint and
mocap neutral immediately behind the cube, and disables nominal
task-timeout enforcement during live recording. The unusable
`2026-07-19_001` trial had no cube motion and was deleted at the operator's
request. Updated focused checks passed with 73 tests and the full suite passed
with 341 tests. A replacement `2026-07-19_001` tuning attempt had 100% hand
detection and contacted the cube, but its commanded base range was too weak
and the cube moved away from the target. That inspected attempt was also
deleted under the operator's authorization. The cube profile now uses stronger
translation/depth gains, faster smoothing response, a smaller depth deadband,
and a bounded larger position step. A subsequent `2026-07-19_002` attempt
showed that the remaining perceived delay was MuJoCo cadence: the generic
2-step loop advanced 0.004 seconds of simulation per roughly 0.033-second
camera frame, leaving 3.3 cm mean and 7.4 cm p95 physical-hand follow error.
Cube recording now advances at least one nominal control period per frame
(17 steps at 30 Hz with the current model), reducing a headless final follow
error from about 2.5 cm to 2 mm. That inspected failed attempt was deleted.
Another post-cadence `2026-07-19_002` attempt showed responsive tracking but a
control-frame mismatch: the simulated palm was horizontal while the operator's
real palm was upright. The cube traveled 27 cm and lifted more than 10 cm,
consistent with an abrupt scoop rather than remaining latency. Cube recording
now starts the robot palm upright to match calibration, moves its neutral
slightly farther behind the cube for a stable reset, and uses moderate
position/depth/orientation damping with a smaller per-frame step cap. The
inspected failed attempt was deleted under the operator's authorization.
Updated focused checks pass with 74 tests and the full suite with 342 tests.
A subsequent 410-frame `2026-07-19_002` trial had 100% hand detection, 0.949
mean tracking confidence, and only 2.3 mm mean physical follow error. The
remaining difficulty was geometric: the operator's palm faced the webcam, but
the simulated fingers rather than the palm normal pointed along the push axis,
and the cube moved 13.3 cm backward. Cube reset now uses a camera-facing,
fingers-up palm, maps webcam-relative depth along the palm normal, and starts
the viewer at a clear three-quarter angle. Palm/finger contacts remain active;
only forearm/wrist collisions below the tabletop are disabled. A headless
regression moves the cube about 18 cm along the table with less than 1 mm
vertical disturbance. Updated focused checks pass with 76 tests, Ruff passes,
and the full suite passes with 344 tests. The inspected failed trial was
deleted under the operator's standing authorization.
A further 418-frame `2026-07-19_002` trial exposed a reversed palm-facing reset
plus the Level 1.13 full-control preset re-enabling live base orientation.
Rendered frames showed the back of the hand at reset, the robot rotate out of
view after about 1.2 seconds, and later reappear sideways on the table. Cube
mode now uses the actual palm-facing identity quaternion and forcibly locks
base orientation while retaining translation, webcam-relative depth, and
finger control. The saved effective teleop snapshot records the resolved
control enable flags. A fixed-orientation headless push moves the cube about
21 cm along the table with less than 1 mm vertical displacement, and the
focused suite passes 76 tests.
A subsequent 756-frame `2026-07-19_002` trial kept the command orientation
fixed but exposed geometry clipping: the visible forearm/wrist crossed the
table at reset, and 15 cm of enabled image-height motion drove the palm into
the table. Contact deflected the physical hand sideways, with 3.6 cm mean and
11.6 cm p95 follow error plus cube lift. Cube mode now hides/disables the
below-table support bodies, fixes hand height, tightens forward/lateral bounds,
reduces depth gain, and ignores real-hand up/down. Named targets use the
matching cube-start lane for a straight planar push. A headless bounded push
ends 8.6 mm from target centre with 0.02 mm vertical drift, fixed orientation,
and 2.2 mm base-follow error. The focused suite passes 76 tests.
On September 2, 2026, `2026-07-19_004` was audited as a clean successful
127-frame planar push: 100% detection, 0.958 mean tracking confidence, 1.4 mm
mean follow error, fixed orientation, 1.1 mm maximum cube lift, successful
five-frame target dwell, clean rendered frames, headless replay, and a quality
pass. It is retained as immutable diagnostic/reference data but does not count
toward the five accepted pilots because the old recorder saved a null operator
label. Push-cube now uses the shared explicit operator-label prompt, and the
quality filter accepts intentionally fixed workspace axes without counting
them as limit hits. Focused checks pass with 79 tests, Ruff passes, and the full
suite passes with 347 tests.
The labeled `2026-09-02_001` retry is visually clean and agrees with recomputed
success: 100% detection, 0.964 confidence, 1.15 mm mean follow error, fixed
orientation, 0.72 mm cube lift, and 2.58 cm final target distance. It remains
raw diagnostic data rather than an accepted pilot because 48 of 147 frames
(32.7%) saturated the left workspace boundary, exceeding the 10% quality
limit. The target-aware lateral workspace now provides 7 cm of steering margin
on each side of the selected start/target lane while preserving fixed height,
orientation, and forward safety limits. Focused checks pass with 79 tests.
The labeled `2026-09-02_002` retry is also visually clean and agrees with
recomputed success: headless replay, 100% detection, 0.960 confidence, 1.09 mm
mean follow error, fixed orientation, 0.60 mm cube lift, and 2.49 cm final
distance all pass. It narrowly fails only the workspace gate because 12.8% of
frames saturated the lateral boundary, above the 10% maximum. It is retained
as diagnostic data. Since named starts and targets are lane-aligned, lateral
position is now fixed to the selected lane together with height/orientation;
only palm-normal forward motion controls the pilot. Focused checks pass with
79 tests.
`2026-09-02_003` is accepted as pilot 1/5 for `push_target_left`: its operator
label and recomputed success agree, headless replay and all quality gates pass,
tracking was 100% at 0.956 mean confidence, mean follow error was 1.44 mm,
maximum cube lift was 0.59 mm, and it ended 2.76 cm from target centre after
the required five-frame dwell. Rendered frames show a clean clipping-free
planar push. The diagnostic `2026-07-19_004`, `2026-09-02_001`, and
`2026-09-02_002` directories were deleted at the operator's request, and the
reports were regenerated. Push-cube summary aggregation now reports target
distribution and one clean success. Four more accepted pilots covering all
three target lanes remain required. Focused checks pass with 79 tests.
Four additional recordings (`2026-09-02_001`, `2026-09-02_002`,
`2026-09-02_004`, and `2026-09-02_005`) were audited on September 2. Together
with `2026-09-02_003`, all five have affirmative operator labels, recomputed
success, zero label disagreement, successful headless replay, and quality
passes. Tracking was present for 100% of every episode, mean confidence ranged
from 0.956 to 0.977, mean base-follow error ranged from 1.44 to 2.33 mm,
maximum cube lift ranged from 0.35 to 0.74 mm, and final planar target distance
ranged from 2.09 to 2.76 cm. Rendered final frames are clean. However, all five
recordings use `push_target_left`, so they do not satisfy the required coverage
of all three target zones. Retain the strongest three left-lane episodes
(`2026-09-02_002`, `2026-09-02_003`, and `2026-09-02_005`) and record one
centre and one right episode. At the operator's request, the redundant
left-lane `2026-09-02_001` and `2026-09-02_004` recordings were deleted and
the reports regenerated; the retained set now contains three clean left-lane
episodes awaiting those two replacements.
The replacement `2026-09-02_006` centre and `2026-09-02_007` right episodes
both pass schema validation, headless replay, relabeling, every quality gate,
and rendered start/middle/final inspection. They have 100% tracking, mean
confidence of 0.954 and 0.933, mean base-follow error of 1.32 and 2.22 mm,
maximum cube lift of 0.63 and 0.48 mm, and final planar target distances of
2.56 and 2.70 cm, respectively. The retained pilot now contains exactly five
clean successes with target distribution left=3, centre=1, right=1, zero label
disagreements, and five quality passes. During verification, replay was found
to default to one MuJoCo step per action even though these recordings saved 17
steps per frame; that default did not reproduce the contact dynamics. Replay
now automatically uses the saved cadence, retains an explicit override, and
falls back to one step for legacy demos. The focused checkpoint suite passes
with 57 tests, Ruff passes, and the full suite passes with 350 tests.
The exactly-five pilot, three-zone coverage, and pipeline-report requirements
are satisfied. On September 2, 2026, the user manually replayed all five
retained episodes and confirmed that their saved cube starts, target markers,
planar pushes, and final target behavior looked correct. Level 2.7G is complete.

Level 2.7H — Scale Button-Press Dataset manually passed on September 2, 2026
after the user completed 50 quality-gated live recordings. Together with the
five retained pilots, the immutable raw dataset contains 55 episodes. All 55
validated and completed semantic headless replay for 5,475 action steps,
recomputed as successful, passed every quality filter, and had zero label
disagreements. The nine training button/depth goals contain six or seven clean
successes each, three interpolated button/depth states remain held out, and the
v3 dataset summary reports `level3_ready: true`. Level 2.7H is complete.
Level 2.7I — Scale Push-Cube Dataset manually passed on September 2, 2026 after
the user completed the live quality-gated collection. The immutable raw dataset
contains 101 episodes and 7,176 action frames. All 101 episodes validated and
completed semantic headless replay using their saved 17-step cadence; all 101
recomputed as successful, passed every quality filter, and had zero label
disagreements. Clean goal coverage is left=33, centre=34, and right=34. Three
interpolated cube start/target-pose states remain held out, and dataset-summary
v4 reports `level3_ready: true`. Focused checks pass with 27 tests, Ruff passes,
and the full suite passes with 362 tests. Level 2.7I is complete.

Level 2.7J — Optional Skill Card Export Metadata did not require manual
verification. Policy-free JSON metadata stubs can be exported for the three
implemented task skills from their task specs and matching dataset-summary
groups. The stubs declare the full Level 1.13 action layout, typed parameters,
transition and terminal-state contracts, dataset provenance/readiness, and
known limitations while leaving the policy checkpoint unset for Level 3.
Automated checks passed on September 2, 2026 using `conda run -n dexvision
pytest tests/test_skill_card_metadata.py` with 7 passed, real CLI exports for
all three skills against the v4 dataset summary, `conda run -n dexvision ruff
check dexvision tests`, and `conda run -n dexvision pytest` with 369 passed.
Level 2.7J is complete.

Level 2.8 — Retargeter B: Fingertip Target Baseline did not require manual
verification. The baseline computes palm-local, palm-width-normalized fingertip
targets from MediaPipe-compatible landmarks, maps them to bounded Shadow Hand
targets with a simple geometric solve, and falls back to the last valid or safe
open targets when solving fails. Automated checks passed on September 2, 2026
using `conda run -n dexvision pytest tests/test_fingertip_retargeter.py` with 7
passed, `conda run -n dexvision ruff check dexvision tests`, and `conda run -n
dexvision pytest` with 376 passed. Level 2.8 is complete.

Level 2.9 — Retargeter C: Optimization Retargeter did not require manual
verification. The bounded solver minimizes palm-local 3D fingertip error,
configured actuator-limit violations, and change from the previous finger
controls. It uses SciPy L-BFGS-B when available, has a deterministic projected-
gradient path when SciPy is absent, clips every output to configured limits,
falls back to last-valid or safe-open targets, and records solve diagnostics.
Automated checks passed on September 2, 2026 using `conda run -n dexvision
pytest tests/test_optimization_retargeter.py` with 7 passed, the three-
retargeter suite with 30 passed, `conda run -n dexvision ruff check dexvision
tests`, and `conda run -n dexvision pytest` with 383 passed. Level 2.9 is
complete.

Level 2.10 — Retargeting Benchmark did not require manual verification. The
benchmark compares curl, fingertip, and optimization retargeters on identical
saved episode streams, saves JSON/CSV metrics and a dependency-free SVG plot,
and recomputes push-cube task success through counterfactual headless MuJoCo
replay. The September 2, 2026 10-episode baseline completed successfully;
the focused metric suite passed with 5 tests, Ruff passed, and the full suite
passed with 388 tests. Level 2.10 is complete. No Level 3 policy learning was
implemented.

Level 2.10B — Full-Dataset Retargeting Benchmark Validation did not require
manual verification. All 101 immutable push-cube episodes and 7,176 frames
were evaluated for curl, fingertip, and optimization retargeters. Versioned
JSON/CSV output now includes deterministic 95% episode-bootstrap intervals;
the SVG contains six metric panels; and headless MuJoCo replay measures actual
distal-fingertip-to-cube distance and contact-frame rate alongside task
success. The README reports the full results and explicitly discloses that the
shared base trajectories were originally collected with curl retargeting, so
independent live per-retargeter trajectories are not claimed. Focused checks
passed with 6 tests, Ruff passed, and the full suite passed with 389 tests.
Level 2 is complete. No Level 3 policy learning was implemented.

Level 2.11 — Learning Readiness Freeze did not require manual verification.
The completed Level 2 raw datasets, rejected attempts, and reports are packaged
as one immutable Git LFS archive with a SHA-256 checksum and release manifest;
the editable `data/demos/` working tree remains ignored. The first reach-policy
offline split, rollout-only held-out targets, initial-state perturbations, and
numerical acceptance gates are frozen in
`configs/level3_evaluation.yaml`. Existing episodes were not rewritten to
invent session ids, and the protocol explicitly forbids a cross-session claim.
No policy, training loop, rollout evaluator, VLM, or planner was implemented.

Level 3.0 — Roadmap Rebaseline did not require manual verification. It
established Level 3 as a bounded learning-feasibility phase and introduced a
later comprehensive data campaign. Level 3.0B corrects the later separation:
Level 4 owns the comprehensive multi-session dataset and immutable release;
Level 5 owns full-scale skill learning, qualification, supervised runtime, and
diverse scripted pilots; Level 6 owns portfolio polish, robustness, and
reproducibility; and Level 7 is future language-guided orchestration. Neither
docs-only checkpoint modifies completed Level 1/2 work or dataset contents.

Level 3.1 — Goal-Conditioned Per-Skill Dataset Loader did not require manual
verification. Automated checks passed on September 3, 2026 with 37 focused
tests and Ruff. The verified Level 2 release checksum passed, all three clean
skill datasets loaded successfully, and the frozen reach split produced 43
training and 12 validation episodes. A full-suite run excluding the unrelated
in-progress roadmap-doc test passed with 399 tests.

Level 3.3 — Behavior-Cloning Training Loop did not require manual verification.
Automated checks passed on September 3, 2026 with 18 focused tests, Ruff, and
the full 415-test suite. The configured 100-epoch CPU training command also
completed on the extracted immutable reach dataset, and its checkpoint digest
verified. No camera, GUI, live teleoperation, or MuJoCo rollout was involved.

Level 3.4 — Frozen Reach Closed-Loop Rollout did not require manual
verification. Automated checks passed on September 3, 2026 with 7 focused
tests, Ruff, and the full 420-test suite. The real checkpoint completed all 35
frozen headless scenarios and saved all 17 successes and 18 failures. It failed
the frozen success and safety gates without any held-out retuning; the exact
results are recorded in `outputs/level3/reach_rollout_v1/report.json`.

Level 3.5A — Button and Push Evaluation Freeze did not require manual
verification. Automated checks passed on September 3, 2026 with 8 focused
tests, 59 evaluation/dataset/task tests, Ruff, and the full 431-test suite. The
two v1 protocols preserve all reserved Level 2 conditions and are frozen before
any button or push policy training.

Level 3.5B — Checkpoint-Selection Repair and Cross-Task Baselines did not
require manual verification. Automated checks passed on September 3, 2026 with
23 focused tests, Ruff, and the full 437-test suite. Fixed-seed reruns reproduced
all best/last checkpoint digests. Corrected reach, button, and push completed
their unchanged frozen matrices and all failed their numerical gates; all
rollout failures and the reach v1-versus-v2 comparison remain in versioned
reports under `outputs/level3/`.

Level 3.6 — Data and Action-Space Diagnostics did not require manual
verification. Automated checks passed on September 3, 2026 with Ruff and the
full 442-test suite. The real CPU/headless matrix completed all 13 report rows
under the frozen task protocols. All common episodes retained their baseline
split in the broader reach comparison, and button/push correctly record that no
broader eligible counterpart exists. The versioned report is under
`outputs/level3/diagnostics_v1/`.

Level 3.7 — Conditional Temporal Baseline did not require manual verification.
The evidence trigger was assessed and not met: prior metrics do not measure
temporal ambiguity or isolate compounding error, and Level 3.6 keeps
compounding error and missing recovery coverage explicitly categorized as
hypotheses. The versioned decision records the measured action-space, safety,
and offline-to-rollout evidence. No sequence dataset, GRU, or temporal training
run was added. Automated checks passed on September 3, 2026 using `conda run -n
dexvision pytest -q tests/test_level3_temporal_decision.py
tests/test_roadmap_docs.py` with 12 passed, `conda run -n dexvision ruff check
dexvision tests`, and `conda run -n dexvision pytest -q` with 445 passed.

Level 3.8 — Feasibility Report and Level 4 Data Requirements did not require
manual verification. The report traces the negative full-action baselines,
checkpoint-selection change, controlled diagnostics, and temporal decision to
their configs, splits, datasets, schemas, checkpoints, protocols, and report
digests. It closes Level 3 with a no-go on policy qualification and a go on the
existing Level 4 workcell-data plan. Automated checks passed on September 3,
2026 using the focused Level 3.8/temporal/roadmap suite with 18 passed, Ruff,
and the full suite with 451 passed.

Level 4.0 — Workcell and Dataset Requirements Freeze did not require manual
verification. The frozen plan and YAML specification define the workcell/data
scope, typed skill and terminal-metric contracts, 79 split-owned coverage
cells, four whole-session split slots, reconstructable action-safety records,
causal online phases, fixed-camera visual conditions, quality gates, and Level
3 failure traceability. Automated checks passed on September 3, 2026 with 9
focused tests, Ruff, and the full 461-test suite. No Level 4 collection or
workcell implementation was started.

Level 4.1 — Workcell Scene, World State, and Task Contracts manually passed on
September 3, 2026. Automated checks passed with 14 focused tests, the
1,200-step headless inspector, repository-wide Ruff, and the full 475-test
suite. The first manual
attempt failed because the fixed camera was locked on an obstructed view and
the viewer exited normally after its hard 1,200-step limit. The corrected
inspector uses a movable overview camera, stays open until explicitly closed,
and exposes keyboard reset, seed, pause, and label controls. The checkpoint is
now also displaying workcell-only entity labels by default after follow-up
review found the unlabeled colors ambiguous and showed that MuJoCo hid the
original label anchors in disabled site group 4. The corrected anchors use
visible site group 0. A later review found that the overview visually reversed
the two return-bin sides and that objects could obscure the setup-slot markers. The
user confirmed the object/setup-slot issue was fixed, but camera-only changes
failed left/right re-verification because the frozen target coordinates encoded
the names opposite to the operator-facing view and the first camera regression
used a reversed horizontal-vector sign. The frozen target centers and MuJoCo
bodies are now corrected together so ids, labels, world state, and metrics
agree. The regression checks frozen-to-runtime coordinate equality and actual
screen-space ordering. The user confirmed that the final viewer passed, with
correct operator-facing return-bin sides and no object/setup-slot overlap.
Level 4.1 is complete.

Level 4.2 — Session-Aware Recording and Phase-Label Schema did not require
manual verification. New Level 4 episodes preserve genuine session/operator
provenance, typed goal/reset metadata, requested/commanded/applied and prior
actions, per-field safety masks/reasons, causal online phases, audited labels,
phase intervals, synchronized state/action/task timestamps, and optional
aligned RGB. Session manifests reject duplicate ids; resumed allocation and
atomic Level 4 writes never overwrite existing episodes. Legacy Level 2
episodes still load without invented Level 4 fields. Automated checks passed
on September 4, 2026 with 31 focused tests, repository-wide Ruff, a validator
module smoke run against a synthetic episode, and the full 489-test suite.
Level 4.2 is complete; Level 4.3A is the next target within the revised Level
4.3 sequence.

Level 4.3 — Collection Pilot and Final Coverage Freeze has automated pilot
recording, review, replay, and coverage-report tooling implemented, but remains incomplete. The
tooling keeps expert acceptance evidence append-only, retains ordinary failures
separately, reports episodes and derived pick/place segments separately, audits
session/cell split ownership, measures phase agreement and collection/storage
statistics, and leaves the count matrix provisional. The workcell recorder now
resolves frozen cells for reach, complete pick/place, push, and press, creates a
genuine append-only session per process, stores all six object states and task
metrics, emits online phases, and restores randomized objects for replay. The
optional dial is explicitly deferred. Automated checks passed on September 4,
2026 with 18 checkpoint tests, 86 controller/workcell regression tests,
repository-wide Ruff, recorder-help and coverage CLI smoke commands, and 508
full-suite tests.
The first manual reach attempt exposed a workcell control-frame defect: the
long forearm root was being positioned as if it were the palm, leaving the hand
cramped against the fixtures. That rejected episode was removed at the user's
request. The weld now offsets the forearm outside the board, reset aligns the
free joint without a transient, the recorder calibrates at the workcell neutral
palm pose, and Level 4 attempts request an explicit operator label. A second
genuine reach attempt was retained as an ordinary failure after it showed that
copying robot palm orientation was still ergonomically infeasible. Reach now
uses an upright, webcam-facing human neutral as a translation clutch, locks the
robot's palm-down orientation, displays a cyan target marker, and uses a closer
neutral plus higher translation gain so the first target does not require
near-edge camera travel. The third reach attempt held the palm within the
distance and orientation gates for 36 frames but displaced two neighboring
objects by roughly 7 cm because the marker was too low in the crowded scene.
The reach marker is now a collision-free pre-grasp target at 0.148 m, live task
state preserves the scene-disturbance metric, and an automated regression
proves five qualifying frames trigger recorder stop. The fourth attempt then
exposed a timing defect: at 30 camera FPS the recorder advanced only one 2 ms
physics step per frame, making visible hand motion about 17 times slower than
real time. It reached only 0.0745 m from the marker and disturbed the scene by
0.0218 m. Workcell recording now enforces 17 physics steps per frame, uses more
responsive filtering, and permits larger safe per-frame target motion. The
fifth retained attempt showed the timing fix was visibly better and entered the
distance gate for 32 frames, but low translation gain forced near-edge camera
travel and the wandering path accumulated 0.0223 m scene disturbance. The next
attempt came no closer than 0.0400 m and displaced `block_large` by 0.1728 m,
confirming that absolute monocular position mapping is not usable in this
cluttered scene. The pending manual trial replaces it with a centered, nonlinear
Cartesian velocity joystick: returning the hand to center stops the robot, a
high transit plane prevents lateral travel through clutter, and descent is
permitted only in a corridor over the target. The actual target object now has
a bright emissive magenta wireframe outline; the cyan cross separately marks
the desired palm position. `--workcell-dry-run` tests this without retaining an
episode or modifying the session manifest. At the user's request, the five
retained failed attempts and their manifest/report were removed from the active
pilot directory and parked recoverably at
`/private/tmp/dexvision_level4_pilot_pre_reset_20260904_001`; the active counter
was reset for `pilot_train_001/episode_000001`. Four clean retained trials now
provide the pivot evidence. Reach recomputed and visibly passed at 0.0162 m
terminal error with the required five-frame dwell and about 0.0021 m maximum
scene disturbance. Pick/place never qualified, moved `block_small` about
0.0486 m and a neighboring cylinder about 0.0450 m, and confirmed that contact
control is not usable through the current mapping. Push never moved the selected
block toward its 0.2884 m-distant target. Button press remained at zero depth;
the commanded base stayed near x=-0.18..-0.16 m while the wall button is near
x=+0.13 m. The user manually judged reach usable, pick/place and push unusable,
and button potentially viable only with a reachable planned approach. No expert
acceptance sidecars have been created, so the collection protocol, final count
freeze, and required visible replay checks remain pending.

The architecture decision is now made: live hand-pose control is retired from
the active workcell pipeline, while deterministic simulator-state experts are
the source for every required skill and scripted correction. Level 4.3 is split
into 4.3A–4.3I:
common expert plus scripted reach; button; constrained push; grasp-and-lift;
place/complete pick-place; replay qualification; small state-only button and
push learnability probes; and final source/count freeze. The first learned
interface is a low-dimensional task-local delta expanded by deterministic code
into the unchanged full action schema. Do not add RGB, an LLM/VLM, a larger
model, a general planner, action chunking without evidence, or a bulk haul.
Do not mark Level 4.3 complete or advance to Level 4.4 until all lettered
checkpoints and visible replay gates pass.

Level 4.3A is complete. `dexvision/sim/level4_expert.py` provides the common
deterministic reset/step boundary and a configuration-owned four-segment reach
through rise, horizontal transit, protected corridor entry, and descent. Every
candidate action is first exercised in copied MuJoCo state and checked for
workspace, joint-limit, table/fixture-contact, and non-target-disturbance
failures. Scripted reach uses the existing `record_demo` logger and unchanged
27-field Level 4 action schema; requested, commanded, and applied rows are
preserved with zero safety interventions, and replay restores the same fully
centered object cage and goal cross. Automated checks passed on September 4, 2026 with 8
focused tests over five randomized seeds, 58 related regression tests,
repository-wide Ruff, and the full 517-test suite. The user confirmed the
corrected visible pre-grasp replay, which advanced work to Level 4.3B.

The first visible replay review initially remained pending because `--speed 0.1`
displayed only about three action updates per second and exposed a real target
cue defect: the cage began at the block center and did not enclose its lower
half. The cage geometry is now centered vertically around the selected entity,
with an automated enclosure regression. The user then accepted the corrected
pre-grasp reach behavior and confirmed that stopping at the cyan cross without
cube interaction was the intended Level 4.3A result.

Level 4.3B is complete. `DeterministicButtonPressExpert` uses a fixed neutral
hand pose and orientation, a task-relative safe approach, button-normal contact
motion, the frozen press-depth/dwell metric, and a complete retract/release
phase. The button was moved into the declared safe workspace after the pilot
proved its prior height unreachable. Copied-state validation and five
randomized record/replay checks reject unintended contacts and meaningful joint
limit excursions; all retained scripted samples have zero safety masks,
reasons, and interventions. The listed 2-test checkpoint suite and 38 related
regression tests passed on September 4, 2026. No manual verification was
required. This advanced work to Level 4.3C.

Level 4.3C visible replay remediation is complete. The first
visible cuboid replay failed review because the block tipped about 45 degrees,
briefly crossed the distance threshold, and rolled back while the recorder
retained the transient success. The remediated controller uses a lower,
partially flexed cuboid index posture, parks non-target objects for standalone
skill trials, and requires table support, at most 10 degrees of tilt, and a
still-qualified metric for five samples after hand release. Five cuboid and
flat-puck resets pass copied-state validation and replay; the 2-test checkpoint
suite, the 33-test combined expert/collection suite, repository-wide Ruff, and
the full 524-test suite pass. The user confirmed on September 4, 2026 that the
new cuboid push slides cleanly without tipping; the accompanying button and
standalone grasp replays also passed visual review.

Level 4.3D is complete. `DeterministicGraspLiftExpert` uses separate
configuration-owned cuboid, cylinder, and flat-puck object-relative templates,
fixed wrist orientations, deterministic retargeter open/full-flexion endpoints,
and one scalar closure synergy per family. Standalone trials retain the selected
object's seeded workcell pose while parking non-target objects on the lower
floor, isolating grasp physics from the pilot-confirmed clutter limitation
without removing objects from the schema. Copied-state and live qualification
require initial table support, a measured two-contact held relation, at least
0.040 m of lift, loss of table support, retention, and ten stable samples below
0.020 m/s. Three seeds per family record and replay deterministically with
planar neighbor disturbance below 0.005 m and zero logged safety interventions.
The listed 2-test checkpoint suite, 52 related regression tests,
repository-wide Ruff, and the full 524-test suite passed on September 4, 2026.
No manual verification was originally required. Visible review noted that the
cuboid rotates substantially during the otherwise stable grasp.

Level 4.3E is complete. The complete controller uses copied-state-validated transport,
descent, release, settle, and retract behavior. After visible review exposed
about 24 degrees of cuboid yaw in the prior top-down grasp, the shared grasp
stage was revised to a shape-aware side-on (hammer-curl) wrist pose. A declared
rotation-only hold keeps the seeded object orientation through lift, stabilize,
transport, and place without constraining translation or release. Cuboid wrist
yaw follows the seeded object yaw; axial objects use their rotational symmetry.
Family release offsets remain configuration-owned, and success requires the
released object to stay within 0.025 m of its target while supported, unheld,
and below linear and angular speed limits. Task-local six-dimensional table
contact and rolling resistance apply only to standalone pick and composed
pick/place, preserving push behavior; the exact overrides and source/target
cues are restored from episode metadata. Ten complete successes span cuboid,
cylinder, and flat-puck families; all recompute and replay within five degrees
of the seeded held orientation, with compatible reach/pick/place segments, zero
safety intervention, and less than 0.005 m neighbor disturbance. Focused place,
segment, standalone grasp, and push suites pass after this remediation; the
related suite passes with 41 tests, repository-wide Ruff passes, and the full
suite passes with 527 tests. The saved viewer now uses an oblique camera that
keeps the hand, object, and named goal sides readable. The user confirmed on
September 4, 2026 that the corrected cuboid pick/place replay worked.

Level 4.3F is complete. The versioned cross-skill audit regenerated, headlessly
replayed, and recomputed two independently seeded accepted episodes for each of
reach, button, push, standalone grasp-and-lift, and complete pick/place. All 10
had complete action/provenance/timestamp evidence, zero safety interventions,
and neighbor disturbance below the configured limit. An ordinary failed episode
is covered by the automated audit and remains rejected without contributing to
expert counts. The required 25-test checkpoint suite passes. The existing
operator-owned pilot directory remains intentionally unchanged and below its
later collection minima. The user confirmed on September 5, 2026 that one
visible replay for all five skills looked good. Last Completed is now 4.3F and
Next Target is 4.3G.

Level 4.3G is complete. Exactly 20 scripted button successes were collected in
whole-session 14/3/3 train/validation/test splits. One frozen 64-by-64 tanh MLP
uses only causal simulator state and emits fixture-frame XYZ deltas through a
deterministic full-action adapter. Physically varied held-out hand resets first
exposed contact-induced `rh_MFJ1` and then `rh_FFJ1` limit failures; both were
diagnosed before changing data volume or model class, and corrected with a
minimal fixed-posture interior margin. The unchanged 20-episode, one-model
recipe then passed all 20 held-out press-and-release rollouts with zero safety,
contact, or invalid-action violations. The listed 5-test checkpoint suite and
37 related regressions pass; repository-wide Ruff and the full 533-test suite
also pass. No manual verification was required. Last Completed is now 4.3G and
Next Target is 4.3H; no 4.3H work has started.

Level 4.3H is complete. The push probe reused the qualified causal
state/phase/previous-delta convention, train-only normalization, 17-step
cadence, shared XYZ adapter, and one 64-by-64 tanh MLP. A learned-only delta
first failed 0/20 because approach error compounded into the workspace bound;
after that diagnosis, the scripted controller remained nominal and the MLP
learned only a sub-millimetre task-frame tracking residual. All 20 held-out
cuboid and flat-puck resets completed with zero board, neighbor, workspace,
joint, contact, tip, or invalid-action violations. No chunking, model/data
increase, or manual verification was required. Copied-state checks also showed
the provisional right-bin cells are infeasible under the current controller;
Level 4.3I must revise that matrix. Last Completed is now 4.3H and Next Target
is 4.3I; no 4.3I work has started.

Level 4.3I is complete. Configuration v2 froze the user-accepted 114–140
episode envelope, 74 source-owned coverage cells, its superseded 98/13/0/3
source minima, the qualified push subset and exclusions, whole-session/held-out
restrictions,
and Git LFS plus immutable-release handling. The user ran the coverage summary
and approved the matrix on September 5, 2026. The checkpoint suite passes with
31 tests, the dataset-plan suite with nine tests, touched-file Ruff checks pass,
and the full suite passes with 537 tests. Last Completed is now 4.3I and Next
Target is 4.4; no 4.4 implementation or collection has started.

Level 4.4 is complete. Configuration v3 expands all 10 reach, 12 push, and 10
button cells into 60 scripted source/split/session/seed assignments. All 60
required episodes passed
append-only review, including 20 reach, 20 push, and 20 press successes across
32/32 complete cells. The report passes session ownership and balance, target
balance, held-out isolation, schema and quality checks, headless replay, terminal
recomputation, and failure auditability with no core issues. Four held-out push
cells use the exact seeds already qualified in Level 4.3H because the generic
test-seed sequence failed the copied-state workspace check; no safety or task
threshold was weakened. The workspace contains 69 attempts and 62 accepted
episodes overall. Historical human-controlled attempts remain locally auditable
but are excluded from the active dataset and all required counts; two rejected
scripted retries also remain preserved.
The listed 10-test checkpoint suite, repository-wide Ruff, and the full 545-test
suite pass. No new manual verification was required. Last Completed is now 4.4
and Next Target is 4.5A; no Level 4.5A work has started. The v3 counts are an
anchor matrix; the prospective Level 4.5B release-scale expansion does not
retroactively change this completed checkpoint.

The team-facing interim findings, four-action measurement table, adopted
scripted-expert pivot, and revised plan are documented in
`docs/level4_pilot_report.md` under **Historical Control-Interface Pilot**.

Active Level 4+ rule: do not add or require live hand-pose control, webcam-based
demonstration collection, or human corrective interventions. Use deterministic
scripted experts for required data and scripted interventions for the frozen
correction cells. Existing legacy code and immutable historical episodes remain
readable but are outside the active training and qualification path.

For checkpoints involving a GUI or MuJoCo viewer, the agent should not mark the
checkpoint complete until the user confirms the manual verification passed.

---

## Agent Selection Rules

When selecting work:

1. Read `AGENTS.md`.
2. Read `docs/CURRENT_STATUS.md`.
3. Read `docs/module_contracts.md`.
4. Read the progress file listed under **Current Progress File**.
5. Select only the **Next Target Checkpoint** unless the user explicitly says otherwise.
6. Do not work on another level unless this file is updated.
7. Do not continue to the next checkpoint after finishing the current one.

---

## Updating This File

After a checkpoint is fully complete:

1. Move the finished checkpoint into **Last Completed Checkpoint**.
2. Set **Next Target Checkpoint** to the next unchecked checkpoint in the active progress file.
3. Update **Current Branch** if needed.
4. Add a note under **Manual Verification Status** if the checkpoint required user verification.

Example:

```text
Last Completed Checkpoint:
Level 1.0 — Repo and Health Check

Next Target Checkpoint:
Level 1.1 — Camera Smoke Test
```

---

## Completion Policy

A checkpoint may be marked complete only when:

```text
automated tests pass
listed smoke-test command runs, when applicable
manual verification is confirmed by the user, when required
progress checkbox is updated
commit is created with an appropriate message
```

Manual checkpoints include, but are not limited to:

```text
camera feed display
hand landmark visual overlay
MuJoCo viewer display
robot joint movement in GUI
demo video capture
```
