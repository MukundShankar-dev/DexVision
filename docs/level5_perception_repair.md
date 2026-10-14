# Checkpoint 5.6 perception repair

Last Completed remains **5.5** and Next Target remains **5.6**. This is a
training/validation repair after a consumed negative held-out experiment.
No held-out images or labels are decoded for repair, no held-out inference is
rerun, and no fresh confirmation or owner acceptance is claimed. The historical
`level5_perception_results.md` and `level5_perception_verification.json` retain
the original negative V1/V2 evidence. No 5.7 work is included.

## Diagnosis and preserved attempts

The accepted place and push/press repair reports from the previous two stages
were reviewed. Their frame-origin, causal-history, physical-contact, geometry,
cadence and immutable-attempt lessons remain applicable. The accepted policies,
decoders, source data and original evaluation gates are unchanged. Place's
privileged collision preview remains explicitly incompatible with vision.

Training RGB shows the blue puck and board share hue. V2's smoothed histogram
merged the puck with the board and admitted background fragments. The new
class-specific saturation threshold and narrow hue range separate these
components. Button hue is fitted independently on TRAIN, within the fixed
calibrated fixture envelope. The detector receives no labels at inference.
Diagnosis artifacts are in `outputs/level5/perception_repair/diagnosis_v3/`.

The original silhouette optimizer converted its projected hull to float32.
Tiny finite-difference steps disappeared, corrupting derivatives. The float64
polygon moments repair removes that numerical failure, but nominal upright
geometry still biases partially visible, rotated and differently sized bodies.
RGB mask centroids are visible-surface centroids, not metric body origins.
The numeric diagnosis used only training and validation.

V3 fits a deterministic kernel regressor for visible-centroid/body-centre pixel
offsets and camera depth. Its 99.15% validation recall and 0.63 mm median look
promising, but 199.5 mm p95 fails. Every detector and kernel candidate is retained
under `outputs/level5/perception_model_v3/`, with source in
`outputs/level5/perception_repair/v3_source/`.

V4 learns pixel offsets and **world height**, then intersects a calibrated ray
at that inferred height. All nine tree candidates pass the aggregate gates;
the selected depth-8/minimum-leaf-4 model has 98.99% recall, 0.063 mm median and
2.56 mm p95. However, parked objects dominate that denominator. A separate
primary-target audit finds only 93.75% recall and 28.59 mm p95, so V4 is retained
as an insufficient repair. Its model, full candidate matrix, validation media
and source snapshot remain under their V4 paths.

## Preserved V5/V6 repairs and frozen V7 method

V5 preserved all 27 detector candidates and three completed pose candidates.
The depth-12/leaf-1 fit stopped because adjacent floats rounded a midpoint to
the upper value, producing an empty child. A dedicated regression test now
checks the exact nonempty partition. Its depth-8/leaf-2 model has 24.09 mm
primary p95, but one puck identity switch fails the gate. No V5 model qualifies.

V6 completes all 12 detector and nine pose candidates. Brightness cutoffs above
50 lose true dark puck observations; an experimental area-weighted tracker
introduces two primary cuboid switches. The selected negative V6 model has
28.55 mm primary p95. V5/V6 directories and source snapshots remain intact.

`configs/level5/perception_repair_v7.yaml` and its SHA256 sidecar freeze the
nine detector and nine pose candidates, feature/label scales, full fitting budget,
selection order, history rules, parent recipe/snapshot and data/source digests
before fitting. Source is preserved in `outputs/level5/perception_repair/v7_source/`.
All candidate models and complete validation reports remain under
`outputs/level5/perception_model_v7/`.

V7 uses minimum 3 pixels and hue margin 8 for cuboids/cylinders, with a separate
TRAIN-fitted button hue and puck saturation floor 170. Puck hue offsets are
[-3,+3], [-3,+8] or [-4,+8] relative to TRAIN median 107. Nearby disconnected
puck fragments may merge at gaps 1/3/6 pixels, with a maximum 64-pixel group
extent. Union changes neither the underlying pixels nor the occluder gap;
features come from the visible union only. The nine complete detector candidates
are evaluated without task identity. The original unweighted tracker is restored.
TRAIN hue quantiles and validation-only miss/identity diagnoses are retained.

Confidence uses a fixed 32-pixel union-area denominator. Lowering a component
cutoff does not inflate confidence; merging evidence adds only actual observed
pixels. The confidence>=0.8 and age<=0.1 s consumer boundary remains unchanged.

Each class has a deterministic weighted CART regressor. Its 34 inputs contain
16 current RGB component geometry/HSV features, a prior-available bit, actual
elapsed time divided by 0.51 s, and 16 prior RGB features. History resets per
episode; absent, same-time, future or expired context has availability zero.
Prior truth positions, identities, physical phase and task state never enter
these features. Tracking and feature extraction precede TRAIN label association.
Matched TRAIN instances receive inverse episode-count weights with a frozen
4x multiplier for primary targets; primary identity affects supervision only.

Learned outputs are body-centre UV offsets divided by 32 pixels and world
height divided by 0.1 m. The engineered decoder uses the exact fixed camera
intrinsic/extrinsic transform to recover a world/metre position. It does not
assume a board plane, read current simulator state, or produce actuator commands.
Depths 8/12/20 and minimum leaves 1/2/4 receive full deterministic fitting and
complete validation, with stable split ties and no stochastic seed/early stop.

Selection requires the original >=95% recall, <=10 mm median, <=25 mm p95,
<=1 identity switch/1000 visible instances and 100% synthetic stale/ambiguity
rejection, including per-core-class recall. V5 onward additionally requires the same
numerical gates on primary targets and primary per-class recall. Missing
observations stay in denominators; pose error is conditional on available
uncertified hypotheses and its count is explicit. These stricter supplemental
gates prevent parked objects masking failure on manipulated targets.

## Frozen V8 pose comparison

V7 retains all nine detector and nine pose candidates. Its selected detector
has 97.60% primary recall and zero switches, but all tree candidates fail the
primary p95 gate at about 28.55 mm. The pose diagnosis shows centroid/body-origin
bias and cylinder-tree splits at elapsed-time values 0.3333333333333336 and
0.33333333333338916. Those represent numerical jitter, not physical motion.

V8 freezes the selected V7 detector as a single candidate before fitting and
rounds only the elapsed-time FEATURE to six decimals. Acquisition timestamps,
tracker expiry, availability and delivery-age checks retain exact source times.
The RGB pose feature count remains34. All nine corrected tree candidates are
retained. Eight kernel ridge candidates use bandwidths 0.5/1/2/4, regularization
0.001/0.01 and at most 256 deterministic farthest-point TRAIN feature centres.
Four local regressors use 1/3/5/9 nearest TRAIN examples with stable ties and
inverse squared feature distance (floor 1e-8) times the frozen supervision
weight. Both use TRAIN mean/std with a 0.02 spread floor. All 21 candidates receive
complete validation before selection; no offline-loss early stop is used.
Outputs/scales and the fixed camera ray decoder remain the same. Exact fitting
statistics, parameters and training exemplars/coefficients are bundled with the
selected model, and the recipe/code/data digests bind them to evaluation.
The independent V8 YAML/checksum and `v8_source` snapshot precede fitting.

## V8 validation results

All 21 pose candidates completed; all four local regressors pass both original
aggregate and primary-target gates. Three neighbours are selected by the frozen
primary-p95 ranking. Validation contains 531 source frames across 16 episodes,
1290 visible core instances and 208 visible primary-target instances.

| Validation denominator | Recall | Position hypotheses | Median | p95 | ID switches |
| --- | --- | --- | --- | --- | --- |
| All visible core instances |99.61% (1285/1290)|1285|0.051 mm|0.83 mm|0|
| Primary manipulated target |97.60% (203/208)|203|0.44 mm|7.10 mm|0|

Primary per-class recall: cuboid 95.38%, cylinder 96.72%, puck 100%, button 100%.
All original median/p95/recall/identity gates and 100% synthetic rejection pass;
no numerical threshold is weakened. Five primary detections remain missing,
and 940 false-positive components remain explicitly counted and visible.
Position counts are hypotheses; certified full typed observations remain zero.
Primary velocity p95 is 29.81 mm/s, exceeding the 20 mm/s push predicate margin;
this is diagnostic and does not qualify downstream skill decisions.

The selected JSON is 3,920,672 bytes. On local macOS/ARM CPU, measured offline
inference median/p95 is 36.0/49.9 ms, with no measured inference latency above
0.1 s. Peak traced Python allocation is 52,708,752 bytes; native-library process
peak is unmeasured and RTX 5070 Ti is untested. The sparse source cadence still
cannot supply 0.1 s freshness between sampled frames or the accepted 34 ms control
cadence. These are validation-only measurements on repeatedly exposed families,
not held-out, unseen-family or runtime qualification.

Artifact readback verifies all 16 panels, 16 complete decoded videos, 531 source
frame records, exact timestamps/validation ownership, source/model/data hashes,
all accepted receipts and historical evidence indices. No held-out inference
or annotation decoding occurred. Owner validation review is pending.

## Commands and pending verification

Commands executed in the `dexvision` Conda environment:

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_perception --config configs/level5/perception_repair_v3.yaml
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_perception --config configs/level5/perception_repair_v4.yaml
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_perception --config configs/level5/perception_repair_v4.yaml --split validation --output-dir outputs/level5/perception_repair/v4_validation
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_perception --config configs/level5/perception_repair_v5.yaml
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_perception --config configs/level5/perception_repair_v6.yaml
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_perception --config configs/level5/perception_repair_v7.yaml
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_perception --config configs/level5/perception_repair_v8.yaml
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_perception --config configs/level5/perception_repair_v8.yaml --split validation --output-dir outputs/level5/perception_repair/v8_validation
conda run --no-capture-output -n dexvision pytest -q tests/test_object_perception.py tests/test_perception_metrics.py tests/test_perception_repair.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision pytest -q
git diff --check
```

Final software checks: **963 passed, one skipped** in the complete repository
suite (741.14s); focused checkpoint/config-lock suite **59 passed**. Ruff and
`git diff --check` pass. The final receipt records commands, failed attempts,
selected parameters, source/data/artifact bindings and pending manual status.

Do not repeat fitting/evaluation into these evidence directories. Historical
recipes are reproduced from their exact source snapshots; current code pins
intentionally refuse altered implementations. Repairs explicitly allow only
validation evaluation, rejecting a default/test request before model/data
loading. A new test set and confirmation protocol must be independently frozen
before any future confirmation; the consumed test is never reusable.

Owner **validation diagnostic** review command on macOS, from the repository:

```bash
open outputs/level5/perception_repair/v8_validation/validation_contact_sheet.png outputs/level5/perception_repair/v8_validation/sequence_*.avi
```

Pass: all scheduled panels and all 16 episode videos are present/readable;
visible boxes/masks align with the object, body-centre crosses represent the
inferred body origin, visual IDs remain with the same object across source
frames, units/acquisition timestamps and displayed gaps agree, and missing,
ambiguous and low-confidence hypotheses visibly reject at the boundary. Full
orientation and semantic identity must display as unavailable. Review every
sequence, including occlusion/reacquisition, fixture and distractor conditions.

Fail: swapped IDs, unexplained large position jumps, misaligned overlays,
missing episodes/conditions, hidden truth-filled estimates, stale data presented
as a new observation, silent acceptance of an invalid result, or unreadable
review media. This review is pending owner confirmation and does not substitute
for held-out confirmation or full typed-schema qualification.

## Qualification limits

Positions and timestamp-derived linear velocities are RGB hypotheses. Full
orientation/angular velocity, instance geometry, semantic IDs, fixture depth,
contacts, held/supported state and causal manipulation phase remain unavailable.
The full `ObjectObservation`/`WorldState` boundary therefore fails closed; no
accepted state bundle is qualified with inferred perception. The sampled image
cadence cannot supply the inherited 0.1 s freshness throughout the accepted
0.034 s control cycle. All five accepted-bundle incompatibilities and sensitivity
diagnostics remain disclosed. Validation success alone cannot complete 5.6.

The algorithm has no neural weights, foundation model or pretrained assets.
Custom source/workcell: Apache-2.0; generated Level 4 data: CC-BY-4.0;
dependencies retain their licenses. CPU latency/model size/Python allocation
measurements are in the validation report. Native-library peak process memory
is unmeasured and RTX 5070 Ti testing is unavailable/untested.

## Owner feedback and Git scope

On October 14, 2026 the owner reported "I think mostly good" and explicitly
authorized commit/push. `level5_perception_review_feedback.json` binds this
positive preliminary feedback to the exact V8 model/report/media index. It does
not assert that every manual pass criterion was confirmed, or complete 5.6.

Source, frozen recipes, tests and documentation are versioned. Generated models,
failed runs, source snapshots and review media remain preserved locally under
the existing `outputs/*` ignore rule, consistent with earlier Level 5 runs. They
are not included in this Git push. Operator-owned datasets remain untouched.
