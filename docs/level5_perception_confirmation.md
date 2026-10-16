# Checkpoint 5.6 fresh confirmation and shared estimate schema

October 14, 2026. Only 5.6 is selected. **Fresh confirmation fails; checkpoint
5.6 remains open.** Last Completed remains 5.5. No 5.7 work is implemented.

## Frozen experiment and immutable data

The owner explicitly authorized new confirmation data/protocol and resolution
of inferred-schema gaps. The selected V8 detector, tracker and three-neighbour
TRAIN regressor were frozen before generation. No fitting or selection used
the new confirmation. Historical baseline V2's consumed test images/labels and
inference were not reused.

`configs/level5/perception_confirmation_v2.yaml` and its checksum define 32 new
simulator sessions: four source cells/classes, four original visual conditions,
and one nominal/one boundary draw per cell/condition. Seeds 93580000–93583301
are disjoint from historical sources and the failed collection attempt. Whole
episode/session ownership is preserved. All **1,641 RGB frames and 32 attempts**
are retained, including **24 source task failures**. There is no trajectory
replay, learned policy, object-pose write after reset, source-success filtering
or replacement. The scripted source is a perception stimulus, not a qualified
manipulation trial. Visibility uses the frozen train/validation convention,
explicitly hiding historical held-out background instances. This is fresh-reset
evidence on exposed families/cells, not unseen-family, human-operator,
real-camera or deployed-sensor evidence.

The exact `level4/visual-camera-v1` calibration is unchanged. Digest-only audit
finds **zero historical RGB pixel collisions** without decoding consumed images
or annotations. Dataset files, source states, metadata, evaluation labels and
per-file checksums are in the new immutable
`datasets/level5-perception-confirmation-v2.tar.gz` release. The archive is
**287,481,232 bytes**, has the Git LFS attribute, and its SHA-256 is
`a9b3bb50caefc9d0851d637f48301abf95ab9344d0b43ebaf529e6b18523d8ab`.
Tracked manifest and exact protocol accompany it. Existing archives, working
data, accepted bundles and historical receipts are unchanged. Local packaging
does not imply remote upload or commit.

V1 collection failed before rendering because JSON sorting changed the finger
actuator order. Its protocol, source snapshot, partial source directory and
failure receipt remain under `outputs/level5/perception_confirmation_v1/`.
V2 preserves ordered actuator names and uses a new seed range. V1 produced no
images or learned inference. The original 5.0 lock is unchanged; config-lock
tests now also check independent confirmation recipes.

## Shared schema and compatibility

`dexvision/perception/world_estimate.py` implements additive
`level5/world-estimate-v1` for both simulator-truth and inferred branches.
`EntityEstimate` separates episode-local observation ids from semantic workcell
ids. Every field records producer, availability, confidence, validity,
acquisition time, units, frame and body origin. Unknown, uncertified, stale,
ambiguous, low-confidence or evaluation-truth input rejects. Hypothesis reads
require explicit diagnostic opt-in. Missing orientation, angular motion,
semantic identity, robot telemetry, contacts/relations and fixture depth stay
**None**. Accepted `ObjectObservation`/`WorldState` types are unchanged; partial
estimates cannot cross the full control boundary.

All **1,641 inferred frames** and **8,118 source truth snapshots** pass typed
roundtrip. With RGB/time fixed, withheld-truth changes leave inferred output
and history unchanged. Separate post-forward render-label oracles validate the
same type on 1,641 frames and **13,128 body origins**; calibrated ray roundtrip
maximum error is **4.83e-16 m**. This tests units/frames/origins, not estimator
accuracy. Cached source snapshots differ from post-forward labels by up to
**1.093 mm**; they are preserved as structural snapshots and are not the
physical pose-parity oracle. Scoring uses post-forward render annotations.
The separate schema audit records these roles without rerunning inference.

Exact accepted transform scalar/batch oracles and field inventories remain
separate. Full inferred feature parity is **unsupported**, with rejection
instead of missing-feature padding. Reach lacks certified semantic geometry
and telemetry; pick lacks full pose, retention/contact and entry/history;
place additionally retains its privileged live-state collision preview;
push lacks measured contact/support/phase and reliable speed; press lacks
measured fixture depth and contact/dwell. All five perception/control tracks
remain blocked. Primary velocity p95 error is **144.82 mm/s**, above push's
20 mm/s margin. Sparse source cadence cannot guarantee 0.1 s freshness at the
accepted 34 ms control interval. Structural schema checks grant no skill,
default registry or deployment admission.

## One-time result

| Metric | All visible core objects | Primary manipulated target |
| --- | ---: | ---: |
| Eligible instances | 4,247 | 925 |
| IoU-matched detections | 4,081 | 759 |
| Recall | 96.09% | **82.05% — fails 95%** |
| Conditional position median | 0.171 mm | 2.174 mm |
| Conditional position p95 | 8.955 mm | **72.640 mm — fails 25 mm** |
| Identity switches | 1 | **1.081/1,000 — fails 1/1,000** |

Aggregate cylinder recall is **91.19%**, below 95%. Primary cuboid/cylinder/puck
recalls are 86.67%/73.33%/92.09%; button recall is 100%. All misses, two missing
primary pose hypotheses, **2,685 false positives** and source failures remain
counted. Synthetic rejection is 100%. CPU inference/envelope p95 is 31.83 ms.
Full control states available are **0/1,641**. Native peak memory and RTX 5070 Ti
remain unmeasured/untested.

Historical baseline V2 test and this distinct fresh confirmation V2 are both
consumed. The original protocol's exclusive reservation refuses evaluation
under a different output path. No rerun, tuning, threshold relaxation or failed
example filtering occurred. Exact bindings and the negative result are in
`docs/level5_perception_confirmation_v2.json`.

This exposes insufficient V8 coverage of ordinary-physics manipulation and
occlusion. Repair requires separately frozen TRAIN/validation evidence and
method selection, followed by another independent confirmation. This
confirmation remains ineligible for fitting or selection. The archive and
schema implementation do not supply a passing perception method.

## Commands and tests

These identify write-once runs in the dexvision environment; existing paths
refuse replacement and the consumed evaluation cannot be repeated.

```bash
conda activate dexvision
python -m dexvision.apps.confirm_perception freeze --config configs/level5/perception_confirmation_v2.yaml --output-dir outputs/level5/perception_confirmation_v2/protocol
python -m dexvision.apps.confirm_perception collect --protocol outputs/level5/perception_confirmation_v2/protocol/protocol.json --output-dir outputs/level5/perception_confirmation_v2/dataset
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m dexvision.apps.confirm_perception evaluate --protocol outputs/level5/perception_confirmation_v2/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v2/dataset --output-dir outputs/level5/perception_confirmation_v2/evaluation
python -m dexvision.apps.confirm_perception package --protocol outputs/level5/perception_confirmation_v2/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v2/dataset --release datasets/level5-perception-confirmation-v2.tar.gz
shasum -a 256 -c datasets/level5-perception-confirmation-v2.tar.gz.sha256
pytest -q tests/test_perception_confirmation.py tests/test_observation_boundary.py tests/test_object_perception.py tests/test_perception_metrics.py tests/test_perception_repair.py tests/test_level5_learning_plan.py
ruff check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
git diff --check
```

macOS collection required graphics-context access outside the sandbox.
Evaluation exits **2** for failed numerical qualification, not an implementation
crash. Focused checks pass **148 tests**; full regression passes **1,052 tests,
one skipped in 772.73 s**. Ruff passes.

## Manual review

Earlier V8 validation review is already owner confirmed and is not requested
again. Fresh acceptance review is **not triggered** because numerical gates
fail. New media is available for diagnostic inspection only:

```bash
open outputs/level5/perception_confirmation_v2/evaluation/heldout_contact_sheet.png outputs/level5/perception_confirmation_v2/evaluation/sequence_*.avi
```

Diagnostic pass: all 16 class/condition panels and 32 videos are readable;
episode/frame/timestamp and units agree; overlays align; visual ids remain
consistent; unavailable semantic identity/orientation and rejected/stale
observations are explicit. Fail: missing clips, swaps, unexplained jumps,
misalignment, hidden truth-filled estimates or silent invalid/stale acceptance.
Visual review cannot override the failed numerical result or complete 5.6.
