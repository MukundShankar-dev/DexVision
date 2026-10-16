# Versioned Dataset Releases

Level 4 v1 is complete and independently retrievable through Git LFS as
`level4-v1.tar.gz`. Source and release metadata are published. Frozen metadata,
schemas, splits, licenses, completion evidence and retrieval instructions are in
[`level4-v1/README.md`](level4-v1/README.md). Level 5 is active; its additive unassisted pick amendment is described below.
The Level 2 release described below remains unchanged.

The editable working tree under `data/demos/` remains ignored by Git. Published
dataset snapshots are immutable archives in this directory and are stored with
Git LFS.

### 5.6 publication and local storage

The owner authorized publishing the 5.6 code, compact model bundles, manifests,
checksums, protocols, acceptance receipt and status documentation while keeping
the large dataset payloads local. The proposed bulk upload was 34.5 GB and
exceeded the owner's reported 10 GiB LFS allowance. Only the V25 and selected
V29 compact model bundles (about 8.2 MB combined) are new Git LFS payloads in
this publication. Earlier published releases and history are unchanged.

The new 5.6 bulk archives and `*.tar.gz.part-*` files are ignored by Git.
Their immutable bytes, archive checksums and scientific manifests remain
preserved locally. The historical multipart manifest retains its original
filename, `level5-perception-lfs-transport-v1.json`, but its 20 data parts are
local storage, not a promise that they can be fetched from GitHub. A clone
alone cannot reproduce these new datasets: obtain the exact payloads from the
owner, verify the committed checksums, and then use the original recipe.

Local disk cleanup removed redundant whole copies of eight large archives and
extracted image/source copies of consumed confirmation V2–V7. All dataset
content remains in intact archives or complete verified parts. The accepted
V8 working dataset, active training inputs, models, protocols, numerical
results, videos and failed-attempt evidence remain available locally.

To recreate the eight whole archives from the retained local parts, run from
the repository root in the dexvision environment:

```bash
python -m dexvision.apps.restore_multipart_releases --manifest datasets/level5-perception-lfs-transport-v1.json
```

The command verifies every part and concatenation and never replaces an
existing archive. It does not download data. After restoration, the original
`.tar.gz.sha256` checks and extraction instructions apply. Per-dataset working
directory restoration commands and byte-verification receipts are recorded
locally under `outputs/maintenance/disk_cleanup_20261016/`.

`docs/level5_perception_publication.json` records the publication scope. This
storage decision does not change scientific qualification or start 5.7.

The Level 2 snapshot contains:

```text
data/demos/raw/
data/demos/rejected/
data/demos/reports/
```

It excludes temporary staging data and the one-off top-level recorder smoke
directory.

This snapshot is the immutable input to the Level 3 learning-feasibility work.
It is not the project's final comprehensive skill dataset and must not be used
to claim cross-session, cross-object, cross-camera, or open-world
generalization. Level 4 has published a separate versioned release with genuine
session ids, broader objects/goals, grasp-lift-place and recovery coverage, and
visual grounding data. Level 5 will train and qualify the full skill set from
that frozen release. The Level 2 archive must not be overwritten when the
Level 4 release is created.

`dexvision_level2_v1_manifest.json` records the archive digest, task counts,
frame counts, report digests, and the known absence of recording-session ids.

After cloning, install Git LFS, download the archive, verify it, and extract it
from the repository root:

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/dexvision_level2_v1.tar.gz.sha256
tar -xzf datasets/dexvision_level2_v1.tar.gz
```

Do not modify an existing release archive. Create a new version when the
dataset intentionally changes.

For a Level 4 release, keep manifests, checksums, and split metadata in Git. A
bounded archive may use Git LFS; if synchronized image streams exceed hosting
quotas, store the immutable payload in documented versioned artifact storage
and keep its retrieval instructions and cryptographic digests here.


## Level 5 unassisted pick amendment v1

`level5-pick-unassisted-v1.tar.gz` is an additive, immutable **state-only** release
for owner-authorized checkpoint 5.3A. It contains 144 training and 48 validation
standalone pick trajectories from the exact-loop deterministic teacher V3, with
no object-orientation hold or object-pose writes after reset in scratch or live
execution. Every frozen collection attempt was reported successful; none was
replaced. The later independent saved-state audit verifies only **191/192**:
one final speed exceeds 0.0200 m/s after recomputation. V1 is preserved as
superseded evidence and must not be described as fully physically verified.
The corrected V2 collection repeats the same cases without replacement.

The six named instances are already exposed in development and are intentionally
included in the new training amendment. This does not alter historical Level 4
split ownership or claim unseen-instance/family generalization. The new whole
simulator sessions are named per split/cell in the manifest (18 training sessions,
six validation sessions); they are deterministic simulation runs, not human,
camera, cross-operator or cross-camera sessions. The source scene is the existing
pick/place task, but recordings terminate at standalone pick success and contain
no completed-place labels or RGB/perception supervision.

The archive contains only canonical `datasets/level5-pick-unassisted-v1/episodes/`
payloads: metadata, result receipts, and compressed causal observations, actions,
safety masks and physical states. It contains no staging directory, training
cache, temporary diagnostic script, model checkpoint or source asset copy.
The original `data/demos/` and all prior release archives remain unchanged.
`level5-pick-unassisted-v1_manifest.json` binds each payload digest, plan, teacher
recipe/feasibility hashes, episode/session assignments, all attempts and physical
parameter/settings provenance. Source simulator assets retain their existing
licenses; they are not redistributed in this archive.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-pick-unassisted-v1.tar.gz.sha256
# Optional inspection extraction; training reads the verified archive directly.
tar -xzf datasets/level5-pick-unassisted-v1.tar.gz
```

The original Level 4 release metadata/assets remain required. Exact collection,
verification, training and qualification commands are in
[`../docs/level5_pick_repair.md`](../docs/level5_pick_repair.md). Never overwrite
this archive. A changed teacher, data selection or content requires a new version.
No remote upload or dataset commit is implied by local packaging.

## Level 5 unassisted pick amendment v2

`level5-pick-unassisted-v2.tar.gz` preserves the same frozen 144 training and
48 validation reset cases and split ownership as V1, using new episode/session
identifiers. Teacher V4 recomputes physical success from post-integration state
on a separate MuJoCo data copy. This fixes the cached-state discrepancy above
without changing live physics, thresholds, fallback order or replacing resets.
All 192 collection attempts report success and zero safety violations; independent
saved-state verification is recorded in the 5.3A method report. V1 stays immutable.

The archive and `level5-pick-unassisted-v2_manifest.json` use the same canonical
three-file episode format and provenance disclosures as V1. V6 and V7 policy recipes
pin both hashes and require the complete passing source audit before training.
This remains exposed-instance, state-only standalone pick data, with no place,
vision, human-demonstration or arbitrary-start qualification claim.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-pick-unassisted-v2.tar.gz.sha256
tar -xzf datasets/level5-pick-unassisted-v2.tar.gz
```

Training can read the verified archive directly; extraction is optional. No
existing release or raw working episode is overwritten, and local packaging does
not imply a remote upload or commit.

## Level 5 unassisted place amendment v2

`level5-place-unassisted-v2.tar.gz` is an additive, immutable state-only place
source for checkpoint 5.4A. Its deterministic expert establishes a physically
held object through unassisted scripted pick and bounded setup, then performs
ordinary-physics placement. Only the place interval supplies training labels:
195 causal pre-action features, applied actions, safety masks and saved physical
states. Setup actions are replayed exactly and excluded from supervision.

The frozen plan contains 144 training and 48 validation resets in disjoint
cells and simulator sessions. Collection retained all attempts: 140/144 and
47/48 succeeded with zero live safety violations. The five failures have no
training payload and remain in the manifest. Independent saved-state MuJoCo
reconstruction passed all 187 successful payloads. The 561 archive members
are the three canonical files for each success; source candidate previews and
failed-attempt staging remain local. The manifest binds plan, teacher screen,
search outcomes, per-file digests, split ownership and the physical audit.

The earlier V1 collection stopped after a board-margin safety violation and
was never published. This V2 archive has new frozen resets and does not
overwrite V1 staging or any Level 4/5 release. All object families and target
types were previously exposed. Session ids denote deterministic simulator
sessions, not human, camera or cross-operator provenance. This release alone
does not qualify a learned place policy.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-place-unassisted-v2.tar.gz.sha256
tar -xzf datasets/level5-place-unassisted-v2.tar.gz
```

See [`../docs/level5_place_repair.md`](../docs/level5_place_repair.md) for
collection and audit commands. The local archive is LFS tracked; packaging
does not imply a remote upload or commit.

## Level 5 unassisted place amendment v3

`level5-place-unassisted-v3.tar.gz` is a new immutable, additive state-only
source for checkpoint 5.4A. V2 had no right-receptacle cell in either source
split and kept slot B only in validation. V3 records all 30 object/target
cells in both training and validation, with disjoint reset seeds, episode ids
and deterministic simulator sessions. The cell identities are deliberately
exposed across splits, so this release does not claim unseen-cell or
unseen-family generalization or change the historical Level 4 split owners.

The frozen plan attempted 240 training and 120 validation resets. It retained
all six failures without replacement: 237/240 training and 117/120 validation
attempts produced successful source payloads. Every live safety counter is
zero; each split exceeds the 95% success and 90% worst-family publication
gates. Independent saved-state MuJoCo reconstruction passed all 354 successful
trajectories. Only the place interval supplies the canonical 195-field
pre-action state and applied-action labels. The unassisted scripted pick/setup
prefix is excluded. The archive has 1,062 canonical episode members; all
candidate previews and failed staging attempts remain local. The manifest
binds their outcomes, the plan, source screen, file digests and physical audit.

V1 stopped without a release; V2 and all historical failed policy reports
remain unchanged. V16 trained on V2 and failed its fresh selection audit, so
this V3 release alone does not qualify a learned policy. Simulator session ids
do not imply human, camera or cross-operator provenance.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-place-unassisted-v3.tar.gz.sha256
tar -xzf datasets/level5-place-unassisted-v3.tar.gz
```

The local archive has the Git LFS filter and its checksum was verified. See
[`../docs/level5_place_repair.md`](../docs/level5_place_repair.md) for the
collection and audit commands. Local packaging does not imply an upload or
commit.

## Level 5.6 fresh perception confirmation v2

`level5-perception-confirmation-v2.tar.gz` is a new immutable **confirmation-only**
RGB/state release: 32 fresh scripted simulator sessions, 1,641 rendered frames,
four core classes/conditions and nominal/boundary resets. Every attempt is
retained, including 24 source task failures. Source actions are generated anew
under ordinary physics, with no learned controller, replay, post-reset object
pose writes or success filtering. Zero historical whole-image pixel collisions
were found. Exposed families/cells and train/validation render visibility remain
explicit; no real-camera, human-session or unseen-family provenance is claimed.

The exact frozen V8 model was evaluated once and **fails** core/primary gates.
This set is consumed and **must never fit/select a model or be retested**.
It does not qualify perception or any skill. The failed pre-render V1 software
attempt has no archive; its evidence is retained locally. V2 never overwrites
any original Level 4/5 release or operator working episode.

The manifest/checksum and exact `_protocol.json` accompany the 287,481,232-byte
LFS archive. Extraction restores only canonical sources/images/labels and
dataset checksums under `datasets/level5-perception-confirmation-v2/`, with no
staging data, training caches or model weights. Source cached world snapshots
are structural evidence, with up to 1.093 mm post-integration pose difference;
canonical position labels in `frames.jsonl` come from post-forward rendering.
Unavailable inferred fields remain explicit. Generated data uses CC-BY-4.0;
existing asset licenses remain unchanged.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-perception-confirmation-v2.tar.gz.sha256
tar -xzf datasets/level5-perception-confirmation-v2.tar.gz
```

See [the method/result report](../docs/level5_perception_confirmation.md) and
[verification receipt](../docs/level5_perception_confirmation_v2.json). Running
the baseline requires the existing local V8/accepted-bundle artifacts, which
are not shipped in this dataset archive. Do not rerun consumed confirmation
after cloning. Packaging implies no remote upload or commit.

## Level 5.6 separate development V1

The owner-authorized repair data is a new immutable LFS release:
`level5-perception-development-v1.tar.gz`, its `.tar.gz.sha256`,
`level5-perception-development-v1_manifest.json`, and exact
`level5-perception-development-v1_protocol.json` / `_protocol.sha256`.
It contains **128 TRAIN sessions / 6,703 frames** and **32 validation sessions /
1,621 frames**, with separate whole-session/episode/reset ownership. There are
zero historical or cross-split whole-image pixel collisions. All source failures
and frames remain included. The archive has 17,133 payload files and is
1,458,141,311 bytes; SHA-256:
`924bff9af9c5d97cc8f9ed608e218e861e3c3f419919b50c06849dd28b9a5e47`.

Seeds 94560000–94563307 (TRAIN) and 94570000–94573301 (validation) are new,
using the same exposed four source cells/families and four render conditions.
No unseen-cell/family, real-camera, human-session or skill success claim is made.
There is no test split. Historical baseline V2 test and fresh confirmation V2
remain consumed and ineligible for fitting/selection/retest. Source truth uses
copied post-forward physics; visual annotations and source state are exact at
object body origins. No accepted bundle or existing release was changed.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-perception-development-v1.tar.gz.sha256
tar -xzf datasets/level5-perception-development-v1.tar.gz
```

The canonical extraction prefix is `datasets/level5-perception-development-v1/`.
The recorded local run used `outputs/level5/perception_development_v1/dataset/`.
Bind any reproduction to a new recipe/output directory. Full method, selection,
confirmation separation and manual requirements are in
[`docs/level5_perception_development.md`](../docs/level5_perception_development.md).

### Development V2 pixel-center render amendment

`level5-perception-development-v2.tar.gz` preserves the **same 160 V1 source
sessions**, source states/actions, 6,703 TRAIN / 1,621 validation frames, all
source failures, exact annotations, mask files and eligible denominators.
It is a new render release, not another independent set of source sessions.
Zero RGB multisampling aligns RGB pixels with integer mask pixel centers;
calibration and physics are unchanged. V1 remains preserved. No consumed test
supplies images, labels, fitting, selection or a retry.

The new LFS archive has 17,133 payload files and is 1,415,969,537 bytes. SHA-256:
`630244ccb8b38437bc43f49a6a8f49db0d228ce447330feb1c0d67aefa869072`.
The matching `.tar.gz.sha256`, `_manifest.json`, `_protocol.json` and
`_protocol.sha256` accompany it. Zero historical/cross-split pixel collisions
and whole-source/label parity are audited. Qualification using this amendment
is bounded to pixel-center renders; it does not establish accuracy on antialiased
inputs or real cameras.

```bash
shasum -a 256 -c datasets/level5-perception-development-v2.tar.gz.sha256
tar -xzf datasets/level5-perception-development-v2.tar.gz
```

Canonical extraction prefix: `datasets/level5-perception-development-v2/`.
Recorded local fitting root: `outputs/level5/perception_development_v2/dataset/`.

### Fresh confirmation V3 — consumed negative result

`level5-perception-confirmation-v3.tar.gz` preserves all 32 new sessions and
1,596 RGB frames, including 14 expert-error endings. Zero historical pixel collisions
were found. V11 passes aggregate gates but fails primary p95 at 51.02 mm and
identity at 1.131/1,000. This set is consumed and cannot train/select or retest.
It has 271,191,582 bytes and 3,293 payload files.
SHA-256: `825595f4847921562af71255f2c904b5b9e398c097ac992fc6ff3e01c3b0500a`. Matching checksum, manifest and exact protocol
companions are retained. Historical baseline V2 and fresh V2 remain unchanged.
See `docs/level5_perception_confirmation_v3_failed.json`.


### Development V3 — expanded independent TRAIN/validation

`level5-perception-development-v3.tar.gz` preserves all 384 new source sessions:
256 TRAIN and 128 validation, with 12,984 / 6,573 RGB frames. It is new ordinary
physics data, with no saved-trajectory replay or outcome filtering. All 186
expert-error endings remain. Raw terminal task flags are a separate diagnostic
(282 false), affected by released button state after a completed press cycle;
this collection does not qualify skill success. There is no test split.

The archive contains 40,271 payload files and is 3,324,915,524 bytes. SHA-256:
`673e10dc10f6cd003756d7672f40b5503158341e20a0aa34e38a8fc8d2741b94`.
Checksum, full file-hash manifest and exact frozen protocol companions are
included beside it. Canonical extraction prefix:
`datasets/level5-perception-development-v3/`; recorded local data root:
`outputs/level5/perception_development_v3/dataset/`. Existing releases are intact.

```bash
shasum -a 256 -c datasets/level5-perception-development-v3.tar.gz.sha256
tar -xzf datasets/level5-perception-development-v3.tar.gz
```


### Development V4 — targeted cuboid coverage amendment

`level5-perception-development-v4.tar.gz` preserves all 384 separately frozen
cuboid sessions across the four original conditions: 256 TRAIN /128 validation,
15,890 /7,798 RGB frames. All 352 expert-error endings are retained; 32 sources
end `scripted_complete`. These are perception stimuli, not policy qualification
trials. No source outcome is filtered, replaced or used to admit a skill.
There is no test split and no historical/cross-split pixel collision.

The archive is 4,106,459,199 bytes with 48,533 payload files. SHA-256:
`15dbe66821ffb9799ad538280fe9bb95b9b0bacfc6bfaeb93cf6a7386ee77061`.
The complete member/checksum readback and Git LFS attribute pass. Matching
checksum, manifest and exact frozen protocol companions are beside the archive.
Source/schema parity passes all 117,668 snapshots and 189,504 rendered body
origins: source/render position mismatch is zero, rotation delta <=2.34e-15,
and camera/world roundtrip error <=2.30e-16 m.

```bash
shasum -a 256 -c datasets/level5-perception-development-v4.tar.gz.sha256
tar -xzf datasets/level5-perception-development-v4.tar.gz
```

Canonical prefix: `datasets/level5-perception-development-v4/`. Recorded local
root: `outputs/level5/perception_development_v4/dataset/`. The derived training
union reuses V3's384 sessions and V4's384, preserving 512 TRAIN /256 validation
whole sessions and 43,245 frames. It copies exact parent images/masks/labels,
adds no physics sessions and leaves both raw parents immutable. No consumed
test enters that union. This remains exposed-family/cell pixel-center simulator
RGB; no real-camera, unseen-family or accepted-control qualification is claimed.


### Fresh confirmation V4 — frozen and consumed once

`level5-perception-confirmation-v4.tar.gz` contains all256 newly seeded source
sessions and13,284 frames across four exposed cells/families and four original
conditions. It was frozen against selected V19 seed3/epoch3,000 before generation;
no prior source trajectory or consumed image/label was reused. Every planned
attempt and expert-error ending is retained. Historical pixel collisions are0.
The protocol-level reservation permits one evaluation; it is currently consumed
and the single evaluation is complete and negative: original/core/primary gates
pass, but cuboid stabilization median13.801mm exceeds10mm. Its receipt is
`docs/level5_perception_confirmation_v4_failed.json`. It can never enter
TRAIN/validation or be evaluated again.

The archive is2,259,403,643 bytes and27,341 payload files. SHA-256:
`96ff7f394f34a57746d8aaba37f2fd784aa4d28469510804427907805558ffd1`.
Complete archive/member readback, Git LFS attribute and external protocol checksum
pass. Checksum, full manifest and exact frozen protocol companions are retained.
The protocol's `model_sha256` is the historical V8 source-parameter ancestor;
`selected_model_sha256` is the actual V19 inference model. The release is V4;
the supported protocol schema remains `level5/perception-confirmation-v3`.
Existing releases and all earlier consumed test sets remain unchanged.

```bash
shasum -a 256 -c datasets/level5-perception-confirmation-v4.tar.gz.sha256
tar -xzf datasets/level5-perception-confirmation-v4.tar.gz
```

Canonical prefix: `datasets/level5-perception-confirmation-v4/`; recorded local
root: `outputs/level5/perception_confirmation_v4/dataset/`. Pixel-center simulator
RGB only; no real-camera, unseen-family or accepted-control qualification claim.

### TRAIN/validation stabilization phase windows V5

`level5-perception-phase-windows-v5.tar.gz` is a frozen derived sampling
amendment of allowed developmentV3/V4 cuboid sources, not new independent
sessions. It includes all5,131 stabilization queries and preceding causal RGB
context (20,524 rows), retaining480 physical parents including14zero-query
parents in coverage. Original whole-source TRAIN/validation ownership remains.
3,408TRAIN/1,723validation queries contain40/20visible primary queries from
10/6physical sources; no outcome or visibility filtering selects queries.
Exact parent images may intentionally recur; there are zero consumed-test
pixel collisions and zero cross-split collisions. No consumed test is a parent.

Archive1,924,712,769bytes,24,247payload files; SHA-256
`156093685697d9b447cb9d9cb16dbae8cc2f188c8db5ae15f3d0b5a4220a9eb1`.
Full archive/member readback and LFS attributes pass. Protocol/manifest/checksum
companions preserve source/code bindings. Raw source states remain in immutable
parent releases, referenced explicitly rather than republished here.
Source-schema audit checks145,061typed snapshots /164,192body origins with zero
position difference and numerical-precision orientation/calibration differences.

```bash
shasum -a 256 -c datasets/level5-perception-phase-windows-v5.tar.gz.sha256
tar -xzf datasets/level5-perception-phase-windows-v5.tar.gz
```

Canonical prefix:`datasets/level5-perception-phase-windows-v5/`; local root
`outputs/level5/perception_development_v5/dataset/`. OnlyTRAIN labels may fit;
validation is selection only. Pixel-center simulator/exposed-family scope; no
real-camera, full-control or checkpoint5.7 qualification.

### Fresh confirmation V5 — frozen against phase-balanced V22

`level5-perception-confirmation-v5.tar.gz` retains all256new99580000-series
ordinary-physics source sessions /13,227RGB frames, without outcome filtering.
It binds selected V22 seed11/epoch3,000 only after all3full-budget candidates
pass full-sequence and dense-query validation. Zero historical/cross-split
pixel collisions. TRAIN/development and all4prior consumed tests are excluded.
The protocol is frozen before generation; this set is permanently ineligible
for fitting/selection. Its one-time evaluation is complete and negative: all
original/core/primary position/identity gates pass, but cuboid release17/18
(94.44%) and settle7/9(77.78%) fail strict95%phase recall. Exact negative receipt:
`docs/level5_perception_confirmation_v5_failed.json`. This set is consumed and
cannot be evaluated again.

Archive2,249,469,632bytes /27,227payload files; SHA-256:
`7d6a8168a71f32a85123fbebdbfd43eaf90973fd5cdd398bd447809aa97385e4`. Full archive/member readback and LFS attributes pass.
External manifest/protocol/checksum companions retain exact bindings. The
protocol schema remains`level5/perception-confirmation-v3`; releaseversionV5.
`model_sha256` names only the historicalV8source-parameter ancestor;
`selected_model_sha256` is the actual V22inference model. The metadata erratum
explicitly identifies executable neural seed budgets; frozen recipes remain
unchanged. Exposed-cell/family pixel-center simulator RGB only; no real-camera
or accepted-control/5.7qualification is claimed.

```bash
shasum -a 256 -c datasets/level5-perception-confirmation-v5.tar.gz.sha256
tar -xzf datasets/level5-perception-confirmation-v5.tar.gz
```

Canonical prefix:`datasets/level5-perception-confirmation-v5/`; recorded local
root:`outputs/level5/perception_confirmation_v5/dataset/`. All prior releases
and consumed tests remain unchanged.

### TRAIN/validation release/settle windows V6

`level5-perception-late-windows-v6.tar.gz` is a separately frozen derived
sampling amendment of developmentV3/V4, with all1,066recorded release/settle
queries plus three causal RGB context frames each (4,264rows). Physical-source
split ownership stays unchanged. All480cuboid parents remain in coverage,
including441zero-query parents;39physical sources render windows. This adds
no new independent sessions and filters no query by visibility/outcome.

Visible TRAIN queries:release473/settle243 from27physical sources each;
validation:release195/settle99 from11sources each. All5consumed tests are
excluded, with zero historical and cross-split pixel collisions. Intentional
parent RGB reuse is explicit. Raw source states remain in unchanged parent
releases; rendered labels/masks/owners/calibration are included here.

Archive257,127,471bytes /3,289files; SHA-256:
`a3f4d3ba9774dff059a5e27aa2b2d72fce4d762e6157eb3797b1fd675c697854`. Full member/checksum readback passes; external
manifest/protocol/checksum companions retain exact bindings.

```bash
shasum -a 256 -c datasets/level5-perception-late-windows-v6.tar.gz.sha256
tar -xzf datasets/level5-perception-late-windows-v6.tar.gz
```

Canonical prefix:`datasets/level5-perception-late-windows-v6/`; local root
`outputs/level5/perception_development_v6/dataset/`. TRAIN masks may fit hue
parameters; validation is selection only. Pixel-center simulator/exposed-family
scope; no real-camera or accepted-control qualification.

### Fresh confirmation V6 — TRAIN-hue frontend with frozen V22pose parameters

`level5-perception-confirmation-v6.tar.gz` retains all512new101580000-series
ordinary-physics source sessions /26,848RGB frames,with no outcome filtering.
The exact selected V23bundle combines TRAIN-fitted cuboid hue9–19with unchanged
V22seed11/epoch3,000pose parameters/scales/ray decoder. Its revised frontend
passes all3candidate full-sequence/stabilization/release/settle validations
before this protocol/data is created. The original neural training frontend
differs; no neural weight updates are claimed for V23.

All5earlier consumed tests and development sources are excluded. Historical
and cross-split pixel collisions are0. The separately frozen512case budget
requires>=8visible instances from>=4physical sources in EACHcube release/settle
phase,plus allunchanged core/global/primary/phase gates. Unsupported late
strata cannot qualify by omission. The single evaluation is complete and negative: cube transportp9537.212mm
exceeds25mm and release pose-hypothesis coverage50/51=98.04%fails99%.
Global/primary gates and late recall pass. Exact negative receipt:
`docs/level5_perception_confirmation_v6_failed.json`. Allsource/media/metrics
are retained; no ownerreview is triggered. This set is permanently ineligible for fitting/selection or an inference retry.

Archive4,566,326,484bytes /55,237payload files; SHA-256:
`3bc8546fc00f35c40914bf3e0f602f9b05f4d63e04950c73e78da2843f63c7bf`. Full archive/member readback passes. External
manifest/protocol/checksum companions preserve exact code/model/data bindings.
The schema remains`level5/perception-confirmation-v3`; releaseversionV6.
`model_sha256` is only the historicalV8source-parameter ancestor,while
`selected_model_sha256` binds the actual V23hue+pose inference bundle.

```bash
shasum -a 256 -c datasets/level5-perception-confirmation-v6.tar.gz.sha256
tar -xzf datasets/level5-perception-confirmation-v6.tar.gz
```

Canonical prefix:`datasets/level5-perception-confirmation-v6/`; local root
`outputs/level5/perception_confirmation_v6/dataset/`. Pixel-center simulator
RGB/exposed-family/cell scope only. No full control,real-camera,unseen-family
or5.7qualification is claimed. Every earlier release remains unchanged.

## 5.6 causal cadence windows V8

`level5-perception-cadence-windows-v8.tar.gz` is an immutable derived TRAIN/
validation RGB amendment, not a new physical-session dataset or confirmation.
It retains all480allowed D3/D4 cube physical parents (320TRAIN/160validation),
including two zero-query parents. It has19,448query windows (13,014TRAIN /
6,434validation), three causal preceding frames per query, and77,792rows.
Visible primary queries number546TRAIN /294validation from99/46physical sources.
The source-frame cadence is5, equal to original D3/D4 sampling. The frozen
reason-string's4x-density claim is incorrect; the immutable correction is
`docs/level5_perception_cadence_metadata_errata.json`. No new sessions/density
claim or outcome/visibility filtering is made. Interrupted all-frame V7 staging
and its protocol remain preserved, unused for fitting/evaluation.

Archive SHA-256:
`ec777faa790beb8ccf9c170f83961c276f8564e3ee57018ccf88790b866b04a5`.
Bytes:3,307,189,778. Complete archive/member readback verifies all41,769files;
LFSfilter is confirmed. Zero consumed-history/cross-split pixel collisions.
Typed-source/body-origin audit checks146,776snapshots and622,336body origins:
source/render position delta0, maximum rotation delta2.34e-15, camera roundtrip
2.29e-16m. This integrity audit is separate from learned perception performance.

Public `_manifest.json`, `_protocol.json`, `_protocol.sha256`, and
`.tar.gz.sha256` companions bind the archive. Parent releases D3/D4 must remain
available for original source-state provenance. All six consumed test sets
remain closed. V24reharvests this amendment plus all dense stabilizationV5/lateV6
windows under the exact runtime, and deduplicates validation by physical source/
frame. Checkpoint5.6 remains open; no5.7work.

```bash
shasum -a 256 -c datasets/level5-perception-cadence-windows-v8.tar.gz.sha256
shasum -a 256 -c datasets/level5-perception-cadence-windows-v8_protocol.sha256
tar -xzf datasets/level5-perception-cadence-windows-v8.tar.gz
```

## 5.6 selected current-RGB inference bundle V25

`level5-perception-model-v25.tar.gz` contains the selected current-frame MLP,
exact TRAIN means/std, UV/world-height output scales, camera calibration,
detector/tracker settings, runtime recipe, historical training recipe, selected
artifact metadata, matching source and asset notices. It performs localization
only. Visual tracking is causal; physical motion, semantic identity, full
orientation, robot/fixture telemetry and contacts remain unavailable. All five
accepted control tracks are unsupported. This is not5.7deployment qualification.

The selected weights are exact V24current-head seed11/source epoch3,000, with
zero V25optimizer updates. Full-sequence and six-phase VALgates pass; two of
three current candidates pass and the failed seed7remains preserved. The
inference-only runtime recipe can relocate visual_root to the extracted bundle
for storage; actual RGBtyped outputs and model digest stay identical. The
full training recipe retains historical input pins, separate from the portable
inference subset. Use CurrentLocalizationBaseline and the current-only engine.

Archive SHA-256:
`f457f30ab9fba132ca5ceec3db575c6f71e724273b92c0779d3b16ca334cc347`.
Bytes4,104,713; all99payload members pass complete readback; LFSfilter confirmed.
Selected model semantic SHA:
`73bda5eff9e99b22f90b89dd7bf071df684b060e22444abc249357d3c93926ab`.
Active current-weight SHA:
`0ba58825982fbfd714f028c3b8a9c4195e9f4ec5befe678e5273025863bbdb85`.
Its later freshV7 result fails one primary IDswitch and is preserved as a
consumed negative. This historical V25 bundle is not the current qualification
candidate; V29/freshV8 are recorded below. Checkpoint5.6 remains open.

```bash
shasum -a 256 -c datasets/level5-perception-model-v25.tar.gz.sha256
tar -xzf datasets/level5-perception-model-v25.tar.gz
```

## 5.6 fresh confirmation V7

`level5-perception-confirmation-v7.tar.gz` preserves all512new ordinary-physics
simulator sources and26,555RGB frames from the frozen103580000-series case
plan. It is test-only, training-ineligible, and permanently consumed by one
V25current-only evaluation. All six earlier consumed sets stay preserved and
closed too. No source outcome/visibility filtering or replacement is allowed.
Zero historical/cross-split pixel collisions; all54,651payload files pass full
archive/member readback. Source/frame ownership and exact selected recipe,
model, calibration, software and implementation pins were frozen before pixels.

SHA-256:`37b3538d232d5cc1fe170e2167f5a20ee6a0a1d52f610dec93f69b27bd5aaa26`.
Bytes:4,516,428,539. Public manifest/protocol/checksum companions accompany
the archive. The protocol requires unchanged original/primary/class-phase gates
and explicit release/settle support>=8visible from>=4physical sources each.
One-time evaluation fails the zero-primary-ID-switch gate (one cuboid/release
switch), while position/recall/late gates pass. The immutable negative receipt
is docs/level5_perception_confirmation_v7_failed.json. No owner review is triggered.

```bash
shasum -a 256 -c datasets/level5-perception-confirmation-v7.tar.gz.sha256
shasum -a 256 -c datasets/level5-perception-confirmation-v7_protocol.sha256
tar -xzf datasets/level5-perception-confirmation-v7.tar.gz
```

V7 final result: strict confirmation fails one primary cuboid/release identity
switch; position/recall and late coverage pass. Receipt:
`docs/level5_perception_confirmation_v7_failed.json`. V7 is consumed, archived,
and closed to repair/selection/repeat evaluation. V26 uses only existing frozen
TRAIN/validation data and exact V25 position weights; no5.7 work.

## 5.6 guarded current-RGB inference bundle V29

`level5-perception-model-v29.tar.gz` is the current inference-only candidate.
Exact selected V25pose weights/means/std/scales/calibration/ray decoder remain
unchanged. Current learned body-origin projection supplies cuboid association
coordinates; a cube-only uncertain replacement guard rolls back vetoed history.
Other-class birth/box association and every original tracker limit remain intact.
Original masks/boxes/current learned positions are published. Physical motion,
full orientation/semantic identity and sensor/contact fields remain unavailable;
all five accepted inferred-control tracks are unsupported. No5.7 qualification.

SHA256:`ee030bd0846bb0a71c0cacae6c72ed331848e05c2e2d845b383375a5ffb66610`.
Bytes:4,125,900; payload files:127.
Complete archive/member readback and LFSfilter verification pass. The portable
runtime_recipe.json retains body_center_tracking_classes, calibration, exact
transforms and all runtime parameters; load selected.json and runtime_recipe.json,
set visual_root only to the extracted directory, then construct
GuardedBodyBaseline from dexvision.perception.guarded_body_localization.
Actual validation RGB/typed outputs are identical after storage relocation.
All matching source and asset notices are included. Original full training
provenance remains separate from the relocatable runtime subset.
Model semantic SHA:`3a25ea7b49cc05ce7d67e89226a7e8c231a7feaea945d4863e7ed56a5cf5cadd`.
Active pose-weight SHA:`0ba58825982fbfd714f028c3b8a9c4195e9f4ec5befe678e5273025863bbdb85`.
Owner manual acceptance was confirmed October 16, 2026; see
`docs/level5_perception_acceptance.json`.

```bash
shasum -a 256 -c datasets/level5-perception-model-v29.tar.gz.sha256
tar -xzf datasets/level5-perception-model-v29.tar.gz
```

## 5.6 fresh confirmation V8

`level5-perception-confirmation-v8.tar.gz` contains all512fresh ordinary-physics
scripted source resets and26,687RGBframes from the independently frozen
105580000-series cases (32per exposed cell/condition). Test-only and
training-ineligible; all outcomes retained, zero historical/cross-split pixel
collisions. The protocol binds exact selected V29weights/runtime/calibration,
source/software/audits before pixels. Mandatory separate release/settle support
of8visible from4physical sources each and every original gate stay unchanged;
added99%overall/95%supported-phase identity coverage prevents vacuous rejection.

SHA256:`515f9e93b2c98eefb11ae2a0fb7e21072040a7903c011360b9d82ac619a27700`.
Bytes:4,537,883,231; all54,915payload files pass full
archive/member readback. LFSfilter confirmed; public manifest/protocol/checksum
companions included. Evaluated once, passes all automated gates; now consumed
and permanently closed to fitting/selection/retry. All seven earlier consumed
sets and failed attempts remain preserved. Schema/body-origin, all512video
readbacks, typed/raw evidence parity, full-control rejection and actual retry
guard pass. Evidence:docs/level5_perception_verification_v29.json.
Owner review of all 16 panels/36 scheduled clips passed on October 16, 2026;
5.6 is complete. `docs/level5_perception_acceptance.json` binds acceptance to
these frozen artifacts. Level 5.7 remains unstarted; no control qualification.

```bash
shasum -a 256 -c datasets/level5-perception-confirmation-v8.tar.gz.sha256
shasum -a 256 -c datasets/level5-perception-confirmation-v8_protocol.sha256
tar -xzf datasets/level5-perception-confirmation-v8.tar.gz
```
