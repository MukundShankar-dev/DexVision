# Level 4.7 visual export

The export is a derived, append-only visual subset of accepted nominal scripted
Level 4 episodes. It does not alter episodes, train perception, audit a release,
or package a dataset. Failures and corrections remain in their existing streams.

## Frozen sampling and camera

`configs/level4_visual_dataset.yaml` freezes one camera named
`workcell_fixed_v1`, 640 x 480 RGB, a 48-degree vertical field of view, position
`[-0.50, -0.62, 0.52]` metres, and look-at `[0, 0, 0.06]` metres. The exported
calibration version is `level4/visual-camera-v1`. The existing XML camera points
away from the work surface and its positive-x position is behind the button
wall. This first image-stream calibration is therefore explicitly set in the
export config; the completed scene and the original episodes remain unchanged.
No source episode is relabeled as having recorded RGB.

Candidate samples are source indices `0, 5, 10, ...`. The exporter restores
saved generalized positions and velocities by their exact recorded names,
restores static entity poses and procedural object scale, then calls
`mj_forward` without integrating actions. RGB and segmentation use the same
render scene. A frame's timestamp is exactly its source `state_timestamps` value;
the saved action timestamp is retained separately. The source episode/frame
index also resolves the complete requested/commanded/applied control record.

Camera intrinsics use integer pixel centers. `camera_from_world` is a 4 x 4
extrinsic matrix with camera axes right, down, forward and metre units. Object
poses contain world-frame translation and a 3 x 3 rotation matrix, representing
six physical pose degrees of freedom. This is a simulator-truth annotation,
not a pose inferred from the image.

## Annotations and conditions

Each image has a lossless uint16 PNG of visible instance ids; zero means
background, robot, or a non-target visual-condition primitive. Annotation rows
map positive mask values to stable object/fixture/target ids and class ids.
Boxes tightly enclose visible mask pixels with exclusive upper x/y edges.
Fully occluded and out-of-frame instances retain their simulator pose, an empty
mask, a null box, and the shared `occluded_or_out_of_frame` status. No amodal
mask or unobservable pose-estimation accuracy is claimed. Multiple geoms of one
entity share an instance id.

The four conditions are:

- Nominal rendering with the source task's geometry and scene isolation.
- Mild illumination at 0.90 times nominal intensity and zero temperature shift.
- Partial occlusion from one camera-facing render-only 3D card, derived from
  the currently visible goal box. The removed goal-mask fraction is measured
  and must stay at or below 0.35. The card is narrowed deterministically when
  necessary to satisfy the measured bound. Missing or too-small visible goals
  retain zero added occlusion with an explicit absence reason.
- Two rigid render-only distractors in bounded peripheral strips outside the
  task geometry. Their poses and dimensions are stored with every frame.

No condition changes physical state or control. Task markers and label sites
are hidden so privileged goal cues do not become perception shortcuts. The
pick/place fixture isolation from the source task is restored.

## Ownership and coverage

The compact matrix selects eight train, four validation, and four test source
episodes for each condition: 64 unique episodes and sessions overall. Selection
uses accepted nominal scripted sources, frozen cell/session ownership, and
sorted episode ids. Each split/condition contains primary examples of cuboid,
cylinder, flat puck, and Start button. Quarantined procedural namespaces,
rejected attempts, failures, and corrections cannot enter this selection.

Held-out object instances and the held-out right-bin target are hidden in
training and validation renders, including their background geometry. Their
raw simulator states remain untouched. Target coverage within a split means
all targets permitted in that split; the union must cover all five targets.
A condition may exist in all three splits, as prescribed by the frozen matrix;
its source sessions and images cannot cross splits.

Some source episodes share visually identical early states. Cross-split exact
RGB duplicates are excluded with their source indices, timestamps, pixel
digests, and retained split recorded in `excluded_frames.json`. All training conditions
are rendered first, followed by validation and then test, so test pixels never
select training frames. They are never
counted as exported frames. Required source counts require actual exported
frames, and missing category/target/condition cells fail the report. No frames
are duplicated to repair shortages.

## Commands and manual acceptance

Run from the repository root:

```bash
conda activate dexvision
python -m dexvision.apps.export_visual_dataset --config configs/level4_visual_dataset.yaml --dataset-dir data/demos/level4 --output-dir data/visual/level4
conda run -n dexvision pytest -q tests/test_render_annotations.py tests/test_visual_alignment.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

Offscreen rendering needs an OpenGL context; it needs no webcam, visible viewer,
or discrete GPU. On macOS a sandbox may block the graphics service even though
headless simulation works. Run the export in the normal desktop environment.
An existing output directory is rejected: intentionally changed exports need
a new output version rather than replacement.

Open the four contact sheets on macOS:

```bash
open data/visual/level4/contact_sheet_nominal.png data/visual/level4/contact_sheet_mild_illumination.png data/visual/level4/contact_sheet_partial_occlusion.png data/visual/level4/contact_sheet_bounded_distractors.png
```

On Windows, open those same PNG files in the image viewer. Each sheet has
three split rows and four primary-category columns, with source episode/frame
labels. `manual_review.json` maps all 48 panels to original RGB files.

**Pass:** inspect every sheet at native resolution; masks cover the named
visible surfaces, boxes tightly enclose them, ids match the objects, and all
three object families plus the button are reviewed in each split. Training
must contain no held-out large block, tall cylinder, heavy puck, right-bin
region, or test-owned source scene. Added occluders and distractors must be
visible and labels must respect their occlusion.

**Fail:** displaced masks/boxes, wrong ids, unexplained missing visible entities,
misaligned source frames, or any held-out/test scene in training. Report the
sheet and source frame. Automated results never satisfy this manual gate;
Level 4.7 remains incomplete until the user confirms acceptance.

## Storage and provenance

`report.json` reports candidate/exported/excluded frame counts, per-cell
coverage, PNG payload bytes, uncompressed RGB/mask size, and total artifact
size before the report itself. The export contains calibration, source-file
SHA-256 values, per-frame RGB/mask file digests, RGB pixel digests, split lists,
annotations, config snapshot, and review sheets. `data/visual/` is ignored by
Git. This checkpoint creates no archive or external upload.

The September 10, 2026 export contains 2,633 frames from 64 source episodes:
1,291 training, 531 validation, and 811 test frames. All 12 condition/split
cells pass. The candidate stream had 2,897 frames; 264 cross-split duplicates
were excluded. PNG payload size is 474,182,625 bytes; artifacts before the
report itself total 498,105,522 bytes. Of 809 partial-occlusion frames, 318 have
positive added goal occlusion; the maximum measured removal is one third.
Already invisible goals remain labeled as such rather than receiving invented
visible annotations. Verification passed with 15 focused tests, all 597 tests
in the full suite, repository-wide Ruff, and an independent read-back audit of
every saved frame and all 64 unchanged source episodes. Manual acceptance passed on September 10, 2026 when the user confirmed
“Looks good.” `manual_review_approval.json` is the append-only confirmation
receipt, bound to the reviewed sheet/report hashes. Original export-time
pending reports remain unchanged as historical records.

Shadow Hand assets retain their Menagerie/Shadow Robot source attribution and
Apache-2.0 license digest. The repository's custom procedural workcell has no
explicit license; the report records that fact instead of inventing one. The
owner must resolve its license before a later release. Payload hosting and
quota selection remain Level 4.9 work, informed by this export's measured size.

Claims are limited to the frozen simulated camera, named rigid entities, and
these bounded conditions. There is no cross-camera or real-world transfer claim.
