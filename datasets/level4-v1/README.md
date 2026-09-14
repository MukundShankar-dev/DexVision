# DexVision / Hand2Bot Level 4 v1

Checkpoint 4.9 release candidate, September 14, 2026. Manual clean-directory
restoration is not yet owner-confirmed; Level 4 remains incomplete and Level 5
has not started. `manifest.json` preserves packaging-time gate status. Subsequent
verification receipts record later events without rewriting that manifest.

## Contents and integrity

The archive is `datasets/level4-v1.tar.gz`: **791,540,241 bytes** compressed,
**1,687,657,576 bytes** expanded, **33,350 regular files**. SHA-256:

```text
05c7a9d58d8466049c2a34ea07ec34e5d8614eaead47b2b7f93fbc4ab7df758b
```

It contains 1,112 active episodes (486 train, 174 validation, 452 test), 2,633
visual RGB/mask pairs with all annotations/calibration/provenance, the passing
74-cell audit, frozen splits, source/config snapshots, historical recording
snapshots for the nine replacements, and license/handoff documents. The 992
expert successes, 90 ordinary failures and 30 corrections remain distinct.
The complete visual source set consists of 64 active episodes. Excluded
collection attempts and staging data are not release members.

`manifest.json` binds every payload file by size and SHA-256. `SHA256SUMS` binds
release metadata, schemas, frozen splits, licenses and handoff. `schemas.json`
contains exact named observation/action layouts; [HANDOFF.md](HANDOFF.md)
defines Level 5 selection/normalization boundaries and known limitations.
The archive's original paths restore below a fresh directory; mutable working
data under `data/demos/` is never overwritten. The existing Level 2 archive and
its checksum/manifest remain unchanged and independently retrievable.

## Verify the local payload

Run from the repository root with the dedicated environment:

```bash
conda run -n dexvision python -m dexvision.apps.verify_dataset_release --release-dir datasets/level4-v1
```

A successful integrity check exits 0 and prints `integrity_passed: true`, 33,350
files, 1,112 active episodes and 2,633 visual frames. It does not automatically
approve the checkpoint. `--require-ready` deliberately rejects a candidate
whose packaging-time publication/manual gates are still pending.

## Required manual clean-directory retrieval

This command downloads the exact LFS object using a fresh Git repository and
empty LFS cache, then streams verification and restores every payload file.
It does not copy the working dataset or use the repository's existing LFS cache.
Requires Git, Git LFS, network access and the `dexvision` Conda environment.
Both target directories must be new; use a fresh suffix for a later attempt.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.verify_dataset_release --release-dir datasets/level4-v1 --download-dir outputs/level4/manual_release_v1/download --restore-dir outputs/level4/manual_release_v1/restored
```

**Pass:** download and command exit 0; `integrity_passed` is true; the reported
counts are 33,350 files, 1,112 episodes and 2,633 visual frames; every expected
file is restored under the requested directory with matching size and SHA-256.
`publication_ready: false` is expected until manual acceptance is recorded.

**Fail:** missing payload, an LFS pointer instead of the archive, transfer or
quota error, any checksum/size mismatch, missing/extra/duplicate/unsafe member,
nonzero exit, or incomplete restoration. Never reuse a partial destination.
Report the error; do not edit data or mark the checkpoint complete.

After this passes, explicitly confirm the manual verification in the task.

## Clean-clone retrieval after committing the release metadata

The LFS object can be retrieved by the command above before Git metadata is
committed. A normal new clone receives the pointer/manifests only after this
change is committed and pushed. No source commit or branch push is performed
by the packager. Once those metadata are available on the selected release ref:

```bash
git clone https://github.com/MukundShankar-dev/DexVision.git dexvision-level4-check
cd dexvision-level4-check
git lfs install
git lfs pull --include="datasets/level4-v1.tar.gz,datasets/dexvision_level2_v1.tar.gz"
conda run -n dexvision python -m dexvision.apps.verify_dataset_release --release-dir datasets/level4-v1 --restore-dir ../dexvision-level4-restored
```

Pin the immutable archive SHA-256 above and the trusted metadata commit when
consuming the release; a mutable branch name alone is not an integrity pin.
On a new machine, create the dedicated environment from `environment.yml`
before running Python verification. A checksum validates bytes, not authenticity:
obtain the manifests from the trusted project repository.

The Level 2 retrieval procedure remains in [../README.md](../README.md).
Its archive is not embedded in the Level 4 archive and has its own digest.

## Licenses, quota and recovery

The owner approved Apache-2.0 for project source/workcell/configuration included
here and CC BY 4.0 for generated data on September 14, 2026. Shadow Hand retains
its upstream Apache-2.0 license and attribution. See [licenses/NOTICE.md](licenses/NOTICE.md)
and the complete license texts. Historical export/audit license-pending notes
remain immutable; this release notice resolves them prospectively.

The owner selected Git LFS on the existing GitHub repository. The verified
`.gitattributes` rule applies `filter=lfs`, `diff=lfs`, `merge=lfs` to this archive.
As checked September 14, 2026, GitHub documents 10 GiB included LFS storage and
monthly download bandwidth for Free/Pro, and a 2 GB per-file cap. This 791.5 MB
archive is below that cap; together with the 50.9 MB Level 2 archive it occupies
about 842.5 MB. See [GitHub quota documentation](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
and [file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).
The account-wide remaining quota was unavailable to the current token; no
remaining-balance or billing-budget claim is made. Other repositories and
retained LFS versions also consume quota. Downloads consume owner bandwidth.

Keep at least 4 GB free disk and one independently verified offline copy. If
retrieval fails, check LFS authentication, publication and account quota; retry
the same immutable object into a new directory. Never overwrite v1 or remove
Level 2 to make space. Changed data requires a new version, audit, manifest,
checksum and release documentation. No paid quota or billing settings are changed.

## Reproduce packaging

The exact input audit is `outputs/level4/audit_v4_final`, dataset source commit
`41a5d9998058178eebc8a4f40dd633a621de04fe`, and owner-selected configuration
`configs/level4_release.yaml`. Packaging uses sorted paths, fixed metadata and a
zero gzip timestamp. The successful command was:

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.prepare_dataset_release --archive datasets/level4-v1.tar.gz --source-commit 41a5d9998058178eebc8a4f40dd633a621de04fe --release-config configs/level4_release.yaml
```

That command now refuses to overwrite the existing release. The archive includes
the exact release-tool source used to build it. Tests use synthetic data, require
no GPU/GUI/webcam, and check deterministic archive bytes, checksum failures,
missing/extra files, traversal, symlinks, LFS pointers and non-overwriting restore.
