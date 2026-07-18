# Troubleshooting Guide

This file starts as a checklist and should be expanded as real issues appear.

---

## Camera Does Not Open

Try:

```bash
python -m dexvision.apps.check_camera --camera-id 0
python -m dexvision.apps.check_camera --camera-id 1
python -m dexvision.apps.check_camera --camera-id 2
```

Likely causes:

```text
wrong camera id
camera used by another app
phone webcam app not connected
Mac camera permissions not granted
Windows privacy settings blocking camera
```

---

## Camera Feed Is Laggy

Try:

```bash
python -m dexvision.apps.check_camera --width 640 --height 480
```

Likely causes:

```text
resolution too high
phone webcam over Wi-Fi
poor lighting causing tracker instability
other apps using camera
```

---

## MediaPipe Does Not Detect Hand

Try:

```text
better lighting
plain background
hand closer to camera
reduce motion speed
check camera mirror setting
```

---

## Finger Curl Values Look Wrong

Check:

```text
landmark indices
left/right hand handling
open-hand calibration
normalization range
whether image is mirrored
```

Run:

```bash
python -m dexvision.apps.check_hand_features
```

---

## Handedness Label Is Swapped

DexVision corrects MediaPipe's selfie-mirror handedness convention for normal
OpenCV camera frames. For the default camera path, your left hand should display
Left and your right hand should display Right.

Run:

```bash
python -m dexvision.apps.check_hand_tracking --camera-id 0
```

If a camera app or driver already provides selfie-mirrored frames, keep
MediaPipe's labels:

```bash
python -m dexvision.apps.check_hand_tracking --camera-id 0 --assume-mirrored-input
```

---

## MuJoCo Import Fails

Check:

```bash
python -m dexvision.apps.health_check
```

Likely causes:

```text
mujoco package not installed
Python version mismatch
environment not activated
```

---

## MuJoCo Window Does Not Open

Try headless test first:

```bash
pytest tests/test_mujoco_load.py
```

If headless works but viewer fails:

```text
graphics driver issue
remote session issue
viewer backend issue
```

---

## Robot Hand Explodes or Jitters

Likely causes:

```text
joint targets outside limits
actuator gains too high
simulation timestep too large
feature values too noisy
no smoothing
```

Check:

```text
joint target clipping
smoothing alpha
control frequency
lost-tracking behavior
```

---

## Tracking Loss Causes Bad Robot Motion

Expected behavior should be one of:

```text
freeze last valid command
slowly return to neutral
disable control until tracking resumes
```

Do not let missing landmarks produce random joint commands.

---

## PyTorch Training Does Not Improve

Check:

```text
dataset/action shapes
normalization
train/val split
whether actions are aligned with observations
whether demo replay works
whether model can overfit 5 demos
```

Always run tiny overfit first.
