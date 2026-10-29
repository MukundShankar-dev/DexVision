"""Perception probes and offline action error (docs/PLAN.md, V3 diagnostics).

A probe asks whether images hold what the task needs: the target object's position and which
object it is. Labels are the simulator's object state recorded in the demos (privileged, used
only to measure, never shown to a policy). Two probes share the same evaluation:

- ``scratch``: the policy's encoder architecture trained alone, supervised on the labels. What
  the cameras and resolution allow at best, before any policy is trained.
- ``frozen``: a small head on a trained policy's frozen image features. What the policy's
  encoder actually learned to represent.

Frames are reported in three buckets, because the hand hides the object as it grasps:
``far`` (every fingertip more than 5 cm from the object centre), ``near`` and ``lifted`` (the
object more than 1 cm above where it started). The thresholds only group the report.

``offline_action_error`` scores a trained policy's predicted action chunk against the expert's
on held-out demo windows. Low offline error with low closed-loop success points to compounding
errors rather than to what the model learned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dexvision.data.demos import load_episode
from dexvision.learning.dataset import DemoWindows
from dexvision.learning.train import Trained
from dexvision.learning.vision import CameraEncoders
from dexvision.sim.workcell import ACTUATOR_TARGETS, ROTATION, TRANSLATION

BUCKETS = ("far", "near", "lifted")
NEAR_M = 0.05  # fingertip to object centre
LIFTED_M = 0.01  # above the start height


@dataclass(frozen=True)
class ProbeFrames:
    """Single frames with labels. N frames, all from whole demo episodes."""

    images: dict[str, np.ndarray]  # {camera: [N, H, W, 3] uint8}
    position: np.ndarray  # [N, 3] object centre, m (privileged label)
    object_index: np.ndarray  # [N] index into the object list
    bucket: np.ndarray  # [N] index into BUCKETS


def frame_buckets(episode: Mapping[str, np.ndarray]) -> np.ndarray:
    """[T] bucket index per step of one episode (needs the object and fingertip fields)."""
    position = episode["obs/object_position"]  # [T, 3]
    tips = episode["obs/fingertip_positions"].reshape(len(position), -1, 3)  # [T, 5, 3]
    near = np.linalg.norm(tips - position[:, None], axis=-1).min(axis=1) < NEAR_M
    lifted = position[:, 2] > position[0, 2] + LIFTED_M
    return np.where(lifted, 2, np.where(near, 1, 0))


def probe_frames(episodes: Sequence[Mapping[str, np.ndarray]], cameras: Sequence[str],
                 objects: Sequence[str]) -> ProbeFrames:
    return ProbeFrames(
        images={c: np.concatenate([e[f"obs/rgb/{c}"] for e in episodes]) for c in cameras},
        position=np.concatenate([e["obs/object_position"] for e in episodes]),
        object_index=np.concatenate([np.full(len(e["action"]),
                                             objects.index(str(e["target_object"])))
                                     for e in episodes]),
        bucket=np.concatenate([frame_buckets(e) for e in episodes]),
    )


def load_probe_frames(paths: Sequence[Path], cameras: Sequence[str], objects: Sequence[str]
                      ) -> ProbeFrames:
    """``probe_frames`` of episode files, filled into preallocated arrays one episode at a time
    so that memory peaks at the frames themselves (gigabytes for a full dataset)."""
    lengths = []
    for path in paths:
        with np.load(path) as data:
            lengths.append(len(data["action"]))
    first = load_episode(paths[0])
    images = {c: np.empty((sum(lengths), *first[f"obs/rgb/{c}"].shape[1:]), np.uint8)
              for c in cameras}
    position, object_index, bucket = np.empty((sum(lengths), 3)), [], []
    start = 0
    for path, n in zip(paths, lengths, strict=True):
        one = probe_frames([load_episode(path)], cameras, objects)
        for c in cameras:
            images[c][start:start + n] = one.images[c]
        position[start:start + n] = one.position
        object_index.append(one.object_index)
        bucket.append(one.bucket)
        start += n
    return ProbeFrames(images, position, np.concatenate(object_index), np.concatenate(bucket))


class ScratchProbe(nn.Module):
    """The policy's image encoder, trained alone: images -> position [3] and object logits."""

    def __init__(self, cameras: Sequence[str], vision: Mapping, n_objects: int) -> None:
        super().__init__()
        self.vision = CameraEncoders(cameras, vision)
        self.head = nn.Linear(self.vision.feature_dim, 3 + n_objects)

    def forward(self, images: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """{camera: [B, H, W, 3] uint8} -> [B, 3 + n_objects]."""
        return self.head(self.vision({c: x[:, None] for c, x in images.items()})[:, 0])


class FeatureProbe(nn.Module):
    """A small head on frozen features: [B, F] -> [B, 3 + n_objects]."""

    def __init__(self, feature_dim: int, n_objects: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 3 + n_objects))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@torch.no_grad()
def frozen_features(vision: CameraEncoders, images: Mapping[str, np.ndarray], device: str,
                    batch: int = 512) -> torch.Tensor:
    """A trained encoder's features of single frames (centre crop): [N, F] on ``device``."""
    vision.eval()
    n = len(next(iter(images.values())))
    out = []
    for i in range(0, n, batch):
        chunk = {c: torch.as_tensor(x[i:i + batch], device=device)[:, None]
                 for c, x in images.items()}
        out.append(vision(chunk)[:, 0])
    return torch.cat(out)


def fit_probe(model: nn.Module, inputs, train: ProbeFrames, *, steps: int, batch: int, lr: float,
              device: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train ``model`` on ``train`` in place. ``inputs(rows)`` -> the model's input for those
    frame rows. Loss: MSE on standardized position plus cross-entropy on the object.
    Returns the position standardization (mean [3], std [3])."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    mean, std = train.position.mean(axis=0), train.position.std(axis=0) + 1e-9
    target = torch.as_tensor((train.position - mean) / std, dtype=torch.float32, device=device)
    labels = torch.as_tensor(train.object_index, device=device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    for _ in range(steps):
        rows = torch.randint(len(train.position), (batch,), generator=generator)
        out = model(inputs(rows))
        i = rows.to(device)
        loss = F.mse_loss(out[:, :3], target[i]) + F.cross_entropy(out[:, 3:], labels[i])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return mean, std


@torch.no_grad()
def probe_report(model: nn.Module, inputs, frames: ProbeFrames,
                 scale: tuple[np.ndarray, np.ndarray], batch: int = 512
                 ) -> dict[str, dict[str, float]]:
    """Per bucket: frames, position error in cm (mean, median, 90th percentile), the error of
    always guessing the training mean position (the no-information reference), and object
    accuracy. ``scale`` is ``fit_probe``'s return value."""
    model.eval()
    mean, std = scale
    predicted, classes = [], []
    for i in range(0, len(frames.position), batch):
        out = model(inputs(torch.arange(i, min(i + batch, len(frames.position))))).cpu().numpy()
        predicted.append(out[:, :3] * std + mean)
        classes.append(out[:, 3:].argmax(axis=1))
    error = np.linalg.norm(np.concatenate(predicted) - frames.position, axis=1) * 100
    guess = np.linalg.norm(mean - frames.position, axis=1) * 100
    correct = np.concatenate(classes) == frames.object_index
    report = {}
    for index, name in [(None, "all"), *enumerate(BUCKETS)]:
        mask = np.ones(len(error), bool) if index is None else frames.bucket == index
        if not mask.any():
            continue
        report[name] = {
            "frames": int(mask.sum()),
            "position_error_cm_mean": round(float(error[mask].mean()), 2),
            "position_error_cm_median": round(float(np.median(error[mask])), 2),
            "position_error_cm_p90": round(float(np.percentile(error[mask], 90)), 2),
            "mean_guess_error_cm": round(float(guess[mask].mean()), 2),
            "object_accuracy": round(float(correct[mask].mean()), 3),
        }
    return report


@torch.no_grad()
def offline_action_error(trained: Trained, episodes: Sequence[Mapping[str, np.ndarray]],
                         n_windows: int, seed: int, device: str = "cpu", batch: int = 256
                         ) -> dict[str, dict[str, float]]:
    """Mean absolute error between the executed part of the predicted chunk and the expert's
    actions, in normalized action units ([-1, 1] per component), on ``n_windows`` windows drawn
    from ``episodes``. Per bucket when the episodes carry object labels, and per action group."""
    horizons = trained.config["horizons"]
    windows = DemoWindows(episodes, trained.fields, horizons["observation"],
                          horizons["prediction"], trained.cameras)
    rows = np.random.default_rng(seed).choice(len(windows), min(n_windows, len(windows)),
                                              replace=False)
    labelled = all("obs/object_position" in e for e in episodes)
    buckets = np.concatenate([frame_buckets(e) for e in episodes]) if labelled else None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    execute = horizons["execution"]
    errors = []
    trained.model.to(device).eval()
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        obs, actions = windows.window(r)
        obs = torch.as_tensor(trained.obs_norm.encode(obs), dtype=torch.float32, device=device)
        images = {c: torch.as_tensor(x, device=device)
                  for c, x in windows.window_images(r).items()} or None
        chunk = trained.model.predict(obs, generator, images).cpu().double().numpy()
        errors.append(np.abs(chunk[:, :execute]
                             - trained.action_norm.encode(actions)[:, :execute]).mean(axis=1))
    error = np.concatenate(errors)  # [n, A]
    groups = {name: s for name, s in (("translation", TRANSLATION), ("rotation", ROTATION),
                                      ("fingers", ACTUATOR_TARGETS))
              if s.stop <= error.shape[1]}  # the workcell's 26-D action
    current = windows.obs_index[rows, -1]  # the window's current step
    report = {}
    for index, name in [(None, "all"), *enumerate(BUCKETS)]:
        if index is not None and buckets is None:
            break
        mask = np.ones(len(rows), bool) if index is None else buckets[current] == index
        if not mask.any():
            continue
        report[name] = {"windows": int(mask.sum()),
                        "all": round(float(error[mask].mean()), 4),
                        **{g: round(float(error[mask][:, s].mean()), 4)
                           for g, s in groups.items()}}
    return report
