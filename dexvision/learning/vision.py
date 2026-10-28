"""Image encoder for the perception track (docs/PLAN.md, V2), following Chi et al. 2023.

One small ResNet-style CNN per camera, GroupNorm instead of BatchNorm (BatchNorm statistics do
not mix well with EMA weights), then spatial-softmax keypoints and a linear projection. It is
trained end to end with the policy. Training uses a random crop per sample and camera (the same
crop for every step of the observation history); acting uses the centre crop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.GroupNorm(groups, cout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)) + self.skip(x))


class SpatialSoftmax(nn.Module):
    """[B, C, H, W] -> [B, 2K]: the expected (x, y) in [-1, 1] of each of K learned keypoints."""

    def __init__(self, channels: int, keypoints: int) -> None:
        super().__init__()
        self.keypoints = nn.Conv2d(channels, keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        weights = self.keypoints(x).flatten(2).softmax(dim=-1)  # [B, K, H*W]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        expected_x = (weights * xs.reshape(1, 1, -1)).sum(-1)
        expected_y = (weights * ys.reshape(1, 1, -1)).sum(-1)
        return torch.cat([expected_x, expected_y], dim=-1)


class ImageEncoder(nn.Module):
    """One camera: [N, 3, crop, crop] float in [-1, 1] -> [N, feature_dim]."""

    def __init__(self, config: Mapping) -> None:
        super().__init__()
        groups, channels = config["group_norm_groups"], list(config["channels"])
        layers: list[nn.Module] = [
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, channels[0]), nn.ReLU(),
        ]
        for i, cout in enumerate(channels):
            cin = channels[max(i - 1, 0)]
            layers.append(ResidualBlock(cin, cout, 1 if i == 0 else 2, groups))
        self.trunk = nn.Sequential(*layers)
        self.pool = SpatialSoftmax(channels[-1], config["keypoints"])
        self.project = nn.Linear(2 * config["keypoints"], config["feature_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.pool(self.trunk(x)))


def crop(images: torch.Tensor, size: int, random: bool) -> torch.Tensor:
    """[B, To, H, W, C] -> [B, To, size, size, C]: a random window per sample (shared over To)
    when ``random``, else the centre."""
    b, _, h, w, _ = images.shape
    if not random:
        top, left = (h - size) // 2, (w - size) // 2
        return images[:, :, top:top + size, left:left + size]
    top = torch.randint(0, h - size + 1, (b,), device=images.device)
    left = torch.randint(0, w - size + 1, (b,), device=images.device)
    span = torch.arange(size, device=images.device)
    rows = (top[:, None] + span)[:, None, :, None]  # [B, 1, size, 1]
    cols = (left[:, None] + span)[:, None, None, :]  # [B, 1, 1, size]
    batch = torch.arange(b, device=images.device)[:, None, None, None]
    steps = torch.arange(images.shape[1], device=images.device)[None, :, None, None]
    return images[batch, steps, rows, cols]


class CameraEncoders(nn.Module):
    """All cameras: {camera: [B, To, H, W, 3] uint8} -> [B, To, cameras * feature_dim]."""

    def __init__(self, cameras: Sequence[str], config: Mapping) -> None:
        super().__init__()
        self.cameras = tuple(cameras)
        self.crop_px = int(config["crop_px"])
        self.feature_dim = int(config["feature_dim"]) * len(self.cameras)
        self.encoders = nn.ModuleDict({c: ImageEncoder(config) for c in self.cameras})

    def forward(self, images: Mapping[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for camera in self.cameras:
            x = crop(images[camera], self.crop_px, random=self.training)
            b, steps = x.shape[:2]
            x = x.flatten(0, 1).permute(0, 3, 1, 2).float() / 127.5 - 1.0  # [B*To, 3, crop, crop]
            features.append(self.encoders[camera](x).view(b, steps, -1))
        return torch.cat(features, dim=-1)
