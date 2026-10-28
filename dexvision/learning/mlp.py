"""MLP behaviour-cloning baseline on the Diffusion Policy interface.

It regresses the whole action chunk [Tp, A] from the flattened observation history [To * D]
(plus the same image features as the diffusion model, when it has cameras) with a mean-squared
error, in the same normalized units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from dexvision.learning.vision import CameraEncoders


class MLPModel(nn.Module):
    name = "mlp"

    def __init__(self, obs_dim: int, action_dim: int, horizons: Mapping, config: Mapping,
                 cameras: Sequence[str] = (), vision: Mapping | None = None) -> None:
        super().__init__()
        self.pred_horizon, self.action_dim = horizons["prediction"], action_dim
        image_dim = int(vision["feature_dim"]) * len(cameras) if cameras else 0
        dims = [(obs_dim + image_dim) * horizons["observation"], *config["hidden_dims"]]
        layers: list[nn.Module] = []
        for cin, cout in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(cin, cout), nn.Mish()]
        layers.append(nn.Linear(dims[-1], self.pred_horizon * action_dim))
        self.net = nn.Sequential(*layers)
        if cameras:  # built after the MLP, so a model without cameras is initialized as before
            self.vision = CameraEncoders(cameras, vision)

    def forward(self, obs: torch.Tensor, images: Mapping | None = None) -> torch.Tensor:
        """obs [B, To, D], images {camera: [B, To, H, W, 3] uint8} -> [B, Tp, A]"""
        if hasattr(self, "vision"):
            obs = torch.cat([obs, self.vision(images)], dim=-1)
        return self.net(obs.flatten(1)).view(-1, self.pred_horizon, self.action_dim)

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, images: Mapping | None = None
             ) -> torch.Tensor:
        return F.mse_loss(self(obs, images), actions)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, generator: torch.Generator | None = None,
                images: Mapping | None = None) -> torch.Tensor:
        """Deterministic; ``generator`` is accepted for interface parity and unused."""
        return self(obs, images).clamp(-1, 1)
