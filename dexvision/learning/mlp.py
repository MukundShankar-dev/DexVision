"""MLP behaviour-cloning baseline on the Diffusion Policy interface.

It regresses the whole action chunk [Tp, A] from the flattened observation history [To * D]
with a mean-squared error, in the same normalized units.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


class MLPModel(nn.Module):
    name = "mlp"

    def __init__(self, obs_dim: int, action_dim: int, horizons: Mapping, config: Mapping) -> None:
        super().__init__()
        self.pred_horizon, self.action_dim = horizons["prediction"], action_dim
        dims = [obs_dim * horizons["observation"], *config["hidden_dims"]]
        layers: list[nn.Module] = []
        for cin, cout in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(cin, cout), nn.Mish()]
        layers.append(nn.Linear(dims[-1], self.pred_horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [B, To, D] -> [B, Tp, A]"""
        return self.net(obs.flatten(1)).view(-1, self.pred_horizon, self.action_dim)

    def loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(obs), actions)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Deterministic; ``generator`` is accepted for interface parity and unused."""
        return self(obs).clamp(-1, 1)
