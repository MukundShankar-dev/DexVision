"""Diffusion Policy (Chi et al. 2023), CNN variant, written in-house.

A 1-D temporal UNet denoises an action chunk [Tp, A]. It is conditioned (FiLM) on the diffusion
step and on the flattened observation history [To * D]. Training is DDPM with epsilon
prediction; sampling is deterministic DDIM given the initial noise.
All tensors here are in normalized units ([-1, 1] on the training range).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


def squared_cosine_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """[T] cumulative signal fractions of the squared-cosine schedule (Nichol & Dhariwal 2021)."""
    def f(t):
        return math.cos((t / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    betas = [min(1 - f(i + 1) / f(i), 0.999) for i in range(timesteps)]
    return torch.cumprod(1 - torch.tensor(betas, dtype=torch.float64), dim=0)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """[B] -> [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        angles = t.float()[:, None] * freqs[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, padding=kernel // 2), nn.GroupNorm(groups, cout), nn.Mish()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLMResidualBlock(nn.Module):
    """Two conv blocks; the condition scales and shifts the first block's output per channel."""

    def __init__(self, cin: int, cout: int, cond_dim: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(cin, cout, kernel, groups)
        self.conv2 = ConvBlock(cout, cout, kernel, groups)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.residual = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x [B, cin, T], cond [B, cond_dim] -> [B, cout, T]"""
        scale, bias = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.conv2(scale * self.conv1(x) + bias)
        return h + self.residual(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, down_dims: list[int], kernel: int,
                 groups: int, step_embedding_dim: int) -> None:
        super().__init__()
        e = step_embedding_dim
        self.step_encoder = nn.Sequential(
            SinusoidalEmbedding(e), nn.Linear(e, 4 * e), nn.Mish(), nn.Linear(4 * e, e)
        )
        cond_dim = cond_dim + e
        dims = [action_dim, *down_dims]
        pairs = list(zip(dims[:-1], dims[1:], strict=True))

        def block(cin, cout):
            return FiLMResidualBlock(cin, cout, cond_dim, kernel, groups)

        self.down = nn.ModuleList()
        for i, (cin, cout) in enumerate(pairs):
            last = i == len(pairs) - 1
            self.down.append(nn.ModuleList([
                block(cin, cout), block(cout, cout),
                nn.Identity() if last else nn.Conv1d(cout, cout, 3, stride=2, padding=1),
            ]))
        mid = down_dims[-1]
        self.mid = nn.ModuleList([block(mid, mid), block(mid, mid)])
        self.up = nn.ModuleList()
        for cin, cout in reversed(pairs[1:]):  # one upsample per downsample
            self.up.append(nn.ModuleList([
                block(2 * cout, cin), block(cin, cin),
                nn.ConvTranspose1d(cin, cin, 4, stride=2, padding=1),
            ]))
        self.final = nn.Sequential(
            ConvBlock(down_dims[0], down_dims[0], kernel, groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x: torch.Tensor, step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x [B, Tp, A], step [B], cond [B, C] -> predicted noise [B, Tp, A]"""
        cond = torch.cat([self.step_encoder(step), cond], dim=-1)
        h = x.transpose(1, 2)
        skips = []
        for res1, res2, downsample in self.down:
            h = res2(res1(h, cond), cond)
            skips.append(h)
            h = downsample(h)
        for res in self.mid:
            h = res(h, cond)
        for res1, res2, upsample in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = upsample(res2(res1(h, cond), cond))
        return self.final(h).transpose(1, 2)


class DiffusionModel(nn.Module):
    """Noise-prediction network plus its DDPM training loss and DDIM sampler."""

    name = "diffusion"

    def __init__(self, obs_dim: int, action_dim: int, horizons: Mapping, config: Mapping) -> None:
        super().__init__()
        self.obs_horizon, self.pred_horizon = horizons["observation"], horizons["prediction"]
        self.action_dim = action_dim
        self.net = ConditionalUnet1D(
            action_dim, obs_dim * self.obs_horizon, list(config["down_dims"]),
            config["kernel_size"], config["group_norm_groups"], config["step_embedding_dim"],
        )
        self.train_timesteps = int(config["train_timesteps"])
        self.inference_steps = int(config["inference_steps"])
        self.register_buffer(
            "alphas_cumprod", squared_cosine_alphas_cumprod(self.train_timesteps).float()
        )

    def loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs [B, To, D], actions [B, Tp, A] (normalized) -> scalar DDPM loss."""
        b = actions.shape[0]
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.train_timesteps, (b,), device=actions.device)
        a = self.alphas_cumprod[t].view(b, 1, 1)
        noisy = a.sqrt() * actions + (1 - a).sqrt() * noise
        return F.mse_loss(self.net(noisy, t, obs.flatten(1)), noise)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """obs [B, To, D] -> action chunk [B, Tp, A] by DDIM (eta = 0) with clipped x0."""
        b = obs.shape[0]
        cond = obs.flatten(1)
        x = torch.randn((b, self.pred_horizon, self.action_dim), generator=generator,
                        device="cpu").to(obs.device)
        stride = self.train_timesteps // self.inference_steps
        steps = list(range(0, self.train_timesteps, stride))[::-1]
        for i, t in enumerate(steps):
            a = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[steps[i + 1]] if i + 1 < len(steps) else a.new_tensor(1.0)
            eps = self.net(x, torch.full((b,), t, device=obs.device), cond)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x
