"""Training loop, EMA weights and checkpoints shared by both models.

A checkpoint is the run record: method config, observation fields, normalizers fitted on the
training data, EMA weights, step, dataset hash and git provenance.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch import nn

from dexvision import REPO_ROOT
from dexvision.learning.dataset import DemoWindows
from dexvision.learning.diffusion import DiffusionModel
from dexvision.learning.mlp import MLPModel
from dexvision.learning.normalize import Normalizer
from dexvision.provenance import code_version

METHOD_CONFIG = REPO_ROOT / "configs" / "learning" / "method.yaml"
MODELS = {"diffusion": DiffusionModel, "mlp": MLPModel}


def load_method_config(path: Path = METHOD_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_model(method: str, obs_dim: int, action_dim: int, config: Mapping) -> nn.Module:
    if method not in MODELS:
        raise KeyError(f"Unknown method '{method}'. Available: {sorted(MODELS)}")
    return MODELS[method](obs_dim, action_dim, config["horizons"], config[method])


@dataclass
class Trained:
    """Everything needed to act: EMA model, normalizers and the run record."""

    method: str
    model: nn.Module
    obs_norm: Normalizer
    action_norm: Normalizer
    fields: tuple[str, ...]
    config: dict
    step: int
    record: dict = field(default_factory=dict)  # dataset hash, provenance, skill

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "method": self.method, "config": self.config, "fields": list(self.fields),
            "obs_norm": self.obs_norm.state(), "action_norm": self.action_norm.state(),
            "obs_dim": len(self.obs_norm.low), "action_dim": len(self.action_norm.low),
            "model": self.model.state_dict(), "step": self.step, "record": self.record,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> Trained:
        data = torch.load(path, map_location=device, weights_only=False)
        model = build_model(data["method"], data["obs_dim"], data["action_dim"], data["config"])
        model.load_state_dict(data["model"])
        model.to(device).eval()
        return cls(data["method"], model, Normalizer.from_state(data["obs_norm"]),
                   Normalizer.from_state(data["action_norm"]), tuple(data["fields"]),
                   data["config"], data["step"], data["record"])


def ema_decay(step: int, power: float, max_decay: float) -> float:
    """Warm-up EMA decay (as in Diffusion Policy): small early, approaching ``max_decay``."""
    return min(max(1 - (1 + step) ** -power, 0.0), max_decay)


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def train(
    method: str,
    windows: DemoWindows,
    config: Mapping,
    *,
    steps: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    record: Mapping | None = None,
    checkpoint: Callable[[Trained], None] | None = None,
    log: Callable[[int, float], None] | None = None,
) -> Trained:
    """Train ``method`` on ``windows``. Returns the EMA model at the last step.

    ``checkpoint`` is called with the EMA model every ``checkpoint_every`` steps and at the end.
    """
    horizons = config["horizons"]
    if (windows.obs_horizon, windows.pred_horizon) != (horizons["observation"],
                                                        horizons["prediction"]):
        raise ValueError("Window horizons differ from the method config")
    cfg = config["training"]
    steps = int(steps if steps is not None else cfg["steps"])
    if torch.device(device).type == "cpu":
        # NNPACK is slower than the default path for short 1-D sequences (measured ~2x on M1).
        torch.backends.nnpack.set_flags(False)
    torch.manual_seed(seed)
    batches = torch.Generator().manual_seed(seed)

    obs_norm = Normalizer.fit(windows.observations)
    action_norm = Normalizer.fit(windows.actions)
    observations = torch.as_tensor(obs_norm.encode(windows.observations), dtype=torch.float32,
                                   device=device)
    actions = torch.as_tensor(action_norm.encode(windows.actions), dtype=torch.float32,
                              device=device)
    obs_index = torch.as_tensor(windows.obs_index, device=device)
    action_index = torch.as_tensor(windows.action_index, device=device)

    model = build_model(method, observations.shape[1], actions.shape[1], config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  betas=tuple(cfg["adam_betas"]), weight_decay=cfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, cfg["warmup_steps"], steps))
    trained = Trained(method, ema, obs_norm, action_norm, windows.fields, dict(config), 0,
                      {**(record or {}), "code": code_version(), "seed": seed,
                       "episodes": windows.n_episodes, "windows": len(windows)})

    model.train()
    for step in range(1, steps + 1):
        i = torch.randint(len(windows), (cfg["batch_size"],), generator=batches).to(device)
        loss = model.loss(observations[obs_index[i]], actions[action_index[i]])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        decay = ema_decay(step, cfg["ema_power"], cfg["ema_max_decay"])
        with torch.no_grad():
            for e, p in zip(ema.parameters(), model.parameters(), strict=True):
                e.lerp_(p, 1 - decay)
            for e, b in zip(ema.buffers(), model.buffers(), strict=True):
                e.copy_(b)
        if log:
            log(step, float(loss))
        if checkpoint and (step % cfg["checkpoint_every"] == 0 or step == steps):
            trained.step = step
            checkpoint(trained)
    trained.step = steps
    return trained
