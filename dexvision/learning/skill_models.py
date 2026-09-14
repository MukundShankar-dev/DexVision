"""Frozen Level 5 MLPs and explicit action heads; no rollout supervisor."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dexvision.learning.skill_datasets import feature_schema, quaternion_product

GROUP_WIDTHS = (3, 3, 2, 18)
GROUP_NAMES = ("base_position", "base_orientation", "wrist", "fingers")


def model_schema(plan: dict, skill: dict, model: str = "baseline") -> dict:
    if model not in {"baseline", "reference"}:
        raise ValueError("Model escalation and expert residuals are not enabled")
    names, _ = feature_schema(plan, skill)
    full = [f["name"] for f in plan["action"]["named_layout"]]
    output = [*full[:3], "base_rotation_vector/x", "base_rotation_vector/y", "base_rotation_vector/z", *full[7:]] if model == "baseline" else full
    return {"version": "level5/policy-schema-v1", "input_names": list(names),
            "observation_width": skill["observation_width"], "goal_width": skill["numeric_goal_width"],
            "output_names": output, "full_action_names": full,
            "observation_contract": plan["observations"]["version"],
            "action_contract": plan["action"]["version"], "model": model,
            "architecture": plan["models"][model]}


class SkillMLP(nn.Module):
    """[B, named input width] -> [B, 26 bounded delta / 27 reference values]."""

    def __init__(self, schema: dict):
        super().__init__()
        self.schema = schema
        architecture = schema["architecture"]
        if architecture["activation"] != "relu":
            raise ValueError("Frozen skill MLP activation must be ReLU")
        width = len(schema["input_names"])
        layers = []
        for hidden in architecture["hidden_sizes"]:
            if not isinstance(hidden, int) or hidden <= 0:
                raise ValueError("Hidden widths must be positive integers")
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        if schema["model"] == "baseline":
            if architecture["heads"] != dict(zip(
                ("base_translation", "base_rotation_vector", "wrist", "fingers"), GROUP_WIDTHS, strict=True
            )):
                raise ValueError("Action heads disagree with frozen schema")
            self.heads = nn.ModuleList(nn.Linear(width, n) for n in GROUP_WIDTHS)
        elif schema["model"] == "reference":
            self.heads = nn.ModuleList([nn.Linear(width, len(schema["output_names"]))])
        else:
            raise ValueError("Unsupported model")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != len(self.schema["input_names"]):
            raise ValueError("Policy input shape disagrees with named schema")
        hidden = self.trunk(inputs)
        if self.schema["model"] == "reference":
            return self.heads[0](hidden)
        groups = [torch.tanh(head(hidden)) for head in self.heads]
        # Componentwise tanh alone permits a sqrt(3)-times-too-large rotation.
        groups[1] = groups[1] / torch.linalg.vector_norm(groups[1], dim=1, keepdim=True).clamp_min(1)
        return torch.cat(groups, dim=1)


def grouped_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 group_weights: dict, *, model: str = "baseline") -> torch.Tensor:
    """Average each relevant group independently, then its configured group weight."""
    widths = GROUP_WIDTHS if model == "baseline" else (3, 4, 2, 18)
    if prediction.shape != target.shape or mask.shape != target.shape or target.shape[1] != sum(widths):
        raise ValueError("Loss target/mask shape mismatch")
    weights = [float(group_weights[n]) for n in GROUP_NAMES]
    if any(not np.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Loss group weights must be nonnegative and nonzero")
    losses, active = [], []
    offset = 0
    for width, weight in zip(widths, weights, strict=True):
        selection = slice(offset, offset + width)
        count = mask[:, selection].sum(dim=1)
        loss = ((prediction[:, selection] - target[:, selection]).square() * mask[:, selection]).sum(dim=1)
        losses.append(loss / count.clamp_min(1) * weight)
        active.append((count > 0).to(loss.dtype) * weight)
        offset += width
    return (torch.stack(losses, dim=1).sum(dim=1) /
            torch.stack(active, dim=1).sum(dim=1).clamp_min(torch.finfo(prediction.dtype).eps)).mean()


def decode_delta(prediction: np.ndarray, prior: np.ndarray, phase: str, plan: dict) -> np.ndarray:
    """Expand bounded heads to full actions; absolute safety checks belong to the supervisor."""
    prediction, prior = np.asarray(prediction, float), np.asarray(prior, float)
    if (prediction.shape != (26,) or prior.shape != (27,)
            or not np.isfinite(np.r_[prediction, prior]).all()
            or np.any(np.abs(prediction) > 1 + 1e-6)
            or np.linalg.norm(prediction[3:6]) > 1 + 1e-6):
        raise ValueError("Invalid bounded action prediction")
    fields = plan["action"]["named_layout"]
    caps = np.array([f["max_change_per_sample"] for f in fields])
    result = prior.copy()
    result[:3] += prediction[:3] * caps[:3]
    result[7:] += prediction[6:] * caps[7:]
    vector = prediction[3:6] * plan["action"]["maximum_angular_change_rad"]
    angle = np.linalg.norm(vector)
    delta = np.r_[np.cos(angle / 2), vector * (np.sin(angle / 2) / angle if angle > 1e-12 else 0.5)]
    if np.linalg.norm(prior[3:7]) < 1e-12:
        raise ValueError("Prior action quaternion is zero")
    q = quaternion_product(delta, prior[3:7] / np.linalg.norm(prior[3:7]))
    result[3:7] = q if q @ prior[3:7] >= 0 else -q
    mask = np.array([plan["action"]["phase_relevance"][phase][f["group"]] for f in fields])
    result[~mask] = prior[~mask]
    return result
