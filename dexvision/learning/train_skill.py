"""Deterministic Level 5 training and resumable validation-candidate artifacts."""
from __future__ import annotations

import copy
import io
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.skill_datasets import SkillDataset, load_skill_datasets
from dexvision.learning.skill_models import SkillMLP, grouped_loss, model_schema
from dexvision.learning.training_manifest import (
    ReleaseSource, build_training_manifest, content_digest, digest_file,
    environment_manifest, load_training_inputs, safe_path,
)


def _write(path: Path, data: bytes) -> None:
    """Atomically replace a file within a newly created or explicitly resumed run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_bytes(data)
    temporary.replace(path)


def _json(path: Path, value: object) -> None:
    _write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _checkpoint(path: Path, payload: dict) -> None:
    stream = io.BytesIO()
    torch.save(payload, stream)
    _write(path, stream.getvalue())
    _write(path.with_suffix(".pt.sha256"), f"{digest_file(path)}  {path.name}\n".encode())


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    sidecar = path.with_suffix(".pt.sha256")
    if sidecar.read_text().strip() != f"{digest_file(path)}  {path.name}":
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    result = torch.load(path, map_location="cpu", weights_only=True)
    if result.get("version") != "level5/training-checkpoint-v1":
        raise ValueError("Unsupported skill checkpoint")
    return result


def _checksum_index(directory: Path) -> None:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    _write(directory / "SHA256SUMS", "".join(
        f"{digest_file(p)}  {p.relative_to(directory).as_posix()}\n" for p in files).encode())


def _verify_run(directory: Path) -> None:
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest_file(safe_path(directory, name)) != expected:
            raise ValueError(f"Run artifact changed: {name}")


def _dataset_digest(dataset: SkillDataset) -> str:
    import hashlib

    ownership = [{"episode_id": e.episode_id, "split": e.split, "session_id": e.session_id,
                  "cell_id": e.cell_id, "episode_digest": e.episode_digest}
                 for e in dataset.episodes]
    digest = hashlib.sha256(content_digest([dataset.identities, ownership]).encode())
    for value in (dataset.inputs, dataset.targets, dataset.masks):
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _rng_state(rng: np.random.Generator) -> dict:
    legacy = np.random.get_state()
    return {"sampler": copy.deepcopy(rng.bit_generator.state), "python": random.getstate(),
            "numpy": [legacy[0], legacy[1].tolist(), *legacy[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict, rng: np.random.Generator) -> None:
    rng.bit_generator.state = state["sampler"]
    random.setstate(state["python"])
    legacy = state["numpy"]
    np.random.set_state((legacy[0], np.array(legacy[1], dtype=np.uint32), *legacy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _offline_loss(model: SkillMLP, dataset: SkillDataset, settings: dict, device: str) -> float:
    model.eval()
    total = 0.0
    batch = settings["batch_size"]
    with torch.no_grad():
        for start in range(0, len(dataset), batch):
            end = min(start + batch, len(dataset))
            arrays = [torch.as_tensor(a[start:end], device=device)
                      for a in (dataset.inputs, dataset.targets, dataset.masks)]
            loss = grouped_loss(model(arrays[0]), arrays[1], arrays[2], settings["loss"]["group_weights"],
                                model=model.schema["model"])
            total += loss.item() * (end - start)
    return total / len(dataset)


def train_skill(datasets: dict[str, SkillDataset], *, schema: dict, settings: dict,
                manifest: dict, output_dir: str | Path, config_snapshot: dict,
                resume: bool = False, stop_after_steps: int | None = None) -> dict:
    """Train one seed, checkpointing every optimizer step, including partial epochs.

    A stop limit models interruption and does not change the run contract.
    Selection remains pending until validation rollout evidence is supplied.
    No test dataset is accepted by this interface.
    """
    if set(datasets) != {"train", "validation"}:
        raise ValueError("Training accepts train and validation only, never test tensors")
    for split, dataset in datasets.items():
        if any(e.split != split for e in dataset.episodes):
            raise ValueError("Dataset split label mismatch")
    train_ids = {e.episode_id for e in datasets["train"].episodes}
    train_sessions = {e.session_id for e in datasets["train"].episodes}
    if any(e.episode_id in train_ids or e.session_id in train_sessions for e in datasets["validation"].episodes):
        raise ValueError("Training/validation episode or session leakage")
    if datasets["train"].normalization != datasets["validation"].normalization:
        raise ValueError("Validation must use the same training-only normalization")
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError("stop_after_steps must be positive")
    if settings["sampling"] != "uniform_cell_then_episode_then_eligible_frame_with_replacement":
        raise ValueError("Unsupported sampling policy")
    if settings["scheduler"] is not None or settings["optimizer"]["name"] != "adam":
        raise ValueError("This frozen protocol uses Adam and no scheduler")
    if settings["batch_size"] <= 0 or settings["max_epochs"] <= 0:
        raise ValueError("Batch size and epochs must be positive")
    seed, device = manifest["seed"], manifest["device"]
    if device not in {"cpu", "cuda"} or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device unavailable; use --device cpu")
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
    rng = np.random.default_rng(seed)
    normalization = datasets["train"].normalization
    contract = {"schema": schema, "settings": settings, "manifest": manifest,
                "normalization": normalization, "config_snapshot": config_snapshot,
                "environment": environment_manifest(device),
                "tensors": {s: _dataset_digest(d) for s, d in datasets.items()}}
    contract_digest = content_digest(contract)
    model = SkillMLP(schema).to(device)
    opt = settings["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opt["learning_rate"], betas=tuple(opt["betas"]),
                                 eps=opt["eps"], weight_decay=opt["weight_decay"])
    directory = Path(output_dir)
    state = {"completed_epochs": 0, "step_in_epoch": 0, "global_step": 0,
             "epoch_loss_sum": 0.0, "epoch_sample_count": 0, "best_offline_loss": None,
             "best_offline_epoch": None, "bad_epochs": 0, "history": [], "candidates": {},
             "sampling_digest": content_digest([]), "finished": False}
    if resume:
        _verify_run(directory)
        checkpoint = load_checkpoint(directory / "last.pt")
        if checkpoint["contract_digest"] != contract_digest:
            raise ValueError("Resume inputs/config/schema/environment or sample assignment changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["training_state"]
        _restore_rng(checkpoint["rng"], rng)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory / "training_manifest.json", manifest)
        _json(directory / "normalization.json", normalization)
        _json(directory / "environment.json", contract["environment"])
        _write(directory / "config_snapshot.yaml", yaml.safe_dump(config_snapshot, sort_keys=True).encode())

    def payload() -> dict:
        return {"version": "level5/training-checkpoint-v1", "contract_digest": contract_digest,
                "schema": schema, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": None, "normalization": normalization, "manifest": manifest,
                "settings": settings, "training_state": copy.deepcopy(state), "rng": _rng_state(rng)}

    def save_progress() -> None:
        _checkpoint(directory / "last.pt", payload())
        _json(directory / "selection.json", {"status": "pending_validation_rollouts",
              "candidates": state["candidates"], "best_offline_epoch": state["best_offline_epoch"],
              "best_offline_loss": state["best_offline_loss"], "test_used": False})
        _checksum_index(directory)

    train = datasets["train"]
    batch_size = settings["batch_size"]
    steps_per_epoch = math.ceil(len(train) / batch_size)
    completed_this_call = 0
    while not state["finished"]:
        model.train()
        indices = train.sample_indices(rng, batch_size)
        identities = [train.identities[i] for i in indices]
        state["sampling_digest"] = content_digest([state["sampling_digest"], identities])
        arrays = [torch.as_tensor(a[indices], device=device) for a in (train.inputs, train.targets, train.masks)]
        optimizer.zero_grad(set_to_none=True)
        loss = grouped_loss(model(arrays[0]), arrays[1], arrays[2], settings["loss"]["group_weights"],
                            model=schema["model"])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        state["global_step"] += 1
        state["step_in_epoch"] += 1
        state["epoch_loss_sum"] += loss.item() * batch_size
        state["epoch_sample_count"] += batch_size
        if state["step_in_epoch"] == steps_per_epoch:
            validation = _offline_loss(model, datasets["validation"], settings, device)
            if not math.isfinite(validation):
                raise ValueError("Non-finite validation loss")
            epoch = state["completed_epochs"] + 1
            improved = state["best_offline_loss"] is None or validation < (
                state["best_offline_loss"] - settings["early_stop"]["minimum_improvement"])
            state["history"].append({"epoch": epoch,
                "training_loss": state["epoch_loss_sum"] / state["epoch_sample_count"],
                "validation_offline_loss": validation})
            state.update(completed_epochs=epoch, step_in_epoch=0, epoch_loss_sum=0.0, epoch_sample_count=0)
            if improved:
                state.update(best_offline_epoch=epoch, best_offline_loss=validation, bad_epochs=0)
                _checkpoint(directory / "best_offline.pt", payload())
            else:
                state["bad_epochs"] += 1
            state["finished"] = epoch >= settings["max_epochs"] or state["bad_epochs"] >= settings["early_stop"]["patience_epochs"]
            if epoch in settings["selection"]["candidate_epochs"] or state["finished"]:
                path = directory / "validation_candidates" / f"epoch_{epoch:04d}.pt"
                _checkpoint(path, payload())
                state["candidates"][str(epoch)] = {"path": path.relative_to(directory).as_posix(),
                    "checkpoint_sha256": digest_file(path), "offline_validation_loss": validation}
            if state["finished"]:
                epoch = state["best_offline_epoch"]
                if str(epoch) not in state["candidates"]:
                    path = directory / "validation_candidates" / f"epoch_{epoch:04d}.pt"
                    _checkpoint(path, load_checkpoint(directory / "best_offline.pt"))
                    state["candidates"][str(epoch)] = {"path": path.relative_to(directory).as_posix(),
                        "checkpoint_sha256": digest_file(path), "offline_validation_loss": state["best_offline_loss"]}
        save_progress()
        completed_this_call += 1
        if stop_after_steps is not None and completed_this_call >= stop_after_steps:
            break
    selection = json.loads((directory / "selection.json").read_text())
    return {"output_dir": str(directory), **state, "selection_status": selection["status"]}


def select_checkpoint(output_dir: str | Path, reports: list[dict]) -> dict:
    """Select using externally supplied validation results; no rollout implementation.

    Every candidate must have evidence bound to its hash, seed and frozen run
    inputs. The caller owns executing the frozen validation matrix in 5.2+.
    """
    directory = Path(output_dir)
    _verify_run(directory)
    checkpoint = load_checkpoint(directory / "last.pt")
    state = checkpoint["training_state"]
    if not state["finished"] or (directory / "selected.pt").exists():
        raise ValueError("Selection requires finished training and no previous selected checkpoint")
    candidates = state["candidates"]
    if len(reports) != len(candidates) or {str(r["epoch"]) for r in reports} != set(candidates):
        raise ValueError("Validation evidence must cover every candidate exactly once")
    metrics = ("safety_violation_count", "success_rate", "worst_cell_success_rate")
    for report in reports:
        candidate = candidates[str(report["epoch"])]
        if (report["split"] != "validation" or report["checkpoint_sha256"] != candidate["checkpoint_sha256"]
                or report["manifest_digest"] != checkpoint["manifest"]["manifest_digest"]
                or report["seed"] != checkpoint["manifest"]["seed"]
                or report["validation_matrix_sha256"] != checkpoint["manifest"]["validation_matrix_sha256"]
                or report["rollout_count"] != checkpoint["manifest"]["validation_rollout_count"]):
            raise ValueError("Selection evidence has incompatible provenance or is held-out test data")
        if any(not isinstance(report[k], (int, float)) or not math.isfinite(report[k]) or report[k] < 0 for k in metrics):
            raise ValueError("Invalid validation selection metrics")
        terminal_error = report["normalized_terminal_error"]
        if terminal_error is None:
            if report.get("terminal_error_unavailable_reason") not in {"rejected_precondition", "missing_terminal_metric"}:
                raise ValueError("Unavailable terminal error requires an explicit reason")
        elif not isinstance(terminal_error, (int, float)) or not math.isfinite(terminal_error) or terminal_error < 0:
            raise ValueError("Invalid normalized terminal error")
        if report["success_rate"] > 1 or report["worst_cell_success_rate"] > 1:
            raise ValueError("Success rates must lie in [0, 1]")
    chosen = min(reports, key=lambda r: (r["safety_violation_count"], -r["success_rate"],
        -r["worst_cell_success_rate"], r["normalized_terminal_error"] if r["normalized_terminal_error"] is not None else math.inf,
        candidates[str(r["epoch"])]["offline_validation_loss"], r["epoch"]))
    source = directory / candidates[str(chosen["epoch"])]["path"]
    _checkpoint(directory / "selected.pt", load_checkpoint(source))
    result = {"status": "selected_from_validation", "epoch": chosen["epoch"],
              "source_checkpoint_sha256": chosen["checkpoint_sha256"], "reports": reports,
              "test_used": False, "qualification_claim": None}
    _json(directory / "selection.json", result)
    _checksum_index(directory)
    return result


def run_training_command(config: Path, *, release_root: Path | None = None, output_dir: Path | None = None,
                         seed: int | None = None, device: str = "cpu", model: str = "baseline",
                         dry_run: bool = False, resume: bool = False,
                         stop_after_steps: int | None = None) -> dict:
    inputs = load_training_inputs(config)
    seed = inputs.plan["training"]["seeds"][0] if seed is None else seed
    if seed not in inputs.plan["training"]["seeds"]:
        raise ValueError("Choose one of the three frozen training seeds")
    if device not in {"cpu", "cuda"} or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device unavailable; use --device cpu")
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(inputs.plan["training"]["deterministic_algorithms"])
    schema = model_schema(inputs.plan, inputs.skill, model)
    manifest = build_training_manifest(inputs, seed=seed, device=device, model=model)
    if resume and output_dir is None:
        raise ValueError("--resume requires the existing --output-dir")
    if output_dir is None:
        from datetime import datetime, timezone

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = inputs.root / inputs.plan["artifacts"]["run_template"].format(
            protocol_slug=inputs.plan["artifacts"]["protocol_slug"], skill=inputs.skill["skill"],
            model=model, seed=seed, run_id=run_id)
    if dry_run:
        # Check one immutable metadata file per selected episode without fitting
        # statistics, loading test actions, constructing MuJoCo, or writing a run.
        names = [f"data/demos/level4/{r['source_path']}/metadata.json"
                 for s in ("train", "validation") for r in inputs.splits[s]["episodes"]
                 if r["episode_id"] in {a["episode_id"] for a in manifest["assignments"][s]}]
        ReleaseSource(inputs, release_root).read_many(names)
        return {"dry_run": True, "skill": inputs.skill["skill"], "device": device, "seed": seed,
                "model": model, "input_width": len(schema["input_names"]),
                "model_parameter_count": sum(p.numel() for p in SkillMLP(schema).parameters()),
                "split_episode_counts": {s: len(a) for s, a in manifest["assignments"].items()},
                "split_frame_counts": {s: sum(a["frame_interval"][1] - a["frame_interval"][0] for a in rows)
                                       for s, rows in manifest["assignments"].items()},
                "stream_counts": manifest["stream_counts"], "inputs": inputs.plan["inputs"],
                "digests": inputs.digests, "output_dir": str(output_dir),
                "normalization_fitted": False, "selection_status": "awaiting_validation_rollouts"}
    datasets, _ = load_skill_datasets(inputs, release_root=release_root, model=model)
    return train_skill(datasets, schema=schema, settings=inputs.plan["training"], manifest=manifest,
                       output_dir=output_dir, config_snapshot=inputs.config_snapshot,
                       resume=resume, stop_after_steps=stop_after_steps)
