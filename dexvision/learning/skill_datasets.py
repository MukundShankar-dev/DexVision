"""Named, causal Level 5 skill tensors with training-only normalization.

Resulting state row t is never the input to action t. MuJoCo forward kinematics
on reset / row t-1 supplies contacts and typed entity state, without stepping.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from dexvision.learning.datasets import quaternion_wxyz_to_rotation_6d
from dexvision.learning.training_manifest import (
    ReleaseSource, TrainingInputs, content_digest, select_streams, skill_interval,
)
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.logging.replay_demo import observation_schema_from_metadata
from dexvision.logging.visual_stream import named_joint_values


def rotation6d(q) -> np.ndarray:
    return quaternion_wxyz_to_rotation_6d(np.asarray(q, float).reshape(1, 4))[0]


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.r_[a[0] * b[0] - a[1:] @ b[1:],
                 a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])]


def relative_rotation(current: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Shortest left-composed rotation vector, invariant to quaternion signs."""
    a, b = np.asarray(current, float), np.asarray(prior, float)
    if (a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a, b]).all()
            or min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12):
        raise ValueError("Invalid action quaternion")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    q = quaternion_product(a, b * np.array([1, -1, -1, -1]))
    if q[0] < 0:
        q = -q
    length = np.linalg.norm(q[1:])
    return np.zeros(3) if length < 1e-12 else q[1:] * (2 * np.arctan2(length, q[0]) / length)


def feature_schema(plan: dict, skill: dict) -> tuple[tuple[str, ...], np.ndarray]:
    """Exact ordered input names and mask of columns that may be normalized."""
    names = [name for g in plan["observations"]["feature_groups"] for name in g["names"]]
    continuous = []
    binary_entity = ("entity.supported.", "entity.held.", "entity.contact_with_hand.",
                     "entity.valid_mask.", "entity.class.", "entity.linear_velocity_valid",
                     "entity.angular_velocity_valid")
    for name in names:
        continuous.append(not name.startswith(("phase.", "previous_safety_masks.", *binary_entity)))
    for field in skill["numeric_goal_layout"]:
        names.extend(f"goal.{field['name']}.{i}" for i in range(field["shape"][0]))
        continuous.extend([field["units"] not in {"bool", "one_hot"}] * field["shape"][0])
    if (len(set(names)) != len(names)
            or len(names) != skill["observation_width"] + skill["numeric_goal_width"]):
        raise ValueError("Invalid frozen feature schema")
    return tuple(names), np.asarray(continuous, bool)


def action_targets(applied: np.ndarray, prior: np.ndarray, safety: np.ndarray,
                   phases: tuple[str, ...], plan: dict, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Return [T, 26] bounded deltas or [T, 27] absolute reference targets/masks."""
    fields = plan["action"]["named_layout"]
    expected = (len(phases), len(fields))
    if any(a.shape != expected for a in (applied, prior, safety)):
        raise ValueError("Action/history/safety shapes disagree with named layout")
    if not np.isfinite(applied).all() or not np.isfinite(prior).all():
        raise ValueError("Non-finite action target or history")
    relevance = plan["action"]["phase_relevance"]
    mask = np.array([[relevance[p][f["group"]] for f in fields] for p in phases], bool) & ~safety.astype(bool)
    if model == "reference":
        return applied.copy(), mask.astype(float)
    if model != "baseline":
        raise ValueError("Only the frozen baseline and reference models are enabled")
    groups = {name: [f["index"] for f in fields if f["group"] == name]
              for name in ("base_position", "base_orientation", "wrist", "fingers")}
    caps = np.asarray([f["max_change_per_sample"] for f in fields], float)
    ordinary = [*groups["base_position"], *groups["wrist"], *groups["fingers"]]
    delta = (applied - prior) / caps
    qidx = groups["base_orientation"]
    rot = np.asarray([relative_rotation(a[qidx], p[qidx]) for a, p in zip(applied, prior, strict=True)])
    rot /= plan["action"]["maximum_angular_change_rad"]
    rotation_active = mask[:, qidx].all(axis=1)
    if np.any(np.linalg.norm(rot[rotation_active], axis=1) > 1 + 1e-5):
        raise ValueError("Expert rotation exceeds the frozen angular rate bound")
    targets = np.concatenate((delta[:, ordinary[:3]], rot, delta[:, ordinary[3:]]), axis=1)
    masks = np.concatenate((mask[:, ordinary[:3]],
                            np.repeat(mask[:, qidx].all(axis=1)[:, None], 3, axis=1),
                            mask[:, ordinary[3:]]), axis=1)
    if np.any(np.abs(targets[masks]) > 1 + 1e-5):
        raise ValueError("Expert delta exceeds the frozen head bounds; do not clip training data")
    targets[~masks] = 0.0
    return targets, masks.astype(float)


@dataclass(frozen=True)
class SkillEpisode:
    """One immutable skill interval; arrays have T rows and declared feature widths."""
    episode_id: str
    split: str
    session_id: str
    cell_id: str
    episode_digest: str
    inputs: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    frames: np.ndarray
    timestamps: np.ndarray
    phases: tuple[str, ...]
    normalization_interval: tuple[int, int] | None
    records: dict

    def __post_init__(self):
        size = len(self.frames)
        if (not size or self.inputs.ndim != 2 or self.targets.ndim != 2
                or self.masks.shape != self.targets.shape
                or any(len(a) != size for a in (self.inputs, self.targets, self.timestamps, self.phases))
                or not all(np.isfinite(a).all() for a in (self.inputs, self.targets, self.masks))
                or np.any(np.diff(self.frames) <= 0)
                or np.any((self.masks != 0) & (self.masks != 1))):
            raise ValueError("Invalid skill tensor shapes, frames, masks or finite values")
        for a in (self.inputs, self.targets, self.masks, self.frames, self.timestamps):
            a.setflags(write=False)


class SkillDataset(Dataset):
    """Torch-compatible samples with stable episode/frame identities."""

    def __init__(self, episodes: list[SkillEpisode], normalization: dict):
        self.episodes = tuple(sorted(episodes, key=lambda e: e.episode_id))
        if not self.episodes or len({e.split for e in episodes}) != 1:
            raise ValueError("A dataset must contain exactly one nonempty split")
        self.normalization = normalization
        mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
        if np.any(std <= 0) or not np.isfinite(np.r_[mean, std]).all():
            raise ValueError("Invalid normalization statistics")
        self.inputs = np.concatenate([(e.inputs - mean) / std for e in self.episodes]).astype(np.float32)
        self.targets = np.concatenate([e.targets for e in self.episodes]).astype(np.float32)
        self.masks = np.concatenate([e.masks for e in self.episodes]).astype(np.float32)
        self.identities = tuple((e.episode_id, int(f)) for e in self.episodes for f in e.frames)
        self.buckets = {}
        offset = 0
        for e in self.episodes:
            self.buckets.setdefault(e.cell_id, []).append(np.arange(offset, offset + len(e.frames)))
            offset += len(e.frames)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        eid, frame = self.identities[index]
        return {"inputs": self.inputs[index], "target": self.targets[index],
                "mask": self.masks[index], "episode_id": eid, "frame": frame}

    def sample_indices(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """Uniform cell, then episode, then eligible frame, with replacement."""
        cells = sorted(self.buckets)
        indices = []
        for _ in range(count):
            episodes = self.buckets[cells[int(rng.integers(len(cells)))]]
            frames = episodes[int(rng.integers(len(episodes)))]
            indices.append(frames[int(rng.integers(len(frames)))])
        return np.asarray(indices, dtype=np.int64)


def fit_normalization(episodes: list[SkillEpisode], names: tuple[str, ...], continuous: np.ndarray,
                      digests: dict, epsilon: float = 1e-8) -> dict:
    """Fit float64 population statistics from explicit train-owner intersections only."""
    values, owners = [], []
    for e in sorted(episodes, key=lambda e: e.episode_id):
        if e.split != "train":
            raise ValueError("Only training episodes may fit normalization")
        if e.normalization_interval is None:
            continue
        start, end = e.normalization_interval
        mask = (e.frames >= start) & (e.frames < end)
        if mask.any():
            values.append(e.inputs[mask])
            owners.append({"episode_id": e.episode_id, "episode_digest": e.episode_digest,
                           "frame_interval": [int(e.frames[mask][0]), int(e.frames[mask][-1]) + 1]})
    if not values or len({r["episode_id"] for r in owners}) != len(owners):
        raise ValueError("No unique training normalization owners")
    all_values = np.concatenate(values).astype(np.float64)
    if all_values.shape[1] != len(names) or continuous.shape != (len(names),):
        raise ValueError("Normalization feature schema mismatch")
    mean, std = all_values.mean(axis=0), all_values.std(axis=0, ddof=0)
    mean[~continuous] = 0
    std[(std < epsilon) | ~continuous] = 1
    result = {"version": "level5/normalization-v1", "ordered_feature_names": list(names),
              "count": len(all_values), "mean": mean.tolist(), "std": std.tolist(),
              "epsilon": epsilon, "variance_ddof": 0, "continuous_mask": continuous.tolist(),
              "training_episode_ids": [o["episode_id"] for o in owners],
              "training_frame_intervals": owners, **digests}
    result["normalization_digest"] = content_digest(result)
    return result


def _named_values(schema, arrays: dict, field: str, names: list[str], index: int) -> np.ndarray:
    layout = schema.layouts[field]
    columns = list(range(*layout.column_range)) if layout.column_range is not None else list(layout.column_indices)
    if len(set(layout.names)) != len(columns):
        raise ValueError(f"Duplicate or incomplete named layout: {field}")
    lookup = dict(zip(layout.names, columns, strict=True))
    return np.asarray(arrays[layout.source_array][index, [lookup[n] for n in names]], float)


def _entity_vector(world, entity_id: str, plan: dict) -> np.ndarray:
    entity = world.require_entity(entity_id)
    relation = world.relation_for(entity_id)
    classes = plan["observations"]["entity"]["class_encoding"]
    category = entity.class_id.removeprefix("rigid_")
    if category in {"spring_button", "pressable_button"}:
        category = "button"
    if category in {"receptacle", "planar_zone", "placement_slot"}:
        category = "target"
    if category not in classes:
        raise ValueError(f"Entity class outside frozen vocabulary: {category}")
    contact = any(entity_id in pair and any(n.startswith("rh_") for n in pair) for pair in world.contacts)
    return np.r_[entity.position, rotation6d(entity.orientation_wxyz),
                 entity.linear_velocity if entity.linear_velocity is not None else np.zeros(3),
                 entity.angular_velocity if entity.angular_velocity is not None else np.zeros(3),
                 relation.supported_by is not None, relation.held_by is not None, contact,
                 entity.confidence, world.timestamp - entity.timestamp, True,
                 [category == c for c in classes], entity.linear_velocity is not None,
                 entity.angular_velocity is not None]


def numeric_goal(skill: dict, row: dict, world) -> np.ndarray:
    """Encode typed numeric fields only; string ids resolve entities, never ordinals."""
    goal, name = row["typed_goal"], skill["skill"]
    if name == "reach_object":
        pose = np.asarray(goal["approach_pose"], float)
        result = np.r_[pose[:3], rotation6d(pose[3:])]
    elif name == "pick_object":
        height = next(c["value"] for c in skill["contract"]["success_metric"]["conditions"]
                      if c["field"] == "object_height_above_support_m")
        pose = goal.get("approach_pose")
        result = np.r_[height, pose[:3] if pose is not None else np.zeros(3),
                       rotation6d(pose[3:]) if pose is not None else np.zeros(6), pose is not None]
    elif name in {"place_held_object", "push_object_to_target"}:
        target_id = goal["target_id" if name == "place_held_object" else "target_zone"]
        target = world.require_entity(target_id)
        target_type = skill["target_type_mapping"][target_id]
        one_hot = [target_type == t for t in ("receptacle", "planar_zone", "placement_slot")]
        result = np.r_[target.position, rotation6d(target.orientation_wxyz), one_hot] if name == "place_held_object" else np.r_[target.position, one_hot]
    elif name == "press_button":
        result = np.array([goal["target_press_depth_m"], goal["target_pressed_state"]], float)
    else:
        raise ValueError(f"Unsupported skill: {name}")
    if result.shape != (skill["numeric_goal_width"],) or not np.isfinite(result).all():
        raise ValueError("Numeric goal does not match the frozen schema")
    return result


def _restore_initial(task, metadata: dict) -> None:
    """Restore archived poses, sizes and dynamics, including historical amendments."""
    initial = metadata["task_config"]["initial_state"]
    wc, env = task.workcell, task.env
    for name in (*wc.config.targets, *wc.config.fixtures):
        env.model.body(name).pos[:] = initial["entity_positions_m"][name]
    # Constructor already applies configured variation once. Saved object poses
    # replace its sampled positions; no newly sampled goal owns this episode.
    for spec in wc.config.objects:
        state = initial["objects"][spec.object_id]
        joint = env.model.joint(state["joint_name"])
        q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
        env.data.qpos[q:q + 7] = [*state["position_m"], *state["orientation_wxyz"]]
        env.data.qvel[v:v + 6] = [*state["linear_velocity_mps"], *state["angular_velocity_radps"]]
    env._mujoco.mj_forward(env.model, env.data)
    task.initial_world_state = wc.get_world_state()
    wc._initial_state = task.initial_world_state
    task.goal = dict(metadata["typed_goal"])
    task._phase = metadata["initial_online_phase"]
    task._configure_tasks()


def _causal_phase(task, world, frame: int) -> str:
    """Reuse the Level 4 physical task state machine on preceding states only.

    This phase can disagree with the expert's saved controller annotation; it
    contains no expert action, future transition, or retrospective label.
    """
    if frame:
        if task.skill_name == "pick_place_sequence":
            task._evaluate_pick_place(world)
        else:
            task._evaluate_single(world, task._task.evaluate(world))
    return task._phase


def vectorize_episode(inputs: TrainingInputs, row: dict, record: dict, *, model: str = "baseline") -> SkillEpisode:
    metadata = record["metadata"]
    for field in ("episode_id", "recording_session_id", "typed_goal", "observation_schema", "action_schema"):
        if metadata[field] != row[field]:
            raise ValueError(f"Episode metadata disagrees with frozen row: {field}")
    schema = observation_schema_from_metadata(metadata)
    schema.extract(record, time_steps=row["frames"])
    count = row["frames"]
    for field in ("applied_actions", "commanded_actions", "requested_actions", "prior_applied_actions",
                  "prior_commanded_actions", "safety_masks"):
        if record[field].shape != (count, 27) or not np.isfinite(record[field]).all():
            raise ValueError(f"Invalid action record: {field}")
    for current, prior in (("applied_actions", "prior_applied_actions"),
                           ("commanded_actions", "prior_commanded_actions")):
        if not np.array_equal(record[current][:-1], record[prior][1:]):
            raise ValueError("Prior action history does not match the preceding applied/commanded action")
    stamps = record["state_timestamps"]
    skew = inputs.requirements["quality_thresholds"]["max_state_action_timestamp_skew_s"]
    dt = inputs.plan["action"]["control_interval_s"]
    for field in ("state_timestamps", "action_timestamps", "task_timestamps", "timestamps"):
        value = record[field]
        if (value.shape != (count,) or not np.isfinite(value).all()
                or np.any(np.abs(value - stamps) > skew) or np.any(np.diff(value) <= 0)):
            raise ValueError("State/action timestamp alignment failed")
    if not np.allclose(np.diff(stamps), dt, atol=1e-8, rtol=0) or abs(stamps[0] - dt) > 1e-8:
        raise ValueError("Saved state interval differs from the frozen control interval")
    start, end = skill_interval(inputs.skill, row)
    groups = inputs.plan["observations"]["feature_groups"]
    by_group = {g["group"]: g for g in groups}
    vectors, phases = [], []
    entity_id = row["entity_id"]
    with WorkcellPilotTask(
        workcell_config=inputs.root / inputs.plan["inputs"]["workcell_config"]["path"],
        dataset_config=inputs.root / inputs.plan["inputs"]["dataset_config"]["path"],
        skill_name=row["skill"], goal_condition_id=row["cell_id"], seed=metadata["random_seed"],
        procedural_variation=metadata["task_config"].get("procedural_variation"),
    ) as task:
        _restore_initial(task, metadata)
        env = task.env
        initial_arrays = {"robot_states": np.zeros((1, record["robot_states"].shape[1]))}
        for field, values in (("robot_qpos", env.data.qpos), ("robot_qvel", env.data.qvel)):
            layout = schema.layouts[field]
            cols = list(range(*layout.column_range)) if layout.column_range is not None else list(layout.column_indices)
            # Verify MuJoCo order against saved names before assigning reset data.
            expected = named_joint_values(env.model, layout, record["robot_states"][0], velocity=field == "robot_qvel")
            if expected.shape != values.shape:
                raise ValueError("Reset state schema differs from the saved model")
            # Build reset values by MuJoCo names, not the saved column order.
            reset_values = []
            suffixes = ("vx", "vy", "vz", "wx", "wy", "wz") if field == "robot_qvel" else (
                "x", "y", "z", "qw", "qx", "qy", "qz")
            for name in layout.names:
                joint_name, _, suffix = name.partition("/")
                joint = env.model.joint(joint_name)
                address = int((joint.dofadr if field == "robot_qvel" else joint.qposadr)[0])
                reset_values.append(values[address + (suffixes.index(suffix) if suffix else 0)])
            initial_arrays["robot_states"][0, cols] = reset_values
        for frame in range(end):
            if frame:
                saved = record["robot_states"][frame - 1]
                env.data.qpos[:] = named_joint_values(env.model, schema.layouts["robot_qpos"], saved, velocity=False)
                env.data.qvel[:] = named_joint_values(env.model, schema.layouts["robot_qvel"], saved, velocity=True)
                env.data.time = float(stamps[frame - 1])
                env.set_mocap_pose(str(task.workcell.config.scene["hand_base_target"]),
                                   position=record["prior_applied_actions"][frame, :3],
                                   orientation_quat=record["prior_applied_actions"][frame, 3:7])
                env.set_joint_targets(dict(zip(metadata["finger_target_names"],
                                               record["prior_applied_actions"][frame, 7:], strict=True)))
                env._mujoco.mj_forward(env.model, env.data)
            world = task.workcell.get_world_state()
            phase = _causal_phase(task, world, frame)
            if frame < start:
                continue
            arrays, index = (record, frame - 1) if frame else (initial_arrays, 0)
            qnames = inputs.plan["observations"]["robot_fields"]["robot_qpos"]
            pose = _named_values(schema, arrays, "robot_qpos", qnames, index)
            parts = {"robot_position": pose[:3], "robot_rotation": rotation6d(pose[3:]),
                     "robot_velocity": _named_values(schema, arrays, "robot_qvel",
                         [n.removeprefix("robot.velocity.") for n in by_group["robot_velocity"]["names"]], index),
                     "prior_commanded_actions": record["prior_commanded_actions"][frame],
                     "prior_applied_actions": record["prior_applied_actions"][frame],
                     "previous_safety_masks": record["safety_masks"][frame - 1] if frame else np.zeros(27),
                     "online_phase": np.array([phase == p for p in inputs.plan["observations"]["phase_vocabulary"]]),
                     "entity": _entity_vector(world, entity_id, inputs.plan)}
            for field in ("finger_joint_positions", "finger_joint_velocities"):
                parts[field] = _named_values(schema, arrays, field,
                    [n.removeprefix(field + ".") for n in by_group[field]["names"]], index)
            for group in groups:
                if parts[group["group"]].shape != tuple(group["shape"]):
                    raise ValueError(f"Feature group shape mismatch: {group['group']}")
            vectors.append(np.r_[np.concatenate([parts[g["group"]] for g in groups]),
                                 numeric_goal(inputs.skill, row, world)])
            phases.append(phase)
    targets, masks = action_targets(record["applied_actions"][start:end],
                                    record["prior_applied_actions"][start:end],
                                    record["safety_masks"][start:end], tuple(phases), inputs.plan, model)
    owners = {o["episode_id"]: o for o in inputs.splits["train"]["normalization_inputs"]}
    owner = owners.get(row["episode_id"])
    records = {name: value[start:end].copy() for name, value in record.items()
               if isinstance(value, np.ndarray) and len(value) == count}
    records["saved_phase_disagreement_count"] = sum(p != str(s) for p, s in
        zip(phases, record["online_phases"][start:end], strict=True))
    return SkillEpisode(row["episode_id"], row["split"], row["recording_session_id"], row["cell_id"],
                        row["episode_digest"], np.asarray(vectors), targets, masks,
                        np.arange(start, end), stamps[start:end].copy(), tuple(phases),
                        tuple(owner["frame_interval"]) if owner else None, records)


def load_skill_datasets(inputs: TrainingInputs, *, release_root: Path | None = None,
                         model: str = "baseline") -> tuple[dict[str, SkillDataset], dict]:
    """Load training/validation expert tensors; held-out test arrays remain unread."""
    selected = select_streams(inputs)["expert_success"]
    rows = selected["train"] + selected["validation"]
    records = ReleaseSource(inputs, release_root).read_episode_records(rows)
    episodes = [vectorize_episode(inputs, r, records[r["episode_id"]], model=model) for r in rows]
    names, continuous = feature_schema(inputs.plan, inputs.skill)
    train = [e for e in episodes if e.split == "train"]
    normalization = fit_normalization(train, names, continuous, inputs.digests,
                                      inputs.plan["data"]["normalization"]["epsilon"])
    datasets = {s: SkillDataset([e for e in episodes if e.split == s], normalization)
                for s in ("train", "validation")}
    return datasets, normalization
