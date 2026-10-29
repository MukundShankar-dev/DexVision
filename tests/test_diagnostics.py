from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml
from test_vision import square_episodes, tiny_config, windows

from dexvision.data.demos import generate, load_data_config, load_episode
from dexvision.eval.diagnose import metric_summary, paired, phases, position_grid
from dexvision.eval.evaluator import camera_rig, run_episode
from dexvision.eval.probe import (
    BUCKETS,
    ScratchProbe,
    fit_probe,
    frame_buckets,
    load_probe_frames,
    offline_action_error,
    probe_frames,
    probe_report,
)
from dexvision.eval.stats import mcnemar_exact
from dexvision.learning.policy import LearnedPolicy
from dexvision.learning.train import Trained, train
from dexvision.policies import make_policy
from dexvision.sim.workcell import Workcell
from dexvision.tasks import TASK_NAMES, make_task
from dexvision.tasks.pick import pick_phase
from dexvision.tasks.spec import load_task_spec


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def rig_or_skip(workcell, cameras):
    try:
        return camera_rig(workcell, cameras)
    except Exception as error:  # no offscreen OpenGL on this machine
        pytest.skip(f"offscreen rendering unavailable: {error}")


# --- task spec variants -------------------------------------------------------------------------

def test_a_variant_replaces_only_the_keys_it_names(tmp_path):
    base = load_task_spec("pick").raw
    (tmp_path / "pick.yaml").write_text(yaml.safe_dump(dict(base)), encoding="utf-8")
    (tmp_path / "pick_x.yaml").write_text(yaml.safe_dump(
        {"extends": "pick", "observation": ["palm_position"], "cameras": ["policy_left"]}),
        encoding="utf-8")
    variant = load_task_spec(tmp_path / "pick_x.yaml")
    assert (variant.name, variant.skill) == ("pick_x", "pick")
    assert variant.observation == ("palm_position",) and variant.cameras == ("policy_left",)
    assert variant.raw["reset"] == base["reset"] and variant.horizon_steps == base["horizon_steps"]
    assert "extends" not in variant.raw


def test_pick_vision_sees_no_object_state_and_has_its_own_test_seeds():
    vision, state = load_task_spec("pick_vision"), load_task_spec("pick")
    assert not [f for f in vision.observation if f.startswith("object_")]
    assert vision.cameras and vision.skill == "pick"
    assert vision.raw["success"] == state.raw["success"]
    assert (vision.splits["train"], vision.splits["val"]) == (state.splits["train"],
                                                              state.splits["val"])
    start, stop = vision.splits["test"]
    for name in TASK_NAMES:
        if name != "pick_vision":
            other_start, other_stop = load_task_spec(name).splits["test"]
            assert stop <= other_start or other_stop <= start, name


# --- pick phases and metrics --------------------------------------------------------------------

@pytest.mark.parametrize(("success", "failure", "metrics", "phase"), [
    (True, None, {}, "success"),
    (False, "object_fell_off_table", {}, "fell_off_table"),
    (False, "timeout", {"first_contact_step": -1}, "no_contact"),
    (False, "timeout", {"first_contact_step": 9, "first_grasp_step": -1}, "contact_no_grasp"),
    (False, "timeout", {"first_contact_step": 9, "first_grasp_step": 30, "max_dwell_steps": 0},
     "grasped_not_lifted"),
    (False, "timeout", {"first_contact_step": 9, "first_grasp_step": 30, "max_dwell_steps": 4},
     "lifted_not_held"),
])
def test_pick_phase_is_the_furthest_stage_reached(success, failure, metrics, phase):
    assert pick_phase(success, failure, metrics) == phase


def test_expert_and_zero_episodes_get_consistent_phase_metrics(workcell):
    task = make_task("pick")
    expert = run_episode(workcell, task, make_policy("expert", workcell.action_space, "pick"),
                         100001)
    assert expert.success and expert.phase == "success"
    m = expert.metrics
    assert 0 < m["first_contact_step"] <= m["first_grasp_step"] < expert.steps
    assert m["max_dwell_steps"] == task.dwell_steps
    assert m["min_grasp_xy_m"] < 0.02 and m["shift_before_grasp_m"] < 0.03

    task.spec = replace(task.spec, horizon_steps=10)
    zero = run_episode(workcell, task, make_policy("zero", workcell.action_space, "pick"), 100001)
    assert zero.phase == "no_contact" and zero.metrics["first_grasp_step"] == -1
    assert zero.metrics["shift_before_grasp_m"] < 1e-3
    assert (zero.metrics["start_x_m"], zero.metrics["start_y_m"]) == (m["start_x_m"],
                                                                      m["start_y_m"])


# --- cameras in the evaluator and in demos ------------------------------------------------------

class SeesImages:
    """Zero action; records which observation keys it was given."""

    name = "sees_images"

    def __init__(self, action_space):
        self._zero = make_policy("zero", action_space, "pick")
        self.keys = set()

    def reset(self, context, seed):
        self._zero.reset(context, seed)

    def act(self, observation):
        self.keys |= set(observation)
        return self._zero.act(observation)


def test_vision_episodes_carry_images_and_match_state_episodes(workcell):
    vision, state = make_task("pick_vision"), make_task("pick")
    vision.spec = replace(vision.spec, horizon_steps=8)
    state.spec = replace(state.spec, horizon_steps=8)
    rig = rig_or_skip(workcell, vision.spec.cameras)
    policy = SeesImages(workcell.action_space)
    seen = run_episode(workcell, vision, policy, 100002, rig=rig)
    rig.close()
    assert {f"rgb/{c}" for c in vision.spec.cameras} <= policy.keys
    assert not [k for k in policy.keys if k.startswith("object_")]
    plain = run_episode(workcell, state, make_policy("zero", workcell.action_space, "pick"),
                        100002)
    assert seen == plain  # rendering only reads the simulation


def test_demos_with_cameras_store_images_and_workers_change_nothing(workcell, tmp_path):
    rig_or_skip(workcell, ("policy_left",)).close()
    config = dict(load_data_config("pick_vision"), episodes=2)
    task = make_task("pick")
    expert = make_policy("expert", workcell.action_space, "pick")
    one = generate(workcell, task, expert, config, tmp_path / "one")
    two = generate(workcell, task, expert, config, tmp_path / "two", workers=2)
    assert one["files"] == two["files"] and one["files"]
    episode = load_episode(next((tmp_path / "one").glob("episode_*.npz")))
    for camera in config["cameras"]:
        frames = episode[f"obs/rgb/{camera}"]
        assert frames.dtype == np.uint8 and frames.shape[0] == len(episode["action"])
    assert "obs/object_position" in episode  # the probe's labels


# --- report breakdowns --------------------------------------------------------------------------

def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(6, 0) == mcnemar_exact(0, 6)


def episode(seed, success, phase, x=0.0, y=0.0, shift=0.0):
    return {"seed": seed, "success": success, "failure": None if success else "timeout",
            "phase": phase, "metrics": {"start_x_m": x, "start_y_m": y,
                                        "shift_before_grasp_m": shift, "first_grasp_step": -1}}


def test_breakdowns_count_phases_cells_and_medians():
    episodes = [episode(0, True, "success", -0.09, -0.05, 0.001),
                episode(1, False, "no_contact", -0.09, 0.05, 0.0),
                episode(2, False, "contact_no_grasp", -0.03, 0.05, 0.04)]
    assert phases(episodes) == {"success": 1, "no_contact": 1, "contact_no_grasp": 1}
    grid = position_grid(episodes, (-0.06, 0.0), (0.04, 0.06))
    assert (grid[0][0].successes, grid[0][0].n) == (1, 1)
    assert (grid[0][2].n, grid[2][2].n, grid[1][1].n) == (1, 1, 0)
    summary = metric_summary(episodes, ["shift_before_grasp_m", "first_grasp_step"])
    assert summary["shift_before_grasp_m"] == {"success": 0.001, "failure": 0.02}
    assert summary["first_grasp_step"] == {"success": None, "failure": None}  # -1 is "never"


def test_paired_comparison_uses_shared_seeds_only():
    this = [episode(0, True, "success"), episode(1, False, "no_contact"),
            episode(2, False, "contact_no_grasp"), episode(9, True, "success")]
    reference = [episode(0, True, "success"), episode(1, True, "success"),
                 episode(2, False, "x")]
    p = paired(this, reference)
    assert (p.both, p.only_this, p.only_reference, p.neither) == (1, 0, 1, 1)
    assert p.reference_only_seeds == (1,) and p.reference_only_phases == {"no_contact": 1}
    with pytest.raises(ValueError):
        paired(this, [episode(5, True, "success")])


# --- blanking, keypoints, probes ----------------------------------------------------------------

def test_training_mean_image_is_saved_and_blanking_ignores_the_camera(tmp_path):
    w = windows(square_episodes(4, 4))
    trained = train("mlp", w, tiny_config(), steps=20, seed=0)
    np.testing.assert_array_equal(
        trained.image_mean["cam"], np.round(w.images["cam"].mean(axis=0)).astype(np.uint8))
    trained.save(tmp_path / "m.pt")
    loaded = Trained.load(tmp_path / "m.pt")
    np.testing.assert_array_equal(loaded.image_mean["cam"], trained.image_mean["cam"])

    def act(policy, frame):
        policy.reset(None, 0)
        return policy.act({"phase": np.zeros(1), "rgb/cam": frame})

    frames = [e["obs/rgb/cam"][0] for e in square_episodes(2, 4, seed=3)]
    blanked = LearnedPolicy(loaded, blank=("cam",))
    assert blanked.name == "mlp_blank-cam"
    np.testing.assert_array_equal(act(blanked, frames[0]), act(blanked, frames[1]))
    np.testing.assert_array_equal(act(blanked, frames[0]),
                                  act(LearnedPolicy(loaded), loaded.image_mean["cam"]))
    assert not np.array_equal(act(LearnedPolicy(loaded), frames[0]),
                              act(LearnedPolicy(loaded), frames[1]))
    with pytest.raises(ValueError):
        LearnedPolicy(loaded, blank=("other",))


def test_keypoints_lie_in_the_crop():
    trained = train("mlp", windows(square_episodes(2, 4)), tiny_config(), steps=2, seed=0)
    images = {"cam": torch.as_tensor(np.stack([e["obs/rgb/cam"][:2]
                                               for e in square_episodes(3, 4)]))}
    points = trained.model.vision.eval().keypoints(images)["cam"]
    assert points.shape == (3, 2, tiny_config()["vision"]["keypoints"], 2)
    assert points.abs().max() <= 1.0


def labelled_squares(n_episodes, seed=0):
    """Square episodes with probe labels: the square is 'the object', fingertips far away; the
    object lifts in the last step of each episode."""
    episodes = square_episodes(n_episodes, 6, seed)
    for i, e in enumerate(episodes):
        x, y = e["action"][0]
        position = np.tile([x, y, 0.0], (6, 1))
        position[-1, 2] = 0.05
        e["obs/object_position"] = position
        e["obs/fingertip_positions"] = np.full((6, 15), 9.0)
        e["target_object"] = np.array(["a", "b"][i % 2])
    return episodes


def test_frame_buckets():
    e = labelled_squares(1)[0]
    np.testing.assert_array_equal(frame_buckets(e), [0, 0, 0, 0, 0, 2])
    e["obs/fingertip_positions"][2] = np.tile(e["obs/object_position"][2], 5)
    assert BUCKETS[frame_buckets(e)[2]] == "near"


def test_probe_frames_from_files_match_in_memory(tmp_path):
    episodes = labelled_squares(3)
    paths = []
    for i, e in enumerate(episodes):
        paths.append(tmp_path / f"episode_{i:06d}.npz")
        np.savez_compressed(paths[-1], **e)
    a = probe_frames(episodes, ("cam",), ["a", "b"])
    b = load_probe_frames(paths, ("cam",), ["a", "b"])
    for field in ("position", "object_index", "bucket"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(a.images["cam"], b.images["cam"])


def test_scratch_probe_reads_position_from_images():
    objects = ["a", "b"]
    train_frames = probe_frames(labelled_squares(40), ("cam",), objects)
    val_frames = probe_frames(labelled_squares(10, seed=1), ("cam",), objects)
    model = ScratchProbe(("cam",), tiny_config()["vision"], len(objects))

    def inputs(frames):
        return lambda rows: {"cam": torch.as_tensor(frames.images["cam"][rows.numpy()])}

    scale = fit_probe(model, inputs(train_frames), train_frames, steps=400, batch=32, lr=3e-3,
                      device="cpu", seed=0)
    report = probe_report(model, inputs(val_frames), val_frames, scale)["all"]
    assert report["frames"] == 60
    assert report["position_error_cm_mean"] < 0.35 * report["mean_guess_error_cm"], report


def test_offline_action_error_is_small_on_training_data():
    episodes = labelled_squares(12)
    w = windows(episodes)
    trained = train("mlp", w, tiny_config(steps=600), seed=0)
    report = offline_action_error(trained, episodes, 50, seed=0)
    assert set(report) >= {"all", "far"} and report["all"]["windows"] == 50
    untrained = train("mlp", w, tiny_config(), steps=1, seed=0)
    before = offline_action_error(untrained, episodes, 50, seed=0)["all"]["all"]
    assert report["all"]["all"] < 0.5 * before
