import mujoco
import numpy as np
import pytest
import yaml

from dexvision.eval.evaluator import camera_rig
from dexvision.sim.appearance import (
    FACTORS,
    AppearanceConfig,
    AppearanceSampler,
    delta_e,
    texture_bank,
)
from dexvision.sim.cameras import CameraConfig
from dexvision.sim.workcell import Placement, Workcell
from dexvision.tasks.spec import TaskSpecError, load_task_spec

LAYOUT = {"puck_light": Placement((-0.06, 0.0), (0.03, 0.03))}
TRAIN = (0, 100000)
CAMERAS = ("policy_right", "policy_left")
VISUAL_FIELDS = ("mat_rgba", "mat_texid", "mat_texrepeat", "geom_rgba", "geom_matid", "light_dir",
                 "light_diffuse", "light_ambient", "light_specular", "tex_data")


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


@pytest.fixture(scope="module")
def sampler(workcell):
    return AppearanceSampler(workcell.model, FACTORS, TRAIN)


def rig_or_skip(workcell, appearance=()):
    try:
        return camera_rig(workcell, CAMERAS, appearance, TRAIN)
    except Exception as error:  # no offscreen OpenGL on this machine
        pytest.skip(f"offscreen rendering unavailable: {error}")


def run(workcell, rig, seed, steps=10):
    """Frames after each step of holding still, and the final state."""
    workcell.reset(seed, LAYOUT)
    if rig:
        rig.start_episode(seed)
    frames = []
    for _ in range(steps):
        workcell.step(workcell.action_space.hold())
        if rig:
            frames.append(rig.render())
    return frames, workcell.data.qpos.copy(), workcell.data.qvel.copy()


def test_texture_bank_leaves_physics_identical_to_the_bare_model(workcell):
    bare = mujoco.MjModel.from_xml_path(str(workcell.model_path))
    for name in ("body_mass", "body_inertia", "body_pos", "geom_size", "geom_friction",
                 "geom_solref", "geom_solimp", "dof_damping", "jnt_stiffness",
                 "actuator_gainprm", "actuator_biasprm", "eq_data", "opt.timestep"):
        owner, field = (name.split(".") if "." in name else ("", name))
        a = getattr(getattr(bare, owner) if owner else bare, field)
        b = getattr(getattr(workcell.model, owner) if owner else workcell.model, field)
        np.testing.assert_array_equal(a, b, err_msg=name)
    trajectories = []
    for model in (bare, workcell.model):
        data, rng = mujoco.MjData(model), np.random.default_rng(3)
        for step in range(1500):
            if step % 50 == 0:
                data.ctrl[:] = rng.uniform(*model.actuator_ctrlrange.T)
            mujoco.mj_step(model, data)
        trajectories.append((data.qpos.copy(), data.qvel.copy()))
    np.testing.assert_array_equal(trajectories[0][0], trajectories[1][0])
    np.testing.assert_array_equal(trajectories[0][1], trajectories[1][1])


def test_nominal_frames_equal_the_bare_models(workcell):
    rig = rig_or_skip(workcell)
    frames, _, _ = run(workcell, rig, 4, steps=1)
    rig.close()
    bare = mujoco.MjModel.from_xml_path(str(workcell.model_path))
    data = mujoco.MjData(bare)
    data.qpos[:], data.qvel[:] = workcell.data.qpos, workcell.data.qvel
    data.mocap_pos[:], data.mocap_quat[:] = workcell.data.mocap_pos, workcell.data.mocap_quat
    mujoco.mj_forward(bare, data)
    config = CameraConfig.load()
    bare.vis.quality.shadowsize = config.shadow_map_px
    renderer = mujoco.Renderer(bare, config.size_px, config.size_px)
    option = mujoco.MjvOption()
    for group in config.hidden_geom_groups:
        option.geomgroup[group] = 0
    for name in CAMERAS:
        renderer.update_scene(data, camera=name, scene_option=option)
        np.testing.assert_array_equal(renderer.render(), frames[-1][name], err_msg=name)
    renderer.close()


def test_varied_appearance_changes_frames_not_physics_or_model(workcell):
    before = {name: getattr(workcell.model, name).copy() for name in VISUAL_FIELDS}
    varied = rig_or_skip(workcell, FACTORS)
    a, qpos_varied, qvel_varied = run(workcell, varied, 7)
    b, _, _ = run(workcell, varied, 7)
    c, _, _ = run(workcell, varied, 8)
    varied.close()
    nominal = rig_or_skip(workcell)
    d, qpos_nominal, qvel_nominal = run(workcell, nominal, 7)
    nominal.close()
    _, qpos_plain, qvel_plain = run(workcell, None, 7)
    assert np.array_equal(qpos_varied, qpos_nominal) and np.array_equal(qpos_varied, qpos_plain)
    assert np.array_equal(qvel_varied, qvel_nominal) and np.array_equal(qvel_varied, qvel_plain)
    for name in CAMERAS:
        assert all(np.array_equal(x[name], y[name]) for x, y in zip(a, b, strict=True))
        assert not np.array_equal(a[-1][name], c[-1][name])
        assert not np.array_equal(a[-1][name], d[-1][name])
    for name in VISUAL_FIELDS:
        np.testing.assert_array_equal(before[name], getattr(workcell.model, name), err_msg=name)


def test_draws_are_deterministic_per_seed_and_differ_across_seeds(sampler):
    assert sampler.draw(5) == sampler.draw(5)
    assert sampler.draw(5) != sampler.draw(6)


def test_each_factor_has_its_own_stream(workcell, sampler):
    full = sampler.draw(11)
    alone = {f: AppearanceSampler(workcell.model, [f], TRAIN).draw(11) for f in FACTORS}
    assert alone["lighting"].lighting == full.lighting and not alone["lighting"].looks
    assert alone["surfaces"].lighting is None and alone["objects"].lighting is None
    merged = {**alone["surfaces"].looks, **alone["objects"].looks}
    assert merged == dict(full.looks)


def test_held_out_textures_appear_only_outside_train_seeds(sampler):
    config = sampler.config
    def textures(seeds):
        return {look.texture for s in seeds for look in sampler.draw(s).looks.values()}
    train = textures(range(0, 300))
    val = textures(range(100000, 100300))
    assert not any(config.held_out(k) for k in train)
    assert any(config.held_out(k) for k in val)
    assert -1 in train and len(train) > 1


def test_texture_bank_is_greyscale_tiles_in_the_luminance_range():
    config = AppearanceConfig.load()
    bank = texture_bank(config)
    lo, hi = config.bank["luminance"]
    assert bank.shape == (config.bank_size, config.bank["size_px"], config.bank["size_px"])
    assert bank.min() >= round(255 * lo) and bank.max() <= round(255 * hi)
    assert len({k.tobytes() for k in bank}) == config.bank_size
    assert np.array_equal(bank, texture_bank(config))


def test_describe_reports_light_textures_and_contrast(sampler):
    appearance = sampler.draw(100003)
    out = sampler.describe(appearance, "puck_light")
    assert {"key_brightness", "table_texture", "object_texture",
            "object_table_delta_e"} <= set(out)
    assert out["object_table_delta_e"] >= 0
    assert delta_e((0.2, 0.3, 0.4), (0.2, 0.3, 0.4)) == 0
    assert delta_e((0, 0, 0), (1, 1, 1)) == pytest.approx(100, abs=0.5)


def test_spec_rejects_appearance_without_cameras(tmp_path):
    raw = yaml.safe_load(load_task_spec("pick").path.read_text(encoding="utf-8"))
    (tmp_path / "pick.yaml").write_text(yaml.safe_dump({**raw, "appearance": ["lighting"]}))
    with pytest.raises(TaskSpecError):
        load_task_spec(tmp_path / "pick.yaml")
    raw = {**raw, "appearance": ["lighting"], "cameras": list(CAMERAS)}
    (tmp_path / "pick_vision.yaml").write_text(yaml.safe_dump(raw))
    assert load_task_spec(tmp_path / "pick_vision.yaml").appearance == ("lighting",)
