import mujoco
import numpy as np
import pytest

from dexvision.sim.cameras import CameraConfig, CameraRig
from dexvision.sim.workcell import Placement, Workcell

LAYOUT = {"block_small": Placement((-0.06, 0.0), (0.03, 0.03))}


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


@pytest.fixture(scope="module")
def rig(workcell):
    try:
        rig = CameraRig(workcell)
    except Exception as error:  # no offscreen OpenGL on this machine
        pytest.skip(f"offscreen rendering unavailable: {error}")
    yield rig
    rig.close()


def run(workcell, rig, seed, steps=10):
    """Images after each step of holding still, and the final state."""
    workcell.reset(seed, LAYOUT)
    images = []
    for _ in range(steps):
        workcell.step(workcell.action_space.hold())
        if rig:
            images.append(rig.render())
    return images, workcell.data.qpos.copy(), workcell.data.qvel.copy()


def test_images_are_square_rgb_per_camera(workcell, rig):
    workcell.reset(0, LAYOUT)
    images = rig.render()
    size = rig.config.size_px
    assert set(images) == set(CameraConfig.load().cameras)
    for image in images.values():
        assert image.shape == (size, size, 3) and image.dtype == np.uint8
    first, second = images.values()
    assert not np.array_equal(first, second)


def test_images_are_deterministic_per_seed(workcell, rig):
    a, _, _ = run(workcell, rig, 5)
    b, _, _ = run(workcell, rig, 5)
    c, _, _ = run(workcell, rig, 6)
    for name in rig.config.cameras:
        assert all(np.array_equal(x[name], y[name]) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[-1][rig.config.cameras[0]], c[-1][rig.config.cameras[0]])


def test_rendering_does_not_change_the_simulation(workcell, rig):
    _, qpos_rendered, qvel_rendered = run(workcell, rig, 3)
    _, qpos_plain, qvel_plain = run(workcell, None, 3)
    assert np.array_equal(qpos_rendered, qpos_plain)
    assert np.array_equal(qvel_rendered, qvel_plain)


def test_camera_poses_never_change(workcell, rig):
    ids = [mujoco.mj_name2id(workcell.model, mujoco.mjtObj.mjOBJ_CAMERA, n)
           for n in rig.config.cameras]
    before = workcell.model.cam_pos[ids].copy(), workcell.model.cam_quat[ids].copy()
    run(workcell, rig, 1)
    run(workcell, rig, 2)
    assert np.array_equal(workcell.model.cam_pos[ids], before[0])
    assert np.array_equal(workcell.model.cam_quat[ids], before[1])


def test_goal_marker_is_not_rendered(workcell, rig):
    workcell.reset(0, LAYOUT)
    without = rig.render()
    workcell.set_goal_marker(np.array([-0.06, 0.06, 0.02]))  # in view of both cameras
    mujoco.mj_forward(workcell.model, workcell.data)
    with_marker = rig.render()
    for name in rig.config.cameras:
        assert np.array_equal(without[name], with_marker[name])
