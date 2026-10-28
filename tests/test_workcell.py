import mujoco
import numpy as np
import pytest

from dexvision.sim.workcell import (
    ACTION_DIM,
    ACTUATOR_TARGETS,
    OBJECT_IDS,
    Placement,
    Workcell,
    WorkcellError,
)

LAYOUT = {
    "block_small": Placement((-0.12, -0.10), (0.015, 0.015)),
    "block_large": Placement((-0.12, 0.00), (0.015, 0.015)),
    "cylinder_short": Placement((-0.12, 0.10), (0.015, 0.015)),
    "cylinder_tall": Placement((-0.04, -0.10), (0.015, 0.015)),
    "puck_light": Placement((-0.04, 0.00), (0.015, 0.015)),
    "puck_heavy": Placement((-0.04, 0.10), (0.015, 0.015)),
}


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def hold(workcell, translation=(0, 0, 0)):
    action = np.zeros(ACTION_DIM)
    action[:3] = translation
    action[ACTUATOR_TARGETS] = workcell.neutral_actuator_targets
    return action


def random_actions(workcell, rng, n):
    actions = rng.uniform(-1, 1, size=(n, ACTION_DIM)) * 0.02
    actions[:, ACTUATOR_TARGETS] = rng.uniform(workcell.ctrl_low, workcell.ctrl_high, size=(n, 20))
    return actions


def test_reset_is_deterministic_per_seed(workcell):
    first = workcell.reset(3, LAYOUT)
    qpos = workcell.data.qpos.copy()
    second = workcell.reset(3, LAYOUT)
    np.testing.assert_array_equal(qpos, workcell.data.qpos)
    block = first.objects["block_small"].position
    np.testing.assert_array_equal(block, second.objects["block_small"].position)
    other = workcell.reset(4, LAYOUT)
    assert not np.allclose(block, other.objects["block_small"].position)


def test_episode_changes_state_only_through_ctrl_and_mocap(workcell, monkeypatch):
    """Replay the exact inputs on a copy with raw mj_step. Any hidden state write diverges."""
    workcell.reset(0, LAYOUT)
    reference = mujoco.MjData(workcell.model)
    mujoco.mj_copyData(reference, workcell.model, workcell.data)
    recorded = []
    original = workcell._apply_inputs

    def capture(targets, positions, orientations):
        recorded.append((targets.copy(), positions.copy(), orientations.copy()))
        original(targets, positions, orientations)

    monkeypatch.setattr(workcell, "_apply_inputs", capture)
    for action in random_actions(workcell, np.random.default_rng(0), 40):
        workcell.step(action)
        workcell.contacts()
        targets, positions, orientations = recorded[-1]
        reference.ctrl[:] = targets
        for position, orientation in zip(positions, orientations, strict=True):
            reference.mocap_pos[0 + workcell._target_mocap] = position
            reference.mocap_quat[workcell._target_mocap] = orientation
            mujoco.mj_step(workcell.model, reference)
        np.testing.assert_array_equal(reference.qpos, workcell.data.qpos)
        np.testing.assert_array_equal(reference.qvel, workcell.data.qvel)


def test_model_is_never_modified(workcell):
    fields = ("body_mass", "body_pos", "body_quat", "geom_size", "geom_friction", "geom_condim",
              "geom_contype", "actuator_gainprm", "actuator_biasprm", "eq_data", "body_gravcomp")
    before = {name: getattr(workcell.model, name).copy() for name in fields}
    rng = np.random.default_rng(1)
    for seed in range(3):
        workcell.reset(seed, LAYOUT if seed % 2 else {"block_small": LAYOUT["block_small"]})
        for action in random_actions(workcell, rng, 10):
            workcell.step(action)
    for name in fields:
        np.testing.assert_array_equal(before[name], getattr(workcell.model, name), err_msg=name)


def test_button_is_placed_at_reset_and_stays_put(workcell):
    face = workcell.reset(0, {}).button_position  # default place
    offset = np.array([0.0, 0.08, -0.05])
    rng = np.random.default_rng(2)
    observation = workcell.reset(0, {}, button_offset=offset)
    np.testing.assert_allclose(observation.button_position, face + offset, atol=1e-9)
    for action in random_actions(workcell, rng, 20):
        observation = workcell.step(action)
    rest = observation.button_position - np.array([observation.button_depth, 0.0, 0.0])
    np.testing.assert_allclose(rest, face + offset, atol=1e-9)
    # the next reset without an offset puts it back for every other task
    np.testing.assert_allclose(workcell.reset(0, {}).button_position, face, atol=1e-12)


def test_hand_tracks_command_in_free_space(workcell):
    workcell.reset(0, LAYOUT)
    for _ in range(10):
        observation = workcell.step(hold(workcell))
    measured, _ = workcell.measured_target_pose()
    assert np.linalg.norm(measured - observation.command_position) < 0.008


def test_blocked_hand_is_measured_not_commanded(workcell):
    """Drive the hand into the empty table: the measurement stops, the command leads by <= 1 cm."""
    workcell.reset(0, {})
    for _ in range(80):
        observation = workcell.step(hold(workcell, translation=(0.0, 0.0, -0.01)))
    measured, _ = workcell.measured_target_pose()
    lead = observation.command_position - measured
    assert lead[2] < -0.008  # the command is below where the hand actually is
    # The limit applies at command time; contact may push the hand back slightly during the step.
    assert np.linalg.norm(lead) <= 0.011
    assert observation.command_position[2] > -0.05  # stopped by the lead limit, not the workspace
    assert observation.fingertip_positions[:, 2].min() > -0.005  # nothing passes through the table
    touching = {c.body1 for c in workcell.contacts()} | {c.body2 for c in workcell.contacts()}
    assert "table" in touching


def test_actuator_targets_change_at_bounded_rate(workcell):
    workcell.reset(0, LAYOUT)
    limit = workcell.action_space.max_target_step
    previous = workcell.observe().actuator_targets
    for action in random_actions(workcell, np.random.default_rng(2), 30):
        current = workcell.step(action).actuator_targets
        assert np.all(np.abs(current - previous) <= limit + 1e-12)
        previous = current
    # 3 rad/s at 20 Hz; tendon actuators drive two joints, so twice that.
    assert sorted(set(np.round(limit, 6))) == [0.15, 0.3]


def test_objects_rest_stably_after_reset(workcell):
    start = workcell.reset(5, LAYOUT)
    for _ in range(20):
        observation = workcell.step(hold(workcell))
    for name in OBJECT_IDS:
        moved = np.linalg.norm(observation.objects[name].position - start.objects[name].position)
        assert moved < 1e-3, name
        assert np.linalg.norm(observation.objects[name].linear_velocity) < 1e-2, name


def test_objects_outside_layout_are_parked_out_of_reach(workcell):
    observation = workcell.reset(0, {"puck_light": LAYOUT["puck_light"]})
    for name in OBJECT_IDS:
        z = observation.objects[name].position[2]
        assert (z > 0) == (name == "puck_light"), name


def test_rejects_bad_inputs():
    fresh = Workcell()
    with pytest.raises(WorkcellError):
        fresh.step(np.zeros(ACTION_DIM))
    fresh.reset(0, {})
    with pytest.raises(WorkcellError):
        fresh.step(np.zeros(ACTION_DIM - 1))
    with pytest.raises(WorkcellError):
        fresh.step(np.full(ACTION_DIM, np.nan))
    with pytest.raises(WorkcellError):
        fresh.reset(0, {"teapot": LAYOUT["block_small"]})


def test_physics_invariants(workcell):
    """Values the honesty argument relies on. Changing them invalidates all results."""
    m = workcell.model
    gravcomp = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): m.body_gravcomp[b]
                for b in range(m.nbody) if m.body_gravcomp[b]}
    assert gravcomp == {"rh_forearm": 1.0}
    for name in OBJECT_IDS:
        geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        assert m.geom_condim[geom] == 4 and m.geom_contype[geom] == 1
    wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tray_left_wall_xp")
    assert m.geom_contype[wall] == 1
