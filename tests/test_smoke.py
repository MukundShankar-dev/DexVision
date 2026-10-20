import mujoco

from dexvision import ASSETS_DIR


def test_workcell_scene_loads():
    model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "mujoco" / "workcell_scene.xml"))
    assert model.nu == 20  # Shadow Hand actuators, including two wrist joints
