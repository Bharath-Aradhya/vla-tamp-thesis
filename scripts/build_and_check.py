"""Build a fixed-base G1 pick scene (table + cube + head camera) and sanity-check it.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\build_and_check.py

Step A (build): copies Menagerie's g1_with_hands.xml into models/g1_fixed.xml with
  - the free base joint removed (pelvis is fixed in the world),
  - the 'stand' keyframe adjusted to match (first 7 qpos values dropped),
  - a head camera added to the torso,
  - meshes pointed at the Menagerie assets folder.
and writes models/scene_pick.xml (floor, table, cube).
Step B (check): loads the scene, holds the stand pose for 2 s, prints where the
cube and right wrist are, and saves head_cam.png and overview.png.

The original Menagerie files are never modified.
NOTE: written without being able to run MuJoCo in the authoring environment;
table/cube/camera positions are first guesses to be tuned from the renders.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"
OUT = ROOT / "models"

CUBE_POSE = "0.38 -0.15 0.775 1 0 0 0"  # x y z + quaternion (w x y z)
CAMERA = '<camera name="head_cam" pos="0.10 0 0.42" xyaxes="0 -1 0 0.766 0 0.643" fovy="75"/>'

SCENE = """<mujoco model="g1_pick_scene">
  <include file="g1_fixed.xml"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0.5 0 2.5" dir="0 0 -1"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <!-- table: top surface at z = 0.75, front edge at x = 0.25 -->
    <geom name="table" type="box" pos="0.45 0 0.375" size="0.2 0.4 0.375" rgba="0.6 0.45 0.3 1"/>
    <!-- 5 cm cube resting on the table, in front of the right hand -->
    <body name="cube" pos="0.38 -0.15 0.775">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" rgba="0.85 0.15 0.15 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected exactly one occurrence of: {old!r} (found {text.count(old)})"
    return text.replace(old, new)


def build():
    text = SRC.read_text(encoding="utf-8")

    # 1. meshes: point to the original assets folder (absolute path, forward slashes)
    assets = (SRC.parent / "assets").as_posix()
    text = replace_once(text, 'meshdir="assets"', f'meshdir="{assets}"')

    # 2. fixed base: remove the free joint
    text = replace_once(text, '<freejoint name="floating_base_joint"/>', "")

    # 3. keyframe: drop the 7 free-joint values (x y z + quaternion) from qpos
    def fix_qpos(m):
        vals = m.group(2).split()
        assert len(vals) == 50, f"unexpected qpos length {len(vals)}"
        # robot joints only (43 values) + the cube's free-joint pose (x y z + quaternion),
        # because the scene adds the cube after the robot and keyframes must cover every joint
        return m.group(1) + " ".join(vals[7:] + CUBE_POSE.split()) + m.group(3)

    text, n = re.subn(r'(<key name="stand"\s+qpos=")([^"]*)(")', fix_qpos, text)
    assert n == 1, "could not find the 'stand' keyframe"

    # 4. head camera on the torso, right after the torso IMU site
    text, n = re.subn(r'(<site name="imu_in_torso"[^>]*/>)', r"\1\n            " + CAMERA, text)
    assert n == 1, "could not find the torso IMU site"

    OUT.mkdir(exist_ok=True)
    (OUT / "g1_fixed.xml").write_text(text, encoding="utf-8")
    (OUT / "scene_pick.xml").write_text(SCENE, encoding="utf-8")
    print("Wrote", OUT / "g1_fixed.xml")
    print("Wrote", OUT / "scene_pick.xml")


def check():
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(OUT / "scene_pick.xml"))
    data = mujoco.MjData(model)
    print(f"\nnq={model.nq} nv={model.nv} nu={model.nu} ncam={model.ncam}")

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
    for _ in range(int(2.0 / model.opt.timestep)):
        data.ctrl[:] = model.key_ctrl[key]
        mujoco.mj_step(model, data)

    cube = data.body("cube").xpos.copy()
    wrist = data.body("right_wrist_yaw_link").xpos.copy()
    print("cube position (x y z):        ", np_fmt(cube), " (resting on table: z ~ 0.775)")
    print("right wrist position (x y z): ", np_fmt(wrist))
    print("wrist -> cube vector:         ", np_fmt(cube - wrist), f" distance {((cube - wrist) ** 2).sum() ** 0.5:.2f} m")

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="head_cam")
    Image.fromarray(renderer.render()).save(ROOT / "head_cam.png")

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, -0.05, 0.9]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 135, -20
    renderer.update_scene(data, camera=cam)
    Image.fromarray(renderer.render()).save(ROOT / "overview.png")
    print("\nSaved head_cam.png and overview.png in", ROOT)


def np_fmt(v):
    return "[" + " ".join(f"{x:+.3f}" for x in v) + "]"


if __name__ == "__main__":
    build()
    if "--build-only" not in sys.argv:
        check()
