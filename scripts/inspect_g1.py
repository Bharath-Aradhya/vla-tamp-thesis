"""Month-1 platform check for the thesis: inspect the Menagerie G1 (with hands).

Usage (run locally, where MuJoCo can be installed):
    pip install mujoco numpy pillow
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
    python inspect_g1.py mujoco_menagerie/unitree_g1/scene_with_hands.xml

NOTE: written without being able to run it in the authoring environment
(MuJoCo could not be installed there), so expect to fix small things.
"""
import sys

import mujoco
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "mujoco_menagerie/unitree_g1/scene_with_hands.xml"
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

print(f"nq={model.nq} nv={model.nv} nu={model.nu} ncam={model.ncam} nbody={model.nbody}")
print(f"timestep={model.opt.timestep}")

# --- right arm + hand: joints, ranges, actuator gains -----------------------
print("\nRight arm / hand actuators:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name and name.startswith("right_") and ("shoulder" in name or "elbow" in name
                                                or "wrist" in name or "hand" in name):
        jid = model.actuator_trnid[i, 0]
        lo, hi = model.jnt_range[jid]
        print(f"  {name:32s} range=[{lo:+.2f}, {hi:+.2f}] kp={model.actuator_gainprm[i, 0]:.0f}")

# --- set the 'stand' keyframe, settle, and check the robot stays up ----------
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
for _ in range(int(2.0 / model.opt.timestep)):
    data.ctrl[:] = model.key_ctrl[key_id] if key_id >= 0 else 0
    mujoco.mj_step(model, data)
print(f"\nPelvis height after 2s holding 'stand': {data.qpos[2]:.3f} m (free base!)")

# --- offscreen rendering: free camera, since the model ships no head camera --
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.3, 0.0, 0.9]
cam.distance, cam.azimuth, cam.elevation = 1.6, 150, -15
renderer.update_scene(data, camera=cam)
img = renderer.render()
try:
    from PIL import Image
    Image.fromarray(img).save("g1_render.png")
    print("Saved g1_render.png", img.shape)
except ImportError:
    print("Rendered image shape:", img.shape, "(install pillow to save it)")
