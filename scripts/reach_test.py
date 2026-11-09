"""First reach-and-grasp test: move the right hand to the cube, close the fingers, lift.

Run from the project root (D:\\vla-tamp-thesis), after build_and_check.py:
    python scripts\\reach_test.py

What it does
  1. Holds the stand pose for 1 s.
  2. Uses inverse kinematics (IK) on the 7 right-arm joints to move a 'grasp point' on the
     hand to a pre-grasp pose beside the cube, then to the cube (side approach).
  3. Closes the fingers, holds, then lifts 10 cm.
  4. Prints the IK errors and the cube height at each stage; saves reach_summary_HHMMSS.png
     (rows = stages, columns = view from the robot's right side and from the front).

This is an exploratory test, NOT expected to succeed on the first run. The grasp point,
hand orientation and thumb closing direction below are educated guesses taken from the
model file; the images show how to correct them.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "scene_pick.xml"

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

# --- tunable guesses ---------------------------------------------------------------------
# Hand frame (= right_wrist_yaw_link): x points along the fingers, fingers curl toward +y,
# index and middle sit side by side along z. Grasp point = middle of the closing region.
GRASP_OFFSET = np.array([0.14, 0.03, 0.0])
# Desired hand orientation in the world: fingers point forward (+x), palm faces left (+y), z up.
R_TARGET = np.eye(3)
GRASP_DZ = 0.02            # raise the grasp point above the cube centre so the hand clears the table
ROT_WEIGHT = 0.3           # orientation matters less than position in this first test
CLOSE = {                  # hand joint targets when closing (thumb direction is a guess)
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.0,
    "right_hand_thumb_2_joint": -1.0,
    "right_hand_index_0_joint": 1.2,
    "right_hand_index_1_joint": 1.4,
    "right_hand_middle_0_joint": 1.2,
    "right_hand_middle_1_joint": 1.4,
}
# -----------------------------------------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)
key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(model, data, key)
data.ctrl[:] = model.key_ctrl[key]


def act(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


arm_act = [act(n) for n in ARM]
arm_joint = [model.joint(n) for n in ARM]
qadr = np.array([j.qposadr[0] for j in arm_joint])
dadr = np.array([j.dofadr[0] for j in arm_joint])
lo = np.array([j.range[0] for j in arm_joint])
hi = np.array([j.range[1] for j in arm_joint])
wrist = model.body("right_wrist_yaw_link").id
cube_body = model.body("cube").id


def step(seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def move(ctrl_target, seconds):
    """Smoothly interpolate all actuator targets, stepping the physics."""
    start = data.ctrl.copy()
    n = int(seconds / model.opt.timestep)
    for i in range(n):
        a = (i + 1) / n
        a = a * a * (3 - 2 * a)
        data.ctrl[:] = (1 - a) * start + a * ctrl_target
        mujoco.mj_step(model, data)


def solve_ik(q_start, target_pos, iters=400):
    """Damped least-squares IK for the 7 arm joints: grasp point -> target_pos, hand axes -> R_TARGET."""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    q = q_start.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        d.qpos[qadr] = q
        mujoco.mj_forward(model, d)
        R = d.xmat[wrist].reshape(3, 3)
        p = d.xpos[wrist] + R @ GRASP_OFFSET
        mujoco.mj_jac(model, d, jacp, jacr, p, wrist)
        e_pos = target_pos - p
        e_rot = 0.5 * sum(np.cross(R[:, i], R_TARGET[:, i]) for i in range(3))
        if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(e_rot) < 0.03:
            break
        J = np.vstack([jacp[:, dadr], ROT_WEIGHT * jacr[:, dadr]])
        err = np.concatenate([e_pos, ROT_WEIGHT * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q = np.clip(q + 0.5 * dq, lo, hi)
    return q, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))


def is_arm(name):
    return any(k in name for k in ("right_shoulder", "right_elbow", "right_wrist", "right_hand"))


def report(point, q_cmd):
    """Compare what the simulation actually did with what IK asked for."""
    R = data.xmat[wrist].reshape(3, 3)
    p = data.xpos[wrist] + R @ GRASP_OFFSET
    track = np.degrees(np.abs(data.qpos[qadr] - q_cmd))
    j = int(np.argmax(track))
    print(f"      simulated grasp point is {np.linalg.norm(point - p)*100:.1f} cm from the target "
          f"(at x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f})")
    print(f"      fingers point along world direction {np.round(R[:, 0], 2)}  (forward would be [1 0 0])")
    print(f"      worst arm-joint tracking error {track[j]:.1f} deg ({ARM[j]})")
    pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        if is_arm(b1) != is_arm(b2):
            pairs.add(f"{b1} <-> {b2}")
    if pairs:
        print("      arm/hand touching:", "; ".join(sorted(pairs)))


def go_to(point, seconds, label):
    q, e_pos, e_rot = solve_ik(data.qpos[qadr].copy(), np.asarray(point, float))
    flag = "" if e_pos < 0.01 else "   <-- NOT REACHED"
    print(f"  {label:<22s} IK position error {e_pos*100:5.1f} cm, orientation error {e_rot:.2f}{flag}")
    target = data.ctrl.copy()
    target[arm_act] = q
    move(target, seconds)
    step(0.3)
    report(np.asarray(point, float), q)


renderer = mujoco.Renderer(model, height=480, width=640)


def snap(label, rows):
    imgs = []
    for az in (90, 180):  # 90: from the robot's right side, 180: from the front
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.2, -0.2, 0.9]
        cam.distance, cam.azimuth, cam.elevation = 1.0, az, -15
        renderer.update_scene(data, camera=cam)
        imgs.append(Image.fromarray(renderer.render()).resize((400, 300)))
    row = Image.new("RGB", (800, 300))
    row.paste(imgs[0], (0, 0))
    row.paste(imgs[1], (400, 0))
    ImageDraw.Draw(row).text((6, 6), label, fill=(255, 255, 255))
    rows.append(row)


def cube_pos():
    return data.xpos[cube_body].copy()


rows = []
step(1.0)
c0 = cube_pos()
print(f"Cube at start: z = {c0[2]:.3f} m")
grasp = c0 + np.array([0.0, 0.0, GRASP_DZ])

print("Reaching (raise the hand clear of the table first, then come in from above):")
go_to([0.12, -0.25, 0.80], 1.5, "raise hand by the body")
go_to([0.30, -0.25, 0.92], 1.5, "move forward above table")
PRE = np.array([-0.06, -0.10, 0.05])
go_to(grasp + PRE, 1.5, "pre-grasp (beside cube)")
snap("1 pre-grasp", rows)
for f in (0.5, 1.0):
    go_to(grasp + (1 - f) * PRE, 1.0, f"approach {int(f*100)}%")
step(0.3)
snap("2 at cube", rows)
print(f"  cube moved during approach: {np.linalg.norm(cube_pos() - c0)*100:.1f} cm")

print("Closing fingers:")
target = data.ctrl.copy()
for name, val in CLOSE.items():
    target[act(name)] = val
move(target, 1.5)
step(0.5)
snap("3 fingers closed", rows)

print("Lifting 10 cm:")
go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, "lift")
step(0.5)
snap("4 after lift", rows)

c1 = cube_pos()
lifted = c1[2] - c0[2]
print(f"\nCube height change after lift: {lifted*100:+.1f} cm   (success would be about +8 to +10 cm)")
print("Result:", "cube lifted with the hand" if lifted > 0.05 else "cube NOT lifted (expected on a first try)")

summary = Image.new("RGB", (800, 300 * len(rows)))
for i, r in enumerate(rows):
    summary.paste(r, (0, 300 * i))
out_name = f"reach_summary_{datetime.now():%H%M%S}.png"
summary.save(ROOT / out_name)
print("Saved", out_name, "in", ROOT)
