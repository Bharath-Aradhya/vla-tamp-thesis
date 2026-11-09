"""Automatic search for a grasp that lifts the cube.

Run from the project root (D:\\vla-tamp-thesis), after build_and_check.py:
    python scripts\\grasp_sweep.py

It repeats the reach-close-lift sequence from reach_test.py for many combinations of
  - where the 'grasp point' sits across the palm (offset y), and
  - how the thumb is positioned when the fingers close (3 thumb joints),
without rendering, and ranks them by how high the cube ends up after the lift.
Then it re-runs the best one with pictures and saves best_grasp_HHMMSS.png.

Expect a few minutes. Progress is printed after every trial.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import itertools
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "scene_pick.xml"

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FINGERS = {"right_hand_index_0_joint": 1.2, "right_hand_index_1_joint": 1.4,
           "right_hand_middle_0_joint": 1.2, "right_hand_middle_1_joint": 1.4}

# --- search grid ----------------------------------------------------------------------
OFFSET_Y = [0.04, 0.055, 0.07]            # grasp point across the palm (m)
THUMB_0 = [-0.8, 0.0, 0.8]
THUMB_1 = [-1.0, -0.3, 0.7]
THUMB_2 = [-1.6, -0.8, 0.0]
GRASP_DX = 0.14                            # grasp point along the fingers (m)
GRASP_DZ = 0.02
PRE = np.array([-0.06, -0.10, 0.05])
R_TARGET = np.eye(3)
ROT_WEIGHT = 0.3
# ----------------------------------------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)
key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
act = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
arm_act = [act(n) for n in ARM]
arm_joint = [model.joint(n) for n in ARM]
qadr = np.array([j.qposadr[0] for j in arm_joint])
dadr = np.array([j.dofadr[0] for j in arm_joint])
lo = np.array([j.range[0] for j in arm_joint])
hi = np.array([j.range[1] for j in arm_joint])
wrist = model.body("right_wrist_yaw_link").id
cube_body = model.body("cube").id
_renderer = None


def step(seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def move(ctrl_target, seconds):
    start = data.ctrl.copy()
    n = int(seconds / model.opt.timestep)
    for i in range(n):
        a = (i + 1) / n
        a = a * a * (3 - 2 * a)
        data.ctrl[:] = (1 - a) * start + a * ctrl_target
        mujoco.mj_step(model, data)


def solve_ik(q_start, target_pos, offset, iters=400):
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    q = q_start.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        d.qpos[qadr] = q
        mujoco.mj_forward(model, d)
        R = d.xmat[wrist].reshape(3, 3)
        p = d.xpos[wrist] + R @ offset
        mujoco.mj_jac(model, d, jacp, jacr, p, wrist)
        e_pos = target_pos - p
        e_rot = 0.5 * sum(np.cross(R[:, i], R_TARGET[:, i]) for i in range(3))
        if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(e_rot) < 0.03:
            break
        J = np.vstack([jacp[:, dadr], ROT_WEIGHT * jacr[:, dadr]])
        err = np.concatenate([e_pos, ROT_WEIGHT * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q = np.clip(q + 0.5 * dq, lo, hi)
    return q, float(np.linalg.norm(e_pos))


def go_to(point, seconds, offset):
    q, e_pos = solve_ik(data.qpos[qadr].copy(), np.asarray(point, float), offset)
    target = data.ctrl.copy()
    target[arm_act] = q
    move(target, seconds)
    step(0.3)
    return e_pos


def snap(label, rows):
    global _renderer
    if _renderer is None:
        _renderer = mujoco.Renderer(model, height=480, width=640)
    imgs = []
    for az in (90, 180):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.2, -0.2, 0.9]
        cam.distance, cam.azimuth, cam.elevation = 1.0, az, -15
        _renderer.update_scene(data, camera=cam)
        imgs.append(Image.fromarray(_renderer.render()).resize((400, 300)))
    row = Image.new("RGB", (800, 300))
    row.paste(imgs[0], (0, 0))
    row.paste(imgs[1], (400, 0))
    ImageDraw.Draw(row).text((6, 6), label, fill=(255, 255, 255))
    rows.append(row)


def run_trial(offset_y, thumb, render=False):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    mujoco.mj_forward(model, data)
    step(1.0)
    offset = np.array([GRASP_DX, offset_y, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, GRASP_DZ])
    rows = []

    go_to([0.12, -0.25, 0.80], 1.5, offset)
    go_to([0.30, -0.25, 0.92], 1.5, offset)
    go_to(grasp + PRE, 1.5, offset)
    if render:
        snap("1 pre-grasp", rows)
    go_to(grasp + 0.5 * PRE, 1.0, offset)
    reach_err = go_to(grasp, 1.0, offset)
    if render:
        snap("2 at cube", rows)

    target = data.ctrl.copy()
    for name, val in FINGERS.items():
        target[act(name)] = val
    target[act("right_hand_thumb_0_joint")] = thumb[0]
    target[act("right_hand_thumb_1_joint")] = thumb[1]
    target[act("right_hand_thumb_2_joint")] = thumb[2]
    move(target, 1.5)
    step(0.5)
    if render:
        snap("3 fingers closed", rows)

    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset)
    step(1.0)
    c1 = data.xpos[cube_body].copy()
    if render:
        snap("4 after lift", rows)
    lift = c1[2] - c0[2]
    palm = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset
    return {"lift": float(lift), "dist": float(np.linalg.norm(c1 - palm)),
            "reach_err": reach_err, "rows": rows}


def main():
    combos = list(itertools.product(OFFSET_Y, THUMB_0, THUMB_1, THUMB_2))
    print(f"{len(combos)} trials. Each prints: grasp offset y | thumb joints 0,1,2 | cube lift | reach error")
    results = []
    t0 = time.time()
    for i, (oy, t0_, t1_, t2_) in enumerate(combos, 1):
        r = run_trial(oy, (t0_, t1_, t2_))
        results.append((r["lift"], oy, (t0_, t1_, t2_), r))
        print(f"[{i:2d}/{len(combos)}] y={oy:.3f} thumb=({t0_:+.1f},{t1_:+.1f},{t2_:+.1f})  "
              f"lift {r['lift']*100:+5.1f} cm  reach err {r['reach_err']*100:4.1f} cm  "
              f"({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda x: -x[0])
    print("\nTop 5 by cube lift (success means about +8 to +10 cm):")
    for lift, oy, th, r in results[:5]:
        print(f"  lift {lift*100:+5.1f} cm | offset y {oy:.3f} | thumb {th} | cube-to-hand distance {r['dist']*100:.1f} cm")

    lift, oy, th, _ = results[0]
    print(f"\nRe-running the best combination with pictures: offset y={oy}, thumb={th}")
    r = run_trial(oy, th, render=True)
    summary = Image.new("RGB", (800, 300 * len(r["rows"])))
    for i, row in enumerate(r["rows"]):
        summary.paste(row, (0, 300 * i))
    name = f"best_grasp_{datetime.now():%H%M%S}.png"
    summary.save(ROOT / name)
    print(f"Best trial lifted the cube by {r['lift']*100:+.1f} cm. Saved {name} in {ROOT}")


if __name__ == "__main__":
    main()
