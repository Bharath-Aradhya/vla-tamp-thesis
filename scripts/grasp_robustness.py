"""Robustness test for the best grasp found by grasp_sweep.py.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\grasp_robustness.py

Repeats the reach-close-lift sequence 30 times with the cube at random positions on the
table and random rotations (the hand is told where the cube is each time), then holds
the cube for 1.5 s more. A trial counts as a SUCCESS if the cube
  - rose at least 6 cm and stayed up after the extra hold, and
  - did not slip in the hand (moved less than 2 cm relative to the wrist during the hold).
Prints one line per trial and a summary, and saves robustness_HHMMSS.csv.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
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

# --- the grasp found by grasp_sweep.py -------------------------------------------------
OFFSET_Y = 0.04                            # grasp point across the palm (m)
THUMB = (0.0, 0.7, -1.6)                   # thumb joints 0, 1, 2 when closing
N_TRIALS = 30
X_RANGE = (0.34, 0.42)                     # random cube position (m), robot frame
Y_RANGE = (-0.20, -0.10)
YAW_RANGE_DEG = 20                         # random cube rotation about the vertical (+/- degrees)
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
cube_qadr = model.joint("cube_joint").qposadr[0]
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


def run_trial(cube_x, cube_y, yaw):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, 0.775]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)
    offset = np.array([GRASP_DX, OFFSET_Y, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, GRASP_DZ])

    go_to([0.12, -0.25, 0.80], 1.5, offset)
    go_to([0.30, -0.25, 0.92], 1.5, offset)
    go_to(grasp + PRE, 1.5, offset)
    go_to(grasp + 0.5 * PRE, 1.0, offset)
    reach_err = go_to(grasp, 1.0, offset)

    target = data.ctrl.copy()
    for name, val in FINGERS.items():
        target[act(name)] = val
    for j, v in enumerate(THUMB):
        target[act(f"right_hand_thumb_{j}_joint")] = v
    move(target, 1.5)
    step(0.5)

    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset)
    step(1.0)

    def cube_in_wrist():
        R = data.xmat[wrist].reshape(3, 3)
        return R.T @ (data.xpos[cube_body] - data.xpos[wrist])

    rel0 = cube_in_wrist()
    step(1.5)                                   # extra hold
    rel1 = cube_in_wrist()
    lift_end = data.xpos[cube_body][2] - c0[2]
    slip = float(np.linalg.norm(rel1 - rel0))
    ok = (lift_end >= 0.06) and (slip <= 0.02)
    return lift_end, slip, reach_err, ok


def main():
    rng = np.random.default_rng(0)
    rows, wins = [], 0
    t_start = time.time()
    for i in range(1, N_TRIALS + 1):
        x = rng.uniform(*X_RANGE)
        y = rng.uniform(*Y_RANGE)
        yaw_deg = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG)
        lift, slip, rerr, ok = run_trial(x, y, np.radians(yaw_deg))
        wins += ok
        rows.append([i, round(x, 3), round(y, 3), round(yaw_deg, 1), round(lift, 3), round(slip, 3), round(rerr, 3), int(ok)])
        print(f"[{i:2d}/{N_TRIALS}] cube x={x:.3f} y={y:+.3f} yaw={yaw_deg:+5.1f} deg | lift {lift*100:+5.1f} cm | "
              f"slip {slip*100:4.1f} cm | reach err {rerr*100:4.1f} cm | {'SUCCESS' if ok else 'fail'}  "
              f"({time.time() - t_start:.0f}s)", flush=True)

    print(f"\nSuccess rate: {wins}/{N_TRIALS} = {100 * wins / N_TRIALS:.0f}%")
    fails = [r for r in rows if not r[-1]]
    if fails:
        print("Failed trials (x, y, yaw):", "; ".join(f"#{r[0]} ({r[1]}, {r[2]}, {r[3]})" for r in fails))
    name = f"robustness_{datetime.now():%H%M%S}.csv"
    with open(ROOT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "cube_x", "cube_y", "yaw_deg", "lift_m", "slip_m", "reach_err_m", "success"])
        w.writerows(rows)
    print("Saved", name, "in", ROOT)


if __name__ == "__main__":
    main()
