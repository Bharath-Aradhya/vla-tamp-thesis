"""Workspace map v2: does using the waist widen the area where the scripted grasp works?

Run from the project root (D:\\vla-tamp-thesis), after build_and_check.py:
    python scripts\\workspace_map_v2.py

Runs the same map as workspace_map.py (cube on a grid of table positions x 2 rotations) twice:
  arm_only   - the 7 right-arm joints move, as before (should reproduce the first map)
  arm_waist  - the waist yaw and waist pitch also take part in the IK (leaning forward / turning
               the torso), with a penalty so they only move when the arm alone cannot get there.
For every trial it records what the SIMULATED hand did (not just the IK's own error), whether
the fingers touched the cube when they closed, and how far the cube was pushed during the
approach. At the end it prints a map and a failure-reason breakdown per mode.

Outputs in the project root: workspace_map_v2_HHMMSS.csv and one PNG per mode.
Expect roughly 8 to 12 minutes in total. CPU only, no GPU.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "scene_pick.xml"

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
WAIST = ["waist_yaw_joint", "waist_pitch_joint"]     # pitch: positive = lean forward
FINGERS = {"right_hand_index_0_joint": 1.2, "right_hand_index_1_joint": 1.4,
           "right_hand_middle_0_joint": 1.2, "right_hand_middle_1_joint": 1.4}

# --- settings ---------------------------------------------------------------------------------
MODES = [("arm_only", 0.0), ("arm_waist", 0.3)]    # (name, waist penalty: 0 = waist locked, 1 = as free as the arm)
WAIST_YAW_LIMIT = 0.6                              # rad each way (about 34 degrees)
WAIST_PITCH_RANGE = (0.0, 0.5)                     # rad, forward lean only
THUMB = (0.0, 0.7, -1.6)
DX, OY, DZ = 0.14, 0.02, 0.02                      # grasp point from grasp_refine.py
X_VALUES = [0.28, 0.32, 0.36, 0.40, 0.44, 0.48]    # cube x (m); the table starts at x = 0.25
Y_VALUES = [0.05, 0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35]
YAWS_DEG = [0.0, 12.0]
PRE = np.array([-0.06, -0.10, 0.05])
R_TARGET = np.eye(3)
ROT_WEIGHT = 0.3
# -----------------------------------------------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)
key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
act = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

JOINTS = ARM + WAIST                               # IK works on these 9 joints (arm first)
N_ARM = len(ARM)
ik_act = [act(n) for n in JOINTS]
ik_joint = [model.joint(n) for n in JOINTS]
qadr = np.array([j.qposadr[0] for j in ik_joint])
dadr = np.array([j.dofadr[0] for j in ik_joint])
lo = np.array([j.range[0] for j in ik_joint])
hi = np.array([j.range[1] for j in ik_joint])
lo[N_ARM], hi[N_ARM] = -WAIST_YAW_LIMIT, WAIST_YAW_LIMIT
lo[N_ARM + 1], hi[N_ARM + 1] = WAIST_PITCH_RANGE
wrist = model.body("right_wrist_yaw_link").id
cube_body = model.body("cube").id
cube_qadr = model.joint("cube_joint").qposadr[0]
waist_scale = 0.0                                  # set per mode in main()


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
    """Damped least squares on 9 joints; waist columns are scaled so the arm is preferred."""
    jscale = np.array([1.0] * N_ARM + [waist_scale] * len(WAIST))
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
        J = np.vstack([jacp[:, dadr], ROT_WEIGHT * jacr[:, dadr]]) * jscale
        err = np.concatenate([e_pos, ROT_WEIGHT * e_rot])
        dq = (J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)) * jscale
        q = np.clip(q + 0.5 * dq, lo, hi)
    return q, float(np.linalg.norm(e_pos))


def hand_state(target_pos, offset):
    """What the simulated hand actually did: distance of the grasp point from the target, and tilt."""
    R = data.xmat[wrist].reshape(3, 3)
    p = data.xpos[wrist] + R @ offset
    cos = np.clip((np.trace(R_TARGET.T @ R) - 1) / 2, -1, 1)
    return float(np.linalg.norm(target_pos - p)), float(np.degrees(np.arccos(cos)))


def go_to(point, seconds, offset):
    point = np.asarray(point, float)
    q, ik_err = solve_ik(data.qpos[qadr].copy(), point, offset)
    target = data.ctrl.copy()
    target[ik_act] = q
    move(target, seconds)
    step(0.3)
    sim_err, tilt = hand_state(point, offset)
    return ik_err, sim_err, tilt


def touches_cube():
    """Number of distinct right-hand bodies currently in contact with the cube."""
    names = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if cube_body in (b1, b2):
            other = b2 if b1 == cube_body else b1
            name = model.body(other).name
            if "right_hand" in name:
                names.add(name)
    return len(names)


def run_trial(cube_x, cube_y, yaw):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, 0.775]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)
    offset = np.array([DX, OY, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, DZ])

    go_to([0.12, -0.25, 0.80], 1.5, offset)
    go_to([0.30, -0.25, 0.92], 1.5, offset)
    go_to(grasp + PRE, 1.5, offset)
    go_to(grasp + 0.5 * PRE, 1.0, offset)
    ik_err, sim_err, tilt = go_to(grasp, 1.0, offset)
    pushed = float(np.linalg.norm(data.xpos[cube_body] - c0))
    waist_yaw, waist_pitch = float(data.qpos[qadr[N_ARM]]), float(data.qpos[qadr[N_ARM + 1]])

    target = data.ctrl.copy()
    for name, val in FINGERS.items():
        target[act(name)] = val
    for j, v in enumerate(THUMB):
        target[act(f"right_hand_thumb_{j}_joint")] = v
    move(target, 1.5)
    step(0.5)
    n_touch = touches_cube()

    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset)
    step(1.0)

    def cube_in_wrist():
        R = data.xmat[wrist].reshape(3, 3)
        return R.T @ (data.xpos[cube_body] - data.xpos[wrist])

    rel0 = cube_in_wrist()
    step(1.5)
    rel1 = cube_in_wrist()
    lift = float(data.xpos[cube_body][2] - c0[2])
    slip = float(np.linalg.norm(rel1 - rel0))
    return dict(lift=lift, slip=slip, ik_err=ik_err, sim_err=sim_err, tilt=tilt, pushed=pushed,
                touches=n_touch, waist_yaw=waist_yaw, waist_pitch=waist_pitch,
                ok=(lift >= 0.06 and slip <= 0.02))


def reason(r):
    """Why a trial failed (first matching reason); 'success' if it did not."""
    if r["ok"]:
        return "success"
    if r["lift"] < -0.3:
        return "cube fell off table"
    if r["ik_err"] > 0.02:
        return "IK cannot reach (out of reach)"
    if r["sim_err"] > 0.02:
        return "IK ok but simulated hand missed"
    if r["touches"] == 0:
        return "hand there but fingers touched nothing"
    if r["lift"] < 0.06:
        return "touched but cube not lifted"
    return "lifted then slipped"


def draw_map(cells, path, title):
    n = len(YAWS_DEG)
    cw, ch, left, top, bottom = 70, 46, 70, 60, 60
    w = left + cw * len(Y_VALUES) + 20
    h = top + ch * len(X_VALUES) + bottom
    img = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.text((8, 8), title, fill=(20, 20, 20))
    for ci, y in enumerate(Y_VALUES):
        d.text((left + ci * cw + 18, top - 18), f"y={y:+.2f}", fill=(40, 40, 40))
    for ri, x in enumerate(reversed(X_VALUES)):
        d.text((8, top + ri * ch + 16), f"x={x:.2f}", fill=(40, 40, 40))
        for ci, y in enumerate(Y_VALUES):
            f = cells[(x, y)] / n
            colour = (int(225 * (1 - f)) + 20, int(190 * f) + 40, 60)
            x0, y0 = left + ci * cw, top + ri * ch
            d.rectangle([x0, y0, x0 + cw - 3, y0 + ch - 3], fill=colour)
            d.text((x0 + 22, y0 + 16), f"{cells[(x, y)]}/{n}", fill=(255, 255, 255))
    d.text((left + cw * len(Y_VALUES) // 2 - 70, top + ch * len(X_VALUES) + 14),
           "ROBOT (looking up the page)", fill=(40, 40, 40))
    d.text((left, top + ch * len(X_VALUES) + 34), "green = cube held in all trials, red = never held",
           fill=(90, 90, 90))
    img.save(path)


def main():
    global waist_scale
    t0 = time.time()
    n = len(YAWS_DEG)
    total = len(X_VALUES) * len(Y_VALUES)
    stamp = f"{datetime.now():%H%M%S}"
    print(f"{len(MODES)} modes x {total} positions x {n} rotations = {len(MODES) * total * n} trials")
    rows = []
    for mode, scale in MODES:
        waist_scale = scale
        cells, reasons, done = {}, Counter(), 0
        print(f"\n===== mode: {mode} (waist penalty {scale}) =====")
        for x in X_VALUES:
            for y in Y_VALUES:
                wins = 0
                for yaw_deg in YAWS_DEG:
                    r = run_trial(x, y, np.radians(yaw_deg))
                    wins += r["ok"]
                    why = reason(r)
                    reasons[why] += 1
                    rows.append([mode, x, y, yaw_deg, round(r["lift"], 3), round(r["slip"], 3),
                                 round(r["ik_err"], 3), round(r["sim_err"], 3), round(r["tilt"], 1),
                                 round(r["pushed"], 3), r["touches"], round(r["waist_yaw"], 2),
                                 round(r["waist_pitch"], 2), int(r["ok"]), why])
                cells[(x, y)] = wins
                done += 1
                print(f"[{done:2d}/{total}] {mode} cube x={x:.2f} y={y:+.2f}: {wins}/{n} held   "
                      f"({time.time() - t0:.0f}s)", flush=True)

        print(f"\nMap for {mode}: rows = distance in front of the robot (far at the top), columns = sideways")
        print("position (robot's left on the left). Entries = trials where the cube was held.\n")
        print("          " + "".join(f"y={y:+.2f}".center(9) for y in Y_VALUES))
        for x in reversed(X_VALUES):
            print(f"x={x:.2f}   " + "".join(f"{cells[(x, y)]}/{n}".center(9) for y in Y_VALUES))
        good = sum(1 for v in cells.values() if v == n)
        print(f"\n{mode}: positions held in all {n} trials: {good} of {total}; "
              f"trials succeeded: {reasons['success']} of {total * n}")
        print("Why the other trials failed:")
        for why, k in reasons.most_common():
            if why != "success":
                print(f"   {k:3d}  {why}")
        draw_map(cells, ROOT / f"workspace_map_v2_{mode}_{stamp}.png", f"{mode} (waist penalty {scale})")

    with open(ROOT / f"workspace_map_v2_{stamp}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "cube_x", "cube_y", "yaw_deg", "lift_m", "slip_m", "ik_err_m", "sim_err_m",
                    "tilt_deg", "pushed_m", "touching_bodies", "waist_yaw_rad", "waist_pitch_rad",
                    "success", "reason"])
        w.writerows(rows)
    print(f"\nSaved workspace_map_v2_{stamp}.csv and one PNG per mode in {ROOT}   "
          f"(total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
