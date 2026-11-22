"""Workspace map: where on the table does the scripted grasp work?

Run from the project root (D:\\vla-tamp-thesis), after build_and_check.py:
    python scripts\\workspace_map.py

Places the cube on a grid of positions (5 distances x 9 sideways positions x 2 rotations =
90 trials), runs the full reach-grasp-lift with the best grasp point from grasp_refine.py,
and records whether the cube was held (rose >= 6 cm, slipped <= 2 cm during an extra hold).
Prints a map, and saves workspace_map_HHMMSS.png (picture) and workspace_map_HHMMSS.csv.
Expect roughly 5 to 15 minutes.
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

# --- map settings ---------------------------------------------------------------------------
THUMB = (0.0, 0.7, -1.6)                   # thumb joints 0, 1, 2 when closing
DX, OY, DZ = 0.14, 0.02, 0.02              # grasp point found by grasp_refine.py (along fingers, across palm, height)
X_VALUES = [0.28, 0.32, 0.36, 0.40, 0.44]  # cube x (m): distance in front of the robot
Y_VALUES = [0.05, 0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35]   # cube y (m): + is the robot's left
YAWS_DEG = [0.0, 12.0]                     # cube rotations tried at every position
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


def run_trial(cube_x, cube_y, yaw, dx, oy, dz):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, 0.775]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)
    offset = np.array([dx, oy, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, dz])

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
    step(1.5)
    rel1 = cube_in_wrist()
    lift_end = data.xpos[cube_body][2] - c0[2]
    slip = float(np.linalg.norm(rel1 - rel0))
    ok = (lift_end >= 0.06) and (slip <= 0.02)
    return lift_end, slip, reach_err, ok


def draw_map(cells, path):
    """Top-down picture: robot at the bottom looking up, its right hand side on the right."""
    n = len(YAWS_DEG)
    cw, ch, left, top, bottom = 70, 46, 70, 40, 60
    w = left + cw * len(Y_VALUES) + 20
    h = top + ch * len(X_VALUES) + bottom
    img = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(img)
    for ci, y in enumerate(Y_VALUES):                    # left to right = robot's left to right
        d.text((left + ci * cw + 18, top - 18), f"y={y:+.2f}", fill=(40, 40, 40))
    for ri, x in enumerate(reversed(X_VALUES)):          # far at the top, near at the bottom
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
    t0 = time.time()
    n = len(YAWS_DEG)
    total = len(X_VALUES) * len(Y_VALUES)
    print(f"Mapping {total} positions x {n} rotations = {total * n} trials")
    cells, rows, done = {}, [], 0
    for x in X_VALUES:
        for y in Y_VALUES:
            wins = 0
            for yaw_deg in YAWS_DEG:
                lift, slip, rerr, ok = run_trial(x, y, np.radians(yaw_deg), DX, OY, DZ)
                wins += ok
                rows.append([x, y, yaw_deg, round(lift, 3), round(slip, 3), round(rerr, 3), int(ok)])
            cells[(x, y)] = wins
            done += 1
            print(f"[{done:2d}/{total}] cube x={x:.2f} y={y:+.2f}: {wins}/{n} held   ({time.time() - t0:.0f}s)", flush=True)

    print("\nMap: rows = distance in front of the robot (far at the top), columns = sideways position")
    print("(left of the robot on the left). Entries = trials where the cube was held.\n")
    print("          " + "".join(f"y={y:+.2f}".center(9) for y in Y_VALUES))
    for x in reversed(X_VALUES):
        print(f"x={x:.2f}   " + "".join(f"{cells[(x, y)]}/{n}".center(9) for y in Y_VALUES))
    good = [(x, y) for (x, y), v in cells.items() if v == n]
    print(f"\nPositions held in all {n} trials: {len(good)} of {total}")

    stamp = f"{datetime.now():%H%M%S}"
    with open(ROOT / f"workspace_map_{stamp}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cube_x", "cube_y", "yaw_deg", "lift_m", "slip_m", "reach_err_m", "success"])
        w.writerows(rows)
    draw_map(cells, ROOT / f"workspace_map_{stamp}.png")
    print(f"Saved workspace_map_{stamp}.png and .csv in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
