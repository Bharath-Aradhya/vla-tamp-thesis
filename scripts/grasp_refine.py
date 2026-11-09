"""Search for the grasp point that works best across many cube positions.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\grasp_refine.py

Step 1 (search): tries 36 grasp-point settings (distance along the fingers, offset across the
         palm, height above the cube centre). Each is scored on the same 6 random cube
         positions/rotations in the reachable region. Thumb pose is fixed to the one that
         worked in grasp_sweep.py.
Step 2 (validate): runs the best setting on 30 NEW random cube poses and reports the success
         rate, using the same strict test as grasp_robustness.py (cube rose >= 6 cm and did
         not slip more than 2 cm during an extra 1.5 s hold).
Prints progress with elapsed time; expect roughly 10-20 minutes in total.
Saves refine_search_HHMMSS.csv and refine_validate_HHMMSS.csv.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
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

# --- search settings ---------------------------------------------------------------------
THUMB = (0.0, 0.7, -1.6)                   # thumb joints 0, 1, 2 when closing (from grasp_sweep.py)
DX_LIST = [0.10, 0.12, 0.14, 0.16]         # grasp point along the fingers from the wrist (m)
OY_LIST = [0.0, 0.02, 0.04]                # grasp point across the palm (m)
DZ_LIST = [0.0, 0.02, 0.04]                # grasp point height above the cube centre (m)
N_SEARCH_POSES = 6
N_VALIDATE = 30
X_RANGE = (0.32, 0.40)                     # random cube position (m), robot frame
Y_RANGE = (-0.19, -0.11)
YAW_RANGE_DEG = 20                         # random cube rotation about the vertical (+/- degrees)
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


def sample_poses(rng, n):
    return [(rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE), np.radians(rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG)))
            for _ in range(n)]


def main():
    t_start = time.time()
    search_poses = sample_poses(np.random.default_rng(1), N_SEARCH_POSES)
    combos = list(itertools.product(DX_LIST, OY_LIST, DZ_LIST))
    print(f"Search: {len(combos)} grasp settings x {N_SEARCH_POSES} cube poses = {len(combos) * N_SEARCH_POSES} trials")
    scored = []
    for i, (dx, oy, dz) in enumerate(combos, 1):
        wins = sum(run_trial(x, y, yaw, dx, oy, dz)[3] for x, y, yaw in search_poses)
        scored.append((wins, dx, oy, dz))
        print(f"[{i:2d}/{len(combos)}] along fingers {dx:.2f}  across palm {oy:.2f}  height {dz:.2f}  ->  "
              f"{wins}/{N_SEARCH_POSES} held   ({time.time() - t_start:.0f}s)", flush=True)

    scored.sort(key=lambda s: -s[0])
    stamp = f"{datetime.now():%H%M%S}"
    with open(ROOT / f"refine_search_{stamp}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wins", "dx", "oy", "dz"])
        w.writerows(scored)
    print("\nTop 5 settings:")
    for wins, dx, oy, dz in scored[:5]:
        print(f"  {wins}/{N_SEARCH_POSES} held | along fingers {dx:.2f} | across palm {oy:.2f} | height {dz:.2f}")

    wins0, dx, oy, dz = scored[0]
    print(f"\nValidating the best setting (dx={dx}, oy={oy}, dz={dz}) on {N_VALIDATE} new cube poses...")
    poses = sample_poses(np.random.default_rng(2), N_VALIDATE)
    rows, wins = [], 0
    for i, (x, y, yaw) in enumerate(poses, 1):
        lift, slip, rerr, ok = run_trial(x, y, yaw, dx, oy, dz)
        wins += ok
        rows.append([i, round(x, 3), round(y, 3), round(float(np.degrees(yaw)), 1), round(lift, 3), round(slip, 3), round(rerr, 3), int(ok)])
        print(f"[{i:2d}/{N_VALIDATE}] cube x={x:.3f} y={y:+.3f} | lift {lift*100:+5.1f} cm | slip {slip*100:4.1f} cm | "
              f"reach err {rerr*100:4.1f} cm | {'SUCCESS' if ok else 'fail'}", flush=True)
    with open(ROOT / f"refine_validate_{stamp}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "cube_x", "cube_y", "yaw_deg", "lift_m", "slip_m", "reach_err_m", "success"])
        w.writerows(rows)
    print(f"\nValidation success rate: {wins}/{N_VALIDATE} = {100 * wins / N_VALIDATE:.0f}%")
    print(f"Best grasp point: along fingers {dx}, across palm {oy}, height {dz}. Saved refine_*_{stamp}.csv in {ROOT}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
