"""Pick-and-place test: carry a block from one candidate location to another and put it down.

Run from the project root (D:\\vla-tamp-thesis), after build_and_check.py:
    python scripts\\pick_place_test.py

Uses the frozen grasp (far waist stance, hybrid hand-yaw policy). For every ordered pair of the
6 candidate locations (30 pairs) it runs 2 trials with a random cube rotation and +-1 cm
start offset: pick up, carry 10 cm above the table, lower, open the hand, retreat. The hand is
steered by where the cube sits in the hand (measured after the lift), so the cube, not the
hand, lands on the target.
A trial counts as OK if the cube ends within 2 cm of the target, upright, resting on the table.
Prints each trial, a summary, and a source-by-target matrix; saves pick_place_HHMMSS.csv.
Expect roughly 4 to 8 minutes. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "scene_pick.xml"

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
WAIST = ["waist_yaw_joint", "waist_pitch_joint"]
FINGERS = {"right_hand_index_0_joint": 1.2, "right_hand_index_1_joint": 1.4,
           "right_hand_middle_0_joint": 1.2, "right_hand_middle_1_joint": 1.4}

# --- settings ---------------------------------------------------------------------------------
STANCE = (0.17, 0.50)                # fixed waist (yaw rad, pitch rad), the 'far' stance
LOCATIONS = [(x, y) for x in (0.32, 0.40, 0.48) for y in (-0.15, -0.25)]
TRIALS_PER_PAIR = 2
JITTER = 0.01
TABLE_Z = 0.775                      # cube centre height when resting on the table
THUMB = (0.0, 0.7, -1.6)
DX, OY, DZ = 0.14, 0.02, 0.02
PRE = np.array([-0.06, -0.10, 0.05])
ROT_WEIGHT = 0.3
# -----------------------------------------------------------------------------------------------

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
waist_act = [act(n) for n in WAIST]
HAND_ACT = [i for i in range(model.nu) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or '').startswith('right_hand')]
waist_qadr = np.array([model.joint(n).qposadr[0] for n in WAIST])
wrist = model.body("right_wrist_yaw_link").id
cube_body = model.body("cube").id
cube_qadr = model.joint("cube_joint").qposadr[0]


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap90(a):
    """Cube rotation about the vertical axis, wrapped into [-45, 45) degrees (a cube repeats every 90)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def hand_yaw(policy, yaw):
    """Hand rotation about the vertical axis (rad) for a cube rotated by yaw."""
    if policy == "plain":
        return 0.0
    w = wrap90(yaw)
    if w < 0:
        return 0.0
    if w <= np.radians(22.5):
        return float(w)
    return float(w - np.pi / 4)


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


def solve_ik(q_start, target_pos, offset, r_des, iters=400):
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
        e_rot = 0.5 * sum(np.cross(R[:, i], r_des[:, i]) for i in range(3))
        if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(e_rot) < 0.03:
            break
        J = np.vstack([jacp[:, dadr], ROT_WEIGHT * jacr[:, dadr]])
        err = np.concatenate([e_pos, ROT_WEIGHT * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q = np.clip(q + 0.5 * dq, lo, hi)
    return q, float(np.linalg.norm(e_pos))


def hand_state(target_pos, offset, r_des):
    R = data.xmat[wrist].reshape(3, 3)
    p = data.xpos[wrist] + R @ offset
    cos = np.clip((np.trace(r_des.T @ R) - 1) / 2, -1, 1)
    return float(np.linalg.norm(target_pos - p)), float(np.degrees(np.arccos(cos)))


def go_to(point, seconds, offset, r_des):
    point = np.asarray(point, float)
    q, ik_err = solve_ik(data.qpos[qadr].copy(), point, offset, r_des)
    target = data.ctrl.copy()
    target[arm_act] = q                       # the waist target is never touched: the stance stays fixed
    move(target, seconds)
    step(0.3)
    sim_err, tilt = hand_state(point, offset, r_des)
    return ik_err, sim_err, tilt


def touches_cube():
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


def cube_yaw():
    w, x, y, z = data.xquat[cube_body]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_trial(src, dst, yaw):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[waist_qadr] = list(STANCE)
    data.ctrl[waist_act] = list(STANCE)
    data.qpos[cube_qadr:cube_qadr + 3] = [src[0], src[1], TABLE_Z]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)

    a = hand_yaw("hybrid", yaw)
    r_des = rot_z(a)
    pre = rot_z(a) @ PRE
    offset = np.array([DX, OY, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, DZ])

    def grasp_point():
        return data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset

    out = dict(stage="ok", err_xy=float("nan"), final_yaw=float("nan"), upright=0, z=float("nan"), carry_slip=float("nan"))

    # --- pick (same as the frozen grasp) ---
    go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    go_to(grasp + pre, 1.5, offset, r_des)
    go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    go_to(grasp, 1.0, offset, r_des)
    target = data.ctrl.copy()
    for name, val in FINGERS.items():
        target[act(name)] = val
    for j, v in enumerate(THUMB):
        target[act(f"right_hand_thumb_{j}_joint")] = v
    move(target, 1.5)
    step(0.5)
    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    step(1.0)
    if data.xpos[cube_body][2] - c0[2] < 0.06:
        out["stage"] = "pick failed"
        return out

    # --- carry: steer so that the CUBE (not the hand) arrives above the target ---
    rel = data.xpos[cube_body] - grasp_point()
    dst3 = np.array([dst[0], dst[1], TABLE_Z])
    go_to(dst3 + np.array([0.0, 0.0, 0.10]) - rel, 1.5, offset, r_des)
    step(0.5)
    out["carry_slip"] = float(np.linalg.norm((data.xpos[cube_body] - grasp_point()) - rel))
    if out["carry_slip"] > 0.03:
        out["stage"] = "cube slipped in the hand while carrying"

    # --- lower, open, retreat ---
    go_to(dst3 + np.array([0.0, 0.0, 0.004]) - rel, 1.2, offset, r_des)
    step(0.3)
    target = data.ctrl.copy()
    target[HAND_ACT] = model.key_ctrl[key][HAND_ACT]
    move(target, 1.0)
    step(0.5)
    go_to(grasp_point() + pre, 1.5, offset, r_des)
    step(1.0)

    c = data.xpos[cube_body].copy()
    out["err_xy"] = float(np.linalg.norm(c[:2] - dst3[:2]))
    out["z"] = float(c[2])
    out["upright"] = int(data.xmat[cube_body].reshape(3, 3)[2, 2] >= 0.98)
    out["final_yaw"] = float(np.degrees(wrap90(cube_yaw())))
    if out["stage"] == "ok":
        if c[2] < 0.5:
            out["stage"] = "cube fell off table"
        elif not out["upright"] or abs(c[2] - TABLE_Z) > 0.01:
            out["stage"] = "cube toppled or not resting on table"
        elif out["err_xy"] > 0.02:
            out["stage"] = "landed more than 2 cm from target"
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(2)
    pairs = [(i, j) for i in range(len(LOCATIONS)) for j in range(len(LOCATIONS)) if i != j]
    total = len(pairs) * TRIALS_PER_PAIR
    print(f"{len(pairs)} location pairs x {TRIALS_PER_PAIR} trials = {total} trials")
    rows, stages, ok_err, matrix, done = [], Counter(), [], {}, 0
    for (i, j) in pairs:
        src, dst = LOCATIONS[i], LOCATIONS[j]
        wins = 0
        for t in range(TRIALS_PER_PAIR):
            s = (src[0] + rng.uniform(-JITTER, JITTER), src[1] + rng.uniform(-JITTER, JITTER))
            yaw_deg = float(rng.uniform(0, 90))
            r = run_trial(s, dst, np.radians(yaw_deg))
            stages[r["stage"]] += 1
            if r["stage"] == "ok":
                wins += 1
                ok_err.append(r["err_xy"])
            done += 1
            print(f"[{done:2d}/{total}] ({src[0]:.2f},{src[1]:+.2f}) -> ({dst[0]:.2f},{dst[1]:+.2f}) "
                  f"cube rotated {yaw_deg:4.1f} deg: {r['stage']}"
                  + (f", {r['err_xy'] * 100:.1f} cm from target" if r["err_xy"] == r["err_xy"] else "")
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([i, j, round(s[0], 3), round(s[1], 3), round(yaw_deg, 1), r["stage"],
                         round(r["err_xy"], 4), round(r["z"], 3), r["upright"], round(r["final_yaw"], 1),
                         round(r["carry_slip"], 3)])
        matrix[(i, j)] = wins

    print(f"\nOK trials (landed within 2 cm, upright, on the table): {stages['ok']} of {total}")
    for k, v in stages.most_common():
        if k != "ok":
            print(f"   {v:3d}  {k}")
    if ok_err:
        e = np.array(ok_err) * 100
        print(f"Landing error of the OK trials: mean {e.mean():.1f} cm, worst {e.max():.1f} cm")
    n = len(LOCATIONS)
    print("\nOK trials per source (rows) and target (columns), out of", TRIALS_PER_PAIR)
    print("location index:  " + "  ".join(f"{k}=({LOCATIONS[k][0]:.2f},{LOCATIONS[k][1]:+.2f})" for k in range(n)))
    print("        " + "".join(f"to {k}  " for k in range(n)))
    for i in range(n):
        print(f"from {i}  " + "".join(("  -   " if i == j else f"  {matrix[(i, j)]}   ") for j in range(n)))

    out = ROOT / f"pick_place_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_index", "dst_index", "start_x", "start_y", "cube_yaw_deg", "stage", "err_xy_m",
                    "final_z_m", "upright", "final_yaw_wrapped_deg", "carry_slip_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
