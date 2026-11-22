"""Hybrid yaw policy: align the hand only where aligning helps.

Run from the project root (D:\\vla-tamp-thesis), after build_and_check.py:
    python scripts\\hybrid_grasp.py

yaw_aligned_grasp.py showed (far stance, waist locked), by cube rotation wrapped into -45..+45 deg:
  plain   (hand straight ahead): works for 0 and for -45..-8 deg, fails for +8..+37 deg
  aligned (hand turned to the cube): works for 0..+22 deg, but turning the hand a long way
          (in particular towards negative angles) often cannot reach or misses.
So the hybrid policy is:
  cube rotation w < 0          -> hand straight ahead (plain already works there)
  0 <= w <= 22.5 deg           -> hand turned by w (aligned works there)
  w > 22.5 deg (to +45)        -> hand turned by w - 45 (0 to -22 deg), which leaves the cube at -45
                                  relative to the hand, an angle where plain worked.  UNTESTED guess.
It is tested on FRESH cube rotations (16 values, 0 to 84 deg, a new random seed), not the ones
the rule was derived from, against 'plain' on the same poses: 6 locations x 16 rotations x 2 = 192 trials.
Prints a table per location, a table per cube rotation, the failure reasons, and saves
hybrid_HHMMSS.csv. Expect roughly 6 to 9 minutes. CPU only.
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
STANCES = [("far", 0.17, 0.50)]                              # (name, waist yaw rad, waist pitch rad)
POLICIES = ["plain", "hybrid"]
CONFIGS = [(s, p) for s in STANCES for p in POLICIES]
LOCATIONS = [(x, y) for x in (0.32, 0.40, 0.48) for y in (-0.15, -0.25)]
N_YAWS = 16                          # cube rotations: 0, 5.6, ... 84 degrees (+- 1.5)
JITTER = 0.01
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


def run_trial(stance_yaw, stance_pitch, cube_x, cube_y, yaw, policy):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[waist_qadr] = [stance_yaw, stance_pitch]
    data.ctrl[waist_act] = [stance_yaw, stance_pitch]
    data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, 0.775]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)

    a = hand_yaw(policy, yaw)
    r_des = rot_z(a)                          # desired hand orientation: straight ahead, turned by a
    pre = rot_z(a) @ PRE                      # approach from the hand's own side, so turn it too
    offset = np.array([DX, OY, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, DZ])

    go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    go_to(grasp + pre, 1.5, offset, r_des)
    go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    ik_err, sim_err, tilt = go_to(grasp, 1.0, offset, r_des)
    pushed = float(np.linalg.norm(data.xpos[cube_body] - c0))

    target = data.ctrl.copy()
    for name, val in FINGERS.items():
        target[act(name)] = val
    for j, v in enumerate(THUMB):
        target[act(f"right_hand_thumb_{j}_joint")] = v
    move(target, 1.5)
    step(0.5)
    n_touch = touches_cube()

    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    step(1.0)

    def cube_in_wrist():
        R = data.xmat[wrist].reshape(3, 3)
        return R.T @ (data.xpos[cube_body] - data.xpos[wrist])

    rel0 = cube_in_wrist()
    step(1.5)
    rel1 = cube_in_wrist()
    lift = float(data.xpos[cube_body][2] - c0[2])
    slip = float(np.linalg.norm(rel1 - rel0))
    return dict(hand_yaw=float(np.degrees(a)), lift=lift, slip=slip, ik_err=ik_err, sim_err=sim_err, tilt=tilt, pushed=pushed,
                touches=n_touch, ok=(lift >= 0.06 and slip <= 0.02))


def reason(r):
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


def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    poses = {}                                # identical cube poses for every configuration
    for (x, y) in LOCATIONS:
        poses[(x, y)] = [(x + rng.uniform(-JITTER, JITTER), y + rng.uniform(-JITTER, JITTER),
                          k * 90.0 / N_YAWS + rng.uniform(-1.5, 1.5)) for k in range(N_YAWS)]
    total = len(CONFIGS) * len(LOCATIONS) * N_YAWS
    print(f"{len(CONFIGS)} configurations x {len(LOCATIONS)} locations x {N_YAWS} rotations = {total} trials")
    rows, by_loc, by_yaw, whys = [], {}, {}, {}
    done = 0
    for (sname, s_yaw, s_pitch), policy in CONFIGS:
        cfg = f"{sname}/{policy}"
        whys[cfg] = Counter()
        by_yaw[cfg] = [0] * N_YAWS
        for (x, y) in LOCATIONS:
            wins = 0
            for k, (cx, cy, yaw_deg) in enumerate(poses[(x, y)]):
                r = run_trial(s_yaw, s_pitch, cx, cy, np.radians(yaw_deg), policy)
                wins += r["ok"]
                by_yaw[cfg][k] += r["ok"]
                why = reason(r)
                whys[cfg][why] += 1
                rows.append([sname, policy, x, y, round(cx, 3), round(cy, 3), round(yaw_deg, 1), round(r["hand_yaw"], 1),
                             round(r["lift"], 3), round(r["slip"], 3), round(r["ik_err"], 3),
                             round(r["sim_err"], 3), round(r["tilt"], 1), round(r["pushed"], 3),
                             r["touches"], int(r["ok"]), why])
                done += 1
            by_loc[(cfg, x, y)] = wins
            print(f"[{done:3d}/{total}] {cfg:15s} x={x:.2f} y={y:+.2f}: {wins}/{N_YAWS} held   "
                  f"({time.time() - t0:.0f}s)", flush=True)

    cfgs = list(whys)
    print(f"\nCube held, out of {N_YAWS} rotations, per location:\n")
    print(f"{'configuration':16s}" + "".join(f"x={x:.2f} y={y:+.2f}  " for (x, y) in LOCATIONS) + "  TOTAL")
    for cfg in cfgs:
        vals = [by_loc[(cfg, x, y)] for (x, y) in LOCATIONS]
        print(f"{cfg:16s}" + "".join(f"{v}/{N_YAWS}".center(17) for v in vals) + f"  {sum(vals)}/{N_YAWS * len(LOCATIONS)}")

    print(f"\nCube held, out of {len(LOCATIONS)} locations, per cube rotation (degrees):\n")
    print(f"{'configuration':16s}" + "".join(f"{k * 90.0 / N_YAWS:5.1f}" for k in range(N_YAWS)))
    for cfg in cfgs:
        print(f"{cfg:16s}" + "".join(f"{v:5d}" for v in by_yaw[cfg]))

    print("\nFailure reasons:")
    for cfg in cfgs:
        fails = {k: v for k, v in whys[cfg].items() if k != "success"}
        print(f"  {cfg}: " + ("; ".join(f"{v} x {k}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])) or "none"))

    out = ROOT / f"hybrid_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stance", "policy", "loc_x", "loc_y", "cube_x", "cube_y", "cube_yaw_deg", "hand_yaw_deg", "lift_m",
                    "slip_m", "ik_err_m", "sim_err_m", "tilt_deg", "pushed_m", "touching_bodies",
                    "success", "reason"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
