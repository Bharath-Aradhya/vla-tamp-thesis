"""Place debug 2: why does the cube end up 5-9 cm from where it was released, and tilted?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\place_debug2.py

The first debug run showed: the cube sits within about 1 cm of the target when the hand is opened,
but ends 5-9 cm away. During the carry the cube is also tilted in the hand. This run
  * prints how far the cube is tilted from lying flat on a face (0 = flat),
  * prints the cube's movement every 0.25 s WHILE the hand opens (in the world and in the hand frame)
    and which parts touch it, to see whether it is pushed, dragged or falls over,
  * prints the IK error and the hand position error of the carry (one carry went 24 cm wrong),
  * compares two releases: 'baseline' (lower to 4 mm above the table, open in 1 s) and
    'gentle' (lower until the cube presses 3 mm into the table, open slowly in 2.5 s).
6 cases x 2 releases, about 1 to 2 minutes. CPU only.
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


def flat_tilt():
    """Degrees the cube is tilted away from lying flat on one of its faces (0 = flat)."""
    R = data.xmat[cube_body].reshape(3, 3)
    return float(np.degrees(np.arccos(min(1.0, np.max(np.abs(R[2, :]))))))


def touching_names():
    names = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if cube_body in (b1, b2):
            other = b2 if b1 == cube_body else b1
            names.add(model.body(other).name.replace("right_hand_", "").replace("_link", "") or "table")
    return ",".join(sorted(names)) or "nothing"


def phase(label, dst3, extra=""):
    c = data.xpos[cube_body]
    print(f"      {label:<20s} height {(c[2] - TABLE_Z) * 1000:+6.1f} mm  from target ({(c[0] - dst3[0]) * 100:+5.1f},{(c[1] - dst3[1]) * 100:+5.1f}) cm  "
          f"tilt {flat_tilt():4.1f} deg  touching: {touching_names()} {extra}")


def open_hand(seconds):
    target = data.ctrl.copy()
    target[HAND_ACT] = model.key_ctrl[key][HAND_ACT]
    start = data.ctrl.copy()
    n = int(seconds / model.opt.timestep)
    every = int(0.25 / model.opt.timestep)
    c_start = data.xpos[cube_body].copy()
    R0 = data.xmat[wrist].reshape(3, 3).copy()
    for i in range(n):
        a = (i + 1) / n
        a = a * a * (3 - 2 * a)
        data.ctrl[:] = (1 - a) * start + a * target
        mujoco.mj_step(model, data)
        if (i + 1) % every == 0:
            d = data.xpos[cube_body] - c_start
            dh = R0.T @ d
            print(f"        open t={(i + 1) * model.opt.timestep:4.2f}s  cube moved world ({d[0] * 100:+.1f},{d[1] * 100:+.1f},{d[2] * 100:+.1f}) cm, "
                  f"hand frame ({dh[0] * 100:+.1f},{dh[1] * 100:+.1f},{dh[2] * 100:+.1f}) cm, tilt {flat_tilt():4.1f}, touching: {touching_names()}")


def run_debug(src, dst, yaw, variant):
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
    dst3 = np.array([dst[0], dst[1], TABLE_Z])
    phase("gripped on table", c0)
    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    step(1.0)
    if data.xpos[cube_body][2] - c0[2] < 0.06:
        print("      pick failed")
        return
    rel = data.xpos[cube_body] - grasp_point()
    phase("lifted at the source", np.array([src[0], src[1], TABLE_Z]))
    ik, sim, _ = go_to(dst3 + np.array([0.0, 0.0, 0.10]) - rel, 1.5, offset, r_des)
    step(0.5)
    phase("carried", dst3 + np.array([0.0, 0.0, 0.10]), f"(IK err {ik * 100:.1f} cm, hand off {sim * 100:.1f} cm)")
    if np.linalg.norm((data.xpos[cube_body] - dst3 - np.array([0.0, 0.0, 0.10]))[:2]) > 0.05:
        print("      carry went wrong, stopping this trial")
        return
    lower = 0.004 if variant == "baseline" else -0.003
    ik, sim, _ = go_to(dst3 + np.array([0.0, 0.0, lower]) - rel, 1.2, offset, r_des)
    step(0.3)
    phase("lowered", dst3, f"(IK err {ik * 100:.1f} cm, hand off {sim * 100:.1f} cm)")
    open_hand(1.0 if variant == "baseline" else 2.5)
    step(0.5)
    phase("hand opened", dst3)
    go_to(grasp_point() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    go_to(grasp_point() + np.array([pre[0], pre[1], 0.0]), 1.2, offset, r_des)
    step(1.0)
    phase("after retreat (final)", dst3)


def main():
    rng = np.random.default_rng(3)
    cases = [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]
    for (i, j) in cases:
        src, dst = LOCATIONS[i], LOCATIONS[j]
        yaw_deg = float(rng.uniform(0, 90))
        for variant in ("baseline", "gentle"):
            print(f"\n({src[0]:.2f},{src[1]:+.2f}) -> ({dst[0]:.2f},{dst[1]:+.2f}), cube rotated {yaw_deg:.0f} deg, release: {variant}")
            run_debug(src, dst, np.radians(yaw_deg), variant)


if __name__ == "__main__":
    main()
