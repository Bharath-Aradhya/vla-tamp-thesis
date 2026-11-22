"""Grasp tilt search: find a grasp that holds the cube FLAT, not tilted.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\grasp_tilt_search.py

place_debug2.py showed the current grasp is a two-contact pinch (middle fingertip + thumb) that
leaves the cube tilted 16-43 degrees in the hand; when the hand opens the cube tips over and is
swept about 7 cm. The model file shows why: index and middle fingers sit 2.85 cm either side of the
palm centre, and the grasp point is 2 cm above the cube centre, so the index finger misses the cube.
This script randomly tries 120 grasp variants at one location (x=0.40, y=-0.20), each with two cube
rotations (0 and 20 degrees, hand turned by the hybrid yaw rule). Varied: the grasp point along the
fingers (DX), across the palm (OY), along the hand's z axis (OZ, new), its height above the cube
centre (DZ), the thumb joints and how far the fingers curl. A variant counts only if the cube is held
(rises >= 6 cm, slips <= 2 cm) at both rotations; those are ranked by the worst tilt of the cube
from lying flat after the lift (smaller = better). The current grasp is run first as a reference.
Expect roughly 8 to 12 minutes. CPU only. Saves grasp_tilt_search_HHMMSS.csv.
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


SEARCH_AT = (0.40, -0.20)
SEARCH_YAWS = (0.0, 20.0)
N_CANDIDATES = 120
BASELINE = dict(dx=0.14, oy=0.02, oz=0.0, dz=0.02, t0=0.0, t1=0.7, t2=-1.6, f0=1.2, f1=1.4)


def flat_tilt():
    """Degrees the cube is tilted away from lying flat on one of its faces (0 = flat)."""
    R = data.xmat[cube_body].reshape(3, 3)
    return float(np.degrees(np.arccos(min(1.0, np.max(np.abs(R[2, :]))))))


def run_candidate(p, yaw_deg):
    yaw = np.radians(yaw_deg)
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[waist_qadr] = list(STANCE)
    data.ctrl[waist_act] = list(STANCE)
    data.qpos[cube_qadr:cube_qadr + 3] = [SEARCH_AT[0], SEARCH_AT[1], TABLE_Z]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)
    a = hand_yaw("hybrid", yaw)
    r_des = rot_z(a)
    pre = rot_z(a) @ PRE
    offset = np.array([p["dx"], p["oy"], p["oz"]])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, p["dz"]])

    go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    go_to(grasp + pre, 1.5, offset, r_des)
    go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    go_to(grasp, 1.0, offset, r_des)
    target = data.ctrl.copy()
    target[act("right_hand_index_0_joint")] = p["f0"]
    target[act("right_hand_middle_0_joint")] = p["f0"]
    target[act("right_hand_index_1_joint")] = p["f1"]
    target[act("right_hand_middle_1_joint")] = p["f1"]
    target[act("right_hand_thumb_0_joint")] = p["t0"]
    target[act("right_hand_thumb_1_joint")] = p["t1"]
    target[act("right_hand_thumb_2_joint")] = p["t2"]
    move(target, 1.5)
    step(0.5)
    n_touch = touches_cube()
    tilt_gripped = flat_tilt()
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
    return dict(lift=lift, slip=slip, tilt=flat_tilt(), tilt_gripped=tilt_gripped, touches=n_touch,
                ok=(lift >= 0.06 and slip <= 0.02))


def evaluate(p):
    rs = [run_candidate(p, y) for y in SEARCH_YAWS]
    ok = all(r["ok"] for r in rs)
    return ok, max(r["tilt"] for r in rs), min(r["touches"] for r in rs), rs


def main():
    t0 = time.time()
    rng = np.random.default_rng(5)
    ok, tilt, touch, rs = evaluate(BASELINE)
    print("Reference (current grasp): " + ", ".join(
        f"rot {y:.0f}: {'held' if r['ok'] else 'NOT held'}, tilt {r['tilt']:.0f} deg, {r['touches']} contacts"
        for y, r in zip(SEARCH_YAWS, rs)), flush=True)
    rows, results = [], []
    for k in range(N_CANDIDATES):
        p = dict(dx=rng.uniform(0.12, 0.16), oy=rng.uniform(0.005, 0.05), oz=rng.uniform(-0.02, 0.02),
                 dz=rng.uniform(0.01, 0.03), t0=rng.uniform(-0.6, 0.6), t1=rng.uniform(0.2, 0.72),
                 t2=rng.uniform(-1.745, -1.0), f0=rng.uniform(0.8, 1.4), f1=rng.uniform(1.0, 1.7))
        ok, tilt, touch, rs = evaluate(p)
        results.append((ok, tilt, touch, p))
        rows.append([k] + [round(p[x], 3) for x in p] + [int(ok), round(tilt, 1), touch,
                    round(rs[0]["tilt"], 1), round(rs[1]["tilt"], 1), round(rs[0]["lift"], 3), round(rs[1]["lift"], 3)])
        print(f"[{k + 1:3d}/{N_CANDIDATES}] {'held both' if ok else 'not held '}  worst tilt {tilt:5.1f} deg  "
              f"contacts {touch}   ({time.time() - t0:.0f}s)", flush=True)

    held = sorted([r for r in results if r[0]], key=lambda r: r[1])
    print(f"\n{len(held)} of {N_CANDIDATES} variants held the cube at both rotations.")
    print("Best by tilt (parameters: grasp point dx,oy,oz,dz in cm; thumb joints 0,1,2; finger curl base,tip):")
    for ok, tilt, touch, p in held[:10]:
        print(f"  tilt {tilt:5.1f} deg, contacts {touch} | dx {p['dx'] * 100:.1f} oy {p['oy'] * 100:.1f} oz {p['oz'] * 100:+.1f} "
              f"dz {p['dz'] * 100:.1f} | thumb {p['t0']:+.2f} {p['t1']:+.2f} {p['t2']:+.2f} | fingers {p['f0']:.2f} {p['f1']:.2f}")
    out = ROOT / f"grasp_tilt_search_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "dx", "oy", "oz", "dz", "t0", "t1", "t2", "f0", "f1", "held_both", "worst_tilt_deg",
                    "min_contacts", "tilt_rot0", "tilt_rot20", "lift_rot0", "lift_rot20"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
