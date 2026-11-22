"""Block size test: does a bigger or taller block sit flat in the hand?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\block_size_test.py

grasp_tilt_search.py found no grasp that held the 5 cm cube flatter than 19 degrees, and almost
every grasp touched the cube with only two hand parts. The model file shows the index and middle
fingers sit 5.7 cm apart (2.85 cm either side of the palm centre, thumb at the centre), taller than
the 5 cm cube, so only one finger can touch it and the cube pivots in the pinch. Block size is a free
choice in our scene, so this tests 6 block shapes: the 5 cm cube (reference), 6 cm and 7 cm cubes and
three tall 'bars' (5x5x8, 5x5x10 and 6x6x9 cm). For each, 3 grasp positions along the fingers
(dx = 13, 14, 15 cm) x 4 cube rotations (0, 15, 30, 60 degrees) at x=0.40, y=-0.20; the hand is
lowered so the middle finger clears the table. Reports, per shape and dx: cubes held, mean and
worst tilt after the lift, and mean number of touching hand parts. 72 trials, about 3 to 5 minutes.
Saves block_size_test_HHMMSS.csv. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE_SRC = ROOT / "models" / "scene_pick.xml"
SCENE_TMP = ROOT / "models" / "_block_test.xml"

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
WAIST = ["waist_yaw_joint", "waist_pitch_joint"]

# --- settings ---------------------------------------------------------------------------------
STANCE = (0.17, 0.50)
SEARCH_AT = (0.40, -0.20)
YAWS = (0.0, 15.0, 30.0, 60.0)
DX_VALUES = (0.13, 0.14, 0.15)
SHAPES = [("5 cm cube (reference)", (0.025, 0.025, 0.025)), ("6 cm cube", (0.03, 0.03, 0.03)),
          ("7 cm cube", (0.035, 0.035, 0.035)), ("5x5x8 cm bar", (0.025, 0.025, 0.04)),
          ("5x5x10 cm bar", (0.025, 0.025, 0.05)), ("6x6x9 cm bar", (0.03, 0.03, 0.045))]
OY = 0.02
THUMB = (0.0, 0.7, -1.6)
FINGER_CURL = (1.2, 1.4)
FINGER_CLEARANCE_Z = 0.0455        # grasp height above the block centre = this minus the block half-height (gives the proven 2 cm for the 5 cm cube; 0 for tall bars)
PRE = np.array([-0.06, -0.10, 0.05])
ROT_WEIGHT = 0.3
# -----------------------------------------------------------------------------------------------

model = data = key = None
arm_act = qadr = dadr = lo = hi = waist_act = waist_qadr = HAND_ACT = None
wrist = cube_body = cube_qadr = None
table_z = 0.775
act = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)


def load_scene(half):
    """Rebuild the scene with a block of half-sizes `half` (m) and refresh all model-dependent globals."""
    global model, data, key, arm_act, qadr, dadr, lo, hi, waist_act, waist_qadr, HAND_ACT
    global wrist, cube_body, cube_qadr, table_z
    text = SCENE_SRC.read_text(encoding="utf-8")
    old_size, old_pos = 'size="0.025 0.025 0.025"', 'pos="0.38 -0.15 0.775"'
    assert text.count(old_size) == 1 and text.count(old_pos) == 1, "scene_pick.xml was changed; update block_size_test.py"
    table_z = 0.75 + half[2]
    text = text.replace(old_size, f'size="{half[0]} {half[1]} {half[2]}"').replace(old_pos, f'pos="0.38 -0.15 {table_z}"')
    SCENE_TMP.write_text(text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(SCENE_TMP))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    arm_act = [act(n) for n in ARM]
    arm_joint = [model.joint(n) for n in ARM]
    qadr = np.array([j.qposadr[0] for j in arm_joint])
    dadr = np.array([j.dofadr[0] for j in arm_joint])
    lo = np.array([j.range[0] for j in arm_joint])
    hi = np.array([j.range[1] for j in arm_joint])
    waist_act = [act(n) for n in WAIST]
    waist_qadr = np.array([model.joint(n).qposadr[0] for n in WAIST])
    HAND_ACT = [i for i in range(model.nu) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith("right_hand")]
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


def run_trial(dx, yaw_deg):
    yaw = np.radians(yaw_deg)
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[waist_qadr] = list(STANCE)
    data.ctrl[waist_act] = list(STANCE)
    data.qpos[cube_qadr:cube_qadr + 3] = [SEARCH_AT[0], SEARCH_AT[1], table_z]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)
    a = hand_yaw("hybrid", yaw)
    r_des = rot_z(a)
    pre = rot_z(a) @ PRE
    half_h = table_z - 0.75
    dz = max(0.0, FINGER_CLEARANCE_Z - half_h)
    offset = np.array([dx, OY, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, dz])

    go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    go_to(grasp + pre, 1.5, offset, r_des)
    go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    go_to(grasp, 1.0, offset, r_des)
    target = data.ctrl.copy()
    for n in ("index", "middle"):
        target[act(f"right_hand_{n}_0_joint")] = FINGER_CURL[0]
        target[act(f"right_hand_{n}_1_joint")] = FINGER_CURL[1]
    for j, v in enumerate(THUMB):
        target[act(f"right_hand_thumb_{j}_joint")] = v
    move(target, 1.5)
    step(0.5)
    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    step(1.0)

    def cube_in_wrist():
        R = data.xmat[wrist].reshape(3, 3)
        return R.T @ (data.xpos[cube_body] - data.xpos[wrist])

    rel0 = cube_in_wrist()
    n_touch = touches_cube()
    step(1.5)
    rel1 = cube_in_wrist()
    lift = float(data.xpos[cube_body][2] - c0[2])
    slip = float(np.linalg.norm(rel1 - rel0))
    return dict(lift=lift, slip=slip, tilt=flat_tilt(), touches=n_touch, ok=(lift >= 0.06 and slip <= 0.02))


def main():
    t0 = time.time()
    rows, summary = [], []
    total = len(SHAPES) * len(DX_VALUES) * len(YAWS)
    done = 0
    for name, half in SHAPES:
        load_scene(half)
        for dx in DX_VALUES:
            rs = []
            for yaw_deg in YAWS:
                r = run_trial(dx, yaw_deg)
                rs.append(r)
                done += 1
                rows.append([name, dx, yaw_deg, int(r["ok"]), round(r["lift"], 3), round(r["slip"], 3),
                             round(r["tilt"], 1), r["touches"]])
            held = sum(r["ok"] for r in rs)
            tilts = [r["tilt"] for r in rs if r["ok"]]
            summary.append((name, dx, held, (np.mean(tilts) if tilts else float("nan")),
                            (max(tilts) if tilts else float("nan")), np.mean([r["touches"] for r in rs])))
            print(f"[{done:2d}/{total}] {name:22s} dx={dx * 100:.0f} cm: held {held}/{len(YAWS)}, "
                  f"tilt of held ones: mean {summary[-1][3]:.0f} worst {summary[-1][4]:.0f} deg   ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'shape':22s} {'dx':>4s} {'held':>6s} {'mean tilt':>10s} {'worst tilt':>11s} {'contacts':>9s}")
    for name, dx, held, mt, wt, ct in summary:
        print(f"{name:22s} {dx * 100:3.0f}c {held:3d}/{len(YAWS)} {mt:9.1f}d {wt:10.1f}d {ct:9.1f}")
    out = ROOT / f"block_size_test_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "dx", "cube_yaw_deg", "held", "lift_m", "slip_m", "tilt_deg", "touching_bodies"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
