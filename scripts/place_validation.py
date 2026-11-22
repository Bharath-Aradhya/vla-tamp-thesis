"""Place validation: the frozen pick-and-place recipe over all 20 pairs of the 5 chosen locations.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\place_validation.py

Recipe (from release_tune.py): pick the 5x5x10 cm bar with the frozen grasp (far waist stance, bar yaw policy hand
yaw), carry it 7 cm above the table, lower it, open the hand while the arm backs up 10 cm along the finger
direction, retreat up and away. The bar is aimed to cancel the measured leftover drift (0.9 cm along the
fingers, 0.35 cm sideways, hand frame).
Optional argument: bar friction (default: the scene's value), e.g.  python scripts\\place_validation.py 0.6
Locations: t1 (0.32,-0.15), t2 (0.40,-0.15), t3 (0.48,-0.15), bin1 (0.32,-0.25), bin2 (0.40,-0.25);
(0.48,-0.25) is dropped because the release drifts furthest there.
60 trials: every ordered pair of the 5 locations x 3 random cube rotations (+-1 cm start offset).
OK = ends within 2 cm of the target, standing upright, resting on the table.
Reports success rate, where each failure happened (pick / carry / landing), landing error, and a
source-by-target matrix. Saves place_validation_HHMMSS.csv. Expect roughly 4 to 7 minutes. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import sys
import time
from collections import Counter
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
FINGERS = {"index_0": 1.2, "index_1": 1.4, "middle_0": 1.2, "middle_1": 1.4}

# --- settings ---------------------------------------------------------------------------------
SHAPE = (0.025, 0.025, 0.05)         # block half-sizes (m): 5 x 5 x 10 cm standing bar
STANCE = (0.17, 0.50)                # fixed waist (yaw, pitch), the 'far' stance
LOCATIONS = [(x, y) for x in (0.32, 0.40, 0.48) for y in (-0.15, -0.25)]
N_YAWS = 16
JITTER = 0.01
DX, OY, DZ = 0.15, 0.02, 0.0         # grasp point (along fingers, across palm) and height above the block centre
THUMB = (0.0, 0.7, -1.6)
PRE = np.array([-0.06, -0.10, 0.05])
ROT_WEIGHT = 0.3
CARRY_CLEARANCE = 0.07               # block centre height above its resting height while carried (m)
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
    if policy == "bar":
        # plain, except the bar rotations where plain failed in the grasp validation (45-52 deg = w -45..-38)
        return float(np.radians(-12.0)) if w < np.radians(-36.0) else 0.0
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


def tilt():
    """Degrees the bar's long axis is tilted away from vertical."""
    return float(np.degrees(np.arccos(min(1.0, abs(data.xmat[cube_body].reshape(3, 3)[2, 2])))))


def hand_part_count():
    return touches_cube()


def do_pick(xy, yaw_deg, policy):
    """Reach, grasp and lift 10 cm. Returns the context needed to continue (carry / place)."""
    yaw = np.radians(yaw_deg)
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qpos[waist_qadr] = list(STANCE)
    data.ctrl[waist_act] = list(STANCE)
    data.qpos[cube_qadr:cube_qadr + 3] = [xy[0], xy[1], table_z]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    step(1.0)
    a = hand_yaw(policy, yaw)
    r_des = rot_z(a)
    pre = rot_z(a) @ PRE
    offset = np.array([DX, OY, 0.0])
    c0 = data.xpos[cube_body].copy()
    grasp = c0 + np.array([0.0, 0.0, DZ])
    go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    go_to(grasp + pre, 1.5, offset, r_des)
    go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    go_to(grasp, 1.0, offset, r_des)
    target = data.ctrl.copy()
    for n, v in FINGERS.items():
        target[act(f"right_hand_{n}_joint")] = v
    for j, v in enumerate(THUMB):
        target[act(f"right_hand_thumb_{j}_joint")] = v
    move(target, 1.5)
    step(0.5)
    go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    step(1.0)
    return dict(r_des=r_des, pre=pre, offset=offset, c0=c0)


NAMES = ["t1", "t2", "t3", "bin1", "bin2"]
PLACE_LOCATIONS = [(0.32, -0.15), (0.40, -0.15), (0.48, -0.15), (0.32, -0.25), (0.40, -0.25)]
BACKUP = 0.10
CALIB = (0.009, 0.0035)               # leftover drift in the hand frame (m): along the fingers, sideways
TRIALS = 3


def place_once(src, dst, yaw_deg, calib=None):
    out = dict(stage="ok", err_xy=float("nan"), tilt_final=float("nan"), z=float("nan"), pick_lift=float("nan"),
               tilt_held=float("nan"), ik_carry=float("nan"), carry_slip=float("nan"), ex_hand=float("nan"), ey_hand=float("nan"))
    calib = CALIB if calib is None else calib
    ctx = do_pick(src, yaw_deg, "bar")
    r_des, pre, offset, c0 = ctx["r_des"], ctx["pre"], ctx["offset"], ctx["c0"]
    out["pick_lift"] = float(data.xpos[cube_body][2] - c0[2])
    out["tilt_held"] = tilt()
    if out["pick_lift"] < 0.06:
        out["stage"] = "pick failed (bar not lifted)"
        return out

    def grasp_point():
        return data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset

    rel = data.xpos[cube_body] - grasp_point()
    dst3 = np.array([dst[0], dst[1], table_z])
    aim = dst3 - r_des @ np.array([calib[0], calib[1], 0.0])
    ik, _, _ = go_to(aim + np.array([0.0, 0.0, CARRY_CLEARANCE]) - rel, 1.5, offset, r_des)
    step(0.5)
    out["ik_carry"] = ik
    out["carry_slip"] = float(np.linalg.norm((data.xpos[cube_body] - grasp_point()) - rel))
    if ik > 0.03:
        out["stage"] = "carry failed (arm could not reach the target)"
        return out
    if out["carry_slip"] > 0.03:
        out["stage"] = "carry failed (bar slipped in the hand)"
        return out
    go_to(aim + np.array([0.0, 0.0, 0.004]) - rel, 1.2, offset, r_des)
    step(0.3)
    q, _ = solve_ik(data.qpos[qadr].copy(), grasp_point() + r_des @ np.array([-BACKUP, 0.0, 0.0]), offset, r_des)
    target = data.ctrl.copy()
    target[arm_act] = q
    target[np.array(HAND_ACT)] = model.key_ctrl[key][HAND_ACT]
    move(target, 1.0)
    step(0.5)
    go_to(grasp_point() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    go_to(grasp_point() + np.array([pre[0], pre[1], 0.0]), 1.2, offset, r_des)
    step(1.0)
    c = data.xpos[cube_body].copy()
    out["err_xy"] = float(np.linalg.norm(c[:2] - dst3[:2]))
    e_hand = r_des.T @ np.array([c[0] - dst3[0], c[1] - dst3[1], 0.0])     # landing error in the hand frame
    out["ex_hand"], out["ey_hand"] = float(e_hand[0]), float(e_hand[1])
    out["z"] = float(c[2])
    out["tilt_final"] = tilt()
    if c[2] < 0.5:
        out["stage"] = "landing failed (bar fell off the table)"
    elif out["tilt_final"] > 8.0 or abs(c[2] - table_z) > 0.01:
        out["stage"] = "landing failed (bar toppled or not on the table)"
    elif out["err_xy"] > 0.02:
        out["stage"] = "landing failed (more than 2 cm from target)"
    return out


def main():
    t0 = time.time()
    load_scene(SHAPE)
    if len(sys.argv) > 1:                                  # optional: bar friction, e.g.  python scripts\\place_validation.py 0.6
        mu = float(sys.argv[1])
        model.geom_friction[model.geom("cube_geom").id, 0] = mu
        print(f"bar friction set to {mu}")
    rng = np.random.default_rng(20)
    pairs = [(i, j) for i in range(5) for j in range(5) if i != j]
    total = len(pairs) * TRIALS
    print(f"{len(pairs)} location pairs x {TRIALS} rotations = {total} trials")
    rows, stages, errs, matrix, done = [], Counter(), [], {}, 0
    for (i, j) in pairs:
        wins = 0
        for t in range(TRIALS):
            s = (PLACE_LOCATIONS[i][0] + rng.uniform(-JITTER, JITTER), PLACE_LOCATIONS[i][1] + rng.uniform(-JITTER, JITTER))
            yaw_deg = float(rng.uniform(0, 90))
            r = place_once(s, PLACE_LOCATIONS[j], yaw_deg)
            stages[r["stage"]] += 1
            if r["stage"] == "ok":
                wins += 1
                errs.append(r["err_xy"])
            done += 1
            print(f"[{done:2d}/{total}] {NAMES[i]:4s} -> {NAMES[j]:4s} rot {yaw_deg:4.1f}: {r['stage']}"
                  + (f", {r['err_xy'] * 100:.1f} cm from target" if r["err_xy"] == r["err_xy"] else "")
                  + (f" (IK err {r['ik_carry'] * 100:.1f} cm)" if r["stage"].startswith("carry failed (arm") else "")
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([NAMES[i], NAMES[j], round(s[0], 3), round(s[1], 3), round(yaw_deg, 1), r["stage"], round(r["err_xy"], 4),
                         round(r["z"], 3), round(r["tilt_final"], 1), round(r["pick_lift"], 3), round(r["tilt_held"], 1),
                         round(r["ik_carry"], 3), round(r["carry_slip"], 3)])
        matrix[(i, j)] = wins

    print(f"\nOK trials (within 2 cm, standing, on the table): {stages['ok']} of {total}  ({100 * stages['ok'] / total:.0f}%)")
    for k, v in stages.most_common():
        if k != "ok":
            print(f"   {v:3d}  {k}")
    if errs:
        e = np.array(errs) * 100
        print(f"Landing error of the OK trials: mean {e.mean():.1f} cm, median {np.median(e):.1f} cm, worst {e.max():.1f} cm")
    print(f"\nOK trials per source (rows) and target (columns), out of {TRIALS}")
    print("        " + "".join(f"to {n:<5s}" for n in NAMES))
    for i in range(5):
        print(f"from {NAMES[i]:<5s}" + "".join(("   -    " if i == j else f"   {matrix[(i, j)]}    ") for j in range(5)))
    out = ROOT / f"place_validation_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "start_x", "start_y", "cube_yaw_deg", "stage", "err_xy_m", "final_z_m", "tilt_final_deg",
                    "pick_lift_m", "tilt_held_deg", "ik_carry_m", "carry_slip_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
