"""Release test: how can the hand let go of the bar without dragging it?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\release_test.py

bar_validation.py: the bar was held flat (tilt ~2 deg) but moved 9.5 cm on average while the hand
opened, so the opening motion itself drags it (the finger tips swing outward along the fingers
as they uncurl). This tries ways to let go, on the same 6 pick-and-place cases for each:
  plain          - open the whole hand in 1 s (reference)
  thumb_first    - open the thumb (0.6 s), then the fingers (0.6 s)
  fingers_first  - open the fingers (0.6 s), then the thumb (0.6 s)
  back_x 4/8/12  - open in 1 s while the arm moves 4, 8 or 12 cm backwards along the finger direction
                   (so the finger tips stay put in the world instead of sweeping forward)
  back_y 3/6     - open in 1 s while the arm moves 3 or 6 cm away from the bar along the palm normal
After opening it retreats (up 6 cm, then away) and measures where the bar ended up.
Prints, per method: how far the bar moved while opening (world and in the hand frame), final distance
from the target, and how many ended standing within 2 cm. 8 methods x 6 cases = 48 trials, about 3 minutes.
Saves release_test_HHMMSS.csv. CPU only.
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


METHODS = [("plain", 0.0), ("thumb_first", 0.0), ("fingers_first", 0.0),
           ("back_x", 0.04), ("back_x", 0.08), ("back_x", 0.12), ("back_y", 0.03), ("back_y", 0.06)]
CASES = [(0, 3, 8.0), (3, 0, 21.0), (1, 4, 72.0), (4, 1, 52.0), (2, 5, 8.0), (5, 2, 39.0)]   # (src, dst, cube rotation)


def release_trial(src, dst, yaw_deg, method, amount):
    ctx = do_pick(src, yaw_deg, "hybrid")
    r_des, pre, offset, c0 = ctx["r_des"], ctx["pre"], ctx["offset"], ctx["c0"]
    if data.xpos[cube_body][2] - c0[2] < 0.06:
        return None

    def grasp_point():
        return data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset

    rel = data.xpos[cube_body] - grasp_point()
    dst3 = np.array([dst[0], dst[1], table_z])
    ik, _, _ = go_to(dst3 + np.array([0.0, 0.0, CARRY_CLEARANCE]) - rel, 1.5, offset, r_des)
    step(0.5)
    if ik > 0.03 or np.linalg.norm((data.xpos[cube_body] - grasp_point()) - rel) > 0.03:
        return None
    go_to(dst3 + np.array([0.0, 0.0, 0.004]) - rel, 1.2, offset, r_des)
    step(0.3)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in HAND_ACT]
    open_vals = model.key_ctrl[key][HAND_ACT]
    thumb = [k for k, n in enumerate(names) if "thumb" in n]
    fingers = [k for k, n in enumerate(names) if "thumb" not in n]
    before = data.xpos[cube_body].copy()
    ik_release = 0.0
    target = data.ctrl.copy()
    hand = np.array(HAND_ACT)
    if method == "plain":
        target[hand] = open_vals
        move(target, 1.0)
    elif method in ("thumb_first", "fingers_first"):
        first, second = (thumb, fingers) if method == "thumb_first" else (fingers, thumb)
        target[hand[first]] = open_vals[first]
        move(target, 0.6)
        target[hand] = open_vals
        move(target, 0.6)
    else:
        v = r_des @ (np.array([-amount, 0.0, 0.0]) if method == "back_x" else np.array([0.0, -amount, 0.0]))
        q, ik_release = solve_ik(data.qpos[qadr].copy(), grasp_point() + v, offset, r_des)
        target[arm_act] = q
        target[hand] = open_vals
        move(target, 1.0)
    step(0.5)
    shift = (data.xpos[cube_body] - before)[:2]
    shift_hand = (r_des.T[:2, :2] @ shift)
    go_to(grasp_point() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    go_to(grasp_point() + np.array([pre[0], pre[1], 0.0]), 1.2, offset, r_des)
    step(1.0)
    c = data.xpos[cube_body].copy()
    err = float(np.linalg.norm(c[:2] - dst3[:2]))
    standing = tilt() <= 8.0 and abs(c[2] - table_z) <= 0.01
    return dict(shift=float(np.linalg.norm(shift)), sx=float(shift_hand[0]), sy=float(shift_hand[1]), err=err,
                standing=standing, ok=bool(standing and err <= 0.02), ik_release=ik_release)


def main():
    t0 = time.time()
    load_scene(SHAPE)
    rows, summary = [], []
    total = len(METHODS) * len(CASES)
    done = 0
    for method, amount in METHODS:
        label = method + (f" {amount * 100:.0f} cm" if amount else "")
        rs = []
        for (i, j, yaw_deg) in CASES:
            r = release_trial(LOCATIONS[i], LOCATIONS[j], yaw_deg, method, amount)
            done += 1
            if r is None:
                print(f"[{done:2d}/{total}] {label:16s} case {i}->{j}: pick or carry failed", flush=True)
                rows.append([label, i, j, yaw_deg, "pick/carry failed"] + [""] * 7)
                continue
            rs.append(r)
            rows.append([label, i, j, yaw_deg, "ok" if r["ok"] else "not ok", round(r["shift"], 4), round(r["sx"], 4),
                         round(r["sy"], 4), round(r["err"], 4), int(r["standing"]), int(r["ok"]), round(r["ik_release"], 3)])
            print(f"[{done:2d}/{total}] {label:16s} case {i}->{j}: bar moved {r['shift'] * 100:4.1f} cm while opening "
                  f"(hand frame x {r['sx'] * 100:+.1f}, y {r['sy'] * 100:+.1f}), ends {r['err'] * 100:4.1f} cm from target, "
                  f"{'standing' if r['standing'] else 'NOT standing'}   ({time.time() - t0:.0f}s)", flush=True)
        summary.append((label, rs))

    print(f"\n{'method':16s} {'cases':>6s} {'moved while opening':>20s} {'hand-frame mean (x,y)':>23s} {'final error':>12s} {'OK':>4s}")
    for label, rs in summary:
        if not rs:
            print(f"{label:16s}   no case reached the release")
            continue
        print(f"{label:16s} {len(rs):6d} {np.mean([r['shift'] for r in rs]) * 100:17.1f} cm "
              f"({np.mean([r['sx'] for r in rs]) * 100:+5.1f}, {np.mean([r['sy'] for r in rs]) * 100:+5.1f}) cm "
              f"{np.mean([r['err'] for r in rs]) * 100:9.1f} cm {sum(r['ok'] for r in rs):3d}/{len(rs)}")
    out = ROOT / f"release_test_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "src", "dst", "cube_yaw_deg", "result", "open_shift_m", "shift_hand_x_m", "shift_hand_y_m",
                    "final_err_m", "standing", "ok", "ik_release_err_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
