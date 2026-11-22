"""Bar validation: grasp and pick-and-place with the 5x5x10 cm standing bar as the thesis block.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\bar_validation.py

block_size_test.py showed the 5x5x10 cm bar is held flat (tilt about 2 deg, 3 contacts, 4/4 at
dx = 15 cm) while every cube tilted 25-39 deg. This script checks it properly:
  PART 1 - grasp: 6 locations x 16 cube rotations (0 to 84 deg, +-1 cm), hand yaw 'plain' vs
           'hybrid' (the rule from hybrid_grasp.py). 192 trials. Reports held/not held, tilt of the
           bar after the lift, failure reasons, per location and per rotation.
  PART 2 - pick-and-place with the hybrid policy: every ordered pair of the 6 locations, 2 trials each
           (60 trials): pick, carry (7 cm above the table), lower, open, retreat. OK = lands within
           2 cm of the target, standing upright, resting on the table. Also records how far the bar
           moved while the hand opened and its tilt when lowered.
Saves bar_validation_grasp_HHMMSS.csv and bar_validation_place_HHMMSS.csv. Expect roughly 7 to 10 minutes.
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


def run_grasp(xy, yaw_deg, policy):
    ctx = do_pick(xy, yaw_deg, policy)

    def cube_in_wrist():
        R = data.xmat[wrist].reshape(3, 3)
        return R.T @ (data.xpos[cube_body] - data.xpos[wrist])

    rel0 = cube_in_wrist()
    n_touch = hand_part_count()
    step(1.5)
    rel1 = cube_in_wrist()
    lift = float(data.xpos[cube_body][2] - ctx["c0"][2])
    slip = float(np.linalg.norm(rel1 - rel0))
    ok = lift >= 0.06 and slip <= 0.02
    why = "held" if ok else ("fell" if lift < -0.3 else ("not lifted" if lift < 0.06 else "slipped"))
    return dict(lift=lift, slip=slip, tilt=tilt(), touches=n_touch, ok=ok, why=why)


def run_place(src, dst, yaw_deg):
    ctx = do_pick(src, yaw_deg, "hybrid")
    r_des, pre, offset, c0 = ctx["r_des"], ctx["pre"], ctx["offset"], ctx["c0"]
    out = dict(stage="ok", err_xy=float("nan"), z=float("nan"), tilt_final=float("nan"), tilt_lowered=float("nan"),
               open_shift=float("nan"), carry_slip=float("nan"), ik_carry=float("nan"))
    if data.xpos[cube_body][2] - c0[2] < 0.06:
        out["stage"] = "pick failed"
        return out

    def grasp_point():
        return data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset

    rel = data.xpos[cube_body] - grasp_point()
    dst3 = np.array([dst[0], dst[1], table_z])
    ik, sim, _ = go_to(dst3 + np.array([0.0, 0.0, CARRY_CLEARANCE]) - rel, 1.5, offset, r_des)
    step(0.5)
    out["ik_carry"] = ik
    out["carry_slip"] = float(np.linalg.norm((data.xpos[cube_body] - grasp_point()) - rel))
    if out["carry_slip"] > 0.03 or ik > 0.03:
        out["stage"] = "carry went wrong (block slipped or arm could not reach)"
        return out
    go_to(dst3 + np.array([0.0, 0.0, 0.004]) - rel, 1.2, offset, r_des)
    step(0.3)
    out["tilt_lowered"] = tilt()
    before = data.xpos[cube_body].copy()
    target = data.ctrl.copy()
    target[HAND_ACT] = model.key_ctrl[key][HAND_ACT]
    move(target, 1.0)
    step(0.5)
    out["open_shift"] = float(np.linalg.norm((data.xpos[cube_body] - before)[:2]))
    go_to(grasp_point() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    go_to(grasp_point() + np.array([pre[0], pre[1], 0.0]), 1.2, offset, r_des)
    step(1.0)
    c = data.xpos[cube_body].copy()
    out["err_xy"] = float(np.linalg.norm(c[:2] - dst3[:2]))
    out["z"] = float(c[2])
    out["tilt_final"] = tilt()
    if c[2] < 0.5:
        out["stage"] = "block fell off table"
    elif out["tilt_final"] > 8.0 or abs(c[2] - table_z) > 0.01:
        out["stage"] = "block toppled or not resting on table"
    elif out["err_xy"] > 0.02:
        out["stage"] = "landed more than 2 cm from target"
    return out


def part1():
    t0 = time.time()
    rng = np.random.default_rng(1)
    poses = {loc: [(loc[0] + rng.uniform(-JITTER, JITTER), loc[1] + rng.uniform(-JITTER, JITTER),
                    k * 90.0 / N_YAWS + rng.uniform(-1.5, 1.5)) for k in range(N_YAWS)] for loc in LOCATIONS}
    total = 2 * len(LOCATIONS) * N_YAWS
    print(f"PART 1: grasp, {total} trials")
    rows, by_loc, by_yaw, whys, tilts, done = [], {}, {}, {}, {}, 0
    for policy in ("plain", "hybrid"):
        whys[policy], by_yaw[policy], tilts[policy] = Counter(), [0] * N_YAWS, []
        for loc in LOCATIONS:
            wins = 0
            for k, (cx, cy, yaw_deg) in enumerate(poses[loc]):
                r = run_grasp((cx, cy), yaw_deg, policy)
                wins += r["ok"]
                by_yaw[policy][k] += r["ok"]
                whys[policy][r["why"]] += 1
                if r["ok"]:
                    tilts[policy].append(r["tilt"])
                rows.append([policy, loc[0], loc[1], round(yaw_deg, 1), int(r["ok"]), r["why"], round(r["lift"], 3),
                             round(r["slip"], 3), round(r["tilt"], 1), r["touches"]])
                done += 1
            by_loc[(policy, loc)] = wins
            print(f"[{done:3d}/{total}] {policy:6s} x={loc[0]:.2f} y={loc[1]:+.2f}: {wins}/{N_YAWS} held   ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nBar held, out of {N_YAWS} rotations, per location:\n")
    print(f"{'policy':8s}" + "".join(f"x={x:.2f} y={y:+.2f}  " for (x, y) in LOCATIONS) + "  TOTAL")
    for policy in ("plain", "hybrid"):
        vals = [by_loc[(policy, loc)] for loc in LOCATIONS]
        print(f"{policy:8s}" + "".join(f"{v}/{N_YAWS}".center(17) for v in vals) + f"  {sum(vals)}/{N_YAWS * len(LOCATIONS)}")
    print(f"\nBar held, out of {len(LOCATIONS)} locations, per cube rotation (degrees):\n")
    print(f"{'policy':8s}" + "".join(f"{k * 90.0 / N_YAWS:5.1f}" for k in range(N_YAWS)))
    for policy in ("plain", "hybrid"):
        print(f"{policy:8s}" + "".join(f"{v:5d}" for v in by_yaw[policy]))
    for policy in ("plain", "hybrid"):
        t = tilts[policy]
        fails = {k: v for k, v in whys[policy].items() if k != "held"}
        print(f"{policy}: tilt of held bars mean {np.mean(t) if t else float('nan'):.1f} deg, worst {max(t) if t else float('nan'):.1f} deg; "
              f"failures: " + ("; ".join(f"{v} x {k}" for k, v in fails.items()) or "none"))
    out = ROOT / f"bar_validation_grasp_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "loc_x", "loc_y", "cube_yaw_deg", "held", "why", "lift_m", "slip_m", "tilt_deg", "touching_bodies"])
        w.writerows(rows)
    print(f"Saved {out.name}\n")


def part2():
    t0 = time.time()
    rng = np.random.default_rng(2)
    pairs = [(i, j) for i in range(len(LOCATIONS)) for j in range(len(LOCATIONS)) if i != j]
    trials = 2
    total = len(pairs) * trials
    print(f"PART 2: pick-and-place, {len(pairs)} pairs x {trials} = {total} trials")
    rows, stages, errs, shifts, matrix, done = [], Counter(), [], [], {}, 0
    for (i, j) in pairs:
        src, dst = LOCATIONS[i], LOCATIONS[j]
        wins = 0
        for t in range(trials):
            s = (src[0] + rng.uniform(-JITTER, JITTER), src[1] + rng.uniform(-JITTER, JITTER))
            yaw_deg = float(rng.uniform(0, 90))
            r = run_place(s, dst, yaw_deg)
            stages[r["stage"]] += 1
            if r["stage"] == "ok":
                wins += 1
                errs.append(r["err_xy"])
            if r["open_shift"] == r["open_shift"]:
                shifts.append(r["open_shift"])
            done += 1
            print(f"[{done:2d}/{total}] ({src[0]:.2f},{src[1]:+.2f}) -> ({dst[0]:.2f},{dst[1]:+.2f}) rot {yaw_deg:4.1f}: {r['stage']}"
                  + (f", {r['err_xy'] * 100:.1f} cm from target, moved {r['open_shift'] * 100:.1f} cm while opening, tilt {r['tilt_lowered']:.0f} deg when lowered"
                     if r["err_xy"] == r["err_xy"] else "") + f"   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([i, j, round(s[0], 3), round(s[1], 3), round(yaw_deg, 1), r["stage"], round(r["err_xy"], 4), round(r["z"], 3),
                         round(r["tilt_final"], 1), round(r["tilt_lowered"], 1), round(r["open_shift"], 4),
                         round(r["carry_slip"], 3), round(r["ik_carry"], 3)])
        matrix[(i, j)] = wins
    print(f"\nOK trials (within 2 cm, standing, on the table): {stages['ok']} of {total}")
    for k, v in stages.most_common():
        if k != "ok":
            print(f"   {v:3d}  {k}")
    if errs:
        e = np.array(errs) * 100
        print(f"Landing error of the OK trials: mean {e.mean():.1f} cm, worst {e.max():.1f} cm")
    if shifts:
        print(f"Movement of the bar while the hand opened: mean {np.mean(shifts) * 100:.1f} cm, worst {np.max(shifts) * 100:.1f} cm")
    n = len(LOCATIONS)
    print("\nOK trials per source (rows) and target (columns), out of", trials)
    print("location index:  " + "  ".join(f"{k}=({LOCATIONS[k][0]:.2f},{LOCATIONS[k][1]:+.2f})" for k in range(n)))
    print("        " + "".join(f"to {k}  " for k in range(n)))
    for i in range(n):
        print(f"from {i}  " + "".join(("  -   " if i == j else f"  {matrix[(i, j)]}   ") for j in range(n)))
    out = ROOT / f"bar_validation_place_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_index", "dst_index", "start_x", "start_y", "cube_yaw_deg", "stage", "err_xy_m", "final_z_m",
                    "tilt_final_deg", "tilt_lowered_deg", "open_shift_m", "carry_slip_m", "ik_carry_m"])
        w.writerows(rows)
    print(f"Saved {out.name}")


def main():
    t0 = time.time()
    load_scene(SHAPE)
    part1()
    part2()
    print(f"\nTotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
