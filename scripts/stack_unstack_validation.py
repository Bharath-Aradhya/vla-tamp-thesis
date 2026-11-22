"""Stack + unstack validation with TWO movable bars (condition-b building blocks).

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\stack_unstack_validation.py

Needs scripts\\place_validation.py next to it (re-uses its IK, motion and grasp code).
Each trial: a base bar stands at t1, t2 or t3 (random rotation, free to move) and a second bar stands at
bin1 or bin2. 1) STACK: pick the second bar and put it on top of the base bar.
2) UNSTACK: pick it off the top again and put it back on the table where it started.
Both bars are 5x5x10 cm and movable, so a push during the approach or a bump during release can disturb the base.
Stack OK   = top bar within 2 cm of the base bar's centre, resting on it (+-1 cm height, tilt <= 8 deg),
             base bar moved less than 2 cm and still upright.
Unstack OK = bar back within 2 cm of its start spot, standing on the table, base bar still within 2 cm and upright.
18 trials (3 bases x 2 sources x 3 rotations). Reports both phases separately, saves stack_unstack_HHMMSS.csv.
Expect roughly 5 to 8 minutes. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_validation as pv

ROOT = pv.ROOT
STACK_SRC = ROOT / "models" / "_stack2_src.xml"
BASES = {"t1": (0.32, -0.15), "t2": (0.40, -0.15), "t3": (0.48, -0.15)}
SOURCES = {"bin1": (0.32, -0.25), "bin2": (0.40, -0.25)}
TRIALS = 3
BAR_H = 0.10                        # bar height (m)
STACK_CLEAR = 0.10                  # bar centre height above its resting height when carried over the stack
UNSTACK_LIFT = 0.04                 # how far the top bar is lifted off the stack (m); enough to clear the base top
UNSTACK_EXTRA = 0.0                 # extra lift before the carry when unstacking (the stack carry uses 0.06)
TABLE_CLEAR = 0.12                  # same when carried to the table (clears the base bar after unstacking)

base_body = base_qadr = None


def build_scene():
    global base_body, base_qadr
    text = (ROOT / "models" / "scene_pick.xml").read_text(encoding="utf-8")
    assert text.count("</worldbody>") == 1, "scene_pick.xml has more than one </worldbody>; tell Claude"
    body = ('<body name="base_bar" pos="0.30 -0.40 0.80">\n'
            '      <freejoint name="base_joint"/>\n'
            '      <geom name="base_geom" type="box" size="0.025 0.025 0.05" mass="0.1" rgba="0.15 0.35 0.85 1"/>\n'
            '    </body>\n')
    STACK_SRC.write_text(text.replace("</worldbody>", body + "</worldbody>"), encoding="utf-8")
    pv.SCENE_SRC = STACK_SRC
    pv.load_scene(pv.SHAPE)
    base_body = pv.model.body("base_bar").id
    base_qadr = pv.model.joint("base_joint").qposadr[0]


def set_bar(qadr, xy, yaw):
    pv.data.qpos[qadr:qadr + 3] = [xy[0], xy[1], pv.table_z]
    pv.data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def bar_yaw(body):
    R = pv.data.xmat[body].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def tilt_of(body):
    return float(np.degrees(np.arccos(min(1.0, abs(pv.data.xmat[body].reshape(3, 3)[2, 2])))))


def reset_trial(src_xy, src_yaw, base_xy, base_yaw):
    d, m = pv.data, pv.model
    import mujoco
    mujoco.mj_resetDataKeyframe(m, d, pv.key)
    d.ctrl[:] = m.key_ctrl[pv.key]
    d.qpos[pv.waist_qadr] = list(pv.STANCE)
    d.ctrl[pv.waist_act] = list(pv.STANCE)
    set_bar(pv.cube_qadr, src_xy, src_yaw)
    set_bar(base_qadr, base_xy, base_yaw)
    mujoco.mj_forward(m, d)
    pv.step(1.0)


def pick(body, dz=None, lift_by=0.10):
    """Reach, grasp and lift the bar `body` by 10 cm from wherever it stands now. dz = grasp height above its centre."""
    d, m = pv.data, pv.model
    dz = pv.DZ if dz is None else dz
    a = pv.hand_yaw("bar", bar_yaw(body))
    r_des = pv.rot_z(a)
    pre = r_des @ pv.PRE
    offset = np.array([pv.DX, pv.OY, 0.0])
    c0 = d.xpos[body].copy()
    grasp = c0 + np.array([0.0, 0.0, dz])
    pv.go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    pv.go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    pv.go_to(grasp + pre + np.array([0.0, 0.0, 0.03]), 1.5, offset, r_des)
    pv.go_to(grasp + pre, 1.0, offset, r_des)
    pv.go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    pv.go_to(grasp, 1.0, offset, r_des)
    target = d.ctrl.copy()
    for n, v in pv.FINGERS.items():
        target[pv.act(f"right_hand_{n}_joint")] = v
    for j, v in enumerate(pv.THUMB):
        target[pv.act(f"right_hand_thumb_{j}_joint")] = v
    pv.move(target, 1.5)
    pv.step(0.5)
    pv.go_to(grasp + np.array([0.0, 0.0, lift_by]), 1.5, offset, r_des)
    pv.step(1.0)
    lift = float(d.xpos[body][2] - c0[2])
    return dict(r_des=r_des, pre=pre, offset=offset, lift=lift)


def carry_and_release(body, ctx, dst3, clearance, extra_lift=0.06):
    """Carry the held bar over dst3 (bar centre target), lower, release backing up, retreat. Returns None or a reason."""
    d, m = pv.data, pv.model
    r_des, pre, offset = ctx["r_des"], ctx["pre"], ctx["offset"]

    def grasp_point():
        return d.xpos[pv.wrist] + d.xmat[pv.wrist].reshape(3, 3) @ offset

    rel = d.xpos[body] - grasp_point()
    aim = dst3 - r_des @ np.array([pv.CALIB[0], pv.CALIB[1], 0.0])
    if extra_lift > 0:
        pv.go_to(grasp_point() + np.array([0.0, 0.0, extra_lift]), 1.0, offset, r_des)   # extra lift to clear both bars
    ik, _, _ = pv.go_to(aim + np.array([0.0, 0.0, clearance]) - rel, 1.8, offset, r_des)
    pv.step(0.5)
    if ik > 0.03:
        return "carry failed (arm could not reach the target)"
    if np.linalg.norm((d.xpos[body] - grasp_point()) - rel) > 0.03:
        return "carry failed (bar slipped in the hand)"
    pv.go_to(aim + np.array([0.0, 0.0, 0.004]) - rel, 1.5, offset, r_des)
    pv.step(0.3)
    q, _ = pv.solve_ik(d.qpos[pv.qadr].copy(), grasp_point() + r_des @ np.array([-pv.BACKUP, 0.0, 0.0]), offset, r_des)
    target = d.ctrl.copy()
    target[pv.arm_act] = q
    target[np.array(pv.HAND_ACT)] = m.key_ctrl[pv.key][pv.HAND_ACT]
    pv.move(target, 1.0)
    pv.step(0.5)
    pv.go_to(grasp_point() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    pv.go_to(grasp_point() + np.array([pre[0], pre[1], 0.0]), 1.2, offset, r_des)
    pv.step(1.0)
    return None


def trial(src_xy, base_xy, rng, unstack_dz=None):
    d = pv.data
    res = dict(stack="ok", unstack="not tried", stack_err=np.nan, unstack_err=np.nan, base_shift=np.nan,
               stack_z=np.nan, yaw_top=np.nan, yaw_base=np.nan)
    yaw_top, yaw_base = float(rng.uniform(0, 90)), float(rng.uniform(0, 90))
    res["yaw_top"], res["yaw_base"] = yaw_top, yaw_base
    reset_trial(src_xy, np.radians(yaw_top), base_xy, np.radians(yaw_base))
    base0 = d.xpos[base_body].copy()

    # ---- phase 1: stack
    ctx = pick(pv.cube_body)
    if ctx["lift"] < 0.06:
        res["stack"] = "pick failed (bar not lifted)"
        return res
    rest = np.array([base0[0], base0[1], base0[2] + BAR_H])
    why = carry_and_release(pv.cube_body, ctx, rest, STACK_CLEAR)
    top = d.xpos[pv.cube_body].copy()
    res["base_shift"] = float(np.linalg.norm(d.xpos[base_body][:2] - base0[:2]))
    res["stack_err"] = float(np.linalg.norm(top[:2] - rest[:2]))
    res["stack_z"] = float(top[2])
    if why:
        res["stack"] = why
    elif top[2] < 0.5:
        res["stack"] = "landing failed (bar fell off)"
    elif top[2] < rest[2] - 0.04:
        res["stack"] = "landing failed (bar slid off the base)"
    elif res["base_shift"] > 0.02 or tilt_of(base_body) > 8.0:
        res["stack"] = "landing failed (base bar was knocked)"
    elif tilt_of(pv.cube_body) > 8.0 or abs(top[2] - rest[2]) > 0.01:
        res["stack"] = "landing failed (bar tilted or not resting on the base)"
    elif res["stack_err"] > 0.02:
        res["stack"] = "landing failed (more than 2 cm from the base centre)"
    if res["stack"] != "ok":
        return res

    # ---- phase 2: unstack, put the bar back where it came from
    res["unstack"] = "ok"
    ctx = pick(pv.cube_body, unstack_dz, UNSTACK_LIFT)
    if ctx["lift"] < 0.6 * UNSTACK_LIFT:
        res["unstack"] = "pick failed (bar not lifted)"
        return res
    dst = np.array([src_xy[0], src_xy[1], pv.table_z])
    why = carry_and_release(pv.cube_body, ctx, dst, TABLE_CLEAR, UNSTACK_EXTRA)
    c = d.xpos[pv.cube_body].copy()
    res["unstack_err"] = float(np.linalg.norm(c[:2] - dst[:2]))
    base_shift = float(np.linalg.norm(d.xpos[base_body][:2] - base0[:2]))
    res["base_shift"] = max(res["base_shift"], base_shift)
    if why:
        res["unstack"] = why
    elif c[2] < 0.5:
        res["unstack"] = "landing failed (bar fell off)"
    elif base_shift > 0.02 or tilt_of(base_body) > 8.0:
        res["unstack"] = "landing failed (base bar was knocked)"
    elif tilt_of(pv.cube_body) > 8.0 or abs(c[2] - pv.table_z) > 0.01:
        res["unstack"] = "landing failed (bar toppled or not on the table)"
    elif res["unstack_err"] > 0.02:
        res["unstack"] = "landing failed (more than 2 cm from the start spot)"
    return res


def main():
    t0 = time.time()
    build_scene()
    rng = np.random.default_rng(40)
    total = len(BASES) * len(SOURCES) * TRIALS
    print(f"{len(BASES)} bases x {len(SOURCES)} sources x {TRIALS} rotations = {total} trials (stack, then unstack)")
    rows, s_stage, u_stage, done = [], Counter(), Counter(), 0
    for bname, bxy in BASES.items():
        for sname, sxy in SOURCES.items():
            for _ in range(TRIALS):
                s = (sxy[0] + rng.uniform(-pv.JITTER, pv.JITTER), sxy[1] + rng.uniform(-pv.JITTER, pv.JITTER))
                b = (bxy[0] + rng.uniform(-pv.JITTER, pv.JITTER), bxy[1] + rng.uniform(-pv.JITTER, pv.JITTER))
                r = trial(s, b, rng)
                s_stage[r["stack"]] += 1
                u_stage[r["unstack"]] += 1
                done += 1
                print(f"[{done:2d}/{total}] {sname} -> on {bname}: stack {r['stack']}; unstack {r['unstack']}"
                      f"   ({time.time() - t0:.0f}s)", flush=True)
                rows.append([sname, bname, round(s[0], 3), round(s[1], 3), round(b[0], 3), round(b[1], 3),
                             round(r["yaw_top"], 1), round(r["yaw_base"], 1), r["stack"], round(r["stack_err"], 4),
                             round(r["stack_z"], 3), r["unstack"], round(r["unstack_err"], 4), round(r["base_shift"], 4)])

    n_unstack = total - sum(v for k, v in s_stage.items() if k != "ok")
    print(f"\nSTACK ok:   {s_stage['ok']} of {total}")
    for k, v in s_stage.most_common():
        if k != "ok":
            print(f"   {v:3d}  {k}")
    print(f"UNSTACK ok: {u_stage['ok']} of {n_unstack} (only tried after a good stack)")
    for k, v in u_stage.most_common():
        if k not in ("ok", "not tried"):
            print(f"   {v:3d}  {k}")
    print(f"Whole stack + unstack cycle ok: {u_stage['ok']} of {total}")
    out = ROOT / f"stack_unstack_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "base", "src_x", "src_y", "base_x", "base_y", "yaw_top_deg", "yaw_base_deg", "stack",
                    "stack_err_m", "stack_z_m", "unstack", "unstack_err_m", "base_shift_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
