"""Stack validation: can the frozen pick-and-place recipe put a bar ON TOP of another bar?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\stack_validation.py

Needs scripts\\place_validation.py next to it (re-uses its pick and place code).
First version: the lower bar is FIXED to the table (a static 5x5x10 cm box), so this only tests the
placing: carry the held bar over it, lower it onto the top, open the hand while backing up 10 cm
(the same release as place_validation.py), retreat. The bar is picked from bin1 or bin2 and stacked
on a base at t1, t2 or t3 (3 bases x 2 sources x 4 random rotations = 24 trials).
OK = the bar ends within 2 cm (sideways) of the base centre, 0.90 m high (+-1 cm), tilted <= 8 deg.
Reports success rate, failure stage, error of the OK trials. Saves stack_validation_HHMMSS.csv.
Expect roughly 3 to 5 minutes. CPU only.
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
STACK_SRC = ROOT / "models" / "_stack_src.xml"
BASES = {"t1": (0.32, -0.15), "t2": (0.40, -0.15), "t3": (0.48, -0.15)}
SOURCES = {"bin1": (0.32, -0.25), "bin2": (0.40, -0.25)}
TRIALS = 4
STACK_CLEARANCE = 0.10          # bar centre height above its resting height while carried over the base (m)
BASE_HALF = (0.025, 0.025, 0.05)


def build_scene(base_xy):
    text = (ROOT / "models" / "scene_pick.xml").read_text(encoding="utf-8")
    assert text.count("</worldbody>") == 1, "scene_pick.xml has more than one </worldbody>; tell Claude"
    geom = (f'<geom name="stack_base" type="box" size="{BASE_HALF[0]} {BASE_HALF[1]} {BASE_HALF[2]}" '
            f'pos="{base_xy[0]} {base_xy[1]} {0.75 + BASE_HALF[2]}" rgba="0.85 0.45 0.2 1"/>\n')
    STACK_SRC.write_text(text.replace("</worldbody>", geom + "</worldbody>"), encoding="utf-8")
    pv.SCENE_SRC = STACK_SRC
    pv.load_scene(pv.SHAPE)


def stack_once(src, base_xy, yaw_deg):
    out = dict(stage="ok", err_xy=float("nan"), z=float("nan"), tilt_final=float("nan"),
               pick_lift=float("nan"), ik_carry=float("nan"), carry_slip=float("nan"))
    ctx = pv.do_pick(src, yaw_deg, "bar")
    r_des, pre, offset, c0 = ctx["r_des"], ctx["pre"], ctx["offset"], ctx["c0"]
    d, m = pv.data, pv.model
    out["pick_lift"] = float(d.xpos[pv.cube_body][2] - c0[2])
    if out["pick_lift"] < 0.06:
        out["stage"] = "pick failed (bar not lifted)"
        return out

    def grasp_point():
        return d.xpos[pv.wrist] + d.xmat[pv.wrist].reshape(3, 3) @ offset

    rel = d.xpos[pv.cube_body] - grasp_point()
    rest_z = pv.table_z + 2 * BASE_HALF[2]                       # bar centre height when resting on the base
    dst3 = np.array([base_xy[0], base_xy[1], rest_z])
    aim = dst3 - r_des @ np.array([pv.CALIB[0], pv.CALIB[1], 0.0])
    pv.go_to(grasp_point() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)      # extra lift, clear the base top
    ik, _, _ = pv.go_to(aim + np.array([0.0, 0.0, STACK_CLEARANCE]) - rel, 1.8, offset, r_des)
    pv.step(0.5)
    out["ik_carry"] = ik
    out["carry_slip"] = float(np.linalg.norm((d.xpos[pv.cube_body] - grasp_point()) - rel))
    if ik > 0.03:
        out["stage"] = "carry failed (arm could not reach the target)"
        return out
    if out["carry_slip"] > 0.03:
        out["stage"] = "carry failed (bar slipped in the hand)"
        return out
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
    c = d.xpos[pv.cube_body].copy()
    out["err_xy"] = float(np.linalg.norm(c[:2] - dst3[:2]))
    out["z"] = float(c[2])
    out["tilt_final"] = pv.tilt()
    if c[2] < 0.5:
        out["stage"] = "landing failed (bar fell off)"
    elif c[2] < rest_z - 0.04:
        out["stage"] = "landing failed (bar slid off the base onto the table)"
    elif out["tilt_final"] > 8.0 or abs(c[2] - rest_z) > 0.01:
        out["stage"] = "landing failed (bar tilted or not resting on the base)"
    elif out["err_xy"] > 0.02:
        out["stage"] = "landing failed (more than 2 cm from the base centre)"
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(30)
    total = len(BASES) * len(SOURCES) * TRIALS
    print(f"{len(BASES)} bases x {len(SOURCES)} sources x {TRIALS} rotations = {total} trials")
    rows, stages, errs, done = [], Counter(), [], 0
    for bname, bxy in BASES.items():
        build_scene(bxy)
        for sname, sxy in SOURCES.items():
            for _ in range(TRIALS):
                s = (sxy[0] + rng.uniform(-pv.JITTER, pv.JITTER), sxy[1] + rng.uniform(-pv.JITTER, pv.JITTER))
                yaw_deg = float(rng.uniform(0, 90))
                r = stack_once(s, bxy, yaw_deg)
                stages[r["stage"]] += 1
                if r["stage"] == "ok":
                    errs.append(r["err_xy"])
                done += 1
                print(f"[{done:2d}/{total}] {sname:4s} -> on {bname}, rot {yaw_deg:4.1f}: {r['stage']}"
                      + (f", {r['err_xy'] * 100:.1f} cm off centre, z {r['z']:.3f}" if r["err_xy"] == r["err_xy"] else "")
                      + f"   ({time.time() - t0:.0f}s)", flush=True)
                rows.append([sname, bname, round(s[0], 3), round(s[1], 3), round(yaw_deg, 1), r["stage"],
                             round(r["err_xy"], 4), round(r["z"], 3), round(r["tilt_final"], 1),
                             round(r["pick_lift"], 3), round(r["ik_carry"], 3), round(r["carry_slip"], 3)])

    print(f"\nStacked OK: {stages['ok']} of {total}  ({100 * stages['ok'] / total:.0f}%)")
    for k, v in stages.most_common():
        if k != "ok":
            print(f"   {v:3d}  {k}")
    if errs:
        e = np.array(errs) * 100
        print(f"Error of the OK trials: mean {e.mean():.1f} cm, median {np.median(e):.1f} cm, worst {e.max():.1f} cm")
    out = ROOT / f"stack_validation_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "base", "start_x", "start_y", "cube_yaw_deg", "stage", "err_xy_m", "final_z_m",
                    "tilt_final_deg", "pick_lift_m", "ik_carry_m", "carry_slip_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
