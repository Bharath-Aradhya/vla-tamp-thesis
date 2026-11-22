"""Closing-sweep test: how far does closing the fingers drag the bar, and can a different grip point avoid it?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\closing_sweep_test.py

The stack debug shows that when the fingers close, the bar slides 5 to 9 cm toward the palm (it pushed the base
bar during stacking, and slid across the base bar's top face during unstacking, tipping it). The frozen grip point
(0.15 m along the fingers, 0.02 m across) starts the bar far out on the fingers. This tries grip points that
start the bar closer to where it ends up, on a bar standing on the TABLE at t2 (0.40, -0.15), 6 rotations each.
Prints per grip point: bar held (lifted >= 6 cm, tilt <= 20 deg), average slide while closing (in the hand's frame:
x = along the fingers, y = across the palm) and average tilt. Saves closing_sweep_HHMMSS.csv.
Needs scripts\\place_validation.py. Expect about 3 minutes. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_validation as pv

GRIPS = [(0.15, 0.02), (0.12, 0.03), (0.10, 0.04), (0.08, 0.05), (0.065, 0.06)]   # (DX, OY) along fingers, across palm
YAWS = [3.0, 17.0, 30.0, 62.0, 75.0, 86.0]
LOC = (0.40, -0.15)


def trial(dx, oy, yaw_deg):
    import mujoco
    d, m = pv.data, pv.model
    yaw = np.radians(yaw_deg)
    mujoco.mj_resetDataKeyframe(m, d, pv.key)
    d.ctrl[:] = m.key_ctrl[pv.key]
    d.qpos[pv.waist_qadr] = list(pv.STANCE)
    d.ctrl[pv.waist_act] = list(pv.STANCE)
    d.qpos[pv.cube_qadr:pv.cube_qadr + 3] = [LOC[0], LOC[1], pv.table_z]
    d.qpos[pv.cube_qadr + 3:pv.cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)
    pv.step(1.0)
    a = pv.hand_yaw("bar", yaw)
    r_des = pv.rot_z(a)
    pre = r_des @ pv.PRE
    offset = np.array([dx, oy, 0.0])
    c0 = d.xpos[pv.cube_body].copy()
    grasp = c0.copy()
    pv.go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    pv.go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    pv.go_to(grasp + pre, 1.5, offset, r_des)
    pv.go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    ik, sim, _ = pv.go_to(grasp, 1.0, offset, r_des)

    def rel():
        R = d.xmat[pv.wrist].reshape(3, 3)
        gp = d.xpos[pv.wrist] + R @ offset
        return R.T @ (d.xpos[pv.cube_body] - gp)

    r0 = rel()
    target = d.ctrl.copy()
    for n, v in pv.FINGERS.items():
        target[pv.act(f"right_hand_{n}_joint")] = v
    for j, v in enumerate(pv.THUMB):
        target[pv.act(f"right_hand_thumb_{j}_joint")] = v
    pv.move(target, 1.5)
    pv.step(0.5)
    r1 = rel()
    pv.go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    pv.step(1.0)
    lift = float(d.xpos[pv.cube_body][2] - c0[2])
    tl = pv.tilt()
    slide = r1 - r0
    return dict(ik=ik, sim=sim, lift=lift, tilt=tl, sx=float(slide[0]), sy=float(slide[1]), sz=float(slide[2]),
                held=lift >= 0.06 and tl <= 20.0)


def main():
    t0 = time.time()
    pv.load_scene(pv.SHAPE)
    rows, summary = [], []
    for dx, oy in GRIPS:
        rs = []
        for y in YAWS:
            r = trial(dx, oy, y)
            rs.append(r)
            rows.append([dx, oy, y, int(r["held"]), round(r["lift"], 3), round(r["tilt"], 1), round(r["sx"], 3),
                         round(r["sy"], 3), round(r["sz"], 3), round(r["ik"], 3), round(r["sim"], 3)])
            print(f"grip ({dx:.3f}, {oy:.3f}) rot {y:4.1f}: {'held' if r['held'] else 'NOT held'}, slide in closing "
                  f"({r['sx'] * 100:+.1f}, {r['sy'] * 100:+.1f}) cm, tilt {r['tilt']:.0f}   ({time.time() - t0:.0f}s)", flush=True)
        summary.append((dx, oy, rs))
    print("\nGrip point (along fingers, across palm) -> held, mean slide while closing (hand frame, cm), mean tilt")
    for dx, oy, rs in summary:
        n = sum(r["held"] for r in rs)
        print(f"  ({dx:.3f}, {oy:.3f}):  held {n}/{len(rs)}   slide x {np.mean([r['sx'] for r in rs]) * 100:+.1f}  "
              f"y {np.mean([r['sy'] for r in rs]) * 100:+.1f}   length {np.mean([np.hypot(r['sx'], r['sy']) for r in rs]) * 100:.1f}"
              f"   tilt {np.mean([r['tilt'] for r in rs]):.1f}")
    out = pv.ROOT / f"closing_sweep_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dx", "oy", "yaw_deg", "held", "lift_m", "tilt_deg", "slide_x_m", "slide_y_m", "slide_z_m", "ik_err_m", "sim_err_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {pv.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
