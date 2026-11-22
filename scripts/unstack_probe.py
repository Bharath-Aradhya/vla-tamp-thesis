"""Unstack probe: is the unstack failure a REACH problem (t1, t3) or a PUSH/TIP problem (t2)? Does lower friction help?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\unstack_probe.py

For friction 1.0 (current) and 0.6 on both bars, and base bars at t1, t2, t3: stack a bar from bin1 onto the base
(as before), then unstack it step by step and record at every step the IK error (can the arm get there), the
simulated hand error (does it get there), how far the base bar has moved and how much it is tilted.
2 rotations each = 12 trials. Prints one line per trial and a summary; saves unstack_probe_HHMMSS.csv with every
step. Needs scripts\\place_validation.py and scripts\\stack_unstack_validation.py (the new version).
Expect about 4 to 5 minutes. CPU only.
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
import stack_unstack_validation as su

FRICTIONS = [1.0, 0.6]
BASE_NAMES = ["t1", "t2", "t3"]
SRC = su.SOURCES["bin1"]
ROTS = 2


def base_shift(base0):
    return float(np.linalg.norm(pv.data.xpos[su.base_body][:2] - base0[:2]))


def probe_unstack(base0, steps):
    """Unstack with per-step logging. Returns (stage, lift)."""
    d, m = pv.data, pv.model
    body = pv.cube_body

    def rec(label, ik=np.nan, sim=np.nan):
        steps.append((label, ik, sim, base_shift(base0), su.tilt_of(su.base_body), su.tilt_of(body)))

    a = pv.hand_yaw("bar", su.bar_yaw(body))
    r_des = pv.rot_z(a)
    pre = r_des @ pv.PRE
    offset = np.array([pv.DX, pv.OY, 0.0])
    c0 = d.xpos[body].copy()
    grasp = c0.copy()
    for label, pt, sec in (("above start 1", [0.12, -0.25, 0.80], 1.5), ("above start 2", [0.30, -0.25, 0.92], 1.5),
                           ("pre-grasp high", grasp + pre + np.array([0.0, 0.0, 0.03]), 1.5),
                           ("pre-grasp", grasp + pre, 1.0), ("half way", grasp + 0.5 * pre, 1.0), ("grasp point", grasp, 1.0)):
        ik, sim, _ = pv.go_to(pt, sec, offset, r_des)
        rec("pick: " + label, ik, sim)
    target = d.ctrl.copy()
    for n, v in pv.FINGERS.items():
        target[pv.act(f"right_hand_{n}_joint")] = v
    for j, v in enumerate(pv.THUMB):
        target[pv.act(f"right_hand_thumb_{j}_joint")] = v
    pv.move(target, 1.5)
    pv.step(0.5)
    rec("pick: fingers closed")
    ik, sim, _ = pv.go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    pv.step(1.0)
    rec("pick: lifted", ik, sim)
    lift = float(d.xpos[body][2] - c0[2])
    if lift < 0.06:
        return "pick failed (bar not lifted)", lift

    gp = lambda: d.xpos[pv.wrist] + d.xmat[pv.wrist].reshape(3, 3) @ offset
    rel = d.xpos[body] - gp()
    dst = np.array([SRC[0], SRC[1], pv.table_z])
    aim = dst - r_des @ np.array([pv.CALIB[0], pv.CALIB[1], 0.0])
    ik, sim, _ = pv.go_to(gp() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    rec("carry: extra lift", ik, sim)
    ik, sim, _ = pv.go_to(aim + np.array([0.0, 0.0, su.TABLE_CLEAR]) - rel, 1.8, offset, r_des)
    pv.step(0.5)
    rec("carry: over target", ik, sim)
    if ik > 0.03:
        return "carry failed (arm could not reach the target)", lift
    ik, sim, _ = pv.go_to(aim + np.array([0.0, 0.0, 0.004]) - rel, 1.5, offset, r_des)
    rec("carry: lowered", ik, sim)
    if base_shift(base0) > 0.02 or su.tilt_of(su.base_body) > 8.0:
        return "base bar was knocked", lift
    return "ok", lift


def main():
    t0 = time.time()
    su.build_scene()
    rng = np.random.default_rng(60)
    rows, summary = [], {}
    for mu in FRICTIONS:
        for geom in ("cube_geom", "base_geom"):
            pv.model.geom_friction[pv.model.geom(geom).id, 0] = mu
        for bname in BASE_NAMES:
            for k in range(ROTS):
                bxy = su.BASES[bname]
                s = (SRC[0] + rng.uniform(-0.01, 0.01), SRC[1] + rng.uniform(-0.01, 0.01))
                b = (bxy[0] + rng.uniform(-0.01, 0.01), bxy[1] + rng.uniform(-0.01, 0.01))
                yt, yb = float(rng.uniform(0, 90)), float(rng.uniform(0, 90))
                su.reset_trial(s, np.radians(yt), b, np.radians(yb))
                base0 = pv.data.xpos[su.base_body].copy()
                ctx = su.pick(pv.cube_body)
                if ctx["lift"] < 0.06:
                    stage_stack = "stack: pick failed"
                else:
                    why = su.carry_and_release(pv.cube_body, ctx, np.array([base0[0], base0[1], base0[2] + su.BAR_H]), su.STACK_CLEAR)
                    top = pv.data.xpos[pv.cube_body]
                    ok = (why is None and abs(top[2] - (base0[2] + su.BAR_H)) < 0.01 and base_shift(base0) < 0.02)
                    stage_stack = "ok" if ok else f"stack: {why or 'landing failed'}"
                steps, stage, lift = [], "not tried", np.nan
                if stage_stack == "ok":
                    stage, lift = probe_unstack(pv.data.xpos[su.base_body].copy() * 0 + base0, steps)
                worst_ik = max((st[1] for st in steps if st[1] == st[1]), default=np.nan)
                worst_at = max((st for st in steps if st[1] == st[1]), key=lambda st: st[1], default=("-",))[0]
                shifts = [st[3] for st in steps]
                print(f"friction {mu:.1f} base {bname} rot {k + 1}: stack {stage_stack}; unstack {stage}"
                      + (f"; worst IK err {worst_ik * 100:.1f} cm at '{worst_at}'; base moved {max(shifts) * 100:.1f} cm"
                         if steps else "") + f"   ({time.time() - t0:.0f}s)", flush=True)
                for st in steps:
                    rows.append([mu, bname, k, stage, *[round(v, 4) if isinstance(v, float) else v for v in st]])
                summary.setdefault((mu, bname), []).append(stage)
    print("\nUnstack result per friction and base location (2 trials each):")
    for (mu, bname), v in summary.items():
        print(f"  friction {mu:.1f}  {bname}: {v}")
    out = pv.ROOT / f"unstack_probe_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["friction", "base", "trial", "unstack_result", "step", "ik_err_m", "sim_err_m", "base_shift_m",
                    "base_tilt_deg", "top_tilt_deg"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {pv.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
