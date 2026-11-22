"""Corner calibration: why does the bar land 2-5 cm off at the far-right corner, and does a corner-specific aim fix it?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\corner_calibration.py

corner_validation.py showed: picking FROM the corner works (17 of 20 to the other slots), but placing INTO the corner
works in only 1 of 20 (bar 2.3-4.7 cm off, or toppled), and corner -> t3 landed 3-5 cm off in 3 of 4 trials.
Phase 1: place into the corner from t1, t2, bin1, bin2 (3 rotations each) with the usual aim correction and measure the
mean landing error in the HAND frame (x along fingers, y across the palm).
Phase 2: add that mean error to the aim correction and place into the corner again (5 sources x 4 rotations).
Phase 3: corner -> t3, 6 trials, to measure the error there (no correction applied, just reported).
Bar friction 0.6. Saves corner_calibration_HHMMSS.csv. Expect about 6 minutes. CPU only.
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

FRICTION = 0.6
CORNER = (0.48, -0.25)
SLOTS = dict(zip(pv.NAMES, pv.PLACE_LOCATIONS))
rows = []


def run(phase, sname, sxy, dname, dxy, yaw, calib, rng, t0):
    s = (sxy[0] + rng.uniform(-pv.JITTER, pv.JITTER), sxy[1] + rng.uniform(-pv.JITTER, pv.JITTER))
    r = pv.place_once(s, dxy, yaw, calib)
    print(f"[{phase}] {sname:4s} -> {dname:4s} rot {yaw:4.1f}: {r['stage']}"
          + (f", {r['err_xy'] * 100:.1f} cm, hand-frame error ({r['ex_hand'] * 100:+.1f}, {r['ey_hand'] * 100:+.1f}) cm"
             if r["err_xy"] == r["err_xy"] else "") + f"   ({time.time() - t0:.0f}s)", flush=True)
    rows.append([phase, sname, dname, round(yaw, 1), r["stage"], round(r["err_xy"], 4), round(r["ex_hand"], 4),
                 round(r["ey_hand"], 4), round(r["tilt_final"], 1), round(calib[0], 4), round(calib[1], 4)])
    return r


def main():
    t0 = time.time()
    pv.load_scene(pv.SHAPE)
    pv.model.geom_friction[pv.model.geom("cube_geom").id, 0] = FRICTION
    rng = np.random.default_rng(90)
    # phase 1: measure
    ex, ey, n_ok1 = [], [], 0
    for sname in ("t1", "t2", "bin1", "bin2"):
        for _ in range(3):
            r = run("1 measure", sname, SLOTS[sname], "r4", CORNER, float(rng.uniform(0, 90)), pv.CALIB, rng, t0)
            n_ok1 += r["stage"] == "ok"
            if r["ex_hand"] == r["ex_hand"] and r["tilt_final"] <= 8.0:
                ex.append(r["ex_hand"])
                ey.append(r["ey_hand"])
    if not ex:
        print("No trial landed standing, cannot calibrate; send me this output.")
        return
    mex, mey = float(np.mean(ex)), float(np.mean(ey))
    new = (pv.CALIB[0] + mex, pv.CALIB[1] + mey)
    print(f"\nPhase 1: {n_ok1} of 12 OK. Mean landing error in the hand frame ({mex * 100:+.1f}, {mey * 100:+.1f}) cm "
          f"(spread {np.std(ex) * 100:.1f}, {np.std(ey) * 100:.1f} cm, from {len(ex)} standing bars).")
    print(f"Corner aim correction: {pv.CALIB[0]:.4f}, {pv.CALIB[1]:.4f}  ->  {new[0]:.4f}, {new[1]:.4f}\n")
    # phase 2: test the new correction
    ok2 = 0
    for sname in ("t1", "t2", "t3", "bin1", "bin2"):
        for _ in range(4):
            r = run("2 corrected", sname, SLOTS[sname], "r4", CORNER, float(rng.uniform(0, 90)), new, rng, t0)
            ok2 += r["stage"] == "ok"
    print(f"\nPhase 2: {ok2} of 20 placed into the corner with the corrected aim.\n")
    # phase 3: corner -> t3
    ex3, ey3, ok3 = [], [], 0
    for _ in range(6):
        r = run("3 corner->t3", "r4", CORNER, "t3", SLOTS["t3"], float(rng.uniform(0, 90)), pv.CALIB, rng, t0)
        ok3 += r["stage"] == "ok"
        if r["ex_hand"] == r["ex_hand"] and r["tilt_final"] <= 8.0:
            ex3.append(r["ex_hand"])
            ey3.append(r["ey_hand"])
    print(f"\nPhase 3: corner -> t3: {ok3} of 6 OK" + (f", mean hand-frame error ({np.mean(ex3) * 100:+.1f}, "
          f"{np.mean(ey3) * 100:+.1f}) cm" if ex3 else ""))
    out = pv.ROOT / f"corner_calibration_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "src", "dst", "yaw_deg", "stage", "err_xy_m", "ex_hand_m", "ey_hand_m", "tilt_final_deg",
                    "calib_x", "calib_y"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {pv.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
